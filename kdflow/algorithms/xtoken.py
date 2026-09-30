"""NVIDIA X-Token P-KL adapted to KDFlow's on-policy rollout pipeline.

The projection and alignment semantics follow the pinned NeMo-RL X-Token
implementation. KDFlow supplies student-generated responses, so this module
changes only the data-generation regime from fixed-corpus to on-policy.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from kdflow.algorithms import register_algorithm
from kdflow.loss.cross_entropy import compute_cross_entropy
from kdflow.utils.logging_utils import init_logger
from xtoken_upstream_token_aligner import TokenAligner


logger = init_logger(__name__)


class _Fp32SparseMM(torch.autograd.Function):
    """FP32 sparse projection with gradient only for the dense probabilities."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, sparse_matrix: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        ctx.sparse_matrix = sparse_matrix
        return torch.sparse.mm(sparse_matrix.t(), dense)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out: torch.Tensor):
        return None, torch.sparse.mm(ctx.sparse_matrix, grad_out)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _chunk_average(values: torch.Tensor, chunk_ids: torch.Tensor, chunks: int):
    valid = chunk_ids >= 0
    sums = torch.zeros((chunks, values.shape[-1]), dtype=values.dtype, device=values.device)
    counts = torch.zeros((chunks,), dtype=torch.float32, device=values.device)
    if valid.any():
        ids = chunk_ids[valid]
        sums.index_add_(0, ids, values[valid])
        counts.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
    return sums / counts.clamp_min(1).unsqueeze(-1), counts


@register_algorithm("xtoken")
class XTokenOnPolicyKD:
    """Projection-KL plus CE on student rollouts using exact X-Token artifacts."""

    def __init__(
        self,
        strategy,
        student_model,
        teacher_lm_head,
        student_tokenizer,
        teacher_tokenizer,
        **kwargs,
    ):
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.teacher_lm_head = teacher_lm_head
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer

        projection_path = Path(self.args.kd.xtoken_projection_path)
        if not projection_path.is_file():
            raise FileNotFoundError(f"X-Token projection not found: {projection_path}")
        actual_sha = _sha256(projection_path)
        expected_sha = self.args.kd.xtoken_projection_sha256.lower()
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"X-Token projection SHA mismatch: expected {expected_sha}, got {actual_sha}"
            )
        projection = torch.load(projection_path, map_location="cpu", weights_only=True)
        indices = projection["indices"]
        likelihoods = projection["likelihoods"]
        if indices.ndim != 2 or indices.shape != likelihoods.shape:
            raise RuntimeError("invalid X-Token projection tensor shapes")
        if indices.shape[0] != len(student_tokenizer):
            raise RuntimeError(
                f"projection student vocab mismatch: {indices.shape[0]} vs {len(student_tokenizer)}"
            )
        if not torch.isfinite(likelihoods).all() or (likelihoods < 0).any():
            raise RuntimeError("X-Token projection contains invalid likelihoods")
        if (likelihoods[indices < 0] != 0).any():
            raise RuntimeError("X-Token projection sentinel entries must have zero weight")

        device = self.teacher_lm_head.weight.device
        self.projection_indices = indices.to(device=device, dtype=torch.long)
        self.projection_weights = likelihoods.to(device=device, dtype=torch.float32)
        self.teacher_vocab_size = len(teacher_tokenizer)
        self.temperature = float(self.args.kd.kd_temperature)
        self.vocab_topk = min(int(self.args.kd.xtoken_vocab_topk), self.teacher_vocab_size)
        self.aligner = TokenAligner(
            student_tokenizer,
            teacher_tokenizer,
            str(projection_path),
            max_comb_len=int(self.args.kd.xtoken_max_comb_len),
        )
        logger.info(
            "Loaded audited X-Token projection sha=%s shape=%s topk=%d",
            actual_sha,
            tuple(indices.shape),
            self.vocab_topk,
        )

    def _sliced_projection(self, teacher_top_indices: torch.Tensor) -> torch.Tensor:
        lookup = torch.full(
            (self.teacher_vocab_size,),
            -1,
            dtype=torch.long,
            device=teacher_top_indices.device,
        )
        lookup[teacher_top_indices] = torch.arange(
            teacher_top_indices.numel(), device=teacher_top_indices.device
        )
        source_rows = torch.arange(
            self.projection_indices.shape[0], device=teacher_top_indices.device
        ).unsqueeze(1).expand_as(self.projection_indices)
        valid_teacher = self.projection_indices >= 0
        mapped_cols = torch.full_like(self.projection_indices, -1)
        mapped_cols[valid_teacher] = lookup[self.projection_indices[valid_teacher]]
        keep = mapped_cols >= 0
        sparse_indices = torch.stack((source_rows[keep], mapped_cols[keep]), dim=0)
        sparse_values = self.projection_weights[keep]
        return torch.sparse_coo_tensor(
            sparse_indices,
            sparse_values,
            (self.projection_indices.shape[0], teacher_top_indices.numel()),
            dtype=torch.float32,
            device=teacher_top_indices.device,
        ).coalesce()

    def training_step(self, micro_batch):
        student_input_ids = micro_batch["stu_input_ids"]
        student_attn_mask = micro_batch["stu_attn_mask"]
        student_loss_mask = micro_batch["stu_loss_mask"].bool()
        teacher_input_ids = micro_batch["tea_input_ids"]
        teacher_loss_mask = micro_batch["tea_loss_mask"].bool()
        teacher_hiddens = micro_batch.get("teacher_hiddens")
        avg_token_num = micro_batch["avg_micro_batch_token_num"]
        if teacher_hiddens is None:
            raise RuntimeError("micro_batch must contain teacher_hiddens for X-Token")
        if student_input_ids.shape[0] != 1:
            raise RuntimeError("X-Token OPD currently requires micro_train_batch_size=1")

        mm_kwargs = {key[3:]: value for key, value in micro_batch.items() if key.startswith("mm_")}
        output = self.student(
            student_input_ids,
            attention_mask=student_attn_mask,
            allgather_logits=True,
            ring_attn_group=self.strategy.ring_attn_group,
            **mm_kwargs,
        )
        student_logits = output["logits"][student_loss_mask]
        student_label_ids = student_input_ids.roll(shifts=-1, dims=1)[student_loss_mask]

        teacher_hiddens = teacher_hiddens.to(self.teacher_lm_head.weight)
        teacher_logits = self.teacher_lm_head(teacher_hiddens)
        teacher_logits = teacher_logits[..., : self.teacher_vocab_size]
        teacher_label_ids = teacher_input_ids.roll(shifts=-1, dims=1)[teacher_loss_mask]

        if student_logits.shape[0] != student_label_ids.numel():
            raise RuntimeError("student predictor/label cardinality mismatch")
        if teacher_logits.shape[0] != teacher_label_ids.numel():
            raise RuntimeError("teacher predictor/label cardinality mismatch")

        alignment = self.aligner.align(
            student_label_ids.reshape(1, -1).cpu(),
            teacher_label_ids.reshape(1, -1).cpu(),
        )
        chunks = alignment.pair_valid.shape[1]
        student_chunk_ids = alignment.student_chunk_id[0].to(student_logits.device)
        teacher_chunk_ids = alignment.teacher_chunk_id[0].to(student_logits.device)
        pair_valid = alignment.pair_valid[0].to(student_logits.device)
        if self.args.kd.xtoken_exact_match_only:
            pair_valid &= alignment.pair_is_correct[0].to(student_logits.device)

        teacher_importance = teacher_logits.detach().max(dim=0).values
        teacher_top_indices = torch.topk(
            teacher_importance, k=self.vocab_topk, sorted=True
        ).indices
        teacher_top_logits = teacher_logits[:, teacher_top_indices]
        teacher_top_log_probs = torch.log_softmax(
            teacher_top_logits.float() / self.temperature, dim=-1
        )

        student_probs = torch.softmax(
            student_logits / self.temperature, dim=-1, dtype=torch.float32
        )
        sparse_projection = self._sliced_projection(teacher_top_indices)
        projected = _Fp32SparseMM.apply(sparse_projection, student_probs.t()).t()

        projected_chunks, projected_counts = _chunk_average(
            projected, student_chunk_ids, chunks
        )
        teacher_chunks, teacher_counts = _chunk_average(
            teacher_top_log_probs, teacher_chunk_ids, chunks
        )
        projected_mass = projected_chunks.sum(dim=-1)
        projected_chunks = projected_chunks / projected_mass.clamp_min(1e-10).unsqueeze(-1)
        projected_log_chunks = projected_chunks.clamp_min(1e-10).log()

        valid = pair_valid & (projected_counts > 0) & (teacher_counts > 0)
        if not valid.any():
            raise RuntimeError("X-Token alignment produced zero valid chunks")
        if self.args.kd.kd_loss_fn == "rkl":
            per_chunk_kd = F.kl_div(
                teacher_chunks, projected_log_chunks, reduction="none", log_target=True
            ).sum(dim=-1)
        elif self.args.kd.kd_loss_fn == "kl":
            per_chunk_kd = F.kl_div(
                projected_log_chunks, teacher_chunks, reduction="none", log_target=True
            ).sum(dim=-1)
        else:
            raise ValueError("X-Token supports kd_loss_fn=kl or rkl")
        kd_loss = per_chunk_kd[valid].mean() * (self.temperature**2)

        ce_loss = compute_cross_entropy(
            student_logits, student_label_ids, reduction="sum"
        ) / avg_token_num
        if self.args.kd.xtoken_dynamic_loss_scaling:
            kd_abs = kd_loss.detach().abs()
            ce_abs = ce_loss.detach().abs()
            kd_scale = torch.where(kd_abs > 0, ce_abs / kd_abs, torch.ones_like(kd_abs))
            loss = kd_scale * kd_loss + ce_loss
        else:
            kd_scale = torch.ones_like(kd_loss)
            loss = (
                self.args.kd.xtoken_kl_loss_weight * kd_loss
                + self.args.kd.xtoken_ce_loss_scale * ce_loss
            )

        with torch.no_grad():
            projection_top1 = projected_chunks.argmax(dim=-1)
            teacher_top1 = teacher_chunks.exp().argmax(dim=-1)
            projection_accuracy = (projection_top1[valid] == teacher_top1[valid]).float().mean()
            token_accuracy = (student_logits.argmax(dim=-1) == student_label_ids).float().mean()
            align_ratio = valid.sum().float() / max(chunks, 1)

        return {
            "loss": loss,
            "kd_loss": kd_loss,
            "ce_loss": ce_loss,
            "kd_loss_scale": kd_scale,
            "accuracy": token_accuracy,
            "proj_accuracy": projection_accuracy,
            "align_ratio": align_ratio,
            "num_valid_chunks": valid.sum().float(),
            "projection_topk_mass": projected_mass[valid].mean(),
            "valid_student_tokens": student_loss_mask.sum().float(),
            "valid_teacher_tokens": teacher_loss_mask.sum().float(),
        }
