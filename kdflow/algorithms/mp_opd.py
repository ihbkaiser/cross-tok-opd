"""Meta-Partitioned Credit Distillation on student on-policy responses.

``mp_opd`` is a KDFlow algorithm mode, not a replacement for distributional
SpanCTKD. It reuses only SimCT's minimal synchronized boundaries and assigns
scalar realized-path credits to contiguous atom partitions.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import time
from pathlib import Path

import torch

from kdflow.algorithms import register_algorithm
from kdflow.energy_cadence import energy_update_due
from kdflow.loss.cross_entropy import compute_cross_entropy

from ._mp_opd_atoms import SimCTAtomizer
from ._mp_opd_credit import (
    build_atom_credits,
    credit_conservation_residual,
    expected_atom_rates,
    hard_partition_loss,
    soft_partition_loss,
    span_tables,
)
from ._mp_opd_energy import MPAtomEnergy, load_energy_checkpoint
from ._mp_opd_gbv_span import (
    atom_logit_sensitivity,
    gbv_partition,
    gbv_partition_metrics,
    gbv_span_costs,
)
from ._mp_opd_oracle import hard_max_partition, span_utility_table
from ._mp_opd_semimarkov import semi_markov_partition
from ._mp_opd_training_diagnostics import (
    logit_gradient_metrics,
    partition_metrics,
)


def fixed_partition(n: int, length: int) -> tuple[tuple[int, int], ...]:
    if n < 0 or length <= 0:
        raise ValueError("n must be nonnegative and length positive")
    return tuple((start, min(start + length, n)) for start in range(0, n, length))


def random_partition(n: int, max_length: int, seed: int,
                     min_length: int = 1) -> tuple[tuple[int, int], ...]:
    """Random spans of length in [min_length, max_length], tiling [0, n) exactly.

    min_length=1 is the historical draw. A short tail is clamped so the last span covers the
    remaining atoms even when fewer than min_length are left, instead of failing.
    """
    if n < 0 or max_length <= 0:
        raise ValueError("n must be nonnegative and max_length positive")
    if min_length <= 0 or min_length > max_length:
        raise ValueError("min_length must be positive and no larger than max_length")
    generator = random.Random(int(seed))
    parts = []
    cursor = 0
    while cursor < n:
        high = min(max_length, n - cursor)
        low = min(min_length, high)
        length = generator.randint(low, high)
        parts.append((cursor, cursor + length))
        cursor += length
    return tuple(parts)


def atom_features(atoms, credits) -> torch.Tensor:
    # Transfer static metadata once, instead of allocating CUDA scalars per
    # atom. Features deliberately stay detached from the student graph.
    with torch.no_grad():
        static = credits.weight.new_tensor([
            (float(a.teacher_token_count), float(a.byte_end - a.byte_start),
             float(a.boundary_type == "one_to_one"),
             float(a.boundary_type == "multi_token"), 1.0)
            for a in atoms
        ])
        # CUDA scalar division can use a different rounding path from tensor
        # division. Group equal counts to retain the original scalar divisor.
        normalized_teacher = torch.empty_like(credits.teacher_log_score)
        groups = {}
        for i, a in enumerate(atoms):
            groups.setdefault(a.teacher_token_count, []).append(i)
        for count, indices in groups.items():
            index = torch.tensor(indices, device=static.device)
            normalized_teacher[index] = credits.teacher_log_score[index] / float(count)
        return torch.stack((
            credits.rate, credits.base_credit, credits.weight, static[:, 0],
            static[:, 1], normalized_teacher,
            credits.student_old_log_score / credits.weight,
            static[:, 2], static[:, 3], static[:, 4],
        ), dim=1).detach()


def _finite_stats(prefix: str, values: torch.Tensor) -> dict[str, torch.Tensor]:
    values = values.detach().float()
    return {
        f"{prefix}_mean": values.mean(),
        f"{prefix}_std": values.std(unbiased=False),
        f"{prefix}_min": values.min(),
        f"{prefix}_max": values.max(),
        f"{prefix}_positive_fraction": (values > 0).float().mean(),
        f"{prefix}_negative_fraction": (values < 0).float().mean(),
    }


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name}={raw!r} must be a boolean value")


def _behavior_parity_metrics(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    rollout_temperature: float,
) -> dict[str, torch.Tensor]:
    """Compare SGLang decode policy logprobs with the trainer policy.

    SGLang applies sampling temperature to output-token logprobs, while its
    input-token/prefill logprobs remain raw model logprobs. The trajectory
    stores output-token logprobs, so parity must apply the rollout temperature
    to the trainer logits as well.
    """
    temperature = float(rollout_temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("MP-OPD parity requires a positive finite rollout temperature")

    if torch.isinf(behavior_log_probs).any():
        raise RuntimeError("behavior logprobs contain infinity")
    # Synthetic terminal events are represented by NaN and intentionally have
    # no behavior probability. Infinity is never a valid sentinel.
    real = ~torch.isnan(behavior_log_probs)
    if not real.any():
        return {}
    actual = (
        (student_logits.detach().float() / temperature)
        .log_softmax(dim=-1)
        .gather(-1, labels.unsqueeze(-1))
        .squeeze(-1)
    )
    delta = (actual[real] - behavior_log_probs[real]).abs()
    if not torch.isfinite(delta).all():
        raise RuntimeError("behavior/trainer logprob parity produced non-finite deltas")

    mean = delta.mean()
    maximum = delta.max()
    p99 = torch.quantile(delta, 0.99)
    # Backend/precision tails can contain isolated finite outliers. Fail on a
    # distributional mismatch while preserving the tail as diagnostics.
    if mean > 0.1 or p99 > 0.5:
        raise RuntimeError(
            "behavior/trainer logprob parity failed: "
            f"mean={mean.item():.6f}, p99={p99.item():.6f}, max={maximum.item():.6f}"
        )
    return {
        "trajectory_logprob_abs_mean": mean,
        "trajectory_logprob_abs_p99": p99,
        "trajectory_logprob_abs_max": maximum,
        "trajectory_logprob_above_0p5_fraction": (delta > 0.5).float().mean(),
    }


@register_algorithm("mp_opd")
class MetaPartitionedOPD:
    """Scalar canonical-path credit with contiguous SimCT atom partitions.

    The real student optimizer never owns the energy parameters. ``soft``
    therefore requires an explicit, separately trained energy checkpoint.
    Oracle mode requires detached per-atom directional scores in the batch and
    is intentionally diagnostic-only.
    """

    FEATURE_DIM = 10

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
        self.atomizer = SimCTAtomizer(student_tokenizer, teacher_tokenizer)
        self.mode = self.args.kd.mp_opd_mode
        self.max_span_length = int(self.args.kd.mp_opd_max_span_length)
        self.min_span_length = int(getattr(self.args.kd, "mp_opd_min_span_length", 1))
        self.fixed_span_length = int(self.args.kd.mp_opd_fixed_span_length)
        self.temperature = float(self.args.kd.mp_opd_partition_temperature)
        # GBV-Span selects a partition from observed credit rates and logit
        # sensitivities; beta and the geometry are locked recipe knobs, and the
        # exact geometry is the only reason the student logits are retained for a
        # non-diagnostic step.
        self.gbv_beta = float(getattr(self.args.kd, "mp_opd_gbv_beta", 1.0))
        self.gbv_geometry = str(getattr(self.args.kd, "mp_opd_gbv_geometry", "token_count"))
        if self.mode == "gbv" and self.gbv_geometry not in {"exact_logit", "token_count"}:
            raise ValueError(f"unsupported mp_opd_gbv_geometry: {self.gbv_geometry}")
        self.gbv_needs_logits = self.mode == "gbv" and self.gbv_geometry == "exact_logit"
        self.host_mask = bool(getattr(self.args.kd, "mp_opd_host_mask", False))
        self.timing_enabled = os.environ.get("MP_OPD_TIMING", "0") == "1"
        # The diagnostics are opt-in because logit-space gradient probes retain
        # the student graph and perform three extra autograd traversals per valid
        # sample. When enabled, scalar locality/pooling metrics and the exact
        # selected-logit gradient comparison are emitted by training_step.
        self.diagnostics_enabled = _env_flag("MP_OPD_DIAGNOSTICS", False)
        self.diagnostics_logit_grad = _env_flag(
            "MP_OPD_DIAGNOSTICS_LOGIT_GRAD", self.diagnostics_enabled
        )
        raw_every = os.environ.get("MP_OPD_DIAGNOSTICS_EVERY", "1")
        try:
            self.diagnostics_every = max(1, int(raw_every))
        except ValueError as error:
            raise ValueError("MP_OPD_DIAGNOSTICS_EVERY must be a positive integer") from error
        self._diagnostic_step = 0
        self.random_seed = int(self.args.kd.mp_opd_random_seed)
        self.energy = None
        self.energy_optimizer = None
        self.student_updates = 0
        self.energy_updates = 0
        self._meta_gradient = False
        if self.mode == "soft":
            checkpoint = self.args.kd.mp_opd_energy_checkpoint
            if not checkpoint:
                raise ValueError("mp_opd soft mode requires mp_opd_energy_checkpoint")
            self.energy = MPAtomEnergy(
                self.FEATURE_DIM,
                hidden_dim=int(self.args.kd.mp_opd_energy_hidden_dim),
                layers=int(self.args.kd.mp_opd_energy_layers),
            ).to(self.teacher_lm_head.weight.device)
            self.energy_optimizer = torch.optim.AdamW(
                self.energy.parameters(), lr=float(self.args.kd.mp_opd_energy_lr), weight_decay=0.0
            )
            load_energy_checkpoint(
                checkpoint,
                self.energy,
                self.energy_optimizer,
                expected_extra_config={"max_span_length": self.max_span_length},
            )
            self.energy.eval()
            if getattr(self.args.kd, "mp_opd_alternating", False):
                # Same initial energy weights as frozen, fresh meta optimizer.
                self.energy_optimizer = torch.optim.AdamW(self.energy.parameters(),
                    lr=self.args.kd.mp_opd_energy_lr, weight_decay=0.)

    def training_state_dict(self):
        return {"student_updates":self.student_updates,"energy_updates":self.energy_updates}

    def load_training_state_dict(self, state):
        self.student_updates=state["student_updates"]
        self.energy_updates=state["energy_updates"]

    def note_optimizer_updates(self, count):
        self.student_updates += count

    def update_energy_full(self, batches, meta_rows, optimizer, *, batch_loader=lambda batch:batch):
        from ._mp_opd_full_meta import full_meta_step, ForwardParameterBridge, memory_event, streamed_inner_losses
        args=self.args.kd
        if not energy_update_due(self.student_updates, args.mp_opd_energy_every):
            return {"mp_opd_energy_updates_total":float(self.energy_updates)}
        device=next(self.student.parameters()).device
        tok=self.student_tokenizer
        encoded=[]
        for row in meta_rows:
            prefix=tok.encode(row['prompt'],add_special_tokens=False)
            answer=tok.encode(row['reference'],add_special_tokens=False)
            if tok.eos_token_id is not None and (not answer or answer[-1]!=tok.eos_token_id):
                answer.append(tok.eos_token_id)
            if not prefix or not answer or len(prefix)+len(answer)>self.args.data.max_len:
                raise ValueError('Meta reference exceeds sequence contract; no silent truncation')
            encoded.append((prefix,answer))
        def meta_loss(rows):
            size=max(len(p)+len(a) for p,a in rows)
            memory_event('meta_forward',device,shape=[len(rows),size])
            pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            ids=torch.full((len(rows),size),pad,dtype=torch.long,device=device)
            mask=torch.zeros_like(ids); response=torch.zeros_like(ids,dtype=torch.bool)
            for i,(p,a) in enumerate(rows):
                ids[i,:len(p)+len(a)]=torch.tensor(p+a,device=device)
                mask[i,:len(p)+len(a)]=1
                response[i,len(p)-1:len(p)+len(a)-1]=True
            logits=self.student(ids,attention_mask=mask,allgather_logits=True,
                ring_attn_group=self.strategy.ring_attn_group)['logits']
            losses=torch.nn.functional.cross_entropy(logits[response].float(),
                ids.roll(-1,1)[response],reduction='none')
            cursor=0; total=losses.new_zeros(())
            for _,a in rows:
                total=total+losses[cursor:cursor+len(a)].mean()/len(encoded);cursor+=len(a)
            return total
        inner=streamed_inner_losses(batches,batch_loader,self.training_step,device)
        outer=[lambda rows=encoded[i:i+args.mp_opd_meta_microbatch_size]:meta_loss(rows)
               for i in range(0,len(encoded),args.mp_opd_meta_microbatch_size)]
        self._meta_gradient=True
        def refresh_parameters():
            # FSDP2.reshard is nonrecursive: discard every cached unsharded
            # parameter view before installing or rolling back virtual weights.
            from torch.distributed.fsdp import FSDPModule
            for module in self.student.modules():
                if isinstance(module,FSDPModule): module.reshard()
        refresh_parameters()
        bridge = ForwardParameterBridge(self.student)
        try:
            result=full_meta_step(tuple(p for p in self.student.parameters() if p.requires_grad),
                optimizer,self.energy,self.energy_optimizer,inner,outer,max_norm=self.args.train.max_norm,
                refresh_parameters=refresh_parameters, parameter_grad=bridge.grad,
                offload_adam_moments=args.mp_opd_offload_adam_moments)
        finally:
            bridge.close()
            self._meta_gradient=False
        self.energy_updates+=1
        result['mp_opd_energy_updates_total']=float(self.energy_updates)
        return result

    def get_energy_params(self):
        """Explicitly separate from ``get_projector_params``/student optimizer."""
        return [] if self.energy is None else list(self.energy.parameters())

    def _partition_loss(
        self,
        credits,
        atoms,
        micro_batch,
        sample_index: int,
        *,
        diagnostics: bool = False,
        diagnostic_seed: int = 0,
        student_logits: torch.Tensor | None = None,
        student_labels: torch.Tensor | None = None,
    ):
        n = len(atoms)
        metrics = {}

        def hard_loss_with_metrics(partition):
            loss = hard_partition_loss(
                credits.current_nll, credits.base_credit, credits.weight, partition
            )
            if diagnostics:
                metrics.update(
                    partition_metrics(
                        credits.base_credit,
                        credits.weight,
                        partition,
                        shuffle_seed=diagnostic_seed,
                    )
                )
                if self.diagnostics_logit_grad and student_logits is not None:
                    metrics.update(
                        logit_gradient_metrics(
                            student_logits,
                            credits.student_token_nll,
                            credits.rate,
                            credits.weight,
                            partition,
                            shuffle_seed=diagnostic_seed,
                        )
                    )
            return loss, metrics

        if self.mode == "atomic":
            partition = fixed_partition(n, 1)
            return hard_loss_with_metrics(partition)
        if self.mode == "fixed":
            partition = fixed_partition(n, self.fixed_span_length)
            return hard_loss_with_metrics(partition)
        if self.mode == "random":
            material = (
                f"{self.random_seed}:{sample_index}:"
                f"{credits.weight.detach().cpu().tolist()}"
            ).encode()
            seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
            partition = random_partition(n, self.max_span_length, seed, self.min_span_length)
            return hard_loss_with_metrics(partition)

        if self.mode == "gbv":
            if self.gbv_geometry == "exact_logit":
                if student_logits is None:
                    raise RuntimeError(
                        "mp_opd gbv exact_logit geometry requires the student logits"
                    )
                if credits.student_token_nll is None:
                    raise RuntimeError(
                        "mp_opd gbv exact_logit geometry requires per-token student NLL"
                    )
                if credits.student_token_nll.numel() != student_logits.shape[0]:
                    raise ValueError(
                        "per-token student NLL and student logits disagree on token count"
                    )
                sensitivity = atom_logit_sensitivity(
                    student_logits,
                    student_labels,
                    tuple((atom.student_start, atom.student_end) for atom in atoms),
                    selected_log_prob=-credits.student_token_nll,
                )
            else:
                # q_i = w_i is the documented token-count approximation; it turns the
                # objective into the weighted Potts special case and costs nothing.
                sensitivity = credits.weight
            gbv_tables = gbv_span_costs(
                credits.rate,
                credits.weight,
                sensitivity,
                self.max_span_length,
                self.gbv_beta,
            )
            gbv_selected = gbv_partition(gbv_tables)
            metrics.update(
                gbv_partition_metrics(
                    gbv_tables, gbv_selected, credits.rate, credits.weight
                )
            )
            metrics["mp_opd_gbv_beta"] = credits.rate.new_tensor(self.gbv_beta)
            metrics["mp_opd_gbv_exact_geometry"] = credits.rate.new_tensor(
                float(self.gbv_geometry == "exact_logit")
            )
            return hard_loss_with_metrics(gbv_selected)

        _b, _w, rates, valid = span_tables(
            credits.base_credit, credits.weight, self.max_span_length
        )
        if self.mode == "oracle":
            supplied = micro_batch.get("mp_opd_atom_directional_scores")
            if supplied is None:
                raise RuntimeError(
                    "mp_opd oracle is instrumentation-only and requires detached "
                    "mp_opd_atom_directional_scores"
                )
            z = supplied[sample_index].to(credits.base_credit).detach()
            if z.numel() != n:
                raise ValueError("oracle directional-score cardinality mismatch")
            utilities, utility_valid = span_utility_table(
                credits.base_credit, credits.weight, z, self.max_span_length
            )
            oracle = hard_max_partition(utilities, utility_valid)
            metrics["mp_opd_delta_pred"] = oracle.score.detach()
            return hard_loss_with_metrics(oracle.partition)

        if self.mode == "soft":
            features = atom_features(atoms, credits)
            energies = self.energy(features, self.max_span_length)
            partition_started = time.perf_counter() if self.timing_enabled else None
            distribution = semi_markov_partition(
                energies, temperature=self.temperature, valid_mask=valid, host_mask=self.host_mask
            )
            partition_seconds = (time.perf_counter() - partition_started
                                 if partition_started is not None else None)
            if not torch.isfinite(distribution.coverage_max_error) or distribution.coverage_max_error > 1e-6:
                raise RuntimeError("MP-OPD partition coverage failed after FP64 DP; inspect before retry")
            rates_per_atom = expected_atom_rates(distribution.marginals, rates)
            metrics.update(
                {
                    "mp_opd_log_z": distribution.log_z.detach(),
                    "mp_opd_partition_entropy": distribution.entropy.detach(),
                    "mp_opd_expected_span_count": distribution.expected_span_count.detach(),
                    "mp_opd_expected_span_length": distribution.expected_span_length.detach(),
                    "mp_opd_marginal_coverage_max_error": distribution.coverage_max_error.detach(),
                    "mp_opd_credit_conservation_residual": credit_conservation_residual(
                        credits.base_credit, credits.weight, rates_per_atom
                    ).detach(),
                }
            )
            if diagnostics:
                atomic_rate = credits.rate.detach()
                effective_rate = rates_per_atom.detach()
                metrics.update(
                    {
                        "mp_opd_diag_effective_rate_std": effective_rate.float().std(unbiased=False),
                        "mp_opd_diag_effective_vs_atomic_rate_rmse": (
                            (effective_rate.float() - atomic_rate.float()).square().mean().sqrt()
                        ),
                        "mp_opd_diag_effective_rate_sign_flip_fraction": (
                            (effective_rate * atomic_rate < 0).float().mean()
                        ),
                    }
                )
            if partition_seconds is not None:
                metrics["mp_opd_semimarkov_wall_seconds"] = torch.tensor(
                    partition_seconds, device=distribution.log_z.device, dtype=torch.float64
                )
            # Student path treats q_phi as fixed; phi has its own optimizer.
            if self._meta_gradient:
                return (credits.current_nll*rates_per_atom).sum(), metrics
            return soft_partition_loss(credits.current_nll, rates_per_atom.detach()), metrics
        raise AssertionError(f"unhandled MP-OPD mode {self.mode}")

    def training_step(self, micro_batch):
        started = time.perf_counter()
        self._diagnostic_step += 1
        diagnostics_due = (
            self.diagnostics_enabled
            and self._diagnostic_step % self.diagnostics_every == 0
        )
        student_input_ids = micro_batch["stu_input_ids"]
        student_attn_mask = micro_batch["stu_attn_mask"]
        student_loss_mask = micro_batch["stu_loss_mask"].bool()
        teacher_input_ids = micro_batch["tea_input_ids"]
        teacher_loss_mask = micro_batch["tea_loss_mask"].bool()
        teacher_hiddens = micro_batch.get("teacher_hiddens")
        avg_token_num = micro_batch["avg_micro_batch_token_num"]
        if teacher_hiddens is None:
            raise RuntimeError("micro_batch must contain teacher_hiddens for MP-OPD")

        mm_kwargs = {key[3:]: value for key, value in micro_batch.items() if key.startswith("mm_")}
        output = self.student(
            student_input_ids,
            attention_mask=student_attn_mask,
            allgather_logits=True,
            ring_attn_group=self.strategy.ring_attn_group,
            **mm_kwargs,
        )
        student_logits_flat = output["logits"][student_loss_mask]
        student_labels = student_input_ids.roll(shifts=-1, dims=1)
        teacher_labels = teacher_input_ids.roll(shifts=-1, dims=1)
        parity_metrics = {}
        behavior = micro_batch.get("stu_behavior_log_probs")
        if behavior is not None:
            selected = behavior[student_loss_mask]
            labels_for_parity = student_labels[student_loss_mask]
            try:
                parity_metrics = _behavior_parity_metrics(
                    student_logits_flat,
                    labels_for_parity,
                    selected,
                    self.args.rollout.temperature,
                )
            except RuntimeError as error:
                from ._parity_capture import capture_failure
                try:
                    capture_failure(self, micro_batch, student_logits_flat,
                                    labels_for_parity, selected,
                                    self.args.rollout.temperature, error)
                except Exception as capture_error:
                    print(f"PARITY_CAPTURE_FAILED: {type(capture_error).__name__}: {capture_error}", flush=True)
                raise
        teacher_logits_flat = self.teacher_lm_head(
            teacher_hiddens.to(self.teacher_lm_head.weight)
        )

        # Keep the zero baseline connected to the student graph. A fail-closed
        # sample can occupy an entire microbatch (the production recipe uses
        # micro_train_batch_size=1); it must contribute zero gradient and
        # explicit invalid-sample telemetry without aborting the surrounding
        # gradient-accumulation window.
        total_loss = student_logits_flat.sum() * 0.0
        total_atoms = total_invalid = total_one = total_multi = 0
        invalid_reasons: dict[str, int] = {}
        covered_student = covered_teacher = masked_eos = 0
        candidate_spans = 0
        byte_lengths = []
        student_lengths = []
        teacher_lengths = []
        credit_values = []
        rate_values = []
        extra_sums: dict[str, torch.Tensor] = {}
        stu_offset = tea_offset = 0
        valid_samples = 0
        for batch_index in range(student_input_ids.shape[0]):
            stu_mask = student_loss_mask[batch_index]
            tea_mask = teacher_loss_mask[batch_index]
            stu_count = int(stu_mask.sum().item())
            tea_count = int(tea_mask.sum().item())
            stu_logits = student_logits_flat[stu_offset : stu_offset + stu_count]
            tea_logits = teacher_logits_flat[tea_offset : tea_offset + tea_count]
            stu_offset += stu_count
            tea_offset += tea_count
            stu_ids = student_labels[batch_index][stu_mask].detach().cpu().tolist()
            tea_ids = teacher_labels[batch_index][tea_mask].detach().cpu().tolist()
            sample_key = hashlib.sha256(
                bytes(str(stu_ids), "utf-8") + b"|" + bytes(str(tea_ids), "utf-8")
            ).hexdigest()[:16]
            atomized = self.atomizer.atomize(stu_ids, tea_ids, sample_id=sample_key)
            masked_eos += atomized.masked_student_eos + atomized.masked_teacher_eos
            if not atomized.valid:
                total_invalid += 1
                reason = atomized.failure_reason or "unknown"
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
                continue
            atoms = atomized.atoms
            stu_label_tensor = torch.tensor(stu_ids, device=stu_logits.device)
            credits = build_atom_credits(
                atoms,
                stu_logits,
                stu_label_tensor,
                tea_logits,
                torch.tensor(tea_ids, device=tea_logits.device),
            )
            sample_loss, sample_metrics = self._partition_loss(
                credits,
                atoms,
                micro_batch,
                batch_index,
                diagnostics=diagnostics_due,
                diagnostic_seed=int(sample_key, 16),
                student_logits=stu_logits if (diagnostics_due or self.gbv_needs_logits) else None,
                student_labels=stu_label_tensor,
            )
            total_loss = total_loss + sample_loss
            for key, value in sample_metrics.items():
                extra_sums[key] = extra_sums.get(key, value.new_zeros(())) + value
            valid_samples += 1
            total_atoms += len(atoms)
            total_one += sum(atom.boundary_type == "one_to_one" for atom in atoms)
            total_multi += sum(atom.boundary_type == "multi_token" for atom in atoms)
            covered_student += atomized.covered_student_events
            covered_teacher += atomized.covered_teacher_events
            candidate_spans += sum(min(self.max_span_length, len(atoms) - i) for i in range(len(atoms)))
            byte_lengths.extend(atom.byte_end - atom.byte_start for atom in atoms)
            student_lengths.extend(atom.student_token_count for atom in atoms)
            teacher_lengths.extend(atom.teacher_token_count for atom in atoms)
            credit_values.append(credits.base_credit)
            rate_values.append(credits.rate)

        kd_loss = total_loss / avg_token_num
        metrics = {
            "loss": kd_loss,
            "kd_loss": kd_loss,
            "mp_opd_diag_enabled": kd_loss.new_tensor(float(self.diagnostics_enabled)),
            "mp_opd_diag_active": kd_loss.new_tensor(float(diagnostics_due)),
            "mp_opd_diag_step": kd_loss.new_tensor(float(self._diagnostic_step)),
            "mp_opd_valid_atom_count": kd_loss.new_tensor(float(total_atoms)),
            "mp_opd_invalid_sample_count": kd_loss.new_tensor(float(total_invalid)),
            "mp_opd_valid_sample_count": kd_loss.new_tensor(float(valid_samples)),
            "mp_opd_one_to_one_atom_ratio": kd_loss.new_tensor(total_one / max(total_atoms, 1)),
            "mp_opd_multi_token_atom_ratio": kd_loss.new_tensor(total_multi / max(total_atoms, 1)),
            "mp_opd_candidate_span_count": kd_loss.new_tensor(float(candidate_spans)),
            "mp_opd_masked_eos_count": kd_loss.new_tensor(float(masked_eos)),
            "valid_student_tokens": kd_loss.new_tensor(float(covered_student)),
            "valid_teacher_tokens": kd_loss.new_tensor(float(covered_teacher)),
            "mp_opd_fail_closed_sample_ratio": kd_loss.new_tensor(
                total_invalid / max(total_invalid + valid_samples, 1)
            ),
            "mp_opd_atomization_and_loss_seconds": kd_loss.new_tensor(time.perf_counter() - started),
        }
        metrics.update(parity_metrics)
        if credit_values:
            metrics.update(_finite_stats("mp_opd_b", torch.cat(credit_values)))
            metrics.update(_finite_stats("mp_opd_r", torch.cat(rate_values)))
        metrics.update(
            {
                "mp_opd_atom_byte_length_mean": kd_loss.new_tensor(sum(byte_lengths) / max(len(byte_lengths), 1)),
                "mp_opd_student_atom_token_length_mean": kd_loss.new_tensor(sum(student_lengths) / max(len(student_lengths), 1)),
                "mp_opd_teacher_atom_token_length_mean": kd_loss.new_tensor(sum(teacher_lengths) / max(len(teacher_lengths), 1)),
            }
        )
        for reason, count in sorted(invalid_reasons.items()):
            metrics[f"mp_opd_invalid_reason_{reason}"] = kd_loss.new_tensor(float(count))
        for key, value in extra_sums.items():
            metrics[key] = value / max(valid_samples, 1)
        if self.args.kd.kd_ratio < 1:
            ce_labels = student_labels[student_loss_mask]
            ce_loss = compute_cross_entropy(student_logits_flat, ce_labels, reduction="sum") / avg_token_num
            metrics["ce_loss"] = ce_loss
            metrics["loss"] = (1 - self.args.kd.kd_ratio) * ce_loss + self.args.kd.kd_ratio * kd_loss
        # One device/host synchronization on the healthy path. Preserve the
        # offending metric name on failure without synchronizing every scalar.
        finite = torch.stack([torch.isfinite(v.detach()).all() for v in metrics.values()])
        if not finite.all():
            for key, ok in zip(metrics, finite.cpu().tolist()):
                if not ok:
                    raise FloatingPointError(f"non-finite MP-OPD metric: {key}")
        return metrics
