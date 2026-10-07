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
from ._mp_opd_credit_transform import (
    CreditTransformSpec,
    credit_transform_from_args,
    production_credit_batch,
)
from ._mp_opd_energy import MPAtomEnergy, load_energy_checkpoint
from ._mp_opd_gbv_span import (
    atom_logit_sensitivity,
    gbv_partition,
    gbv_partition_metrics,
    gbv_span_costs,
)
from ._mp_opd_grass_span import (
    CREDIT_CONSERVATION_REL_TOL,
    GrassNoiseEstimator,
    atom_head_gram,
    grass_candidate_metrics,
    grass_credit_residual,
    grass_gram_diagnostics,
    grass_partition,
    grass_partition_metrics,
    grass_span_costs,
    grass_span_ids,
    grass_shrink,
    grass_update_energy,
    hard_pooled_credits,
)
from ._mp_opd_grass_chunk import (
    atom_chunk_ids_from_tokens,
    apply_chunk_shrinkage,
    chunk_boundary_diagnostics,
    chunk_gram_diagnostics,
    chunk_head_gram,
    chunk_partition,
    chunk_shadow_metrics,
    grass_chunk_metrics,
    grass_chunk_tables,
    run_chunk_assignment,
)
from ._mp_opd_grass_span import grass_cosine
from ._mp_opd_airs import (
    airs_metrics,
    airs_precision_buckets,
    airs_shrinkage,
)
from ._mp_opd_oracle import hard_max_partition, span_utility_table
from ._mp_opd_semimarkov import semi_markov_partition
from ._mp_opd_training_diagnostics import (
    logit_gradient_metrics,
    partition_metrics,
)


# Modes that shrink credit with the shared SURE geometry. GRASS-DP searches for
# the partition; GRASS-Chunk takes it from the upstream alignment. Both need the
# same logits, the same head-input hidden states and the same noise-scale state.
_GRASS_MODES = frozenset({"grass", "grass_chunk"})

# AIRS needs none of the GRASS machinery: no logits, no hidden states, no Gram. It
# shares only the noise estimator, because the method note fixes the same MAD-of-
# adjacent-differences scale so the two methods are compared under one noise model.
_AIRS_MODES = frozenset({"airs"})


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
    logits = student_logits.detach().float()
    actual = (
        (logits / temperature)
        .log_softmax(dim=-1)
        .gather(-1, labels.unsqueeze(-1))
        .squeeze(-1)
    )
    delta = (actual[real] - behavior_log_probs[real]).abs()
    if not torch.isfinite(delta).all():
        raise RuntimeError("behavior/trainer logprob parity produced non-finite deltas")

    # Diagnostic-only: SGLang's standard sampler divides the logits by the request
    # temperature in place and returns log(softmax(...)). If `temperatures` is 1.0
    # for a stretch of decode steps that division is a no-op and the stored logprob
    # is the RAW one, which is the 2026-09 temperature bug in a partial form. The
    # fraction below is the calibrated detector for exactly that: it reads ~0.99 on
    # a B200 when the engine genuinely returns raw logprobs, and 0.004-0.037 when it
    # scales correctly (measured over 123k tokens, google/gemma-2-2b-it). It changes
    # nothing about the pass/fail decision; it makes the failure self-describing.
    # With temperature == 1 the two references coincide and the statistic carries no
    # information, so it is reported as NaN rather than a misleading 0.
    if abs(temperature - 1.0) < 1e-9:
        closer_to_raw = torch.tensor(float("nan"))
    else:
        raw = logits.log_softmax(dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        closer_to_raw = (
            (behavior_log_probs[real] - raw[real]).abs()
            < (behavior_log_probs[real] - actual[real]).abs()
        ).float().mean()

    mean = delta.mean()
    maximum = delta.max()
    p99 = torch.quantile(delta, 0.99)
    # Backend/precision tails can contain isolated finite outliers. Fail on a
    # distributional mismatch while preserving the tail as diagnostics.
    if mean > 0.1 or p99 > 0.5:
        diagnosis = ""
        if torch.isfinite(closer_to_raw) and closer_to_raw.item() > 0.1:
            diagnosis = (
                "; closer_to_raw="
                f"{closer_to_raw.item():.4f} means the engine returned RAW, unscaled "
                "logprobs for part or all of the trajectory (expected <= 0.04)"
            )
        raise RuntimeError(
            "behavior/trainer logprob parity failed: "
            f"mean={mean.item():.6f}, p99={p99.item():.6f}, max={maximum.item():.6f}"
            + diagnosis
        )
    return {
        "trajectory_logprob_abs_mean": mean,
        "trajectory_logprob_abs_p99": p99,
        "trajectory_logprob_abs_max": maximum,
        "trajectory_logprob_above_0p5_fraction": (delta > 0.5).float().mean(),
        "trajectory_logprob_closer_to_raw_fraction": closer_to_raw,
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
        # GRASS shrinks credits with a SURE-optimal strength derived from the
        # exact output-head atom Gram, so unlike GBV it cannot be reduced to a
        # token-count proxy: it needs the logits *and* the hidden state that the
        # LM head actually consumes.
        self.grass_geometry = str(getattr(self.args.kd, "mp_opd_grass_geometry", "exact_head"))
        if self.mode in _GRASS_MODES and self.grass_geometry not in {"exact_head", "diag"}:
            raise ValueError(f"unsupported mp_opd_grass_geometry: {self.grass_geometry}")
        self.grass_eps_d = float(getattr(self.args.kd, "mp_opd_grass_eps_d", 0.0))
        self.grass_negative_tol_rel = float(
            getattr(self.args.kd, "mp_opd_grass_negative_tol_rel", 1e-6)
        )
        self.grass_shadow = bool(getattr(self.args.kd, "mp_opd_grass_shadow", False))
        # GRASS-Chunk shares the statistic with GRASS-DP and differs only in where
        # the partition comes from, so it needs the same logits and the same head
        # input hidden states, and the same noise-scale state.
        self.grass_chunk_source = str(
            getattr(self.args.kd, "mp_opd_grass_chunk_source", "xtoken")
        )
        self.grass_chunk_run_length = int(
            getattr(self.args.kd, "mp_opd_grass_chunk_run_length", 2)
        )
        self.grass_chunk_straddle = str(
            getattr(self.args.kd, "mp_opd_grass_chunk_straddle", "singleton")
        )
        self.grass_chunk_shadow = bool(
            getattr(self.args.kd, "mp_opd_grass_chunk_shadow", False)
        )
        # The native synchronized chunk of section 2.1 is produced by the audited
        # cross-tokenizer aligner, the same one kd_algorithm='xtoken' uses. It is
        # built once here, and the projection is re-verified by digest here rather
        # than trusted, because a run whose chunks came from a different
        # projection is not the run the recipe names.
        # Built only for the mode that consumes it. Gating on the source alone let
        # the default mp_opd_grass_chunk_source='xtoken' build this for every mode,
        # so any mp_opd run without MP_XTOKEN_PROJECTION_PATH died in __init__ on
        # Path(None) before a single step - gbv, soft and atomic included, not just
        # grass_chunk.
        self.grass_aligner = (
            self._build_grass_aligner()
            if self.mode == "grass_chunk" and self.grass_chunk_source == "xtoken"
            else None
        )
        self._grass_modes = _GRASS_MODES
        self.grass_needs_logits = self.mode in _GRASS_MODES
        # Both geometries need the head input: the diag ablation drops the
        # off-diagonal band, but a multi-token atom's diagonal block still mixes
        # its own tokens through h_t, so the hidden states are not optional there.
        self.grass_needs_hidden = self.mode in _GRASS_MODES
        self.grass_noise = (
            GrassNoiseEstimator(
                rho=float(getattr(self.args.kd, "mp_opd_grass_sigma_rho", 0.99)),
                min_adjacent_pairs=int(getattr(self.args.kd, "mp_opd_grass_sigma_min_pairs", 8)),
            )
            if self.mode in _GRASS_MODES
            else None
        )
        self.grass_softcap = self._detect_final_logit_softcap()
        self.grass_head_bias = self._detect_head_bias()
        self.airs_warmup_steps = int(getattr(self.args.kd, "mp_opd_airs_warmup_steps", 20))
        self.airs_noise = (
            GrassNoiseEstimator(
                rho=float(getattr(self.args.kd, "mp_opd_airs_sigma_rho", 0.99)),
                min_adjacent_pairs=int(
                    getattr(self.args.kd, "mp_opd_airs_sigma_min_pairs", 8)
                ),
            )
            if self.mode in _AIRS_MODES
            else None
        )
        # Cross-atom credit operator. Atomic *is* the identity operator here, so
        # mode 'atomic' and mode 'kernel' with transform 'identity' share one path.
        self.credit_spec: CreditTransformSpec = credit_transform_from_args(self.args.kd)
        # Fail-closed regression probe: recompute the historical Atomic pooled loss
        # next to the identity-operator loss and raise on any drift.
        self.credit_identity_check = _env_flag("MP_OPD_CREDIT_IDENTITY_CHECK", False)
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

    def _build_grass_aligner(self):
        """The native cross-tokenizer synchronized chunker of section 2.1.

        The digest is re-verified here instead of trusting the args layer: a run
        whose chunks came from a different projection than the one the recipe
        names is a different experiment, and nothing downstream would notice.
        """
        projection_path = Path(self.args.kd.xtoken_projection_path)
        if not projection_path.is_file():
            raise FileNotFoundError(
                f"GRASS-Chunk X-Token projection not found: {projection_path}"
            )
        digest = hashlib.sha256(projection_path.read_bytes()).hexdigest()
        expected = str(self.args.kd.xtoken_projection_sha256).lower()
        if digest != expected:
            raise RuntimeError(
                "GRASS-Chunk X-Token projection SHA mismatch: expected "
                f"{expected}, got {digest}"
            )
        try:
            from xtoken_upstream_token_aligner import TokenAligner
        except ImportError as error:
            raise RuntimeError(
                "mp_opd_grass_chunk_source='xtoken' needs the vendored "
                "xtoken_upstream_token_aligner on PYTHONPATH "
                "(experiments/modal/vendor), the same dependency "
                "kd_algorithm='xtoken' has. Use "
                "mp_opd_grass_chunk_source='run' for the fixed-run baseline."
            ) from error
        return TokenAligner(
            self.student_tokenizer,
            self.teacher_tokenizer,
            str(projection_path),
            max_comb_len=int(self.args.kd.xtoken_max_comb_len),
        )

    def _detect_final_logit_softcap(self):
        """Final-logit softcapping of the student head, or ``None``.

        GRASS needs the *actual* head Jacobian. Gemma-2 squashes its final logits
        with ``sc * tanh(u/sc)``, so the returned logits are not ``W h + b`` and an
        unsquashed ``delta_t`` would describe a head the model does not have. The
        value is read from the model config rather than assumed, and
        ``MP_OPD_GRASS_SOFTCAP`` overrides it for hosts where the config is not
        reachable (``none``/``0`` to force an unsquashed head).
        """
        config = getattr(self.student, "model_config", None)
        softcap = getattr(config, "final_logit_softcapping", None) if config is not None else None
        if softcap is None and config is not None:
            # Some wrappers nest the decoder config. Missing this silently
            # describes an unsquashed head the student does not have, which changes
            # every entry of H and is exactly the section 17.23 failure signature,
            # so both spellings are consulted before giving up.
            text_config = getattr(config, "text_config", None)
            softcap = getattr(text_config, "final_logit_softcapping", None)
        override = os.environ.get("MP_OPD_GRASS_SOFTCAP", "").strip().lower()
        if override:
            if override in {"none", "0", "null", "off"}:
                return None
            try:
                return float(override)
            except ValueError as error:
                raise ValueError(
                    f"MP_OPD_GRASS_SOFTCAP={override!r} must be a number, 'none' or '0'"
                ) from error
        return None if softcap is None else float(softcap)

    def _detect_head_bias(self) -> bool:
        """Whether the student output head carries a trainable bias.

        A bias adds ``<sum_t delta_t, sum_s delta_s>`` to every Gram block. None of
        the student models this repo trains has one, but silently dropping the
        term would make ``H`` only approximately exact, so it is detected.
        """
        try:
            head = self.student.model.get_output_embeddings()
        except (AttributeError, NotImplementedError):
            head = None
        return getattr(head, "bias", None) is not None

    def training_state_dict(self):
        state = {"student_updates": self.student_updates, "energy_updates": self.energy_updates}
        # The GRASS noise scale is a running estimate, not a parameter, but losing
        # it on resume silently restarts GRASS from a deflated sigma and makes the
        # early steps of a resumed run look like the atomic failure signature.
        if self.grass_noise is not None:
            state["mp_opd_grass_noise"] = self.grass_noise.state_dict()
        # AIRS has its own noise state and its own warm-up counter. Both have to
        # survive a resume: a reset counter would replay the warm-up, and a reset
        # estimator would restart the EMA from a deflated variance.
        if self.airs_noise is not None:
            state["mp_opd_airs_noise"] = self.airs_noise.state_dict()
        # The warm-up position is read off student_updates, which is already in the
        # state and is counted once per optimizer update, so a resume continues the
        # warm-up where it stopped instead of replaying it.
        return state

    def load_training_state_dict(self, state):
        self.student_updates=state["student_updates"]
        self.energy_updates=state["energy_updates"]
        if self.grass_noise is not None and "mp_opd_grass_noise" in state:
            self.grass_noise.load_state_dict(state["mp_opd_grass_noise"])
        if self.airs_noise is not None and "mp_opd_airs_noise" in state:
            self.airs_noise.load_state_dict(state["mp_opd_airs_noise"])

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

    def _effective_credit(self, credits, n: int, micro_batch, sample_index: int):
        """Apply the configured credit operator to one sample's atomic credits.

        Production calls this once per sample, so the packed-batch axes are the
        degenerate ones: one sequence, every listed atom valid. Atoms the atomizer
        rejected never reach here, and the masked EOS tokens live outside every
        atom, exactly as in the Atomic credit path.
        """
        metadata = {}
        if self.credit_spec.name == "external":
            supplied = micro_batch.get("mp_opd_atom_future_advantage")
            if supplied is None:
                raise RuntimeError(
                    "the external credit operator requires a detached "
                    "mp_opd_atom_future_advantage per sample"
                )
            advantage = supplied[sample_index]
            if torch.as_tensor(advantage).numel() != n:
                raise ValueError("future-advantage cardinality mismatch")
            metadata["future_advantage"] = advantage
        batch = production_credit_batch(
            credits.rate, credits.base_credit, credits.weight, metadata
        )
        output = self.credit_spec.transform(batch, training=True)
        return output.effective_credit, output.diagnostics

    def _airs_loss(self, credits, atoms, *, sample_index=None):
        """AIRS: shrink each atomic credit toward zero by its own reliability.

        The whole method is the substitution ``r -> lambda * r`` where
        ``lambda_i = [1 - sigma2 / (w_i * r_i^2)]_+``. No atom is merged with
        another, no span is formed and no Gram is built, so the update keeps the
        atomic supervision resolution and changes nothing else in the loss.

        Credit mass is deliberately *not* restored afterwards: renormalising to
        the original weighted mass would divide out exactly the suppression the
        method exists to apply.
        """
        if self.airs_noise is None:
            raise RuntimeError("mp_opd airs mode requires the AIRS noise estimator")
        # One noise update per valid response, matching GRASS, so the two methods
        # see the same estimator driven by the same statistic.
        noise = self.airs_noise.update(credits.rate, credits.weight)
        # Measured in optimizer updates, not micro-batches: with grad accumulation a
        # micro-batch counter would end a 20-step warm-up after roughly one step and
        # make the configured length a lie. student_updates is the same counter the
        # checkpoint carries, so a resume does not replay the warm-up.
        completed_updates = int(self.student_updates)
        enabled = completed_updates >= self.airs_warmup_steps
        shrinkage = airs_shrinkage(
            credits.rate,
            credits.weight,
            self.airs_noise.sigma2,
            enabled=enabled,
        )
        metrics = airs_metrics(
            credits.rate, credits.weight, shrinkage, enabled=enabled
        )
        metrics.update(airs_precision_buckets(credits.weight, shrinkage))
        metrics.update(
            {
                "mp_opd_airs_noise_valid_pairs": credits.rate.new_tensor(
                    float(noise["valid_pairs"])
                ),
                "mp_opd_airs_noise_sigma_batch": credits.rate.new_tensor(
                    float(noise["sigma_batch"])
                ),
                "mp_opd_airs_noise_sigma_batch_variance": credits.rate.new_tensor(
                    float(noise["sigma_batch_variance"])
                ),
                "mp_opd_airs_noise_sigma2": credits.rate.new_tensor(
                    float(self.airs_noise.sigma2)
                ),
                "mp_opd_airs_noise_z_abs_mean": credits.rate.new_tensor(
                    float(noise["z_abs_mean"])
                ),
                "mp_opd_airs_noise_z_abs_p99": credits.rate.new_tensor(
                    float(noise["z_abs_quantile"])
                ),
                "mp_opd_airs_warmup_remaining": credits.rate.new_tensor(
                    float(max(self.airs_warmup_steps - completed_updates, 0))
                ),
            }
        )
        loss = soft_partition_loss(credits.current_nll, shrinkage["credit"])
        return loss, metrics

    def _grass_loss(
        self,
        credits,
        atoms,
        *,
        student_logits: torch.Tensor | None,
        student_labels: torch.Tensor | None,
        student_hidden: torch.Tensor | None,
    ):
        """GRASS: pick spans and a shrinkage strength by local output-head risk.

        Everything the selector needs is already on the batch - atomic credits,
        student logits and the hidden state the LM head consumes - so the mode
        costs one ``output_hidden_states`` flag and no extra forward pass. The
        rest of the OPD loss, clipping and optimizer pipeline is untouched.
        """
        if student_logits is None or student_labels is None:
            raise RuntimeError("mp_opd grass mode requires the student logits and labels")
        if student_hidden is None:
            raise RuntimeError(
                "mp_opd grass mode requires the student LM-head hidden states"
            )
        if credits.student_token_nll is None:
            raise RuntimeError("mp_opd grass mode requires per-token student NLL")
        if credits.student_token_nll.numel() != student_logits.shape[0]:
            raise ValueError(
                "per-token student NLL and student logits disagree on token count"
            )

        atom_ranges = tuple((atom.student_start, atom.student_end) for atom in atoms)
        covered = atom_ranges[-1][1]
        # One noise-scale update per valid response; the estimator is training-level
        # so the EMA survives across steps and across a resume.
        noise = self.grass_noise.update(credits.rate, credits.weight)
        head = atom_head_gram(
            student_logits[:covered],
            student_hidden[:covered],
            student_labels[:covered],
            atom_ranges,
            self.max_span_length,
            selected_log_prob=-credits.student_token_nll[:covered],
            softcap=self.grass_softcap,
            head_bias=self.grass_head_bias,
            diagonal_only=self.grass_geometry == "diag",
        )
        tables = grass_span_costs(
            credits.rate,
            credits.weight,
            head.gram,
            self.grass_noise.sigma2,
            self.max_span_length,
            eps_d=self.grass_eps_d,
            negative_tol_rel=self.grass_negative_tol_rel,
        )
        partition, _cost, margins = grass_partition(tables)
        shrunk, _exact_residual = grass_shrink(
            credits.rate, credits.weight, partition, tables.alpha
        )
        # The assertion runs on the coefficients the loss really multiplies by, not
        # on the float64 intermediate: section 18.1 is about the update, and the
        # update sees float32 rates.
        effective = shrunk.to(credits.rate.dtype)
        residual = grass_credit_residual(credits.rate, credits.weight, effective)
        # The tolerance is scaled by sum |w_i r_i|, not |sum w_i r_i|. The signed
        # sum is conserved exactly and can cancel to ~0 on a response with mixed-sign
        # atoms, while the float32 round-off of `effective` is proportional to the
        # unsigned magnitude. Scaling on the signed value would abort a healthy run
        # over rounding.
        magnitude = float((credits.weight * credits.rate).abs().sum())
        tolerance = CREDIT_CONSERVATION_REL_TOL * max(magnitude, 1.0)
        if float(residual) > tolerance:
            raise RuntimeError(
                "GRASS violated weighted credit conservation: max response residual "
                f"{float(residual):.6g} exceeds {tolerance:.6g}; the shrinkage is "
                "not credit-preserving and the run is not comparable"
            )

        metrics = grass_partition_metrics(
            tables,
            partition,
            credits.rate,
            credits.weight,
            head.gram,
            effective,
            margins,
            self.max_span_length,
            conservation_error=residual,
        )
        metrics.update(grass_candidate_metrics(tables, partition))
        metrics.update(grass_gram_diagnostics(head.gram, self.max_span_length))
        metrics.update(
            {
                "mp_opd_grass_exact_geometry": credits.rate.new_tensor(
                    float(self.grass_geometry == "exact_head")
                ),
                "mp_opd_grass_gram_symmetry_error": credits.rate.new_tensor(
                    head.symmetry_error
                ),
                "mp_opd_grass_atom_token_count": credits.rate.new_tensor(
                    float(head.token_count)
                ),
                "mp_opd_grass_sigma_batch": credits.rate.new_tensor(noise["sigma_batch"]),
                "mp_opd_grass_sigma_batch_variance": credits.rate.new_tensor(
                    noise["sigma_batch_variance"]
                ),
                "mp_opd_grass_noise_valid_pairs": credits.rate.new_tensor(
                    noise["valid_pairs"]
                ),
                "mp_opd_grass_noise_z_abs_mean": credits.rate.new_tensor(
                    noise["z_abs_mean"]
                ),
                "mp_opd_grass_noise_z_abs_p99": credits.rate.new_tensor(
                    noise["z_abs_quantile"]
                ),
                "mp_opd_grass_softcap": credits.rate.new_tensor(
                    float(self.grass_softcap or 0.0)
                ),
                "mp_opd_grass_head_bias": credits.rate.new_tensor(
                    float(self.grass_head_bias)
                ),
            }
        )
        if self.grass_shadow:
            metrics.update(
                self._grass_shadow(credits, partition, head.gram, effective)
            )
        return soft_partition_loss(credits.current_nll, effective), metrics

    def _grass_chunk_assignment(self, atoms, student_ids, teacher_ids):
        """Map this response's atoms onto the upstream alignment chunks.

        With ``source='xtoken'`` the aligner is run on the same loss-masked token
        axis the atomizer used, so a chunk boundary and an atom boundary live in
        the same coordinates and their disagreement is directly measurable.
        """
        n = len(atoms)
        if self.grass_chunk_source == "run":
            return run_chunk_assignment(n, self.grass_chunk_run_length)
        alignment = self.grass_aligner.align(
            torch.tensor([list(student_ids)], dtype=torch.long),
            torch.tensor([list(teacher_ids)], dtype=torch.long),
        )
        token_chunk_ids = alignment.student_chunk_id[0].tolist()
        atom_ranges = tuple((atom.student_start, atom.student_end) for atom in atoms)
        return atom_chunk_ids_from_tokens(
            token_chunk_ids, atom_ranges, policy=self.grass_chunk_straddle
        )

    def _grass_chunk_loss(
        self,
        credits,
        atoms,
        *,
        student_logits: torch.Tensor | None,
        student_labels: torch.Tensor | None,
        student_hidden: torch.Tensor | None,
        student_ids: list[int] | None = None,
        teacher_ids: list[int] | None = None,
    ):
        """GRASS-Chunk: shrink inside alignment chunks, never search for them.

        The statistic, the Gram and the conservation invariant are the GRASS ones,
        reused unchanged. What is removed is the whole decision layer: no
        candidate spans, no maximum span length, no dynamic program, no
        selection cost. The partition is the upstream alignment's, and the SURE
        gain is reported without ever being allowed to move a boundary.
        """
        if student_logits is None or student_labels is None:
            raise RuntimeError(
                "mp_opd grass_chunk mode requires the student logits and labels"
            )
        if student_hidden is None:
            raise RuntimeError(
                "mp_opd grass_chunk mode requires the student LM-head hidden states"
            )
        if credits.student_token_nll is None:
            raise RuntimeError(
                "mp_opd grass_chunk mode requires per-token student NLL"
            )
        if credits.student_token_nll.numel() != student_logits.shape[0]:
            raise ValueError(
                "per-token student NLL and student logits disagree on token count"
            )
        if self.grass_chunk_source == "xtoken" and (
            student_ids is None or teacher_ids is None
        ):
            raise RuntimeError(
                "mp_opd grass_chunk_source='xtoken' needs the sampled student and "
                "teacher token ids for the alignment"
            )

        atom_ranges = tuple((atom.student_start, atom.student_end) for atom in atoms)
        covered = atom_ranges[-1][1]
        assignment = self._grass_chunk_assignment(atoms, student_ids, teacher_ids)
        partition, noncontiguous = chunk_partition(assignment)
        span_ids = grass_span_ids(partition, len(atoms), credits.rate.device)
        # Section 11: a pair that crosses an alignment chunk boundary is exactly
        # where the latent credit is allowed to jump, so it must not inform the
        # noise scale. Every response here is single-chunked, so all pairs whose
        # two atoms fall in different chunks are excluded.
        within_pair = span_ids[:-1] == span_ids[1:] if span_ids.numel() > 1 else None
        noise = self.grass_noise.update(
            credits.rate, credits.weight, same_group=within_pair
        )
        head = chunk_head_gram(
            student_logits[:covered],
            student_hidden[:covered],
            student_labels[:covered],
            atom_ranges,
            partition,
            selected_log_prob=-credits.student_token_nll[:covered],
            softcap=self.grass_softcap,
            head_bias=self.grass_head_bias,
            diagonal_only=self.grass_geometry == "diag",
        )
        tables = grass_chunk_tables(
            credits.rate,
            credits.weight,
            head.gram,
            partition,
            self.grass_noise.sigma2,
            eps_d=self.grass_eps_d,
            negative_tol_rel=self.grass_negative_tol_rel,
        )
        shrunk, _exact_residual = apply_chunk_shrinkage(
            credits.rate, credits.weight, tables
        )
        effective = shrunk.to(credits.rate.dtype)
        residual = grass_credit_residual(credits.rate, credits.weight, effective)
        magnitude = float((credits.weight * credits.rate).abs().sum())
        tolerance = CREDIT_CONSERVATION_REL_TOL * max(magnitude, 1.0)
        if float(residual) > tolerance:
            raise RuntimeError(
                "GRASS-Chunk violated weighted credit conservation: max response "
                f"residual {float(residual):.6g} exceeds {tolerance:.6g}; the "
                "shrinkage is not credit-preserving and the run is not comparable"
            )

        metrics = grass_chunk_metrics(
            tables,
            credits.rate,
            credits.weight,
            head.gram,
            effective,
            residual,
            assignment=assignment,
            noncontiguous_splits=noncontiguous,
            source=self.grass_chunk_source,
        )
        metrics.update(chunk_gram_diagnostics(head.gram, tables))
        for key, value in chunk_boundary_diagnostics(
            credits.rate, credits.weight, head.gram, partition
        ).items():
            metrics[f"mp_opd_grass_chunk_boundary_{key}"] = credits.rate.new_tensor(
                float(value)
            )
        metrics.update(
            {
                "mp_opd_grass_chunk_exact_geometry": credits.rate.new_tensor(
                    float(self.grass_geometry == "exact_head")
                ),
                "mp_opd_grass_chunk_gram_symmetry_error": credits.rate.new_tensor(
                    head.symmetry_error
                ),
                "mp_opd_grass_chunk_atom_token_count": credits.rate.new_tensor(
                    float(head.token_count)
                ),
                "mp_opd_grass_chunk_sigma_batch": credits.rate.new_tensor(
                    noise["sigma_batch"]
                ),
                "mp_opd_grass_chunk_sigma_batch_variance": credits.rate.new_tensor(
                    noise["sigma_batch_variance"]
                ),
                "mp_opd_grass_chunk_noise_valid_pairs": credits.rate.new_tensor(
                    noise["valid_pairs"]
                ),
                "mp_opd_grass_chunk_noise_pairs_excluded": credits.rate.new_tensor(
                    float(max(span_ids.numel() - 1, 0) - int(noise["valid_pairs"]))
                ),
                "mp_opd_grass_chunk_noise_z_abs_mean": credits.rate.new_tensor(
                    noise["z_abs_mean"]
                ),
                "mp_opd_grass_chunk_noise_z_abs_p99": credits.rate.new_tensor(
                    noise["z_abs_quantile"]
                ),
                "mp_opd_grass_chunk_softcap": credits.rate.new_tensor(
                    float(self.grass_softcap or 0.0)
                ),
                "mp_opd_grass_chunk_head_bias": credits.rate.new_tensor(
                    float(self.grass_head_bias)
                ),
            }
        )
        if self.grass_chunk_shadow:
            metrics.update(
                chunk_shadow_metrics(
                    head.gram, credits.rate, credits.weight, partition, effective
                )
            )
        return soft_partition_loss(credits.current_nll, effective), metrics

    def _grass_shadow(self, credits, partition, gram, shrunk):
        """Section 17.12: what the alternative selectors would have done here.

        Pure read-out on the observed batch - no alternative changes the update -
        but it is the only way, after the run, to tell a genuinely different GRASS
        apart from one that merely reproduced a fixed-k tiling.
        """
        rate = credits.rate
        n = rate.numel()
        alternatives = {
            "atomic": fixed_partition(n, 1),
            "fixed2": fixed_partition(n, 2),
            "fixed3": fixed_partition(n, 3),
        }
        # The GBV comparator uses the token-count geometry and the same beta the run
        # was configured with, so the shadow answers "would GBV have done this?"
        # rather than introducing a second hand-tuned beta.
        alternatives["gbv"] = gbv_partition(
            gbv_span_costs(
                rate, credits.weight, credits.weight, self.max_span_length, self.gbv_beta
            )
        )
        atomic_ids = torch.arange(n, device=rate.device)
        grass_ids = grass_span_ids(partition, n, rate.device)
        atomic_energy = grass_update_energy(gram, atomic_ids, rate)
        grass_energy = grass_update_energy(gram, grass_ids, shrunk)
        metrics = {}
        for name, candidate in alternatives.items():
            pooled = hard_pooled_credits(credits.base_credit, credits.weight, candidate)
            candidate_ids = grass_span_ids(candidate, n, rate.device)
            candidate_energy = grass_update_energy(gram, candidate_ids, pooled)
            cross = grass_update_energy(gram, candidate_ids, pooled, rate)
            cross_grass = grass_update_energy(gram, candidate_ids, pooled, shrunk)
            prefix = f"mp_opd_grass_shadow_{name}"
            metrics[f"{prefix}_span_count"] = rate.new_tensor(float(len(candidate)))
            metrics[f"{prefix}_span_length_mean"] = rate.new_tensor(
                n / max(len(candidate), 1)
            )
            metrics[f"{prefix}_credit_l2_change"] = (pooled - rate).to(torch.float64).norm()
            metrics[f"{prefix}_head_energy_ratio"] = (
                candidate_energy / atomic_energy.clamp_min(1e-300)
            )
            metrics[f"{prefix}_head_cosine_to_atomic"] = grass_cosine(
                cross, atomic_energy, candidate_energy
            )
            metrics[f"{prefix}_head_cosine_to_grass"] = grass_cosine(
                cross_grass, grass_energy, candidate_energy
            )
        metrics["mp_opd_grass_shadow_grass_span_count"] = rate.new_tensor(float(len(partition)))
        metrics["mp_opd_grass_shadow_grass_span_length_mean"] = rate.new_tensor(
            n / max(len(partition), 1)
        )
        return metrics

    def _dpca_loss(
        self,
        credits,
        atoms,
        micro_batch,
        sample_index: int,
        student_logits: torch.Tensor | None = None,
        student_labels: torch.Tensor | None = None,
    ):
        """DPCA: semantic-prior advantage on the rollout behaviour likelihood ratio.

        Unlike every other mode this does not consume a partition. The gradient is
        ``sum_i A_i * d(-log p(sampled_i))/d(theta)``, so the differentiable input
        is the student's per-token log-probability and not ``credits.current_nll``.

        Both sides of the likelihood ratio live on the rollout's tempered
        distribution: ``prior`` is what the engine reported, and the current
        log-probability is recomputed here with the same temperature, which is what
        ``verl``'s actor does. ``L_T`` is the per-atom teacher total already carried
        in ``AtomCreditTensors``; ``L_S`` is summed from ``prior`` itself, matching
        upstream, and both are expanded back to token resolution because the
        advantage is per token.
        """
        from ._mp_opd_dpca import (
            DPCAConfig,
            dpca_atom_advantages,
            dpca_metrics_to_tensors,
            dpca_policy_loss,
            rollout_temperature_log_probs,
        )

        kd = self.args.kd
        config = DPCAConfig(
            clip_ratio_low=float(kd.mp_opd_dpca_clip_ratio_low),
            clip_ratio_high=float(kd.mp_opd_dpca_clip_ratio_high),
            clip_ratio_c=float(kd.mp_opd_dpca_clip_ratio_c),
            adv_clamp=float(kd.mp_opd_dpca_adv_clamp),
        )

        behaviour = micro_batch.get("stu_behavior_log_probs")
        if behaviour is None:
            if kd.mp_opd_dpca_require_behavior:
                raise RuntimeError(
                    "mp_opd dpca needs the rollout engine's behaviour log-probabilities as the "
                    "denominator of its likelihood ratio, but stu_behavior_log_probs is absent. "
                    "That happens whenever rollout.exact_token_trajectory is False. DPCA without "
                    "the ratio is not DPCA, so this fails closed instead of assuming ratio == 1."
                )
            raise RuntimeError(
                "mp_opd dpca has no behaviour log-probabilities and "
                "mp_opd_dpca_require_behavior is False, which leaves no loss to compute"
            )

        loss_mask = micro_batch["stu_loss_mask"][sample_index]
        prior = behaviour[sample_index][loss_mask]
        token_nll = credits.student_token_nll
        if prior.numel() != token_nll.numel():
            raise ValueError(
                f"behaviour log-prob cardinality {prior.numel()} does not match the "
                f"{token_nll.numel()} student loss tokens"
            )
        prior = prior.detach()

        # The loss mask is deliberately wider than the atoms. trajectory_tokens
        # appends a synthetic EOS sentinel that the atomizer excludes from credit
        # ("the atomizer excludes it from credit"), so the mask carries positions
        # that own no atom and, because behaviour is NaN outside the sampled span,
        # no engine log-prob either. student_logits was already sliced by this same
        # mask and atom.student_start/end index it directly, so restricting both
        # tensors to atom-covered mask positions is what keeps the semantic prior
        # and the per-atom credit aligned.
        covered = torch.zeros(prior.numel(), dtype=torch.bool, device=prior.device)
        for atom in atoms:
            covered[atom.student_start : atom.student_end] = True
        if not bool(covered.any()):
            raise RuntimeError("mp_opd dpca sample has no atom-covered loss position")
        prior = prior[covered]
        token_nll = token_nll[covered]
        # behaviour logprobs are collated on CPU while credits come from the
        # student forward on CUDA. DPCA mixes both in one objective, so align them
        # here or the metrics dict ends up with tensors on two devices.
        prior = prior.to(device=token_nll.device)

        if not torch.isfinite(prior).all():
            raise RuntimeError(
                "mp_opd dpca received non-finite behaviour log-probabilities on an "
                "atom-covered position; the engine did not report logprobs for the whole "
                "generated span"
            )
        prior = prior.float()

        counts = credits.weight.long()
        if int(counts.sum().item()) != prior.numel():
            raise ValueError("atom student token counts do not cover the atom-selected response")

        advantages = dpca_atom_advantages(
            credits.teacher_log_score,
            counts,
            prior,
            config.adv_clamp,
        )

        # The current policy's log-probability has to be recomputed on the same
        # tempered distribution the engine sampled from. credits.student_token_nll
        # is built from raw logits, so reusing it would put pi_theta on one side of
        # the ratio and pi_T on the other.
        if student_logits is None or student_labels is None:
            raise RuntimeError("mp_opd dpca requires the student logits to rebuild pi_T")
        current_log_probs = rollout_temperature_log_probs(
            student_logits, student_labels, float(self.args.rollout.temperature)
        )
        if current_log_probs.shape != prior.shape:
            raise ValueError(
                f"tempered student log-probs cover {current_log_probs.numel()} tokens "
                f"but the atom-selected prior covers {prior.numel()}"
            )
        token_nll = -current_log_probs[covered].to(prior.device)

        loss, metrics = dpca_policy_loss(
            prior,
            -token_nll,
            advantages,
            torch.ones_like(prior),
            config,
        )
        metrics = dpca_metrics_to_tensors(metrics, prior.device)
        metrics["mp_opd_dpca_advantage_mean"] = advantages.detach().mean()
        metrics["mp_opd_dpca_advantage_abs_max"] = advantages.detach().abs().max()
        metrics["mp_opd_dpca_advantage_clamped_frac"] = (
            (advantages.detach().abs() >= config.adv_clamp).float().mean()
        )
        # Every metric from here up must live on the objective's device.
        # training_step stacks them into one tensor, and torch.as_tensor(float)
        # silently lands on CPU.
        metrics["mp_opd_dpca_atoms"] = torch.as_tensor(float(len(atoms)), device=prior.device)
        return loss, metrics

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
        student_hidden: torch.Tensor | None = None,
        student_ids: list[int] | None = None,
        teacher_ids: list[int] | None = None,
    ):
        n = len(atoms)
        metrics = {}

        def add_diagnostics(partition):
            if not diagnostics:
                return
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
                        # Atoms can leave tokens outside every atom (masked EOS), so the
                        # probe needs the ranges instead of assuming a full cover.
                        atom_ranges=tuple(
                            (atom.student_start, atom.student_end) for atom in atoms
                        ),
                    )
                )

        def hard_loss_with_metrics(partition):
            loss = hard_partition_loss(
                credits.current_nll, credits.base_credit, credits.weight, partition
            )
            add_diagnostics(partition)
            return loss, metrics

        def credit_operator_loss(partition):
            """``sum_i stopgrad(A_i) NLL_i`` with ``A = K r`` from the credit operator.

            Atomic is this path with ``K = I``. The operator never redefines
            ``b_i``/``w_i``/``r_i`` and never touches atomization, masking or NLL.
            """
            effective, operator_metrics = self._effective_credit(
                credits, n, micro_batch, sample_index
            )
            metrics.update(operator_metrics)
            loss = soft_partition_loss(credits.current_nll, effective)
            if self.credit_identity_check and self.credit_spec.name == "identity":
                legacy = hard_partition_loss(
                    credits.current_nll, credits.base_credit, credits.weight, partition
                )
                drift = (loss - legacy).detach().abs()
                relative = drift / (1.0 + legacy.detach().abs())
                metrics["mp_opd_credit_identity_abs_diff"] = drift
                metrics["mp_opd_credit_identity_rel_diff"] = relative
                if float(relative) > 1e-5:
                    raise RuntimeError(
                        "identity credit operator no longer reproduces the historical "
                        f"Atomic pooled loss: |drift|={float(drift):.6g}, "
                        f"relative={float(relative):.6g} > 1e-5"
                    )
            add_diagnostics(partition)
            return loss, metrics

        if self.mode in {"atomic", "kernel"}:
            return credit_operator_loss(fixed_partition(n, 1))
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

        if self.mode == "airs":
            return self._airs_loss(
                credits, atoms, sample_index=sample_index
            )

        if self.mode == "grass":
            return self._grass_loss(                credits,
                atoms,
                student_logits=student_logits,
                student_labels=student_labels,
                student_hidden=student_hidden,
            )

        if self.mode == "grass_chunk":
            return self._grass_chunk_loss(
                credits,
                atoms,
                student_logits=student_logits,
                student_labels=student_labels,
                student_hidden=student_hidden,
                student_ids=student_ids,
                teacher_ids=teacher_ids,
            )

        _b, _w, rates, valid = span_tables(
            credits.base_credit, credits.weight, self.max_span_length
        )
        if self.mode == "dpca":
            return self._dpca_loss(
                credits,
                atoms,
                micro_batch,
                sample_index,
                student_logits=student_logits,
                student_labels=student_labels,
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
            # GRASS needs the state the LM head actually consumes, so the last
            # decoder hidden state is retained instead of reconstructed. Off for
            # every other mode: it costs a full [batch, seq, hidden] tensor.
            output_hidden_states=self.grass_needs_hidden,
            **mm_kwargs,
        )
        student_logits_flat = output["logits"][student_loss_mask]
        if self.grass_needs_hidden:
            # The hidden state is the input of the head at the position that produced
            # each logit row, so it is masked with exactly the same index as the
            # logits. Only the last entry is the head input; the per-layer stack is
            # dropped immediately because it is [layers, batch, seq, hidden] of
            # memory nobody downstream reads.
            student_hiddens_flat = output["hidden_states"][-1][student_loss_mask]
            output["hidden_states"] = ()
        else:
            student_hiddens_flat = None
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
            stu_offset_before = stu_offset
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
                student_logits=stu_logits if (diagnostics_due or self.gbv_needs_logits or self.grass_needs_logits or self.mode == "dpca") else None,
                student_labels=stu_label_tensor,
                student_hidden=(
                    None
                    if student_hiddens_flat is None
                    else student_hiddens_flat[stu_offset_before : stu_offset_before + stu_count]
                ),
                student_ids=stu_ids if self.mode == "grass_chunk" else None,
                teacher_ids=tea_ids if self.mode == "grass_chunk" else None,
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
