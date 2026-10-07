"""DPCA: semantic-prior-clipped policy gradient for MP-OPD.

This module is a port of ``verl.trainer.ppo.core_algos.compute_policy_loss_opd``
(ivanniu/On-Policy-Distill @ 927a8264, registered as ``"opd"``) onto the MP-OPD
atom path. It is a port rather than a copy because kdflow has no verl dependency:
the objective has to live next to the other MP-OPD modes.

Why this is a different loss, not another partition
--------------------------------------------------
Every existing MP-OPD mode computes ``rate * current_nll``, whose gradient is
``sum_i rate_i * d(nll_i)/d(theta)`` -- imitation that pulls the student towards
the teacher's log-probability *rate*. DPCA is a policy gradient: it weights the
log-probability of the token the student actually sampled. The differentiable
quantity differs, so no existing helper can be reused for the gradient:

* ``credits.current_nll``   -- full-vocabulary NLL per token (the imitation term)
* ``credits.student_token_nll`` -- ``-log p(sampled token)``, which is what a
  policy gradient differentiates (available since 4e6da65 made it part of
  ``AtomCreditTensors``)

The likelihood ratio *denominator* is a second, separate blocker. Nothing in
kdflow previously needed it, so ``rollout_log_probs`` was hardcoded to ``None``
and the behaviour log-probs (``stu_behavior_log_probs``) existed only for the
parity probe. DPCA consumes them as a loss term, which makes
``exact_token_trajectory`` a load-bearing requirement rather than an optional
parity aid; ``_behaviour_log_probs`` raises when it is missing instead of
silently substituting 1.0.

Cross-tokenizer chunking
-----------------------
The paper aligns teacher and student text into synchronized chunks before
computing ``L_T`` and ``L_S``. Here a chunk is one MP-OPD atom: the atomizer
already produces the minimal contiguous student-token spans that a teacher
chunk aligns to, together with the teacher/student token counts that give
``L_S`` and ``L_T`` exactly. Chunk boundaries therefore agree with the rest of
MP-OPD rather than with a second, independent aligner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from ._mp_opd_atoms import MPAtom
from ._mp_opd_credit import AtomCreditTensors

__all__ = [
    "DPCAConfig",
    "dpca_atom_advantages",
    "dpca_policy_loss",
]


@dataclass(frozen=True)
class DPCAConfig:
    """Paper hyper-parameters (``dpca_paper8b.json``); override only deliberately."""

    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    clip_ratio_c: float = 10.0
    adv_clamp: float = 10.0
    kl_clamp: float = 20.0


def rollout_temperature_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """log p_T(label) per row, on the rollout's tempered distribution.

    ``verl``'s actor divides the logits by the rollout temperature before
    ``logprobs_from_logits`` (``dp_actor._forward_micro_batch``), and the rollout
    engine applies the same temperature to the logprobs it returns. Both sides of
    the DPCA ratio must therefore sit on ``pi_T``. Comparing raw logits against a
    tempered prior would make ``clip`` bite on temperature rather than on policy
    drift, and with one update per rollout the corrected ratio collapses back to 1.
    """
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"rollout temperature must be positive and finite, got {temperature}")
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.numel():
        raise ValueError("logits must be [tokens,vocab] and labels [tokens]")
    scaled = logits.float() / temperature
    chosen = scaled.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return chosen - torch.logsumexp(scaled, dim=-1)


def dpca_atom_coverage(
    atom_spans: Sequence[tuple[int, int]],
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Boolean mask over loss-mask positions that at least one atom owns.

    The loss mask is deliberately wider than the atoms: ``trajectory_tokens``
    appends a synthetic EOS sentinel that the atomizer excludes from credit, so
    the mask carries positions with no atom and no engine log-prob. Everything
    token-shaped in the objective has to be narrowed with this one mask.

    ``width`` is the *unfiltered* mask width, because the tensors it will be
    applied to are still full width at that point. Narrowing an already-narrowed
    tensor is the mistake that makes a 236-vs-234 mismatch.
    """
    covered = torch.zeros(int(width), dtype=torch.bool, device=device)
    for start, end in atom_spans:
        covered[int(start) : int(end)] = True
    if not bool(covered.any()):
        raise RuntimeError("mp_opd dpca sample has no atom-covered loss position")
    return covered


def dpca_atom_advantages(
    teacher_log_score: torch.Tensor,
    counts: torch.Tensor,
    prior_log_probs: torch.Tensor,
    adv_clamp: float | None,
) -> torch.Tensor:
    """Semantic-prior advantage per student token, from per-atom chunk totals.

    Reproduces ``compute_policy_loss_opd``: inside each synchronized chunk the
    teacher and prior log-probabilities are summed (``L_T``, ``L_S``), every token
    inherits the chunk's ratio via ``log q_i = (L_T / L_S) * log p_i``, and the
    advantage is ``A_i = (L_T / L_S - 1) * log p_i``.

    The ratio is formed from **per-atom** totals and only then expanded to token
    resolution. Computing ``L_T / log p_i`` token by token instead would divide a
    whole chunk's teacher mass by a single token's log-probability, which is a
    different objective that happens to look plausible.

    ``L_S == 0`` is the degenerate chunk upstream handles by spreading the teacher
    budget uniformly across it; it is reproduced rather than clamped, because a
    clamped denominator turns a real signal into a spurious one.
    """
    counts = counts.long()
    n_atoms = int(counts.numel())
    if teacher_log_score.shape != (n_atoms,):
        raise ValueError("teacher_log_score must be one total per atom")
    if int(counts.sum().item()) != prior_log_probs.numel():
        raise ValueError(
            f"atom token counts total {int(counts.sum().item())} but the prior covers "
            f"{prior_log_probs.numel()} tokens"
        )

    teacher_chunk = teacher_log_score.float()
    # Upstream sums ``prior_log_probs`` for L_S and uses the *same array* for the
    # per-token log p_i. Deriving the chunk total from a different quantity would
    # silently mix two conventions inside one formula.
    atom_index = torch.repeat_interleave(
        torch.arange(n_atoms, device=prior_log_probs.device), counts
    )
    prior_chunk = torch.zeros(n_atoms, dtype=torch.float32, device=prior_log_probs.device)
    prior_chunk = prior_chunk.index_add(0, atom_index, prior_log_probs.float())
    degenerate = prior_chunk.abs() < 1e-8
    safe_prior_chunk = torch.where(degenerate, torch.ones_like(prior_chunk), prior_chunk)
    ratio = teacher_chunk / safe_prior_chunk

    # The ratio must be expanded to token resolution BEFORE it is combined with
    # prior_log_probs. Multiplying a per-atom vector by a per-token vector
    # broadcasts instead of failing: it errors when the atom count differs from
    # the token count, and silently produces an atom-count-times-too-long tensor
    # when there is exactly one atom.
    per_token_ratio = torch.repeat_interleave(ratio, counts)
    uniform_share = torch.repeat_interleave(teacher_chunk / counts.clamp_min(1).float(), counts)
    target = torch.where(
        torch.repeat_interleave(degenerate, counts), uniform_share, per_token_ratio * prior_log_probs
    )

    advantages = target - prior_log_probs
    if adv_clamp is not None:
        advantages = torch.clamp(advantages, min=-adv_clamp, max=adv_clamp)
    return advantages


def dpca_policy_loss(
    prior_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    config: DPCAConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped policy objective, matching ``compute_policy_loss`` term by term.

    The dual clip uses upstream's form literally: ``pg_losses3 = -A * c`` is a
    constant, so the lower branch is selected whenever it is strictly smaller,
    which is equivalent to clamping the ratio at ``c`` only for negative
    advantages. Rewriting it as an explicit ratio clamp would silently change
    the objective.
    """
    negative_approx_kl = log_probs - prior_log_probs
    negative_approx_kl = torch.clamp(
        negative_approx_kl, min=-config.kl_clamp, max=config.kl_clamp
    )
    ratio = torch.exp(negative_approx_kl)
    # Upstream reports ``masked_mean(-negative_approx_kl)``. This is the mean of a
    # log-ratio, not a divergence, so it is only a readable divergence estimate
    # once the two policies agree; upstream still negates it, and matching that
    # sign is what makes the reported number comparable with the reference runs.
    ppo_kl = -(negative_approx_kl * response_mask).sum() / response_mask.sum().clamp_min(1.0)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - config.clip_ratio_low, 1 + config.clip_ratio_high
    )
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    clip_mask = (pg_losses2 > pg_losses1).float()
    pg_clipfrac = (clip_mask * response_mask).sum() / response_mask.sum().clamp_min(1.0)

    pg_losses3 = -advantages * config.clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    lower_mask = ((clip_pg_losses1 > pg_losses3) & (advantages < 0)).float()
    pg_clipfrac_lower = (lower_mask * response_mask).sum() / response_mask.sum().clamp_min(1.0)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    denom = response_mask.sum().clamp_min(1.0)
    pg_loss = (pg_losses * response_mask).sum() / denom

    metrics = {
        "mp_opd_dpca_ppo_kl": float(ppo_kl.detach().item()),
        "mp_opd_dpca_clipfrac": float(pg_clipfrac.detach().item()),
        "mp_opd_dpca_clipfrac_lower": float(pg_clipfrac_lower.detach().item()),
        "mp_opd_dpca_ratio_mean": float(ratio.detach().mean().item()),
    }
    return pg_loss, metrics


def dpca_metrics_to_tensors(metrics, device):
    """Materialize the scalar DPCA metrics on the objective's device.

    ``dpca_policy_loss`` reports Python floats, and ``.item()`` has already
    discarded where they were computed. A bare ``torch.as_tensor`` therefore puts
    every metric on CPU while the loss-derived metrics beside them stay on CUDA,
    and ``training_step`` then dies on ``torch.stack``. Pin the device here so the
    conversion cannot silently default.
    """
    return {key: torch.as_tensor(value, device=device) for key, value in metrics.items()}