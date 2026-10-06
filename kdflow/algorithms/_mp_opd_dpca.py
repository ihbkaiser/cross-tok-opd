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

from dataclasses import dataclass
from typing import Sequence

import torch

from ._mp_opd_atoms import MPAtom
from ._mp_opd_credit import AtomCreditTensors

__all__ = [
    "DPCAConfig",
    "dpca_target_log_probs",
    "dpca_policy_loss",
    "dpca_advantages",
]


@dataclass(frozen=True)
class DPCAConfig:
    """Paper hyper-parameters (``dpca_paper8b.json``); override only deliberately."""

    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    clip_ratio_c: float = 10.0
    adv_clamp: float = 10.0
    kl_clamp: float = 20.0


def _per_atom_prior(prior_log_probs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Sum prior log-probability inside each atom, so chunks never cross atoms."""
    # Repeated cumulative sums give every atom's inclusive prefix boundary; the
    # exclusive boundary is the previous atom's inclusive one, and atom 0 starts
    # at 0.
    inclusive = torch.cumsum(prior_log_probs, dim=0)
    starts = torch.cat([inclusive.new_zeros(1), inclusive[:-1]])
    return (inclusive - starts)[weight.long().bool()]


def dpca_target_log_probs(
    prior_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    weight: torch.Tensor,
    teacher_chunk_logp: torch.Tensor,
) -> torch.Tensor:
    """Semantic prior assignment ``log q_i = (L_T/L_S) * log p_i``, per atom.

    Mirrors ``compute_policy_loss_opd``: teacher and student log-probabilities are
    summed inside each synchronized chunk, and every token in the chunk inherits
    the chunk's ratio. ``teacher_chunk_logp`` is supplied by the caller because
    summing it here would repeat the detach-and-mask bookkeeping that already
    happened when the credits were built.

    ``L_S == 0`` is the degenerate case the upstream handles by spreading the
    teacher budget uniformly across the chunk; it is reproduced rather than
    silently clamped, because a clamped denominator would turn a real signal
    into a spurious one.
    """
    if prior_log_probs.shape != teacher_log_probs.shape:
        raise ValueError("prior_log_probs and teacher_log_probs must be aligned")
    prior_chunk_logp = _per_atom_prior(prior_log_probs, weight)

    with torch.no_grad():
        valid = weight.long().bool()
        safe_prior_chunk = torch.where(
            prior_chunk_logp.abs() < 1e-8, torch.ones_like(prior_chunk_logp), prior_chunk_logp
        )
        ratio = teacher_chunk_logp / safe_prior_chunk
        uniform = (prior_chunk_logp.abs() < 1e-8) & valid
        per_token = torch.where(
            uniform,
            teacher_chunk_logp / weight.clamp_min(1.0),
            ratio * prior_log_probs,
        )
        return torch.where(valid, per_token, prior_log_probs)


def dpca_advantages(
    target_log_probs: torch.Tensor,
    prior_log_probs: torch.Tensor,
    adv_clamp: float,
) -> torch.Tensor:
    """``A_i = log q_i - log p_i``, optionally clamped to ``+-loss_max_clamp``."""
    advantages = target_log_probs - prior_log_probs
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
    ppo_kl = (negative_approx_kl * response_mask).sum() / response_mask.sum().clamp_min(1.0)

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