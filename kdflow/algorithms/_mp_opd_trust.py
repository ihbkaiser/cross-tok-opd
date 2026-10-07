"""TRUST v1.0 - trusted reference update scaling for cross-tokenizer OPD.

Mismatch supervision may carry signal, but its direction and scale should be
calibrated against the more reliable strict/matched supervision. TRUST does not
touch relative credits inside mismatch spans: per calibration unit it scales the
whole mismatch update by ``lambda = max(0, <gS, gM> / ||gM||^2)`` and keeps the
strict update whole.

Two modes share this file: TRUST-R (one lambda per response) and TRUST-B (one
lambda per micro-batch). The only intended difference between them is the
calibration aggregation level; the strict/mismatch partition, the proxy, and
the guards are identical.

The strict/mismatch partition comes from the atomizer, not from a threshold:
an atom whose student and teacher spans are both exactly one token over the
same bytes (``boundary_type == "one_to_one"``) is matched supervision, and an
atom that straddles a tokenization boundary (``multi_token``) is mismatch. Any
atom with another boundary type fails closed, because silently filing it under
either side would change what the calibration means.

The proxy is the GRASS LM-head geometry, reused unchanged: with per-atom
coefficients ``q_i = r_i`` (the OPD atom loss is a plain token-sum, so the
outer coefficient is 1 and the response weight is a plain sum) and token
reduction ``a_it = 1`` (``current_nll`` is a token sum over the atom), the
effective coefficients are exactly the rates the loss consumes. The Gram code
is shared, so a TRUST number and a GRASS number read the same geometry.

Deliberate v1 deviations from the spec text, all fail-safe rather than silent:

* Median/p10/p25/p75/p90 keys are not emitted. Metrics reduce per micro-batch
  by sum-then-mean, which cannot express a percentile; emitting a mean under a
  ``_median`` key would lie about the statistic. Means and fractions are exact
  (production uses micro_train_batch_size=1, where the mean is the value).
* ``mp_opd_trust_scope`` is a float code (0.0 = response, 1.0 = batch) because
  the metric table only carries finite tensors.
* Samples or units without both sides log 0.0 for undefined quantities (never
  NaN: the trainer fails fast on non-finite metrics), with the eligibility
  fractions (``has_both``/``zero_*``) as the interpretive key.
* TRUST requires ``exact_head`` geometry. The ``diag`` ablation zeroes every
  cross-atom term, which would force every dot product to zero and every lambda
  to zero: a mode that can never act. Failing closed beats a silent no-op.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

# Section 9 fixes this: a mismatch norm at or below it carries no usable
# direction, so the calibration that would divide by it is skipped.
TRUST_EPS_G = 1e-12
# Only used to keep norm denominators away from zero; it never decides whether
# a unit calibrates or what lambda it gets.
TRUST_EPS_NORM = 1e-12


@dataclass(frozen=True)
class TrustUnitResult:
    """One calibration unit's lambda plus everything its diagnostics derive from."""

    lam: float
    dot: float
    gs_norm2: float
    gm_norm2: float
    cosine: float
    norm_ratio: float
    kappa: float
    has_strict: bool
    has_mismatch: bool
    calibrated: bool
    update_cosine: float
    update_norm_ratio: float
    n_strict: int
    n_mismatch: int


def _require_finite(name: str, *tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"TRUST received a non-finite {name}")


def trust_calibrate(
    qs: torch.Tensor,
    qm: torch.Tensor,
    gram_ss: torch.Tensor,
    gram_mm: torch.Tensor,
    gram_sm: torch.Tensor,
    *,
    eps_g: float = TRUST_EPS_G,
    eps_norm: float = TRUST_EPS_NORM,
) -> TrustUnitResult:
    """Calibrate one strict/mismatch pair from its Gram blocks.

    ``qs``/``qm`` are the effective strict/mismatch coefficients and
    ``gram_ss``/``gram_mm``/``gram_sm`` the matching blocks of the exact
    LM-head Gram. Everything is read in fp32 and detached by the caller; the
    returned lambda is a Python float, so it can never carry a graph.
    """
    _require_finite("coefficients", qs, qm)
    _require_finite("Gram blocks", gram_ss, gram_mm, gram_sm)
    if not (float(eps_g) >= 0.0):
        raise ValueError(f"TRUST eps_g must be non-negative, got {eps_g!r}")
    qs = qs.detach().to(torch.float32).reshape(-1)
    qm = qm.detach().to(torch.float32).reshape(-1)
    n_s, n_m = int(qs.numel()), int(qm.numel())
    if (n_s, n_m) != (int(gram_ss.shape[0]), int(gram_mm.shape[0])):
        raise ValueError(
            f"TRUST Gram blocks {tuple(gram_ss.shape)}/{tuple(gram_mm.shape)} "
            f"do not match coefficient counts ({n_s}, {n_m})"
        )
    if tuple(gram_sm.shape) != (n_s, n_m):
        raise ValueError(
            f"TRUST cross Gram {tuple(gram_sm.shape)} does not match ({n_s}, {n_m})"
        )

    # Empty sides have no geometry to read: the scalars below are defined as
    # zero rather than derived from a matmul over an empty axis.
    gs2 = float(qs @ gram_ss @ qs) if n_s else 0.0
    gm2 = float(qm @ gram_mm @ qm) if n_m else 0.0
    dot = float(qs @ gram_sm @ qm) if (n_s and n_m) else 0.0
    for name, value in (("gs_norm2", gs2), ("gm_norm2", gm2), ("dot", dot)):
        if not abs(value) < float("inf"):
            raise ValueError(f"TRUST {name} is non-finite")
    # A Gram diagonal is a squared norm, so a negative below is roundoff, not
    # signal; clamping keeps the square roots below honest without touching
    # the dot product that decides the calibration.
    gs2 = max(gs2, 0.0)
    gm2 = max(gm2, 0.0)

    has_strict = n_s > 0
    has_mismatch = n_m > 0
    calibrated = bool(has_strict and has_mismatch and gm2 > float(eps_g))
    lam = max(0.0, dot / gm2) if calibrated else 0.0

    gs = gs2**0.5
    gm = gm2**0.5
    cosine = dot / (gs * gm + float(eps_norm))
    norm_ratio = gm / (gs + float(eps_norm))
    kappa = lam * gm / (gs + float(eps_norm))

    # g_pre = gS + gM and g_post = gS + lam gM, expanded in Gram scalars so no
    # vector is ever formed. When both updates vanish they are identical.
    pre2 = gs2 + gm2 + 2.0 * dot
    post2 = gs2 + lam * lam * gm2 + 2.0 * lam * dot
    inner = gs2 + lam * gm2 + (1.0 + lam) * dot
    pre2 = max(pre2, 0.0)
    post2 = max(post2, 0.0)
    if pre2 <= 0.0 and post2 <= 0.0:
        update_cosine = 1.0
    elif pre2 <= 0.0 or post2 <= 0.0:
        update_cosine = 0.0
    else:
        update_cosine = inner / (pre2**0.5 * post2**0.5 + float(eps_norm))
    update_norm_ratio = post2**0.5 / (pre2**0.5 + float(eps_norm))

    return TrustUnitResult(
        lam=lam,
        dot=dot,
        gs_norm2=gs2,
        gm_norm2=gm2,
        cosine=cosine,
        norm_ratio=norm_ratio,
        kappa=kappa,
        has_strict=has_strict,
        has_mismatch=has_mismatch,
        calibrated=calibrated,
        update_cosine=update_cosine,
        update_norm_ratio=update_norm_ratio,
        n_strict=n_s,
        n_mismatch=n_m,
    )


def strict_mask(boundary_types: Sequence[str]) -> torch.Tensor:
    """Strict-side mask from atomizer boundary types, failing closed.

    ``one_to_one`` is matched supervision, ``multi_token`` is mismatch. A
    third kind would silently change the calibration under either choice, so
    it raises instead of being filed anywhere.
    """
    mask = []
    for kind in boundary_types:
        if kind == "one_to_one":
            mask.append(True)
        elif kind == "multi_token":
            mask.append(False)
        else:
            raise ValueError(f"TRUST has no side for boundary type {kind!r}")
    return torch.tensor(mask, dtype=torch.bool)


def trust_response_loss(
    current_nll: torch.Tensor,
    rate: torch.Tensor,
    is_strict: torch.Tensor,
    is_mismatch: torch.Tensor,
    result: TrustUnitResult,
) -> torch.Tensor:
    """``L_S + stopgrad(lambda) L_M`` with exactly the OPD loss coefficients.

    Each side is a ``(rates.detach() * nll).sum()``, the same construction as
    ``soft_partition_loss``: the strict side keeps its credits whole and the
    mismatch side is scaled by the detached lambda. The lambda is a Python
    float by construction, so no graph can pass through the calibration rule.
    """
    if not (bool(is_strict.any()) or bool(is_mismatch.any())):
        raise ValueError("TRUST needs at least one strict or mismatch atom")
    if bool((is_strict & is_mismatch).any()):
        raise ValueError("TRUST strict and mismatch sets must be disjoint")
    zero = current_nll.new_zeros(())
    strict = (torch.where(is_strict, rate.detach(), zero) * current_nll).sum()
    mismatch = (torch.where(is_mismatch, rate.detach(), zero) * current_nll).sum()
    return strict + float(result.lam) * mismatch


def _counts(is_strict: torch.Tensor, token_counts: torch.Tensor) -> dict[str, float]:
    """Raw per-unit coverage counts; the caller decides the aggregation."""
    is_mismatch = ~is_strict
    n_s = int(is_strict.sum())
    n_m = int(is_mismatch.sum())
    total = n_s + n_m
    return {
        "strict_span_count": float(n_s),
        "mismatch_span_count": float(n_m),
        # One atom is one supervision span here: there is no higher grouping
        # for TRUST to aggregate, so the span and atom keys coincide by
        # definition rather than by coincidence.
        "strict_atom_count": float(n_s),
        "mismatch_atom_count": float(n_m),
        "strict_token_count": float(token_counts[is_strict].sum()),
        "mismatch_token_count": float(token_counts[is_mismatch].sum()),
        "strict_span_fraction": float(n_s / total) if total else 0.0,
        "mismatch_span_fraction": float(n_m / total) if total else 0.0,
        "has_strict": float(n_s > 0),
        "has_mismatch": float(n_m > 0),
        "has_both": float(n_s > 0 and n_m > 0),
        "zero_strict": float(n_s == 0),
        "zero_mismatch": float(n_m == 0),
    }


def trust_response_metrics(
    result: TrustUnitResult,
    is_strict: torch.Tensor,
    token_counts: torch.Tensor,
    *,
    scope: str,
) -> dict[str, float]:
    """Per-unit metric values; the training loop averages them over responses."""
    if scope not in ("response", "batch"):
        raise ValueError(f"TRUST scope must be response|batch, got {scope!r}")
    out = _counts(is_strict, token_counts)
    out.update(
        {
            "cosine": result.cosine,
            "cosine_negative": float(result.cosine < 0.0),
            "mismatch_to_strict_norm_ratio": result.norm_ratio,
            "lambda": result.lam,
            "lambda_zero": float(result.lam <= 0.0),
            "lambda_gt1": float(result.lam > 1.0),
            "calibrated": float(result.calibrated),
            "calibrated_mismatch_norm_ratio": result.kappa,
            "update_cosine_pre_post": result.update_cosine,
            "update_norm_ratio_post_pre": result.update_norm_ratio,
            "scope": 0.0 if scope == "response" else 1.0,
        }
    )
    return out


def trust_batch_metrics(
    results: TrustUnitResult,
    counts: dict[str, float],
) -> dict[str, float]:
    """Single micro-batch values for TRUST-B; logged as-is, never averaged."""
    return {
        "batch_strict_span_count": counts["strict_span_count"],
        "batch_mismatch_span_count": counts["mismatch_span_count"],
        "batch_strict_atom_count": counts["strict_atom_count"],
        "batch_mismatch_atom_count": counts["mismatch_atom_count"],
        "batch_strict_token_count": counts["strict_token_count"],
        "batch_mismatch_token_count": counts["mismatch_token_count"],
        "batch_strict_span_fraction": counts["strict_span_fraction"],
        "batch_mismatch_span_fraction": counts["mismatch_span_fraction"],
        "cosine": results.cosine,
        "mismatch_to_strict_norm_ratio": results.norm_ratio,
        "lambda": results.lam,
        "batch_has_strict": float(results.has_strict),
        "batch_has_mismatch": float(results.has_mismatch),
        "batch_calibrated": float(results.calibrated),
        "calibrated_mismatch_norm_ratio": results.kappa,
        "update_cosine_pre_post": results.update_cosine,
        "update_norm_ratio_post_pre": results.update_norm_ratio,
        "scope": 1.0,
    }
