"""TRUST-P: project mismatch onto the strict non-conflict halfspace.

This is a new batch-level algorithm, not a correction to TRUST-R/B. Projection
is a known operation; efficacy or research novelty is not established here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ._mp_opd_head_pair import checked_geometry


@dataclass(frozen=True)
class TrustProjectedResult:
    strict_scale: float
    gs_norm2: float
    gm_norm2: float
    dot: float
    n_strict: int
    n_mismatch: int
    no_anchor: bool
    projected: bool
    mismatch_retained_norm_ratio: float
    strict_descent: float
    update_norm_ratio: float
    update_cosine: float


def trust_p_calibrate_scalars(gs2, gm2, dot, n_s, n_m, *, eps_g=1e-12):
    """One decision for the entire optimizer batch, including cross-responses.

    M_safe = M - min(0,<S,M>)/||S||^2 S; g = S+M_safe. Therefore
    L = stopgrad(a) L_S + L_M with a=1-min(0,dot)/||S||^2.
    With no usable S anchor, keep Atomic and explicitly waive the guarantee.
    """
    if not (math.isfinite(float(eps_g)) and float(eps_g) >= 0):
        raise ValueError("TRUST-P eps_g must be finite and nonnegative")
    if n_s < 0 or n_m < 0:
        raise ValueError("TRUST-P atom counts must be nonnegative")
    pair = checked_geometry(gs2, gm2, dot)
    s, m, d = pair.first_norm2, pair.second_norm2, pair.dot
    no_anchor = n_s == 0 or s <= float(eps_g)
    correction = -min(0., d)/s if not no_anchor else 0.
    a = 1 + correction
    if not math.isfinite(a) or a > torch.finfo(torch.float32).max:
        raise FloatingPointError("TRUST-P strict scale is not representable in fp32")
    # Loss coefficients are fp32. Round towards the protected halfspace, rather
    # than letting nearest rounding reintroduce a tiny opposing component.
    rounded = torch.tensor(a, dtype=torch.float32)
    if float(rounded) < a:
        rounded = torch.nextafter(rounded, torch.tensor(float("inf")))
    a = float(rounded)
    # Orthogonal component survives unchanged; only the opposing parallel
    # component is removed. Projection cannot increase the M norm.
    orthogonal2 = max(m - d*d/s, 0.) if correction > 0 else 0.
    retained2 = orthogonal2 + (a-1+d/s)**2*s if correction > 0 else m
    # Complete the square to avoid catastrophic cancellation for opposing arms.
    post2 = (orthogonal2 + (a + d/s)**2*s) if correction > 0 else max(s + m + 2*d, 0.)
    pre2 = max(s + m + 2*d, 0.)
    cross = a*s + m + (a+1)*d
    cosine = cross/math.sqrt(pre2*post2) if pre2 > 0 and post2 > 0 else (1. if pre2 == post2 == 0 else 0.)
    scalars = (a, retained2, post2, pre2, cross)
    if not all(math.isfinite(v) for v in scalars):
        raise FloatingPointError("non-finite TRUST-P calibration")
    return TrustProjectedResult(
        a, s, m, d, int(n_s), int(n_m), no_anchor, correction > 0,
        math.sqrt(retained2/m) if m > 0 else 0., a*s+d,
        math.sqrt(post2/pre2) if pre2 > 0 else 0., max(-1., min(cosine, 1.)),
    )


def trust_p_response_loss(current_nll, rate, is_strict, is_mismatch, result):
    if bool((is_strict & is_mismatch).any()) or not bool((is_strict | is_mismatch).all()):
        raise ValueError("TRUST-P sides must be disjoint and cover all atoms")
    coefficient = torch.where(is_strict, float(result.strict_scale), 1.).to(rate)
    return (rate.detach() * coefficient * current_nll).sum()


def trust_p_batch_metrics(result, counts):
    r = result
    out = {"batch_"+key: float(value) for key, value in counts.items()}
    out.update({
        "strict_scale": r.strict_scale,
        "mismatch_scale": 1.,
        "strict_head_norm2": r.gs_norm2,
        "mismatch_head_norm2": r.gm_norm2,
        "strict_mismatch_dot": r.dot,
        "strict_mismatch_cosine": r.dot/math.sqrt(r.gs_norm2*r.gm_norm2) if r.gs_norm2 > 0 and r.gm_norm2 > 0 else 0.,
        "projected": float(r.projected),
        "no_anchor": float(r.no_anchor),
        "batch_has_strict": float(r.n_strict > 0),
        "batch_has_mismatch": float(r.n_mismatch > 0),
        "mismatch_retained_norm_ratio": r.mismatch_retained_norm_ratio,
        "strict_descent": r.strict_descent,
        "strict_descent_ratio": r.strict_descent/r.gs_norm2 if r.gs_norm2 > 0 else 0.,
        "update_norm_ratio_post_pre": r.update_norm_ratio,
        "update_cosine_pre_post": r.update_cosine,
        "update_ratio_defined": float(r.gs_norm2+r.gm_norm2+2*r.dot > 0),
        "scope": 1.,
    })
    return out
