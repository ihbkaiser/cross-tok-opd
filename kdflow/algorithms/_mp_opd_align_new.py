"""ALIGN-NEW: local ALIGN candidate, guarded by the full-response head update."""
from __future__ import annotations

import math
import torch

from ._mp_opd_head_pair import checked_geometry


def align_new_guard(rate, candidate, first_norm2, second_norm2, dot, *,
                    min_descent_fraction=.1, eps_g=1e-12):
    """Largest beta in [0,1] retaining rho of Atomic first-order descent.

    g(beta) = (1-beta) g_atomic + beta g_candidate. Require
    <g_atomic,g(beta)> >= rho ||g_atomic||^2, with rho fixed in the recipe.
    No descent is defined for a vanished anchor: return Atomic and flag it.
    Scalars are detached; there is no gradient through the geometry/guard.
    """
    rho, eps = float(min_descent_fraction), float(eps_g)
    if not (math.isfinite(rho) and 0 <= rho <= 1):
        raise ValueError("ALIGN-NEW minimum descent fraction must be in [0,1]")
    if not (math.isfinite(eps) and eps >= 0):
        raise ValueError("ALIGN-NEW eps_g must be finite and nonnegative")
    if rate.ndim != 1 or candidate.shape != rate.shape:
        raise ValueError("ALIGN-NEW rates must have the same flat shape")
    if not all(bool(torch.isfinite(t).all()) for t in (rate, candidate)):
        raise FloatingPointError("non-finite ALIGN-NEW credit")
    pair = checked_geometry(first_norm2, second_norm2, dot)
    a, b, d = pair.first_norm2, pair.second_norm2, pair.dot
    no_anchor = a <= eps
    beta = 0.0 if no_anchor else (min(1., (1-rho)*a/(a-d)) if d < rho*a else 1.)
    # Move infinitesimally into the feasible interval to absorb coefficient
    # rounding, while retaining the largest feasible beta to numerical precision.
    if 0 < beta < 1:
        beta *= 1 - 1e-6
    effective = rate.detach() + beta * (candidate.detach().to(rate.dtype) - rate.detach())
    if not bool(torch.isfinite(effective).all()):
        raise FloatingPointError("non-finite guarded ALIGN-NEW credit")
    post_dot = (1-beta)*a + beta*d
    post2 = max((1-beta)**2*a + beta**2*b + 2*beta*(1-beta)*d, 0.)
    cosine = post_dot / math.sqrt(a*post2) if a > 0 and post2 > 0 else (1. if a == post2 == 0 else 0.)
    candidate_cos = d / math.sqrt(a*b) if a > 0 and b > 0 else 0.
    def m(value):
        return rate.detach().new_tensor(float(value))
    metrics = {
        "mp_opd_align_new_beta": m(beta),
        "mp_opd_align_new_guard_active_fraction": m(beta < 1),
        "mp_opd_align_new_no_anchor_fraction": m(no_anchor),
        "mp_opd_align_new_min_descent_fraction": m(rho),
        "mp_opd_align_new_atomic_head_norm2": m(a),
        "mp_opd_align_new_candidate_head_norm2": m(b),
        "mp_opd_align_new_candidate_atomic_dot": m(d),
        "mp_opd_align_new_candidate_head_update_cosine": m(max(-1., min(candidate_cos, 1.))),
        "mp_opd_align_new_head_update_cosine": m(max(-1., min(cosine, 1.))),
        "mp_opd_align_new_head_update_norm_ratio": m(math.sqrt(post2/a) if a > 0 else 0.),
        "mp_opd_align_new_retained_descent_fraction": m(post_dot/a if a > 0 else 0.),
    }
    return effective, metrics
