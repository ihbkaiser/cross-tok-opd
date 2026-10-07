"""AIRS: Atomic Independent Risk Shrinkage.

AIRS keeps the aligned atom as the supervision resolution and suppresses only the
*reliability* of each atomic credit. There is no span, no chunk, no dynamic
program, no Gram matrix and no mixing between neighbouring credits: every atom
receives its own scalar shrinkage coefficient and nothing else changes downstream.

For an atom with credit ``r_i``, precision ``w_i`` and global noise scale
``sigma2``, the heteroskedastic working model is ``Var(eps_i) = sigma2 / w_i``.
Differentiating the SURE risk of the fixed linear estimator ``mu_hat = lambda * r``
gives ``lambda_i = [1 - sigma_i^2 / r_i^2]_+``, so an atom whose credit is not large
relative to its own uncertainty is attenuated toward zero instead of having credit
borrowed from a neighbour.

Two properties make this different from every other MP-OPD mode here, and both are
deliberate rather than incidental:

* **Credit is not conserved.** Shrinking toward zero lowers the weighted credit
  mass on purpose. A renormalisation that restores ``sum_i w_i r~_i == sum_i w_i r_i``
  would undo the suppression, so none is applied and none may be added.
* **Nothing else moves.** The OPD loss, clipping and optimizer see exactly the same
  tensors they see in the atomic baseline apart from the substituted credit.
"""
from __future__ import annotations

import torch

# Weights are floored before they are inverted. The value is the one fixed by the
# method note; it is a numerical floor, not a modelling parameter, so it is not
# configurable.
AIRS_EPS_W: float = 1e-8
# Below this per-atom noise variance the atom is treated as noiseless and kept at
# full strength. Without it, a cold estimator that has not yet seen enough pairs
# would report a small variance and silently damp every credit towards zero.
AIRS_EPS_NOISE: float = 1e-12
# The credit RMS ratio is reported with this floor so a batch whose credits are all
# zero cannot produce a division by zero in the diagnostic.
AIRS_EPS_RATIO: float = 1e-12


def airs_shrinkage(
    rate: torch.Tensor,
    weight: torch.Tensor,
    sigma2: float,
    *,
    enabled: bool = True,
    eps_w: float = AIRS_EPS_W,
    eps_noise: float = AIRS_EPS_NOISE,
) -> dict[str, torch.Tensor]:
    """Apply the per-atom AIRS rule.

    Parameters
    ----------
    rate:
        Atomic credits ``r_i`` for one response, shape ``[n]``.
    weight:
        Positive precisions ``w_i``, shape ``[n]``.
    sigma2:
        Global noise *variance* from the running estimator. Shared by every atom.
    enabled:
        ``False`` during warm-up, which keeps ``lambda = 1`` so the method observes
        the unmodified credit stream until the noise scale is calibrated.

    The branch is written out rather than folded into
    ``1 - sigma2 / (r^2 + eps)``: the additive epsilon moves the ``r^2 <= sigma_i^2``
    threshold, and that threshold is the entire rule. ``sigma_i^2`` is computed from
    the floored weight, and an atom with ``r_i == 0`` lands on ``lambda = 0``, which
    is consistent with the required assumption that a zero credit means no update.

    Returns
    -------
    dict with ``credit`` (the shrunk credit), ``lambda``, ``snr``, ``atom_variance``
    and the floored ``weight``.
    """
    r = rate.detach()
    if r.ndim != 1:
        raise ValueError("rate must be one-dimensional")
    w = weight.detach().to(r.dtype)
    if w.shape != r.shape:
        raise ValueError("rate and weight must have the same shape")
    if not torch.isfinite(r).all():
        raise ValueError("rate must be finite")

    sigma2 = float(sigma2)
    if not torch.isfinite(torch.tensor(sigma2)) or sigma2 < 0.0:
        raise ValueError("sigma2 must be finite and nonnegative")

    w_safe = w.clamp_min(float(eps_w))
    atom_variance = sigma2 / w_safe
    r_squared = r * r

    if not enabled:
        coefficient = torch.ones_like(r)
    else:
        # An explicit branch, because the two regions differ by more than a factor:
        # sigma2_i <= eps_noise means the noise scale is not yet trusted, not that
        # the credit is large.
        coefficient = torch.where(
            atom_variance <= float(eps_noise),
            torch.ones_like(r),
            torch.where(
                r_squared <= atom_variance,
                torch.zeros_like(r),
                1.0 - atom_variance / r_squared.clamp_min(torch.finfo(r.dtype).tiny),
            ),
        )
    coefficient = coefficient.clamp(0.0, 1.0)

    return {
        "credit": coefficient * r,
        "lambda": coefficient,
        "snr": r_squared / atom_variance.clamp_min(float(eps_noise)),
        "atom_variance": atom_variance,
        "weight": w_safe,
    }


def airs_metrics(
    rate: torch.Tensor,
    weight: torch.Tensor,
    shrinkage: dict[str, torch.Tensor],
    *,
    enabled: bool,
    eps_ratio: float = AIRS_EPS_RATIO,
) -> dict[str, torch.Tensor]:
    """Diagnostics of section 13, restricted to what AIRS itself needs.

    The credit RMS ratio is the headline number: because AIRS does not conserve
    credit, a value far below 1 is the intended effect, and a value at 1 means the
    shrinkage did nothing. Both are reported so the effect cannot be mistaken for a
    no-op.
    """
    r = rate.detach().to(torch.float64)
    credit = shrinkage["credit"].detach().to(torch.float64)
    coefficient = shrinkage["lambda"].detach().to(torch.float64)
    snr = shrinkage["snr"].detach().to(torch.float64)

    count = max(r.numel(), 1)
    rms_before = torch.sqrt((r * r).sum() / count)
    rms_after = torch.sqrt((credit * credit).sum() / count)
    finite_snr = snr[torch.isfinite(snr)]

    def _quantile(values: torch.Tensor, q: float) -> float:
        if values.numel() == 0:
            return 0.0
        return float(values.quantile(q))

    positive = r > 0
    negative = r < 0
    metrics: dict[str, torch.Tensor] = {
        "mp_opd_airs_enabled": torch.ones((), dtype=torch.float64),
        "mp_opd_airs_atom_count": r.new_tensor(float(r.numel())),
        "mp_opd_airs_lambda_mean": coefficient.mean(),
        "mp_opd_airs_lambda_median": coefficient.median(),
        "mp_opd_airs_lambda_p25": coefficient.new_tensor(_quantile(coefficient, 0.25)),
        "mp_opd_airs_lambda_p75": coefficient.new_tensor(_quantile(coefficient, 0.75)),
        "mp_opd_airs_lambda_p90": coefficient.new_tensor(_quantile(coefficient, 0.90)),
        "mp_opd_airs_kill_fraction": (coefficient <= 0.0).to(torch.float64).mean(),
        "mp_opd_airs_keep_fraction": (coefficient >= 0.9).to(torch.float64).mean(),
        "mp_opd_airs_credit_mean_before": r.mean(),
        "mp_opd_airs_credit_abs_mean_before": r.abs().mean(),
        "mp_opd_airs_credit_rms_before": rms_before,
        "mp_opd_airs_credit_rms_after": rms_after,
        "mp_opd_airs_credit_rms_ratio": rms_after
        / (rms_before + float(eps_ratio)),
        "mp_opd_airs_credit_weighted_mass_ratio": (
            (weight.detach().to(torch.float64) * credit).sum()
            / ((weight.detach().to(torch.float64) * r).sum().abs() + float(eps_ratio))
        ),
        "mp_opd_airs_snr_mean": finite_snr.mean() if finite_snr.numel() else r.new_zeros(()),
        "mp_opd_airs_snr_median": r.new_tensor(_quantile(finite_snr, 0.5)),
        "mp_opd_airs_snr_p25": r.new_tensor(_quantile(finite_snr, 0.25)),
        "mp_opd_airs_snr_p75": r.new_tensor(_quantile(finite_snr, 0.75)),
        "mp_opd_airs_snr_p90": r.new_tensor(_quantile(finite_snr, 0.90)),
        "mp_opd_airs_snr_p99": r.new_tensor(_quantile(finite_snr, 0.99)),
        "mp_opd_airs_snr_le_1": (finite_snr <= 1.0).to(torch.float64).mean()
        if finite_snr.numel()
        else r.new_zeros(()),
        "mp_opd_airs_snr_1_2": ((finite_snr > 1.0) & (finite_snr <= 2.0))
        .to(torch.float64)
        .mean()
        if finite_snr.numel()
        else r.new_zeros(()),
        "mp_opd_airs_snr_2_5": ((finite_snr > 2.0) & (finite_snr <= 5.0))
        .to(torch.float64)
        .mean()
        if finite_snr.numel()
        else r.new_zeros(()),
        "mp_opd_airs_snr_gt_5": (finite_snr > 5.0).to(torch.float64).mean()
        if finite_snr.numel()
        else r.new_zeros(()),
        "mp_opd_airs_lambda_mean_positive": coefficient[positive].mean()
        if bool(positive.any())
        else r.new_zeros(()),
        "mp_opd_airs_lambda_mean_negative": coefficient[negative].mean()
        if bool(negative.any())
        else r.new_zeros(()),
    }
    # Warm-up must be visible in the log: an all-ones lambda is the whole run in
    # disguise if it is only inferable from the other metrics.
    metrics["mp_opd_airs_enabled"] = r.new_tensor(float(bool(enabled)))
    return metrics


def airs_precision_buckets(
    weight: torch.Tensor, shrinkage: dict[str, torch.Tensor], *, buckets: int = 4
) -> dict[str, torch.Tensor]:
    """Per-precision-quantile SNR and lambda (section 13.6).

    AIRS is supposed to key off SNR, which mixes credit magnitude and precision.
    If it instead collapses because ``w`` has a pathological scale, the bucket means
    separate that from a genuine reliability effect.
    """
    w = weight.detach().to(torch.float64)
    coefficient = shrinkage["lambda"].detach().to(torch.float64)
    snr = shrinkage["snr"].detach().to(torch.float64)
    metrics: dict[str, torch.Tensor] = {}
    buckets = max(int(buckets), 1)
    for index in range(buckets):
        low = int(round(index * w.numel() / buckets))
        high = int(round((index + 1) * w.numel() / buckets))
        if high <= low:
            continue
        chunk_coefficient = coefficient[low:high]
        chunk_snr = snr[low:high]
        finite = chunk_snr[torch.isfinite(chunk_snr)]
        metrics[f"mp_opd_airs_bucket{index}_weight_mean"] = w[low:high].mean()
        metrics[f"mp_opd_airs_bucket{index}_lambda_mean"] = chunk_coefficient.mean()
        metrics[f"mp_opd_airs_bucket{index}_kill_fraction"] = (
            (chunk_coefficient <= 0.0).to(torch.float64).mean()
        )
        metrics[f"mp_opd_airs_bucket{index}_snr_mean"] = (
            finite.mean() if finite.numel() else w.new_zeros(())
        )
    return metrics
