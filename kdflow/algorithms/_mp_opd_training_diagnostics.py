"""Optional diagnostics for the MP-OPD locality/pooling hypothesis.

The production objective stays unchanged. These helpers report detached scalar
metrics that answer four separate questions:

* Are neighboring atomic rates correlated?
* Does a partition reduce within-group rate noise, or hide sign changes?
* Does contiguous grouping differ from a same-length shuffled control?
* Does pooling change the gradient direction or magnitude at the student-logit
  interface?

The last item is deliberately called a *logit-space* gradient diagnostic. It is
the exact gradient with respect to the selected student logits, not a claim
about the full parameter-space gradient after the transformer Jacobian.
"""

from __future__ import annotations

import random
from typing import Sequence

import torch


def partition_rate_vector(
    base: torch.Tensor,
    weight: torch.Tensor,
    partition: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Broadcast each pooled group rate back to its atomic positions."""
    if base.ndim != 1 or weight.shape != base.shape:
        raise ValueError("base/weight must be aligned one-dimensional tensors")
    result = torch.empty_like(base)
    cursor = 0
    for start, end in partition:
        if start != cursor or not (start < end <= base.numel()):
            raise ValueError("partition must cover atoms once, contiguously, in order")
        result[start:end] = base[start:end].sum() / weight[start:end].sum()
        cursor = end
    if cursor != base.numel():
        raise ValueError("partition does not cover all atoms")
    return result.detach()


def _safe_std(values: torch.Tensor) -> torch.Tensor:
    return values.detach().float().std(unbiased=False) if values.numel() else values.new_zeros(())


def _safe_corr(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.numel() < 2 or right.numel() < 2:
        return left.new_zeros(())
    left = left.detach().float()
    right = right.detach().float()
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denom = left_centered.square().mean().sqrt() * right_centered.square().mean().sqrt()
    if not torch.isfinite(denom) or denom <= 1e-12:
        return left.new_zeros(())
    return (left_centered * right_centered).mean() / denom


def _lag_metrics(rate: torch.Tensor, lag: int) -> dict[str, torch.Tensor]:
    prefix = f"mp_opd_diag_lag{lag}"
    if rate.numel() <= lag:
        zero = rate.new_zeros(())
        return {
            f"{prefix}_corr": zero,
            f"{prefix}_abs_diff": zero,
            f"{prefix}_sign_flip_fraction": zero,
            f"{prefix}_pair_count": zero,
        }
    left = rate[:-lag]
    right = rate[lag:]
    return {
        f"{prefix}_corr": _safe_corr(left, right),
        f"{prefix}_abs_diff": (left - right).detach().float().abs().mean(),
        f"{prefix}_sign_flip_fraction": ((left * right) < 0).detach().float().mean(),
        f"{prefix}_pair_count": rate.new_tensor(float(left.numel())),
    }


def _shuffled_rate(rate: torch.Tensor, seed: int) -> torch.Tensor:
    if rate.numel() < 2:
        return rate.detach().clone()
    generator = random.Random(int(seed))
    indices = list(range(rate.numel()))
    generator.shuffle(indices)
    index = torch.tensor(indices, device=rate.device, dtype=torch.long)
    return rate.detach().index_select(0, index)


def partition_metrics(
    base: torch.Tensor,
    weight: torch.Tensor,
    partition: Sequence[tuple[int, int]],
    *,
    shuffle_seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Return detached locality, pooling, and shuffled-control metrics."""
    rate = (base / weight).detach()
    pooled = partition_rate_vector(base, weight, partition)
    residual = (weight * pooled).sum() - base.sum()
    within = rate - pooled
    lengths = rate.new_tensor([end - start for start, end in partition])
    sign_flip = ((rate * pooled) < 0).float()

    shuffled = _shuffled_rate(rate, shuffle_seed)
    # Preserve token-weighted total mass in the shuffled control. This is a
    # counterfactual rate assignment only; it does not affect the production
    # loss.
    shuffled_base = shuffled * weight
    shuffled_pooled = partition_rate_vector(shuffled_base, weight, partition)
    shuffled_within = shuffled - shuffled_pooled
    atomic_abs_mass = (weight * rate.abs()).sum()
    pooled_abs_mass = (weight * pooled.abs()).sum()
    pooled_abs_mass_ratio = pooled_abs_mass / atomic_abs_mass.clamp_min(1e-12)

    result = {
        "mp_opd_diag_atom_rate_std": _safe_std(rate),
        "mp_opd_diag_group_rate_std": _safe_std(
            pooled[torch.tensor([start for start, _ in partition], device=rate.device)]
        ),
        "mp_opd_diag_group_count": rate.new_tensor(float(len(partition))),
        "mp_opd_diag_group_length_mean": lengths.float().mean(),
        "mp_opd_diag_group_length_std": _safe_std(lengths),
        "mp_opd_diag_within_group_rate_rmse": within.float().square().mean().sqrt(),
        "mp_opd_diag_within_group_rate_abs_dev": within.float().abs().mean(),
        "mp_opd_diag_group_sign_flip_fraction": sign_flip.mean(),
        "mp_opd_diag_pooled_vs_atomic_rate_rmse": within.float().square().mean().sqrt(),
        "mp_opd_diag_rate_conservation_residual": residual.detach().float(),
        "mp_opd_diag_atomic_abs_credit_mass": atomic_abs_mass.detach().float(),
        "mp_opd_diag_pooled_abs_credit_mass": pooled_abs_mass.detach().float(),
        "mp_opd_diag_pooled_abs_mass_ratio": pooled_abs_mass_ratio.detach().float(),
        "mp_opd_diag_abs_mass_cancelled_fraction": (1.0 - pooled_abs_mass_ratio).detach().float(),
        "mp_opd_diag_shuffled_within_group_rate_rmse": shuffled_within.float().square().mean().sqrt(),
        "mp_opd_diag_shuffled_within_group_rate_abs_dev": shuffled_within.float().abs().mean(),
        "mp_opd_diag_actual_minus_shuffled_within_rmse": (
            within.float().square().mean().sqrt()
            - shuffled_within.float().square().mean().sqrt()
        ),
    }
    result.update(_lag_metrics(rate, 1))
    result.update(_lag_metrics(rate, 2))
    return {key: value.detach() for key, value in result.items()}


def logit_gradient_metrics(
    student_logits: torch.Tensor,
    token_nll: torch.Tensor,
    atom_rate: torch.Tensor,
    weight: torch.Tensor,
    partition: Sequence[tuple[int, int]],
    *,
    shuffle_seed: int = 0,
    atom_ranges: Sequence[tuple[int, int]] | None = None,
) -> dict[str, torch.Tensor]:
    """Compare exact atomic/pooled gradients with respect to student logits.

    ``atom_ranges`` are the atoms' half-open token slices. They are required whenever
    the atoms do not cover the whole token axis, which is the normal case: the atomizer
    leaves masked-EOS and other non-atomized tokens outside every atom. Tokens outside
    every atom carry no credit in the production loss either, so their per-token rate is
    zero here; that is exact, not filler. Without the ranges a non-covering atom set
    still fails closed instead of silently comparing misaligned vectors.
    """
    if token_nll is None:
        return {}
    if student_logits.ndim != 2 or token_nll.ndim != 1:
        raise ValueError("student_logits must be [tokens,vocab] and token_nll [tokens]")
    counts = weight.detach().round().long()
    tokens = int(token_nll.numel())
    if int(counts.sum().item()) > tokens:
        raise ValueError("atom token weights exceed token_nll")
    if atom_ranges is None and int(counts.sum().item()) != tokens:
        raise ValueError(
            "atom token weights do not cover token_nll; pass atom_ranges so gaps can be placed"
        )

    def to_token_axis(values: torch.Tensor) -> torch.Tensor:
        if atom_ranges is None:
            return torch.repeat_interleave(values, counts)
        out = token_nll.new_zeros(tokens)
        cursor = 0
        for (start, end), value, count in zip(atom_ranges, values, counts.tolist()):
            if start < cursor or not start < end <= tokens:
                raise ValueError(
                    "atom ranges must be ordered, non-overlapping and inside the token axis"
                )
            if int(count) != end - start:
                raise ValueError("atom token count and atom range disagree")
            out[start:end] = value
            cursor = end
        return out

    pooled_atom_rate = partition_rate_vector(atom_rate * weight, weight, partition)
    shuffled_atom_rate = _shuffled_rate(atom_rate.detach(), shuffle_seed)
    shuffled_pooled_atom_rate = partition_rate_vector(
        shuffled_atom_rate * weight, weight, partition
    )
    token_atomic_rate = to_token_axis(atom_rate.detach())
    token_pooled_rate = to_token_axis(pooled_atom_rate)
    token_shuffled_rate = to_token_axis(shuffled_pooled_atom_rate)
    atomic_loss = (token_atomic_rate * token_nll).sum()
    pooled_loss = (token_pooled_rate * token_nll).sum()
    shuffled_loss = (token_shuffled_rate * token_nll).sum()
    atomic_grad = torch.autograd.grad(
        atomic_loss,
        student_logits,
        retain_graph=True,
        allow_unused=True,
    )[0]
    pooled_grad = torch.autograd.grad(
        pooled_loss,
        student_logits,
        retain_graph=True,
        allow_unused=True,
    )[0]
    shuffled_grad = torch.autograd.grad(
        shuffled_loss,
        student_logits,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if atomic_grad is None:
        atomic_grad = torch.zeros_like(student_logits)
    if pooled_grad is None:
        pooled_grad = torch.zeros_like(student_logits)
    if shuffled_grad is None:
        shuffled_grad = torch.zeros_like(student_logits)
    # Match MP-OPD's token-normalized loss scale. The cosine and all ratios are
    # invariant to this factor, while the norms become comparable across
    # samples with different response lengths.
    normalizer = weight.sum().detach().float().clamp_min(1.0)
    atomic_grad = atomic_grad / normalizer
    pooled_grad = pooled_grad / normalizer
    shuffled_grad = shuffled_grad / normalizer
    atomic_grad = atomic_grad.detach().float()
    pooled_grad = pooled_grad.detach().float()
    shuffled_grad = shuffled_grad.detach().float()
    delta = pooled_grad - atomic_grad
    atomic_norm = atomic_grad.square().sum().sqrt()
    pooled_norm = pooled_grad.square().sum().sqrt()
    delta_norm = delta.square().sum().sqrt()
    denominator = atomic_norm * pooled_norm
    cosine = torch.where(
        denominator > 1e-12,
        (atomic_grad * pooled_grad).sum() / denominator,
        atomic_norm.new_zeros(()),
    )
    shuffled_norm = shuffled_grad.square().sum().sqrt()
    actual_shuffled_denominator = pooled_norm * shuffled_norm
    actual_shuffled_cosine = torch.where(
        actual_shuffled_denominator > 1e-12,
        (pooled_grad * shuffled_grad).sum() / actual_shuffled_denominator,
        atomic_norm.new_zeros(()),
    )
    return {
        "mp_opd_diag_logit_grad_atomic_norm": atomic_norm,
        "mp_opd_diag_logit_grad_pooled_norm": pooled_norm,
        "mp_opd_diag_logit_grad_norm_ratio": pooled_norm / atomic_norm.clamp_min(1e-12),
        "mp_opd_diag_logit_grad_cosine": cosine,
        "mp_opd_diag_logit_grad_delta_norm": delta_norm,
        "mp_opd_diag_logit_grad_delta_ratio": delta_norm / atomic_norm.clamp_min(1e-12),
        "mp_opd_diag_shuffled_logit_grad_norm_ratio": shuffled_norm / atomic_norm.clamp_min(1e-12),
        "mp_opd_diag_atomic_shuffled_logit_grad_cosine": torch.where(
            atomic_norm * shuffled_norm > 1e-12,
            (atomic_grad * shuffled_grad).sum() / (atomic_norm * shuffled_norm),
            atomic_norm.new_zeros(()),
        ),
        "mp_opd_diag_pooled_shuffled_logit_grad_cosine": actual_shuffled_cosine,
    }
