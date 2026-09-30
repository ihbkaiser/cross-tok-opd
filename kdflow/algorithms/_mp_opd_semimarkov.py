"""Exact O(nL) semi-Markov partition dynamic program."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SemiMarkovResult:
    log_z: torch.Tensor
    marginals: torch.Tensor
    coverage: torch.Tensor
    coverage_max_error: torch.Tensor
    entropy: torch.Tensor
    expected_span_count: torch.Tensor
    expected_span_length: torch.Tensor


def semi_markov_partition(
    energies: torch.Tensor,
    *,
    temperature: float = 1.0,
    valid_mask: torch.Tensor | None = None,
    host_mask: bool = False,
) -> SemiMarkovResult:
    if energies.ndim != 2:
        raise ValueError("energies must have shape [n,L]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    n, max_length = energies.shape
    device = energies.device
    if n == 0:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        empty = torch.empty_like(energies, dtype=torch.float32)
        return SemiMarkovResult(zero, empty, torch.empty(0, device=device), zero, zero, zero, zero)
    if max_length < 1:
        raise ValueError("L must be at least one for n>0")

    # Long atom sequences accumulate a large log partition. FP32 cancellation
    # in alpha + score + beta - logZ then corrupts full-cover marginals.
    # Keep the small O(nL) DP in FP64; gradients still flow to the energy dtype.
    score = energies.double() / float(temperature)
    geometric = torch.zeros_like(score, dtype=torch.bool)
    for start in range(n):
        geometric[start, : min(max_length, n - start)] = True
    mask = geometric if valid_mask is None else geometric & valid_mask.bool()
    score = score.masked_fill(~mask, -torch.inf)

    alpha_values = [score.new_zeros(())]
    for end in range(1, n + 1):
        terms = []
        for span_length in range(1, min(max_length, end) + 1):
            start = end - span_length
            terms.append(alpha_values[start] + score[start, span_length - 1])
        alpha_values.append(torch.logsumexp(torch.stack(terms), dim=0))
    alpha = torch.stack(alpha_values)
    if not torch.isfinite(alpha[n]):
        raise ValueError("all valid full-cover partitions are masked")

    beta_values: list[torch.Tensor] = [score.new_zeros(()) for _ in range(n + 1)]
    beta_values[n] = score.new_zeros(())
    for start in range(n - 1, -1, -1):
        terms = []
        for span_length in range(1, min(max_length, n - start) + 1):
            end = start + span_length
            terms.append(score[start, span_length - 1] + beta_values[end])
        beta_values[start] = torch.logsumexp(torch.stack(terms), dim=0)
    beta = torch.stack(beta_values)

    marginals = score.new_zeros((n, max_length))
    # Candidate-only metadata optimization. DP values and floating-point
    # operations remain on the original device and in the original order.
    mask_rows = mask.detach().cpu().tolist() if host_mask else None
    for start in range(n):
        for span_length in range(1, min(max_length, n - start) + 1):
            allowed = (mask_rows[start][span_length - 1]
                       if mask_rows is not None else mask[start, span_length - 1])
            if allowed:
                end = start + span_length
                marginals[start, span_length - 1] = torch.exp(
                    alpha[start] + score[start, span_length - 1] + beta[end] - alpha[n]
                )

    coverage = score.new_zeros(n)
    length_mass = score.new_zeros(())
    for start in range(n):
        for offset in range(max_length):
            end = start + offset + 1
            if end <= n:
                marginal = marginals[start, offset]
                coverage[start:end] += marginal
                length_mass += marginal * float(offset + 1)
    expected_count = marginals.sum()
    expected_length = length_mass / expected_count.clamp_min(torch.finfo(score.dtype).tiny)
    finite_score = torch.where(mask, score, torch.zeros_like(score))
    entropy = alpha[n] - (marginals * finite_score).sum()
    return SemiMarkovResult(
        alpha[n], marginals, coverage, (coverage - 1).abs().max(), entropy,
        expected_count, expected_length,
    )
