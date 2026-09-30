"""Hard partition oracle and exact prefix utility tables."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import torch


@dataclass(frozen=True)
class OracleResult:
    score: torch.Tensor
    partition: tuple[tuple[int, int], ...]


def span_utility_table(
    base_credit: torch.Tensor,
    weight: torch.Tensor,
    atom_directional_scores: torch.Tensor,
    max_span_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute U_[i,j) = -r_[i,j) * sum z_i in O(nL)."""
    if not (base_credit.shape == weight.shape == atom_directional_scores.shape):
        raise ValueError("base_credit, weight and z must have the same shape")
    if (weight <= 0).any():
        raise ValueError("weights must be positive")
    n = base_credit.numel()
    length = min(max(int(max_span_length), 1), max(n, 1))
    utilities = base_credit.new_full((n, length), -torch.inf, dtype=torch.float32)
    valid = torch.zeros((n, length), dtype=torch.bool, device=base_credit.device)
    pb = torch.cat((base_credit.float().new_zeros(1), base_credit.float().cumsum(0)))
    pw = torch.cat((weight.float().new_zeros(1), weight.float().cumsum(0)))
    pz = torch.cat((atom_directional_scores.float().new_zeros(1), atom_directional_scores.float().cumsum(0)))
    for start in range(n):
        for offset in range(length):
            end = start + offset + 1
            if end <= n:
                rate = (pb[end] - pb[start]) / (pw[end] - pw[start])
                utilities[start, offset] = -rate * (pz[end] - pz[start])
                valid[start, offset] = True
    return utilities, valid


def hard_max_partition(utilities: torch.Tensor, valid_mask: torch.Tensor | None = None) -> OracleResult:
    if utilities.ndim != 2:
        raise ValueError("utilities must have shape [n,L]")
    n, max_length = utilities.shape
    if n == 0:
        return OracleResult(utilities.new_zeros(()), ())
    mask = torch.ones_like(utilities, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    best = [utilities.new_tensor(-torch.inf) for _ in range(n + 1)]
    back = [-1] * (n + 1)
    best[0] = utilities.new_zeros(())
    for end in range(1, n + 1):
        candidates = []
        starts = []
        for span_length in range(1, min(max_length, end) + 1):
            start = end - span_length
            if mask[start, span_length - 1]:
                candidates.append(best[start] + utilities[start, span_length - 1])
                starts.append(start)
        if candidates:
            values = torch.stack(candidates)
            index = int(values.argmax().item())
            best[end] = values[index]
            back[end] = starts[index]
    if back[n] < 0 or not torch.isfinite(best[n]):
        raise ValueError("no valid full-cover partition")
    parts = []
    end = n
    while end:
        start = back[end]
        parts.append((start, end))
        end = start
    parts.reverse()
    return OracleResult(best[n], tuple(parts))


def enumerate_partitions(n: int, max_span_length: int) -> Iterable[tuple[tuple[int, int], ...]]:
    if n == 0:
        yield ()
        return
    def visit(start: int, parts: list[tuple[int, int]]):
        if start == n:
            yield tuple(parts)
            return
        for length in range(1, min(max_span_length, n - start) + 1):
            parts.append((start, start + length))
            yield from visit(start + length, parts)
            parts.pop()
    yield from visit(0, [])


def partition_score(table: torch.Tensor, partition: tuple[tuple[int, int], ...]) -> torch.Tensor:
    total = table.new_zeros(())
    for start, end in partition:
        total = total + table[start, end - start - 1]
    return total
