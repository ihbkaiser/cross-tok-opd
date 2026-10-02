"""Gradient bias-variance span selection for MP-OPD (GBV-Span).

Every candidate contiguous span pays two scale-free costs:

    C_beta(c) = D_g(c) / E_g + beta * (Q_c / W_c) / Q_0

``D_g(c)`` is the observed squared gradient distortion of forcing the span's atoms
to share one pooled credit rate, ``Q_c / W_c`` is the span's effective degrees of
freedom under the token-noise model of the method note, ``E_g`` is the total
observed directional energy and ``Q_0`` the atomic degrees of freedom. The optimal
full-cover partition is the exact minimum of the additive cost, found with the same
O(n L_max) dynamic program the oracle mode already uses.

This module only *selects* a partition. The loss stays MP-OPD's hard pooled loss
(``hard_partition_loss``), so signed-credit conservation is untouched: for every
full-cover partition ``sum_i w_i (A_pi r)_i == sum_i b_i`` exactly.

Numerical notes that are easy to get wrong later:

* cost tables are accumulated in float64 because the distortion term is a
  difference of prefix sums (``S_qr2 - 2 rbar S_qr + rbar^2 Q``) and the interesting
  case is nearly-equal rates, where float32 cancellation would dominate;
* ``E_g`` is compared against a relative floor, and a degenerate draw returns the
  coarsest admissible partition instead of dividing by a numerically-zero energy;
* the squared-softmax sum is accumulated in float64 even though the logits are
  bf16/fp32, because it is a pure reduction with no extra memory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import torch

from ..fused_logprob import _logsumexp_chunked, accumulation_dtype
from ._mp_opd_oracle import hard_max_partition, partition_score

# Relative floor for the total directional energy. Below it every rate is equal to
# within float64 noise, the distortion term carries no signal, and the objective is
# minimized by the coarsest admissible tiling (documented degenerate branch).
ENERGY_RELATIVE_FLOOR = 1e-12

# Span-length fractions reported as telemetry. The method note names spans 1..4.
REPORTED_SPAN_LENGTHS = (1, 2, 3, 4)

_DEFAULT_VOCAB_CHUNK = 16384
_VOCAB_CHUNK_FLAG = "MP_OPD_GBV_VOCAB_CHUNK"


def gbv_vocab_chunk() -> int:
    """Vocabulary chunk for the exact-sensitivity pass, overridable per host.

    A memory knob, not a recipe knob: it changes how the same number is computed,
    never which number. Invalid values raise instead of silently picking a chunk.
    """
    raw = os.environ.get(_VOCAB_CHUNK_FLAG, "").strip()
    if not raw:
        return _DEFAULT_VOCAB_CHUNK
    try:
        chunk = int(raw)
    except ValueError as error:
        raise ValueError(f"{_VOCAB_CHUNK_FLAG}={raw!r} must be an integer") from error
    if chunk < 1:
        raise ValueError(f"{_VOCAB_CHUNK_FLAG} must be >= 1")
    return chunk


@dataclass(frozen=True)
class GbvSpanCosts:
    """Cost tables and the normalizers they were divided by.

    ``costs``/``valid`` have shape ``[n, L]`` indexed ``[start, length-1]``.
    ``distortion`` and ``retained_dof`` are the two terms of ``costs`` kept apart so
    telemetry can show which one drove a decision; ``costs`` is
    ``distortion + beta * retained_dof``. Invalid entries carry ``+inf`` cost so a
    masked maximum can never pick them.
    """

    costs: torch.Tensor
    valid: torch.Tensor
    distortion: torch.Tensor
    retained_dof: torch.Tensor
    energy: torch.Tensor
    atomic_dof: torch.Tensor
    beta: float
    degenerate: bool


def _as_vector(name: str, value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector")
    return value


def token_logit_sensitivity(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    selected_log_prob: torch.Tensor | None = None,
    vocab_chunk: int | None = None,
) -> torch.Tensor:
    """Per-token ``||p_t - e_{y_t}||^2 = 1 - 2 p_t(y_t) + sum_v p_t(v)^2``.

    This is the exact squared norm of the token-NLL logit gradient, so atom
    sensitivities need no ``autograd.grad`` call. ``selected_log_prob``
    (``log p_t(y_t)``) may be supplied to skip the chunked log-sum-exp pass; the
    MP-OPD credit path already holds it as ``-credits.student_token_nll``. Both
    routes derive ``log Z`` from the same selected logit, so the probabilities stay
    consistent with the production credit operator.

    Runs under ``no_grad`` and returns a detached vector: sensitivities select a
    partition and must never carry gradient into the student.
    """
    if student_logits.ndim != 2:
        raise ValueError("student_logits must be [tokens,vocab]")
    rows, vocab = student_logits.shape
    if rows == 0:
        raise ValueError("student_logits must contain at least one token")
    labels = _as_vector("labels", labels)
    if labels.numel() != rows:
        raise ValueError("labels must align with student_logits rows")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= vocab):
        raise ValueError("labels must be within [0, vocab)")
    if selected_log_prob is not None:
        selected_log_prob = _as_vector("selected_log_prob", selected_log_prob)
        if selected_log_prob.numel() != rows:
            raise ValueError("selected_log_prob must align with student_logits rows")
    chunk = int(vocab_chunk) if vocab_chunk is not None else gbv_vocab_chunk()
    if chunk < 1:
        raise ValueError("vocab_chunk must be >= 1")

    with torch.no_grad():
        acc = accumulation_dtype(student_logits.dtype)
        selected_logit = student_logits.gather(
            1, labels.long().unsqueeze(1)
        ).squeeze(1).to(torch.float64)
        if selected_log_prob is None:
            log_partition = _logsumexp_chunked(student_logits, chunk).to(torch.float64)
            log_selected = selected_logit - log_partition
        else:
            log_selected = selected_log_prob.detach().to(torch.float64)
            # log Z = z_y - log p(y): subtracting it keeps every exponent <= 0, so
            # the squared-softmax sum cannot overflow without a max pass.
            log_partition = selected_logit - log_selected
        squared_sum = torch.zeros(rows, dtype=torch.float64, device=student_logits.device)
        for start in range(0, vocab, chunk):
            stop = min(start + chunk, vocab)
            block = student_logits[:, start:stop].to(acc).to(torch.float64)
            squared_sum += torch.exp(2.0 * (block - log_partition.unsqueeze(-1))).sum(dim=-1)
        selected_probability = torch.exp(log_selected)
        # ||p - e_y||^2 is a squared norm: clamp float64 round-off, never the signal.
        return (1.0 - 2.0 * selected_probability + squared_sum).clamp_min(0.0)


def atom_logit_sensitivity(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    atom_ranges: Sequence[tuple[int, int]],
    *,
    selected_log_prob: torch.Tensor | None = None,
    vocab_chunk: int | None = None,
) -> torch.Tensor:
    """Sum :func:`token_logit_sensitivity` over contiguous atom ranges.

    ``atom_ranges`` are half-open ``[start, end)`` slices into the same flattened
    student-token axis the credit path uses. They must be ordered and must not
    overlap: an overlap would attribute one token to two atoms, so it raises.

    Gaps are allowed and are not an error. The atomizer can leave tokens outside every
    atom (masked EOS and other non-atomized positions), and the credit path already
    ignores them; those tokens contribute to no atom's sensitivity, which is exactly
    the per-atom sum this objective is defined on. Demanding a perfect cover was
    measured wrong on a real microbatch (B200 smoke, 2026-10-02).
    """
    if not atom_ranges:
        raise ValueError("at least one atom range is required")
    if student_logits.ndim != 2:
        raise ValueError("student_logits must be [tokens,vocab]")
    tokens = int(student_logits.shape[0])
    cursor = 0
    for start, end in atom_ranges:
        if start < cursor:
            raise ValueError("atom ranges must be ordered and must not overlap")
        if not start < end <= tokens:
            raise ValueError("atom ranges must stay inside the token axis")
        cursor = end
    per_token = token_logit_sensitivity(
        student_logits,
        labels,
        selected_log_prob=selected_log_prob,
        vocab_chunk=vocab_chunk,
    )
    return torch.stack([per_token[start:end].sum() for start, end in atom_ranges])


def coarsest_admissible_partition(n: int, max_span_length: int) -> tuple[tuple[int, int], ...]:
    """Spans of exactly ``max_span_length`` with a clamped tail; same as fixed mode."""
    if n < 0 or max_span_length <= 0:
        raise ValueError("n must be nonnegative and max_span_length positive")
    return tuple((start, min(start + max_span_length, n)) for start in range(0, n, max_span_length))


def gbv_span_costs(
    rate: torch.Tensor,
    weight: torch.Tensor,
    sensitivity: torch.Tensor,
    max_span_length: int,
    beta: float,
) -> GbvSpanCosts:
    """Scale-free span costs for one response's atom sequence.

    ``rate`` is the atomic credit rate ``b_i / w_i``, ``weight`` the atom token mass
    and ``sensitivity`` the atom logit sensitivity ``q_i``. ``q_i = w_i`` is the
    token-count approximation; the caller must label it as such, because it turns the
    objective into the weighted Potts special case.
    """
    rate = _as_vector("rate", rate)
    weight = _as_vector("weight", weight)
    sensitivity = _as_vector("sensitivity", sensitivity)
    if not (rate.shape == weight.shape == sensitivity.shape):
        raise ValueError("rate, weight and sensitivity must share one shape")
    n = rate.numel()
    if n == 0:
        raise ValueError("at least one atom is required")
    if not torch.isfinite(rate).all():
        raise ValueError("rates must be finite")
    if not torch.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError("weights must be finite and positive")
    if not torch.isfinite(sensitivity).all() or (sensitivity < 0).any():
        raise ValueError("sensitivities must be finite and nonnegative")
    max_span_length = int(max_span_length)
    if max_span_length < 1:
        raise ValueError("max_span_length must be positive")
    beta = float(beta)
    if beta < 0 or not torch.isfinite(torch.tensor(beta)):
        raise ValueError("beta must be finite and nonnegative")

    length = min(max_span_length, n)
    device = rate.device
    r = rate.detach().to(torch.float64)
    w = weight.detach().to(torch.float64)
    q = sensitivity.detach().to(torch.float64)

    zero = torch.zeros(1, dtype=torch.float64, device=device)
    pw = torch.cat((zero, w.cumsum(0)))
    pb = torch.cat((zero, (w * r).cumsum(0)))
    pq = torch.cat((zero, q.cumsum(0)))
    pqr = torch.cat((zero, (q * r).cumsum(0)))
    pqr2 = torch.cat((zero, (q * r * r).cumsum(0)))
    pqw = torch.cat((zero, (q / w).cumsum(0)))

    mean_rate = pb[-1] / pw[-1]
    energy = 0.5 * (pqr2[-1] - 2.0 * mean_rate * pqr[-1] + mean_rate * mean_rate * pq[-1])
    energy = energy.clamp_min(0.0)
    atomic_dof = pqw[-1]
    if not torch.isfinite(atomic_dof) or atomic_dof <= 0:
        raise ValueError("atomic degrees of freedom must be positive and finite")
    energy_scale = 0.5 * pqr2[-1]
    degenerate = bool(
        energy <= ENERGY_RELATIVE_FLOOR * torch.clamp_min(energy_scale, torch.tensor(1e-300, dtype=torch.float64, device=device))
    )

    costs = torch.full((n, length), float("inf"), dtype=torch.float64, device=device)
    distortion = torch.zeros((n, length), dtype=torch.float64, device=device)
    retained_dof = torch.zeros((n, length), dtype=torch.float64, device=device)
    valid = torch.zeros((n, length), dtype=torch.bool, device=device)
    for start in range(n):
        for offset in range(length):
            end = start + offset + 1
            if end > n:
                continue
            span_weight = pw[end] - pw[start]
            span_rate = (pb[end] - pb[start]) / span_weight
            span_q = pq[end] - pq[start]
            span_qr = pqr[end] - pqr[start]
            span_qr2 = pqr2[end] - pqr2[start]
            span_distortion = 0.5 * (
                span_qr2 - 2.0 * span_rate * span_qr + span_rate * span_rate * span_q
            )
            span_distortion = span_distortion.clamp_min(0.0)
            span_dof = (span_q / span_weight) / atomic_dof
            distortion[start, offset] = 0.0 if degenerate else span_distortion / energy
            retained_dof[start, offset] = span_dof
            costs[start, offset] = distortion[start, offset] + beta * span_dof
            valid[start, offset] = True
    return GbvSpanCosts(
        costs=costs,
        valid=valid,
        distortion=distortion,
        retained_dof=retained_dof,
        energy=energy,
        atomic_dof=atomic_dof,
        beta=beta,
        degenerate=degenerate,
    )


def gbv_partition(tables: GbvSpanCosts) -> tuple[tuple[int, int], ...]:
    """Exact minimum-cost full-cover partition of the cost tables.

    Ties fall to the shortest admissible span because the dynamic program scans span
    lengths in increasing order and keeps the first maximum; the degenerate branch
    returns the coarsest admissible partition instead.
    """
    n = tables.costs.shape[0]
    if tables.degenerate:
        return coarsest_admissible_partition(n, tables.costs.shape[1])
    return hard_max_partition(-tables.costs, tables.valid).partition


def _pooled_rate(
    rate: torch.Tensor, weight: torch.Tensor, start: int, end: int
) -> torch.Tensor:
    return (rate[start:end] * weight[start:end]).sum() / weight[start:end].sum()


def gbv_partition_metrics(
    tables: GbvSpanCosts,
    partition: Sequence[tuple[int, int]],
    rate: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Detached telemetry for one selected partition.

    ``mp_opd_gbv_boundary_strength_mean`` is defined here as the mean absolute jump of
    the pooled rate across the selected span boundaries; the method note lists the
    name but not its definition, and this is the reading used by the report.
    """
    n, _length = tables.costs.shape
    rate = _as_vector("rate", rate).detach()
    weight = _as_vector("weight", weight).detach()
    if rate.numel() != n or weight.numel() != n:
        raise ValueError("rate and weight must align with the cost tables")
    cursor = 0
    lengths: list[int] = []
    pooled: list[torch.Tensor] = []
    for start, end in partition:
        if start != cursor or not start < end <= n:
            raise ValueError("partition must cover the atoms once, contiguously, in order")
        lengths.append(end - start)
        pooled.append(_pooled_rate(rate, weight, start, end))
        cursor = end
    if cursor != n:
        raise ValueError("partition does not cover all atoms")

    span_count = len(lengths)
    total_cost = partition_score(tables.costs, partition)
    distortion_term = partition_score(tables.distortion, partition)
    retained_fraction = partition_score(tables.retained_dof, partition)
    length_tensor = tables.costs.new_tensor([float(value) for value in lengths])
    metrics = {
        "mp_opd_gbv_total_cost": total_cost,
        "mp_opd_gbv_distortion_term": distortion_term,
        "mp_opd_gbv_dof_term": total_cost.new_tensor(tables.beta) * retained_fraction,
        "mp_opd_gbv_retained_dof_fraction": retained_fraction,
        "mp_opd_gbv_selected_span_count": total_cost.new_tensor(float(span_count)),
        "mp_opd_gbv_selected_span_length_mean": length_tensor.mean(),
        "mp_opd_gbv_span_length_max": total_cost.new_tensor(float(max(lengths))),
        "mp_opd_gbv_degenerate": total_cost.new_tensor(float(tables.degenerate)),
    }
    for reported in REPORTED_SPAN_LENGTHS:
        fraction = sum(1 for value in lengths if value == reported) / span_count
        metrics[f"mp_opd_gbv_span_{reported}_fraction"] = total_cost.new_tensor(float(fraction))
    if span_count >= 2:
        jumps = torch.stack(
            [(pooled[index] - pooled[index + 1]).abs() for index in range(span_count - 1)]
        )
        metrics["mp_opd_gbv_boundary_strength_mean"] = jumps.mean()
    else:
        metrics["mp_opd_gbv_boundary_strength_mean"] = total_cost.new_zeros(())
    return {key: value.detach() for key, value in metrics.items()}
