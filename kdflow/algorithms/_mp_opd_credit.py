"""Dimensionally consistent scalar path credit for MP-OPD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import os

import torch

from ..fused_logprob import masked_selected_logprobs
from ._mp_opd_atoms import MPAtom


@dataclass(frozen=True)
class AtomCreditTensors:
    teacher_log_score: torch.Tensor
    student_old_log_score: torch.Tensor
    base_credit: torch.Tensor
    weight: torch.Tensor
    rate: torch.Tensor
    current_nll: torch.Tensor
    # Per-student-token NLL is kept only so the optional MP-OPD diagnostics can
    # compare atomic and pooled weighting in logit space without recomputing a
    # second log_softmax. Existing diagnostic-only callers construct this
    # dataclass positionally, so the field remains optional for compatibility.
    student_token_nll: torch.Tensor | None = None


_FUSED_CREDIT_FLAG = "MP_OPD_FUSED_CREDIT"
_FUSED_CREDIT_CHUNK_FLAG = "MP_OPD_FUSED_CREDIT_CHUNK"
_FUSED_OFF = frozenset({"", "0", "false", "no", "off", "none"})
_FUSED_ON = frozenset({"1", "true", "yes", "on", "masked"})
_DEFAULT_VOCAB_CHUNK = 16384


def fused_credit_enabled() -> bool:
    """Whether the credit path uses the memory-lean selected-log-prob operator.

    Default off, so the qualified vehicle keeps the production log_softmax route until
    the fused path has its own measured memory trace. A value that is neither clearly
    on nor clearly off raises instead of silently picking a path: a typo must not let an
    experiment be reported as evidence for code it never ran.
    """
    raw = os.environ.get(_FUSED_CREDIT_FLAG, "").strip().lower()
    if raw in _FUSED_OFF:
        return False
    if raw in _FUSED_ON:
        return True
    raise ValueError(f"{_FUSED_CREDIT_FLAG}={raw!r} is not a recognised value")


def fused_credit_vocab_chunk() -> int:
    raw = os.environ.get(_FUSED_CREDIT_CHUNK_FLAG, "").strip()
    if not raw:
        return _DEFAULT_VOCAB_CHUNK
    try:
        chunk = int(raw)
    except ValueError as error:
        raise ValueError(
            f"{_FUSED_CREDIT_CHUNK_FLAG}={raw!r} must be an integer"
        ) from error
    if chunk < 1:
        raise ValueError(f"{_FUSED_CREDIT_CHUNK_FLAG} must be >= 1")
    return chunk


def realized_token_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """log p(label) per row, from the same logits the production path consumes.

    The fused branch never builds an fp32 copy of the logits and never runs a full
    log_softmax; it keeps the value (and both derivatives the energy step needs)
    identical in fp32 accumulation. At fp64 input it accumulates in fp64 where the
    production route downcasts to fp32 - training logits are bf16, so both paths
    accumulate in fp32 there.
    """
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.numel():
        raise ValueError("logits must be [tokens,vocab] and labels [tokens]")
    if fused_credit_enabled():
        return masked_selected_logprobs(
            logits, labels, vocab_chunk=fused_credit_vocab_chunk()
        )
    return torch.log_softmax(logits.float(), dim=-1).gather(1, labels.long().unsqueeze(1)).squeeze(1)


def build_atom_credits(
    atoms: Sequence[MPAtom],
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_labels: torch.Tensor,
) -> AtomCreditTensors:
    if not atoms:
        raise ValueError("at least one valid atom is required")
    student_logp = realized_token_log_probs(student_logits, student_labels)
    teacher_logp = realized_token_log_probs(teacher_logits.detach(), teacher_labels).detach()
    l_t, l_s, h, weights = [], [], [], []
    for atom in atoms:
        if not atom.valid or atom.student_token_count <= 0:
            raise ValueError("invalid atom in credit computation")
        l_t.append(teacher_logp[atom.teacher_start : atom.teacher_end].sum())
        score = student_logp[atom.student_start : atom.student_end].sum()
        l_s.append(score.detach())
        h.append(-score)
        weights.append(atom.student_token_count)
    teacher_score = torch.stack(l_t).detach()
    student_old = torch.stack(l_s).detach()
    weight = torch.tensor(weights, dtype=torch.float32, device=student_logits.device)
    base = (teacher_score - student_old).detach()
    rate = (base / weight).detach()
    return AtomCreditTensors(
        teacher_score,
        student_old,
        base,
        weight,
        rate,
        torch.stack(h),
        -student_logp,
    )


def span_tables(base: torch.Tensor, weight: torch.Tensor, max_span_length: int):
    if base.ndim != 1 or weight.shape != base.shape or (weight <= 0).any():
        raise ValueError("base/weight must be aligned vectors with positive weights")
    n = base.numel()
    length = min(max(int(max_span_length), 1), max(n, 1))
    b_span = base.new_zeros((n, length))
    w_span = weight.new_zeros((n, length))
    valid = torch.zeros((n, length), dtype=torch.bool, device=base.device)
    pb = torch.cat((base.new_zeros(1), base.cumsum(0)))
    pw = torch.cat((weight.new_zeros(1), weight.cumsum(0)))
    # Keep exactly the same prefix subtraction, but launch per span length
    # rather than per atom/length pair (thousands of tiny CUDA operations).
    for offset in range(length):
        width = offset + 1
        count = n - offset
        if count > 0:
            b_span[:count, offset] = pb[width:] - pb[:count]
            w_span[:count, offset] = pw[width:] - pw[:count]
            valid[:count, offset] = True
    rate = torch.where(valid, b_span / w_span.clamp_min(1), torch.zeros_like(b_span)).detach()
    return b_span.detach(), w_span.detach(), rate, valid


def hard_partition_loss(
    current_nll: torch.Tensor,
    base: torch.Tensor,
    weight: torch.Tensor,
    partition: Sequence[tuple[int, int]],
) -> torch.Tensor:
    total = current_nll.new_zeros(())
    cursor = 0
    for start, end in partition:
        if start != cursor or not (start < end <= current_nll.numel()):
            raise ValueError("partition must cover atoms once, contiguously, in order")
        rate = base[start:end].sum() / weight[start:end].sum()
        total = total + rate.detach() * current_nll[start:end].sum()
        cursor = end
    if cursor != current_nll.numel():
        raise ValueError("partition does not cover all atoms")
    return total


def expected_atom_rates(span_marginals: torch.Tensor, span_rates: torch.Tensor) -> torch.Tensor:
    if span_marginals.shape != span_rates.shape:
        raise ValueError("marginals and rates must have the same [n,L] shape")
    n, length = span_marginals.shape
    result = span_rates.new_zeros(n)
    # For each output atom preserve the original order: increasing span start,
    # then increasing span length. Parallelize across output atoms, not across
    # their reductions. L=2 now uses three vector operations, not ~2*n slices.
    for distance in range(min(length, n) - 1, -1, -1):
        for offset in range(distance, min(length, n)):
            count = n - offset
            result[distance:count + distance] += (
                span_marginals[:count, offset] * span_rates[:count, offset]
            ).to(result.dtype)  # Match scalar-tensor promotion before addition.
    return result


def soft_partition_loss(current_nll: torch.Tensor, atom_rates: torch.Tensor) -> torch.Tensor:
    if current_nll.shape != atom_rates.shape:
        raise ValueError("current_nll and atom_rates must be aligned")
    return (atom_rates.detach() * current_nll).sum()


def credit_conservation_residual(
    base: torch.Tensor, weight: torch.Tensor, atom_rates: torch.Tensor
) -> torch.Tensor:
    return (weight * atom_rates).sum() - base.sum()
