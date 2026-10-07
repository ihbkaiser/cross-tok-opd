"""ALIGN v1.0 - aligned local interference gradient neutralization.

GRASS-Chunk asks whether atomic credits inside an aligned chunk should be shrunk.
ALIGN asks a different question and leaves the credits alone: the credits may
already be right, while the *directions* the atoms push the head in disagree, and
one atom undoes another. The response is the standard PCGrad projection, run in
coefficient space against the exact LM-head Gram, so no gradient surgery touches the
backward pass and no pooling, noise model or SURE gain is involved.

The distinction from GRASS matters for reading a result. Shrinkage can only pull a
credit towards zero, so its effect is monotone and bounded. A projection can add
to one credit and subtract from another, so it may move a credit either way and may
redistribute weight between atoms that both looked fine beforehand. A run where ALIGN
helps is evidence about interference; a run where it does not is evidence against
it, which is the opposite of a tuning knob that failed to find a good setting.

Everything here is deterministic. The projection order is cyclic and fixed, there is
no sampling, and the same chunk and the same Gram as GRASS-Chunk are used, so a
resumed or distributed run recomputes the same coefficients from the same inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

# Section 4 fixes this: a diagonal at or below it carries no usable direction, so
# the projection that would divide by it is skipped rather than attempted.
ALIGN_EPS_H = 1e-12
# Only used to keep a norm denominator away from zero; it never decides which pairs
# conflict or which projections run.
ALIGN_EPS_NORM = 1e-12


@dataclass(frozen=True)
class AlignChunkResult:
    """One chunk's surgery plus everything its diagnostics are derived from."""

    rate: torch.Tensor
    conflict_fraction_pre: float
    conflict_fraction_post: float
    projections: int
    considered: int
    skipped_zero_reference: int
    skipped_degenerate_diagonal: int
    row_relative_change_mean: float


def _cyclic_order(m: int, start: int) -> list[int]:
    """``start+1 .. m-1, 0 .. start-1`` - the deterministic cyclic order of §3."""
    return list(range(start + 1, m)) + list(range(0, start))


def _off_diagonal_conflict_fraction(matrix: torch.Tensor) -> float:
    """Fraction of unordered off-diagonal pairs that are negative.

    Read from the strict upper triangle only. Counting both triangles would count
    every unordered pair twice and would admit the diagonal, which can never be a
    conflict because a Gram diagonal is a squared norm.
    """
    size = int(matrix.shape[0])
    if size < 2:
        return 0.0
    upper = torch.triu(matrix, diagonal=1)
    pairs = size * (size - 1) / 2
    return float((upper < 0).sum()) / float(pairs)


def align_chunk(
    rate: torch.Tensor,
    gram: torch.Tensor,
    *,
    eps_h: float = ALIGN_EPS_H,
) -> AlignChunkResult:
    """Remove negative pairwise interference inside one chunk.

    ``rate`` is ``[m]`` and ``gram`` is the ``[m, m]`` exact LM-head Gram of those
    same atoms. Row ``i`` of the coefficient matrix holds the current modified
    contribution of atom ``i``, so it starts as ``diag(rate)`` and each projection
    rewrites exactly one entry. The sum of the rows is the credit handed to the loss,
    which is what lets the whole surgery stay in the coordinates the loss already
    uses, without ever forming a vector.
    """
    rate = rate.detach()
    m = int(rate.numel())
    if m <= 1:
        # Section 4: a singleton has nothing to interfere with, so it comes back
        # untouched instead of being passed through a projection that would be a no-op.
        return AlignChunkResult(
            rate=rate.clone(),
            conflict_fraction_pre=0.0,
            conflict_fraction_post=0.0,
            projections=0,
            considered=0,
            skipped_zero_reference=0,
            skipped_degenerate_diagonal=0,
            row_relative_change_mean=0.0,
        )
    if tuple(gram.shape) != (m, m):
        raise ValueError(
            f"ALIGN chunk Gram is {tuple(gram.shape)}, expected ({m}, {m})"
        )
    if not bool(torch.isfinite(rate).all()):
        raise ValueError("ALIGN received a non-finite credit")
    if not bool(torch.isfinite(gram).all()):
        raise ValueError("ALIGN received a non-finite Gram")

    # Section 4: the projection arithmetic is fp32. The Gram it reads is already
    # fp32, so promoting would imply an accuracy the inputs do not carry.
    r = rate.to(torch.float32)
    h = gram.to(torch.float32)

    # Pre-surgery conflict uses the original credits and the original Gram, so a
    # pair counts as conflicting exactly when r_i r_j H_ij < 0.
    conflict_pre = _off_diagonal_conflict_fraction(r.unsqueeze(0) * h * r.unsqueeze(1))

    a = torch.diag(r.clone())
    # The conflict score of row i against reference j is r_j * (A H)[i, j], so the
    # whole double loop only needs A H. Recomputed one row at a time: a projection
    # changes one entry of one row, so no other row of A H moves.
    rows = a @ h
    projections = 0
    considered = 0
    skipped_zero_reference = 0
    skipped_degenerate_diagonal = 0

    for i in range(m):
        for j in _cyclic_order(m, i):
            if float(r[j]) == 0.0:
                # Section 3 makes r_j g_j the reference, and a zero credit is the zero
                # vector: projecting against it would divide by a vanishing norm and
                # change nothing.
                skipped_zero_reference += 1
                continue
            h_jj = float(h[j, j])
            if h_jj <= eps_h:
                skipped_degenerate_diagonal += 1
                continue
            considered += 1
            current_dot_with_gj = float(rows[i, j])
            if float(r[j]) * current_dot_with_gj < 0.0:
                a[i, j] -= current_dot_with_gj / h_jj
                rows[i] = a[i] @ h
                projections += 1

    rate_out = a.sum(dim=0)
    if not bool(torch.isfinite(rate_out).all()):
        raise ValueError("ALIGN produced a non-finite credit")

    # Post-surgery conflict is measured on the modified contributions, which is
    # A H A^T and not diag(rate_out) H: two individually acceptable credits can
    # still oppose each other once a projection has mixed them.
    post = _off_diagonal_conflict_fraction(a @ h @ a.T)
    return AlignChunkResult(
        rate=rate_out,
        conflict_fraction_pre=conflict_pre,
        conflict_fraction_post=post,
        projections=projections,
        considered=considered,
        skipped_zero_reference=skipped_zero_reference,
        skipped_degenerate_diagonal=skipped_degenerate_diagonal,
        row_relative_change_mean=_row_relative_change(r, h, a),
    )


def _row_relative_change(r: torch.Tensor, h: torch.Tensor, a: torch.Tensor) -> float:
    """Mean over atoms of ``||v~_i - v_i||_H / (||v_i||_H + eps)``.

    Measured in head geometry rather than in raw coefficients, because a coefficient
    change that leaves the head direction alone is not an intervention worth
    reporting. The H-norm of a coefficient vector ``x`` is ``sqrt(x^T H x)``. An atom
    whose original contribution has zero norm is skipped rather than counted as
    zero, which would read as "untouched" when it means "undefined".
    """
    if h.dtype != torch.float64:
        h64 = h.to(torch.float64)
    else:
        h64 = h
    diagonal = torch.diagonal(h64)
    total = 0.0
    count = 0
    for i in range(int(r.numel())):
        original_sq = float(r[i]) ** 2 * float(diagonal[i])
        if original_sq <= 0.0:
            continue
        # v~_i - v_i in coefficient space: both live in the same basis g, so the
        # difference is coefficient-wise and a[i] minus the original single entry.
        difference = a[i].to(torch.float64).clone()
        difference[i] -= float(r[i])
        delta_sq = float(difference @ h64 @ difference)
        total += (delta_sq**0.5) / (original_sq**0.5 + ALIGN_EPS_NORM)
        count += 1
    return total / count if count else 0.0


def align_chunks(
    rate: torch.Tensor,
    gram: torch.Tensor,
    partition: Sequence[tuple[int, int]],
    *,
    eps_h: float = ALIGN_EPS_H,
) -> dict[str, object]:
    """Run ALIGN over every chunk and scatter the credits back to atom positions.

    The Gram is block diagonal, so each chunk is transformed from its own block and
    no cross-chunk pair is ever considered - the independence §2 assumes.
    """
    rate = rate.detach()
    n = int(rate.numel())
    effective = rate.clone()
    results: list[AlignChunkResult] = []

    projections = 0
    considered = 0
    skipped_zero_reference = 0
    skipped_degenerate_diagonal = 0
    conflict_pre_weighted = 0.0
    conflict_post_weighted = 0.0
    row_change_weighted = 0.0
    multi_atom_weight = 0.0

    for start, end in partition:
        if end - start <= 1:
            continue
        result = align_chunk(rate[start:end], gram[start:end, start:end], eps_h=eps_h)
        effective[start:end] = result.rate.to(effective.dtype)
        results.append(result)
        size = float(end - start)
        multi_atom_weight += size
        projections += result.projections
        considered += result.considered
        skipped_zero_reference += result.skipped_zero_reference
        skipped_degenerate_diagonal += result.skipped_degenerate_diagonal
        # Chunk-level fractions are averaged by atom count, not by chunk count: a
        # 12-atom chunk and a 2-atom chunk would otherwise carry the same weight.
        conflict_pre_weighted += result.conflict_fraction_pre * size
        conflict_post_weighted += result.conflict_fraction_post * size
        row_change_weighted += result.row_relative_change_mean * size

    if not bool(torch.isfinite(effective).all()):
        raise ValueError("ALIGN produced a non-finite credit")

    denominator = multi_atom_weight if multi_atom_weight > 0 else 1.0
    considered_total = (
        considered + skipped_zero_reference + skipped_degenerate_diagonal
    )
    return {
        "rate": effective,
        "conflict_fraction_pre": conflict_pre_weighted / denominator,
        "conflict_fraction_post": conflict_post_weighted / denominator,
        "projection_fraction": projections / considered if considered else 0.0,
        "projections": float(projections),
        "considered_pairs": float(considered),
        "skipped_zero_reference": float(skipped_zero_reference),
        "skipped_degenerate_diagonal": float(skipped_degenerate_diagonal),
        "row_relative_change_mean": row_change_weighted / denominator,
        "multi_atom_chunk_fraction": multi_atom_weight / float(n) if n else 0.0,
        "degenerate_hdiag_skip_fraction": (
            skipped_degenerate_diagonal / considered_total if considered_total else 0.0
        ),
        "results": results,
    }


def align_metrics(
    rate: torch.Tensor,
    effective: torch.Tensor,
    gram: torch.Tensor,
    stats: dict[str, object],
) -> dict[str, torch.Tensor]:
    """The ``mp_opd_align_*`` diagnostics of §6.

    Accumulated in fp64: these are ratios of norms and dot products over the whole
    credit vector, which is cheap next to the Gram, and the precision keeps a
    near-orthogonal pair from reporting a cosine outside [-1, 1].
    """
    r = rate.detach().to(torch.float64)
    e = effective.detach().to(torch.float64)
    h = gram.detach().to(torch.float64)
    n = max(r.numel(), 1)

    change = float((e - r).norm()) / (float(r.norm()) + ALIGN_EPS_NORM)
    nonzero = (r != 0) & (e != 0)
    sign_flip = float(((r[nonzero] * e[nonzero]) < 0).sum()) / float(
        max(int(nonzero.sum()), 1)
    )

    r_h_r = float(r @ h @ r)
    e_h_e = float(e @ h @ e)
    cosine = float(r @ h @ e) / (
        (max(r_h_r, 0.0) ** 0.5) * (max(e_h_e, 0.0) ** 0.5) + ALIGN_EPS_NORM
    )
    norm_ratio = (max(e_h_e, 0.0) ** 0.5) / (
        max(r_h_r, 0.0) + ALIGN_EPS_NORM
    ) ** 0.5

    def metric(value: float) -> torch.Tensor:
        return rate.detach().new_tensor(float(value))

    return {
        "mp_opd_align_conflict_fraction_pre": metric(stats["conflict_fraction_pre"]),
        "mp_opd_align_conflict_fraction_post": metric(stats["conflict_fraction_post"]),
        "mp_opd_align_projection_fraction": metric(stats["projection_fraction"]),
        "mp_opd_align_credit_relative_l2_change": metric(change),
        "mp_opd_align_credit_sign_flip_fraction": metric(sign_flip),
        "mp_opd_align_row_relative_change_mean": metric(
            stats["row_relative_change_mean"]
        ),
        "mp_opd_align_head_update_cosine": metric(cosine),
        "mp_opd_align_head_update_norm_ratio": metric(norm_ratio),
        "mp_opd_align_multi_atom_chunk_fraction": metric(
            stats["multi_atom_chunk_fraction"]
        ),
        "mp_opd_align_degenerate_hdiag_skip_fraction": metric(
            stats["degenerate_hdiag_skip_fraction"]
        ),
        "mp_opd_align_projection_count": metric(stats["projections"]),
        "mp_opd_align_considered_pairs": metric(stats["considered_pairs"]),
        "mp_opd_align_atom_count": metric(float(n)),
    }
