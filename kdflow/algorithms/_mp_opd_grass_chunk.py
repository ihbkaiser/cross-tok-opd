"""GRASS-Chunk: gradient-risk adaptive credit shrinkage inside fixed alignment chunks.

This is the no-DP sibling of :mod:`_mp_opd_grass_span`. The SURE machinery is
deliberately *identical* - same working noise model, same exact output-head Gram,
same ``D_c``/``V_c`` closed forms, same closed-form shrinkage - and exactly one
thing changes:

    GRASS-DP    chooses the partition by searching candidate spans with a
                dynamic program, then shrinks inside each selected span;
    GRASS-Chunk takes the partition from the upstream cross-tokenizer alignment
                and only shrinks inside it.

Keeping the statistic fixed and removing the search is the point. GRASS-DP bundles
two hypotheses - that risk-adaptive shrinkage helps, and that SURE-driven boundary
search helps - and a difference between it and a plain pooled baseline cannot say
which one carried it. GRASS-Chunk isolates the first.

What is therefore *absent* on purpose, asserted by
``tests/mp_opd/test_grass_chunk.py``:

* no candidate-span enumeration;
* no maximum span length of its own (section 6.3: the native chunk is used exactly
  as delivered, and its cost is *measured* rather than truncated);
* no dynamic programming and no partition-selection cost;
* SURE gain is logged and never used to accept, reject, split, merge or reorder a
  chunk (section 10).

Reuse, not duplication: the Gram, the noise estimator, the shrink and the
conservation check all come from :mod:`_mp_opd_grass_span`. The per-chunk
strengths are laid out in that module's ``[start, length - 1]`` table shape so
:func:`grass_shrink` - and with it the exact conservation invariant - is the very
same code path GRASS-DP trains on rather than a second copy that could drift.

Scope, stated plainly: this implements sections 3-14 of the method note and the
telemetry of section 18. It does **not** implement exact full-Transformer gradient
geometry, nor a global risk optimum; both are listed as non-claims in
``docs/mp_opd_design.md`` and must be reported as such.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from ._mp_opd_grass_span import (
    GrassHeadGram,
    _as_vector,
    _partition_index,
    _shrunk_strength,
    atom_head_gram,
    grass_cosine,
    grass_span_ids,
    grass_shrink,
    grass_update_energy,
    hard_pooled_credits,
)

# Chunk-length buckets reported as telemetry. A fixed reporting range, independent
# of the recipe, so two runs whose native chunks have different lengths stay
# comparable. Chunk length here is an atom count, not a token count.
REPORTED_CHUNK_LENGTHS = (1, 2, 3, 4, 8, 16)

STRADDLE_SINGLETON = "singleton"
STRADDLE_MAJORITY = "majority"
STRADDLE_POLICIES = (STRADDLE_SINGLETON, STRADDLE_MAJORITY)


def hard_chunk_credits(
    rate: torch.Tensor, weight: torch.Tensor, partition: Sequence[tuple[int, int]]
) -> torch.Tensor:
    """Hard alignment-chunk pooling, the ``alpha = 1`` comparator of section 15.2.

    Delegated to :func:`hard_pooled_credits` on purpose: the comparator is only
    worth reporting if it is the *same* code that Fixed-k and GBV train on, and a
    second copy of the pooling arithmetic could differ from theirs by exactly the
    amount the comparison is meant to expose.
    """
    return hard_pooled_credits(rate * weight, weight, partition)


# ---------------------------------------------------------------------------
# Section 2: chunk structure supplied by the alignment procedure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtomChunkAssignment:
    """Which upstream chunk each atom belongs to, and what that cost to decide.

    ``ids[i]`` is the aligner's chunk id for atom ``i``, or a negative sentinel
    when the atom could not be assigned. ``straddling`` and ``unaligned`` count
    the two cases the method note's precondition ("every valid atom belongs to
    exactly one chunk") does not cover; they are counted rather than smoothed
    over, because either one silently changes which atoms are treated as
    neighbours.
    """

    ids: tuple[int, ...]
    straddling: int
    unaligned: int
    policy: str


def atom_chunk_ids_from_tokens(
    token_chunk_ids: Sequence[int],
    atom_ranges: Sequence[tuple[int, int]],
    *,
    policy: str = STRADDLE_SINGLETON,
) -> AtomChunkAssignment:
    """Project token-level alignment chunk ids onto atoms.

    The aligner labels every *token*; the credit is carried by *atoms*. A chunk
    boundary falling strictly inside an atom is the only disagreement the two
    structures can have, and it is resolved by ``policy`` rather than by moving a
    boundary.

    * ``singleton`` - the atom is left unassigned and becomes its own chunk, so
      its credit is preserved untouched instead of being pooled with a neighbour
      the alignment never related it to;
    * ``majority`` - the atom joins the chunk holding most of its tokens, ties
      broken by the smaller chunk id so the choice is deterministic.

    Both keep weighted credit conservation exact; they differ only in which
    atoms end up sharing a chunk, and both are counted and reported.
    """
    if policy not in STRADDLE_POLICIES:
        raise ValueError(f"policy must be one of {STRADDLE_POLICIES}")
    ids: list[int] = []
    straddling = 0
    unaligned = 0
    sentinel = 0
    for start, end in atom_ranges:
        window = [int(value) for value in token_chunk_ids[start:end]]
        present = [value for value in window if value >= 0]
        if not present:
            # Each unassigned atom gets its own fresh sentinel so it becomes a
            # singleton chunk rather than being merged with its neighbours.
            sentinel -= 1
            ids.append(sentinel)
            unaligned += 1
            continue
        distinct = set(present)
        if len(distinct) == 1:
            ids.append(present[0])
            continue
        straddling += 1
        if policy == STRADDLE_MAJORITY:
            counts: dict[int, int] = {}
            for value in present:
                counts[value] = counts.get(value, 0) + 1
            ids.append(min(counts, key=lambda value: (-counts[value], value)))
        else:
            sentinel -= 1
            ids.append(sentinel)
    return AtomChunkAssignment(
        ids=tuple(ids), straddling=straddling, unaligned=unaligned, policy=policy
    )


def run_chunk_assignment(n: int, run_length: int) -> AtomChunkAssignment:
    """Fixed-run chunks - the alignment-free baseline of section 2.1.

    A fixed count is explicitly *not* the intended core chunk definition. It exists
    so the mode can be exercised without an audited cross-tokenizer projection, and
    every metric emitted from it carries ``chunk_source=run`` so such a run can
    never be mistaken for one on native alignment chunks.
    """
    if n <= 0:
        raise ValueError("at least one atom is required")
    run_length = int(run_length)
    if run_length < 1:
        raise ValueError("run_length must be positive")
    return AtomChunkAssignment(
        ids=tuple(index // run_length for index in range(n)),
        straddling=0,
        unaligned=0,
        policy="run",
    )


def chunk_partition(
    assignment: AtomChunkAssignment,
) -> tuple[tuple[tuple[int, int], ...], int]:
    """``(full-cover partition, non-contiguous chunk id count)`` over atom indices.

    The downstream shrink is written against a *contiguous* partition, and
    rewriting it for an arbitrary grouping would create a second, untested credit
    path. A chunk id that reappears after a different id therefore cannot form one
    group: the run is split at the change and the id is counted, because merging
    across the gap would silently reorder which atoms count as neighbours.

    The count is over *ids with more than one run*, not over repeated positions: a
    three-atom chunk is one chunk, and counting its second and third atom as splits
    would make the ordinary case look pathological.
    """
    ids = assignment.ids
    n = len(ids)
    if n == 0:
        raise ValueError("at least one atom is required")
    spans: list[tuple[int, int]] = []
    runs_of: dict[int, int] = {}
    current = ids[0]
    run_index = 0
    runs_of[current] = 1
    for index in range(1, n + 1):
        if index < n and ids[index] == current:
            continue
        spans.append((run_index, index))
        run_index = index
        if index < n:
            current = ids[index]
            runs_of[current] = runs_of.get(current, 0) + 1
    split_ids = sum(1 for count in runs_of.values() if count > 1)
    return tuple(spans), split_ids


# ---------------------------------------------------------------------------
# Section 6.2: blockwise within-chunk head Gram
# ---------------------------------------------------------------------------


def chunk_head_gram(
    logits: torch.Tensor,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    atom_ranges: Sequence[tuple[int, int]],
    partition: Sequence[tuple[int, int]],
    *,
    selected_log_prob: torch.Tensor | None = None,
    softcap: float | None = None,
    token_weight: torch.Tensor | None = None,
    head_bias: bool = False,
    diagonal_only: bool = False,
    row_chunk_atoms: int | None = None,
    vocab_chunk: int | None = None,
) -> GrassHeadGram:
    """Exact ``H_c`` per fixed chunk, assembled into one ``[n, n]`` matrix.

    Section 6.2 asks only for pairs *inside* one chunk, so the Gram is computed
    chunk by chunk: each call is the same exact banded routine GRASS-DP uses, run
    on that chunk's token slice with the chunk's own atom count as the admissible
    span length. A single global band wide enough for the longest chunk would
    instead evaluate every atom pair within that distance of every other atom
    regardless of chunk, which for a handful of long chunks is orders of magnitude
    more work than the method needs.

    Across chunks the returned matrix is exactly zero, because section 12 decides
    each ``alpha_c`` from ``H_c`` alone and ignores cross-chunk terms.
    """
    n = len(atom_ranges)
    # float32, like the per-chunk routine it assembles: holding the assembly in
    # float64 would advertise precision the fp32 blocks do not have while doubling
    # the [n, n] buffer - 134 MB at n = 4096.
    gram = torch.zeros((n, n), dtype=torch.float32, device=logits.device)
    symmetry_error = 0.0
    covered = 0
    for start, end in partition:
        token_low = int(atom_ranges[start][0])
        token_high = int(atom_ranges[end - 1][1])
        sub_ranges = tuple(
            (atom_ranges[index][0] - token_low, atom_ranges[index][1] - token_low)
            for index in range(start, end)
        )
        block = atom_head_gram(
            logits[token_low:token_high],
            hidden[token_low:token_high],
            labels[token_low:token_high],
            sub_ranges,
            end - start,
            selected_log_prob=None
            if selected_log_prob is None
            else selected_log_prob[token_low:token_high],
            softcap=softcap,
            token_weight=None if token_weight is None else token_weight[token_low:token_high],
            head_bias=head_bias,
            diagonal_only=diagonal_only,
            row_chunk_atoms=row_chunk_atoms,
            vocab_chunk=vocab_chunk,
        )
        symmetry_error = max(symmetry_error, block.symmetry_error)
        covered += block.token_count
        gram[start:end, start:end] = block.gram.to(gram.dtype)
    return GrassHeadGram(
        gram=gram,
        diagonal_only=bool(diagonal_only),
        symmetry_error=symmetry_error,
        token_count=covered,
        atom_count=n,
    )


# ---------------------------------------------------------------------------
# Sections 7-10: per-chunk SURE statistics and shrinkage strength
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GrassChunkTables:
    """Per-chunk ``D_c``, ``V_c``, ``alpha_c`` and the diagnostic SURE gain.

    Every statistical vector is indexed by *chunk*. There is no candidate axis in
    this mode, and keeping one would be a step back toward the dynamic program.
    """

    partition: tuple[tuple[int, int], ...]
    lengths: torch.Tensor
    alpha: torch.Tensor
    distortion: torch.Tensor
    variance: torch.Tensor
    trace: torch.Tensor
    gain: torch.Tensor
    sigma2: float
    eps_d: float
    negative_tol: float
    degenerate: bool

    @property
    def chunk_count(self) -> int:
        return len(self.partition)


def _chunk_strength(
    distortion: torch.Tensor, variance: torch.Tensor, eps_d: float
) -> torch.Tensor:
    """``clip(V/D, 0, 1)`` with the conservative branches of section 9.1.

    Shared with GRASS-DP verbatim: the two modes are meant to differ in *where
    the partition comes from* and nowhere else, so a second copy of the branch
    rule would make that comparison unverifiable. The last branch is the
    conservative one the method note insists on - when the geometry cannot
    separate atomic from pooled credit, leave the credit alone.
    """
    return _shrunk_strength(distortion, variance, eps_d)


def grass_chunk_tables(
    rate: torch.Tensor,
    weight: torch.Tensor,
    gram: torch.Tensor,
    partition: Sequence[tuple[int, int]],
    sigma2: float,
    *,
    eps_d: float = 0.0,
    negative_tol_rel: float = 1e-6,
) -> GrassChunkTables:
    """``D_c``, ``V_c``, ``alpha_c`` and ``G_c`` for every fixed chunk.

    The closed forms are the ones GRASS-DP uses, and for the same reason: both
    ``A_c r_c = r_c - rbar_c 1`` and
    ``A_c Sigma_c = sigma^2 (diag(1/w) - 11^T/W)``, the latter PSD by
    Cauchy-Schwarz, so ``V_c >= 0`` holds by construction and a materially
    negative value is a fault rather than a quantity to tune. What is new here is
    only that the group is *given* instead of searched.
    """
    r = _as_vector("rate", rate).detach().to(torch.float64)
    w = _as_vector("weight", weight).detach().to(torch.float64)
    if r.shape != w.shape:
        raise ValueError("rate and weight must share one shape")
    n = r.numel()
    if n == 0:
        raise ValueError("at least one atom is required")
    if not torch.isfinite(r).all():
        raise ValueError("rates must be finite")
    if not torch.isfinite(w).all() or (w <= 0).any():
        raise ValueError("weights must be finite and positive")
    if gram.ndim != 2 or tuple(gram.shape) != (n, n):
        raise ValueError("gram must be [n,n] and align with the credit vectors")
    if not torch.isfinite(gram).all():
        raise ValueError("the head Gram must be finite")
    sigma2 = float(sigma2)
    if not math.isfinite(sigma2) or sigma2 < 0:
        raise ValueError("sigma2 must be finite and nonnegative")
    eps_d = float(eps_d)
    if eps_d < 0 or not math.isfinite(eps_d):
        raise ValueError("eps_d must be finite and nonnegative")
    negative_tol_rel = float(negative_tol_rel)
    if negative_tol_rel < 0 or not math.isfinite(negative_tol_rel):
        raise ValueError("negative_tol_rel must be finite and nonnegative")

    spans = tuple((int(start), int(end)) for start, end in partition)
    # _partition_index owns the contiguity/coverage rule, so the two modes reject
    # a malformed partition with the same message.
    lengths, _span_ids, _starts = _partition_index(spans, n, r.device)

    # Section 17.2: symmetrize numerically before forming D_c and V_c.
    h = gram.detach().to(torch.float64)
    h = (h + h.T) * 0.5
    diagonal = torch.diagonal(h).contiguous()
    row_sum_prefix = torch.cat((h.new_zeros(1), h.sum(dim=1).cumsum(0)))
    weight_prefix = torch.cat((h.new_zeros(1), w.cumsum(0)))
    weighted_rate_prefix = torch.cat((h.new_zeros(1), (w * r).cumsum(0)))
    # Per-atom trace contribution, prefix-summed so a chunk's trace is a scalar
    # difference rather than a gather that would have to be padded.
    trace_prefix = torch.cat((h.new_zeros(1), (diagonal / w).cumsum(0)))

    starts = torch.tensor([start for start, _end in spans], dtype=torch.int64, device=r.device)
    ends = torch.tensor([end for _start, end in spans], dtype=torch.int64, device=r.device)
    positions = torch.arange(int(lengths.max()), device=r.device).view(1, -1)
    # Chunks have different lengths, so the block gather is padded to the widest one.
    # Padded slots point at their own chunk's first atom - always a valid index - and
    # are then zeroed by multiplying a masked deviation, so no junk can reach D_c.
    inside = positions < lengths.unsqueeze(-1)
    index = torch.where(inside, positions + starts.unsqueeze(-1), starts.unsqueeze(-1))

    chunk_weight = weight_prefix[ends] - weight_prefix[starts]
    chunk_rate = (weighted_rate_prefix[ends] - weighted_rate_prefix[starts]) / chunk_weight
    deviation = (r[index] - chunk_rate.unsqueeze(-1)) * inside
    # `index` already holds absolute atom ids, so the block is the two-axis gather -
    # adding the two index tensors instead would address start+i + start+j.
    block = h[index.unsqueeze(-1), index.unsqueeze(-2)]
    distortion = (deviation.unsqueeze(-1) * block * deviation.unsqueeze(-2)).sum(dim=(1, 2))
    trace = sigma2 * (trace_prefix[ends] - trace_prefix[starts])
    cross = row_sum_prefix[ends] - row_sum_prefix[starts]
    variance = trace - sigma2 * cross / chunk_weight

    non_singleton = lengths > 1
    # Section 4 / 13.5: a singleton chunk is pinned to alpha = 0 and reported as
    # such, so the degenerate branch can never pool an atom with itself.
    alpha = torch.where(
        non_singleton, _chunk_strength(distortion, variance, eps_d), torch.zeros_like(variance)
    )
    # Both statistics are identically zero for a singleton chunk: there is no
    # second atom, so there is no within-chunk heterogeneity to observe and no
    # within-chunk trace to sum. They are pinned to exact zero rather than left to
    # the prefix-sum difference below, which would report the cancellation residue
    # of sigma^2 * H_ii/w_i - sigma^2 * H_ii/w_i as if it were a real quantity.
    zeroed = torch.zeros_like(trace)
    trace = torch.where(non_singleton, trace, zeroed)
    variance = torch.where(non_singleton, variance, zeroed)
    # Section 9.1: a tiny negative D is clamped to zero before it is used, and
    # the raw value is kept in the reported table so a real sign failure surfaces
    # in the pathology counter instead of being hidden here.
    usable_distortion = torch.where(
        distortion > eps_d, distortion, torch.zeros_like(distortion)
    )
    # Section 10: logged, never used to accept, reject, split, merge or reorder.
    gain = 2.0 * alpha * variance - alpha * alpha * usable_distortion

    # The pathology tolerance needs a scale that survives a response of nothing but
    # singleton chunks. Taking it from the *reported* trace would give exactly zero
    # there, so the float64 cancellation residue (~1e-18) would be counted as a sign
    # failure on every sample. D_c and V_c both scale with the Gram diagonal, so that
    # is the honest scale.
    negative_tol = negative_tol_rel * max(
        float(trace.max()), float(diagonal.abs().max()), float(torch.finfo(torch.float64).tiny)
    )
    return GrassChunkTables(
        partition=spans,
        lengths=lengths,
        alpha=alpha,
        distortion=distortion,
        variance=variance,
        trace=trace,
        gain=gain,
        sigma2=sigma2,
        eps_d=eps_d,
        negative_tol=negative_tol,
        degenerate=sigma2 <= 0.0,
    )


def chunk_alpha_table(
    tables: GrassChunkTables, n: int, device: torch.device
) -> torch.Tensor:
    """Lay the per-chunk strengths out in the ``[start, length - 1]`` table shape.

    Reusing that layout is what lets :func:`grass_shrink` - and the conservation
    invariant it enforces - stay the same code path GRASS-DP trains on.
    """
    width = max(int(tables.lengths.max()), 1)
    table = torch.zeros((n, width), dtype=torch.float64, device=device)
    for index, (start, end) in enumerate(tables.partition):
        table[start, end - start - 1] = tables.alpha[index]
    return table


def chunk_span_ids(
    partition: Sequence[tuple[int, int]], n: int, device: torch.device
) -> torch.Tensor:
    """Chunk index of every atom, under the same name GRASS-DP uses for spans.

    A chunk *is* a span here - the only thing that differs is how its boundaries
    were chosen - so the Gram masking, the energy read-out and the adjacency mask
    all operate on this one vector.
    """
    return grass_span_ids(partition, n, device)


def apply_chunk_shrinkage(
    rate: torch.Tensor,
    weight: torch.Tensor,
    tables: GrassChunkTables,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shrink in place of the partition search and return ``(shrunk, residual)``.

    Delegates to :func:`grass_shrink`, which is the shared implementation whose
    conservation invariant section 17.1 makes mandatory.
    """
    n = rate.numel()
    table = chunk_alpha_table(tables, n, rate.device)
    shrunk, residual = grass_shrink(rate, weight, tables.partition, table)
    return shrunk, residual


# ---------------------------------------------------------------------------
# Section 18: telemetry
# ---------------------------------------------------------------------------


def chunk_boundary_diagnostics(
    rate: torch.Tensor,
    weight: torch.Tensor,
    gram: torch.Tensor | None,
    partition: Sequence[tuple[int, int]],
) -> dict[str, float]:
    """Section 18.7: are the upstream boundaries statistically meaningful at all?

    Within-chunk and cross-chunk adjacent pairs are compared on the same
    quantities. If the two populations are indistinguishable, the fixed chunk
    structure is a weak inductive bias for credit shrinkage, and that is worth
    knowing before attributing any result to the method.

    Keys are unprefixed so the caller owns the metric namespace.
    """
    r = _as_vector("rate", rate).detach().to(torch.float64)
    w = _as_vector("weight", weight).detach().to(torch.float64)
    n = r.numel()
    inside = torch.ones(max(n - 1, 0), dtype=torch.bool, device=r.device)
    for start, end in partition:
        if end - start < 2:
            continue
        # Pair index i joins atoms i and i+1, so a chunk [start, end) owns exactly the
        # pairs i in [start, end - 2]. The pair i = end - 1 reaches into the next chunk
        # and is the single boundary pair this chunk must keep out of the "within"
        # population. Clearing the range [start, end - 1] instead would drop precisely
        # the pairs the chunk owns and keep the boundary one - the inverse of the
        # intent, and it makes the two populations describe opposite things.
        if end - 1 < inside.numel():
            inside[end - 1] = False
    scale = (1.0 / w[:-1] + 1.0 / w[1:]).clamp_min(1e-300).sqrt()
    absolute = (r[1:] - r[:-1]).abs()
    normalized = absolute / scale
    metrics = {
        "adjacent_pairs_within": float(inside.sum()),
        "adjacent_pairs_cross": float((~inside).sum()),
    }
    for name, selection in (("within", inside), ("cross", ~inside)):
        metrics[f"{name}_abs_diff_mean"] = (
            float(absolute[selection].mean()) if bool(selection.any()) else 0.0
        )
        metrics[f"{name}_normalized_diff_mean"] = (
            float(normalized[selection].mean()) if bool(selection.any()) else 0.0
        )
        metrics[f"{name}_gradient_cosine_mean"] = 0.0
    if gram is not None and n > 1:
        h = gram.detach().to(torch.float64)
        diagonal = torch.diagonal(h)
        denominator = (diagonal[:-1] * diagonal[1:]).clamp_min(1e-300).sqrt()
        cosine = h[:-1, 1:] / denominator
        for name, selection in (("within", inside), ("cross", ~inside)):
            if bool(selection.any()):
                metrics[f"{name}_gradient_cosine_mean"] = float(cosine[selection].mean())
    return metrics


def grass_chunk_metrics(
    tables: GrassChunkTables,
    rate: torch.Tensor,
    weight: torch.Tensor,
    gram: torch.Tensor,
    shrunk: torch.Tensor,
    conservation_error: torch.Tensor,
    *,
    assignment: AtomChunkAssignment,
    noncontiguous_splits: int,
    source: str,
) -> dict[str, torch.Tensor]:
    """Sections 18.2-18.4, 18.6 and 18.8 for one response."""
    r = _as_vector("rate", rate).detach().to(torch.float64)
    shrunk64 = shrunk.detach().to(torch.float64)
    n = r.numel()
    lengths = tables.lengths
    alpha = tables.alpha
    tolerance = r.new_tensor(tables.negative_tol)
    positive = tables.distortion > 0
    gamma = torch.where(
        positive,
        tables.variance / tables.distortion.clamp_min(1e-300),
        torch.zeros_like(tables.variance),
    )
    span_ids = chunk_span_ids(tables.partition, n, r.device)
    atomic_energy = grass_update_energy(gram, torch.arange(n, device=r.device), r)
    chunk_energy = grass_update_energy(gram, span_ids, shrunk64)
    metrics = {
        "mp_opd_grass_chunk_count": r.new_tensor(float(tables.chunk_count)),
        "mp_opd_grass_chunk_source": r.new_tensor(1.0 if source == "xtoken" else 0.0),
        "mp_opd_grass_chunk_length_mean": lengths.to(torch.float64).mean(),
        "mp_opd_grass_chunk_length_max": lengths.max().to(torch.float64),
        "mp_opd_grass_chunk_singleton_fraction": (lengths == 1).to(torch.float64).mean(),
        "mp_opd_grass_chunk_non_singleton_fraction": (lengths > 1)
        .to(torch.float64)
        .mean(),
        "mp_opd_grass_chunk_alpha_mean": alpha.mean(),
        "mp_opd_grass_chunk_alpha_median": alpha.median(),
        "mp_opd_grass_chunk_alpha_atomic_fraction": (alpha < 0.1).to(torch.float64).mean(),
        "mp_opd_grass_chunk_alpha_hard_fraction": (alpha > 0.9).to(torch.float64).mean(),
        "mp_opd_grass_chunk_alpha_clip_low_fraction": (alpha <= 1e-12)
        .to(torch.float64)
        .mean(),
        "mp_opd_grass_chunk_alpha_clip_high_fraction": (alpha >= 1.0 - 1e-12)
        .to(torch.float64)
        .mean(),
        "mp_opd_grass_chunk_distortion_mean": tables.distortion.mean(),
        "mp_opd_grass_chunk_distortion_median": tables.distortion.median(),
        "mp_opd_grass_chunk_variance_mean": tables.variance.mean(),
        "mp_opd_grass_chunk_variance_median": tables.variance.median(),
        "mp_opd_grass_chunk_gamma_mean": gamma.mean(),
        "mp_opd_grass_chunk_sigma2": r.new_tensor(tables.sigma2),
        "mp_opd_grass_chunk_degenerate": r.new_tensor(float(tables.degenerate)),
        # Section 10: a diagnostic, never a decision input.
        "mp_opd_grass_chunk_sure_gain_mean": tables.gain.mean(),
        "mp_opd_grass_chunk_sure_gain_median": tables.gain.median(),
        "mp_opd_grass_chunk_sure_gain_positive_fraction": (tables.gain > 0)
        .to(torch.float64)
        .mean(),
        "mp_opd_grass_chunk_pathology_negative_d_fraction": (
            tables.distortion < -tolerance
        ).to(torch.float64).mean(),
        "mp_opd_grass_chunk_pathology_negative_v_fraction": (
            tables.variance < -tolerance
        ).to(torch.float64).mean(),
        "mp_opd_grass_chunk_head_update_energy_atomic": atomic_energy,
        "mp_opd_grass_chunk_head_update_energy_shrunk": chunk_energy,
        "mp_opd_grass_chunk_head_update_cosine": grass_cosine(
            grass_update_energy(gram, span_ids, r, shrunk64), atomic_energy, chunk_energy
        ),
        "mp_opd_grass_chunk_credit_relative_l2_change": (shrunk64 - r).norm()
        / r.norm().clamp_min(1e-300),
        "mp_opd_grass_chunk_credit_sign_change_fraction": (r * shrunk64 < 0)
        .to(torch.float64)
        .mean(),
        # Section 18.8: the conservation invariant, measured on what is trained.
        "mp_opd_grass_chunk_conservation_error": conservation_error.detach(),
        "mp_opd_grass_chunk_mapping_straddling_atoms": r.new_tensor(
            float(assignment.straddling)
        ),
        "mp_opd_grass_chunk_mapping_unaligned_atoms": r.new_tensor(
            float(assignment.unaligned)
        ),
        "mp_opd_grass_chunk_mapping_noncontiguous_splits": r.new_tensor(
            float(noncontiguous_splits)
        ),
    }
    for reported in REPORTED_CHUNK_LENGTHS:
        selection = lengths == reported
        if not bool(selection.any()):
            continue
        metrics[f"mp_opd_grass_chunk_length_{reported}_fraction"] = (
            selection.to(torch.float64).mean()
        )
        metrics[f"mp_opd_grass_chunk_length_{reported}_alpha_mean"] = alpha[selection].mean()
        metrics[f"mp_opd_grass_chunk_length_{reported}_sure_gain_mean"] = tables.gain[
            selection
        ].mean()
    return {key: value.detach() for key, value in metrics.items()}


def chunk_gram_diagnostics(
    gram: torch.Tensor, tables: GrassChunkTables, *, max_sampled: int = 8
) -> dict[str, torch.Tensor]:
    """Section 17.3 / 18.6: PSD check on a *sampled* subset of chunks.

    Eigendecomposing every chunk inside the training loop would cost more than the
    Gram itself, so the sample is bounded and the bound is itself reported.
    """
    h = gram.detach().to(torch.float64)
    diagonal = torch.diagonal(h)
    metrics = {
        "mp_opd_grass_chunk_gram_diagonal_mean": diagonal.mean(),
        "mp_opd_grass_chunk_gram_diagonal_max": diagonal.max(),
    }
    indices = [
        index for index, (start, end) in enumerate(tables.partition) if end - start >= 2
    ]
    if not indices:
        return {key: value.detach() for key, value in metrics.items()}
    step = max(1, len(indices) // max(int(max_sampled), 1))
    sampled = indices[::step][: int(max_sampled)]
    smallest = torch.stack(
        [
            torch.linalg.eigvalsh(h[start:end, start:end]).min()
            for start, end in (tables.partition[index] for index in sampled)
        ]
    )
    metrics["mp_opd_grass_chunk_gram_sampled_chunks"] = h.new_tensor(float(len(sampled)))
    metrics["mp_opd_grass_chunk_gram_sampled_min_eigenvalue"] = smallest.min()
    metrics["mp_opd_grass_chunk_gram_sampled_negative_fraction"] = (
        (smallest < -1e-8 * smallest.abs().clamp_min(1e-300).max())
    ).to(torch.float64).mean()
    cosines = []
    for index in sampled:
        start, end = tables.partition[index]
        block = h[start:end, start:end]
        rows = torch.arange(end - start - 1, device=h.device).unsqueeze(-1)
        columns = rows + 1
        denominator = (
            diagonal[start:end][rows] * diagonal[start:end][columns]
        ).clamp_min(1e-300).sqrt()
        cosines.append((block[rows, columns] / denominator).reshape(-1))
    cosine = cosines[0] if len(cosines) == 1 else torch.cat(cosines)
    metrics["mp_opd_grass_chunk_gram_neighbour_cosine_mean"] = cosine.mean()
    metrics["mp_opd_grass_chunk_gram_neighbour_cosine_negative_fraction"] = (
        (cosine < 0).to(torch.float64).mean()
    )
    return {key: value.detach() for key, value in metrics.items()}


def chunk_shadow_metrics(
    gram: torch.Tensor,
    rate: torch.Tensor,
    weight: torch.Tensor,
    partition: Sequence[tuple[int, int]],
    shrunk: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Hard chunk pooling and atomic on the same batch, for the section 20 table.

    Both share this mode's boundaries; only the amount of shrinkage differs.
    """
    r = rate.detach().to(torch.float64)
    shrunk64 = shrunk.detach().to(torch.float64)
    n = r.numel()
    pooled = hard_chunk_credits(r, weight, partition)
    span_ids = chunk_span_ids(partition, n, r.device)
    atomic_ids = torch.arange(n, device=r.device)
    atomic_energy = grass_update_energy(gram, atomic_ids, r)
    pooled_energy = grass_update_energy(gram, span_ids, pooled)
    chunk_energy = grass_update_energy(gram, span_ids, shrunk64)
    return {
        "mp_opd_grass_chunk_shadow_hard_chunk_head_cosine_to_grass": grass_cosine(
            grass_update_energy(gram, span_ids, pooled, shrunk64),
            pooled_energy,
            chunk_energy,
        ),
        "mp_opd_grass_chunk_shadow_atomic_head_cosine_to_grass": grass_cosine(
            grass_update_energy(gram, atomic_ids, r, shrunk64), atomic_energy, chunk_energy
        ),
        "mp_opd_grass_chunk_shadow_hard_chunk_credit_l2_change": (pooled - r).norm(),
        "mp_opd_grass_chunk_shadow_hard_chunk_credit_relative_l2_change": (pooled - r).norm()
        / r.norm().clamp_min(1e-300),
    }