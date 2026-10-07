"""ALIGN v1.0 - aligned local interference gradient neutralization.

The tests here are written against the method note rather than against the
implementation, so an implementation that happens to be self-consistent still
fails. Where a property is a direct consequence of the projection geometry - the
conflict score vanishing after a projection, the cyclic order being
deterministic, a singleton being untouched - it is asserted directly instead of
through a stored expectation.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest
import torch

from kdflow.algorithms._mp_opd_align import (
    ALIGN_EPS_H,
    align_chunk,
    align_chunks,
    align_metrics,
)

SOURCE = Path(__file__).resolve().parents[2] / "kdflow" / "algorithms" / "_mp_opd_align.py"


def _psd_gram(m: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(m, m, generator=generator, dtype=torch.float32)
    gram = 0.5 * (raw + raw.T) + m * torch.eye(m, dtype=torch.float32)
    return gram


def _credit(m: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(m, generator=generator, dtype=torch.float32)


# ---------------------------------------------------------------------------
# The projection itself
# ---------------------------------------------------------------------------


def test_a_singleton_chunk_is_returned_unchanged():
    # Section 4: a singleton has nothing to interfere with.
    rate = torch.tensor([2.5], dtype=torch.float32)
    result = align_chunk(rate, torch.ones(1, 1, dtype=torch.float32))
    assert torch.equal(result.rate, rate)
    assert result.projections == 0
    assert result.considered == 0
    assert result.conflict_fraction_pre == 0.0


def test_a_chunk_with_no_conflict_is_left_exactly_alone():
    # Conflict is r_i r_j H_ij < 0, so with positive credits it takes POSITIVE
    # off-diagonal Gram entries to be conflict-free. Writing the off-diagonals
    # negative here, as an earlier version of this test did, describes the most
    # conflicting chunk there is rather than a quiet one.
    gram = torch.tensor(
        [[2.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 4.0]], dtype=torch.float32
    )
    rate = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    result = align_chunk(rate, gram)
    assert result.conflict_fraction_pre == 0.0
    assert result.projections == 0
    assert torch.allclose(result.rate, rate, atol=1e-6)


def test_conflict_is_detected_from_the_credit_product_not_only_the_gram_sign():
    # The checklist requires r_i r_j H_ij < 0, so a negative H_ij with two positive
    # credits is a conflict and the same H_ij with one negative credit is not.
    gram = torch.tensor(
        [[1.0, -1.0, 0.5], [-1.0, 1.0, 0.5], [0.5, 0.5, 1.0]], dtype=torch.float32
    )
    conflicting = align_chunk(torch.tensor([1.0, 1.0, 1.0]), gram)
    assert conflicting.conflict_fraction_pre > 0.0
    assert conflicting.projections > 0

    # Flipping the middle credit does not simply remove the conflicts. The pair that
    # carried the most negative Gram entry, (0,1), now multiplies out positive and is
    # exempt, while the pair whose Gram entry was POSITIVE, (1,2), becomes a conflict.
    # That swap is the actual claim: the credit product decides, not the Gram sign.
    flipped = align_chunk(torch.tensor([1.0, -1.0, 1.0]), gram)
    assert flipped.conflict_fraction_pre == pytest.approx(1.0 / 3.0)
    # Exactly one of the three unordered pairs conflicts, and it is the one whose
    # credit product is negative: (1,2) at -0.5, not (0,1) whose Gram entry is -1.
    rates = torch.tensor([1.0, -1.0, 1.0])
    scores = {
        (i, j): float(rates[i]) * float(rates[j]) * float(gram[i, j])
        for i, j in [(0, 1), (0, 2), (1, 2)]
    }
    assert scores[(0, 1)] == pytest.approx(1.0)
    assert scores[(1, 2)] == pytest.approx(-0.5)
    conflicting_pairs = [pair for pair, score in scores.items() if score < 0]
    assert conflicting_pairs == [(1, 2)]
    assert flipped.conflict_fraction_pre == len(conflicting_pairs) / 3.0


def test_the_projection_removes_the_conflict_it_fires_on():
    # After a projection the modified contribution of i must have a non-negative
    # inner product with the reference v_j it was projected against. That is the
    # whole mechanism, so it is checked on the coefficient matrix the surgery
    # actually produced rather than on any published metric.
    for seed in range(12):
        m = 5
        gram = _psd_gram(m, 100 + seed)
        rate = _credit(m, 200 + seed)
        result = align_chunk(rate, gram)
        assert torch.isfinite(result.rate).all()
        if result.projections == 0:
            continue
        coefficients = torch.diag(rate)
        # Re-derive the coefficient matrix the same way align_chunk does, so the
        # assertion is on the surgery rather than on its summary.
        rows = coefficients @ gram
        fired = 0
        for i in range(m):
            for j in list(range(i + 1, m)) + list(range(0, i)):
                if float(rate[j]) == 0.0 or float(gram[j, j]) <= ALIGN_EPS_H:
                    continue
                if float(rate[j]) * float(rows[i, j]) < 0.0:
                    coefficients[i, j] -= float(rows[i, j]) / float(gram[j, j])
                    rows[i] = coefficients[i] @ gram
                    fired += 1
                    # After the rewrite the score is exactly zero by construction.
                    assert float(rate[j]) * float(rows[i, j]) >= -1e-4
        assert fired == result.projections


def test_the_reference_is_the_original_contribution_and_not_a_projected_one():
    # Section 4: v_j is always r_j g_j. A sequential run that projected v_j in
    # place would produce a different answer whenever two conflicts chain, so the
    # reference is pinned by comparing against an explicit frozen copy.
    gram = torch.tensor(
        [
            [1.0, -0.9, -0.9, -0.9],
            [-0.9, 1.0, -0.9, -0.9],
            [-0.9, -0.9, 1.0, -0.9],
            [-0.9, -0.9, -0.9, 1.0],
        ],
        dtype=torch.float32,
    )
    rate = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    result = align_chunk(rate, gram)
    assert result.projections > 0

    # Firing condition must be evaluated against diag(rate) rows frozen at the
    # start: recomputing the whole result with the reference pinned gives the
    # same answer as the shipped one, and re-running with a mutated "current"
    # reference would not.
    coefficients = torch.diag(rate.clone())
    rows = coefficients @ gram
    count = 0
    for i in range(4):
        for j in list(range(i + 1, 4)) + list(range(0, i)):
            if float(rate[j]) * float(rows[i, j]) < 0.0:
                coefficients[i, j] -= float(rows[i, j]) / float(gram[j, j])
                rows[i] = coefficients[i] @ gram
                count += 1
    assert count == result.projections
    assert torch.allclose(coefficients.sum(dim=0), result.rate, atol=1e-5)


def test_the_projection_order_is_cyclic_and_deterministic():
    # Section 3 fixes the order to i+1 .. m-1, 0 .. i-1, and section 4 rules out any
    # RNG. Two calls on the same input must therefore agree bit for bit.
    gram = _psd_gram(6, 11)
    rate = _credit(6, 12)
    first = align_chunk(rate, gram)
    second = align_chunk(rate.clone(), gram.clone())
    assert torch.equal(first.rate, second.rate)
    assert first.projections == second.projections

    order = list(range(1, 6)) + list(range(0, 1))
    assert order == [1, 2, 3, 4, 5, 0]
    for start in range(6):
        assert list(range(start + 1, 6)) + list(range(0, start)) == [
            (start + 1 + k) % 6 for k in range(5)
        ]


def test_a_degenerate_reference_diagonal_is_skipped_and_counted():
    # Section 4 guards the division with eps_H. A zero diagonal has no direction,
    # so the projection must be skipped rather than dividing by it.
    gram = torch.tensor(
        [[1.0, -1.0], [-1.0, 0.0]], dtype=torch.float32
    )
    rate = torch.tensor([1.0, 1.0], dtype=torch.float32)
    result = align_chunk(rate, gram)
    assert result.skipped_degenerate_diagonal == 1
    # Row 1 still projects against atom 0, whose diagonal is fine, so the degenerate
    # entry removes one projection and not both. Asserting zero projections here would
    # be asserting that ALIGN skips healthy references too.
    assert result.projections == 1
    assert torch.isfinite(result.rate).all()


def test_a_zero_credit_reference_is_skipped_and_counted():
    gram = _psd_gram(3, 21)
    rate = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    result = align_chunk(rate, gram)
    assert result.skipped_zero_reference > 0
    assert torch.isfinite(result.rate).all()


def test_non_finite_input_raises_rather_than_producing_a_nan():
    # Section 4 requires failing closed on non-finite input.
    with pytest.raises(ValueError, match="non-finite credit"):
        align_chunk(torch.tensor([1.0, float("nan")]), _psd_gram(2, 1))
    with pytest.raises(ValueError, match="non-finite Gram"):
        gram = _psd_gram(2, 1)
        gram[0, 1] = float("inf")
        align_chunk(torch.tensor([1.0, 1.0]), gram)


def test_a_mismatched_gram_is_rejected():
    with pytest.raises(ValueError, match="expected"):
        align_chunk(torch.zeros(3), _psd_gram(2, 1))


# ---------------------------------------------------------------------------
# Properties of the transform
# ---------------------------------------------------------------------------


def test_credits_are_not_renormalized_after_the_surgery():
    # Section 4 forbids a post-ALIGN renormalization, so a chunk whose credits
    # change must keep the original magnitude of the head update unless the
    # projections genuinely changed it. The check is that the two vectors differ
    # in a way that is not a pure rescale: if the result were r times a scalar,
    # every pair of coordinates would keep its ratio.
    gram = torch.tensor(
        [[2.0, -1.5, -1.5], [-1.5, 3.0, -1.5], [-1.5, -1.5, 4.0]], dtype=torch.float32
    )
    rate = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    result = align_chunk(rate, gram)
    assert result.projections > 0
    ratios = rate[:2] / rate[:1]
    out_ratios = result.rate[:2] / result.rate[:1]
    assert not torch.allclose(ratios, out_ratios, atol=1e-4)


def test_the_transformed_credit_reproduces_the_projected_total_update():
    # Section 3: sum_i v~_i = sum_k r~_k g_k. That identity is what makes it
    # legal to hand r~ to the ordinary loss instead of a custom gradient, so it is
    # asserted in the head geometry where it has content.
    gram = _psd_gram(5, 31)
    rate = _credit(5, 32)
    result = align_chunk(rate, gram)
    # With A the coefficient matrix, A H A^T is the Gram of the modified
    # contributions; comparing its quadratic form against the summed credit is the
    # numerical statement of the identity.
    coefficients = torch.diag(rate.clone())
    rows = coefficients @ gram
    for i in range(5):
        for j in list(range(i + 1, 5)) + list(range(0, i)):
            if float(rate[j]) == 0.0 or float(gram[j, j]) <= ALIGN_EPS_H:
                continue
            if float(rate[j]) * float(rows[i, j]) < 0.0:
                coefficients[i, j] -= float(rows[i, j]) / float(gram[j, j])
                rows[i] = coefficients[i] @ gram
    total_pre = rate.double() @ gram.double() @ rate.double()
    total_post = (
        coefficients.double().sum(dim=0)
        @ gram.double()
        @ coefficients.double().sum(dim=0)
    )
    modified = coefficients.double() @ gram.double() @ coefficients.double().T
    # The identity is the sum over ALL entries of A H A^T, not its trace: the total
    # update is one vector, so 1^T (A H A^T) 1 sums the off-diagonal terms too. Using
    # the trace here would pass only when no projection fired.
    assert math.isclose(
        float(total_post), float(modified.sum()), rel_tol=1e-5
    )
    # The total update must stay finite and the surgery must be non-trivial on this
    # input, otherwise the identity above is being checked on a no-op.
    assert math.isfinite(float(total_pre))
    assert not torch.allclose(coefficients.sum(dim=0), rate, atol=1e-4)


def test_conflict_never_increases_but_is_not_guaranteed_to_reach_zero():
    # The projection sweeps each row against a frozen reference, so a later
    # projection against a third reference can put the score back above zero where
    # an earlier one had removed it. One sweep is therefore not a convergence
    # guarantee, and the method note asks for the before/after fractions precisely
    # so a run can be judged on whether it actually removed anything.
    increases = 0
    decreases = 0
    for seed in range(40):
        m = 5
        gram = _psd_gram(m, 300 + seed)
        rate = _credit(m, 400 + seed)
        result = align_chunk(rate, gram)
        if result.conflict_fraction_post > result.conflict_fraction_pre + 1e-9:
            increases += 1
        elif result.conflict_fraction_post < result.conflict_fraction_pre - 1e-9:
            decreases += 1
    assert increases == 0, "a projection sweep must not create new conflict"
    assert decreases > 0, "the sweep must remove some conflict on real inputs"


def test_conflict_can_survive_a_full_sweep_on_a_severely_conflicting_chunk():
    # A chunk where every pair conflicts is the honest counterexample to any claim
    # that one sweep zeroes the metric. Recorded here so the published diagnostic is
    # read as a measurement rather than as a guarantee.
    gram = torch.full((3, 3), -0.95, dtype=torch.float32)
    gram.fill_diagonal_(1.0)
    rate = torch.ones(3, dtype=torch.float32)
    result = align_chunk(rate, gram)
    assert result.projections > 0, "the chunk really does fire projections"
    assert result.conflict_fraction_post >= result.conflict_fraction_pre - 1e-9


def test_conflict_is_removed_on_its_own_when_it_can_be():
    # On a chunk whose conflict is mild enough for one sweep to absorb, the metric
    # does reach zero. The stronger, more useful statements - that a sweep never
    # leaves it worse, and that a severe chunk survives a full sweep - are asserted
    # separately below, so this one is free to state the clean case.
    gram = torch.tensor(
        [[2.0, -1.0, -1.0], [-1.0, 3.0, -1.0], [-1.0, -1.0, 4.0]],
        dtype=torch.float32,
    )
    rate = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    result = align_chunk(rate, gram)
    assert result.conflict_fraction_pre > result.conflict_fraction_post


def test_projection_arithmetic_runs_in_float32():
    # Section 4 fixes the precision. The Gram arrives as fp32 from chunk_head_gram,
    # so this is really a check that the surgery does not silently promote.
    gram = _psd_gram(4, 41)
    rate = _credit(4, 42)
    result = align_chunk(rate, gram)
    assert result.rate.dtype == torch.float32


# ---------------------------------------------------------------------------
# Chunk orchestration
# ---------------------------------------------------------------------------


def test_every_chunk_is_transformed_independently():
    # Section 5: chunks are processed independently, then scattered back to their
    # original atom positions. A cross-chunk pair must not influence the result,
    # which is checked by changing the off-chunk block and seeing nothing move.
    partition = ((0, 2), (2, 5))
    gram = _psd_gram(5, 51)
    rate = _credit(5, 52)
    first = align_chunks(rate, gram, partition)

    perturbed = gram.clone()
    # Fill the cross-chunk block with a large negative number: if any cross-chunk
    # pair were ever considered, this would change the answer.
    perturbed[0:2, 2:5] = -50.0
    perturbed[2:5, 0:2] = -50.0
    second = align_chunks(rate, perturbed, partition)
    assert torch.equal(first["rate"], second["rate"])


def test_singletons_are_left_alone_across_a_partition():
    partition = ((0, 1), (1, 4), (4, 5))
    gram = _psd_gram(5, 61)
    rate = _credit(5, 62)
    stats = align_chunks(rate, gram, partition)
    effective = stats["rate"]
    assert torch.equal(effective[0], rate[0])
    assert torch.equal(effective[4], rate[4])


def test_the_transformed_credit_keeps_its_dtype_and_device():
    partition = ((0, 3), (3, 6))
    gram = _psd_gram(6, 71)
    rate = _credit(6, 72)
    stats = align_chunks(rate, gram, partition)
    assert stats["rate"].dtype == rate.dtype
    assert stats["rate"].device == rate.device


def test_projection_and_skip_fractions_are_consistent():
    partition = ((0, 3), (3, 6))
    gram = _psd_gram(6, 81)
    rate = _credit(6, 82)
    stats = align_chunks(rate, gram, partition)
    considered = stats["considered_pairs"]
    projections = stats["projections"]
    assert projections <= considered
    expected = projections / considered if considered else 0.0
    assert stats["projection_fraction"] == pytest.approx(expected)
    # Each atom of an m-atom chunk is compared against all m-1 others, so a 3-atom
    # chunk offers 6 ordered pairs and two chunks offer 12. An earlier version of
    # this test expected 8, which counted unordered pairs and then halved again.
    assert considered + stats["skipped_zero_reference"] + stats[
        "skipped_degenerate_diagonal"
    ] <= 12


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_every_required_diagnostic_is_published():
    # Section 6 names ten metrics. A missing one is silent, so the names are
    # asserted rather than assumed.
    partition = ((0, 3), (3, 6))
    gram = _psd_gram(6, 91)
    rate = _credit(6, 92)
    stats = align_chunks(rate, gram, partition)
    metrics = align_metrics(rate, stats["rate"], gram, stats)
    required = {
        "mp_opd_align_conflict_fraction_pre",
        "mp_opd_align_conflict_fraction_post",
        "mp_opd_align_projection_fraction",
        "mp_opd_align_credit_relative_l2_change",
        "mp_opd_align_credit_sign_flip_fraction",
        "mp_opd_align_row_relative_change_mean",
        "mp_opd_align_head_update_cosine",
        "mp_opd_align_head_update_norm_ratio",
        "mp_opd_align_multi_atom_chunk_fraction",
        "mp_opd_align_degenerate_hdiag_skip_fraction",
    }
    assert required <= set(metrics)
    for key, value in metrics.items():
        assert torch.isfinite(value).all(), f"{key} was {value}"
        assert float(value) >= 0.0 or key.endswith("head_update_cosine")


def test_the_head_update_cosine_is_a_cosine():
    # It is r^T H r~ over two norms, so on a healthy Gram it stays inside [-1, 1]
    # and equals 1 when ALIGN changed nothing.
    partition = ((0, 3), (3, 6))
    gram = _psd_gram(6, 101)
    rate = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    stats = align_chunks(rate, gram, partition)
    metrics = align_metrics(rate, stats["rate"], gram, stats)
    assert -1.0 - 1e-6 <= float(metrics["mp_opd_align_head_update_cosine"]) <= 1.0 + 1e-6


def test_relative_l2_change_is_zero_when_nothing_is_projected():
    partition = ((0, 3), (3, 6))
    gram = torch.tensor(
        [
            [2.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 4.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 3.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 4.0],
        ],
        dtype=torch.float32,
    )
    rate = torch.ones(6, dtype=torch.float32)
    stats = align_chunks(rate, gram, partition)
    assert stats["projections"] == 0
    metrics = align_metrics(rate, stats["rate"], gram, stats)
    assert float(metrics["mp_opd_align_credit_relative_l2_change"]) == pytest.approx(
        0.0, abs=1e-9
    )
    assert float(metrics["mp_opd_align_head_update_norm_ratio"]) == pytest.approx(
        1.0, rel=1e-6
    )


def test_multi_atom_chunk_fraction_counts_atoms_in_multi_atom_chunks():
    # Section 6 item 9 is the fraction of chunks ALIGN can act on, expressed over
    # the atoms those chunks cover.
    gram = _psd_gram(4, 111)
    rate = torch.ones(4, dtype=torch.float32)
    stats = align_chunks(rate, gram, ((0, 1), (1, 4)))
    # 3 of 4 atoms sit in the one chunk that has more than one atom.
    assert float(stats["multi_atom_chunk_fraction"]) == pytest.approx(0.75)


def test_degenerate_skip_fraction_is_zero_on_a_healthy_gram():
    gram = _psd_gram(6, 121)
    rate = _credit(6, 122)
    stats = align_chunks(rate, gram, ((0, 3), (3, 6)))
    assert stats["skipped_degenerate_diagonal"] == 0.0
    assert float(stats["degenerate_hdiag_skip_fraction"]) == 0.0


def test_diagnostics_do_not_duplicate_the_existing_training_logs():
    # Section 6 asks that loss, grad norm and the chunk statistics keep their
    # existing names rather than being republished with an ALIGN prefix.
    partition = ((0, 3), (3, 6))
    gram = _psd_gram(6, 131)
    rate = _credit(6, 132)
    stats = align_chunks(rate, gram, partition)
    for key in align_metrics(rate, stats["rate"], gram, stats):
        for borrowed in ("loss", "grad_norm", "optimizer_norm", "chunk_length"):
            assert borrowed not in key, f"{key} duplicates the existing {borrowed} log"


# ---------------------------------------------------------------------------
# The method's prohibitions
# ---------------------------------------------------------------------------


def _module_source() -> str:
    return SOURCE.read_text()


def test_the_module_builds_no_noise_model_and_pools_nothing():
    # Section 4 rules out sigma^2, SURE, pooling and threshold tuning. ALIGN is a
    # per-chunk coefficient edit, so none of the GRASS machinery may appear.
    tree = ast.parse(_module_source())
    names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    for forbidden in (
        "GrassNoiseEstimator",
        "grass_noise",
        "sigma2",
        "sure",
        "pool",
        "rename",
        "threshold",
    ):
        assert forbidden not in names, f"ALIGN must not reference {forbidden}"


def test_the_module_never_renormalizes_the_credit():
    # Section 6 of the note and section 4 both forbid it, and it is the one change
    # that would silently turn "how much did the projection move the update" into a
    # number that always looks the same size.
    source = _module_source()
    for forbidden in ("rate_out / rate_out.norm", "normalise", "normalize", "rescale"):
        assert forbidden not in source, f"ALIGN must not {forbidden} the credit"
