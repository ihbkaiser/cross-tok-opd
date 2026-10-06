"""GRASS-Chunk: fixed alignment chunks, no partition search.

The interesting properties of this mode are mostly *absences*: no span search, no
maximum span length, no dynamic program, and a SURE gain that is reported but
never allowed to move a boundary. Several tests below assert those absences
structurally, because they are exactly what could regress silently - a chunk mode
that quietly grew a search would still pass every numerical test.
"""
import ast as _ast
import re
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kdflow.algorithms._mp_opd_grass_chunk import (
    STRADDLE_MAJORITY,
    STRADDLE_SINGLETON,
    AtomChunkAssignment,
    apply_chunk_shrinkage,
    chunk_boundary_diagnostics,
    chunk_gram_diagnostics,
    chunk_head_gram,
    chunk_partition,
    chunk_shadow_metrics,
    chunk_span_ids,
    grass_chunk_metrics,
    grass_chunk_tables,
    hard_chunk_credits,
    run_chunk_assignment,
)
from kdflow.algorithms._mp_opd_grass_chunk import atom_chunk_ids_from_tokens
from kdflow.algorithms._mp_opd_grass_span import (
    CREDIT_CONSERVATION_REL_TOL,
    GrassNoiseEstimator,
    atom_head_gram,
    grass_credit_residual,
)


def _credits(n: int, seed: int = 7, spread: float = 1.0):
    generator = torch.Generator().manual_seed(seed)
    weight = (1.0 + torch.rand(n, generator=generator, dtype=torch.float64) * 4.0)
    rate = spread * torch.randn(n, generator=generator, dtype=torch.float64)
    return rate, weight


def _psd(n: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(n, n, generator=generator, dtype=torch.float64)
    gram = 0.5 * (raw + raw.T)
    values, vectors = torch.linalg.eigh(gram)
    return vectors @ torch.diag(values.clamp_min(1e-6)) @ vectors.T


def _chunked_gram(partition, gram):
    """Zero every cross-chunk entry, which is what chunk_head_gram produces."""
    ids = chunk_span_ids(partition, gram.shape[0], gram.device)
    return gram * (ids.unsqueeze(-1) == ids.unsqueeze(0))


# ---------------------------------------------------------------------------
# Section 2: chunk structure comes from upstream, never from a search
# ---------------------------------------------------------------------------


def test_token_chunk_ids_project_onto_atoms():
    token_ids = [0, 0, 1, 1, 2, 2]
    assignment = atom_chunk_ids_from_tokens(token_ids, ((0, 2), (2, 4), (4, 6)))
    assert assignment.ids == (0, 1, 2)
    assert assignment.straddling == 0
    assert assignment.unaligned == 0
    assert assignment.policy == STRADDLE_SINGLETON


def test_unaligned_and_straddling_atoms_are_counted_not_hidden():
    # Atom 1 spans tokens 2..4, which the aligner split across chunks 1 and 2;
    # atom 2 covers two tokens the aligner could not assign at all.
    token_ids = [0, 0, 1, 2, -1, -1]
    ranges = ((0, 2), (2, 4), (4, 6))
    for policy in (STRADDLE_SINGLETON, STRADDLE_MAJORITY):
        assignment = atom_chunk_ids_from_tokens(token_ids, ranges, policy=policy)
        assert assignment.straddling == 1
        assert assignment.unaligned == 1
        partition, _splits = chunk_partition(assignment)
        # Three atoms, and neither a straddling nor an unaligned atom is merged
        # with a neighbour the alignment never related it to.
        assert partition == ((0, 1), (1, 2), (2, 3))


def test_majority_straddle_policy_joins_the_dominant_chunk():
    # Atom 1 covers tokens [1, 2, 2]: chunk 2 holds two of the three tokens.
    token_ids = [0, 0, 1, 2, 2, 2]
    ranges = ((0, 2), (2, 5), (5, 6))

    majority = atom_chunk_ids_from_tokens(token_ids, ranges, policy=STRADDLE_MAJORITY)
    assert majority.straddling == 1
    assert majority.ids == (0, 2, 2)
    majority_partition, _ = chunk_partition(majority)
    assert majority_partition == ((0, 1), (1, 3))

    split = atom_chunk_ids_from_tokens(token_ids, ranges)
    assert split.ids[1] < 0
    split_partition, _ = chunk_partition(split)
    assert split_partition == ((0, 1), (1, 2), (2, 3))


def test_majority_breaks_ties_towards_the_smaller_chunk_id():
    # Atom 1 covers one token from each chunk, so the counts tie exactly.
    token_ids = [7, 7, 9, 5]
    assignment = atom_chunk_ids_from_tokens(
        token_ids, ((0, 2), (2, 4)), policy=STRADDLE_MAJORITY
    )
    assert assignment.straddling == 1
    assert assignment.ids == (7, 5)


def test_a_reappearing_chunk_id_splits_rather_than_merging():
    assignment = AtomChunkAssignment(ids=(0, 1, 0), straddling=0, unaligned=0, policy="test")
    partition, splits = chunk_partition(assignment)
    assert partition == ((0, 1), (1, 2), (2, 3))
    assert splits == 1


def test_a_multi_atom_chunk_is_not_counted_as_splits():
    """Regression: the counter used to count repeated *positions*, not repeated ids."""
    assignment = run_chunk_assignment(6, 4)
    partition, splits = chunk_partition(assignment)
    assert partition == ((0, 4), (4, 6))
    assert splits == 0


def test_run_assignment_is_the_documented_baseline():
    assignment = run_chunk_assignment(5, 2)
    assert assignment.ids == (0, 0, 1, 1, 2)
    assert assignment.policy == "run"
    partition, splits = chunk_partition(assignment)
    assert partition == ((0, 2), (2, 4), (4, 5))
    assert splits == 0


def test_chunk_partition_rejects_an_empty_atoms_list():
    with pytest.raises(ValueError):
        chunk_partition(AtomChunkAssignment(ids=(), straddling=0, unaligned=0, policy="t"))
    with pytest.raises(ValueError):
        atom_chunk_ids_from_tokens([0, 0], ((0, 2),), policy="nonsense")
    with pytest.raises(ValueError):
        run_chunk_assignment(0, 2)
    with pytest.raises(ValueError):
        run_chunk_assignment(4, 0)


# ---------------------------------------------------------------------------
# Section 6.2: the exact head Gram, computed per chunk
# ---------------------------------------------------------------------------


def test_chunk_head_gram_matches_the_unrestricted_gram_inside_each_chunk():
    torch.manual_seed(21)
    tokens = 8
    logits = torch.randn(tokens, 13, dtype=torch.float64)
    hidden = torch.randn(tokens, 4, dtype=torch.float64)
    labels = torch.arange(tokens)
    ranges = ((0, 1), (1, 3), (3, 4), (4, 8))
    partition = ((0, 2), (2, 4))

    blockwise = chunk_head_gram(logits, hidden, labels, ranges, partition)
    # The same atoms computed in one pass with a band wide enough to reach every
    # within-chunk pair must agree exactly on those entries.
    wide = atom_head_gram(logits, hidden, labels, ranges, 2)
    expected = _chunked_gram(partition, wide.gram).to(torch.float64)
    # chunk_head_gram accumulates each chunk in float64 while atom_head_gram's
    # accumulator is float32, so the comparison happens in float64 against a
    # tolerance the fp32 path can actually resolve rather than an absolute 1e-10.
    scale = max(float(expected.abs().max()), 1.0) * 1e-6
    assert torch.allclose(blockwise.gram.to(torch.float64), expected, atol=scale)
    # Nothing is computed across a chunk boundary.
    ids = chunk_span_ids(partition, 4, logits.device)
    assert float(blockwise.gram[ids.unsqueeze(-1) != ids.unsqueeze(0)].abs().max()) == 0.0
    assert blockwise.atom_count == 4
    assert blockwise.token_count == tokens


def test_chunk_head_gram_singleton_chunks_cost_only_their_diagonal():
    torch.manual_seed(22)
    logits = torch.randn(4, 7, dtype=torch.float64)
    hidden = torch.randn(4, 3, dtype=torch.float64)
    labels = torch.arange(4)
    ranges = tuple((index, index + 1) for index in range(4))
    singles = chunk_head_gram(logits, hidden, labels, ranges, tuple(
        (index, index + 1) for index in range(4)
    ))
    full = atom_head_gram(logits, hidden, labels, ranges, 4)
    diagonal = torch.diagonal(full.gram).to(torch.float64)
    scale = max(float(diagonal.abs().max()), 1.0) * 1e-6
    assert torch.allclose(torch.diagonal(singles.gram).to(torch.float64), diagonal, atol=scale)


def test_chunk_head_gram_diag_ablation_zeroes_the_off_diagonal():
    torch.manual_seed(23)
    logits = torch.randn(5, 9, dtype=torch.float64)
    hidden = torch.randn(5, 3, dtype=torch.float64)
    labels = torch.arange(5)
    ranges = ((0, 2), (2, 5))
    diag = chunk_head_gram(
        logits, hidden, labels, ranges, ((0, 2),), diagonal_only=True
    )
    assert float(diag.gram[0, 1]) == 0.0
    assert float(diag.gram[1, 0]) == 0.0
    assert float(diag.gram[0, 0]) > 0.0


# ---------------------------------------------------------------------------
# Sections 7-9: per-chunk D, V, alpha
# ---------------------------------------------------------------------------


def test_closed_form_matches_an_explicit_matrix_computation():
    n = 6
    rate, weight = _credits(n, seed=31)
    gram = _psd(n, 31)
    partition = ((0, 3), (3, 6))
    sigma2 = 0.25
    tables = grass_chunk_tables(rate, weight, gram, partition, sigma2)

    for index, (start, end) in enumerate(partition):
        block = gram[start:end, start:end]
        w = weight[start:end]
        r = rate[start:end]
        pooled = (w * r).sum() / w.sum()
        deviation = r - pooled
        expected_d = float(deviation @ block @ deviation)
        sigma = sigma2 * torch.diag(1.0 / w)
        pooling = torch.ones(end - start, 1, dtype=block.dtype) @ (w / w.sum()).unsqueeze(0)
        a = torch.eye(end - start, dtype=block.dtype) - pooling
        expected_v = float(torch.trace(block @ a @ sigma))
        # Each half is asserted on its own so a failure names the term that is wrong
        # rather than only the difference of two of them.
        expected_trace = sigma2 * float((torch.diagonal(block) / w).sum())
        expected_cross = sigma2 * float(block.sum()) / float(w.sum())
        assert float(tables.distortion[index]) == pytest.approx(expected_d, rel=1e-10)
        assert float(tables.trace[index]) == pytest.approx(expected_trace, rel=1e-10)
        # Recover the entry sum the routine actually used, rather than trusting that
        # two spellings of the same reduction agree.
        recovered_cross = (
            (float(tables.trace[index]) - float(tables.variance[index]))
            * float(w.sum())
            / sigma2
        )
        assert recovered_cross == pytest.approx(float(block.sum()), rel=1e-10)
        assert float(tables.variance[index]) == pytest.approx(expected_v, rel=1e-10)


def test_singleton_chunks_are_pinned_to_zero_and_never_pool():
    n = 5
    rate, weight = _credits(n, seed=32)
    gram = _psd(n, 32)
    partition = tuple((index, index + 1) for index in range(n))
    tables = grass_chunk_tables(rate, weight, gram, partition, 10.0)
    assert torch.all(tables.alpha == 0)
    # V_c is zero by construction for a singleton; a residual of order 1e-18 from
    # the float64 cancellation is expected and must not leak into the update.
    assert float(tables.variance.abs().max()) < 1e-12
    assert float(tables.trace.abs().max()) == 0.0
    shrunk, residual = apply_chunk_shrinkage(rate, weight, tables)
    assert torch.allclose(shrunk, rate.to(torch.float64))


def test_alpha_matches_the_closed_form_including_the_degenerate_branches():
    n = 6
    rate, weight = _credits(n, seed=33)
    gram = _psd(n, 33)
    partition = ((0, 3), (3, 6))
    tables = grass_chunk_tables(rate, weight, gram, partition, 0.5)
    for index in range(2):
        d = float(tables.distortion[index])
        v = float(tables.variance[index])
        assert float(tables.alpha[index]) == pytest.approx(min(max(v / d, 0.0), 1.0), rel=1e-12)

    # sigma^2 -> 0 means V -> 0 means alpha -> 0: the atomic limit of section 17.4.
    quiet = grass_chunk_tables(rate, weight, gram, partition, 0.0)
    assert torch.all(quiet.alpha == 0)
    assert quiet.degenerate


def test_zero_noise_recovers_atomic_credit_exactly():
    n = 7
    rate, weight = _credits(n, seed=34, spread=3.0)
    gram = _chunked_gram(((0, 4), (4, 7)), _psd(n, 34))
    tables = grass_chunk_tables(rate, weight, gram, ((0, 4), (4, 7)), 0.0)
    shrunk, _ = apply_chunk_shrinkage(rate, weight, tables)
    assert torch.allclose(shrunk, rate.to(torch.float64))


def test_full_pooling_recovers_hard_chunk_pooling():
    n = 6
    rate, weight = _credits(n, seed=35, spread=3.0)
    partition = ((0, 2), (2, 6))
    gram = _chunked_gram(partition, _psd(n, 35))
    # An enormous noise scale drives every strength to the alpha = 1 boundary.
    tables = grass_chunk_tables(rate, weight, gram, partition, 1e6)
    assert torch.all(tables.alpha == 1)
    shrunk, _ = apply_chunk_shrinkage(rate, weight, tables)
    assert torch.allclose(shrunk, hard_chunk_credits(rate, weight, partition), atol=1e-12)


def test_constant_credit_with_noise_pools_every_chunk():
    n = 6
    rate = torch.full((n,), 0.4, dtype=torch.float64)
    weight = torch.arange(1.0, n + 1.0, dtype=torch.float64)
    gram = _chunked_gram(((0, 3), (3, 6)), _psd(n, 36))
    tables = grass_chunk_tables(rate, weight, gram, ((0, 3), (3, 6)), 1.0)
    assert torch.all(tables.alpha == 1)
    shrunk, residual = apply_chunk_shrinkage(rate, weight, tables)
    assert torch.allclose(shrunk, rate, atol=1e-12)
    assert float(residual) < 1e-12


def test_shrinkage_conserves_weighted_credit_in_float32_and_float64():
    n = 9
    rate, weight = _credits(n, seed=37, spread=4.0)
    partition = ((0, 1), (1, 5), (5, 6), (6, 9))
    gram = _chunked_gram(partition, _psd(n, 37))
    tables = grass_chunk_tables(rate, weight, gram, partition, 0.3)
    shrunk, exact_residual = apply_chunk_shrinkage(rate, weight, tables)
    assert float(exact_residual) < 1e-12
    for effective in (shrunk, shrunk.to(torch.float32)):
        residual = grass_credit_residual(rate, weight, effective)
        magnitude = float((weight * rate).abs().sum())
        assert float(residual) <= CREDIT_CONSERVATION_REL_TOL * max(magnitude, 1.0)


def test_loss_gradient_is_the_shrunk_credit_rate():
    from kdflow.algorithms._mp_opd_credit import soft_partition_loss

    n = 5
    rate, weight = _credits(n, seed=38)
    partition = ((0, 2), (2, 5))
    gram = _chunked_gram(partition, _psd(n, 38))
    tables = grass_chunk_tables(rate, weight, gram, partition, 0.2)
    shrunk, _ = apply_chunk_shrinkage(rate, weight, tables)
    nll = torch.rand(n, dtype=torch.float32, requires_grad=True)
    loss = soft_partition_loss(nll, shrunk.to(torch.float32))
    loss.backward()
    assert torch.allclose(nll.grad, shrunk.to(torch.float32), atol=1e-7)


def test_negative_statistics_are_reported_not_hidden():
    n = 5
    rate, weight = _credits(n, seed=39)
    indefinite = torch.eye(n, dtype=torch.float64)
    indefinite[0, 1] = indefinite[1, 0] = 3.0
    tables = grass_chunk_tables(rate, weight, indefinite, ((0, 2), (2, 5)), 1.0)
    metrics = grass_chunk_metrics(
        tables,
        rate,
        weight,
        indefinite,
        rate,
        torch.zeros((), dtype=torch.float64),
        assignment=run_chunk_assignment(n, 2),
        noncontiguous_splits=0,
        source="run",
    )
    assert float(metrics["mp_opd_grass_chunk_pathology_negative_d_fraction"]) > 0.0
    assert float(metrics["mp_opd_grass_chunk_pathology_negative_v_fraction"]) > 0.0
    assert tables.negative_tol >= 0.0


def test_an_all_singleton_response_reports_no_spurious_sign_failure():
    """The pathology tolerance must not collapse to zero when every chunk is length 1."""
    n = 5
    rate, weight = _credits(n, seed=42, spread=4.0)
    gram = _chunked_gram(tuple((i, i + 1) for i in range(n)), _psd(n, 42))
    partition = tuple((index, index + 1) for index in range(n))
    tables = grass_chunk_tables(rate, weight, gram, partition, 1.0)
    assert tables.negative_tol > 0.0
    metrics = grass_chunk_metrics(
        tables, rate, weight, gram, rate, torch.zeros((), dtype=torch.float64),
        assignment=run_chunk_assignment(n, 1), noncontiguous_splits=0, source="run",
    )
    assert float(metrics["mp_opd_grass_chunk_pathology_negative_d_fraction"]) == 0.0
    assert float(metrics["mp_opd_grass_chunk_pathology_negative_v_fraction"]) == 0.0


def test_invalid_chunk_table_inputs_raise():
    rate, weight = _credits(4, seed=41)
    gram = torch.eye(4, dtype=torch.float64)
    with pytest.raises(ValueError):
        grass_chunk_tables(rate, weight, torch.eye(3, dtype=torch.float64), ((0, 4),), 1.0)
    with pytest.raises(ValueError):
        grass_chunk_tables(rate, weight, gram, ((0, 4),), -1.0)
    with pytest.raises(ValueError):
        grass_chunk_tables(rate, weight, gram, ((0, 3),), 1.0)
    with pytest.raises(ValueError):
        grass_chunk_tables(rate, weight, gram, ((0, 2), (3, 4)), 1.0)
    with pytest.raises(ValueError):
        grass_chunk_tables(rate, torch.zeros(4, dtype=torch.float64), gram, ((0, 4),), 1.0)
    with pytest.raises(ValueError):
        grass_chunk_tables(torch.zeros(0), torch.zeros(0), torch.zeros(0, 0), (), 1.0)


# ---------------------------------------------------------------------------
# Section 11: within-chunk adjacency only
# ---------------------------------------------------------------------------


def test_noise_estimator_ignores_pairs_that_cross_a_chunk_boundary():
    estimator = GrassNoiseEstimator(rho=0.9, min_adjacent_pairs=1)
    generator = torch.Generator().manual_seed(51)
    # A MAD scale estimate carries roughly a 10% relative sampling error per hundred
    # pairs. At forty pairs that error is larger than the effect under test, so the
    # assertion would only measure the seed. Twelve hundred pairs puts it at ~1.5%.
    n, sigma, weight_value = 1200, 0.05, 4.0
    weight = torch.full((n,), weight_value, dtype=torch.float64)
    # The latent credit jumps between the two chunks, so exactly one adjacent pair -
    # the boundary one - carries a step. A spike on a single atom would contaminate
    # its other neighbour too and would not test what this is about.
    latent = torch.full((n,), 0.5, dtype=torch.float64)
    latent[n // 2 :] += 60.0
    noise = sigma * torch.randn(n, generator=generator, dtype=torch.float64) / weight.sqrt()
    rate = latent + noise
    same_chunk = torch.ones(n - 1, dtype=torch.bool)
    same_chunk[n // 2 - 1] = False

    inside = estimator.update(rate, weight, same_group=same_chunk)
    assert inside["valid_pairs"] == n - 2
    # Ground truth is the variance of the differences the estimator was handed, not
    # the planted sigma^2: MAD and the second moment are both noisy estimators, and
    # at ~1200 pairs they disagree by a few per cent routinely. A 12% band leaves
    # room for that sampling spread while still failing on any real bias - the
    # earlier masked pair would have shown up as a large *downward* error here.
    standardized = (rate[1:] - rate[:-1]) / (1.0 / weight[:-1] + 1.0 / weight[1:]).sqrt()
    assert estimator.sigma2 == pytest.approx(float(standardized[same_chunk].var()), rel=0.12)
    assert estimator.sigma2 == pytest.approx(sigma**2, rel=0.15)


def test_the_masked_pair_is_a_large_outlier_a_nonrobust_scale_would_absorb():
    """Control: the excluded pair is genuinely enormous, not a negligible extra.

    MAD is robust to a single outlier by design, so 'the masked estimate is
    smaller' is not a meaningful claim - one outlier cannot move a median. The
    meaningful claim is that the pair being excluded carries a step orders of
    magnitude above the noise, which is what would wreck a quadratic scale.
    """
    generator = torch.Generator().manual_seed(51)
    n, sigma, weight_value = 1200, 0.05, 4.0
    weight = torch.full((n,), weight_value, dtype=torch.float64)
    latent = torch.full((n,), 0.5, dtype=torch.float64)
    latent[n // 2 :] += 60.0
    noise = sigma * torch.randn(n, generator=generator, dtype=torch.float64) / weight.sqrt()
    rate = latent + noise

    scale = (1.0 / weight[:-1] + 1.0 / weight[1:]).sqrt()
    z = (rate[1:] - rate[:-1]) / scale
    boundary = n // 2 - 1
    assert float(z[boundary].abs()) > 50.0
    assert float((z**2).mean()) > 50.0 * float((z[z.abs() < 5.0] ** 2).mean())


def test_noise_estimator_rejects_a_mismatched_group_mask():
    estimator = GrassNoiseEstimator()
    rate, weight = _credits(6, seed=52)
    with pytest.raises(ValueError):
        estimator.update(rate, weight, same_group=torch.ones(3, dtype=torch.bool))


def test_chunk_noise_excludes_exactly_the_cross_boundary_pairs():
    n = 9
    rate, weight = _credits(n, seed=53)
    partition = ((0, 4), (4, 9))
    ids = chunk_span_ids(partition, n, rate.device)
    within = ids[:-1] == ids[1:]
    estimator = GrassNoiseEstimator(min_adjacent_pairs=1)
    diagnostics = estimator.observe(rate, weight, same_group=within)
    assert diagnostics["valid_pairs"] == n - 2


# ---------------------------------------------------------------------------
# Section 18: telemetry
# ---------------------------------------------------------------------------


def test_chunk_metrics_cover_the_required_readouts_and_stay_detached():
    n = 8
    rate, weight = _credits(n, seed=61, spread=3.0)
    partition = ((0, 1), (1, 4), (4, 8))
    gram = _chunked_gram(partition, _psd(n, 61))
    tables = grass_chunk_tables(rate, weight, gram, partition, 0.4)
    shrunk, residual = apply_chunk_shrinkage(rate, weight, tables)
    metrics = grass_chunk_metrics(
        tables,
        rate,
        weight,
        gram,
        shrunk,
        residual,
        assignment=AtomChunkAssignment(ids=(0, 1, 1, 1, 2, 2, 2, 2), straddling=1,
                                       unaligned=0, policy=STRADDLE_SINGLETON),
        noncontiguous_splits=0,
        source="xtoken",
    )
    for key in (
        "mp_opd_grass_chunk_count",
        "mp_opd_grass_chunk_length_mean",
        "mp_opd_grass_chunk_singleton_fraction",
        "mp_opd_grass_chunk_alpha_mean",
        "mp_opd_grass_chunk_alpha_atomic_fraction",
        "mp_opd_grass_chunk_alpha_hard_fraction",
        "mp_opd_grass_chunk_distortion_mean",
        "mp_opd_grass_chunk_variance_mean",
        "mp_opd_grass_chunk_sure_gain_mean",
        "mp_opd_grass_chunk_sure_gain_positive_fraction",
        "mp_opd_grass_chunk_conservation_error",
        "mp_opd_grass_chunk_mapping_straddling_atoms",
        "mp_opd_grass_chunk_head_update_cosine",
        "mp_opd_grass_chunk_length_1_fraction",
        "mp_opd_grass_chunk_length_4_fraction",
    ):
        assert key in metrics, key
        assert torch.isfinite(metrics[key]).all(), key
        assert not metrics[key].requires_grad, key
    assert float(metrics["mp_opd_grass_chunk_source"]) == 1.0
    assert float(metrics["mp_opd_grass_chunk_length_1_fraction"]) == pytest.approx(1 / 3)
    assert float(metrics["mp_opd_grass_chunk_mapping_straddling_atoms"]) == 1.0


def test_run_source_is_tagged_so_it_is_never_mistaken_for_a_native_chunk():
    n = 4
    rate, weight = _credits(n, seed=62)
    partition = ((0, 2), (2, 4))
    gram = _chunked_gram(partition, _psd(n, 62))
    tables = grass_chunk_tables(rate, weight, gram, partition, 0.1)
    metrics = grass_chunk_metrics(
        tables, rate, weight, gram, rate, torch.zeros((), dtype=torch.float64),
        assignment=run_chunk_assignment(n, 2), noncontiguous_splits=0, source="run",
    )
    assert float(metrics["mp_opd_grass_chunk_source"]) == 0.0


def test_gram_diagnostics_sample_instead_of_eigendecomposing_every_chunk():
    n = 12
    rate, weight = _credits(n, seed=63)
    partition = tuple((index, index + 2) for index in range(0, n, 2))
    gram = _chunked_gram(partition, _psd(n, 63))
    tables = grass_chunk_tables(rate, weight, gram, partition, 0.2)
    metrics = chunk_gram_diagnostics(gram, tables, max_sampled=3)
    assert 1 <= float(metrics["mp_opd_grass_chunk_gram_sampled_chunks"]) <= 3
    assert float(metrics["mp_opd_grass_chunk_gram_sampled_min_eigenvalue"]) >= -1e-8
    assert float(metrics["mp_opd_grass_chunk_gram_sampled_negative_fraction"]) == 0.0
    assert "mp_opd_grass_chunk_gram_neighbour_cosine_mean" in metrics


def test_gram_diagnostics_short_circuit_on_all_singleton_chunks():
    n = 3
    rate, weight = _credits(n, seed=64)
    partition = tuple((index, index + 1) for index in range(n))
    gram = _chunked_gram(partition, _psd(n, 64))
    tables = grass_chunk_tables(rate, weight, gram, partition, 0.2)
    metrics = chunk_gram_diagnostics(gram, tables)
    assert "mp_opd_grass_chunk_gram_sampled_chunks" not in metrics


def test_boundary_diagnostics_separate_within_from_cross_chunk_neighbours():
    n = 8
    partition = ((0, 4), (4, 8))
    # The latent rate is constant inside each chunk, so a within-chunk difference is
    # pure noise while the single cross-chunk pair carries a real step. With
    # independent random credits inside the chunks the two populations overlap and
    # the comparison measures the seed rather than the partition.
    generator = torch.Generator().manual_seed(65)
    weight = torch.full((n,), 2.0, dtype=torch.float64)
    latent = torch.full((n,), 0.5, dtype=torch.float64)
    latent[4:] += 4.0
    rate = latent + 0.05 * torch.randn(n, generator=generator, dtype=torch.float64) / weight.sqrt()
    gram = _chunked_gram(partition, _psd(n, 65))
    metrics = chunk_boundary_diagnostics(rate, weight, gram, partition)
    assert metrics["adjacent_pairs_within"] == n - 2
    assert metrics["adjacent_pairs_cross"] == 1
    # The cross-chunk pair is the largest jump, because atom 3 to 4 is the
    # partition boundary.
    assert metrics["cross_abs_diff_mean"] > 10.0 * metrics["within_abs_diff_mean"]
    assert "cross_gradient_cosine_mean" in metrics


def test_boundary_diagnostics_handle_the_first_chunk_starting_at_zero():
    n = 4
    rate, weight = _credits(n, seed=66)
    metrics = chunk_boundary_diagnostics(rate, weight, None, ((0, 2), (2, 4)))
    # Pairs are (0,1), (1,2), (2,3). Only (1,2) crosses a boundary, and it is pair
    # index 1 - the first chunk starting at zero must not shift that index.
    assert metrics["adjacent_pairs_within"] == 2
    assert metrics["adjacent_pairs_cross"] == 1
    difference = (rate[1:] - rate[:-1]).abs()
    assert metrics["cross_abs_diff_mean"] == pytest.approx(float(difference[1]))
    assert metrics["within_abs_diff_mean"] == pytest.approx(
        float((difference[0] + difference[2]) / 2)
    )
    assert metrics["within_gradient_cosine_mean"] == 0.0


def test_shadow_reports_hard_chunk_and_atomic_without_changing_anything():
    n = 6
    rate, weight = _credits(n, seed=67, spread=3.0)
    partition = ((0, 3), (3, 6))
    gram = _chunked_gram(partition, _psd(n, 67))
    tables = grass_chunk_tables(rate, weight, gram, partition, 0.3)
    shrunk, _ = apply_chunk_shrinkage(rate, weight, tables)
    metrics = chunk_shadow_metrics(gram, rate, weight, partition, shrunk)
    for key in (
        "mp_opd_grass_chunk_shadow_hard_chunk_head_cosine_to_grass",
        "mp_opd_grass_chunk_shadow_atomic_head_cosine_to_grass",
        "mp_opd_grass_chunk_shadow_hard_chunk_credit_l2_change",
    ):
        assert key in metrics
        assert torch.isfinite(metrics[key]).all()


def test_hard_chunk_credits_match_a_python_loop():
    n = 7
    rate, weight = _credits(n, seed=68)
    partition = ((0, 3), (3, 4), (4, 7))
    pooled = hard_chunk_credits(rate, weight, partition)
    expected = torch.zeros(n, dtype=torch.float64)
    for start, end in partition:
        expected[start:end] = (rate[start:end] * weight[start:end]).sum() / weight[
            start:end
        ].sum()
    assert torch.allclose(pooled, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# The absences: this is what must never regress
# ---------------------------------------------------------------------------


def test_the_chunk_module_contains_no_search_or_dynamic_program():
    source = (ROOT / "kdflow/algorithms/_mp_opd_grass_chunk.py").read_text(
        encoding="utf-8"
    )
    tree = _ast.parse(source)
    names = {
        node.id for node in _ast.walk(tree) if isinstance(node, _ast.Name)
    } | {
        node.attr for node in _ast.walk(tree) if isinstance(node, _ast.Attribute)
    }
    for banned in ("grass_partition", "grass_span_costs", "gbv_span_costs"):
        assert banned not in names, f"{banned} reintroduces a search into grass_chunk"
    # No cost table at all: there is nothing to minimise.
    assert "costs" not in {name.id for name in _ast.walk(tree) if isinstance(name, _ast.Name)}


def test_the_sure_gain_never_feeds_back_into_the_update():
    """Section 10: the gain is logged; it must not move a boundary."""
    tree = _ast.parse(
        (ROOT / "kdflow/algorithms/_mp_opd_grass_chunk.py").read_text(encoding="utf-8")
    )
    functions = {
        node.name: node for node in _ast.walk(tree) if isinstance(node, _ast.FunctionDef)
    }

    def identifiers(node):
        # Prose is checked as names, not as text: 'against' contains 'gain'.
        return {
            child.id for child in _ast.walk(node) if isinstance(child, _ast.Name)
        } | {
            child.attr for child in _ast.walk(node) if isinstance(child, _ast.Attribute)
        }

    # The only function that decides the update reads the strengths, never the gain.
    shrink = functions["apply_chunk_shrinkage"]
    assert "gain" not in identifiers(shrink)
    assert "chunk_alpha_table" in _ast.unparse(shrink)
    assert "grass_shrink" in _ast.unparse(shrink)
    # And nothing else in the module orders, filters or reshapes chunks by the gain.
    # Only the table that records it and the telemetry that reports it may read it.
    readers = {
        name
        for name, node in functions.items()
        if "gain" in identifiers(node)
    }
    assert readers <= {"grass_chunk_tables", "grass_chunk_metrics"}, readers


def _field_specs(source: str):
    """``{name: (default, choices)}`` for every ``field(...)`` declaration."""
    specs = {}
    for node in _ast.walk(_ast.parse(source)):
        if not isinstance(node, _ast.AnnAssign) or node.value is None:
            continue
        if not (
            isinstance(node.value, _ast.Call)
            and isinstance(node.value.func, _ast.Name)
            and node.value.func.id == "field"
        ):
            continue
        default = choices = None
        for keyword in node.value.keywords:
            try:
                if keyword.arg == "default":
                    default = _ast.literal_eval(keyword.value)
                elif keyword.arg == "metadata":
                    choices = _ast.literal_eval(keyword.value).get("choices")
            except (ValueError, SyntaxError):
                pass
        if isinstance(node.target, _ast.Name):
            specs[node.target.id] = (default, choices)
    return specs


def test_args_declare_the_chunk_knobs_and_fail_closed():
    source = (ROOT / "kdflow/arguments/distillation_args.py").read_text(
        encoding="utf-8"
    )
    specs = _field_specs(source)
    assert specs["mp_opd_grass_chunk_source"] == ("xtoken", ["xtoken", "run"])
    assert specs["mp_opd_grass_chunk_straddle"] == (
        "singleton",
        ["singleton", "majority"],
    )
    assert specs["mp_opd_grass_chunk_run_length"] == (2, None)
    assert specs["mp_opd_grass_chunk_shadow"] == (False, None)
    assert "grass_chunk" in specs["mp_opd_mode"][1]

    guards = [
        _ast.unparse(node)
        for node in _ast.walk(_ast.parse(source))
        if isinstance(node, _ast.If) and "raise ValueError" in _ast.unparse(node)
    ]
    # The native chunk needs the audited projection; a run must not fall back to
    # fixed runs while claiming to use alignment chunks.
    assert any(
        "xtoken_projection_path" in guard and "grass_chunk" in guard for guard in guards
    )
    assert any(
        "grass_chunk" in guard and "mp_opd_credit_transform" in guard for guard in guards
    )


def test_every_advertised_mode_is_actually_accepted():
    """The `choices` metadata and the runtime guard are two separate lists.

    They drifted before: `kernel` was accepted but never advertised, and a mode
    added to `choices` alone produces a run that dies at argument validation with
    a message about a mode the config said was valid. One list, one source of
    truth is not available here, so the invariant is asserted instead.
    """
    source = (ROOT / "kdflow/arguments/distillation_args.py").read_text(
        encoding="utf-8"
    )
    advertised = set(_field_specs(source)["mp_opd_mode"][1])
    accepted = set()
    for node in _ast.walk(_ast.parse(source)):
        if not isinstance(node, _ast.If) or isinstance(node.test, _ast.UnaryOp):
            continue
        text = _ast.unparse(node.test)
        if "mp_opd_mode not in" not in text:
            continue
        for literal in _ast.walk(node.test):
            if isinstance(literal, _ast.Set):
                accepted |= {element.value for element in literal.elts}
    assert advertised, "could not read the advertised mode list"
    assert accepted, "could not read the accepted mode list"
    assert advertised <= accepted, advertised - accepted
    # Everything mp_opd dispatches must also be advertised to the user.
    mp_opd_source = (ROOT / "kdflow/algorithms/mp_opd.py").read_text(encoding="utf-8")
    dispatched = set(re.findall(r"self\.mode == \"([a-z_]+)\"", mp_opd_source))
    # `atomic` and `kernel` share one branch, so `==` alone under-counts.
    for group in re.findall(r"self\.mode in \{([^}]*)\}", mp_opd_source):
        dispatched |= set(re.findall(r"\"([a-z_]+)\"", group))
    assert "atomic" in dispatched and "kernel" in dispatched
    assert dispatched <= advertised, dispatched - advertised


def test_mode_dispatch_reaches_both_grass_modes():
    source = (ROOT / "kdflow/algorithms/mp_opd.py").read_text(encoding="utf-8")
    body = _ast.unparse(_ast.parse(source))
    assert "if self.mode == 'grass_chunk':" in body
    assert "self._grass_chunk_loss(" in body
    assert "_GRASS_MODES = frozenset" in body