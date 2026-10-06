"""GRASS: exact head Gram, SURE shrinkage strength, DP, and credit conservation.

The method note makes five claims that a CPU-only unit test can pin without a GPU:

* ``H_ij`` equals an explicit, per-token ``sum_t sum_s (delta_t^T delta_s)(h_t^T h_s)``
  for the output-head gradients, including the Gemma-2 final-logit-softcap factor;
* ``alpha* = clip(V_c / D_c, 0, 1)`` is what the tables return, with the documented
  degenerate branches;
* the dynamic program equals brute force and never returns a worse total cost;
* the shrunk credits conserve ``sum_i w_i r_i`` exactly, and ``alpha = 1`` reproduces
  the hard pooled rate of GBV/Fixed-k;
* the four limiting behaviours of section 18.4 hold: a vanishing noise scale sends
  ``alpha`` to 0, a constant-credit span with real noise pools, the diagonal ablation
  is recovered by zeroing the off-diagonal band, and singletons are free.
"""
import ast as _ast
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kdflow.algorithms._mp_opd_grass_span import (
    CREDIT_CONSERVATION_REL_TOL,
    GrassNoiseEstimator,
    atom_head_gram,
    grass_candidate_metrics,
    grass_cosine,
    grass_credit_residual,
    grass_gram_diagnostics,
    grass_partition,
    grass_partition_metrics,
    grass_span_costs,
    grass_span_ids,
    grass_shrink,
    grass_update_energy,
    hard_pooled_credits,
)

ARGS = ROOT / "kdflow/arguments/distillation_args.py"


def _credits(n: int, seed: int = 3, spread: float = 1.0):
    generator = torch.Generator().manual_seed(seed)
    rate = spread * torch.randn(n, generator=generator, dtype=torch.float64)
    weight = torch.randint(1, 5, (n,), generator=generator).double()
    return rate, weight


def _brute_force_gram(logits, hidden, labels, atom_ranges, softcap=None, token_weight=None):
    """Reference atom Gram: build every head gradient explicitly, then inner-product."""
    rows = logits.to(torch.float64)
    log_partition = torch.logsumexp(rows, dim=-1)
    probabilities = torch.exp(rows - log_partition.unsqueeze(-1))
    if softcap is not None:
        normalized = (rows / float(softcap)).clamp(-1.0, 1.0)
        probabilities = probabilities * (1.0 - normalized * normalized)
    if token_weight is not None:
        probabilities = probabilities * token_weight.to(torch.float64).unsqueeze(-1)
    delta = probabilities.clone()
    delta[torch.arange(labels.numel()), labels.long()] -= 1.0
    grads = [
        torch.stack(
            [
                (delta[start:end].unsqueeze(-1) * hidden[start:end].to(torch.float64).unsqueeze(1))
                .sum(dim=0)
            ]
        ).reshape(-1)
        for start, end in atom_ranges
    ]
    return torch.stack([torch.stack([g @ other for other in grads]) for g in grads])


def _enforce_psd(gram: torch.Tensor, floor: float = 1e-6) -> torch.Tensor:
    gram = 0.5 * (gram + gram.T)
    values, vectors = torch.linalg.eigh(gram)
    return vectors @ torch.diag(values.clamp_min(floor)) @ vectors.T


@pytest.mark.parametrize(
    "atom_ranges,softcap",
    [
        (((0, 1), (1, 2), (2, 3)), None),
        (((0, 1), (1, 2), (2, 3)), 30.0),
        (((0, 2), (2, 4), (4, 5)), None),
        (((0, 3), (3, 5), (5, 6), (6, 8)), None),
    ],
)
def test_atom_head_gram_matches_explicit_head_gradient(atom_ranges, softcap):
    """Section 6: the banded Gram is the exact output-head gradient inner product."""
    torch.manual_seed(11)
    tokens, vocab, hidden_size = 8, 23, 5
    logits = torch.randn(tokens, vocab, dtype=torch.float64)
    hidden = torch.randn(tokens, hidden_size, dtype=torch.float64)
    labels = torch.tensor([0, 3, 7, 7, 12, 19, 22, 1])
    covered = atom_ranges[-1][1]
    result = atom_head_gram(
        logits,
        hidden,
        labels,
        atom_ranges,
        4,
        softcap=softcap,
        # The supplied route is validated against the atom-covered token axis, not
        # the full one: passing the whole sequence asks the Gram to resolve atoms
        # that tile only part of it.
        selected_log_prob=(
            logits[:covered].gather(1, labels[:covered].unsqueeze(1)).squeeze(1)
            - torch.logsumexp(logits[:covered], dim=-1)
        ),
    )
    expected = _brute_force_gram(logits[:covered], hidden[:covered], labels[:covered],
                                 atom_ranges, softcap)
    assert torch.allclose(result.gram, expected, atol=1e-10, rtol=1e-9)
    assert result.symmetry_error < 1e-12
    assert result.atom_count == len(atom_ranges)
    assert result.token_count == atom_ranges[-1][1]


@pytest.mark.parametrize(
    "atom_ranges",
    [
        ((0, 1), (1, 3), (3, 4), (4, 6)),
        ((0, 2), (2, 4), (4, 5), (5, 6)),
        ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)),
    ],
)
def test_weighted_gram_matches_the_explicit_head_gradient(atom_ranges):
    """Regression: unequal token weights across atoms broke the delta identity.

    The correction -v_s q_t[y_s] - v_t q_s[y_t] was paired with the wrong scale on
    each half. Equal weights cancel the mistake, so every unweighted case passed;
    only a weight that differs between two atoms exposes it. The reference below
    carries the same weights through explicit head gradients.
    """
    torch.manual_seed(19)
    tokens, vocab, hidden_size = 6, 17, 4
    logits = torch.randn(tokens, vocab, dtype=torch.float64)
    hidden = torch.randn(tokens, hidden_size, dtype=torch.float64)
    labels = torch.tensor([0, 5, 5, 11, 16, 2])
    weight = torch.tensor([1.0, 2.0, 2.0, 0.5, 1.5, 3.0], dtype=torch.float64)
    result = atom_head_gram(logits, hidden, labels, atom_ranges, 4, token_weight=weight)
    expected = _brute_force_gram(logits, hidden, labels, atom_ranges, None, weight)
    # The routine accumulates in float32, so the reference is matched at fp32
    # resolution rather than at the float64 exactness of the inputs.
    tolerance = max(float(expected.abs().max()), 1.0) * 1e-5
    assert torch.allclose(result.gram.to(torch.float64), expected, atol=tolerance)
    assert result.symmetry_error < 1e-6


def test_row_chunking_agrees_with_a_single_pass_under_unequal_weights():
    """Regression: chunked rows read the window position instead of the row index."""
    torch.manual_seed(20)
    tokens, vocab, hidden_size = 9, 13, 4
    logits = torch.randn(tokens, vocab, dtype=torch.float64)
    hidden = torch.randn(tokens, hidden_size, dtype=torch.float64)
    labels = torch.tensor([0, 1, 2, 3, 12, 4, 5, 6, 7])
    weight = torch.tensor([1.0, 2.0, 0.5, 1.0, 3.0, 1.0, 2.0, 1.0, 4.0], dtype=torch.float64)
    # Multi-token atoms make the row/window offset real rather than hypothetical.
    ranges = ((0, 1), (1, 3), (3, 4), (4, 6), (6, 9))
    reference = atom_head_gram(logits, hidden, labels, ranges, 3, token_weight=weight,
                               row_chunk_atoms=len(ranges))
    for chunk in (1, 2, 3):
        chunked = atom_head_gram(logits, hidden, labels, ranges, 3, token_weight=weight,
                                 row_chunk_atoms=chunk)
        assert torch.allclose(chunked.gram, reference.gram, atol=1e-6), chunk
        assert chunked.symmetry_error < 1e-5


def test_atom_head_gram_band_is_zero_outside_the_candidate_window():
    torch.manual_seed(12)
    tokens, vocab = 9, 17
    logits = torch.randn(tokens, vocab, dtype=torch.float64)
    hidden = torch.randn(tokens, 4, dtype=torch.float64)
    labels = torch.arange(tokens)
    ranges = tuple((index, index + 1) for index in range(tokens))
    result = atom_head_gram(logits, hidden, labels, ranges, 3)
    mask = (result.gram != 0).nonzero()
    for row, column in mask.tolist():
        assert abs(row - column) <= 2


def test_atom_head_gram_uses_the_supplied_selected_logprob_route():
    """The two probability routes must agree; MP-OPD always supplies the first."""
    torch.manual_seed(13)
    logits = torch.randn(6, 19, dtype=torch.float64)
    hidden = torch.randn(6, 3, dtype=torch.float64)
    labels = torch.tensor([1, 1, 2, 3, 4, 5])
    ranges = ((0, 2), (2, 3), (3, 6))
    probabilities = logits.softmax(dim=-1)
    supplied = atom_head_gram(
        logits,
        hidden,
        labels,
        ranges,
        3,
        selected_log_prob=probabilities.gather(1, labels.unsqueeze(1)).squeeze(1).log(),
    )
    recovered = atom_head_gram(logits, hidden, labels, ranges, 3, vocab_chunk=4)
    assert torch.allclose(supplied.gram, recovered.gram, atol=1e-12)


def test_diagonal_geometry_is_the_section_15_ablation():
    torch.manual_seed(14)
    logits = torch.randn(8, 13, dtype=torch.float64)
    hidden = torch.randn(8, 4, dtype=torch.float64)
    labels = torch.arange(8)
    ranges = tuple((index, index + 1) for index in range(8))
    full = atom_head_gram(logits, hidden, labels, ranges, 4)
    diagonal = atom_head_gram(logits, hidden, labels, ranges, 4, diagonal_only=True)
    assert torch.allclose(
        diagonal.gram, torch.diag(torch.diagonal(full.gram)), atol=1e-12
    )
    assert bool((diagonal.gram == torch.diag(torch.diagonal(diagonal.gram))).all())


def test_token_weight_scales_the_gram_exactly():
    """``m_t`` is a per-token scalar, so it must enter as an exact Gram rescaling."""
    torch.manual_seed(15)
    logits = torch.randn(6, 11, dtype=torch.float64)
    hidden = torch.randn(6, 3, dtype=torch.float64)
    labels = torch.arange(6)
    ranges = ((0, 1), (1, 3), (3, 4), (4, 6))
    base = atom_head_gram(logits, hidden, labels, ranges, 4)
    # Atom 1 owns tokens 1 and 2; scaling both by 2 scales that block by exactly 4
    # and leaves the single-token atoms untouched.
    weight = torch.tensor([1.0, 2.0, 2.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    scaled = atom_head_gram(logits, hidden, labels, ranges, 4, token_weight=weight)
    assert float(scaled.gram[1, 1]) == pytest.approx(float(base.gram[1, 1]) * 4.0, rel=1e-10)
    assert float(scaled.gram[0, 0]) == pytest.approx(float(base.gram[0, 0]), rel=1e-10)
    assert float(scaled.gram[3, 3]) == pytest.approx(float(base.gram[3, 3]), rel=1e-10)
    # One of the two atoms is scaled, so the cross block is scaled by 2.
    assert float(scaled.gram[0, 1]) == pytest.approx(float(base.gram[0, 1]) * 2.0, rel=1e-10)


def test_softcap_factor_is_the_actual_head_jacobian():
    """Without the softcap factor the Gram would describe a head the model lacks."""
    torch.manual_seed(16)
    logits = torch.randn(5, 9, dtype=torch.float64)
    hidden = torch.randn(5, 3, dtype=torch.float64)
    labels = torch.arange(5)
    ranges = tuple((index, index + 1) for index in range(5))
    probabilities = logits.softmax(dim=-1)
    plain = atom_head_gram(logits, hidden, labels, ranges, 4)
    capped = atom_head_gram(logits, hidden, labels, ranges, 4, softcap=30.0)
    jacobian = 1.0 - (logits / 30.0) ** 2
    # ||delta_t * s_t||^2 = sum_v p^2 s^2 - 2 p[y] s[y]^2 + s[y]^2
    expected = (
        (probabilities**2 * jacobian**2).sum(dim=-1)
        - 2.0 * probabilities.gather(1, labels.unsqueeze(1)).squeeze(1) * jacobian.gather(1, labels.unsqueeze(1)).squeeze(1) ** 2
        + jacobian.gather(1, labels.unsqueeze(1)).squeeze(1) ** 2
    )
    expected = expected * (hidden**2).sum(dim=-1)
    # atom_head_gram accumulates in float32 on purpose; the reference is float64.
    diagonal = torch.diagonal(capped.gram).to(torch.float64)
    assert torch.allclose(diagonal, expected, atol=float(expected.abs().max()) * 1e-5)
    # An unsquashed proxy would report ||p_t||^2 instead. It still has to carry the
    # label half - ||delta_t||^2 = sum_v p^2 - 2 p[y] + 1 - or it is not the quantity
    # the Gram diagonal reduces to, and the comparison below is meaningless.
    unsquashed = (
        (probabilities**2).sum(dim=-1)
        - 2.0 * probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
        + 1.0
    ) * (hidden**2).sum(dim=-1)
    plain_diagonal = torch.diagonal(plain.gram).to(torch.float64)
    assert not torch.allclose(plain_diagonal, expected, atol=1e-6)
    assert torch.allclose(plain_diagonal, unsquashed, atol=float(unsquashed.abs().max()) * 1e-4)


def test_gram_survives_bf16_inputs_and_stays_detached():
    torch.manual_seed(17)
    logits = torch.randn(7, 64, dtype=torch.bfloat16, requires_grad=True)
    hidden = torch.randn(7, 6, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.arange(7)
    ranges = ((0, 2), (2, 4), (4, 7))
    result = atom_head_gram(logits, hidden, labels, ranges, 3)
    assert not result.gram.requires_grad
    assert torch.isfinite(result.gram).all()


def test_gram_rejects_gaps_and_overlaps():
    logits = torch.randn(6, 11, dtype=torch.float64)
    hidden = torch.randn(6, 3, dtype=torch.float64)
    labels = torch.arange(6)
    with pytest.raises(ValueError):
        atom_head_gram(logits, hidden, labels, ((0, 2), (3, 6)), 3)
    with pytest.raises(ValueError):
        atom_head_gram(logits, hidden, labels, ((0, 3), (2, 6)), 3)
    with pytest.raises(ValueError):
        atom_head_gram(logits, hidden, labels, ((0, 7),), 3)
    with pytest.raises(ValueError):
        atom_head_gram(logits, hidden, labels, (), 3)


def test_row_chunking_does_not_change_the_gram():
    """The chunk knob changes how ``H`` is computed, never which ``H``."""
    torch.manual_seed(18)
    logits = torch.randn(12, 31, dtype=torch.float64)
    hidden = torch.randn(12, 5, dtype=torch.float64)
    labels = torch.arange(12)
    ranges = tuple((index, index + 1) for index in range(12))
    reference = atom_head_gram(logits, hidden, labels, ranges, 4, row_chunk_atoms=12)
    for chunk in (1, 3, 5):
        chunked = atom_head_gram(logits, hidden, labels, ranges, 4, row_chunk_atoms=chunk)
        assert torch.allclose(chunked.gram, reference.gram, atol=1e-12)


def test_span_costs_match_the_closed_form_of_the_method_note():
    n, max_span, sigma2 = 7, 4, 0.3
    rate, weight = _credits(n, seed=21)
    generator = torch.Generator().manual_seed(n)
    gram = _enforce_psd(
        torch.randn(n, n, generator=generator, dtype=torch.float64) * 0.4
        @ (torch.randn(n, n, generator=generator, dtype=torch.float64) * 0.4).T
        + torch.eye(n, dtype=torch.float64)
    )
    tables = grass_span_costs(rate, weight, gram, sigma2, max_span)
    for start in range(n):
        for end in range(start + 1, min(n, start + max_span) + 1):
            index = end - start - 1
            span_rate = rate[start:end]
            span_weight = weight[start:end]
            pooled = (span_rate * span_weight).sum() / span_weight.sum()
            deviation = span_rate - pooled
            block = gram[start:end, start:end]
            expected_d = deviation @ block @ deviation
            expected_v = sigma2 * (
                (torch.diagonal(block) / span_weight).sum()
                - block.sum() / span_weight.sum()
            )
            expected_alpha = min(max(float(expected_v / expected_d), 0.0), 1.0)
            assert float(tables.distortion[start, index]) == pytest.approx(
                float(expected_d), rel=1e-10, abs=1e-14
            )
            assert float(tables.variance[start, index]) == pytest.approx(
                float(expected_v), rel=1e-10, abs=1e-14
            )
            assert float(tables.alpha[start, index]) == pytest.approx(
                expected_alpha, rel=1e-9, abs=1e-12
            )
            assert float(tables.costs[start, index]) == pytest.approx(
                expected_alpha**2 * float(expected_d) - 2.0 * expected_alpha * float(expected_v),
                rel=1e-9,
                abs=1e-12,
            )


def test_singletons_are_free_and_have_zero_strength():
    n = 6
    rate, weight = _credits(n, seed=22)
    gram = torch.eye(n, dtype=torch.float64) * 2.0
    tables = grass_span_costs(rate, weight, gram, 0.5, 4)
    assert torch.allclose(tables.costs[:, 0], torch.zeros(n, dtype=torch.float64))
    assert torch.allclose(tables.alpha[:, 0], torch.zeros(n, dtype=torch.float64))


def test_noise_free_estimate_sends_every_strength_to_zero():
    """Section 18.4: ``sigma^2 -> 0`` implies ``alpha -> 0``."""
    n = 8
    rate, weight = _credits(n, seed=23, spread=4.0)
    gram = _enforce_psd(torch.randn(n, n, dtype=torch.float64).abs() + torch.eye(n, dtype=torch.float64))
    tables = grass_span_costs(rate, weight, gram, 0.0, 4)
    assert torch.allclose(tables.alpha, torch.zeros_like(tables.alpha))
    partition, cost, _margins = grass_partition(tables)
    assert partition == tuple((index, index + 1) for index in range(n))
    assert float(cost) == 0.0
    assert tables.degenerate


def test_constant_credit_with_real_noise_pools_fully():
    """Section 18.4: constant ``r`` inside a span plus noise implies strong pooling."""
    n = 6
    rate = torch.full((n,), 0.3, dtype=torch.float64)
    weight = torch.ones(n, dtype=torch.float64)
    gram = _enforce_psd(torch.randn(n, n, dtype=torch.float64).abs() + torch.eye(n, dtype=torch.float64))
    tables = grass_span_costs(rate, weight, gram, 0.2, 4)
    # Slice alpha to the candidate widths first: masking the full [n, length] table
    # with a [n, length - 1] mask indexes the wrong axis.
    multi = tables.valid[:, 1:]
    assert float(tables.alpha[:, 1:][multi].min()) == pytest.approx(1.0)
    shrunk, residual = grass_shrink(rate, weight, grass_partition(tables)[0], tables.alpha)
    assert float(residual) < 1e-12
    assert float(shrunk.max() - shrunk.min()) < 1e-12


def test_hard_pooling_limit_reproduces_the_pooled_rate():
    """Section 15: forcing ``alpha = 1`` must reproduce hard broadcast pooling."""
    n = 9
    rate, weight = _credits(n, seed=24)
    partition = ((0, 2), (2, 5), (5, 6), (6, 9))
    alpha = torch.zeros((n, 4), dtype=torch.float64)
    for start, end in partition:
        alpha[start, end - start - 1] = 1.0
    shrunk, residual = grass_shrink(rate, weight, partition, alpha)
    expected = hard_pooled_credits(rate * weight, weight, partition)
    assert torch.allclose(shrunk, expected, atol=1e-12)
    assert float(residual) < 1e-12


@pytest.mark.parametrize("n,max_span,seed", [(6, 2, 31), (7, 4, 32), (10, 3, 33)])
def test_dynamic_program_equals_brute_force(n, max_span, seed):
    from kdflow.algorithms._mp_opd_oracle import enumerate_partitions, partition_score

    rate, weight = _credits(n, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    gram = _enforce_psd(
        torch.randn(n, n, generator=generator, dtype=torch.float64)
        @ torch.randn(n, n, generator=generator, dtype=torch.float64).T
        + torch.eye(n, dtype=torch.float64)
    )
    tables = grass_span_costs(rate, weight, gram, 0.25, max_span)
    partition, cost, _margins = grass_partition(tables)
    brute = min(
        enumerate_partitions(n, max_span), key=lambda part: float(partition_score(tables.costs, part))
    )
    assert float(cost) == pytest.approx(
        float(partition_score(tables.costs, brute)), rel=1e-12
    )
    cursor = 0
    for start, end in partition:
        assert start == cursor
        cursor = end
    assert cursor == n


def test_margin_is_best_minus_second_best():
    n = 6
    rate, weight = _credits(n, seed=34)
    gram = torch.eye(n, dtype=torch.float64)
    tables = grass_span_costs(rate, weight, gram, 0.1, 3)
    partition, cost, margins = grass_partition(tables)
    assert margins.shape == (n + 1,)
    assert float(margins[0]) == 0.0
    # Positions with a single admissible predecessor have no margin to report;
    # they stay +inf so the near-tie fraction is not inflated by construction.
    assert float(margins[1]) == float("inf")
    # The margin is best - second best, so it is non-positive by construction: a
    # negative value is the ordinary case, not a sign failure, and a zero one is the
    # exact tie the metric is looking for. Asserting >= 0 asked for something the
    # definition cannot produce.
    decidable = margins[torch.isfinite(margins)]
    assert bool((decidable <= 0.0).all())
    assert float(decidable.min()) < 0.0
    # Whatever the margin says, the reported cost must still be the chosen path's.
    chosen = float(sum(float(tables.costs[start, end - start - 1]) for start, end in partition))
    assert float(cost) == pytest.approx(chosen, rel=1e-12)


@pytest.mark.parametrize("n,seed", [(8, 41), (13, 42), (21, 43)])
def test_shrinkage_conserves_weighted_credit_exactly(n, seed):
    rate, weight = _credits(n, seed=seed, spread=3.0)
    generator = torch.Generator().manual_seed(seed)
    gram = _enforce_psd(
        torch.randn(n, n, generator=generator, dtype=torch.float64)
        @ torch.randn(n, n, generator=generator, dtype=torch.float64).T
        + torch.eye(n, dtype=torch.float64)
    )
    tables = grass_span_costs(rate, weight, gram, 0.4, 4)
    partition, _cost, margins = grass_partition(tables)
    shrunk, residual = grass_shrink(rate, weight, partition, tables.alpha)
    assert float(residual) == pytest.approx(0.0, abs=1e-12)
    assert float((weight * shrunk).sum() - (weight * rate).sum()) == pytest.approx(0.0, abs=1e-12)
    assert float(grass_credit_residual(rate, weight, shrunk)) < 1e-12
    metrics = grass_partition_metrics(
        tables,
        partition,
        rate,
        weight,
        gram,
        shrunk,
        margins,
        4,
        conservation_error=residual,
    )
    assert float(metrics["mp_opd_grass_credit_conservation_error"]) < 1e-12


def test_float32_rounding_still_satisfies_the_guard():
    """The loss consumes fp32 rates, so the guard is stated against that budget."""
    n = 32
    rate, weight = _credits(n, seed=44, spread=50.0)
    gram = _enforce_psd(torch.randn(n, n, dtype=torch.float64).abs() + torch.eye(n, dtype=torch.float64))
    tables = grass_span_costs(rate, weight, gram, 0.4, 4)
    partition, _cost, margins = grass_partition(tables)
    shrunk, _residual = grass_shrink(rate, weight, partition, tables.alpha)
    residual = grass_credit_residual(
        rate.float(), weight.float(), shrunk.to(torch.float32)
    )
    budget = float((weight.float() * rate.float()).sum().abs())
    assert float(residual) <= CREDIT_CONSERVATION_REL_TOL * max(budget, 1.0)


def test_loss_gradient_is_the_shrunk_credit_rate():
    """The update GRASS produces is ``sum_i r~_i NLL_i`` and nothing else."""
    from kdflow.algorithms._mp_opd_credit import soft_partition_loss

    n = 10
    rate, weight = _credits(n, seed=45)
    gram = _enforce_psd(torch.randn(n, n, dtype=torch.float64).abs() + torch.eye(n, dtype=torch.float64))
    tables = grass_span_costs(rate, weight, gram, 0.3, 3)
    partition, _cost, margins = grass_partition(tables)
    shrunk, _residual = grass_shrink(rate, weight, partition, tables.alpha)
    nll = torch.randn(n, dtype=torch.float32, requires_grad=True)
    loss = soft_partition_loss(nll, shrunk.to(torch.float32))
    gradient = torch.autograd.grad(loss, nll)[0]
    assert torch.allclose(gradient, shrunk.to(torch.float32), atol=1e-6)


def test_noise_estimator_recovers_a_planted_scale_and_is_robust_to_outliers():
    estimator = GrassNoiseEstimator(rho=0.9, min_adjacent_pairs=8)
    generator = torch.Generator().manual_seed(51)
    n, sigma, weight_value = 400, 0.05, 4.0
    weight = torch.full((n,), weight_value, dtype=torch.float64)
    latent = torch.full((n,), 0.5, dtype=torch.float64)
    # The working model is Var(epsilon_i) = sigma^2 / w_i, so the standardised
    # differences have unit scale in sigma by construction.
    noise = sigma * torch.randn(n, generator=generator, dtype=torch.float64) / weight.sqrt()
    # Genuine credit outliers must not move the median-based scale.
    noise[0] = 40.0
    noise[1] = -25.0
    rate = latent + noise
    for _ in range(40):
        estimator.update(rate, weight)
    assert estimator.sigma2 == pytest.approx(sigma**2, rel=0.25)
    assert estimator.sigma2 > 0.0


def test_noise_estimator_keeps_the_previous_scale_when_pairs_are_too_few():
    estimator = GrassNoiseEstimator(rho=0.9, min_adjacent_pairs=8)
    rate = torch.tensor([1.0, 1.4, 1.2], dtype=torch.float64)
    weight = torch.ones(3, dtype=torch.float64)
    assert estimator.sigma2 == 0.0
    estimator.update(rate, weight)
    assert estimator.sigma2 == 0.0
    diagnostics = estimator.observe(rate, weight)
    assert diagnostics["valid_pairs"] == 2.0


def test_noise_estimator_state_round_trips():
    estimator = GrassNoiseEstimator(rho=0.95, min_adjacent_pairs=4)
    rate, weight = _credits(64, seed=61)
    for _ in range(5):
        estimator.update(rate, weight)
    restored = GrassNoiseEstimator(rho=0.95, min_adjacent_pairs=4)
    restored.load_state_dict(estimator.state_dict())
    assert restored.sigma2 == pytest.approx(estimator.sigma2, rel=1e-12)
    with pytest.raises(ValueError):
        GrassNoiseEstimator(rho=0.5).load_state_dict(estimator.state_dict())


def test_negative_statistics_are_reported_not_hidden():
    """An indefinite Gram must surface through the pathology counters.

    The credits are chosen so the sign is provable rather than sampled: with equal
    weights, ``r = (1, -1, 0, 0, 0)`` makes the (0, 2) span deviate by
    ``(+1/2, -1/2)``, and against ``H = [[1, 3], [3, 1]]`` that block gives
    ``1/4 - 3/2 + 1/4 = -1``, an order of magnitude below any plausible tolerance.
    """
    n = 5
    rate = torch.tensor([1.0, -1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    weight = torch.ones(n, dtype=torch.float64)
    # Strongly positive off-diagonal with a small diagonal is the shape that makes
    # tr(H A Sigma) negative while D stays positive - the exact failure the
    # counters exist to expose.
    indefinite = torch.eye(n, dtype=torch.float64)
    indefinite[0, 1] = indefinite[1, 0] = 3.0
    tables = grass_span_costs(rate, weight, indefinite, 1.0, 3)
    assert float(tables.distortion[0, 1]) == pytest.approx(-1.0, rel=1e-12)
    metrics = grass_candidate_metrics(tables)
    assert float(metrics["mp_opd_grass_pathology_negative_d_fraction"]) > 0.0
    assert float(metrics["mp_opd_grass_pathology_negative_v_fraction"]) > 0.0
    assert tables.negative_tol >= 0.0
    # The clamping to zero happens on the value used for alpha, never on the value
    # the counter reads: a negative D must still be visible in the reported table.
    assert float(tables.distortion[0, 1]) < 0.0


def test_candidate_and_gram_metrics_are_detached_and_finite():
    n = 11
    rate, weight = _credits(n, seed=81)
    gram = _enforce_psd(torch.randn(n, n, dtype=torch.float64).abs() + torch.eye(n, dtype=torch.float64))
    tables = grass_span_costs(rate, weight, gram, 0.2, 4)
    partition, _cost, margins = grass_partition(tables)
    shrunk, residual = grass_shrink(rate, weight, partition, tables.alpha)
    candidates = grass_candidate_metrics(tables, partition)
    selected = grass_partition_metrics(
        tables, partition, rate, weight, gram, shrunk, margins, 4, conservation_error=residual
    )
    gram_metrics = grass_gram_diagnostics(gram, 4)
    everything = {**candidates, **selected, **gram_metrics}
    assert everything
    assert all(isinstance(value, torch.Tensor) for value in everything.values())
    assert all(not value.requires_grad for value in everything.values())
    assert all(bool(torch.isfinite(value).all()) for value in everything.values())
    for key in (
        "mp_opd_grass_total_cost",
        "mp_opd_grass_alpha_mean",
        "mp_opd_grass_sigma2",
        "mp_opd_grass_head_update_cosine",
        "mp_opd_grass_gram_offdiagonal_negative_fraction",
    ):
        assert key in everything
    assert float(everything["mp_opd_grass_gram_max_span_negative_fraction"]) == 0.0


def test_head_update_energy_masks_pairs_outside_the_span():
    """Head-update comparisons must count only pairs the selector actually merged."""
    n = 6
    rate, _weight = _credits(n, seed=91)
    generator = torch.Generator().manual_seed(91)
    gram = _enforce_psd(torch.randn(n, n, generator=generator, dtype=torch.float64))
    atomic_ids = grass_span_ids(tuple((i, i + 1) for i in range(n)), n, rate.device)
    coarse_ids = grass_span_ids(((0, n),), n, rate.device)
    atomic = grass_update_energy(gram, atomic_ids, rate)
    assert float(atomic) == pytest.approx(
        float((torch.diagonal(gram) * rate * rate).sum()), rel=1e-12
    )
    coarse = grass_update_energy(gram, coarse_ids, rate)
    assert float(coarse) == pytest.approx(float(rate @ gram @ rate), rel=1e-12)
    assert float(grass_update_energy(gram, coarse_ids, rate, rate)) == pytest.approx(
        float(coarse), rel=1e-12
    )
    # A different span choice is a genuinely different head-space update: it drops
    # every cross-span pair. Whether that raises or lowers the energy depends on the
    # sign of those terms, so the claim under test is the masked value itself, not an
    # ordering the quadratic form does not guarantee.
    partial_ids = grass_span_ids(((0, 3), (3, 6)), n, rate.device)
    partial = grass_update_energy(gram, partial_ids, rate)
    blocked = 0.0
    for start, end in ((0, 3), (3, 6)):
        piece = rate[start:end]
        blocked += float(piece @ gram[start:end, start:end] @ piece)
    assert float(partial) == pytest.approx(blocked, rel=1e-12)
    # Dropping the cross-span pairs is the whole point: the masked value differs from
    # the unmasked one unless those pairs happen to contribute nothing.
    assert float(partial) != pytest.approx(float(coarse), rel=1e-6)
    cross = grass_update_energy(gram, partial_ids, rate, rate * 0.5)
    assert float(grass_cosine(cross, partial, partial * 0.25)) == pytest.approx(
        1.0, rel=1e-10
    )


def test_narrow_span_budgets_report_only_the_buckets_that_exist():
    """`run_single_gpu.py` defaults max_span_length to 2, and short responses have
    fewer atoms than the bucket count, so the telemetry must shrink with the table."""
    for n, span in ((5, 3), (3, 4), (2, 4), (6, 1)):
        rate, weight = _credits(n, seed=131)
        gram = _enforce_psd(
            torch.randn(n, n, generator=torch.Generator().manual_seed(n * 10 + span),
                        dtype=torch.float64)
        )
        tables = grass_span_costs(rate, weight, gram, 0.5, span)
        partition, _cost, _margins = grass_partition(tables)
        candidates = grass_candidate_metrics(tables, partition)
        for reported in (1, 2, 3, 4):
            key = f"mp_opd_grass_candidate_span_{reported}_count"
            if reported <= min(span, n):
                assert key in candidates, f"missing bucket {reported} for n={n}, span={span}"
            else:
                assert key not in candidates
        selected = grass_partition_metrics(
            tables, partition, rate, weight, gram, rate, torch.zeros(n + 1, dtype=torch.float64),
            span, conservation_error=torch.zeros((), dtype=torch.float64),
        )
        for reported in (1, 2, 3, 4):
            # The partition histogram is over observed lengths, so it always reports.
            assert f"mp_opd_grass_span_{reported}_fraction" in selected


def test_grass_mode_rejects_a_cross_atom_credit_operator():
    """A silent `A = K r` drop would produce a run that is not the recipe's."""
    source = (ROOT / "kdflow/arguments/distillation_args.py").read_text(encoding="utf-8")
    tree = _ast.parse(source)
    guards = []
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.If):
            continue
        rendered = _ast.unparse(node.test)
        if "grass" not in rendered or "credit_transform" not in rendered:
            continue
        body = _ast.unparse(node)
        if "raise ValueError" in body:
            guards.append(body)
    assert guards, "mp_opd_mode='grass' must be guarded against a credit operator"
    assert any("identity" in guard for guard in guards)


def test_hard_pooled_credits_match_a_python_loop():
    n = 7
    rate, weight = _credits(n, seed=101)
    partition = ((0, 3), (3, 4), (4, 7))
    pooled = hard_pooled_credits(rate * weight, weight, partition)
    expected = torch.zeros(n, dtype=torch.float64)
    for start, end in partition:
        expected[start:end] = (rate[start:end] * weight[start:end]).sum() / weight[start:end].sum()
    assert torch.allclose(pooled, expected, atol=1e-12)


def test_invalid_cost_inputs_raise():
    rate, weight = _credits(4, seed=111)
    gram = torch.eye(4, dtype=torch.float64)
    with pytest.raises(ValueError):
        grass_span_costs(rate, weight, torch.eye(3, dtype=torch.float64), 1.0, 2)
    with pytest.raises(ValueError):
        grass_span_costs(rate, weight, gram, -1.0, 2)
    with pytest.raises(ValueError):
        grass_span_costs(rate, torch.zeros(4, dtype=torch.float64), gram, 1.0, 2)
    with pytest.raises(ValueError):
        grass_span_costs(torch.zeros(0), torch.zeros(0), torch.zeros(0, 0), 1.0, 2)


def test_shrink_rejects_a_non_covering_partition_and_out_of_range_strength():
    rate, weight = _credits(6, seed=121)
    alpha = torch.zeros((6, 4), dtype=torch.float64)
    with pytest.raises(ValueError):
        grass_shrink(rate, weight, ((0, 2), (3, 6)), alpha)
    with pytest.raises(ValueError):
        grass_shrink(rate, weight, ((0, 2), (2, 5)), alpha)
    alpha[0, 1] = 1.5
    with pytest.raises(ValueError):
        grass_shrink(rate, weight, ((0, 2), (2, 6)), alpha)


def test_args_declare_the_grass_knobs_without_importing_transformers():
    source = ARGS.read_text()
    tree = _ast.parse(source)
    fields = {}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.AnnAssign) and getattr(node.target, "id", "").startswith("mp_opd_grass"):
            fields[node.target.id] = _ast.literal_eval(
                {kw.arg: kw.value for kw in node.value.keywords}["default"]
            )
    assert fields["mp_opd_grass_geometry"] == "exact_head"
    assert fields["mp_opd_grass_sigma_rho"] == 0.99
    assert fields["mp_opd_grass_sigma_min_pairs"] == 8
    assert fields["mp_opd_grass_eps_d"] == 0.0
    assert fields["mp_opd_grass_shadow"] is False
    assert '"grass"' in source
    assert "unsupported mp_opd_grass_geometry" in source
    assert "mp_opd_grass_sigma_rho must lie in [0, 1)" in source


def test_args_dataclass_accepts_grass_when_transformers_is_present():
    pytest.importorskip("transformers")
    from kdflow.arguments.distillation_args import DistillationArguments

    defaults = DistillationArguments.__dataclass_fields__
    assert defaults["mp_opd_grass_geometry"].default == "exact_head"
    assert defaults["mp_opd_grass_sigma_rho"].default == 0.99
    assert "grass" in str(defaults["mp_opd_mode"].metadata)