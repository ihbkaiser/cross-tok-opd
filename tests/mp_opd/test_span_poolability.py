"""Unit tests for the span poolability analyzer.

These are pure-stdlib: no torch, no GPU, so they run on the agent machine and
on the node. The central claim under test is that the spec's closed form
``V_within = (k - 1) * sigma2 / W_c`` equals the matrix trace it claims to
shortcut, **for unequal weights** as well as equal ones.
"""

import importlib.util
import math
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / 'scripts' / 'diagnostics' / 'analyze_span_poolability.py'
SPEC = importlib.util.spec_from_file_location('analyze_span_poolability', MODULE_PATH)
assert SPEC and SPEC.loader
analyzer = importlib.util.module_from_spec(SPEC)
# dataclasses resolves annotations through sys.modules, so register before exec.
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


WEIGHT_CASES = [
    [1.0, 1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0, 2.0],
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 1.0, 1.0, 3.0],
    [0.5, 1.5, 2.0, 4.0, 1.0],
]


@pytest.mark.parametrize('weights', WEIGHT_CASES)
def test_constant_sigma2_collapses_to_the_closed_form(weights):
    sigma2 = 0.37
    closed = analyzer.v_within(weights, [sigma2] * len(weights))
    matrix = analyzer.v_within_from_trace(weights, sigma2)
    assert math.isclose(closed, matrix, rel_tol=1e-12, abs_tol=1e-15)
    # And the closed form is exactly (k - 1) sigma2 / W_c.
    assert math.isclose(closed, (len(weights) - 1) * sigma2 / sum(weights),
                        rel_tol=1e-12, abs_tol=1e-15)


def test_v_within_is_monotone_in_each_sigma2_and_the_weights():
    weights = [1.0, 2.0, 3.0, 4.0]
    base = analyzer.v_within(weights, [1.0] * 4)
    bumped = analyzer.v_within(weights, [1.0, 1.0, 1.0, 5.0])
    assert bumped > base
    # Doubling every weight halves V_within: the (1/W_c) factor is a per-unit
    # normalisation, so more weight means less variance per unit of credit mass.
    heavier = analyzer.v_within([2.0 * v for v in weights], [1.0] * 4)
    assert heavier == pytest.approx(base / 2.0)


@pytest.mark.parametrize('weights', WEIGHT_CASES)
def test_weighted_centering_is_idempotent_and_weight_adjoint(weights):
    center, _ = analyzer.weighted_center_operator(weights)
    count = len(weights)
    # P = 1 w^T / W_c so C = I - P; C must stay unchanged when applied twice.
    for i in range(count):
        for j in range(count):
            once = center[i][j]
            twice = sum(center[i][m] * center[m][j] for m in range(count))
            assert math.isclose(once, twice, rel_tol=1e-12, abs_tol=1e-15)
    # tr(C) = k - 1 for every positive weight vector.
    assert math.isclose(analyzer.weighted_center_trace(weights), count - 1, abs_tol=1e-12)


def test_v_between_is_zero_for_constant_rates_and_positive_otherwise():
    weights = [1.0, 2.0, 3.0, 4.0]
    assert analyzer.v_between([2.5] * 4, weights) == pytest.approx(0.0, abs=1e-15)
    assert analyzer.v_between([1.0, 2.0, 3.0, 4.0], weights) > 0.0


def test_v_between_uses_the_weighted_mean_not_the_plain_mean():
    rates, weights = [0.0, 10.0], [1.0, 3.0]
    # weighted mean = 7.5, plain mean = 5.0
    expected = (1.0 * 56.25 + 3.0 * 6.25) / 4.0
    assert analyzer.v_between(rates, weights) == pytest.approx(expected, rel=1e-12)


def test_shrinking_rate_spread_raises_poolability():
    weights = [1.0, 2.0, 3.0, 4.0]
    tight = analyzer.v_between([0.1, 0.11, 0.09, 0.1], weights)
    loose = analyzer.v_between([0.0, 1.0, 2.0, 3.0], weights)
    noise = analyzer.v_within(weights, [1.0] * 4)
    assert analyzer.poolability(tight, noise, 1e-12) > analyzer.poolability(loose, noise, 1e-12)


def test_poolability_is_clipped_to_the_unit_interval():
    assert analyzer.poolability(1e-12, 5.0, 1e-12) == pytest.approx(1.0)
    # Identical observed rates with non-zero noise dispersion means every
    # difference is explained by noise, so S_p is 1 — not 0.
    assert analyzer.poolability(0.0, 1.0, 1e-12) == pytest.approx(1.0)
    assert 0.0 <= analyzer.poolability(1.0, 0.5, 1e-12) <= 1.0
    # A degenerate denominator (zero spread *and* zero noise) must not divide.
    assert analyzer.poolability(0.0, 0.0, 1e-12) == 0.0


def test_single_atom_span_is_degenerate():
    assert analyzer.v_within([2.0], [1.0]) == 0.0
    assert analyzer.v_between([1.0], [2.0]) == 0.0
    assert analyzer.span_residual_scale([1.0], [2.0]) == 0.0


def test_noise_estimate_recovers_sigma_for_unit_weights():
    # Adjacent differences of i.i.d. N(0, sigma^2) have sigma^2 for unit w.
    rates = [0.31, -0.12, 0.44, -0.29, 0.08, 0.37, -0.41, 0.19]
    estimate = analyzer.robust_noise_estimate(rates, [1.0] * len(rates))
    assert estimate['pairs'] == len(rates) - 1
    assert estimate['sigma2_robust'] == pytest.approx(estimate['sigma_robust'] ** 2, rel=1e-12)
    assert estimate['sigma_robust'] > 0.0


def test_noise_estimate_is_zero_without_enough_atoms():
    assert analyzer.robust_noise_estimate([1.0], [1.0])['sigma2_robust'] == 0.0


def test_ema_interpolates_between_batch_and_previous():
    assert analyzer.ema(1.0, 0.0, 0.5) == pytest.approx(0.5)
    assert analyzer.ema(2.0, 2.0, 0.9) == pytest.approx(2.0)
    assert analyzer.ema(0.0, 4.0, 0.75) == pytest.approx(1.0)


def test_span_residual_scale_is_a_diagnostic_not_a_noise_estimate():
    rates, weights = [1.0, -1.0], [1.0, 1.0]
    # Two atoms with genuinely opposite credit: a large residual scale here is
    # signal, which is exactly why it must not feed V_within.
    assert analyzer.span_residual_scale(rates, weights) > 0.0


def test_risk_reduction_matches_the_spec_formula():
    assert analyzer.risk_reduction(3.0, 2.0, 4.0, 1e-12) == pytest.approx(0.25)
    assert analyzer.risk_reduction(1.0, 2.0, 1.0, 1e-12) == pytest.approx(-1.0)


def test_sliding_spans_covers_every_admissible_window():
    assert list(analyzer.sliding_spans(4, 4)) == [(0, 2), (1, 2), (2, 2), (0, 3), (1, 3), (0, 4)]
    assert list(analyzer.sliding_spans(1, 4)) == []
    assert list(analyzer.sliding_spans(5, 2)) == [(0, 2), (1, 2), (2, 2), (3, 2)]


def test_gate_is_logged_and_never_enforced_without_grass_inputs():
    gates = {'s_p': 0.5, 'r_p': 0.02, 'alpha_star': 0.1}
    rows = analyzer.evaluate_sample(0, [0.1, 0.5, 0.2, 0.9], [1.0, 2.0, 1.0, 3.0],
                                    sigma2=[0.01] * 4, max_span=4, eps=1e-12,
                                    risk_by_span=None, gates=gates)
    assert rows and all(row.gate_pass is None for row in rows)
    assert all('GRASS' in row.gate_missing for row in rows)
    summary = analyzer.summarise(rows)
    assert summary['gate_enforced'] is False
    assert summary['gate_evaluated_spans'] == 0


def test_gate_trips_only_when_all_three_inputs_clear_their_threshold():
    gates = {'s_p': 0.5, 'r_p': 0.02, 'alpha_star': 0.1}
    span = {(0, 2): {'r_p': 0.05, 'alpha_star': 0.2}}
    rows = analyzer.evaluate_sample(0, [0.1, 0.1, 0.1, 0.1], [1.0, 1.0, 1.0, 1.0],
                                    sigma2=[0.5] * 4, max_span=4, eps=1e-12,
                                    risk_by_span=span, gates=gates)
    row = next(r for r in rows if (r.start, r.length) == (0, 2))
    assert row.gate_pass is True
    failing = {(0, 2): {'r_p': 0.001, 'alpha_star': 0.2}}
    rows = analyzer.evaluate_sample(0, [0.1, 0.1, 0.1, 0.1], [1.0, 1.0, 1.0, 1.0],
                                    sigma2=[0.5] * 4, max_span=4, eps=1e-12,
                                    risk_by_span=failing, gates=gates)
    row = next(r for r in rows if (r.start, r.length) == (0, 2))
    assert row.gate_pass is False


def test_summary_reports_every_span_length_separately():
    gates = {'s_p': 0.5, 'r_p': 0.02, 'alpha_star': 0.1}
    rows = analyzer.evaluate_sample(0, [0.1, 0.4, 0.2, 0.8, 0.3], [1.0, 2.0, 1.0, 3.0, 2.0],
                                    sigma2=[0.02] * 5, max_span=3, eps=1e-12,
                                    risk_by_span=None, gates=gates)
    summary = analyzer.summarise(rows)
    assert set(summary['by_span_length']) == {'2', '3'}
    assert summary['spans'] == len(rows)


# --- position-dependent sigma2 ------------------------------------------------

def _heteroskedastic_series(sigma_first: float, sigma_last: float, count: int) -> list[float]:
    """i.i.d. Gaussian with an exactly known per-atom variance, via Box-Muller."""
    import random
    rng = random.Random(7)
    values = []
    for i in range(count):
        sigma = sigma_first + (sigma_last - sigma_first) * i / max(1, count - 1)
        u1, u2 = rng.random(), rng.random()
        values.append(sigma * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2))
    return values


def test_local_sigma2_tracks_a_known_19_fold_variance_ramp():
    rates = _heteroskedastic_series(0.2, 0.044, 400)
    weights = [1.0] * len(rates)
    global_sigma2 = 0.01
    local = analyzer.local_sigma2(rates, weights, 41, global_sigma2)
    head = sum(local[:40]) / 40.0
    tail = sum(local[-40:]) / 40.0
    true_head, true_tail = 0.2 ** 2, 0.044 ** 2
    # The property that matters: one global value cannot represent this ramp,
    # while the local estimate lands near the truth at both ends.
    assert head / tail > 5.0
    assert abs(head - true_head) < abs(global_sigma2 - true_head)
    assert abs(tail - true_tail) < abs(global_sigma2 - true_tail)
    # It must not be unbiased either: a ~40-pair MAD plus a shrink target that is
    # 5x the local truth pulls the tail up. Asserted so the bias stays visible.
    assert tail >= true_tail


def test_local_sigma2_shrinks_towards_the_global_value():
    rates = [0.0, 0.0, 0.0, 0.0, 0.0]
    weights = [1.0] * 5
    # Constant rates give a local MAD of exactly zero: there is nothing to shrink,
    # so the global value is used as-is rather than pulled toward a zero.
    local = analyzer.local_sigma2(rates, weights, 3, 0.25)
    assert all(v == pytest.approx(0.25) for v in local)
    # With a real (tiny) local estimate the shrink is strictly between the two.
    series = [0.0, 1e-3, -1e-3, 2e-3, -2e-3, 3e-3, -3e-3, 4e-3, -4e-3]
    weights9 = [1.0] * len(series)
    tiny = analyzer.local_sigma2(series, weights9, 5, 0.25)
    assert all(0.0 < v < 0.25 for v in tiny)


def test_local_sigma2_shrink_is_scale_invariant():
    # Multiplying the whole series by 10 must multiply sigma2 by 100 exactly;
    # a linear shrink towards a global that does not scale would break this.
    base = [0.0, 0.3, -0.2, 0.5, -0.4, 0.1, 0.2]
    weights = [1.0] * len(base)
    small = analyzer.local_sigma2(base, weights, 5, 0.01)
    large = analyzer.local_sigma2([10.0 * v for v in base], weights, 5, 1.0)
    assert all(b == pytest.approx(a * 100.0, rel=1e-9) for a, b in zip(small, large))


def test_local_sigma2_is_finite_and_positive_for_zero_weights():
    rates = [0.1, -0.2, 0.3]
    local = analyzer.local_sigma2(rates, [1.0, 0.0, 1.0], 5, 0.5)
    assert all(math.isfinite(v) and v > 0.0 for v in local)


def test_standardized_residuals_use_the_local_scale():
    rates = [0.1, -0.1, 0.2]
    weights = [1.0, 1.0, 1.0]
    tight = analyzer.standardized_residuals(rates, weights, [0.01] * 3)
    loose = analyzer.standardized_residuals(rates, weights, [0.04] * 3)
    # A larger assumed sigma2 shrinks the standardized residual.
    assert all(abs(a) > abs(b) for a, b in zip(tight, loose))


def test_describe_reports_the_tail_not_just_the_mean():
    values = [0.0] * 990 + [40.0] * 10
    stats = analyzer.describe(values)
    assert stats['median'] == pytest.approx(0.0)
    assert stats['max'] == pytest.approx(40.0)
    # The heavy tail has to show up as kurtosis, otherwise it is invisible in the mean.
    assert stats['kurtosis_median_scaled'] > 3.0
    assert stats['mean'] < 1.0
