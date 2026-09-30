import itertools

import pytest
import torch

from kdflow.algorithms._mp_opd_credit import (
    credit_conservation_residual,
    expected_atom_rates,
    hard_partition_loss,
    span_tables,
)
from kdflow.algorithms._mp_opd_functional import centered_finite_difference
from kdflow.algorithms._mp_opd_oracle import (
    enumerate_partitions,
    hard_max_partition,
    partition_score,
    span_utility_table,
)
from kdflow.algorithms._mp_opd_semimarkov import semi_markov_partition


def test_credit_conservation_random_signed_base_positive_integer_weight():
    torch.manual_seed(3)
    base = torch.randn(7)
    weight = torch.randint(1, 5, (7,)).float()
    _, _, rates, valid = span_tables(base, weight, 4)
    mu = semi_markov_partition(torch.randn_like(rates), valid_mask=valid).marginals
    atom_rates = expected_atom_rates(mu, rates)
    assert abs(float(credit_conservation_residual(base, weight, atom_rates))) < 2e-6


def test_l1_exact_atomic_loss_and_gradient_equality():
    h = torch.randn(5, dtype=torch.float64, requires_grad=True)
    base = torch.tensor([-2.0, 0.0, 1.0, 4.0, -0.5], dtype=torch.float64)
    weight = torch.tensor([1, 2, 3, 1, 2], dtype=torch.float64)
    partition = tuple((i, i + 1) for i in range(5))
    hard = hard_partition_loss(h, base, weight, partition)
    expected = ((base / weight) * h).sum()
    assert torch.equal(hard, expected)
    assert torch.equal(torch.autograd.grad(hard, h, retain_graph=True)[0], torch.autograd.grad(expected, h)[0])


def test_constant_rate_merge_preserves_loss_and_gradient():
    h = torch.randn(4, requires_grad=True)
    weight = torch.tensor([1.0, 2.0, 4.0, 3.0])
    base = 0.75 * weight
    atomic = hard_partition_loss(h, base, weight, ((0, 1), (1, 2), (2, 3), (3, 4)))
    merged = hard_partition_loss(h, base, weight, ((0, 4),))
    assert torch.equal(atomic, merged)
    assert torch.equal(torch.autograd.grad(atomic, h, retain_graph=True)[0], torch.autograd.grad(merged, h)[0])


@pytest.mark.parametrize("base", [torch.zeros(4), torch.tensor([1e6, -1e6, 1e-6, -1e-6])])
def test_zero_and_large_signed_credit_remain_finite(base):
    h = torch.randn(4, requires_grad=True)
    loss = hard_partition_loss(h, base, torch.ones(4), ((0, 2), (2, 4)))
    assert torch.isfinite(loss)
    assert torch.isfinite(torch.autograd.grad(loss, h)[0]).all()


def test_hard_dp_equals_bruteforce_argmax():
    torch.manual_seed(8)
    table = torch.randn(7, 4)
    valid = torch.zeros_like(table, dtype=torch.bool)
    for i in range(7):
        valid[i, : min(4, 7 - i)] = True
    table = table.masked_fill(~valid, -torch.inf)
    oracle = hard_max_partition(table, valid)
    brute = max(enumerate_partitions(7, 4), key=lambda part: float(partition_score(table, part)))
    assert float(oracle.score) == pytest.approx(float(partition_score(table, brute)))


def test_prefix_utility_matches_explicit_per_span_gradient_inner_product_and_sign():
    theta = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)
    x = torch.tensor([[1.0, 2.0], [-0.5, 0.7], [0.3, -1.1]], dtype=torch.float64)
    score = x @ theta
    v = torch.tensor([0.8, -0.4], dtype=torch.float64)
    z = torch.stack([(v * torch.autograd.grad(score[i], theta, retain_graph=True)[0]).sum() for i in range(3)])
    base = torch.tensor([0.4, -0.6, 1.2], dtype=torch.float64)
    weight = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
    table, _ = span_utility_table(base, weight, z, 3)
    for start in range(3):
        for end in range(start + 1, 4):
            rate = base[start:end].sum() / weight[start:end].sum()
            ell = rate * (-(score[start:end]).sum())
            explicit = (v * torch.autograd.grad(ell, theta, retain_graph=True)[0]).sum()
            assert float(table[start, end - start - 1]) == pytest.approx(float(explicit))


def test_centered_finite_difference_matches_autograd_directional_derivative():
    value = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)
    direction = torch.tensor([0.7, 0.1], dtype=torch.float64)
    fn = lambda x: (x.sin() + 0.2 * x.square()).sum()
    analytic = (torch.autograd.grad(fn(value), value)[0] * direction).sum()
    finite = centered_finite_difference(fn, value.detach(), direction, 1e-5)
    assert float((analytic - finite).abs()) < 1e-8
