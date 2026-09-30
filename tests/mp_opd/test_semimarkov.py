import itertools
import math

import pytest
import torch

from kdflow.algorithms._mp_opd_oracle import enumerate_partitions
from kdflow.algorithms._mp_opd_semimarkov import semi_markov_partition


def score_partition(energies, partition):
    return sum(energies[i, j - i - 1] for i, j in partition)


@pytest.mark.parametrize("n,max_length", [(1, 4), (2, 2), (5, 3), (8, 4)])
def test_dp_matches_bruteforce_logz_marginals_and_coverage(n, max_length):
    torch.manual_seed(100 + n)
    energies = torch.randn(n, min(max_length, n), dtype=torch.float64)
    result = semi_markov_partition(energies, temperature=0.8)
    partitions = list(enumerate_partitions(n, max_length))
    scores = torch.stack([score_partition(energies, part) / 0.8 for part in partitions])
    brute_logz = torch.logsumexp(scores, 0)
    brute_mu = torch.zeros_like(result.marginals, dtype=torch.float64)
    probabilities = torch.softmax(scores, 0)
    for probability, partition in zip(probabilities, partitions):
        for start, end in partition:
            brute_mu[start, end - start - 1] += probability
    assert torch.allclose(result.log_z.double(), brute_logz, atol=2e-6)
    assert torch.allclose(result.marginals.double(), brute_mu, atol=2e-6)
    assert float(result.coverage_max_error) < 1e-6
    assert torch.allclose(probabilities.sum(), torch.tensor(1.0, dtype=probabilities.dtype))


def test_n_zero_is_neutral_and_all_invalid_fails_closed():
    empty = semi_markov_partition(torch.empty(0, 1))
    assert empty.marginals.shape == (0, 1) and float(empty.log_z) == 0.0
    with pytest.raises(ValueError, match="masked"):
        semi_markov_partition(torch.zeros(2, 2), valid_mask=torch.zeros(2, 2, dtype=torch.bool))


def test_mixed_precision_input_runs_dp_in_float64_without_nonfinite_outputs():
    energies = torch.tensor([[100.0, -100.0], [75.0, 0.0]], dtype=torch.float16)
    result = semi_markov_partition(energies)
    assert result.log_z.dtype == torch.float64
    assert all(torch.isfinite(value).all() for value in (result.log_z, result.marginals, result.entropy))


def test_long_chain_full_coverage_and_gradient():
    energies = torch.full((2048, 2), 3.0, requires_grad=True)
    result = semi_markov_partition(energies)
    assert float(result.coverage_max_error.detach()) < 1e-7
    result.log_z.backward()
    assert torch.isfinite(energies.grad).all()
    assert torch.allclose(energies.grad.double(), result.marginals, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.6])
def test_host_mask_preserves_outputs_and_first_gradient(dtype, temperature):
    energies = torch.randn(7, 4, dtype=dtype, requires_grad=True)
    valid = torch.tensor(
        [[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 0], [1, 0, 0, 0],
         [1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool
    )
    ref = semi_markov_partition(energies, temperature=temperature, valid_mask=valid)
    cand = semi_markov_partition(energies, temperature=temperature, valid_mask=valid, host_mask=True)
    for left, right in zip(ref.__dict__.values(), cand.__dict__.values()):
        assert torch.equal(left, right), (left, right)
    ref_grad = torch.autograd.grad(ref.log_z, energies, create_graph=True)[0]
    cand_grad = torch.autograd.grad(cand.log_z, energies, create_graph=True)[0]
    assert torch.equal(ref_grad, cand_grad)


def test_host_mask_preserves_higher_order_derivative_and_failure_semantics():
    values = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    ref = semi_markov_partition(values, valid_mask=torch.ones(4, 3, dtype=torch.bool))
    cand = semi_markov_partition(values, valid_mask=torch.ones(4, 3, dtype=torch.bool), host_mask=True)
    ref_g = torch.autograd.grad(ref.log_z, values, create_graph=True)[0]
    cand_g = torch.autograd.grad(cand.log_z, values, create_graph=True)[0]
    ref_h = torch.autograd.grad((ref_g.square()).sum(), values)[0]
    cand_h = torch.autograd.grad((cand_g.square()).sum(), values)[0]
    assert torch.equal(ref_h, cand_h)
    invalid = torch.zeros(3, 2, dtype=torch.bool)
    for host in (False, True):
        with pytest.raises(ValueError, match="masked"):
            semi_markov_partition(torch.zeros(3, 2), valid_mask=invalid, host_mask=host)
