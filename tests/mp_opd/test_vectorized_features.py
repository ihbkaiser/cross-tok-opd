"""Compare optimized preprocessing with the original scalar implementation."""
from types import SimpleNamespace
import pytest
import torch
from kdflow.algorithms.mp_opd import atom_features
from kdflow.algorithms._mp_opd_credit import span_tables, expected_atom_rates


def scalar_features(atoms, c):
    rows = []
    for i, a in enumerate(atoms):
        w = c.weight[i]
        rows.append(torch.stack((c.rate[i], c.base_credit[i], w,
            w.new_tensor(float(a.teacher_token_count)),
            w.new_tensor(float(a.byte_end - a.byte_start)),
            c.teacher_log_score[i] / float(a.teacher_token_count),
            c.student_old_log_score[i] / w,
            w.new_tensor(float(a.boundary_type == 'one_to_one')),
            w.new_tensor(float(a.boundary_type == 'multi_token')), w.new_tensor(1.))))
    return torch.stack(rows).detach()


def scalar_spans(base, weight, maximum):
    n = base.numel(); length = min(max(int(maximum), 1), max(n, 1))
    b = base.new_zeros((n, length)); w = weight.new_zeros((n, length))
    valid = torch.zeros_like(b, dtype=torch.bool)
    pb = torch.cat((base.new_zeros(1), base.cumsum(0)))
    pw = torch.cat((weight.new_zeros(1), weight.cumsum(0)))
    for start in range(n):
        for offset in range(length):
            end = start + offset + 1
            if end <= n:
                b[start, offset] = pb[end] - pb[start]
                w[start, offset] = pw[end] - pw[start]
                valid[start, offset] = True
    rate = torch.where(valid, b / w.clamp_min(1), torch.zeros_like(b)).detach()
    return b.detach(), w.detach(), rate, valid


def fixture(n, device='cpu', dtype=torch.float32):
    atoms = [SimpleNamespace(teacher_token_count=1+i%3, byte_start=i*4,
        byte_end=i*4+3, boundary_type='multi_token' if i%3 else 'one_to_one') for i in range(n)]
    base = torch.randn(n, device=device, dtype=dtype)
    weight = torch.randint(1, 4, (n,), device=device).to(dtype)
    c = SimpleNamespace(base_credit=base, weight=weight, rate=base/weight,
        teacher_log_score=base-2, student_old_log_score=base-3)
    return atoms, c


def scalar_rates(marginals, rates):
    n, length = rates.shape
    result = rates.new_zeros(n)
    for start in range(n):
        for offset in range(length):
            end = start + offset + 1
            if end <= n:
                result[start:end] += marginals[start,offset] * rates[start,offset]
    return result


@pytest.mark.parametrize('n,length', [(0,2),(1,4),(7,2),(17,5),(400,2)])
def test_expected_rates_bitwise(n, length):
    torch.manual_seed(42)
    q = torch.randn(n,length,dtype=torch.float64)
    r = torch.randn(n,length,dtype=torch.float32)
    assert torch.equal(expected_atom_rates(q,r), scalar_rates(q,r))


def test_expected_rates_first_and_second_derivatives():
    q = torch.randn(7,3,dtype=torch.float64,requires_grad=True)
    r = torch.randn_like(q,requires_grad=True)
    ga = torch.autograd.grad(expected_atom_rates(q,r).square().sum(),(q,r),create_graph=True)
    ge = torch.autograd.grad(scalar_rates(q,r).square().sum(),(q,r),create_graph=True)
    for a,b in zip(ga,ge): torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    ha = torch.autograd.grad(sum(g.square().sum() for g in ga),(q,r))
    he = torch.autograd.grad(sum(g.square().sum() for g in ge),(q,r))
    for a,b in zip(ha,he): torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('n', [1, 2, 17, 400])
@pytest.mark.parametrize('maximum', [1, 2, 5])
def test_span_tables_bitwise(n, maximum):
    _, c = fixture(n)
    for actual, expected in zip(span_tables(c.base_credit, c.weight, maximum),
                                scalar_spans(c.base_credit, c.weight, maximum)):
        assert torch.equal(actual, expected)
        assert not actual.requires_grad


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_features_bitwise_and_energy_gradient(dtype):
    atoms, c = fixture(41, dtype=dtype)
    c.base_credit.requires_grad_(True)
    actual = atom_features(atoms, c); expected = scalar_features(atoms, c)
    assert torch.equal(actual, expected)
    assert not actual.requires_grad
    energy = torch.nn.Linear(10, 2).to(dtype)
    ga = torch.autograd.grad(energy(actual).square().sum(), tuple(energy.parameters()))
    ge = torch.autograd.grad(energy(expected).square().sum(), tuple(energy.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(ga, ge))


def test_empty_spans():
    for a, b in zip(span_tables(torch.empty(0), torch.empty(0), 2),
                    scalar_spans(torch.empty(0), torch.empty(0), 2)):
        assert torch.equal(a, b)
