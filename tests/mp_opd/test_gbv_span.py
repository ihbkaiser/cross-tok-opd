"""GBV-Span: exact cost tables, exact DP, credit conservation, and the Potts limit.

The method note claims four things that can be pinned by unit test without a GPU:
the cost tables equal their closed form, the dynamic program equals brute force,
the selected partition still conserves signed credit and applies the existing hard
pooled loss, and ``q_i = w_i`` collapses the objective to the weighted Potts form.
"""
import ast as _ast
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kdflow.algorithms._mp_opd_credit import hard_partition_loss
from kdflow.algorithms._mp_opd_gbv_span import (
    ENERGY_RELATIVE_FLOOR,
    atom_logit_sensitivity,
    coarsest_admissible_partition,
    gbv_partition,
    gbv_partition_metrics,
    gbv_span_costs,
    token_logit_sensitivity,
)
from kdflow.algorithms._mp_opd_oracle import enumerate_partitions, partition_score

ARGS = ROOT / "kdflow/arguments/distillation_args.py"


def _draw(n: int, seed: int = 11, spread: float = 1.0):
    generator = torch.Generator().manual_seed(seed)
    rate = spread * torch.randn(n, generator=generator, dtype=torch.float64)
    weight = torch.randint(1, 5, (n,), generator=generator).double()
    sensitivity = 0.5 + torch.rand(n, generator=generator, dtype=torch.float64)
    return rate, weight, sensitivity


def _explicit_potts_cost(rate, weight, sensitivity, span, beta, energy, atomic_dof):
    start, end = span
    span_weight = weight[start:end].sum()
    span_rate = (rate[start:end] * weight[start:end]).sum() / span_weight
    distortion = 0.5 * (sensitivity[start:end] * (rate[start:end] - span_rate).square()).sum()
    dof = (sensitivity[start:end].sum() / span_weight) / atomic_dof
    return distortion / energy + beta * dof


@pytest.mark.parametrize("n,max_span,beta", [(6, 2, 0.3), (7, 4, 1.0), (9, 3, 0.0), (12, 5, 3.0)])
def test_cost_table_matches_closed_form(n, max_span, beta):
    rate, weight, sensitivity = _draw(n, seed=n + int(10 * beta))
    tables = gbv_span_costs(rate, weight, sensitivity, max_span, beta)
    mean_rate = (weight * rate).sum() / weight.sum()
    energy = 0.5 * (sensitivity * (rate - mean_rate).square()).sum()
    atomic_dof = (sensitivity / weight).sum()
    assert float(tables.energy) == pytest.approx(float(energy), rel=1e-12)
    assert float(tables.atomic_dof) == pytest.approx(float(atomic_dof), rel=1e-12)
    for start in range(n):
        for end in range(start + 1, min(n, start + max_span) + 1):
            expected = _explicit_potts_cost(
                rate, weight, sensitivity, (start, end), beta, energy, atomic_dof
            )
            assert float(tables.costs[start, end - start - 1]) == pytest.approx(
                float(expected), rel=1e-11, abs=1e-14
            )
            assert bool(tables.valid[start, end - start - 1])


@pytest.mark.parametrize("n,max_span,beta,seed", [(6, 2, 0.3, 1), (7, 4, 1.0, 2), (10, 3, 2.5, 3)])
def test_dp_equals_bruteforce_minimum_cost(n, max_span, beta, seed):
    rate, weight, sensitivity = _draw(n, seed=seed)
    tables = gbv_span_costs(rate, weight, sensitivity, max_span, beta)
    assert not tables.degenerate
    selected = gbv_partition(tables)
    brute = min(
        enumerate_partitions(n, max_span),
        key=lambda part: float(partition_score(tables.costs, part)),
    )
    assert float(partition_score(tables.costs, selected)) == pytest.approx(
        float(partition_score(tables.costs, brute)), rel=1e-12
    )


def test_selected_partition_conserves_signed_credit_and_reuses_hard_loss():
    n = 8
    rate, weight, sensitivity = _draw(n, seed=7)
    base = rate * weight
    tables = gbv_span_costs(rate, weight, sensitivity, 3, 1.0)
    partition = gbv_partition(tables)
    pooled = torch.zeros(n, dtype=torch.float64)
    for start, end in partition:
        pooled[start:end] = (base[start:end].sum() / weight[start:end].sum())
    assert float((weight * pooled).sum() - base.sum()) == pytest.approx(0.0, abs=1e-12)
    nll = torch.randn(n, dtype=torch.float64, requires_grad=True)
    loss = hard_partition_loss(nll, base, weight, partition)
    expected = sum(
        (base[start:end].sum() / weight[start:end].sum()) * nll[start:end].sum()
        for start, end in partition
    )
    assert float(loss.detach()) == pytest.approx(float(expected.detach()), rel=1e-12)
    gradient = torch.autograd.grad(loss, nll, retain_graph=True)[0]
    assert torch.equal(
        gradient,
        torch.cat(
            [
                torch.full((end - start,), float(base[start:end].sum() / weight[start:end].sum()),
                           dtype=torch.float64)
                for start, end in partition
            ]
        ),
    )


def test_constant_rate_is_degenerate_and_returns_coarsest_tiling():
    n = 7
    weight = torch.tensor([1.0, 3.0, 2.0, 4.0, 1.0, 2.0, 5.0], dtype=torch.float64)
    rate = torch.full((n,), 0.25, dtype=torch.float64)
    sensitivity = torch.linspace(0.5, 1.5, n, dtype=torch.float64)
    tables = gbv_span_costs(rate, weight, sensitivity, 3, 1.0)
    assert tables.degenerate
    assert float(tables.energy) <= ENERGY_RELATIVE_FLOOR * float(0.5 * (sensitivity * rate.square()).sum())
    assert gbv_partition(tables) == coarsest_admissible_partition(n, 3)
    assert gbv_partition(tables) == ((0, 3), (3, 6), (6, 7))


def test_beta_extremes_move_between_atomic_and_coarsest():
    n = 9
    rate, weight, sensitivity = _draw(n, seed=5, spread=2.0)
    atomic = gbv_partition(gbv_span_costs(rate, weight, sensitivity, 4, 0.0))
    assert atomic == tuple((index, index + 1) for index in range(n))
    coarsest = gbv_partition(gbv_span_costs(rate, weight, sensitivity, 4, 1e6))
    assert coarsest == coarsest_admissible_partition(n, 4)


def test_token_count_geometry_is_the_weighted_potts_special_case():
    n = 8
    rate, weight, _ = _draw(n, seed=13)
    tables = gbv_span_costs(rate, weight, weight, 4, 1.0)
    mean_rate = (weight * rate).sum() / weight.sum()
    energy = 0.5 * (weight * (rate - mean_rate).square()).sum()
    for start in range(n):
        for offset in range(min(4, n - start)):
            assert float(tables.retained_dof[start, offset]) == pytest.approx(1.0 / n, rel=1e-12)
    partition = gbv_partition(tables)
    potts = 0.5 * sum(
        (weight[start:end] * (rate[start:end] - (rate[start:end] * weight[start:end]).sum()
                              / weight[start:end].sum()).square()).sum()
        for start, end in partition
    )
    assert float(partition_score(tables.costs, partition)) == pytest.approx(
        float(potts / energy + 1.0 * len(partition) / n), rel=1e-11
    )
    metrics = gbv_partition_metrics(tables, partition, rate, weight)
    assert float(metrics["mp_opd_gbv_retained_dof_fraction"]) == pytest.approx(
        len(partition) / n, rel=1e-12
    )


@pytest.mark.parametrize("vocab_chunk", [1, 7, 4096])
def test_token_logit_sensitivity_matches_explicit_softmax(vocab_chunk):
    torch.manual_seed(4)
    logits = torch.randn(5, 23, dtype=torch.float64)
    labels = torch.tensor([0, 5, 22, 11, 3])
    probabilities = torch.softmax(logits, dim=-1)
    one_hot = torch.nn.functional.one_hot(labels, logits.shape[1]).to(torch.float64)
    expected = (probabilities - one_hot).square().sum(dim=-1)
    assert torch.allclose(
        token_logit_sensitivity(logits, labels, vocab_chunk=vocab_chunk), expected, atol=1e-14
    )
    selected_log_prob = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1).log()
    assert torch.allclose(
        token_logit_sensitivity(
            logits, labels, selected_log_prob=selected_log_prob, vocab_chunk=vocab_chunk
        ),
        expected,
        atol=1e-14,
    )


def test_sensitivity_is_detached_and_survives_bf16_logits():
    torch.manual_seed(6)
    logits = torch.randn(4, 17, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.tensor([1, 2, 3, 4])
    per_token = token_logit_sensitivity(logits, labels, vocab_chunk=5)
    assert not per_token.requires_grad
    reference = logits.detach().to(torch.float64)
    expected = (
        torch.softmax(reference, dim=-1)
        - torch.nn.functional.one_hot(labels, 17).to(torch.float64)
    ).square().sum(dim=-1)
    assert torch.allclose(per_token, expected, atol=1e-3)


def test_atom_sensitivity_sums_ranges_and_allows_gaps_but_not_overlap():
    torch.manual_seed(9)
    logits = torch.randn(6, 13, dtype=torch.float64)
    labels = torch.tensor([0, 1, 2, 3, 4, 5])
    ranges = ((0, 2), (2, 3), (3, 6))
    atoms = atom_logit_sensitivity(logits, labels, ranges, vocab_chunk=4)
    per_token = token_logit_sensitivity(logits, labels, vocab_chunk=4)
    assert torch.allclose(
        atoms, torch.stack([per_token[start:end].sum() for start, end in ranges])
    )
    # A real microbatch leaves non-atomized tokens (masked EOS) between atoms, so a gap
    # is legitimate: those tokens belong to no atom and contribute to no sensitivity.
    gapped = ((0, 2), (3, 6))
    assert torch.allclose(
        atom_logit_sensitivity(logits, labels, gapped, vocab_chunk=4),
        torch.stack([per_token[0:2].sum(), per_token[3:6].sum()]),
    )
    with pytest.raises(ValueError):
        atom_logit_sensitivity(logits, labels, ((0, 3), (2, 6)))
    with pytest.raises(ValueError):
        atom_logit_sensitivity(logits, labels, ((3, 4), (0, 2)))
    with pytest.raises(ValueError):
        atom_logit_sensitivity(logits, labels, ((0, 2), (2, 7)))
    with pytest.raises(ValueError):
        atom_logit_sensitivity(logits, labels, ())


@pytest.mark.parametrize(
    "rate,weight,sensitivity,max_span,beta",
    [
        ([1.0, 2.0], [1.0, 1.0], [1.0], 2, 1.0),
        ([1.0, 2.0], [1.0, 1.0], [1.0, 1.0], 0, 1.0),
        ([1.0, 2.0], [1.0, 1.0], [1.0, 1.0], 2, -0.5),
        ([1.0, 2.0], [1.0, 0.0], [1.0, 1.0], 2, 1.0),
        ([1.0, 2.0], [1.0, 1.0], [1.0, -1.0], 2, 1.0),
    ],
)
def test_invalid_inputs_raise(rate, weight, sensitivity, max_span, beta):
    with pytest.raises(ValueError):
        gbv_span_costs(
            torch.tensor(rate), torch.tensor(weight), torch.tensor(sensitivity), max_span, beta
        )


def test_empty_atom_sequence_raises():
    with pytest.raises(ValueError):
        gbv_span_costs(
            torch.zeros(0), torch.zeros(0), torch.zeros(0), 4, 1.0
        )


def test_partition_metrics_cover_the_documented_telemetry():
    n = 10
    rate, weight, sensitivity = _draw(n, seed=17)
    tables = gbv_span_costs(rate, weight, sensitivity, 4, 0.7)
    partition = gbv_partition(tables)
    metrics = gbv_partition_metrics(tables, partition, rate, weight)
    documented = {
        "mp_opd_gbv_total_cost",
        "mp_opd_gbv_distortion_term",
        "mp_opd_gbv_dof_term",
        "mp_opd_gbv_retained_dof_fraction",
        "mp_opd_gbv_selected_span_count",
        "mp_opd_gbv_selected_span_length_mean",
        "mp_opd_gbv_span_1_fraction",
        "mp_opd_gbv_span_2_fraction",
        "mp_opd_gbv_span_3_fraction",
        "mp_opd_gbv_span_4_fraction",
        "mp_opd_gbv_boundary_strength_mean",
    }
    assert documented <= set(metrics)
    assert all(isinstance(value, torch.Tensor) for value in metrics.values())
    assert all(not value.requires_grad for value in metrics.values())
    fractions = sum(float(metrics[f"mp_opd_gbv_span_{length}_fraction"]) for length in (1, 2, 3, 4))
    assert fractions == pytest.approx(1.0, abs=1e-12)
    assert float(metrics["mp_opd_gbv_selected_span_count"]) == len(partition)
    assert float(metrics["mp_opd_gbv_total_cost"]) == pytest.approx(
        float(metrics["mp_opd_gbv_distortion_term"]) + float(metrics["mp_opd_gbv_dof_term"]),
        rel=1e-12,
        abs=1e-14,
    )
    single_rate, single_weight, single_sensitivity = _draw(3, seed=19)
    single_tables = gbv_span_costs(single_rate, single_weight, single_sensitivity, 4, 0.7)
    single = gbv_partition_metrics(
        single_tables, ((0, 3),), single_rate, single_weight
    )
    assert float(single["mp_opd_gbv_boundary_strength_mean"]) == 0.0


def test_args_declare_the_gbv_knobs():
    """Source-level check so it holds in environments without transformers."""
    source = ARGS.read_text()
    tree = _ast.parse(source)
    fields = {}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.AnnAssign) and getattr(node.target, "id", "").startswith("mp_opd_gbv"):
            fields[node.target.id] = _ast.literal_eval(
                {kw.arg: kw.value for kw in node.value.keywords}["default"]
            )
    assert fields == {"mp_opd_gbv_beta": 1.0, "mp_opd_gbv_geometry": "token_count"}
    assert '"gbv"' in source
    assert "unsupported mp_opd_gbv_geometry" in source
    assert "mp_opd_gbv_beta must be finite and nonnegative" in source


def test_args_dataclass_accepts_gbv_when_transformers_is_present():
    pytest.importorskip("transformers")
    from kdflow.arguments.distillation_args import DistillationArguments

    defaults = DistillationArguments.__dataclass_fields__
    assert defaults["mp_opd_gbv_beta"].default == 1.0
    assert defaults["mp_opd_gbv_geometry"].default == "token_count"
