"""Cross-atom credit operators: the identity is Atomic, the rest are pre-registered probes.

The specification (`cross_atom_credit_experiment_spec.md` section 10) makes five
properties mandatory before any training run: identity passthrough, the convex
forward transfer of section 4.2, sequence-boundary isolation, mask-boundary
isolation, shuffle determinism, and a numerical Atomic regression against the
historical pooled loss. Everything here runs on CPU torch with no model, no
teacher and no GPU: an operator that cannot be pinned this way cannot be trusted
inside a training step.
"""
import ast as _ast
from pathlib import Path

import pytest
import torch

from kdflow.algorithms._mp_opd_credit import hard_partition_loss
from kdflow.algorithms._mp_opd_credit_transform import (
    CREDIT_TRANSFORM_CHOICES,
    METRIC_PREFIX,
    AtomCreditBatch,
    BackwardNeighborCreditTransform,
    CausalKernelCreditTransform,
    CreditTransform,
    ExternalCreditAugmentationTransform,
    ForwardNeighborCreditTransform,
    IdentityCreditTransform,
    ShuffledNeighborCreditTransform,
    build_credit_transform,
    credit_transform_from_args,
    kernel_weights,
    neighbor_reach,
    production_credit_batch,
)

ARGS = Path(__file__).resolve().parents[2] / "kdflow/arguments/distillation_args.py"


def batch_of(rate, *, seq_ids=None, valid=None, advantage=None, weight=None):
    """One packed batch from a list of rates, with optional masks and sequences."""
    rate = torch.tensor(rate, dtype=torch.float32)
    n = rate.numel()
    seq_ids = torch.zeros(n, dtype=torch.long) if seq_ids is None else torch.tensor(seq_ids)
    valid = torch.ones(n, dtype=torch.bool) if valid is None else torch.tensor(valid, dtype=torch.bool)
    if n == 0:
        positions = torch.zeros(0, dtype=torch.long)
    else:
        starts = torch.cat((torch.ones(1, dtype=torch.bool), seq_ids[1:] != seq_ids[:-1]))
        group_start = torch.cummax(
            torch.where(starts, torch.arange(n), torch.zeros(n, dtype=torch.long)), 0
        ).values
        positions = torch.arange(n, dtype=torch.long) - group_start
    metadata = {} if advantage is None else {"future_advantage": torch.tensor(advantage)}
    return AtomCreditBatch(
        rate_credit=rate,
        base_credit=rate if weight is None else rate * weight,
        token_count=torch.ones(n) if weight is None else torch.as_tensor(weight, dtype=torch.float32),
        valid_mask=valid,
        seq_ids=seq_ids,
        atom_positions=positions,
        metadata=metadata,
    )


def draw(n, seed=7):
    generator = torch.Generator().manual_seed(seed)
    return 2.0 * torch.randn(n, generator=generator) - 0.5


# ---------------------------------------------------------------------------
# Section 10.1 -- identity
# ---------------------------------------------------------------------------


def test_identity_reproduces_the_atomic_rate_exactly():
    rate = draw(9)
    batch = batch_of(rate.tolist())
    out = IdentityCreditTransform()(batch, training=True)
    assert torch.equal(out.effective_credit, batch.rate_credit)
    assert float(out.diagnostics[f"{METRIC_PREFIX}mean_abs_delta"]) == 0.0
    assert float(out.diagnostics[f"{METRIC_PREFIX}transfer_fraction"]) == 0.0


# ---------------------------------------------------------------------------
# Section 10.2 -- forward convex transfer, the exact worked example
# ---------------------------------------------------------------------------


def test_forward_convex_matches_the_spec_worked_example():
    batch = batch_of([1.0, 2.0, 3.0])
    out = ForwardNeighborCreditTransform(0.25)(batch, training=True)
    assert torch.allclose(out.effective_credit, torch.tensor([1.25, 2.25, 3.0]))


def test_backward_convex_is_the_matched_mirror():
    batch = batch_of([1.0, 2.0, 3.0])
    out = BackwardNeighborCreditTransform(0.25)(batch, training=True)
    assert torch.allclose(out.effective_credit, torch.tensor([1.0, 1.75, 2.75]))


def test_raw_additive_transfer_is_not_the_convex_one():
    batch = batch_of([1.0, 2.0, 3.0])
    out = ForwardNeighborCreditTransform(0.25, convex=False)(batch, training=True)
    assert torch.allclose(out.effective_credit, torch.tensor([1.5, 2.75, 3.0]))


def test_lambda_zero_is_identity_for_every_neighbor_operator():
    batch = batch_of(draw(12, seed=3).tolist())
    for transform in (
        ForwardNeighborCreditTransform(0.0),
        BackwardNeighborCreditTransform(0.0),
        ShuffledNeighborCreditTransform(0.0, seed=1),
    ):
        out = transform(batch, training=True)
        assert torch.equal(out.effective_credit, batch.rate_credit), transform.name


def test_lambda_outside_the_unit_interval_is_rejected():
    for factory in (
        lambda value: ForwardNeighborCreditTransform(value),
        lambda value: BackwardNeighborCreditTransform(value),
        lambda value: ShuffledNeighborCreditTransform(value),
    ):
        for value in (-0.1, 1.1):
            with pytest.raises(ValueError):
                factory(value)


# ---------------------------------------------------------------------------
# Section 10.3 -- packed-sequence isolation
# ---------------------------------------------------------------------------


def test_no_transfer_across_a_sequence_boundary():
    batch = batch_of([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], seq_ids=[0, 0, 0, 1, 1, 1])
    forward = ForwardNeighborCreditTransform(0.25)(batch, training=True).effective_credit
    assert torch.allclose(forward, torch.tensor([1.25, 2.25, 3.0, 12.5, 22.5, 30.0]))
    backward = BackwardNeighborCreditTransform(0.25)(batch, training=True).effective_credit
    assert torch.allclose(backward, torch.tensor([1.0, 1.75, 2.75, 10.0, 17.5, 27.5]))


def test_reachability_is_sequence_local():
    batch = batch_of([1.0, 2.0, 3.0, 4.0], seq_ids=[0, 0, 1, 1])
    assert neighbor_reach(batch, 1).tolist() == [True, False, True, False]
    assert neighbor_reach(batch, -1).tolist() == [False, True, False, True]
    assert neighbor_reach(batch, 2).tolist() == [False, False, False, False]


def test_kernel_never_crosses_a_sequence_boundary():
    batch = batch_of([1.0, 2.0, 100.0, 200.0], seq_ids=[0, 0, 1, 1])
    out = CausalKernelCreditTransform(horizon=2, direction="forward")(batch, training=True)
    assert torch.allclose(out.effective_credit, torch.tensor([1.5, 2.0, 150.0, 200.0]))


# ---------------------------------------------------------------------------
# Section 10.4 -- mask boundary
# ---------------------------------------------------------------------------


def test_a_masked_atom_neither_sends_nor_receives_credit():
    batch = batch_of([1.0, 50.0, 3.0, 4.0], valid=[True, False, True, True])
    out = ForwardNeighborCreditTransform(0.5)(batch, training=True).effective_credit
    # Atom 0 may not take the masked atom 1's credit, and atom 1 keeps its own.
    assert torch.allclose(out, torch.tensor([1.0, 50.0, 3.5, 4.0]))
    for transform in (
        BackwardNeighborCreditTransform(0.5),
        CausalKernelCreditTransform(horizon=3, direction="symmetric"),
    ):
        applied = transform(batch, training=True).effective_credit
        assert float(applied[1]) == 50.0
        assert float(applied[0]) == 1.0, transform.name
    # A shuffle partner is always another *valid* atom, so the masked atom's credit
    # can never appear anywhere else.
    shuffled = ShuffledNeighborCreditTransform(1.0, seed=5)(batch, training=True).effective_credit
    assert float(shuffled[1]) == 50.0
    assert 50.0 not in shuffled[[0, 2, 3]].tolist()
    assert set(shuffled[[0, 2, 3]].tolist()) == {1.0, 3.0, 4.0}


def test_reachability_stops_at_a_masked_atom():
    batch = batch_of([1.0, 2.0, 3.0, 4.0], valid=[True, True, False, True])
    assert neighbor_reach(batch, 1).tolist() == [True, False, False, False]
    assert neighbor_reach(batch, 2).tolist() == [False, False, False, False]


# ---------------------------------------------------------------------------
# Section 10.5 -- shuffle determinism and marginal preservation
# ---------------------------------------------------------------------------


def test_shuffle_is_deterministic_per_seed_and_seed_sensitive():
    batch = batch_of(draw(64, seed=11).tolist())
    first = ShuffledNeighborCreditTransform(1.0, seed=3)(batch, training=True).effective_credit
    again = ShuffledNeighborCreditTransform(1.0, seed=3)(batch, training=True).effective_credit
    other = ShuffledNeighborCreditTransform(1.0, seed=4)(batch, training=True).effective_credit
    assert torch.equal(first, again)
    assert not torch.allclose(first, other)


def test_shuffle_preserves_the_marginal_credit_multiset():
    rate = draw(40, seed=13)
    batch = batch_of(rate.tolist())
    # lambda = 1 makes the operator exactly the shuffled source credit.
    shuffled = ShuffledNeighborCreditTransform(1.0, seed=9)(batch, training=True).effective_credit
    assert torch.allclose(torch.sort(shuffled).values, torch.sort(batch.rate_credit).values)
    assert not torch.allclose(shuffled, batch.rate_credit)


def test_shuffle_partners_stay_inside_their_own_sequence():
    # Two sequences with disjoint value ranges: a cross-sequence partner would show up
    # immediately as an out-of-range credit on one side.
    rate = [1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0]
    batch = batch_of(rate, seq_ids=[0, 0, 0, 0, 1, 1, 1, 1])
    shuffled = ShuffledNeighborCreditTransform(1.0, seed=2)(batch, training=True).effective_credit
    assert set(shuffled[:4].tolist()) == {1.0, 2.0, 3.0, 4.0}
    assert set(shuffled[4:].tolist()) == {100.0, 200.0, 300.0, 400.0}


def test_shuffle_of_a_single_atom_sequence_is_identity():
    batch = batch_of([5.0], seq_ids=[0])
    out = ShuffledNeighborCreditTransform(1.0, seed=2)(batch, training=True)
    assert torch.allclose(out.effective_credit, torch.tensor([5.0]))


def test_shuffle_leaves_a_masked_atom_untouched():
    batch = batch_of([1.0, 7.0, 3.0, 4.0], valid=[True, False, True, True])
    applied = ShuffledNeighborCreditTransform(1.0, seed=2)(batch, training=True).effective_credit
    assert float(applied[1]) == 7.0
    assert set(applied[[0, 2, 3]].tolist()) == {1.0, 3.0, 4.0}


# ---------------------------------------------------------------------------
# Section 5 -- static causal kernels and their matched controls
# ---------------------------------------------------------------------------


def test_weights_are_normalized_for_both_kernel_families():
    for kind in ("uniform", "exponential"):
        for horizon in (1, 2, 4, 8):
            weights = kernel_weights(kind, horizon, 0.5)
            assert float(weights.sum()) == pytest.approx(1.0, rel=1e-12)
            assert weights.numel() == horizon
    with pytest.raises(ValueError):
        kernel_weights("triangular", 3, 0.5)
    with pytest.raises(ValueError):
        kernel_weights("uniform", 0, 0.5)
    with pytest.raises(ValueError):
        kernel_weights("exponential", 3, 0.0)


def test_uniform_kernel_with_horizon_two_is_the_forward_half_transfer():
    batch = batch_of(draw(11, seed=21).tolist())
    kernel = CausalKernelCreditTransform(horizon=2, kind="uniform", direction="forward")
    neighbor = ForwardNeighborCreditTransform(0.5)
    assert torch.allclose(
        kernel(batch, training=True).effective_credit,
        neighbor(batch, training=True).effective_credit,
    )


def test_horizon_one_is_atomic_for_every_direction():
    batch = batch_of(draw(7, seed=5).tolist())
    for direction in ("forward", "backward", "symmetric"):
        out = CausalKernelCreditTransform(horizon=1, direction=direction)(batch, training=True)
        assert torch.allclose(out.effective_credit, batch.rate_credit)


def test_backward_kernel_is_the_time_reversal_of_the_forward_kernel():
    rate = draw(9, seed=17)
    forward = CausalKernelCreditTransform(horizon=3, direction="forward")(
        batch_of(rate.tolist()), training=True
    ).effective_credit
    backward = CausalKernelCreditTransform(horizon=3, direction="backward")(
        batch_of(rate.flip(0).tolist()), training=True
    ).effective_credit
    assert torch.allclose(backward, forward.flip(0), atol=1e-6)


def test_symmetric_kernel_is_the_mean_of_the_two_directions():
    batch = batch_of(draw(13, seed=23).tolist())
    forward = CausalKernelCreditTransform(horizon=4, direction="forward")(
        batch, training=True
    ).effective_credit
    backward = CausalKernelCreditTransform(horizon=4, direction="backward")(
        batch, training=True
    ).effective_credit
    symmetric = CausalKernelCreditTransform(horizon=4, direction="symmetric")(
        batch, training=True
    ).effective_credit
    assert torch.allclose(symmetric, 0.5 * (forward + backward))


def test_exponential_kernel_matches_its_closed_form_on_interior_atoms():
    rate = draw(8, seed=29)
    decay, horizon = 0.5, 4
    batch = batch_of(rate.tolist())
    applied = CausalKernelCreditTransform(
        horizon=horizon, kind="exponential", decay=decay, direction="forward"
    )(batch, training=True).effective_credit
    weights = torch.tensor([decay ** step for step in range(horizon)])
    for index in range(rate.numel() - (horizon - 1)):
        expected = float((weights * rate[index : index + horizon]).sum() / weights.sum())
        assert float(applied[index]) == pytest.approx(expected, rel=1e-6)


def test_kernel_truncates_at_the_sequence_end_instead_of_shrinking_to_zero():
    rate = draw(6, seed=31)
    batch = batch_of(rate.tolist())
    applied = CausalKernelCreditTransform(horizon=3, direction="forward")(
        batch, training=True
    ).effective_credit
    assert float(applied[-1]) == pytest.approx(float(rate[-1]))
    assert float(applied[-2]) == pytest.approx(float((rate[-2] + rate[-1]) / 2))


def test_kernel_normalization_falls_back_to_atomic_on_a_masked_run():
    batch = batch_of([1.0, 2.0, 3.0, 4.0], valid=[True, False, True, True])
    applied = CausalKernelCreditTransform(horizon=4, direction="forward")(
        batch, training=True
    ).effective_credit
    assert float(applied[0]) == 1.0
    assert float(applied[1]) == 2.0
    assert float(applied[2]) == pytest.approx(3.5)
    assert float(applied[3]) == 4.0


# ---------------------------------------------------------------------------
# Section 3.3/3.4 -- offline oracle-ish augmentation and scale matching
# ---------------------------------------------------------------------------


def test_external_operator_adds_the_advantage_verbatim():
    rate = draw(6, seed=37)
    advantage = torch.linspace(-1.0, 1.0, 6)
    batch = batch_of(rate.tolist(), advantage=advantage.tolist())
    for alpha in (0.1, 0.25, 0.5):
        applied = ExternalCreditAugmentationTransform(alpha)(batch, training=True).effective_credit
        assert torch.allclose(applied, batch.rate_credit + alpha * advantage, atol=1e-6)


def test_external_rms_matching_equalises_the_credit_scale():
    rate = draw(16, seed=41)
    advantage = torch.linspace(-2.0, 2.0, 16) * 10.0
    batch = batch_of(rate.tolist(), advantage=advantage.tolist())
    matched = ExternalCreditAugmentationTransform(0.5, scale_match="rms")(
        batch, training=True
    ).effective_credit
    atomic_rms = float(batch.rate_credit.pow(2).mean().sqrt())
    assert float(matched.pow(2).mean().sqrt()) == pytest.approx(atomic_rms, rel=1e-5)
    raw = ExternalCreditAugmentationTransform(0.5)(batch, training=True).effective_credit
    assert float(raw.pow(2).mean().sqrt()) > 2.0 * atomic_rms


def test_external_operator_ignores_masked_atoms_and_requires_an_advantage():
    rate = draw(5, seed=43)
    advantage = [1.0, 1.0, 1.0, 1.0, 1.0]
    batch = batch_of(
        rate.tolist(), valid=[True, False, True, True, True], advantage=advantage
    )
    applied = ExternalCreditAugmentationTransform(0.5)(batch, training=True).effective_credit
    assert float(applied[1]) == float(rate[1])
    with pytest.raises(RuntimeError):
        ExternalCreditAugmentationTransform(0.5)(batch_of(rate.tolist()), training=True)
    with pytest.raises(ValueError):
        ExternalCreditAugmentationTransform(-0.1)
    with pytest.raises(ValueError):
        ExternalCreditAugmentationTransform(0.1, scale_match="zscore")


def test_external_operator_rejects_misaligned_advantages():
    batch = batch_of(draw(5, seed=47).tolist(), advantage=[1.0, 1.0])
    with pytest.raises(ValueError):
        ExternalCreditAugmentationTransform(0.25)(batch, training=True)


# ---------------------------------------------------------------------------
# Section 10.6 -- the mandatory Atomic regression
# ---------------------------------------------------------------------------


def legacy_atomic_loss(current_nll, base, weight):
    """Historical Atomic loss: the singleton partition of `hard_partition_loss`."""
    partition = tuple((start, start + 1) for start in range(current_nll.numel()))
    return hard_partition_loss(current_nll, base, weight, partition)


@pytest.mark.parametrize("n,seed", [(1, 1), (4, 2), (37, 3), (128, 4)])
def test_identity_operator_equals_the_historical_atomic_loss(n, seed):
    generator = torch.Generator().manual_seed(seed)
    rate = torch.randn(n, generator=generator) * 0.3
    weight = torch.randint(1, 6, (n,), generator=generator).float()
    nll = torch.rand(n, generator=generator) * 4.0
    batch = AtomCreditBatch(
        rate_credit=rate,
        base_credit=rate * weight,
        token_count=weight,
        valid_mask=torch.ones(n, dtype=torch.bool),
        seq_ids=torch.zeros(n, dtype=torch.long),
        atom_positions=torch.arange(n),
    )
    identity = IdentityCreditTransform()(batch, training=True).effective_credit
    operator_loss = (identity * nll).sum()
    reference = legacy_atomic_loss(nll, batch.base_credit, weight)
    assert float(operator_loss) == pytest.approx(float(reference), rel=1e-5, abs=1e-5)


def test_non_identity_operators_change_the_loss_but_not_the_atom_nll():
    rate = draw(12, seed=53)
    weight = torch.ones(12)
    nll = torch.linspace(0.5, 2.0, 12)
    batch = AtomCreditBatch(
        rate_credit=rate,
        base_credit=rate,
        token_count=weight,
        valid_mask=torch.ones(12, dtype=torch.bool),
        seq_ids=torch.zeros(12, dtype=torch.long),
        atom_positions=torch.arange(12),
    )
    atomic = float((batch.rate_credit * nll).sum())
    for transform in (
        ForwardNeighborCreditTransform(0.25),
        BackwardNeighborCreditTransform(0.25),
        ShuffledNeighborCreditTransform(0.25, seed=1),
    ):
        applied = transform(batch, training=True).effective_credit
        assert applied.shape == nll.shape
        assert float((applied * nll).sum()) != pytest.approx(atomic, rel=1e-6)


def test_credit_operators_do_not_inflate_the_credit_scale():
    """Section 3.4: an operator must not win by enlarging credit or gradient magnitude.

    Convex transfers, permutation sources and normalized kernels are all weighted
    means of the same atomic credits, so the mean is preserved and the RMS can only
    shrink -- never grow.
    """
    rate = draw(64, seed=59)
    batch = batch_of(rate.tolist())
    atomic_mean = float(rate.mean())
    atomic_rms = float(rate.pow(2).mean().sqrt())
    for transform in (
        ForwardNeighborCreditTransform(0.25),
        BackwardNeighborCreditTransform(0.25),
        ShuffledNeighborCreditTransform(0.25, seed=3),
        CausalKernelCreditTransform(horizon=4, kind="exponential", decay=0.5),
        CausalKernelCreditTransform(horizon=2, kind="uniform"),
    ):
        applied = transform(batch, training=True).effective_credit
        assert abs(float(applied.mean()) - atomic_mean) <= 0.05 * atomic_rms, transform.name
        assert float(applied.pow(2).mean().sqrt()) <= 1.01 * atomic_rms, transform.name


# ---------------------------------------------------------------------------
# Telemetry contract (spec section 11) and the fail-closed finite check
# ---------------------------------------------------------------------------


DOCUMENTED_METRICS = {
    f"{METRIC_PREFIX}transform_code",
    f"{METRIC_PREFIX}valid_atom_count",
    f"{METRIC_PREFIX}mean_atomic",
    f"{METRIC_PREFIX}std_atomic",
    f"{METRIC_PREFIX}rms_atomic",
    f"{METRIC_PREFIX}mean_effective",
    f"{METRIC_PREFIX}std_effective",
    f"{METRIC_PREFIX}rms_effective",
    f"{METRIC_PREFIX}corr_effective_atomic",
    f"{METRIC_PREFIX}mean_abs_delta",
    f"{METRIC_PREFIX}transfer_fraction",
    f"{METRIC_PREFIX}sign_flip_fraction",
    f"{METRIC_PREFIX}neighbor_product_mean",
    f"{METRIC_PREFIX}neighbor_sign_agreement",
    f"{METRIC_PREFIX}neighbor_pair_count",
}


def all_transforms():
    return [
        IdentityCreditTransform(),
        ForwardNeighborCreditTransform(0.25),
        BackwardNeighborCreditTransform(0.25),
        ShuffledNeighborCreditTransform(0.25, seed=3),
        CausalKernelCreditTransform(horizon=3, kind="uniform", direction="forward"),
        CausalKernelCreditTransform(horizon=3, kind="exponential", decay=0.7, direction="symmetric"),
    ]


def test_every_operator_emits_finite_documented_telemetry():
    for transform in all_transforms():
        batch = batch_of(draw(15, seed=61).tolist())
        out = transform(batch, training=True)
        assert DOCUMENTED_METRICS <= set(out.diagnostics)
        assert all(isinstance(value, torch.Tensor) for value in out.diagnostics.values())
        assert all(not value.requires_grad for value in out.diagnostics.values())
        assert all(torch.isfinite(value).all() for value in out.diagnostics.values()), transform.name
        assert float(out.diagnostics[f"{METRIC_PREFIX}valid_atom_count"]) == 15


@pytest.mark.parametrize(
    "rate,valid",
    [
        ([1.0], [True]),
        ([2.0, 2.0, 2.0], [True, True, True]),
        ([1.0, 2.0], [False, False]),
        ([0.0, 0.0, 0.0, 0.0], [True, True, True, True]),
    ],
)
def test_degenerate_credit_draws_never_emit_nan(rate, valid):
    """The trainer raises on a non-finite metric, so zero variance must report zero."""
    batch = batch_of(rate, valid=valid)
    for transform in all_transforms():
        out = transform(batch, training=True)
        assert all(torch.isfinite(value).all() for value in out.diagnostics.values()), (
            transform.name,
            {key: float(value) for key, value in out.diagnostics.items()},
        )


def test_empty_batch_is_handled_without_nan():
    batch = batch_of([])
    for transform in all_transforms():
        out = transform(batch, training=True)
        assert out.effective_credit.numel() == 0
        assert all(torch.isfinite(value).all() for value in out.diagnostics.values())


def test_neighbor_telemetry_reports_the_local_correlation_it_measures():
    batch = batch_of([1.0, 2.0, 3.0, 4.0])
    out = IdentityCreditTransform()(batch, training=True)
    assert float(out.diagnostics[f"{METRIC_PREFIX}neighbor_product_mean"]) == pytest.approx(
        (1 * 2 + 2 * 3 + 3 * 4) / 3
    )
    assert float(out.diagnostics[f"{METRIC_PREFIX}neighbor_sign_agreement"]) == 1.0
    assert float(out.diagnostics[f"{METRIC_PREFIX}neighbor_pair_count"]) == 3


def test_operator_telemetry_exposes_the_registered_parameters():
    out = CausalKernelCreditTransform(horizon=4, kind="exponential", decay=0.5)(
        batch_of(draw(9, seed=67).tolist()), training=True
    )
    assert float(out.diagnostics[f"{METRIC_PREFIX}horizon"]) == 4.0
    assert float(out.diagnostics[f"{METRIC_PREFIX}decay"]) == 0.5
    # The reported lambda is the realized one-step weight, not the raw decay.
    assert float(out.diagnostics[f"{METRIC_PREFIX}lambda"]) == pytest.approx(0.5 / 1.875)
    assert float(out.diagnostics[f"{METRIC_PREFIX}weight_mass"]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Batch geometry validation
# ---------------------------------------------------------------------------


def test_batch_validation_is_fail_closed():
    rate = torch.zeros(3)
    common = dict(
        rate_credit=rate,
        base_credit=rate.clone(),
        token_count=torch.ones(3),
        valid_mask=torch.ones(3, dtype=torch.bool),
        seq_ids=torch.zeros(3, dtype=torch.long),
        atom_positions=torch.arange(3),
    )
    AtomCreditBatch(**common)
    with pytest.raises(ValueError):
        AtomCreditBatch(**{**common, "token_count": torch.ones(2)})
    with pytest.raises(ValueError):
        AtomCreditBatch(**{**common, "valid_mask": torch.ones(3)})
    with pytest.raises(ValueError):
        AtomCreditBatch(**{**common, "seq_ids": torch.tensor([0, 2, 1])})
    with pytest.raises(ValueError):
        AtomCreditBatch(**{**common, "atom_positions": torch.tensor([0, 2, 1])})
    with pytest.raises(ValueError):
        AtomCreditBatch(**{**common, "rate_credit": torch.zeros(3, 1)})


# ---------------------------------------------------------------------------
# Factory, protocol and argument surface
# ---------------------------------------------------------------------------


def test_factory_covers_every_registered_choice_and_satisfies_the_protocol():
    for name in CREDIT_TRANSFORM_CHOICES:
        spec = build_credit_transform(name, lam=0.25, horizon=3, shuffle_seed=5)
        assert spec.name == name
        assert isinstance(spec.transform, CreditTransform)
        record = spec.invocation_record()
        assert record["credit_transform"] == name
        assert record["credit_transform_code"] == spec.code
    with pytest.raises(ValueError):
        build_credit_transform("pooled")


def test_factory_reads_the_argument_namespace():
    class Args:
        mp_opd_credit_transform = "causal_kernel"
        mp_opd_credit_lambda = 0.1
        mp_opd_credit_convex = True
        mp_opd_credit_horizon = 4
        mp_opd_credit_kernel = "exponential"
        mp_opd_credit_decay = 0.25
        mp_opd_credit_direction = "backward"
        mp_opd_credit_shuffle_seed = 7
        mp_opd_credit_alpha = 0.5
        mp_opd_credit_scale_match = "rms"

    spec = credit_transform_from_args(Args())
    assert spec.name == "causal_kernel"
    assert isinstance(spec.transform, CausalKernelCreditTransform)
    assert spec.transform.direction == "backward"
    assert spec.transform.horizon == 4
    assert spec.parameters["decay"] == 0.25


def test_production_batch_helper_builds_one_valid_sequence():
    """The trainer and the offline probes must share one batch geometry."""
    rate = draw(6, seed=71)
    batch = production_credit_batch(rate, rate.clone(), torch.ones(6))
    assert batch.seq_ids.tolist() == [0] * 6
    assert batch.atom_positions.tolist() == [0, 1, 2, 3, 4, 5]
    assert bool(batch.valid_mask.all())
    assert batch.metadata == {}
    carried = production_credit_batch(rate, rate.clone(), torch.ones(6), {"future_advantage": rate})
    assert torch.equal(carried.metadata["future_advantage"], rate)
    # Boundary fallback still applies at the ends of the single sequence.
    applied = ForwardNeighborCreditTransform(0.5)(batch, training=True).effective_credit
    assert float(applied[-1]) == pytest.approx(float(rate[-1]))


def test_args_declare_the_credit_knobs():
    """Source-level check so it holds in environments without transformers."""
    source = ARGS.read_text()
    tree = _ast.parse(source)
    fields = {}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.AnnAssign) and getattr(node.target, "id", "").startswith(
            "mp_opd_credit"
        ):
            keywords = {keyword.arg: keyword.value for keyword in node.value.keywords}
            fields[node.target.id] = _ast.literal_eval(keywords["default"])
    assert fields == {
        "mp_opd_credit_transform": "identity",
        "mp_opd_credit_lambda": 0.25,
        "mp_opd_credit_convex": True,
        "mp_opd_credit_horizon": 2,
        "mp_opd_credit_kernel": "uniform",
        "mp_opd_credit_decay": 0.5,
        "mp_opd_credit_direction": "forward",
        "mp_opd_credit_shuffle_seed": 43,
        "mp_opd_credit_alpha": 0.25,
        "mp_opd_credit_scale_match": "raw",
    }
    assert '"kernel"' in source
    assert "unsupported mp_opd_credit_transform" in source
    assert "mp_opd_credit_lambda must be in [0, 1]" in source


def test_args_dataclass_accepts_the_kernel_mode_when_transformers_is_present():
    pytest.importorskip("transformers")
    from kdflow.arguments.distillation_args import DistillationArguments

    defaults = DistillationArguments.__dataclass_fields__
    assert defaults["mp_opd_credit_transform"].default == "identity"
    assert defaults["mp_opd_credit_lambda"].default == 0.25
    assert defaults["mp_opd_credit_direction"].default == "forward"
    accepted = dict(
        kd_algorithm="mp_opd",
        mp_opd_mode="kernel",
        mp_opd_credit_transform="forward",
        mp_opd_credit_lambda=0.25,
    )
    DistillationArguments(**accepted)
    for overrides in (
        {"mp_opd_credit_lambda": 1.5},
        {"mp_opd_credit_lambda": -0.1},
        {"mp_opd_credit_transform": "pooled"},
        {"mp_opd_credit_horizon": 0},
        {"mp_opd_credit_kernel": "triangular"},
        {"mp_opd_credit_direction": "sideways"},
        {"mp_opd_credit_decay": 0.0},
        {"mp_opd_credit_scale_match": "zscore"},
        {"mp_opd_credit_alpha": -1.0},
    ):
        with pytest.raises(ValueError):
            DistillationArguments(**{**accepted, **overrides})
    with pytest.raises(ValueError):
        # Atomic is the identity operator by definition.
        DistillationArguments(
            kd_algorithm="mp_opd", mp_opd_mode="atomic", mp_opd_credit_transform="forward"
        )
    with pytest.raises(ValueError):
        DistillationArguments(kd_algorithm="mp_opd", mp_opd_mode="pooled")
