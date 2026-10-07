"""AIRS: per-atom risk shrinkage, no span, no Gram, no DP.

Two families of test live here. The first pins the closed form of the rule against
hand-computed values, so a sign or a threshold cannot drift silently. The second
pins the *absences*: AIRS must not form spans, must not mix neighbouring credits
and must not restore the credit mass it removed. Those are exactly the properties
that would regress quietly, because adding any of them still produces a decreasing
loss - it just stops being the method.

The unequal-precision cases are load-bearing. A rule implemented over uniform
weights collapses to a single global scale, so a bug that only shows up when the
weights differ would survive any test built on equal weights alone.
"""
import ast as _ast
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from kdflow.algorithms._mp_opd_airs import (
    AIRS_EPS_NOISE,
    AIRS_EPS_W,
    airs_metrics,
    airs_precision_buckets,
    airs_shrinkage,
)

MODE = "mp_opd_mode"
SOURCE = ROOT / "kdflow" / "algorithms" / "mp_opd.py"


def _function_body(name: str, *, drop_docstrings: bool = True) -> str:
    """AST dump of one function body.

    A missing function has to be a named failure, not a StopIteration: a silently
    skipped structural assertion is exactly the kind of green test that hides a
    missing implementation.

    Docstrings are dropped by default. This module's own prose explains that AIRS
    uses no Gram and no chunk, so a naive substring scan over the body would match
    its own documentation and fail a correct implementation.
    """
    tree = _ast.parse(SOURCE.read_text())
    for node in _ast.walk(tree):
        if isinstance(node, _ast.FunctionDef) and node.name == name:
            body = list(node.body)
            if drop_docstrings and (
                body
                and isinstance(body[0], _ast.Expr)
                and isinstance(body[0].value, _ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            return _ast.dump(_ast.Module(body=body, type_ignores=[]))
    raise AssertionError(f"{name} is not defined in {SOURCE}")


def _credits(spread: float = 1.0, *, weight_spread: float = 0.0, seed: int = 901):
    """Credits with optionally unequal precisions.

    ``weight_spread > 0`` gives each atom its own precision, which is the only way
    to see the ``sigma2 / w_i`` dependence at all: with equal weights every atom
    shares one variance and the per-atom rule degenerates to a global scale.
    """
    generator = torch.Generator().manual_seed(seed)
    n = 12
    rate = spread * torch.randn(n, generator=generator, dtype=torch.float64)
    if weight_spread:
        weight = 1.0 + weight_spread * torch.rand(
            n, generator=generator, dtype=torch.float64
        )
    else:
        weight = torch.ones(n, dtype=torch.float64)
    return rate, weight


def test_lambda_matches_the_closed_form():
    # r = 4, w = 2, sigma2 = 8  =>  sigma_i^2 = 4, lambda = 1 - 4/16 = 0.75
    rate = torch.tensor([4.0], dtype=torch.float64)
    weight = torch.tensor([2.0], dtype=torch.float64)
    out = airs_shrinkage(rate, weight, 8.0)
    assert float(out["lambda"][0]) == pytest.approx(0.75, rel=1e-12)
    assert float(out["credit"][0]) == pytest.approx(3.0, rel=1e-12)


def test_snr_regimes_follow_the_specification():
    # sigma2 = 1.0, w = 1.0 => sigma_i^2 = 1.
    # SNR <= 1 kills, SNR = 2 gives 0.5, large SNR approaches 1.
    rate = torch.tensor([1.0, math.sqrt(2.0), 10.0], dtype=torch.float64)
    weight = torch.ones(3, dtype=torch.float64)
    out = airs_shrinkage(rate, weight, 1.0)
    lam = out["lambda"].tolist()
    assert lam[0] == pytest.approx(0.0, abs=1e-12)
    assert lam[1] == pytest.approx(0.5, rel=1e-12)
    # r = 10 with sigma_i^2 = 1 is SNR 100, so lambda is 0.99 exactly - it approaches 1
    # but never reaches it, and it must never exceed 1 either.
    assert lam[2] == pytest.approx(0.99, rel=1e-12)
    assert 0.99 < 1.0
    snr = out["snr"].tolist()
    assert snr[0] == pytest.approx(1.0, rel=1e-12)
    assert snr[1] == pytest.approx(2.0, rel=1e-12)
    assert snr[2] == pytest.approx(100.0, rel=1e-12)


def test_zero_credit_is_neutral_rather_than_division_by_zero():
    # The rule is defined as a branch, so r = 0 must land on lambda = 0 instead of
    # producing a NaN the way 1 - sigma2 / r**2 would.
    rate = torch.tensor([0.0, 1.0], dtype=torch.float64)
    weight = torch.tensor([1.0, 1.0], dtype=torch.float64)
    out = airs_shrinkage(rate, weight, 2.0)
    assert float(out["credit"][0]) == 0.0
    assert math.isfinite(float(out["credit"][0]))
    assert torch.isfinite(out["lambda"]).all()
    assert torch.isfinite(out["snr"]).all()


def test_lambda_is_bounded_on_both_sides():
    rate, weight = _credits(3.0, weight_spread=2.0)
    out = airs_shrinkage(rate, weight, 1e6)
    lam = out["lambda"]
    assert float(lam.min()) >= 0.0
    assert float(lam.max()) <= 1.0
    assert torch.isfinite(out["credit"]).all()


def test_uncalibrated_noise_keeps_every_atom_at_full_strength():
    # sigma2 = 0 means the estimator has not identified a scale yet. Every atom
    # must stay at lambda = 1: a small but nonzero estimate during warm-up would
    # otherwise suppress most of the run.
    rate, weight = _credits(0.1)
    out = airs_shrinkage(rate, weight, 0.0, enabled=True)
    assert torch.equal(out["lambda"], torch.ones_like(out["lambda"]))


def test_warmup_disables_shrinkage_entirely():
    rate, weight = _credits(0.05)
    on = airs_shrinkage(rate, weight, 1.0, enabled=True)
    off = airs_shrinkage(rate, weight, 1.0, enabled=False)
    assert torch.equal(off["lambda"], torch.ones_like(off["lambda"]))
    assert torch.allclose(off["credit"], rate)
    # The same inputs with shrinkage on must actually differ, otherwise the warm-up
    # test would pass for a rule that never suppresses anything.
    assert not torch.allclose(on["lambda"], torch.ones_like(on["lambda"]))


def test_shrinkage_is_independent_across_atoms():
    # Shrinking one atom must not move any other atom's coefficient: that is the
    # whole difference between an independent per-atom rule and a span/DP rule.
    rate = torch.tensor([5.0, 1.0, 4.0, 2.0], dtype=torch.float64)
    weight = torch.ones(4, dtype=torch.float64)
    base = airs_shrinkage(rate, weight, 1.0)["lambda"]
    for index in range(4):
        perturbed = rate.clone()
        perturbed[index] = perturbed[index] * 1.7 + 3.0
        moved = airs_shrinkage(perturbed, weight, 1.0)["lambda"]
        others = [i for i in range(4) if i != index]
        assert torch.equal(base[others], moved[others])


def test_unequal_precision_changes_the_coefficient_per_atom():
    # Same credit, different precision: the higher-precision atom must keep more of
    # it. A rule that ignored w would pass every equal-weight test and be wrong.
    rate = torch.tensor([3.0, 3.0], dtype=torch.float64)
    weight = torch.tensor([1.0, 100.0], dtype=torch.float64)
    out = airs_shrinkage(rate, weight, 1.0)
    assert float(out["lambda"][1]) > float(out["lambda"][0])
    assert float(out["snr"][1]) > float(out["snr"][0])


def test_weight_is_floored_before_inversion():
    # A zero or negative precision would make 1/w explode; the floor is applied
    # before the division, not after.
    rate = torch.tensor([1.0, 1.0], dtype=torch.float64)
    weight = torch.tensor([0.0, -1.0], dtype=torch.float64)
    out = airs_shrinkage(rate, weight, 1.0)
    assert torch.isfinite(out["credit"]).all()
    assert torch.isfinite(out["snr"]).all()
    assert float(out["weight"][0]) == pytest.approx(AIRS_EPS_W, rel=1e-12)
    assert float(out["weight"][1]) == pytest.approx(AIRS_EPS_W, rel=1e-12)


def test_eps_noise_guards_a_tiny_but_positive_scale():
    rate = torch.tensor([1e-3], dtype=torch.float64)
    weight = torch.tensor([1.0], dtype=torch.float64)
    out = airs_shrinkage(rate, weight, AIRS_EPS_NOISE / 10.0)
    assert float(out["lambda"][0]) == 1.0


def test_credit_mass_is_reduced_and_never_restored():
    # Shrinking toward zero lowers the weighted mass on purpose. If any downstream
    # step renormalised it back, the suppression would be undone.
    rate, weight = _credits(0.5, weight_spread=1.0)
    out = airs_shrinkage(rate, weight, 0.5)
    before = (weight * rate).abs().sum()
    after = (weight * out["credit"]).abs().sum()
    assert float(after) < float(before)


def test_metrics_report_the_effect_rather_than_claiming_success():
    rate, weight = _credits(0.5, weight_spread=1.0)
    out = airs_shrinkage(rate, weight, 0.5)
    metrics = airs_metrics(rate, weight, out, enabled=True)
    assert float(metrics["mp_opd_airs_credit_rms_ratio"]) < 1.0
    assert float(metrics["mp_opd_airs_kill_fraction"]) > 0.0
    assert 0.0 < float(metrics["mp_opd_airs_lambda_mean"]) <= 1.0
    assert float(metrics["mp_opd_airs_enabled"]) == 1.0
    for key, value in metrics.items():
        assert torch.isfinite(torch.as_tensor(value)).all(), f"{key} was {value}"


def test_warmup_is_visible_in_the_metrics():
    # An all-ones lambda must be identifiable as warm-up, otherwise a run that never
    # leaves warm-up looks like a healthy run whose shrinkage had no effect.
    rate, weight = _credits(0.5, weight_spread=1.0)
    out = airs_shrinkage(rate, weight, 0.5, enabled=False)
    metrics = airs_metrics(rate, weight, out, enabled=False)
    assert float(metrics["mp_opd_airs_enabled"]) == 0.0
    assert float(metrics["mp_opd_airs_lambda_mean"]) == 1.0
    assert float(metrics["mp_opd_airs_credit_rms_ratio"]) == pytest.approx(
        1.0, rel=1e-9
    )


def test_sign_symmetry_of_the_coefficient():
    # The rule depends on r^2, so equal-magnitude opposite credits must receive the
    # same lambda. A sign leak would show up here and nowhere else.
    rate = torch.tensor([4.0, -4.0, 2.0, -2.0], dtype=torch.float64)
    weight = torch.ones(4, dtype=torch.float64)
    lam = airs_shrinkage(rate, weight, 1.0)["lambda"]
    assert float(lam[0]) == pytest.approx(float(lam[1]), rel=1e-12)
    assert float(lam[2]) == pytest.approx(float(lam[3]), rel=1e-12)


def test_precision_buckets_separate_low_from_high_precision():
    rate = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    weight = torch.tensor([0.01, 0.1, 10.0, 100.0], dtype=torch.float64)
    out = airs_shrinkage(rate, weight, 1.0)
    buckets = airs_precision_buckets(weight, out, buckets=2)
    assert all(torch.isfinite(v).all() for v in buckets.values())
    low = float(buckets["mp_opd_airs_bucket0_lambda_mean"])
    high = float(buckets["mp_opd_airs_bucket1_lambda_mean"])
    assert high > low


def test_non_finite_rate_is_rejected():
    with pytest.raises(ValueError):
        airs_shrinkage(torch.tensor([1.0, float("nan")]), torch.ones(2), 1.0)
    with pytest.raises(ValueError):
        airs_shrinkage(torch.tensor([1.0, 1.0]), torch.ones(2), -1.0)
    with pytest.raises(ValueError):
        airs_shrinkage(torch.tensor([1.0, 1.0]), torch.ones(3), 1.0)


def test_mode_is_registered_in_the_argument_validation():
    from kdflow.arguments.distillation_args import DistillationArguments

    # __post_init__ is where this dataclass validates; there is no validate() method,
    # so constructing the arguments is the assertion.
    args = DistillationArguments(kd_algorithm="mp_opd", mp_opd_mode="airs")
    assert args.mp_opd_mode == "airs"
    assert args.mp_opd_credit_transform == "identity"


def test_airs_rejects_a_cross_atom_credit_operator():
    # AIRS shrinks each atom on its own; composing it with a cross-atom operator
    # would change which atoms are mixed, so the combination must fail closed rather
    # than run something the recipe does not name. "causal_kernel" is the real
    # cross-atom operator in the advertised choices - "kernel" is a mode, not a
    # transform, and would fail for an unrelated reason.
    from kdflow.arguments.distillation_args import DistillationArguments

    with pytest.raises(ValueError, match="airs"):
        DistillationArguments(
            kd_algorithm="mp_opd",
            mp_opd_mode="airs",
            mp_opd_credit_transform="causal_kernel",
        )


def test_airs_rejects_an_out_of_range_ema_rate():
    from kdflow.arguments.distillation_args import DistillationArguments

    with pytest.raises(ValueError, match="mp_opd_airs_sigma_rho"):
        DistillationArguments(
            kd_algorithm="mp_opd", mp_opd_mode="airs", mp_opd_airs_sigma_rho=1.0
        )


def test_loss_path_never_builds_a_gram_or_a_span():
    # Structural, because a numeric test cannot tell an independent per-atom rule
    # from one that quietly pools neighbours and happens to give the same answer on
    # the chosen inputs.
    body = _function_body("_airs_loss")
    for forbidden in ("grass_span_costs", "grass_partition", "chunk_", "gram", "dpca"):
        assert forbidden not in body, f"_airs_loss must not reference {forbidden}"
    assert "airs_shrinkage" in body
    assert "soft_partition_loss" in body


def test_airs_needs_no_student_logits_or_hidden_states():
    # AIRS is the one shrinkage mode that requires neither, because the coefficient
    # depends only on credit and precision. Keeping that true is what makes the mode
    # cheap, so it is asserted rather than left to drift.
    tree = _ast.parse(SOURCE.read_text())
    grass_modes = next(
        node
        for node in _ast.walk(tree)
        if isinstance(node, _ast.Assign)
        and any(getattr(t, "id", None) == "_GRASS_MODES" for t in node.targets)
    )
    assert "airs" not in _ast.dump(grass_modes)


def test_warmup_state_survives_a_resume():
    # The noise estimator and the warm-up position are both training-level. Losing
    # the estimator restarts the EMA from a deflated variance. The warm-up position
    # is read from student_updates, which the checkpoint already carries, so a resume
    # cannot silently replay the warm-up.
    source = SOURCE.read_text()
    assert "mp_opd_airs_noise" in source
    tree = _ast.parse(source)
    body = _function_body("_airs_loss")
    assert "student_updates" in body
    assert "airs_updates" not in body, (
        "the warm-up must be measured in optimizer updates, not a per-micro-batch "
        "counter that grad accumulation would advance several times per step"
    )


def test_no_airs_specific_renormalization_in_the_loss():
    # The suppression is the method. Any division that restores the original credit
    # mass would cancel it, so the loss must contain no such correction.
    body = _function_body("_airs_loss")
    for forbidden in ("credit_conservation_residual", "renormal", "rescale"):
        assert forbidden not in body
