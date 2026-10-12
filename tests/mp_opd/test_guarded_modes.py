"""Independent gradient oracles and known counterexamples for new modes."""
import math
from types import SimpleNamespace

import pytest
import torch

from kdflow.algorithms._mp_opd_align import align_chunks, align_metrics
from kdflow.algorithms._mp_opd_align_new import align_new_guard
from kdflow.algorithms._mp_opd_head_pair import head_pair_geometry
from kdflow.algorithms._mp_opd_trust import trust_calibrate_scalars
from kdflow.algorithms._mp_opd_trust_batch import TrustHeadAccumulator
from kdflow.algorithms._mp_opd_trust_p import (
    trust_p_calibrate_scalars, trust_p_response_loss,
)
from test_numeric_fallback import _make_algo


@pytest.mark.parametrize("softcap", [None, 2.])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("tile", [1, 4, 20])
def test_full_response_pair_matches_autograd_including_cross_chunk(softcap, bias, tile):
    torch.manual_seed(156)
    h = torch.randn(6, 3)
    w = torch.randn(7, 3, requires_grad=True)
    b = torch.randn(7, requires_grad=True)
    z = h @ w.T + (b if bias else 0)
    if softcap is not None:
        z = softcap * torch.tanh(z/softcap)
    labels = torch.tensor([0, 3, 6, 1, 4, 2])
    lp = z.log_softmax(-1)[torch.arange(6), labels]
    # Deliberate tail: only the covered prefix belongs to the atom loss.
    ranges = [(0, 1), (1, 3), (3, 5)]
    r, e = torch.tensor([1., -.4, .8]), torch.tensor([.3, .7, -.2])
    token_r = r.repeat_interleave(torch.tensor([1, 2, 2]))
    token_e = e.repeat_interleave(torch.tensor([1, 2, 2]))
    params = (w, b) if bias else (w,)
    ga = torch.autograd.grad(-(token_r*lp[:5]).sum(), params, retain_graph=True)
    gb = torch.autograd.grad(-(token_e*lp[:5]).sum(), params)
    actual = head_pair_geometry(z, h, labels, ranges, r, e,
                               selected_log_prob=lp.detach(), softcap=softcap,
                               head_bias=bias, vocab_chunk=tile)
    expected = [sum(float(g.double().square().sum()) for g in ga),
                sum(float(g.double().square().sum()) for g in gb),
                sum(float((a.double()*b.double()).sum()) for a, b in zip(ga, gb))]
    assert [actual.first_norm2, actual.second_norm2, actual.dot] == pytest.approx(expected, rel=2e-6, abs=2e-6)


def test_geometry_is_not_changed_by_outer_autocast():
    torch.manual_seed(7)
    z, h = torch.randn(4, 6), torch.randn(4, 3)
    labels = torch.tensor([1, 2, 3, 4])
    lp = z.log_softmax(-1)[torch.arange(4), labels]
    args = (z, h, labels, [(0, 2), (2, 4)], torch.tensor([.2, 1.]), torch.tensor([1., -.4]))
    base = head_pair_geometry(*args, selected_log_prob=lp)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = head_pair_geometry(*args, selected_log_prob=lp)
    assert actual == base


def test_run_length_two_block_cosine_hid_reversed_total_update():
    gradients = torch.tensor([[1., 0.], [-1., 1.], [-.4, -1.]])
    r = torch.ones(3)
    full = gradients @ gradients.T
    block = full.clone()
    block[:2, 2] = 0; block[2, :2] = 0
    stats = align_chunks(r, block, [(0, 2), (2, 3)])
    candidate = stats["rate"]
    assert candidate.tolist() == pytest.approx([2., 1.5, 1.])
    old = align_metrics(r, candidate, block, stats)
    assert float(old["mp_opd_align_head_update_cosine"]) == pytest.approx(.9460502863)
    g0, g1 = r @ gradients, candidate @ gradients
    assert float(g0 @ g1) == pytest.approx(-.04, abs=1e-7)
    safe, metrics = align_new_guard(r, candidate, float(g0 @ g0), float(g1 @ g1), float(g0 @ g1))
    actual = safe @ gradients
    assert float(metrics["mp_opd_align_new_beta"]) == pytest.approx(.72, abs=1e-6)
    assert float(g0 @ actual) >= .1*float(g0 @ g0)-1e-7
    assert float(metrics["mp_opd_align_new_candidate_head_update_cosine"]) < 0
    assert float(metrics["mp_opd_align_new_retained_descent_fraction"]) >= .1
    assert float(metrics["mp_opd_align_new_head_update_cosine"]) == pytest.approx(
        float(g0 @ actual)/(float(g0.norm())*float(actual.norm())), abs=1e-6)


def test_three_atom_sequential_projection_also_needs_guard():
    gradients = torch.tensor([[-2., -2.], [0., 1.], [1., 0.]])
    r = torch.ones(3); gram = gradients @ gradients.T
    candidate = align_chunks(r, gram, [(0, 3)])["rate"]
    g0, g1 = r @ gradients, candidate @ gradients
    assert float(g0 @ g1) == pytest.approx(-.5)
    safe, _ = align_new_guard(r, candidate, float(g0 @ g0), float(g1 @ g1), float(g0 @ g1))
    assert float(g0 @ (safe @ gradients)) >= .1*float(g0 @ g0)-1e-6


@pytest.mark.parametrize("rho", [0., .1, .5, 1.])
def test_align_guard_matches_independent_vector_constraint(rho):
    generator = torch.Generator().manual_seed(19)
    for _ in range(30):
        g = torch.randn(5, 8, generator=generator)
        r, candidate = torch.randn(5, generator=generator), torch.randn(5, generator=generator)
        g0, g1 = r @ g, candidate @ g
        out, metrics = align_new_guard(r, candidate, float(g0.double().square().sum()),
                                      float(g1.double().square().sum()), float(g0.double() @ g1.double()),
                                      min_descent_fraction=rho)
        assert float(g0 @ (out @ g)) >= rho*float(g0 @ g0)-2e-5
        assert 0 <= float(metrics["mp_opd_align_new_beta"]) <= 1


def test_align_zero_anchor_returns_atomic_and_detaches_guard():
    r = torch.tensor([0., 0.], requires_grad=True)
    candidate = torch.tensor([1., -1.], requires_grad=True)
    out, metrics = align_new_guard(r, candidate, 0., 2., 0.)
    assert torch.equal(out, r) and not out.requires_grad
    assert float(metrics["mp_opd_align_new_no_anchor_fraction"]) == 1


@pytest.mark.parametrize("mismatch", [[0., 1.], [-.1, 1.], [1., 10.], [-2., 0.]])
def test_trust_p_retains_orthogonal_signal_and_protects_strict(mismatch):
    s, m = torch.tensor([1., 0.]), torch.tensor(mismatch)
    result = trust_p_calibrate_scalars(float(s @ s), float(m @ m), float(s @ m), 1, 1)
    safe_m = m + (result.strict_scale-1)*s
    update = s + safe_m
    # Orthogonal signal is unchanged, even if a small opposing component exists.
    assert float(safe_m[1]) == pytest.approx(float(m[1]))
    assert float(s @ safe_m) >= -1e-6
    assert float(s @ update) >= float(s @ s)-1e-6
    assert float(safe_m.norm()) <= float(m.norm())+1e-6
    assert result.strict_descent == pytest.approx(float(s @ update))
    if mismatch in ([0., 1.], [-.1, 1.]):
        native = trust_calibrate_scalars(float(s @ s), float(m @ m), float(s @ m), 1, 1)
        assert native.lam == 0  # Existing mode deliberately remains unchanged.
        assert float(update[1]) == 1


@pytest.mark.parametrize("gs2,n_s", [(0., 0), (0., 2), (1e-14, 2)])
def test_trust_p_no_anchor_keeps_atomic(gs2, n_s):
    result = trust_p_calibrate_scalars(gs2, 1., 0., n_s, 2)
    assert result.no_anchor and result.strict_scale == 1
    rate = torch.tensor([.7, -.2], requires_grad=True)
    nll = torch.tensor([2., 3.], requires_grad=True)
    loss = trust_p_response_loss(nll, rate, torch.tensor([True, False]), torch.tensor([False, True]), result)
    grad = torch.autograd.grad(loss, (nll, rate), allow_unused=True)
    assert grad[0].tolist() == pytest.approx(rate.tolist()) and grad[1] is None


@pytest.mark.parametrize("values", [(-1., 1., 0.), (1., 1., 2.), (math.nan, 1., 0.)])
def test_new_calibration_rejects_invalid_geometry(values):
    with pytest.raises((ValueError, FloatingPointError)):
        trust_p_calibrate_scalars(*values, 1, 1)
    with pytest.raises((ValueError, FloatingPointError)):
        align_new_guard(torch.ones(2), torch.zeros(2), *values)


def test_align_new_actual_loss_gradient_equals_guarded_credit(monkeypatch):
    algo = _make_algo("align_new")
    # Use a manufactured PSD counterexample to test dispatch and loss separately
    # from head geometry, which has its own independent autograd oracle above.
    g = torch.tensor([[1., 0.], [-1., 1.], [-.4, -1.]])
    full = g @ g.T; block = full.clone(); block[:2, 2] = 0; block[2, :2] = 0
    import kdflow.algorithms.mp_opd as module
    monkeypatch.setattr(module, "chunk_head_gram", lambda *a, **k:
                        SimpleNamespace(gram=block, symmetry_error=0., token_count=3))
    def pair(*a, **k):
        r, e = a[4:6]; a0, a1 = r @ g, e @ g
        return SimpleNamespace(first_norm2=float(a0 @ a0), second_norm2=float(a1 @ a1), dot=float(a0 @ a1))
    monkeypatch.setattr(module, "head_pair_geometry", pair)
    nll = torch.ones(3, requires_grad=True)
    credits = SimpleNamespace(rate=torch.ones(3), weight=torch.ones(3), base_credit=torch.ones(3),
                              current_nll=nll, student_token_nll=torch.ones(3))
    atoms = [SimpleNamespace(student_start=i, student_end=i+1, student_token_count=1) for i in range(3)]
    loss, metrics = algo._partition_loss(credits, atoms, {}, 0,
                                        student_logits=torch.zeros(3, 4), student_labels=torch.ones(3, dtype=torch.long),
                                        student_hidden=torch.zeros(3, 2))
    actual_rate = torch.autograd.grad(loss, nll)[0]
    g0 = g.sum(0)
    assert float(g0 @ (actual_rate @ g)) >= .1*float(g0 @ g0)-1e-7
    assert "mp_opd_align_head_update_cosine" not in metrics
    assert "mp_opd_align_new_local_head_update_cosine" in metrics
    assert float(metrics["mp_opd_numeric_fallback_fraction"]) == 0


def test_align_new_missing_geometry_fails_closed():
    algo = _make_algo("align_new")
    credits = SimpleNamespace(rate=torch.ones(1), weight=torch.ones(1), base_credit=torch.ones(1),
                              current_nll=torch.ones(1), student_token_nll=torch.ones(1))
    with pytest.raises(RuntimeError, match="requires the student logits"):
        algo._partition_loss(credits, [SimpleNamespace()], {}, 0)


@pytest.mark.parametrize("mode", ["align_new", "trust_p"])
def test_new_modes_reject_disabled_exact_geometry(monkeypatch, mode):
    monkeypatch.setenv("MP_OPD_GRASS_EXACT_FP32", "0")
    with pytest.raises(ValueError, match="exact fp32"):
        _make_algo(mode)


@pytest.mark.parametrize("mode", ["align_new", "trust_p"])
def test_new_modes_resume_contract_is_specific_and_checked(mode):
    algo = _make_algo(mode)
    algo.student_updates = 80
    state = algo.training_state_dict()
    restored = _make_algo(mode)
    restored.load_training_state_dict(state)
    assert restored.student_updates == 80
    other = _make_algo("trust_p" if mode == "align_new" else "align_new")
    with pytest.raises(ValueError, match="resume contract"):
        other.load_training_state_dict(state)
    del state["mp_opd_guarded_mode"]
    with pytest.raises(ValueError, match="resume contract"):
        restored.load_training_state_dict(state)


def test_align_new_resume_rejects_changed_guard_recipe():
    algo = _make_algo("align_new")
    state = algo.training_state_dict()
    algo.align_new_min_descent_fraction = 1.
    with pytest.raises(ValueError, match="resume contract"):
        algo.load_training_state_dict(state)


def test_projected_trust_cannot_checkpoint_pending_calibration():
    algo = _make_algo("trust_p")
    algo._trust_batch_result = trust_p_calibrate_scalars(1., 1., 0., 1, 1)
    with pytest.raises(RuntimeError, match="optimizer boundaries"):
        algo.training_state_dict()


def test_projected_batch_accumulator_matches_independent_strict_mismatch_autograd():
    torch.manual_seed(18)
    h = torch.randn(4, 3)
    w = torch.randn(5, 3, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3]); z = h @ w.T
    lp = z.log_softmax(-1)[torch.arange(4), labels]
    rate = torch.tensor([-.5, .2]); strict = torch.tensor([True, False])
    s = torch.autograd.grad(-rate[0]*lp[:2].sum(), w, retain_graph=True)[0]
    m = torch.autograd.grad(-rate[1]*lp[2:].sum(), w)[0]
    acc = TrustHeadAccumulator(vocab_chunk=2, force_fp32=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        acc.add_response(z, h, labels, [(0, 2), (2, 4)], rate, strict, lp.detach())
    actual = acc.finalize(calibrate=trust_p_calibrate_scalars)
    expected = trust_p_calibrate_scalars(float(s.double().square().sum()),
                                        float(m.double().square().sum()), float(s.double().flatten() @ m.double().flatten()), 1, 1)
    assert actual.strict_scale == pytest.approx(expected.strict_scale, abs=1e-6, rel=1e-6)
    assert actual.strict_descent == pytest.approx(expected.strict_descent, abs=1e-6, rel=1e-6)


def test_projected_trust_metrics_include_actual_fp32_coefficient_rounding():
    # Tiny opposing M: rounding a upwards adds an epsilon-sized parallel
    # component. Relative M ratios must not pretend this is exact zero.
    result = trust_p_calibrate_scalars(1., 1e-20, -1e-10, 1, 1)
    actual_m = -1e-10 + result.strict_scale-1
    assert result.mismatch_retained_norm_ratio == pytest.approx(abs(actual_m)/1e-10)
    assert result.strict_descent == pytest.approx(result.strict_scale-1e-10)
