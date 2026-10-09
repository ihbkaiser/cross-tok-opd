"""Temporal pooling oracles, including the production B64/M4 actor fit body."""
import copy
import json
import os
from types import SimpleNamespace

import pytest
import torch

from kdflow.algorithms._mp_opd_atoms import MPAtom, AtomizationResult
from kdflow.algorithms._mp_opd_credit import build_atom_credits
from kdflow.algorithms._mp_opd_grass_chunk import (
    apply_chunk_shrinkage, grass_chunk_tables, scale_chunk_shrinkage,
    temporal_pooling_scale,
)
from kdflow.algorithms._mp_opd_grass_span import GrassNoiseEstimator
from kdflow.training_checkpoint import capture_rng, restore_rng
from test_numeric_fallback import _make_algo
from test_trust_batch_replay import _fit_body, _setup


@pytest.mark.parametrize("update,expected", [(1, 0), (19, 0), (20, 0),
    (21, 1/140), (90, .5), (159, 139/140), (160, 1), (240, 1)])
def test_schedule_uses_one_based_optimizer_update(update, expected):
    assert temporal_pooling_scale(update, 20, 160) == pytest.approx(expected)


@pytest.mark.parametrize("update,start,end", [(0, 20, 160), (1, -1, 160),
                                             (1, 20, 20), (1, 160, 20)])
def test_invalid_schedule_fails_closed(update, start, end):
    with pytest.raises(ValueError):
        temporal_pooling_scale(update, start, end)


@pytest.mark.parametrize("scale", [0., .5, 1.])
def test_scaled_tables_match_formula_and_conserve_weighted_credit(scale):
    r = torch.tensor([2., -1., 4.], dtype=torch.float64)
    w = torch.tensor([1., 2., 3.], dtype=torch.float64)
    raw = grass_chunk_tables(r, w, torch.eye(3), [(0, 2), (2, 3)], 1.)
    original_alpha = raw.alpha.clone()
    effective = scale_chunk_shrinkage(raw, scale)
    result, residual = apply_chunk_shrinkage(r, w, effective)
    # Weighted mean of the first two atoms is zero, the singleton is unchanged.
    expected = r.clone()
    expected[:2] *= 1-scale*raw.alpha[0]
    assert torch.allclose(result, expected)
    assert torch.equal(raw.alpha, original_alpha)
    assert effective.alpha[1] == 0
    assert residual < 1e-12
    assert torch.allclose(effective.gain,
        2*effective.alpha*raw.variance-effective.alpha.square()*raw.distortion)
    if scale == 0:
        assert torch.equal(result, r)
        assert torch.equal(effective.gain, torch.zeros_like(effective.gain))


def test_atomic_warmup_still_updates_noise_ema():
    algo = _make_algo("grass_chunk_temporal")
    algo.grass_noise = GrassNoiseEstimator(min_adjacent_pairs=3)
    logits = torch.randn(6, 4)
    labels = torch.arange(6) % 4
    nll = -logits.log_softmax(-1)[torch.arange(6), labels]
    current = nll.clone().requires_grad_()
    credits = SimpleNamespace(rate=torch.tensor([0., 1., 0., 4., 0., 9.]),
        weight=torch.ones(6), current_nll=current, student_token_nll=nll)
    atoms = tuple(MPAtom("test", i, i+1, i, i+1, i, i+1, 1, 1) for i in range(6))
    loss, metrics = algo._grass_chunk_loss(credits, atoms, student_logits=logits,
        student_labels=labels, student_hidden=torch.randn(6, 2))
    assert algo.grass_noise.state_dict()["updates"] == 1
    assert algo.grass_noise.sigma2 > 0
    assert metrics["mp_opd_grass_chunk_temporal_scale"] == 0
    assert metrics["mp_opd_grass_chunk_raw_alpha_mean"] > 0
    assert metrics["mp_opd_grass_chunk_effective_alpha_mean"] == 0
    assert metrics["mp_opd_grass_chunk_sure_gain_mean"] == 0
    assert torch.equal(torch.autograd.grad(loss, current)[0], credits.rate)


def test_resume_keeps_progress_noise_and_rejects_recipe_drift():
    algo = _make_algo("grass_chunk_temporal")
    algo.note_optimizer_updates(89)
    algo.grass_noise = GrassNoiseEstimator(initial_variance=.7)
    state = copy.deepcopy(algo.training_state_dict())
    resumed = _make_algo("grass_chunk_temporal")
    resumed.load_training_state_dict(state)
    assert resumed.student_updates == 89
    assert resumed.grass_noise.sigma2 == pytest.approx(.7)
    assert temporal_pooling_scale(resumed.student_updates+1,
        resumed.grass_temporal_start, resumed.grass_temporal_end) == .5
    resumed.grass_temporal_start = 40
    with pytest.raises(ValueError, match="resume contract"):
        resumed.load_training_state_dict(state)
    resumed.grass_temporal_start = 20
    resumed.grass_chunk_run_length = 3
    with pytest.raises(ValueError, match="resume contract"):
        resumed.load_training_state_dict(state)
    missing_noise = copy.deepcopy(state)
    del missing_noise["mp_opd_grass_noise"]
    with pytest.raises(ValueError, match="saved noise state"):
        _make_algo("grass_chunk_temporal").load_training_state_dict(missing_noise)
    del state["mp_opd_grass_chunk_temporal"]
    with pytest.raises(ValueError, match="resume contract"):
        _make_algo("grass_chunk_temporal").load_training_state_dict(state)


def test_temporal_actor_requires_exactly_one_accumulation_window():
    algo = _make_algo("grass_chunk_temporal")
    algo.strategy.accumulated_gradient = 16
    assert algo.prepare_optimizer_batch([None]*16) is None
    with pytest.raises(ValueError, match="complete optimizer"):
        algo.prepare_optimizer_batch([None]*32)
    algo.strategy.step = 1
    with pytest.raises(ValueError, match="complete optimizer"):
        algo.prepare_optimizer_batch([None]*16)


def _oracle_rates(credits, gram, run_length, scale, sigma2):
    """Direct scalar D/V formula; does not call GRASS's table/shrink helpers."""
    r, w = credits.rate.double(), credits.weight.double()
    result = r.clone()
    for start in range(0, len(r), run_length):
        end = min(start+run_length, len(r))
        if end-start == 1:
            continue
        rc, wc, hc = r[start:end], w[start:end], gram[start:end, start:end]
        mean = (rc*wc).sum()/wc.sum()
        deviation = rc-mean
        distortion = deviation @ hc @ deviation
        variance = sigma2*((hc.diag()/wc).sum()-hc.sum()/wc.sum())
        if distortion > 0:
            alpha = (variance/distortion).clamp(0, 1)
        else:
            alpha = rc.new_tensor(float(variance > 0))
        result[start:end] = rc-scale*alpha*(rc-mean)
    return result.to(credits.rate.dtype)


@pytest.mark.parametrize("scale", [0., .5, 1.])
def test_oracle_handles_identical_gradients_without_zero_over_zero(scale):
    credits = SimpleNamespace(rate=torch.tensor([2., -1.]), weight=torch.ones(2))
    rates = _oracle_rates(credits, torch.ones(2, 2, dtype=torch.float64), 2, scale, .8)
    assert torch.isfinite(rates).all()
    assert torch.equal(rates, credits.rate)


@pytest.mark.parametrize("run_length", [2, 3])
@pytest.mark.parametrize("update,scale", [(20, 0.), (90, .5), (160, 1.)])
def test_actor_b64_m4_matches_explicit_head_autograd_oracle(monkeypatch, run_length, update, scale):
    device = torch.device(os.environ.get("AUDIT_TEST_DEVICE", "cpu"))
    if device.type == "cpu":
        monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
        for name in ("reset_peak_memory_stats", "synchronize"):
            monkeypatch.setattr(torch.cuda, name, lambda *a, **kw: None)
        for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
            monkeypatch.setattr(torch.cuda, name, lambda *a, **kw: 0)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a: SimpleNamespace(total_memory=0))
    model, previous, _ = _setup(device)
    reference = copy.deepcopy(model)
    algo = _make_algo("grass_chunk_temporal")
    algo.student = model
    algo.teacher_lm_head = previous.teacher_lm_head
    algo.grass_head_bias = True
    algo.grass_chunk_run_length = run_length
    algo.grass_noise = GrassNoiseEstimator(initial_variance=.8, min_adjacent_pairs=100)
    algo.student_updates = update-1
    algo.strategy.accumulated_gradient = 16
    atoms = tuple(MPAtom("test", i, i+1, i, i+1, i, i+1, 1, 1) for i in range(4))
    algo.atomizer = SimpleNamespace(atomize=lambda *a, **kw: AtomizationResult(atoms, True, None, 4, 4))
    batches = []
    for _ in range(16):
        ids = torch.randint(0, 3, (4, 5), device=device)
        mask = torch.tensor([[1, 1, 1, 1, 0]]*4, dtype=torch.bool, device=device)
        batches.append(dict(stu_input_ids=ids, tea_input_ids=ids.clone(),
            stu_loss_mask=mask, tea_loss_mask=mask, stu_attn_mask=torch.ones_like(ids),
            teacher_hiddens=torch.randn(16, 2, device=device),
            avg_micro_batch_token_num=torch.tensor(16., device=device)))
    initial_rng = capture_rng()
    optim = torch.optim.SGD(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=1.)
    actual_gradients = []
    class Strategy:
        step = 0
        accumulated_gradient = 16
        def backward(self, loss, *args):
            self.step = (self.step+1) % 16
            (loss/16).backward()
        def optimizer_step(self, optimizer, student, schedule):
            if self.step == 0:
                actual_gradients.extend(p.grad.detach().clone() for p in student.parameters())
                optimizer.step(); optimizer.zero_grad(); schedule.step()
    actor = SimpleNamespace(student=model, kd_algorithm=algo, optim=optim,
        scheduler=scheduler, strategy=Strategy(), args=algo.args, _world_size=1)
    status = _fit_body()(actor, batches)
    assert status["optimizer_updates"] == 1
    assert algo.student_updates == update
    assert status["mp_opd_grass_chunk_temporal_optimizer_update"] == update
    assert status["mp_opd_grass_chunk_temporal_scale"] == scale
    assert status["mp_opd_numeric_fallback_fraction"] == 0
    assert status["mp_opd_grass_chunk_conservation_error"] < 1e-5
    assert status["mp_opd_grass_chunk_effective_alpha_mean"] == pytest.approx(
        scale*status["mp_opd_grass_chunk_raw_alpha_mean"], abs=1e-7)
    assert len(model.frames) == 16
    restore_rng(initial_rng)
    for batch in batches:
        mask = batch["stu_loss_mask"]
        logits = reference(batch["stu_input_ids"])["logits"][mask]
        labels = batch["stu_input_ids"].roll(-1, 1)[mask]
        teacher = algo.teacher_lm_head(batch["teacher_hiddens"])
        loss = 0
        for i in range(4):
            sl = slice(i*4, (i+1)*4)
            credits = build_atom_credits(atoms, logits[sl], labels[sl], teacher[sl], labels[sl])
            gradients = []
            for nll in credits.current_nll:
                parts = torch.autograd.grad(nll, (reference.head.weight, reference.head.bias), retain_graph=True)
                gradients.append(torch.cat([p.detach().double().flatten() for p in parts]))
            gs = torch.stack(gradients)
            rates = _oracle_rates(credits, gs@gs.T, run_length, scale, .8)
            loss = loss+(rates.detach()*credits.current_nll).sum()
        (loss/(16*16)).backward()
    grad_error = max(float((a-p.grad).abs().max()) for a, p in zip(actual_gradients, reference.parameters()))
    assert grad_error < 3e-6
    torch.optim.SGD(reference.parameters(), lr=.01).step()
    parameter_error = max(float((a-b).detach().abs().max()) for a, b in zip(model.parameters(), reference.parameters()))
    assert parameter_error < 3e-6
    print("TEMPORAL_B64_M4_JSON="+json.dumps(dict(device=str(device), update=update,
        run_length=run_length, scale=scale, optimizer_updates=1,
        raw_alpha_mean=status["mp_opd_grass_chunk_raw_alpha_mean"],
        effective_alpha_mean=status["mp_opd_grass_chunk_effective_alpha_mean"],
        conservation_error=status["mp_opd_grass_chunk_conservation_error"],
        max_gradient_error=grad_error, max_parameter_error=parameter_error)))
