"""Run the actual actor fit body and MP-OPD loss through a complete B64/M4 update."""
import ast
import copy
import os
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kdflow.algorithms._mp_opd_atoms import MPAtom, AtomizationResult
from kdflow.algorithms._mp_opd_credit import build_atom_credits
from kdflow.algorithms._mp_opd_trust import trust_calibrate_scalars
from kdflow.algorithms._mp_opd_trust_p import trust_p_calibrate_scalars
from kdflow.metric_reduction import reduce_sparse_metrics
from kdflow.training_checkpoint import capture_rng, restore_rng
from test_numeric_fallback import _make_algo

ROOT = Path(__file__).resolve().parents[2]
ATOMS = (
    MPAtom("test", 0, 1, 0, 1, 0, 1, 1, 1),
    MPAtom("test", 1, 3, 1, 3, 1, 3, 2, 2, boundary_type="multi_token"),
)


class Student(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(3, 2)
        self.dropout = torch.nn.Dropout(.25)
        self.head = torch.nn.Linear(2, 3)
        self.frames = []

    def forward(self, ids, **kwargs):
        hidden = self.dropout(self.emb(ids))
        logits = self.head(hidden)
        self.frames.append((hidden.detach().clone(), logits.detach().clone()))
        return {"logits": logits, "hidden_states": (hidden,)}


def _setup(device, mode="trust_b"):
    torch.manual_seed(287)
    model = Student().to(device)
    algo = _make_algo(mode)
    algo.student = model
    algo.teacher_lm_head = torch.nn.Linear(2, 3).to(device)
    algo.grass_head_bias = True
    algo.atomizer = SimpleNamespace(atomize=lambda *args, **kwargs:
                                   AtomizationResult(ATOMS, True, None, 3, 3))
    algo.strategy.accumulated_gradient = 16
    batches = []
    for _ in range(16):
        ids = torch.randint(0, 3, (4, 4), device=device)
        mask = torch.tensor([[1, 1, 1, 0]]*4, dtype=torch.bool, device=device)
        batches.append(dict(stu_input_ids=ids, tea_input_ids=ids.clone(),
                            stu_loss_mask=mask, tea_loss_mask=mask,
                            stu_attn_mask=torch.ones_like(ids),
                            teacher_hiddens=torch.randn(12, 2, device=device),
                            avg_micro_batch_token_num=torch.tensor(12., device=device)))
    return model, algo, batches


def _fit_body():
    path = ROOT / "kdflow/ray/train/student_actor.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StudentRayActor")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "fit")
    env = dict(torch=torch, np=np, time=time, defaultdict=defaultdict,
               ray=SimpleNamespace(ObjectRef=type("ObjectRef", (), {})),
               reduce_sparse_metrics=reduce_sparse_metrics)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), env)
    return env["fit"]


def _reference_coefficients(algo, batch, logits):
    mask = batch["stu_loss_mask"]
    selected = logits[mask]
    labels = batch["stu_input_ids"].roll(-1, 1)[mask]
    teacher = algo.teacher_lm_head(batch["teacher_hiddens"])
    return [build_atom_credits(ATOMS, selected[i*3:(i+1)*3], labels[i*3:(i+1)*3],
                               teacher[i*3:(i+1)*3], labels[i*3:(i+1)*3])
            for i in range(4)]


@pytest.mark.parametrize("mode", ["trust_b", "trust_p"])
def test_actor_b64_m4_replay_matches_independent_full_batch_lambda_and_update(monkeypatch, mode):
    device = torch.device(os.environ.get("AUDIT_TEST_DEVICE", "cpu"))
    if device.type == "cpu":
        monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
        for name in ("reset_peak_memory_stats", "synchronize"):
            monkeypatch.setattr(torch.cuda, name, lambda *args, **kwargs: None)
        for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
            monkeypatch.setattr(torch.cuda, name, lambda *args, **kwargs: 0)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *args: SimpleNamespace(total_memory=0))
    model, algo, batches = _setup(device, mode)
    reference = copy.deepcopy(model)
    initial_rng = capture_rng()
    optim = torch.optim.SGD(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=1.)

    class Strategy:
        step = 0
        accumulated_gradient = 16
        def backward(self, loss, *args):
            self.step = (self.step+1) % 16
            (loss/16).backward()
        def optimizer_step(self, optimizer, model, schedule):
            if self.step == 0:
                optimizer.step(); optimizer.zero_grad(); schedule.step()

    actor = SimpleNamespace(student=model, kd_algorithm=algo, optim=optim,
                            scheduler=scheduler, strategy=Strategy(), args=algo.args, _world_size=1)
    status = _fit_body()(actor, batches)
    assert status["optimizer_updates"] == 1
    prefix = "mp_opd_trust_p_" if mode == "trust_p" else "mp_opd_trust_"
    assert status[prefix+"full_batch"] == 1
    assert status[prefix+"calibration_response_count"] == 64
    assert len(model.frames) == 32
    for (h1, l1), (h2, l2) in zip(model.frames[:16], model.frames[16:]):
        assert torch.equal(h1, h2) and torch.equal(l1, l2), "dropout replay changed"

    # Independent head-autograd oracle over all 64 responses, including cross-microbatch terms.
    gs = [torch.zeros_like(reference.head.weight), torch.zeros_like(reference.head.bias)]
    gm = [torch.zeros_like(reference.head.weight), torch.zeros_like(reference.head.bias)]
    for batch, (hidden, _) in zip(batches, model.frames[:16]):
        logits = reference.head(hidden)
        credits = _reference_coefficients(algo, batch, logits)
        ls = sum(c.rate[0]*c.current_nll[0] for c in credits)
        lm = sum(c.rate[1]*c.current_nll[1] for c in credits)
        for target, loss in ((gs, ls), (gm, lm)):
            grads = torch.autograd.grad(loss, (reference.head.weight, reference.head.bias), retain_graph=True)
            for dst, src in zip(target, grads):
                dst.add_(src)
    calibrate = trust_p_calibrate_scalars if mode == "trust_p" else trust_calibrate_scalars
    result = calibrate(sum(float(g.double().square().sum()) for g in gs),
                                    sum(float(g.double().square().sum()) for g in gm),
                                    sum(float((s.double()*m.double()).sum()) for s, m in zip(gs, gm)), 64, 64)
    if mode == "trust_p":
        strict_scale, mismatch_scale = result.strict_scale, 1.
        assert status[prefix+"strict_scale"] == pytest.approx(result.strict_scale, rel=3e-5, abs=3e-6)
    else:
        strict_scale, mismatch_scale = 1., result.lam
        assert status[prefix+"lambda"] == pytest.approx(result.lam, rel=3e-5, abs=3e-6)
    restore_rng(initial_rng)
    for batch in batches:
        logits = reference(batch["stu_input_ids"])["logits"]
        credits = _reference_coefficients(algo, batch, logits)
        loss = sum(strict_scale*c.rate[0]*c.current_nll[0]+mismatch_scale*c.rate[1]*c.current_nll[1] for c in credits)
        (loss/(12*16)).backward()
    with torch.no_grad():
        for actual, ref in zip(model.parameters(), reference.parameters()):
            expected = ref - .01*ref.grad
            assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)
    assert algo._trust_batch_result is None


def test_full_batch_trust_refuses_backward_without_calibration():
    model, algo, batches = _setup(torch.device("cpu"))
    with pytest.raises(RuntimeError, match="prepare_optimizer_batch"):
        algo.training_step(batches[0])


def test_incomplete_accumulation_window_is_rejected():
    _, algo, batches = _setup(torch.device("cpu"))
    with pytest.raises(ValueError, match="complete optimizer"):
        algo.prepare_optimizer_batch(batches[:2])


def test_calibration_cannot_start_mid_optimizer_update():
    _, algo, batches = _setup(torch.device("cpu"))
    algo.strategy.step = 1
    with pytest.raises(ValueError, match="optimizer boundary"):
        algo.prepare_optimizer_batch(batches)


def test_projected_trust_rejects_different_microbatch_loss_normalizers():
    _, algo, batches = _setup(torch.device("cpu"), "trust_p")
    batches[1]["avg_micro_batch_token_num"] = torch.tensor(13.)
    with pytest.raises(ValueError, match="common loss normalizer"):
        algo.prepare_optimizer_batch(batches)


def test_align_new_b64_m4_update_matches_independent_two_atom_projection_and_guard(monkeypatch):
    # Single-pass ALIGN path through the real actor; independent head autograd
    # defines H and closed-form two-atom surgery, rather than calling ALIGN code.
    device = torch.device(os.environ.get("AUDIT_TEST_DEVICE", "cpu"))
    if device.type == "cpu":
        monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
        for name in ("reset_peak_memory_stats", "synchronize"):
            monkeypatch.setattr(torch.cuda, name, lambda *a, **k: None)
        for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
            monkeypatch.setattr(torch.cuda, name, lambda *a, **k: 0)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a: SimpleNamespace(total_memory=0))
    model, algo, batches = _setup(device, "align_new")
    reference = copy.deepcopy(model)
    rng = capture_rng()
    optim = torch.optim.SGD(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=1.)
    class Strategy:
        step = 0
        accumulated_gradient = 16
        def backward(self, loss, *args):
            self.step = (self.step+1) % 16
            (loss/16).backward()
        def optimizer_step(self, optimizer, student, schedule):
            if self.step == 0:
                optimizer.step(); optimizer.zero_grad(); schedule.step()
    actor = SimpleNamespace(student=model, kd_algorithm=algo, optim=optim,
                            scheduler=scheduler, strategy=Strategy(), args=algo.args, _world_size=1)
    status = _fit_body()(actor, batches)
    assert status["optimizer_updates"] == 1 and len(model.frames) == 16
    assert status["mp_opd_numeric_fallback_fraction"] == 0
    restore_rng(rng)
    for batch in batches:
        logits = reference(batch["stu_input_ids"])["logits"]
        credits = _reference_coefficients(algo, batch, logits)
        loss = 0
        for c in credits:
            gs = []
            for nll in c.current_nll:
                parts = torch.autograd.grad(nll, (reference.head.weight, reference.head.bias), retain_graph=True)
                gs.append(torch.cat([p.detach().double().flatten() for p in parts]))
            gradients = torch.stack(gs)
            h = gradients @ gradients.T
            r = c.rate.detach().double(); e = r.clone()
            if float(r[0]*r[1]*h[0, 1]) < 0:
                if h[0, 0] > 1e-12: e[0] -= r[1]*h[0, 1]/h[0, 0]
                if h[1, 1] > 1e-12: e[1] -= r[0]*h[0, 1]/h[1, 1]
            g0, g1 = r @ gradients, e @ gradients
            a, d = float(g0 @ g0), float(g0 @ g1)
            beta = 0. if a <= 1e-12 else (min(1., .9*a/(a-d)) if d < .1*a else 1.)
            if 0 < beta < 1: beta *= 1-1e-6
            q = (r+beta*(e-r)).to(c.current_nll)
            loss = loss+(q*c.current_nll).sum()
        (loss/(12*16)).backward()
    torch.optim.SGD(reference.parameters(), lr=.01).step()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-5)
