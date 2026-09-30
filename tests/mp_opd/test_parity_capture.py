import json
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from kdflow.algorithms._parity_capture import capture_failure


def test_capture_saves_replay_inputs_without_teacher_or_logits(tmp_path, monkeypatch):
    monkeypatch.setenv("MP_PARITY_CAPTURE_DIR", str(tmp_path))
    called = []
    def save(*args):
        called.append(args[-1])
    algorithm = SimpleNamespace(student=SimpleNamespace(training=True), student_tokenizer=None,
        strategy=SimpleNamespace(save_model=save),
        args=SimpleNamespace(model=SimpleNamespace(attn_implementation="eager")))
    batch = dict(stu_input_ids=torch.tensor([[1, 2, 3], [1, 3, 0]]),
        stu_attn_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
        stu_loss_mask=torch.tensor([[True, True, False], [True, False, False]]),
        stu_behavior_log_probs=torch.tensor([[-1., -2., float("nan")], [-3., float("nan"), float("nan")]]),
        teacher_hiddens=torch.ones(2, 3, 7))
    logits = torch.zeros(3, 4, requires_grad=True)
    capture_failure(algorithm, batch, logits, torch.tensor([2, 3, 3]),
                    torch.tensor([-1., -2., -3.]), .6, RuntimeError("parity"))
    path = next(tmp_path.iterdir())
    saved = torch.load(path / "batch.pt", weights_only=True)
    assert "teacher_hiddens" not in saved and "logits" not in saved
    assert saved["actual_log_probs"].shape == (3,)
    assert not saved["actual_log_probs"].requires_grad
    assert saved["positions"].tolist() == [[0, 0], [0, 1], [1, 0]]
    meta = json.loads((path / "metadata.json").read_text())
    assert meta["checkpoint_complete"] and len(called) == 1
    assert [s["tokens"] for s in meta["per_sample"]] == [2, 1]


def test_disabled_does_not_touch_model(tmp_path, monkeypatch):
    monkeypatch.delenv("MP_PARITY_CAPTURE_DIR", raising=False)
    capture_failure(None, None, None, None, None, None, None)
    assert not list(tmp_path.iterdir())


def test_multi_rank_refused_before_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("MP_PARITY_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    with pytest.raises(RuntimeError, match="exactly one"):
        capture_failure(None, None, None, None, None, None, None)
    assert not list(tmp_path.iterdir())


def test_original_gate_remains_strict():
    source = Path(__file__).resolve().parents[2] / "kdflow/algorithms/mp_opd.py"
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "_behavior_parity_metrics")
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
    gate = namespace[fn.name]
    logits, labels = torch.zeros(100, 4), torch.zeros(100, dtype=torch.long)
    behavior = logits.log_softmax(-1)[:, 0]
    gate(logits, labels, behavior, 1.)
    behavior[:5] += .85
    with pytest.raises(RuntimeError, match="parity failed"):
        gate(logits, labels, behavior, 1.)
