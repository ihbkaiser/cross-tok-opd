"""Unit tests for per-step live eval plans (no GPU, no servers, no torch).

build_plan is pure stdlib and runs anywhere. ensure_state needs eval_queue
(fcntl) plus a working scorer, so it only runs where those exist.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "scripts" / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

_SPEC = importlib.util.spec_from_file_location(
    "plan_live_checkpoint", EVAL_DIR / "plan_live_checkpoint.py"
)
P = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(P)

import contract_eval as E
import queue_data as D

COUNTS = {"gsm8k": 1319, "math500": 500, "mbpp": 500, "live-code-bench-v6": 1055}


def _prepared_root(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for bench, count in COUNTS.items():
        items = [{"id": f"{bench}-{i}", "prompt": "p", "messages": [{"role": "user", "content": "p"}]}
                 for i in range(count)]
        (path / f"{bench}.json").write_text(json.dumps(
            {"benchmark": bench, "items": items, "source": {}, "profile": D.PROFILE}))
    return path


def _checkpoint(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "tokenizer_config.json").write_text("{}")
    (path / "tokenizer.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"\x00" * 64)
    return path


def test_build_plan_writes_conventional_layout(tmp_path):
    prep = _prepared_root(tmp_path / "prep")
    ckpt = _checkpoint(tmp_path / "ckpt" / "step40")
    case = tmp_path / "case"
    plan_path, plan = P.build_plan(case, "myrun", 40, "grass", ckpt, prep)
    assert plan_path == case / "eval" / "myrun-step40" / "plan.json"
    assert plan_path.is_file()
    assert plan["schema"] == "eval-queue-v1"
    assert plan["profile"] == D.PROFILE
    assert plan["seeds"] == [42, 43, 44]
    assert plan["jobs"][0]["id"] == "myrun-step40"
    assert plan["jobs"][0]["step"] == 40
    assert plan["source"] == D.script_hashes()
    assert set(plan["data"]) == set(COUNTS)
    # Idempotent: same inputs rebuild the identical plan.
    _, plan2 = P.build_plan(case, "myrun", 40, "grass", ckpt, prep)
    assert plan2 == plan


def test_build_plan_rejects_drift_and_garbage(tmp_path):
    prep = _prepared_root(tmp_path / "prep")
    ckpt = _checkpoint(tmp_path / "ckpt" / "step40")
    case = tmp_path / "case"
    P.build_plan(case, "myrun", 40, "grass", ckpt, prep)
    (ckpt / "model.safetensors").write_bytes(b"\x01" * 64)
    with pytest.raises(ValueError, match="drift"):
        P.build_plan(case, "myrun", 40, "grass", ckpt, prep)
    with pytest.raises(ValueError):
        P.build_plan(case, "myrun", 0, "grass", ckpt, prep)
    (prep / "gsm8k.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        P.build_plan(case, "other", 80, "grass", ckpt, prep)


def test_build_plan_rejects_wrong_profile(tmp_path):
    prep = _prepared_root(tmp_path / "prep")
    bad = json.loads((prep / "mbpp.json").read_text())
    bad["profile"] = "paper-spec"
    (prep / "mbpp.json").write_text(json.dumps(bad))
    ckpt = _checkpoint(tmp_path / "ckpt" / "step40")
    with pytest.raises(ValueError, match="profile"):
        P.build_plan(tmp_path / "case", "myrun", 40, "grass", ckpt, prep)


def test_build_plan_rejects_incomplete_checkpoint(tmp_path):
    prep = _prepared_root(tmp_path / "prep")
    ckpt = tmp_path / "ckpt" / "step40"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text("{}")
    with pytest.raises(ValueError):
        P.build_plan(tmp_path / "case", "myrun", 40, "grass", ckpt, prep)


def test_ensure_state_needs_unix_and_scorer(tmp_path):
    try:
        import eval_queue  # noqa: F401
    except ImportError:
        pytest.skip("eval_queue needs fcntl (Unix)")
    prep = _prepared_root(tmp_path / "prep")
    ckpt = _checkpoint(tmp_path / "ckpt" / "step40")
    plan_path, _ = P.build_plan(tmp_path / "case", "myrun", 40, "grass", ckpt, prep)
    state_path = P.ensure_state(plan_path, json.loads(plan_path.read_text()))
    state = json.loads(state_path.read_text())
    assert state["plan_sha256"] == E.file_hash(plan_path)
    assert isinstance(state["qualification"], list) and state["qualification"]
    # Second call is a no-op read.
    assert P.ensure_state(plan_path, json.loads(plan_path.read_text())) == state_path
