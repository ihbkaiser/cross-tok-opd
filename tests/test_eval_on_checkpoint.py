"""Unit tests for in-process eval generation (no GPU, no servers, no training).

These tests pin the on-disk contract that score-spool consumes. They cannot
prove end-to-end generation on a B200; that part runs on the node and is
reported, not asserted, here.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "scripts" / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import importlib.util

_SPEC = importlib.util.spec_from_file_location(
    "eval_on_checkpoint", ROOT / "kdflow" / "utils" / "eval_on_checkpoint.py"
)
M = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(M)

import contract_eval as E
import queue_data as D


class FakeE:
    CAPS = dict(E.CAPS)
    COUNTS = dict(E.COUNTS)

    @staticmethod
    def generation_payload(model, item, benchmark, seed):
        return E.generation_payload(model, item, benchmark, seed)

    @staticmethod
    def digest(value):
        return E.digest(value)

    @staticmethod
    def encoded(value):
        return E.encoded(value)

    @staticmethod
    def validate_response(result):
        E.validate_response(result)


def _item(i):
    return {
        "id": f"q{i}",
        "messages": [{"role": "user", "content": f"solve {i}"}],
    }


def _ok_response():
    return {"choices": [{"finish_reason": "stop", "message": {"content": "42"}}]}


def test_parse_steps_default_convention():
    assert M.parse_steps("40,80,120,160,200,240,280,312") == [40, 80, 120, 160, 200, 240, 280, 312]


def test_parse_steps_rejects_non_positive():
    with pytest.raises(ValueError):
        M.parse_steps("40,0,80")
    with pytest.raises(ValueError):
        M.parse_steps("40,-8")


def test_served_model_id_requires_exactly_one(monkeypatch):
    monkeypatch.setattr(M, "_json", lambda *a, **k: {"data": [{"id": "eval-gemma"}]})
    assert M.served_model_id("http://x") == "eval-gemma"
    monkeypatch.setattr(M, "_json", lambda *a, **k: {"data": []})
    with pytest.raises(ValueError):
        M.served_model_id("http://x")
    monkeypatch.setattr(M, "_json", lambda *a, **k: {"data": [{"id": "a"}, {"id": "b"}]})
    with pytest.raises(ValueError):
        M.served_model_id("http://x")


def test_server_record_is_deterministic():
    checkpoint = {"sha256": "abc", "path": "/ckpt/step40"}
    first = M.server_record(served_model="eval-gemma", step=40, checkpoint=checkpoint)
    second = M.server_record(served_model="eval-gemma", step=40, checkpoint=checkpoint)
    assert first == second
    assert first["kind"] == "in-process-rollout"
    assert first["served_model"] == "eval-gemma"
    assert first["step"] == 40


def test_cell_contract_matches_queue_shape():
    try:
        import eval_queue as Q
    except ImportError:
        pytest.skip("eval_queue needs fcntl (Unix); runs on Linux/CI")
    job = {"id": "run-step40", "checkpoint": {"sha256": "deadbeef"}}
    data = {"gsm8k": {"sha256": "datahash"}}
    server = M.server_record(served_model="eval-gemma", step=40,
                             checkpoint={"sha256": "deadbeef", "path": "/x"})
    mine = M._cell_contract("planhash", job, data, "gsm8k", 42, server, D.PROFILE)
    theirs = Q.cell_contract("planhash", job, data, "gsm8k", 42, server)
    assert mine == theirs


def _fakes(calls):
    def render_texts(items):
        return {item_id: f"TEXT:{item_id}" for item_id in items}

    def generate_texts(prompt_texts, params):
        calls.append((list(prompt_texts), list(params)))
        return [
            {"text": f"OUT:{text}", "meta_info": {"finish_reason": {"type": "stop"}}}
            for text in prompt_texts
        ]

    return render_texts, generate_texts


def test_generate_cell_round_trip_and_resume(tmp_path):
    items = {f"q{i}": _item(i) for i in range(3)}
    contract = {"plan_sha256": "p", "checkpoint_sha256": "c", "data_sha256": "d",
                "benchmark": "gsm8k", "seed": 42, "profile": D.PROFILE, "server": {"kind": "t"}}
    calls = []
    render_texts, generate_texts = _fakes(calls)
    cell = tmp_path / "cell"
    n = M._generate_cell(E=FakeE, cell=cell, items=items, benchmark="gsm8k", seed=42,
                         served_model="eval-gemma", contract=contract,
                         render_texts=render_texts, generate_texts=generate_texts)
    assert n == 3
    assert len(calls) == 1
    prompts, params = calls[0]
    assert prompts == ["TEXT:q0", "TEXT:q1", "TEXT:q2"]
    assert [p["sampling_seed"] for p in params] == [
        FakeE.generation_payload("eval-gemma", items[f"q{i}"], "gsm8k", 42)["seed"]
        for i in range(3)
    ]
    assert all(p["temperature"] == 0.6 and p["top_p"] == 0.95 for p in params)
    rows = [json.loads(line) for line in (cell / "responses.jsonl").read_text().splitlines()]
    assert {r["id"] for r in rows} == set(items)
    for row in rows:
        expected = FakeE.generation_payload("eval-gemma", items[row["id"]], "gsm8k", 42)
        assert row["seed"] == 42
        assert row["request_sha256"] == FakeE.digest(FakeE.encoded(expected))
        assert row["response"]["choices"][0]["message"]["content"] == f"OUT:TEXT:{row['id']}"
    marker = json.loads((cell / "generation-complete.json").read_text())
    assert marker["contract"] == contract
    assert marker["count"] == 3
    # A valid marker resumes without touching the server again.
    calls.clear()
    assert M._generate_cell(E=FakeE, cell=cell, items=items, benchmark="gsm8k", seed=42,
                            served_model="eval-gemma", contract=contract,
                            render_texts=render_texts, generate_texts=generate_texts) == 3
    assert calls == []


def test_generate_cell_refuses_contract_drift(tmp_path):
    items = {f"q{i}": _item(i) for i in range(2)}
    contract = {"plan_sha256": "p", "server": {"kind": "t"}}
    render_texts, generate_texts = _fakes([])
    cell = tmp_path / "cell"
    M._generate_cell(E=FakeE, cell=cell, items=items, benchmark="gsm8k", seed=42,
                     served_model="eval-gemma", contract=contract,
                     render_texts=render_texts, generate_texts=generate_texts)
    with pytest.raises(ValueError):
        M._generate_cell(E=FakeE, cell=cell, items=items, benchmark="gsm8k", seed=42,
                         served_model="eval-gemma",
                         contract={**contract, "plan_sha256": "other"},
                         render_texts=render_texts, generate_texts=generate_texts)


def test_generate_cell_refuses_interrupted_spool(tmp_path):
    render_texts, generate_texts = _fakes([])
    cell = tmp_path / "cell"
    cell.mkdir()
    (cell / "responses.jsonl").write_text('{"id": "q0"}\n')
    with pytest.raises(ValueError, match="interrupted generation spool"):
        M._generate_cell(E=FakeE, cell=cell, items={"q0": _item(0)}, benchmark="gsm8k",
                         seed=42, served_model="eval-gemma",
                         contract={"plan_sha256": "p"},
                         render_texts=render_texts, generate_texts=generate_texts)


def test_adapt_router_output_maps_finish_reasons():
    stop = M.adapt_router_output({"text": "hi", "meta_info": {"finish_reason": {"type": "stop"}}})
    assert stop["choices"][0]["finish_reason"] == "stop"
    assert stop["choices"][0]["message"]["content"] == "hi"
    length = M.adapt_router_output({"text": "hi", "meta_info": {"finish_reason": {"type": "length"}}})
    assert length["choices"][0]["finish_reason"] == "length"
    with pytest.raises(ValueError):
        M.adapt_router_output({"text": "hi", "meta_info": {"finish_reason": {"type": "greedy"}}})
    with pytest.raises(ValueError):
        M.adapt_router_output({"text": "hi", "meta_info": {}})
    with pytest.raises(ValueError):
        M.adapt_router_output({"meta_info": {"finish_reason": {"type": "stop"}}})
    FakeE.validate_response(stop)
    FakeE.validate_response(length)


def test_router_sampling_params_carry_the_contract_seed():
    payload = FakeE.generation_payload("eval-gemma", _item(0), "math500", 43)
    params = M.router_sampling_params(FakeE, "math500", payload)
    assert params == {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 4096,
                      "sampling_seed": payload["seed"]}


def test_render_prompt_texts_uses_the_given_template():
    items = {"a": _item(0), "b": _item(1)}
    texts = M.render_prompt_texts(lambda messages, **kw: f"T:{messages[0]['content']}", items)
    assert texts == {"a": "T:solve 0", "b": "T:solve 1"}
    with pytest.raises(ValueError):
        M.render_prompt_texts(lambda messages, **kw: "", items)
