import json, os
from pathlib import Path
import pytest

from experiments.modal.provenance_prepare import PreparationError, load_receipt, prepare, write_receipt

def _tree(tmp_path):
    roots = {}
    for role, files in {"student": {"config.json": "s", "weights.safetensors": "weights", "ignored.csv": "x"}, "teacher": {"config.json": "t", "weights.safetensors": "teacher"}, "dataset": {"prompts.parquet": "data"}, "meta": {"selected.parquet": "meta"}, "energy": {"energy.pt": "energy"}}.items():
        d = tmp_path / role; d.mkdir()
        for name, value in files.items(): (d / name).write_text(value)
        roots[role] = str(d)
    source = tmp_path / "source"; source.mkdir(); (source / "a.py").write_text("print(1)\n")
    return roots, source

def test_uncached_cached_contract_equality_and_progress(tmp_path):
    roots, source = _tree(tmp_path); events = []
    receipt = prepare(roots=roots, source_root=str(source), source_commit="abc", snapshot_id="snap", progress=events.append)
    path = tmp_path / "sidecar" / "receipt.json"; stored = write_receipt(path, receipt)
    loaded = load_receipt(path, expected_sha256=stored["receipt_sha256"], expected_source_commit="abc", expected_snapshot_id="snap")
    assert loaded["launcher_models"] == receipt["launcher_models"] and loaded["pipeline_inputs"] == receipt["pipeline_inputs"]
    assert loaded["counts"]["files"] == 7 and loaded["counts"]["bytes"] > 0
    assert len(events) >= 7 and all(e["event"] == "read" for e in events)
    assert "ignored.csv" not in loaded["launcher_models"]["student"]

def test_changed_content_same_size_and_mtime_rejected(tmp_path):
    roots, source = _tree(tmp_path); path = tmp_path / "r.json"
    stored = write_receipt(path, prepare(roots=roots, source_root=str(source), source_commit="abc", snapshot_id="snap"))
    target = Path(roots["student"]) / "config.json"; st = target.stat(); target.write_text("z"); os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns))
    with pytest.raises(PreparationError): load_receipt(path, expected_sha256=stored["receipt_sha256"])

@pytest.mark.parametrize("mutator", [lambda d: d.pop("receipt_sha256"), lambda d: d.update(status="partial"), lambda d: d.update(snapshot_id="wrong"), lambda d: d.update(launcher_models_sha256="bad")])
def test_invalid_partial_wrong_snapshot_or_coverage_fails(tmp_path, mutator):
    roots, source = _tree(tmp_path); path = tmp_path / "r.json"
    write_receipt(path, prepare(roots=roots, source_root=str(source), source_commit="abc", snapshot_id="snap"))
    value = json.loads(path.read_text()); mutator(value); path.write_text(json.dumps(value))
    with pytest.raises(PreparationError): load_receipt(path, expected_source_commit="abc", expected_snapshot_id="snap")

def test_sidecar_inside_input_is_rejected_and_gate_does_not_dispatch(tmp_path):
    roots, source = _tree(tmp_path)
    with pytest.raises(PreparationError): write_receipt(Path(roots["student"]) / "bad.json", prepare(roots=roots, source_root=str(source)))
    called = []
    try: load_receipt(tmp_path / "missing.json")
    except PreparationError: pass
    assert called == []
