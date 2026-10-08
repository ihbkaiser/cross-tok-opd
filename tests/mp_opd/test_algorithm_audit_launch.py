"""Offline tests of the node handoff's asset/provenance/GPU guard contracts."""
import json
from types import SimpleNamespace

import pytest

from experiments.runai import launch_algorithm_audit_fix as launch


def test_launch_manifest_paths_are_used_without_host_defaults(tmp_path):
    path = tmp_path/"launch-config.json"
    path.write_text(json.dumps({"options": {
        "student_name_or_path": "/my/student", "teacher_name_or_path": "/my/teacher",
        "train_dataset_path": "/my/dataset"}}))
    assert launch.asset_paths(path) == dict(student="/my/student", teacher="/my/teacher", dataset="/my/dataset")


def test_plan_is_read_only_and_pins_batch_scope_gpu3_and_native_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    assets = {k: str(tmp_path/k) for k in ("student", "teacher", "dataset")}
    for path in assets.values():
        launch.Path(path).touch()
    reference = tmp_path/"campaign.json"
    reference.write_text(json.dumps(assets))
    (tmp_path/"ALGORITHM_AUDIT_RECEIPT.json").write_text(json.dumps(dict(commit="abc", source_sha256={})))
    wrapper = tmp_path/"wrapper.sh"; wrapper.touch()
    args = SimpleNamespace(reference=reference, runtime_wrapper=wrapper,
                           output=tmp_path/"fresh", seed=42, mode="trust_b", updates=2)
    monkeypatch.setenv("MP_RESUME", "1")
    monkeypatch.setenv("MP_GRASS_CHUNK_SOURCE", "run")
    monkeypatch.setenv("MP_ENERGY_CHECKPOINT", "stale")
    plan, environment = launch.build_plan(args)
    assert not (tmp_path/"fresh").exists()
    assert plan["gpu"] == 3 and plan["batch"] == 64 and plan["microbatch"] == 4
    assert environment["MP_RESUME"] == "0"
    assert environment["MP_TRUST_SCOPE"] == "batch"
    assert environment["MP_GRASS_CHUNK_SOURCE"] == "xtoken"
    assert "MP_ENERGY_CHECKPOINT" not in environment
    (tmp_path/"fresh").mkdir()
    with pytest.raises(FileExistsError, match="fresh output"):
        launch.build_plan(args)


def test_uncommitted_qualified_bundle_reports_the_tree_instead_of_a_fake_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(launch, "ROOT", tmp_path)
    assets = {k: str(tmp_path/k) for k in ("student", "teacher", "dataset")}
    for path in assets.values():
        launch.Path(path).touch()
    reference = tmp_path/"campaign.json"; reference.write_text(json.dumps(assets))
    (tmp_path/"ALGORITHM_AUDIT_RECEIPT.json").write_text(json.dumps(dict(commit=None, staged_tree="123", source_sha256={})))
    wrapper = tmp_path/"wrapper.sh"; wrapper.touch()
    args = SimpleNamespace(reference=reference, runtime_wrapper=wrapper,
                           output=tmp_path/"fresh", seed=42, mode="trust_b", updates=2)
    plan, environment = launch.build_plan(args)
    assert plan["source_commit"] is None and plan["source_tree"] == "123"
    assert environment["MP_SOURCE_COMMIT"] == "uncommitted-tree:123"
    assert "GPG commit blocked" in environment["MP_SOURCE_DIRTY"]


def test_gpu_guard_rejects_a_holder_only_on_gpu3(monkeypatch):
    def query(command, **kwargs):
        if "--id=3" in command:
            return "GPU-target, 10\n"
        return "GPU-other, 2\nGPU-target, 765\n"
    monkeypatch.setattr(launch.subprocess, "check_output", query)
    with pytest.raises(RuntimeError, match="765"):
        launch.require_idle_gpu()
