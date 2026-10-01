"""Cheap Modal CPU probe for P0-A launcher admission."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/opt/overlay")
SOURCE_COMMIT = os.environ.get("P0A_SOURCE_COMMIT", "unknown")
RUN_ID = os.environ.get("P0A_RUN_ID", "simct-p0a-admission-preflight-20260916")

app = modal.App(RUN_ID)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir(str(ROOT / "experiments/runai"), "/opt/overlay/experiments/runai")
    .add_local_dir(str(ROOT / "experiments/modal"), "/opt/overlay/experiments/modal")
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@app.function(image=image, cpu=1, memory=2048, timeout=300, retries=0)
def admission_probe(source_commit: str, run_id: str) -> dict[str, object]:
    root = Path("/opt/overlay")
    fixture = Path("/tmp/p0a-admission-fixture")
    out = fixture / "run"
    fixture.mkdir(parents=True, exist_ok=True)
    for role in ("student", "teacher"):
        model = fixture / role
        model.mkdir(exist_ok=True)
        (model / "config.json").write_text("{}\n")
    for name, content in (("prompts.parquet", b"probe-dataset"), ("energy.pt", b"probe-energy"), ("meta.parquet", b"probe-meta")):
        (fixture / name).write_bytes(content)
    env = dict(
        os.environ,
        MP_PREFLIGHT_ONLY="1",
        MP_STUDENT_PATH=str(fixture / "student"),
        MP_TEACHER_PATH=str(fixture / "teacher"),
        MP_DATASET_PATH=str(fixture / "prompts.parquet"),
        MP_ENERGY_CHECKPOINT=str(fixture / "energy.pt"),
        MP_META_PATH=str(fixture / "meta.parquet"),
        MP_ALTERNATING="1",
        MP_ATTN_IMPLEMENTATION="eager",
        MP_OFFLOAD_ADAM_MOMENTS="1",
        MP_MICRO_TRAIN_BATCH_SIZE="1",
        MP_META_MICRO_BATCH_SIZE="4",
        MP_ALGORITHM="mp_opd",
        MP_MAX_SPAN_LENGTH="2",
        MP_FIXED_SPAN_LENGTH="2",
        MP_ENERGY_EVERY="1",
        MP_ENERGY_LR="0.001",
        MP_QUALIFICATION_POLICY="required",
        MP_QUALIFICATION_STATUS="pass",
        MP_SOURCE_COMMIT=source_commit,
        MP_SOURCE_DIRTY="false",
        CUDA_VISIBLE_DEVICES="0",
    )
    command = ["python", str(root / "experiments/runai/run_single_gpu.py"), "soft", "312", str(out)]
    import subprocess
    result = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, timeout=120)
    config_path = out / "launch-config.json"
    payload = {
        "run_id": run_id,
        "source_commit": source_commit,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path) if config_path.is_file() else None,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    if result.returncode != 0:
        raise RuntimeError("P0-A Modal admission probe failed")
    config = json.loads(config_path.read_text())
    assert config["contract"]["execution_updates"] == 312
    assert config["contract"]["scheduler_horizon"] == 312
    assert config["options"]["micro_train_batch_size"] == 1
    assert config["options"]["mp_opd_meta_microbatch_size"] == 4
    assert config["options"]["attn_implementation"] == "eager"
    assert config["options"]["mp_opd_offload_adam_moments"] is True
    assert "ADMISSION_PASS: exact soft alternating micro1/meta4 full-run contract" in result.stdout
    return payload


@app.local_entrypoint()
def main() -> None:
    import subprocess
    commit = os.environ.get("P0A_SOURCE_COMMIT")
    if not commit:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result = admission_probe.remote(commit, RUN_ID)
    print("P0A_MODAL_PROBE_PASS", json.dumps(result, sort_keys=True), flush=True)
