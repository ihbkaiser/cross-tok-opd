"""Prepare and run one evidence-gated X-Token Phase A workload on Modal.

The data/model preparation is CPU-only.  The A10:2 function performs the full
projection, transport canary, exact config gate, and exactly one 10-update
training invocation.  Every durable artifact lives on a dedicated Modal Volume.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

import modal

from xtoken_phase_a_contract import (
    APP_NAME,
    CONTAINER_MEMORY_MIB,
    DATA_REVISION,
    EPHEMERAL_DISK_MIB,
    GPU,
    GPU_COUNT,
    NATIVE_LOCK_SHA256,
    NEMO_REVISION,
    OPTIMIZER_UPDATES,
    OVERLAY_NAME,
    RUN_ID,
    SECRET_NAME,
    STUDENT_REPO,
    STUDENT_REVISION,
    STUDENT_WEIGHT_SHA256,
    TARGET_NAME,
    TEACHER_REPO,
    TEACHER_REVISION,
    VOLUME_NAME,
    operational_contract,
    scientific_contract,
)

REMOTE_REPO = Path("/opt/repo")
LOCAL_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE_REPO
REMOTE_ENVIRONMENTS = REMOTE_REPO / "experiments/environments"
REMOTE_XTOKEN_SCRIPTS = REMOTE_REPO / "experiments/xtoken/scripts"
PYTHON = "/opt/xtoken/venv/bin/python"
RUNTIME = Path("/vol/runtime")

SUPPORT_FILES = (
    "prepare_source.py",
    "prepare_xtoken_target.py",
    "build_cpu_logits_overlay.py",
    "cpu_logits_packed_transport.py",
    "cpu_logits_canary.py",
    "validate_xtoken_target.py",
    "xtoken_target.py",
)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("ca-certificates", "curl", "git", "build-essential", "libnuma1")
    .pip_install("uv==0.12.7")
    .add_local_dir(
        str(LOCAL_ROOT / "experiments/environments/xtoken"),
        remote_path=str(REMOTE_ENVIRONMENTS / "xtoken"),
        copy=True,
    )
)
image = image.run_commands(
    "uv python install 3.13.15",
    (
        "env -u UV_INDEX_URL -u UV_DEFAULT_INDEX -u PIP_INDEX_URL "
        "-u PIP_TRUSTED_HOST UV_PROJECT_ENVIRONMENT=/opt/xtoken/venv "
        "UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300 uv sync --project "
        f"{REMOTE_ENVIRONMENTS / 'xtoken'} --locked"
    ),
)
# Operational-only artifact reader.  This exact version is already pinned in
# the native X-Token lock; placing it after the expensive sync preserves that
# cached dependency layer and does not alter the training environment.
image = image.pip_install("tensorboard==2.21.0")
for filename in SUPPORT_FILES:
    image = image.add_local_file(
        str(LOCAL_ROOT / "experiments/environments" / filename),
        remote_path=str(REMOTE_ENVIRONMENTS / filename),
        copy=True,
    )
image = image.add_local_file(
    str(Path(__file__).with_name("xtoken_phase_a_contract.py")),
    remote_path="/root/xtoken_phase_a_contract.py",
    copy=True,
)
image = image.add_local_file(
    str(LOCAL_ROOT / "experiments/xtoken/scripts/run_logged.py"),
    remote_path=str(REMOTE_XTOKEN_SCRIPTS / "run_logged.py"),
    copy=True,
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
hf_secret = modal.Secret.from_name(SECRET_NAME)


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sanitize(text: str, token: str | None = None) -> str:
    if token:
        text = text.replace(token, "[REDACTED]")
    text = re.sub(r"https?://[^\s]+", "[REMOTE_URL]", text)
    return text


def clean_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("UV_", "PIP_"))
        and not any(
            marker in key.upper()
            for marker in ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "KAGGLE_KEY")
        )
    }
    environment.update(
        PATH=f"/opt/xtoken/venv/bin:{environment.get('PATH', '/usr/local/bin:/usr/bin:/bin')}",
        HOME="/root",
        LANG="C.UTF-8",
        PYTHONUNBUFFERED="1",
        UV_CACHE_DIR="/tmp/uv-cache",
        UV_NO_CACHE="false",
        UV_LINK_MODE="copy",
        UV_PYTHON_INSTALL_DIR="/tmp/uv-python",
        UV_PROJECT_ENVIRONMENT="/opt/xtoken/venv",
        VDT_EPHEMERAL_UV_ROOT="/tmp/xtoken-runtime",
        HF_HOME=str(RUNTIME / "hf"),
        WANDB_MODE="disabled",
        TORCH_CUDA_ARCH_LIST="8.6",
        RAY_USAGE_STATS_ENABLED="0",
        NEMO_RL_PY_EXECUTABLES_SYSTEM="1",
        NEMO_RL_VENV_DIR=str(RUNTIME / "worker-venvs"),
        NCCL_CUMEM_HOST_ENABLE="0",
        OMP_NUM_THREADS="4",
    )
    return environment


def initialize_runtime() -> None:
    RUNTIME.mkdir(mode=0o700, parents=True, exist_ok=False)
    (RUNTIME / "xtoken").mkdir(mode=0o700)
    (RUNTIME / "xtoken/venv").symlink_to("/opt/xtoken/venv", target_is_directory=True)
    contract = {
        "run_id": RUN_ID,
        "created_at": utcnow(),
        "scientific": scientific_contract(),
        "operational": operational_contract(),
        "native_lock_sha256": NATIVE_LOCK_SHA256,
    }
    atomic_json(RUNTIME / "modal-run.json", contract)


def run_capture(
    command: list[str],
    *,
    name: str,
    cwd: Path,
    timeout: int,
    environment: dict[str, str],
    token: str | None = None,
) -> None:
    evidence = RUNTIME / "modal-evidence" / name
    evidence.mkdir(parents=True, exist_ok=False)
    started = utcnow()
    proc = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    safe_output = sanitize(proc.stdout + "\n" + proc.stderr, token)
    (evidence / "stdout.log").write_text(safe_output)
    atomic_json(
        evidence / "result.json",
        {
            "name": name,
            "started_at": started,
            "finished_at": utcnow(),
            "exit_code": proc.returncode,
            "timeout_seconds": timeout,
            "command": command,
        },
    )
    print(json.dumps({"phase": name, "exit_code": proc.returncode}), flush=True)
    if proc.returncode:
        raise RuntimeError(f"{name} failed with exit code {proc.returncode}")


def download_student(token: str, environment: dict[str, str]) -> None:
    child = dict(environment)
    child.update(
        HF_TOKEN=token,
        HF_HUB_DISABLE_TELEMETRY="1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="0",
        HF_HUB_DOWNLOAD_TIMEOUT="300",
    )
    for phase, pattern in (("metadata", "*.json"), ("weights", "*.safetensors")):
        command = [
            "/opt/xtoken/venv/bin/hf",
            "download",
            STUDENT_REPO,
            "--revision",
            STUDENT_REVISION,
            "--cache-dir",
            str(RUNTIME / "hf/hub"),
            "--include",
            pattern,
            "--max-workers",
            "4",
            "--quiet",
        ]
        run_capture(
            command,
            name=f"student-{phase}",
            cwd=RUNTIME,
            timeout=5400,
            environment=child,
            token=token,
        )
    snapshot = (
        RUNTIME
        / "hf/hub/models--meta-llama--Llama-3.2-1B/snapshots"
        / STUDENT_REVISION
    )
    weight = snapshot / "model.safetensors"
    if sha256(weight) != STUDENT_WEIGHT_SHA256:
        raise ValueError("student weight checksum mismatch")
    for required in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if not (snapshot / required).is_file():
            raise FileNotFoundError(f"missing student metadata: {required}")
    records = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(snapshot.iterdir())
        if path.is_file()
    ]
    student = {
        "role": "student",
        "repo": STUDENT_REPO,
        "revision": STUDENT_REVISION,
        "snapshot": str(snapshot),
        "weights_verified": True,
        "files": records,
    }
    atomic_json(
        RUNTIME / "previous-models.json",
        {
            "complete": True,
            "models": [student],
            "credential_scope": "student download only",
            "token_persisted": False,
        },
    )


def run_target_phase(phase: str, environment: dict[str, str]) -> None:
    command = [
        PYTHON,
        str(REMOTE_ENVIRONMENTS / "xtoken_target.py"),
        phase,
        "--root",
        str(RUNTIME),
        "--steps",
        str(OPTIMIZER_UPDATES),
        "--run-name",
        TARGET_NAME,
    ]
    run_capture(
        command,
        name=f"target-{phase}",
        cwd=REMOTE_REPO,
        timeout=7200 if phase == "projection" else 3600,
        environment=environment,
    )


@app.function(
    image=image,
    volumes={"/vol": volume},
    secrets=[hf_secret],
    cpu=8,
    memory=32_768,
    timeout=10_800,
    retries=0,
    single_use_containers=True,
)
def prepare() -> dict[str, object]:
    result: dict[str, object] = {
        "run_id": RUN_ID,
        "stage": "prepare",
        "status": "running",
        "started_at": utcnow(),
    }
    token = os.environ.pop("HF_TOKEN", "")
    try:
        if not token.startswith("hf_"):
            raise RuntimeError("designated HF token unavailable")
        initialize_runtime()
        environment = clean_environment()
        run_capture(
            [PYTHON, str(REMOTE_ENVIRONMENTS / "prepare_source.py"), "--root", str(RUNTIME)],
            name="prepare-source",
            cwd=REMOTE_REPO,
            timeout=600,
            environment=environment,
        )
        if sha256(RUNTIME / "NeMo-RL/uv.lock") != "95f63521d28a2a4104ff372c5985fe63826ab27d6901b78bada1ab1a89a81bf7":
            raise ValueError("upstream lock drift")
        download_student(token, environment)
        token = ""
        run_target_phase("models", environment)
        run_target_phase("data", environment)
        manifest = json.loads((RUNTIME / TARGET_NAME / "models.json").read_text())
        repos = [(item["repo"], item["revision"]) for item in manifest["models"]]
        if repos != [(STUDENT_REPO, STUDENT_REVISION), (TEACHER_REPO, TEACHER_REVISION)]:
            raise ValueError("model lineage mismatch")
        data = json.loads((RUNTIME / TARGET_NAME / "data/manifest.json").read_text())
        if data["revision"] != DATA_REVISION or data["optimizer_updates"] != OPTIMIZER_UPDATES:
            raise ValueError("data lineage mismatch")
        result.update(
            status="prepared",
            models_verified=True,
            data_verified=True,
            raw_rows=data["raw_rows"],
            complete_packs=data["complete_packs"],
        )
    except BaseException as error:
        result.update(
            status="stopped",
            error_type=type(error).__name__,
            error=sanitize(str(error), token),
        )
    finally:
        token = ""
        result["finished_at"] = utcnow()
        if RUNTIME.exists():
            atomic_json(RUNTIME / "prepare-result.json", result)
            volume.commit()
    return result


def resource_gate() -> dict[str, object]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
        env=clean_environment(),
    )
    rows = [line.strip().split(", ") for line in proc.stdout.splitlines() if line.strip()]
    if len(rows) != GPU_COUNT or any("A10" not in row[0] for row in rows):
        raise RuntimeError("exactly two idle A10 GPUs required")
    used_mib = [int(row[2]) for row in rows]
    if max(used_mib) > 256:
        raise RuntimeError("GPUs are not idle")
    physical_memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    memory_limit_bytes = physical_memory_bytes
    memory_limit_source = "sysconf"
    for candidate in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        if not candidate.is_file():
            continue
        raw_limit = candidate.read_text().strip()
        if raw_limit != "max":
            parsed_limit = int(raw_limit)
            # cgroup v1 represents "unlimited" with an enormous integer.
            if parsed_limit < 1 << 60:
                memory_limit_bytes = min(physical_memory_bytes, parsed_limit)
        memory_limit_source = str(candidate)
        break
    if memory_limit_bytes < 120 * 1024**3:
        raise RuntimeError("less than 120 GiB container RAM")
    if shutil.disk_usage(RUNTIME).free < 15 * 1024**3:
        raise RuntimeError("less than 15 GiB durable free storage")
    return {
        "gpus": [row[0] for row in rows],
        "gpu_memory_total_mib": [int(row[1]) for row in rows],
        "gpu_memory_used_mib": used_mib,
        "container_memory_limit_bytes": memory_limit_bytes,
        "container_memory_limit_source": memory_limit_source,
        "durable_free_bytes": shutil.disk_usage(RUNTIME).free,
    }


def training_environment(source: Path) -> dict[str, str]:
    environment = clean_environment()
    environment.update(
        PYTHONPATH=str(source),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        CUDA_VISIBLE_DEVICES="0,1",
    )
    return environment


def extract_tensorboard(target: Path) -> dict[str, object]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    scalars: dict[str, list[dict[str, float | int]]] = {}
    event_files = sorted(target.glob("logs/**/events.out.tfevents.*"))
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            scalars.setdefault(tag, []).extend(
                {"step": int(event.step), "value": float(event.value), "wall_time": event.wall_time}
                for event in accumulator.Scalars(tag)
            )
    for values in scalars.values():
        values.sort(key=lambda item: (item["step"], item["wall_time"]))
    candidates = {
        tag: values
        for tag, values in scalars.items()
        if any(marker in tag.lower() for marker in ("loss", "kl", "grad", "time", "token"))
    }
    expected_zero_based = set(range(OPTIMIZER_UPDATES))
    expected_one_based = set(range(1, OPTIMIZER_UPDATES + 1))
    complete = {
        tag: values
        for tag, values in candidates.items()
        if (
            {item["step"] for item in values} >= expected_zero_based
            or {item["step"] for item in values} >= expected_one_based
        )
        and all(math.isfinite(float(item["value"])) for item in values)
    }
    return {
        "event_files": [str(path) for path in event_files],
        "event_file_sha256": {str(path): sha256(path) for path in event_files},
        "scalars": scalars,
        "finite_complete_candidate_tags": sorted(complete),
        "metric_gate_pass": len(complete) >= 3,
    }


@app.function(
    image=image,
    volumes={"/vol": volume},
    gpu=GPU,
    cpu=16,
    memory=CONTAINER_MEMORY_MIB,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    timeout=14_400,
    retries=0,
    single_use_containers=True,
)
def train() -> dict[str, object]:
    result: dict[str, object] = {
        "run_id": RUN_ID,
        "stage": "train",
        "status": "running",
        "started_at": utcnow(),
        "training_invoked": False,
        "automatic_retry": False,
    }
    try:
        volume.reload()
        prepare_result = json.loads((RUNTIME / "prepare-result.json").read_text())
        if prepare_result["status"] != "prepared":
            raise RuntimeError("CPU preparation did not pass")
        if (RUNTIME / "train-result.json").exists():
            raise FileExistsError("this unique training run already has terminal evidence")
        result["resource_gate"] = resource_gate()
        environment = training_environment(RUNTIME / "NeMo-RL")
        run_target_phase("projection", environment)
        run_target_phase("config", environment)

        sys.path.insert(0, str(REMOTE_ENVIRONMENTS))
        sys.path.insert(0, str(REMOTE_XTOKEN_SCRIPTS))
        from build_cpu_logits_overlay import build
        from run_logged import run_logged
        from xtoken_target import overrides

        original = RUNTIME / "NeMo-RL"
        overlay = RUNTIME / OVERLAY_NAME
        overlay_manifest = build(
            original,
            overlay,
            REMOTE_ENVIRONMENTS / "cpu_logits_packed_transport.py",
        )
        if overlay_manifest["upstream_commit"] != NEMO_REVISION:
            raise ValueError("overlay source lineage mismatch")
        # Match xtoken_target.py, which canonicalizes the mounted root before
        # rendering target-backed projection, data, and logger paths.
        target = (RUNTIME / TARGET_NAME).resolve()
        models = json.loads((target / "models.json").read_text())
        student, teacher = [item["snapshot"] for item in models["models"]]
        settings = overrides(target, student, teacher, OPTIMIZER_UPDATES)
        base = [
            "uv",
            "run",
            "--project",
            str(REMOTE_ENVIRONMENTS / "xtoken"),
            "--locked",
            "--no-sync",
            "python",
        ]
        environment = training_environment(overlay)
        before_config_bytes = (target / "config-resolved.json").read_bytes()
        before_config = json.loads(before_config_bytes)
        commands = [
            (
                "canary",
                base
                + [
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=2",
                    str(REMOTE_ENVIRONMENTS / "cpu_logits_canary.py"),
                ],
                300,
            ),
            (
                "overlay-config",
                base
                + [
                    str(REMOTE_ENVIRONMENTS / "validate_xtoken_target.py"),
                    "--expected-steps",
                    str(OPTIMIZER_UPDATES),
                    *settings,
                ],
                600,
            ),
            (
                "train",
                base + [str(overlay / "examples/run_xtoken_off_policy_distillation.py"), *settings],
                1800,
            ),
        ]
        attempt = {
            "run_id": RUN_ID,
            "created_at": utcnow(),
            "commands": commands,
            "scientific": scientific_contract(),
            "operational": operational_contract(),
            "overlay_manifest": overlay_manifest,
            "resolved_config_before_sha256": hashlib.sha256(before_config_bytes).hexdigest(),
        }
        with (target / "modal-overlay-train-attempt-1.json").open("x") as stream:
            json.dump(attempt, stream, indent=2)
        phases = []
        for phase, command, timeout in commands:
            if phase in ("canary", "train"):
                resource_gate()
            if phase == "train":
                after_config_bytes = (target / "config-resolved.json").read_bytes()
                after_config = json.loads(after_config_bytes)
                result["config_identity"] = {
                    "before_sha256": hashlib.sha256(before_config_bytes).hexdigest(),
                    "after_sha256": hashlib.sha256(after_config_bytes).hexdigest(),
                    "semantic_equal": after_config == before_config,
                }
                if after_config != before_config:
                    changed_top_level = sorted(
                        key
                        for key in set(before_config) | set(after_config)
                        if before_config.get(key) != after_config.get(key)
                    )
                    result["config_identity"]["changed_top_level"] = changed_top_level
                    raise ValueError(
                        "overlay changed resolved scientific configuration fields: "
                        + ", ".join(changed_top_level)
                    )
                result["training_invoked"] = True
            rc, evidence = run_logged(
                command,
                cwd=overlay,
                root=RUNTIME / "evidence",
                name=f"modal-a10x2-{phase}",
                timeout=timeout,
                env=environment,
            )
            phase_result = {
                "phase": phase,
                "exit_code": rc,
                "evidence": str(evidence),
                "stdout_sha256": sha256(evidence / "stdout.log"),
            }
            phases.append(phase_result)
            result["phases"] = phases
            atomic_json(RUNTIME / "train-result.json", result)
            if rc:
                raise RuntimeError(f"{phase} failed; no automatic retry")
            if phase == "canary":
                text = (evidence / "stdout.log").read_text()
                if text.count('"cross_process_values_bitwise": true') != 4:
                    raise ValueError("two-rank/two-GPU transport canary did not fully pass")
        train_text = (Path(phases[-1]["evidence"]) / "stdout.log").read_text()
        metrics = extract_tensorboard(target)
        atomic_json(target / "modal-verified-metrics.json", metrics)
        result.update(
            status="training_process_complete_pending_review",
            process_exit_code=0,
            max_steps_message="Max steps reached, stopping training." in train_text,
            tensorboard_metric_gate=metrics["metric_gate_pass"],
            complete_metric_tags=metrics["finite_complete_candidate_tags"],
        )
    except BaseException as error:
        result.update(
            status="stopped",
            process_exit_code=1,
            error_type=type(error).__name__,
            error=sanitize(str(error)),
            traceback=sanitize("".join(traceback.format_exception_only(type(error), error))).strip(),
        )
    finally:
        result["finished_at"] = utcnow()
        if RUNTIME.exists():
            atomic_json(RUNTIME / "train-result.json", result)
            volume.commit()
    return result


@app.function(
    image=image,
    volumes={"/vol": volume},
    cpu=4,
    memory=8192,
    timeout=600,
    retries=0,
    single_use_containers=True,
)
def finalize() -> dict[str, object]:
    volume.reload()
    train_result = json.loads((RUNTIME / "train-result.json").read_text())
    phases = train_result.get("phases", [])
    if not train_result.get("training_invoked") or not phases:
        raise RuntimeError("no invoked training evidence to finalize")
    train_phase = phases[-1]
    if train_phase.get("phase") != "train" or train_phase.get("exit_code") != 0:
        raise RuntimeError("training process did not complete successfully")
    train_log = Path(train_phase["evidence"]) / "stdout.log"
    train_text = train_log.read_text()
    observed_steps = [int(value) for value in re.findall(r"Step (\d+)/10", train_text)]
    if observed_steps != list(range(1, OPTIMIZER_UPDATES + 1)):
        raise RuntimeError(f"unexpected optimizer step sequence: {observed_steps}")
    if "Max steps reached, stopping training." not in train_text:
        raise RuntimeError("missing max-steps completion marker")
    target = (RUNTIME / TARGET_NAME).resolve()
    metrics = extract_tensorboard(target)
    if not metrics["metric_gate_pass"]:
        raise RuntimeError("TensorBoard finite/complete metric gate failed")
    atomic_json(target / "modal-verified-metrics.json", metrics)
    result = {
        "run_id": RUN_ID,
        "status": "completed",
        "finalization_only": True,
        "training_process_exit_code": 0,
        "training_stdout_sha256": sha256(train_log),
        "optimizer_updates": OPTIMIZER_UPDATES,
        "observed_steps": observed_steps,
        "max_steps_message": True,
        "tensorboard_metric_gate": True,
        "complete_metric_tags": metrics["finite_complete_candidate_tags"],
        "event_file_sha256": metrics["event_file_sha256"],
        "scientific": scientific_contract(),
        "operational": operational_contract(),
        "finalized_at": utcnow(),
    }
    atomic_json(RUNTIME / "finalization-result.json", result)
    volume.commit()
    return result


@app.local_entrypoint()
def main(stage: str = "all") -> None:
    if stage not in {"all", "prepare", "train", "finalize"}:
        raise ValueError("stage must be all, prepare, train, or finalize")
    if stage in {"all", "prepare"}:
        preparation = prepare.remote()
        print(json.dumps({"modal_result": preparation}, sort_keys=True), flush=True)
        if preparation["status"] != "prepared":
            raise SystemExit(1)
    if stage in {"all", "train"}:
        training = train.remote()
        print(json.dumps({"modal_result": training}, sort_keys=True), flush=True)
        if training["status"] == "stopped":
            raise SystemExit(1)
    if stage == "finalize":
        finalization = finalize.remote()
        print(json.dumps({"modal_result": finalization}, sort_keys=True), flush=True)
