"""Run one matched, evidence-gated 10-update SimCT OPD workload on Modal."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import modal


APP_NAME = "vdt-simct-opd-l40sx1-no1ceboy-20260904-r10"
RUN_ID = "simct-opd-l40sx1-10update-20260904-r10"
VOLUME_NAME = "vdt-simct-opd-l40sx1-no1ceboy-20260904-r10"
SOURCE_VOLUME_NAME = "vdt-xtoken-phase-a-no1ceboy-20260904-r14"
WANDB_SECRET_NAME = "vdt-xtoken-wandb-no1ceboy"
WANDB_ENTITY = "kieusontung8-hanoi-university-of-science-and-technology"
WANDB_PROJECT = "vdt-simct-tunix-reproduction"
WANDB_RUN_ID = "simct-opd-r10-9d5c801"
WANDB_RUN_NAME = "simct-opd-modal-l40sx1-10update-r10"

STUDENT_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
TEACHER_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
SOURCE_DATA_SHA256 = "9f2a6a657e5e7575eb90bce59df5e385a68efdd37ce165b881e757f993a10b5c"
SIMCT_LOCK_SHA256 = "c1b851b81615d0bb38d47b868948dffa9e81f96f461f1e5f9dff25dc78ea3f8d"
SPAN_CTKD_SHA256 = "ca5b52bdf96fd690853e92eb674afffdfb883e669c0a7a07cd18189e67f63cfe"
REPO_HEAD = "9d5c801ebbaed8c1b7acf7a235876da32a2afb65"

REMOTE_REPO = Path("/opt/repo")
LOCAL_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE_REPO
PYTHON = "/opt/kdflow/venv/bin/python"
SOURCE_ROOT = Path("/source/runtime")
TARGET_ROOT = SOURCE_ROOT / "target-4b-2048-b64-full-10steps-modal-a10x2-r14"
OUTPUT_ROOT = Path("/runs") / RUN_ID
STUDENT = (
    SOURCE_ROOT
    / "hf/hub/models--meta-llama--Llama-3.2-1B/snapshots"
    / STUDENT_REVISION
)
TEACHER = SOURCE_ROOT / "hf/hub/models--Qwen--Qwen3-4B/snapshots" / TEACHER_REVISION
SOURCE_DATA = TARGET_ROOT / "data/formal-logic-prefix.parquet"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def scientific_contract() -> dict[str, object]:
    return {
        "regime": "on-policy distillation from student rollouts",
        "algorithm": "SimCT span_ctkd reverse-KL",
        "implementation": "exact vectorized gathers plus row-chunked recompute RKL with full telemetry",
        "span_ctkd_sha256": SPAN_CTKD_SHA256,
        "student": "meta-llama/Llama-3.2-1B",
        "student_revision": STUDENT_REVISION,
        "teacher": "Qwen/Qwen3-4B",
        "teacher_revision": TEACHER_REVISION,
        "optimizer_updates": 10,
        "sequence_length": 2048,
        "prompt_max_tokens": 240,
        "generation_max_tokens": 1808,
        "teacher_alignment_context_length": 4096,
        "teacher_forward_chunk_microbatches": 8,
        "rollout_batch_size": 64,
        "train_batch_size": 64,
        "micro_train_batch_size": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": 42,
        "kd_loss_fn": "rkl",
        "kd_ratio": 1.0,
        "span_gh_mask_threshold": 2.0,
        "data_source_sha256": SOURCE_DATA_SHA256,
        "prompt_transform": (
            "first 640 rows; longest student-tokenizer prefix whose final encoded "
            "prompt including special tokens is <=240 tokens"
        ),
    }


image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("ca-certificates", "git", "build-essential", "libnuma1")
    .pip_install("uv==0.12.7")
    .add_local_file(str(LOCAL_ROOT / "pyproject.toml"), "/opt/repo/pyproject.toml", copy=True)
    .add_local_file(str(LOCAL_ROOT / "requirements.txt"), "/opt/repo/requirements.txt", copy=True)
    .add_local_file(str(LOCAL_ROOT / "README.md"), "/opt/repo/README.md", copy=True)
    .add_local_dir(str(LOCAL_ROOT / "kdflow"), "/opt/repo/kdflow", copy=True)
    .add_local_dir(str(LOCAL_ROOT / "tests"), "/opt/repo/tests", copy=True)
    .add_local_dir(
        str(LOCAL_ROOT / "experiments/environments/simct"),
        "/opt/repo/experiments/environments/simct",
        copy=True,
    )
    # KDFlow discovers every algorithm module at import time.  SimCT does not
    # execute X-Token, but the registry still imports that module, so its
    # upstream adapter must be packaged for a complete source tree.
    .add_local_file(
        str(LOCAL_ROOT / "experiments/modal/vendor/xtoken_upstream_token_aligner.py"),
        "/opt/kdflow-vdt/xtoken_upstream_token_aligner.py",
        copy=True,
    )
    .add_local_file(
        str(LOCAL_ROOT / "experiments/modal/prepare_opd_prompts.py"),
        "/opt/kdflow-vdt/prepare_opd_prompts.py",
        copy=True,
    )
)
image = image.run_commands(
    "uv python install 3.12.12",
    (
        "env -u UV_INDEX_URL -u UV_DEFAULT_INDEX -u PIP_INDEX_URL "
        "-u PIP_TRUSTED_HOST UV_PROJECT_ENVIRONMENT=/opt/kdflow/venv "
        "UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300 uv sync --project "
        "/opt/repo/experiments/environments/simct --locked"
    ),
)

app = modal.App(APP_NAME)
source_volume = modal.Volume.from_name(SOURCE_VOLUME_NAME, create_if_missing=False)
output_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
wandb_secret = modal.Secret.from_name(WANDB_SECRET_NAME)


def clean_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("UV_", "PIP_"))
        and not any(marker in key.upper() for marker in ("HF_TOKEN", "KAGGLE", "PASSWORD"))
    }
    nvidia_root = Path("/opt/kdflow/venv/lib/python3.12/site-packages/nvidia")
    cuda_library_dirs = [
        "/usr/local/cuda/lib64",
        "/usr/local/nvidia/lib",
        "/usr/local/nvidia/lib64",
    ]
    cuda_library_dirs.extend(str(path) for path in sorted(nvidia_root.glob("*/lib")))
    inherited_ld_path = environment.get("LD_LIBRARY_PATH")
    if inherited_ld_path:
        cuda_library_dirs.append(inherited_ld_path)
    environment.update(
        PATH=f"/opt/kdflow/venv/bin:{environment.get('PATH', '/usr/local/bin:/usr/bin:/bin')}",
        PYTHONPATH="/opt/kdflow-vdt:/opt/repo",
        PYTHONUNBUFFERED="1",
        HF_HOME=str(SOURCE_ROOT / "hf"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        NCCL_CUMEM_HOST_ENABLE="0",
        RAY_USAGE_STATS_ENABLED="0",
        WANDB_MODE="online",
        WANDB_SILENT="true",
        TORCH_CUDA_ARCH_LIST="8.6",
        OMP_NUM_THREADS="4",
        CUDA_HOME="/usr/local/cuda",
        LD_LIBRARY_PATH=":".join(cuda_library_dirs),
    )
    return environment


def prepare_prompts(environment: dict[str, str]) -> dict[str, object]:
    destination = OUTPUT_ROOT / "data/opd-prompts.jsonl"
    manifest = OUTPUT_ROOT / "data/manifest.json"
    process = subprocess.run(
        [
            PYTHON,
            "/opt/kdflow-vdt/prepare_opd_prompts.py",
            "--source", str(SOURCE_DATA),
            "--source-sha256", SOURCE_DATA_SHA256,
            "--student", str(STUDENT),
            "--destination", str(destination),
            "--manifest", str(manifest),
            "--rows", "640",
            "--max-tokens", "240",
        ],
        cwd=REMOTE_REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    (OUTPUT_ROOT / "prepare-prompts.log").write_text(process.stdout + process.stderr)
    if process.returncode:
        raise RuntimeError(f"prompt preparation failed with exit code {process.returncode}")
    return json.loads(manifest.read_text())


def run_logged(command: list[str], environment: dict[str, str]) -> dict[str, object]:
    log_path = OUTPUT_ROOT / "logs/train.log"
    log_path.parent.mkdir(parents=True, exist_ok=False)
    secret = environment.get("WANDB_API_KEY", "")
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=REMOTE_REPO,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    with log_path.open("x", encoding="utf-8") as stream:
        for line in process.stdout:
            safe = line.replace(secret, "[REDACTED]") if secret else line
            stream.write(safe)
    return_code = process.wait(timeout=30)
    result = {
        "exit_code": return_code,
        "wall_time_seconds": time.monotonic() - started,
        "stdout_sha256": sha256(log_path),
    }
    atomic_json(OUTPUT_ROOT / "train-process.json", result)
    return result


@app.function(
    image=image,
    gpu="L40S:1",
    cpu=8,
    memory=49_152,
    ephemeral_disk=524_288,
    timeout=21_600,
    retries=0,
    single_use_containers=True,
    volumes={"/source": source_volume, "/runs": output_volume},
    secrets=[wandb_secret],
)
def train() -> dict[str, object]:
    result: dict[str, object] = {
        "run_id": RUN_ID,
        "status": "starting",
        "created_at": utcnow(),
        "scientific": scientific_contract(),
        "operational": {
            "repo_head": REPO_HEAD,
            "native_lock_sha256": SIMCT_LOCK_SHA256,
            "gpu": "L40S:1",
            "wandb_entity": WANDB_ENTITY,
            "wandb_project": WANDB_PROJECT,
            "wandb_run_id": WANDB_RUN_ID,
        },
    }
    try:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
        atomic_json(OUTPUT_ROOT / "contract.json", result)
        for required in (STUDENT / "config.json", TEACHER / "config.json", SOURCE_DATA):
            if not required.is_file():
                raise FileNotFoundError(str(required))
        if sha256(SOURCE_DATA) != SOURCE_DATA_SHA256:
            raise RuntimeError("source data SHA mismatch")
        if sha256(REMOTE_REPO / "kdflow/algorithms/span_ctkd.py") != SPAN_CTKD_SHA256:
            raise RuntimeError("SimCT implementation SHA mismatch")
        if sha256(REMOTE_REPO / "experiments/environments/simct/uv.lock") != SIMCT_LOCK_SHA256:
            raise RuntimeError("native environment lock SHA mismatch")

        environment = clean_environment()
        linker = subprocess.run(
            [
                PYTHON,
                "-c",
                (
                    "import ctypes,shutil; ctypes.CDLL('libcudart.so.12'); "
                    "assert shutil.which('nvcc'); print('libcudart_load=pass nvcc=pass')"
                ),
            ],
            cwd=REMOTE_REPO,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        (OUTPUT_ROOT / "cuda-linker-gate.log").write_text(linker.stdout + linker.stderr)
        if linker.returncode:
            raise RuntimeError(f"CUDA linker gate failed with exit code {linker.returncode}")
        prompt_manifest = prepare_prompts(environment)
        unit = subprocess.run(
            [
                PYTHON,
                "-m",
                "pytest",
                "-q",
                "tests/test_span_ctkd_metrics.py",
                "tests/test_teacher_forward_chunking.py",
            ],
            cwd=REMOTE_REPO,
            env=environment,
            capture_output=True,
            text=True,
            timeout=300,
        )
        unit_log = (unit.stdout + "\n" + unit.stderr).replace(
            environment.get("WANDB_API_KEY", ""), "[REDACTED]"
        )
        (OUTPUT_ROOT / "unit-test.log").write_text(unit_log)
        if unit.returncode:
            raise RuntimeError(f"SimCT unit test failed with exit code {unit.returncode}")

        save_path = OUTPUT_ROOT / "checkpoint"
        command = [
            PYTHON,
            "-m",
            "kdflow.cli.train_kd_on_policy",
            "--num_nodes", "1",
            "--num_gpus_per_node", "1",
            "--backend", "fsdp2",
            "--student_name_or_path", str(STUDENT),
            "--teacher_name_or_path", str(TEACHER),
            "--attn_implementation", "sdpa",
            "--num_epochs", "1",
            "--train_batch_size", "64",
            "--micro_train_batch_size", "1",
            "--learning_rate", "5e-5",
            "--lr_warmup_ratio", "0.0",
            "--lr_scheduler", "cosine_with_min_lr",
            "--weight_decay", "0.0",
            "--max_norm", "1.0",
            "--gradient_checkpointing", "True",
            "--enable_sleep", "True",
            "--bf16", "True",
            "--full_determinism", "True",
            "--seed", "42",
            "--save_path", str(save_path),
            "--train_dataset_path", str(OUTPUT_ROOT / "data/opd-prompts.jsonl"),
            "--input_key", "text",
            "--apply_chat_template", "False",
            "--max_samples", "640",
            "--prompt_max_len", "240",
            "--max_len", "2048",
            "--preprocess_num_workers", "8",
            "--rollout_num_engines", "1",
            "--rollout_tp_size", "1",
            "--rollout_mem_fraction_static", "0.20",
            "--rollout_batch_size", "64",
            "--generate_max_len", "1808",
            "--n_samples_per_prompt", "1",
            "--temperature", "1.0",
            "--top_p", "1.0",
            "--teacher_tp_size", "1",
            "--teacher_pp_size", "1",
            "--teacher_ep_size", "1",
            "--teacher_dp_size", "1",
            "--teacher_mem_fraction_static", "0.28",
            "--teacher_context_length", "4096",
            "--teacher_forward_n_batches", "8",
            "--kd_ratio", "1.0",
            "--kd_algorithm", "span_ctkd",
            "--kd_loss_fn", "rkl",
            "--kd_temperature", "1.0",
            "--span_gh_mask_threshold", "2.0",
            "--logging_steps", "1",
            "--use_wandb", "True",
            "--wandb_org", WANDB_ENTITY,
            "--wandb_project", WANDB_PROJECT,
            "--wandb_group", "xtoken-opd-vs-simct-opd-llama1b-qwen4b",
            "--wandb_run_name", WANDB_RUN_NAME,
            "--wandb_run_id", WANDB_RUN_ID,
            "--wandb_job_type", "opd-train",
            "--wandb_tags", "opd,simct,span-ctkd,rkl,modal,l40sx1,10-update,llama-3.2-1b,qwen3-4b",
            "--wandb_mode", "online",
            "--wandb_dir", str(OUTPUT_ROOT / "wandb"),
        ]
        atomic_json(
            OUTPUT_ROOT / "invocation.json",
            {"command": command, "prompt_manifest": prompt_manifest, "created_at": utcnow()},
        )
        result["status"] = "training"
        atomic_json(OUTPUT_ROOT / "result.json", result)
        process = run_logged(command, environment)
        result["process"] = process
        if process["exit_code"] != 0:
            raise RuntimeError("training process failed; no automatic retry")
        summary_path = save_path / "run-summary.json"
        if not summary_path.is_file():
            raise RuntimeError("missing training run-summary.json")
        summary = json.loads(summary_path.read_text())
        if summary.get("optimizer_updates") != 10:
            raise RuntimeError(f"optimizer update gate failed: {summary}")
        result.update(status="completed_pending_wandb_audit", training_summary=summary)
    except BaseException as error:
        result.update(
            status="stopped",
            error_type=type(error).__name__,
            error=str(error),
            traceback="".join(traceback.format_exception_only(type(error), error)).strip(),
        )
    finally:
        result["finished_at"] = utcnow()
        if OUTPUT_ROOT.exists():
            atomic_json(OUTPUT_ROOT / "result.json", result)
            output_volume.commit()
    return result


@app.local_entrypoint()
def main() -> None:
    result = train.remote()
    print(json.dumps({"modal_result": result}, sort_keys=True), flush=True)
    if result["status"] == "stopped":
        raise SystemExit(1)
