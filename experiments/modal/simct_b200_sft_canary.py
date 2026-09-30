"""Build the portable B200 stack and run a five-update public-model SFT canary."""

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


APP_NAME = "vdt-simct-b200-sft-no1ceboy-20260904-r6"
RUN_ID = "simct-b200-sft-5update-20260904-r6"
VOLUME_NAME = "vdt-simct-b200-qualification-no1ceboy-r6"
ASSET_VOLUME_NAME = "vdt-simct-b200-assets-no1ceboy-v2-public-qwen"
HF_SECRET_NAME = "vdt-xtoken-hf-no1ceboy"
WANDB_SECRET_NAME = "vdt-xtoken-wandb-no1ceboy"
WANDB_ENTITY = "kieusontung8-hanoi-university-of-science-and-technology"
WANDB_PROJECT = "vdt-simct-tunix-reproduction"
WANDB_RUN_ID = "simct-b200-sft-r6-9d5c801"

STUDENT_ID = "Qwen/Qwen2.5-1.5B-Instruct"
STUDENT_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
TEACHER_ID = "microsoft/Phi-4-mini-instruct"
TEACHER_REVISION = "cfbefacb99257ffa30c83adab238a50856ac3083"
DATASET_ID = "openai/gsm8k"
DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
REPO_HEAD = "9d5c801ebbaed8c1b7acf7a235876da32a2afb65"

REMOTE_ROOT = Path("/opt/repo")
LOCAL_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE_ROOT
ASSET_ROOT = Path("/assets")
OUTPUT_ROOT = Path("/runs") / RUN_ID
SFT_PYTHON = "/opt/sft/venv/bin/python"
SFT_CLI = "/opt/sft/venv/bin/llamafactory-cli"
SIMCT_PYTHON = "/opt/simct/venv/bin/python"


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


asset_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "datasets==4.0.0",
    "huggingface-hub==1.30.0",
)

runtime_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("ca-certificates", "git", "build-essential", "libnuma1")
    .pip_install("uv==0.12.7")
    .add_local_file(
        str(LOCAL_ROOT / "experiments/environments/simct-b200-sft/pyproject.toml"),
        "/opt/env/sft/pyproject.toml",
        copy=True,
    )
    .add_local_file(
        str(LOCAL_ROOT / "experiments/environments/simct-b200-sft/uv.lock"),
        "/opt/env/sft/uv.lock",
        copy=True,
    )
    .add_local_file(
        str(LOCAL_ROOT / "experiments/environments/simct-b200/pyproject.toml"),
        "/opt/repo/experiments/environments/simct-b200/pyproject.toml",
        copy=True,
    )
    .add_local_file(
        str(LOCAL_ROOT / "experiments/environments/simct-b200/uv.lock"),
        "/opt/repo/experiments/environments/simct-b200/uv.lock",
        copy=True,
    )
    .add_local_file(str(LOCAL_ROOT / "pyproject.toml"), "/opt/repo/pyproject.toml", copy=True)
    .add_local_file(str(LOCAL_ROOT / "requirements.txt"), "/opt/repo/requirements.txt", copy=True)
    .add_local_dir(str(LOCAL_ROOT / "kdflow"), "/opt/repo/kdflow", copy=True)
    .add_local_file(
        str(LOCAL_ROOT / "experiments/environments/b200_gate.py"),
        "/opt/repo/experiments/environments/b200_gate.py",
        copy=True,
    )
    .add_local_file(
        str(LOCAL_ROOT / "experiments/environments/patch_llamafactory_json_path.py"),
        "/opt/patches/patch_llamafactory_json_path.py",
        copy=True,
    )
    .run_commands(
        (
            "env -u UV_INDEX_URL -u UV_DEFAULT_INDEX -u PIP_INDEX_URL "
            "-u PIP_TRUSTED_HOST UV_PROJECT_ENVIRONMENT=/opt/sft/venv "
            "UV_LINK_MODE=copy UV_HTTP_TIMEOUT=600 uv sync --project /opt/env/sft --locked"
        ),
        "/opt/sft/venv/bin/python /opt/patches/patch_llamafactory_json_path.py",
        (
            "env -u UV_INDEX_URL -u UV_DEFAULT_INDEX -u PIP_INDEX_URL "
            "-u PIP_TRUSTED_HOST UV_PROJECT_ENVIRONMENT=/opt/simct/venv "
            "UV_LINK_MODE=copy UV_HTTP_TIMEOUT=600 uv sync "
            "--project /opt/repo/experiments/environments/simct-b200 --locked"
        ),
    )
)

app = modal.App(APP_NAME)
assets = modal.Volume.from_name(ASSET_VOLUME_NAME, create_if_missing=True)
outputs = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)
wandb_secret = modal.Secret.from_name(WANDB_SECRET_NAME)


@app.function(
    image=asset_image,
    cpu=4,
    memory=16_384,
    timeout=7_200,
    retries=0,
    volumes={"/assets": assets},
    secrets=[hf_secret],
)
def prepare_assets() -> dict[str, object]:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    model_cache = ASSET_ROOT / "hf"
    student = Path(
        snapshot_download(
            STUDENT_ID,
            revision=STUDENT_REVISION,
            cache_dir=model_cache,
        )
    )
    teacher = Path(
        snapshot_download(
            TEACHER_ID,
            revision=TEACHER_REVISION,
            cache_dir=model_cache,
        )
    )
    dataset = load_dataset(
        DATASET_ID,
        "main",
        split="train[:640]",
        revision=DATASET_REVISION,
    )
    data_root = ASSET_ROOT / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    examples = [
        {
            "conversations": [
                {"from": "human", "value": str(row["question"])},
                {"from": "gpt", "value": str(row["answer"])},
            ]
        }
        for row in dataset
    ]
    data_path = data_root / "gsm8k-sft-canary.json"
    data_path.write_text(json.dumps(examples, ensure_ascii=False) + "\n")
    dataset_info = {
        "gsm8k_sft_canary": {
            "file_name": data_path.name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }
    }
    atomic_json(data_root / "dataset_info.json", dataset_info)
    manifest = {
        "status": "ready",
        "created_at": utcnow(),
        "student": {"id": STUDENT_ID, "revision": STUDENT_REVISION, "path": str(student)},
        "teacher": {"id": TEACHER_ID, "revision": TEACHER_REVISION, "path": str(teacher)},
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "rows": len(examples),
            "sha256": sha256(data_path),
        },
    }
    atomic_json(ASSET_ROOT / "manifest.json", manifest)
    assets.commit()
    return manifest


def clean_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("UV_", "PIP_"))
        and not any(marker in key.upper() for marker in ("HF_TOKEN", "KAGGLE", "PASSWORD"))
    }
    environment.update(
        PATH=f"/opt/sft/venv/bin:{environment.get('PATH', '/usr/local/bin:/usr/bin:/bin')}",
        PYTHONUNBUFFERED="1",
        HF_HOME=str(ASSET_ROOT / "hf"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        TORCH_CUDA_ARCH_LIST="10.0",
        CUDAARCHS="100",
        CMAKE_CUDA_ARCHITECTURES="100",
        FORCE_TORCHRUN="1",
        NNODES="1",
        NPROC_PER_NODE="1",
        WANDB_MODE="online",
        WANDB_SILENT="true",
        WANDB_RUN_ID=WANDB_RUN_ID,
        WANDB_RESUME="never",
    )
    return environment


def run_logged(command: list[str], environment: dict[str, str], log_path: Path) -> dict[str, object]:
    secret = environment.get("WANDB_API_KEY", "")
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=REMOTE_ROOT,
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
            stream.write(line.replace(secret, "[REDACTED]") if secret else line)
    return {
        "exit_code": process.wait(timeout=30),
        "wall_time_seconds": time.monotonic() - started,
        "stdout_sha256": sha256(log_path),
    }


@app.function(
    image=runtime_image,
    gpu="B200",
    cpu=12,
    memory=65_536,
    # Modal currently requires at least 512 GiB of ephemeral disk for B200.
    ephemeral_disk=524_288,
    timeout=7_200,
    retries=0,
    single_use_containers=True,
    volumes={"/assets": assets, "/runs": outputs},
    secrets=[wandb_secret],
)
def sft_canary() -> dict[str, object]:
    result: dict[str, object] = {
        "run_id": RUN_ID,
        "status": "starting",
        "created_at": utcnow(),
        "repo_head": REPO_HEAD,
        "qualification_only": True,
        "scientific_target_blocked": "google/gemma-2-2b-it requires Hugging Face approval",
        "optimizer_updates": 5,
    }
    try:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
        environment = clean_environment()
        manifest = json.loads((ASSET_ROOT / "manifest.json").read_text())
        if manifest["student"]["revision"] != STUDENT_REVISION:
            raise RuntimeError("student revision mismatch")
        if manifest["teacher"]["revision"] != TEACHER_REVISION:
            raise RuntimeError("teacher revision mismatch")

        gates: dict[str, object] = {}
        for name, python in (("sft", SFT_PYTHON), ("simct", SIMCT_PYTHON)):
            gate = subprocess.run(
                [python, "/opt/repo/experiments/environments/b200_gate.py"],
                cwd=REMOTE_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
            )
            gate_log = OUTPUT_ROOT / f"{name}-environment-gate.log"
            gate_log.write_text(gate.stdout + gate.stderr)
            if gate.returncode:
                raise RuntimeError(f"{name} B200 environment gate failed")
            marker = next(
                line.removeprefix("SIMCT_B200_ENV_JSON=")
                for line in gate.stdout.splitlines()
                if line.startswith("SIMCT_B200_ENV_JSON=")
            )
            gates[name] = json.loads(marker)

        student_path = manifest["student"]["path"]
        config = {
            "model_name_or_path": student_path,
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "full",
            "dataset_dir": str(ASSET_ROOT / "data"),
            "dataset": "gsm8k_sft_canary",
            "template": "qwen",
            "cutoff_len": 2048,
            "preprocessing_num_workers": 8,
            "packing": False,
            "output_dir": str(OUTPUT_ROOT / "checkpoint"),
            "logging_steps": 1,
            "save_strategy": "steps",
            "save_steps": 5,
            "save_total_limit": 1,
            "save_only_model": False,
            "overwrite_output_dir": False,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 16,
            "learning_rate": 2e-6,
            "max_steps": 5,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "bf16": True,
            "pure_bf16": True,
            "gradient_checkpointing": True,
            "ddp_timeout": 180000000,
            "report_to": "wandb",
            "run_name": "simct-b200-qwen25-sft-5update-r6",
        }
        config_path = OUTPUT_ROOT / "sft-canary.json"
        atomic_json(config_path, config)
        atomic_json(
            OUTPUT_ROOT / "contract.json",
            {
                **result,
                "environment_gates": gates,
                "assets": manifest,
                "sft": {
                    "student": STUDENT_ID,
                    "student_revision": STUDENT_REVISION,
                    "dataset": DATASET_ID,
                    "dataset_revision": DATASET_REVISION,
                    "micro_batch_size": 4,
                    "gradient_accumulation_steps": 16,
                    "effective_batch_size": 64,
                    "max_steps": 5,
                    "cutoff_len": 2048,
                    "learning_rate": 2e-6,
                    "warmup_ratio": 0.05,
                },
                "wandb": {
                    "entity": WANDB_ENTITY,
                    "project": WANDB_PROJECT,
                    "run_id": WANDB_RUN_ID,
                },
            },
        )
        environment.update(
            WANDB_ENTITY=WANDB_ENTITY,
            WANDB_PROJECT=WANDB_PROJECT,
        )
        result["status"] = "training"
        atomic_json(OUTPUT_ROOT / "result.json", result)
        process = run_logged(
            [SFT_CLI, "train", str(config_path)],
            environment,
            OUTPUT_ROOT / "sft.log",
        )
        result["process"] = process
        if process["exit_code"] != 0:
            raise RuntimeError("SFT canary failed; no automatic retry")

        state_files = sorted((OUTPUT_ROOT / "checkpoint").glob("checkpoint-*/trainer_state.json"))
        if len(state_files) != 1:
            raise RuntimeError(f"expected one trainer state, found {len(state_files)}")
        trainer_state = json.loads(state_files[0].read_text())
        if int(trainer_state.get("global_step", -1)) != 5:
            raise RuntimeError("SFT optimizer-step gate failed")
        losses = [row.get("loss") for row in trainer_state.get("log_history", []) if "loss" in row]
        if len(losses) < 5 or not all(isinstance(value, (int, float)) for value in losses):
            raise RuntimeError("missing finite per-step SFT losses")
        result.update(
            status="completed",
            environment_gates=gates,
            trainer_state_sha256=sha256(state_files[0]),
            loss_values=losses,
        )
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
            outputs.commit()
    return result


@app.local_entrypoint()
def main() -> None:
    asset_result = prepare_assets.remote()
    print("SIMCT_B200_ASSET_JSON=" + json.dumps(asset_result, sort_keys=True), flush=True)
    result = sft_canary.remote()
    print("SIMCT_B200_SFT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    if result["status"] != "completed":
        raise SystemExit(1)
