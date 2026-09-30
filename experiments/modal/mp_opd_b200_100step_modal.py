"""Run one evidence-gated 100-update MP-OPD atomic canary on a Modal B200."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import traceback

import modal


APP_NAME = "vdt-mp-opd-b200-atomic-no1ceboy-20260905-r7"
RUN_ID = "mp-opd-b200-atomic-100update-20260905-r7"
VOLUME_NAME = "vdt-mp-opd-b200-atomic-no1ceboy-20260905-r7"
SOURCE_VOLUME_NAME = "vdt-xtoken-phase-a-no1ceboy-20260904-r14"
WANDB_SECRET_NAME = "vdt-xtoken-wandb-no1ceboy"
WANDB_ENTITY = "kieusontung8-hanoi-university-of-science-and-technology"
WANDB_PROJECT = "vdt-simct-tunix-reproduction"
WANDB_RUN_ID = "mp-opd-b200-atomic-r7-38c54ef"
WANDB_RUN_NAME = "mp-opd-b200-atomic-100update-r7"

STUDENT_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
TEACHER_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
SOURCE_DATA_SHA256 = "9f2a6a657e5e7575eb90bce59df5e385a68efdd37ce165b881e757f993a10b5c"
B200_LOCK_SHA256 = "ae9d1c9acf536a1b2e65e515207c4386cd3f6bcb33155446def0b64cb98444e3"
MP_OPD_SHA256 = "61adc81fea034de5da56845488ddb54cd73a7890d98f515a3ef02d056efed37a"
MP_OPD_ATOMS_SHA256 = "4f6ce015ab81ea62a1bc2aa5d2bd66a47e2dbb986e01e9813b3a1527be35ce61"
MP_OPD_CREDIT_SHA256 = "c9421290454ef85cd99cbbd166a38e386bc6d28c350aeebd5d5fdc51b06408df"
MP_OPD_SEMIMARKOV_SHA256 = "b8880fed36bbf724ae7053de47df39f251ddff9822b2927f3a85a670f1c69044"
RING_ATTN_UTILS_SHA256 = "02e55701ec24de36e7512675c8e3ce94480702120c4b6cdfcc3a90bf0bf2e19b"
LOGGING_ARGS_SHA256 = "19b125ad7d5e3935a56a6f325108c1c2ecdbb47bfb66c39987113faad436fad0"
TENSORBOARD_UTILS_SHA256 = "30327d6986eb9223a11a8726a40cfc0609948ab147401468ff6993ceaf2c9429"
ON_POLICY_TRAINER_SHA256 = "e3aadcb1040d53d81d162c45eb3714bbbd59e840afd59bbea3ede76c12dde364"
TENSORBOARD_TEST_SHA256 = "1625f729e3485576dfb8bf05874263666c3aa118a999e46325d811c7a28d69e6"
LOGGING_REGRESSION_TEST_SHA256 = "ac848c6168421c26212f9e1a1a84488626a3269edc7007d19edd34d3590e8642"
REPO_BASE_HEAD = "38c54efd4aaffbd45e07d8a97dbfdc8bc30c1d56"

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

UNIQUE_PROMPTS = 640
PROMPT_REPEATS = 10
TRAINING_ROWS = UNIQUE_PROMPTS * PROMPT_REPEATS
ROLLOUT_BATCH_SIZE = 64
TRAIN_BATCH_SIZE = 64
OPTIMIZER_UPDATES = TRAINING_ROWS // ROLLOUT_BATCH_SIZE


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
        "evidence_label": "implementation_validation",
        "regime": "on-policy distillation from student rollouts",
        "algorithm": "MP-OPD scalar canonical-path credit",
        "mp_opd_mode": "atomic",
        "learned_partition_claim": False,
        "student": "meta-llama/Llama-3.2-1B",
        "student_revision": STUDENT_REVISION,
        "teacher": "Qwen/Qwen3-4B",
        "teacher_revision": TEACHER_REVISION,
        "optimizer_updates": OPTIMIZER_UPDATES,
        "sequence_length": 2048,
        "prompt_max_tokens": 240,
        "generation_max_tokens": 1808,
        "teacher_alignment_context_length": 4096,
        "teacher_forward_chunk_microbatches": 8,
        "rollout_batch_size": ROLLOUT_BATCH_SIZE,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "micro_train_batch_size": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "learning_rate": 5e-5,
        "lr_scheduler": "cosine_with_min_lr",
        "lr_scheduler_horizon_steps": OPTIMIZER_UPDATES,
        "lr_warmup_ratio": 0.05,
        "seed": 43,
        "kd_ratio": 1.0,
        "mp_opd_max_span_length": 4,
        "tensorboard_enabled": True,
        "tensorboard_log_dir": str(OUTPUT_ROOT / "tensorboard"),
        "data_source_sha256": SOURCE_DATA_SHA256,
        "unique_prompt_rows": UNIQUE_PROMPTS,
        "deterministic_prompt_repeats": PROMPT_REPEATS,
        "training_rows": TRAINING_ROWS,
        "optimizer_step_equation": (
            f"({TRAINING_ROWS}/{ROLLOUT_BATCH_SIZE}) * "
            f"({ROLLOUT_BATCH_SIZE}*1/{TRAIN_BATCH_SIZE}) = {OPTIMIZER_UPDATES}"
        ),
        "prompt_transform": (
            "first 640 source rows; longest student-tokenizer prefix whose final "
            "encoded prompt including special tokens is <=240 tokens; repeated "
            "deterministically ten times with repeat_index provenance"
        ),
    }


image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("ca-certificates", "git", "build-essential", "libnuma1")
    .pip_install("uv==0.12.7")
    .add_local_file(str(LOCAL_ROOT / "pyproject.toml"), "/opt/repo/pyproject.toml", copy=True)
    .add_local_file(
        str(LOCAL_ROOT / "requirements.txt"), "/opt/repo/requirements.txt", copy=True
    )
    .add_local_file(str(LOCAL_ROOT / "README.md"), "/opt/repo/README.md", copy=True)
    .add_local_dir(str(LOCAL_ROOT / "kdflow"), "/opt/repo/kdflow", copy=True)
    .add_local_dir(str(LOCAL_ROOT / "tests"), "/opt/repo/tests", copy=True)
    .add_local_dir(
        str(LOCAL_ROOT / "experiments/environments/simct-b200"),
        "/opt/repo/experiments/environments/simct-b200",
        copy=True,
    )
    .add_local_file(
        str(LOCAL_ROOT / "experiments/environments/b200_gate.py"),
        "/opt/repo/experiments/environments/b200_gate.py",
        copy=True,
    )
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
    .run_commands(
        "uv python install 3.12.12",
        (
            "env -u UV_INDEX_URL -u UV_DEFAULT_INDEX -u PIP_INDEX_URL "
            "-u PIP_TRUSTED_HOST UV_PROJECT_ENVIRONMENT=/opt/kdflow/venv "
            "UV_LINK_MODE=copy UV_HTTP_TIMEOUT=600 uv sync "
            "--project /opt/repo/experiments/environments/simct-b200 --locked"
        ),
    )
)

app = modal.App(APP_NAME)
source_volume = modal.Volume.from_name(SOURCE_VOLUME_NAME, create_if_missing=False)
output_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
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
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        TOKENIZERS_PARALLELISM="false",
        NCCL_CUMEM_HOST_ENABLE="0",
        RAY_USAGE_STATS_ENABLED="0",
        WANDB_MODE="online",
        WANDB_SILENT="true",
        TORCH_CUDA_ARCH_LIST="10.0",
        CUDAARCHS="100",
        CMAKE_CUDA_ARCHITECTURES="100",
        OMP_NUM_THREADS="4",
        CUDA_HOME="/usr/local/cuda",
        LD_LIBRARY_PATH=":".join(cuda_library_dirs),
    )
    return environment


def prepare_prompts(environment: dict[str, str]) -> dict[str, object]:
    base_path = OUTPUT_ROOT / "data/opd-prompts-640.jsonl"
    base_manifest_path = OUTPUT_ROOT / "data/base-manifest.json"
    process = subprocess.run(
        [
            PYTHON,
            "/opt/kdflow-vdt/prepare_opd_prompts.py",
            "--source",
            str(SOURCE_DATA),
            "--source-sha256",
            SOURCE_DATA_SHA256,
            "--student",
            str(STUDENT),
            "--destination",
            str(base_path),
            "--manifest",
            str(base_manifest_path),
            "--rows",
            str(UNIQUE_PROMPTS),
            "--max-tokens",
            "240",
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

    base_rows = [json.loads(line) for line in base_path.read_text().splitlines() if line]
    if len(base_rows) != UNIQUE_PROMPTS:
        raise RuntimeError(f"expected {UNIQUE_PROMPTS} base prompts, found {len(base_rows)}")
    training_path = OUTPUT_ROOT / "data/opd-prompts-6400.jsonl"
    with training_path.open("x", encoding="utf-8") as stream:
        for repeat_index in range(PROMPT_REPEATS):
            for row in base_rows:
                record = dict(row)
                record["repeat_index"] = repeat_index
                stream.write(json.dumps(record, sort_keys=True) + "\n")

    base_manifest = json.loads(base_manifest_path.read_text())
    manifest = {
        **base_manifest,
        "base_output_sha256": base_manifest["output_sha256"],
        "output_sha256": sha256(training_path),
        "unique_rows": UNIQUE_PROMPTS,
        "repeat_count": PROMPT_REPEATS,
        "rows": TRAINING_ROWS,
        "training_path": str(training_path),
    }
    atomic_json(OUTPUT_ROOT / "data/manifest.json", manifest)
    return manifest


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
            stream.write(line.replace(secret, "[REDACTED]") if secret else line)
    result = {
        "exit_code": process.wait(timeout=30),
        "wall_time_seconds": time.monotonic() - started,
        "stdout_sha256": sha256(log_path),
    }
    atomic_json(OUTPUT_ROOT / "train-process.json", result)
    return result


@app.function(
    image=image,
    cpu=4,
    memory=16_384,
    timeout=1_800,
    retries=0,
    single_use_containers=True,
    volumes={"/source": source_volume},
)
def cpu_preflight() -> dict[str, object]:
    result: dict[str, object] = {
        "status": "starting",
        "created_at": utcnow(),
        "run_id": RUN_ID,
        "gpu_allocated": False,
    }
    try:
        for required in (STUDENT / "config.json", TEACHER / "config.json", SOURCE_DATA):
            if not required.is_file():
                raise FileNotFoundError(str(required))
        if sha256(SOURCE_DATA) != SOURCE_DATA_SHA256:
            raise RuntimeError("source data SHA mismatch")
        if sha256(REMOTE_REPO / "experiments/environments/simct-b200/uv.lock") != B200_LOCK_SHA256:
            raise RuntimeError("native B200 environment lock SHA mismatch")

        environment = clean_environment()
        environment["KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT"] = "1"
        destination = Path("/tmp/mp-opd-preflight/data/prompts.jsonl")
        manifest_path = Path("/tmp/mp-opd-preflight/data/manifest.json")
        prompt_probe = subprocess.run(
            [
                PYTHON,
                "/opt/kdflow-vdt/prepare_opd_prompts.py",
                "--source",
                str(SOURCE_DATA),
                "--source-sha256",
                SOURCE_DATA_SHA256,
                "--student",
                str(STUDENT),
                "--destination",
                str(destination),
                "--manifest",
                str(manifest_path),
                "--rows",
                "2",
                "--max-tokens",
                "240",
            ],
            cwd=REMOTE_REPO,
            env=environment,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if prompt_probe.returncode:
            raise RuntimeError(
                "CPU tokenizer/data probe failed: "
                + (prompt_probe.stderr or prompt_probe.stdout)[-2000:]
            )
        prompt_manifest = json.loads(manifest_path.read_text())
        unit = subprocess.run(
            [
                PYTHON,
                "-m",
                "pytest",
                "-q",
                "tests/mp_opd",
                "tests/test_span_ctkd_metrics.py",
                "tests/test_xtoken_algorithm.py",
                "tests/test_ring_attn_utils.py",
                "tests/test_tensorboard_logging.py",
                "tests/test_on_policy_logging.py",
            ],
            cwd=REMOTE_REPO,
            env=environment,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if unit.returncode:
            raise RuntimeError("CPU MP-OPD regression gate failed: " + unit.stdout[-2000:])
        result.update(
            status="pass",
            prompt_probe_manifest=prompt_manifest,
            unit_test_tail=unit.stdout[-1000:],
        )
    except BaseException as error:
        result.update(
            status="stopped",
            error_type=type(error).__name__,
            error=str(error),
        )
    result["finished_at"] = utcnow()
    return result


@app.function(
    image=image,
    gpu="B200",
    cpu=16,
    memory=98_304,
    ephemeral_disk=524_288,
    timeout=21_600,
    retries=0,
    single_use_containers=True,
    volumes={"/source": source_volume, "/runs": output_volume},
    secrets=[wandb_secret],
)
def train(preflight_result: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {
        "run_id": RUN_ID,
        "status": "starting",
        "created_at": utcnow(),
        "scientific": scientific_contract(),
        "cpu_preflight": preflight_result,
        "operational": {
            "repo_base_head": REPO_BASE_HEAD,
            "retry_of": "mp-opd-b200-atomic-100update-20260904-r6",
            "retry_reason": (
                "r6 completed optimizer updates 1 and 2, then stopped while logging "
                "update 3 because a step-local MP-OPD invalid-reason metric was absent "
                "and its retained empty accumulator was formatted as a scalar; r7 "
                "drops inactive dynamic metrics between logging steps without changing "
                "the scientific configuration"
            ),
            "native_lock_sha256": B200_LOCK_SHA256,
            "gpu": "B200:1",
            "wandb_entity": WANDB_ENTITY,
            "wandb_project": WANDB_PROJECT,
            "wandb_run_id": WANDB_RUN_ID,
        },
    }
    try:
        if preflight_result.get("status") != "pass":
            raise RuntimeError("CPU preflight did not pass")
        if OPTIMIZER_UPDATES != 100:
            raise RuntimeError(f"static optimizer-step equation is {OPTIMIZER_UPDATES}, not 100")
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
        atomic_json(OUTPUT_ROOT / "contract.json", result)
        for required in (STUDENT / "config.json", TEACHER / "config.json", SOURCE_DATA):
            if not required.is_file():
                raise FileNotFoundError(str(required))
        if sha256(SOURCE_DATA) != SOURCE_DATA_SHA256:
            raise RuntimeError("source data SHA mismatch")
        source_hashes = {
            "kdflow/algorithms/mp_opd.py": MP_OPD_SHA256,
            "kdflow/algorithms/_mp_opd_atoms.py": MP_OPD_ATOMS_SHA256,
            "kdflow/algorithms/_mp_opd_credit.py": MP_OPD_CREDIT_SHA256,
            "kdflow/algorithms/_mp_opd_semimarkov.py": MP_OPD_SEMIMARKOV_SHA256,
            "kdflow/models/ring_attn_utils.py": RING_ATTN_UTILS_SHA256,
            "kdflow/arguments/logging_args.py": LOGGING_ARGS_SHA256,
            "kdflow/utils/tensorboard_utils.py": TENSORBOARD_UTILS_SHA256,
            "kdflow/trainer/on_policy_kd_trainer.py": ON_POLICY_TRAINER_SHA256,
            "tests/test_tensorboard_logging.py": TENSORBOARD_TEST_SHA256,
            "tests/test_on_policy_logging.py": LOGGING_REGRESSION_TEST_SHA256,
        }
        for name, expected in source_hashes.items():
            actual = sha256(REMOTE_REPO / name)
            if actual != expected:
                raise RuntimeError(f"source SHA mismatch for {name}")
        if sha256(REMOTE_REPO / "experiments/environments/simct-b200/uv.lock") != B200_LOCK_SHA256:
            raise RuntimeError("native B200 environment lock SHA mismatch")

        environment = clean_environment()
        environment.update(
            WANDB_ENTITY=WANDB_ENTITY,
            WANDB_PROJECT=WANDB_PROJECT,
            WANDB_RUN_ID=WANDB_RUN_ID,
            WANDB_RESUME="never",
        )
        gate = subprocess.run(
            [PYTHON, "/opt/repo/experiments/environments/b200_gate.py"],
            cwd=REMOTE_REPO,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        (OUTPUT_ROOT / "b200-environment-gate.log").write_text(gate.stdout + gate.stderr)
        if gate.returncode:
            raise RuntimeError("B200 environment gate failed")
        marker = next(
            line.removeprefix("SIMCT_B200_ENV_JSON=")
            for line in gate.stdout.splitlines()
            if line.startswith("SIMCT_B200_ENV_JSON=")
        )
        environment_gate = json.loads(marker)
        prompt_manifest = prepare_prompts(environment)

        save_path = OUTPUT_ROOT / "checkpoint"
        command = [
            PYTHON,
            "-m",
            "kdflow.cli.train_kd_on_policy",
            "--num_nodes",
            "1",
            "--num_gpus_per_node",
            "1",
            "--backend",
            "fsdp2",
            "--student_name_or_path",
            str(STUDENT),
            "--teacher_name_or_path",
            str(TEACHER),
            "--attn_implementation",
            "sdpa",
            "--num_epochs",
            "1",
            "--train_batch_size",
            str(TRAIN_BATCH_SIZE),
            "--micro_train_batch_size",
            "1",
            "--learning_rate",
            "5e-5",
            "--lr_warmup_ratio",
            "0.05",
            "--lr_scheduler",
            "cosine_with_min_lr",
            "--lr_scheduler_horizon_steps",
            str(OPTIMIZER_UPDATES),
            "--weight_decay",
            "0.0",
            "--max_norm",
            "1.0",
            "--gradient_checkpointing",
            "True",
            "--enable_sleep",
            "True",
            "--bf16",
            "True",
            "--full_determinism",
            "True",
            "--seed",
            "43",
            "--save_path",
            str(save_path),
            "--ckpt_path",
            str(OUTPUT_ROOT / "checkpoints"),
            "--train_dataset_path",
            str(OUTPUT_ROOT / "data/opd-prompts-6400.jsonl"),
            "--input_key",
            "text",
            "--apply_chat_template",
            "False",
            "--max_samples",
            str(TRAINING_ROWS),
            "--prompt_max_len",
            "240",
            "--max_len",
            "2048",
            "--preprocess_num_workers",
            "8",
            "--rollout_num_engines",
            "1",
            "--rollout_tp_size",
            "1",
            "--rollout_mem_fraction_static",
            "0.20",
            "--rollout_batch_size",
            str(ROLLOUT_BATCH_SIZE),
            "--generate_max_len",
            "1808",
            "--n_samples_per_prompt",
            "1",
            "--temperature",
            "1.0",
            "--top_p",
            "1.0",
            "--teacher_tp_size",
            "1",
            "--teacher_pp_size",
            "1",
            "--teacher_ep_size",
            "1",
            "--teacher_dp_size",
            "1",
            "--teacher_mem_fraction_static",
            "0.28",
            "--teacher_context_length",
            "4096",
            "--teacher_forward_n_batches",
            "8",
            "--kd_algorithm",
            "mp_opd",
            "--kd_ratio",
            "1.0",
            "--mp_opd_mode",
            "atomic",
            "--mp_opd_max_span_length",
            "4",
            "--mp_opd_fixed_span_length",
            "2",
            "--mp_opd_random_seed",
            "43",
            "--mp_opd_partition_temperature",
            "1.0",
            "--save_steps",
            str(OPTIMIZER_UPDATES),
            "--logging_steps",
            "1",
            "--use_tensorboard",
            "True",
            "--tensorboard_log_dir",
            str(OUTPUT_ROOT / "tensorboard"),
            "--tensorboard_flush_secs",
            "10",
            "--use_wandb",
            "True",
            "--wandb_org",
            WANDB_ENTITY,
            "--wandb_project",
            WANDB_PROJECT,
            "--wandb_group",
            "mp-opd-b200-llama1b-qwen4b",
            "--wandb_run_name",
            WANDB_RUN_NAME,
            "--wandb_run_id",
            WANDB_RUN_ID,
            "--wandb_job_type",
            "implementation-validation",
            "--wandb_tags",
            "mp-opd,atomic,b200,100-update,implementation-validation,llama-3.2-1b,qwen3-4b",
            "--wandb_mode",
            "online",
            "--wandb_dir",
            str(OUTPUT_ROOT / "wandb"),
        ]
        atomic_json(
            OUTPUT_ROOT / "invocation.json",
            {
                "command": command,
                "prompt_manifest": prompt_manifest,
                "environment_gate": environment_gate,
                "source_hashes": source_hashes,
                "created_at": utcnow(),
            },
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
        if summary.get("optimizer_updates") != OPTIMIZER_UPDATES:
            raise RuntimeError(f"optimizer update gate failed: {summary}")
        if summary.get("kd_algorithm") != "mp_opd":
            raise RuntimeError(f"algorithm identity gate failed: {summary}")
        if not math.isfinite(float(summary.get("total_time_seconds", float("nan")))):
            raise RuntimeError(f"non-finite training time: {summary}")
        result.update(
            status="completed_pending_wandb_audit",
            training_summary=summary,
            environment_gate=environment_gate,
            prompt_manifest=prompt_manifest,
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
            output_volume.commit()
    return result


@app.local_entrypoint()
def preflight() -> None:
    """Run and print the CPU-only gate without allocating a B200."""
    result = cpu_preflight.remote()
    print("MP_OPD_CPU_PREFLIGHT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    if result["status"] != "pass":
        raise SystemExit(1)


@app.local_entrypoint()
def main() -> None:
    preflight_result = cpu_preflight.remote()
    print("MP_OPD_CPU_PREFLIGHT_JSON=" + json.dumps(preflight_result, sort_keys=True), flush=True)
    if preflight_result["status"] != "pass":
        raise SystemExit(1)
    result = train.remote(preflight_result)
    print(json.dumps({"modal_result": result}, sort_keys=True), flush=True)
    if result["status"] == "stopped":
        raise SystemExit(1)
