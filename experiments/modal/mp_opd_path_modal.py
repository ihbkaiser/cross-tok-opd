"""One bounded corrected-trajectory atomic diagnostic on lhtu05."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import modal

IMAGE = "docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f"
RUN_ID = "mp-opd-path-atomic-r8-lhtu05-20260907"
ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/opt/overlay")
PYTHON = "/opt/venvs/simct-b200/bin/python"
STUDENT_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
TEACHER_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
ENTITY = "kieusontung8-hanoi-university-of-science-and-technology"
PROJECT = "vdt-simct-tunix-reproduction"
cache = modal.Volume.from_name("vdt-mp-opd-path-cache", create_if_missing=False)
outputs = modal.Volume.from_name("vdt-mp-opd-path-r8-lhtu05-20260907", create_if_missing=True)
image = modal.Image.from_registry(IMAGE).entrypoint([]).env({
    "PYTHONPATH": "/opt/overlay/experiments/modal/vendor:/opt/overlay"
}).add_local_dir(
    str(ROOT / "kdflow"), "/opt/overlay/kdflow", ignore=["**/__pycache__/**", "**/*.pyc"]
).add_local_dir(str(ROOT / "tests"), "/opt/overlay/tests", ignore=["**/__pycache__/**", "**/*.pyc"]).add_local_dir(
    str(ROOT / "experiments/modal"), "/opt/overlay/experiments/modal", ignore=["**/__pycache__/**", "**/*.pyc"]
)
app = modal.App("vdt-mp-opd-path-r8-lhtu05")


@app.function(image=image, cpu=4, memory=16384, timeout=1800, retries=0,
              volumes={"/model-cache": cache},
              secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")])
def prepare():
    """Download the pinned snapshots on remote CPU; no GPU is allocated."""
    code = '''
import hashlib, importlib.metadata, json, os
from pathlib import Path
from huggingface_hub import snapshot_download
import wandb
wandb.Api(timeout=40).run("kieusontung8-hanoi-university-of-science-and-technology/vdt-simct-tunix-reproduction/mp-opd-b200-atomic-r7-38c54ef")
print("WANDB_EXISTING_RUN_ACCESS=pass", flush=True)
root = Path("/model-cache")
data = root / "data/opd-prompts-6400.jsonl"
assert sum(1 for _ in data.open()) == 6400
print("DATA_SHA256=" + hashlib.sha256(data.read_bytes()).hexdigest(), flush=True)
engine = Path(importlib.metadata.distribution("sglang").locate_file("sglang/srt/entrypoints/engine.py"))
text = engine.read_text()
start = text.index("    def generate(")
print("ENGINE_GENERATE_SIGNATURE=" + text[start:start+2600], flush=True)
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
for model, revision in [
    ("meta-llama/Llama-3.2-1B", "4e20de362430cd3b72f300e6b0f18e50e7166e08"),
    ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"),
]:
    path = snapshot_download(model, revision=revision, cache_dir="/model-cache/hf/hub", token=token,
        allow_patterns=["*.json", "*.safetensors", "*.model", "merges.txt", "vocab.json"], max_workers=4)
    print("PINNED_MODEL_READY=" + json.dumps({"model": model, "revision": revision, "path": path}), flush=True)
print("REMOTE_PREPARE_PASS", flush=True)
'''
    download_env = dict(os.environ, HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0")
    completed = subprocess.run([PYTHON, "-c", code], env=download_env, timeout=1650)
    cache.commit()
    if completed.returncode:
        raise RuntimeError("Pinned model preparation failed; training was not started")
    return {"status": "pass", "image": IMAGE}


DATA_SHA256 = "a0d248ab7c23e09380260a337aad580fabc13fb6d013ce4a9ae20a64f7e9ee9c"


def runtime_environment():
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(("UV_", "PIP_")) or key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
            env.pop(key)
    env.update(PYTHONPATH="/opt/overlay/experiments/modal/vendor:/opt/overlay",
               PATH="/opt/venvs/simct-b200/bin:" + env.get("PATH", ""),
               HF_HOME="/model-cache/hf", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_IMPLICIT_TOKEN="1", TOKENIZERS_PARALLELISM="false",
               RAY_USAGE_STATS_ENABLED="0", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4",
               NCCL_CUMEM_HOST_ENABLE="0", WANDB_SILENT="true")
    nvidia_root = Path("/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia")
    libraries = ["/usr/local/cuda/lib64", "/usr/local/nvidia/lib", "/usr/local/nvidia/lib64"]
    libraries.extend(str(path) for path in sorted(nvidia_root.glob("*/lib")))
    if env.get("LD_LIBRARY_PATH"):
        libraries.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(libraries)
    return env


@app.function(image=image, cpu=4, memory=16384, timeout=1200, retries=0,
              volumes={"/model-cache": cache, "/runs": outputs},
              secrets=[modal.Secret.from_name("wandb-secret")])
def preflight():
    env = runtime_environment()
    env["KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT"] = "1"
    destination = Path("/runs") / RUN_ID
    destination.mkdir(parents=True, exist_ok=True)
    data = Path("/model-cache/data/opd-prompts-6400.jsonl")
    if hashlib.sha256(data.read_bytes()).hexdigest() != DATA_SHA256:
        raise RuntimeError("r7 input data hash mismatch")
    tests = subprocess.run([PYTHON, "-m", "pytest", "-q", "tests/mp_opd",
        "tests/test_trajectory.py", "tests/test_exact_trajectory_integration.py", "tests/test_wandb_schema.py",
        "tests/test_on_policy_logging.py", "tests/test_tensorboard_logging.py"],
        cwd="/opt/overlay", env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1000)
    (destination / "preflight.log").write_text(tests.stdout)
    outputs.commit()
    print(tests.stdout[-6000:], flush=True)
    if tests.returncode:
        raise RuntimeError("CPU regression preflight failed")
    return {"status": "pass", "data_sha256": DATA_SHA256, "image": IMAGE}


def train_impl(source_commit: str, preflight_result: dict):
    from kdflow.wandb_schema import build_wandb_tags, IMPLEMENTATION_VALIDATION_JOB_TYPE
    if preflight_result.get("status") != "pass":
        raise RuntimeError("preflight is required")
    OUTPUT_ROOT = Path("/runs") / RUN_ID
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT_ROOT / "invocation.json").exists():
        raise RuntimeError("existing training invocation: refusing duplicate run")
    STUDENT = Path("/model-cache/hf/hub/models--meta-llama--Llama-3.2-1B/snapshots") / STUDENT_REVISION
    TEACHER = Path("/model-cache/hf/hub/models--Qwen--Qwen3-4B/snapshots") / TEACHER_REVISION
    for model in (STUDENT, TEACHER):
        if not (model / "config.json").is_file() or not list(model.glob("*.safetensors")):
            raise RuntimeError("pinned model is missing")
    TRAIN_BATCH_SIZE = ROLLOUT_BATCH_SIZE = 64
    TRAINING_ROWS = 6400
    OPTIMIZER_UPDATES = 100
    WANDB_ENTITY, WANDB_PROJECT = ENTITY, PROJECT
    WANDB_RUN_NAME = WANDB_RUN_ID = RUN_ID
    WANDB_TAGS = build_wandb_tags(method="mp-opd", regime="on-policy", objective="canonical-path-credit",
        variant="atomic-exact-trajectory", platform="modal", accelerator="b200x1", budget="30-update",
        stage=IMPLEMENTATION_VALIDATION_JOB_TYPE, student="llama-3.2-1b", teacher="qwen3-4b")
    save_path = OUTPUT_ROOT / "checkpoint"
    command = [
        PYTHON,
        '-m',
        'kdflow.cli.train_kd_on_policy',
        '--num_nodes',
        '1',
        '--num_gpus_per_node',
        '1',
        '--backend',
        'fsdp2',
        '--student_name_or_path',
        str(STUDENT),
        '--teacher_name_or_path',
        str(TEACHER),
        '--attn_implementation',
        'sdpa',
        '--num_epochs',
        '1',
        '--train_batch_size',
        str(TRAIN_BATCH_SIZE),
        '--micro_train_batch_size',
        '1',
        '--learning_rate',
        '5e-5',
        '--lr_warmup_ratio',
        '0.05',
        '--lr_scheduler',
        'cosine_with_min_lr',
        '--lr_scheduler_horizon_steps',
        str(OPTIMIZER_UPDATES),
        '--weight_decay',
        '0.0',
        '--max_norm',
        '1.0',
        '--gradient_checkpointing',
        'True',
        '--enable_sleep',
        'True',
        '--bf16',
        'True',
        '--full_determinism',
        'True',
        '--seed',
        '43',
        '--save_path',
        str(save_path),
        '--ckpt_path',
        str(OUTPUT_ROOT / 'checkpoints'),
        '--train_dataset_path',
        str(OUTPUT_ROOT / 'data/opd-prompts-6400.jsonl'),
        '--input_key',
        'text',
        '--apply_chat_template',
        'False',
        '--max_samples',
        str(TRAINING_ROWS),
        '--prompt_max_len',
        '240',
        '--max_len',
        '2048',
        '--preprocess_num_workers',
        '8',
        '--rollout_num_engines',
        '1',
        '--rollout_tp_size',
        '1',
        '--rollout_mem_fraction_static',
        '0.20',
        '--rollout_batch_size',
        str(ROLLOUT_BATCH_SIZE),
        '--generate_max_len',
        '1808',
        '--n_samples_per_prompt',
        '1',
        '--temperature',
        '1.0',
        '--top_p',
        '1.0',
        '--teacher_tp_size',
        '1',
        '--teacher_pp_size',
        '1',
        '--teacher_ep_size',
        '1',
        '--teacher_dp_size',
        '1',
        '--teacher_mem_fraction_static',
        '0.28',
        '--teacher_context_length',
        '4096',
        '--teacher_forward_n_batches',
        '8',
        '--kd_algorithm',
        'mp_opd',
        '--kd_ratio',
        '1.0',
        '--mp_opd_mode',
        'atomic',
        '--mp_opd_max_span_length',
        '4',
        '--mp_opd_fixed_span_length',
        '2',
        '--mp_opd_random_seed',
        '43',
        '--mp_opd_partition_temperature',
        '1.0',
        '--save_steps',
        str(OPTIMIZER_UPDATES),
        '--logging_steps',
        '1',
        '--use_tensorboard',
        'True',
        '--tensorboard_log_dir',
        str(OUTPUT_ROOT / 'tensorboard'),
        '--tensorboard_flush_secs',
        '10',
        '--use_wandb',
        'True',
        '--wandb_org',
        WANDB_ENTITY,
        '--wandb_project',
        WANDB_PROJECT,
        '--wandb_group',
        'mp-opd-b200-llama1b-qwen4b',
        '--wandb_run_name',
        WANDB_RUN_NAME,
        '--wandb_run_id',
        WANDB_RUN_ID,
        '--wandb_job_type',
        IMPLEMENTATION_VALIDATION_JOB_TYPE,
        '--wandb_tags',
        WANDB_TAGS,
        '--wandb_mode',
        'online',
        '--wandb_dir',
        str(OUTPUT_ROOT / 'wandb'),
    ]
    command[command.index("--train_dataset_path") + 1] = "/model-cache/data/opd-prompts-6400.jsonl"
    command[command.index("--save_steps") + 1] = "10"
    command.extend(["--exact_token_trajectory", "True", "--diagnostic_max_updates", "30",
                    "--diagnostic_collapse_gate", "True"])
    environment = runtime_environment()
    environment.update(WANDB_ENTITY=ENTITY, WANDB_PROJECT=PROJECT, WANDB_RUN_ID=RUN_ID, WANDB_RESUME="never")
    gate = subprocess.run([PYTHON, "/opt/simct/experiments/environments/b200_gate.py"],
        env=environment, text=True, capture_output=True, timeout=120)
    (OUTPUT_ROOT / "runtime-gate.log").write_text(gate.stdout + gate.stderr)
    if gate.returncode:
        outputs.commit()
        raise RuntimeError("B200 runtime gate failed; training was not started")
    runtime = json.loads(next(line.split("=", 1)[1] for line in gate.stdout.splitlines()
                              if line.startswith("SIMCT_B200_ENV_JSON=")))
    if runtime["gpu_count"] != 1 or runtime["torch"] != "2.11.0+cu130":
        raise RuntimeError("runtime differs from the pinned single-B200 contract")
    print("MP_OPD_B200_RUNTIME=" + json.dumps(runtime), flush=True)
    source_hashes = {str(p.relative_to("/opt/overlay")): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path("/opt/overlay/kdflow").rglob("*.py")}
    contract = dict(run_id=RUN_ID, source_commit=source_commit, source_hashes=source_hashes,
        image=IMAGE, runtime=runtime, data_sha256=DATA_SHA256, command=command, preflight=preflight_result,
        stop_rules={"max_updates": 30, "process_timeout_seconds": 3600, "collapse_consecutive_batches": 2,
                    "empty_fraction": 0.25, "mean_length_relative_to_first_batch": 0.1,
                    "logprob_abs_mean_max": 0.1, "logprob_abs_max_max": 0.5},
        evidence_label="implementation_validation", terminal_eos_objective="masked, legacy denominator sentinel")
    (OUTPUT_ROOT / "invocation.json").write_text(json.dumps(contract, indent=2) + "\n")
    outputs.commit()
    started = time.monotonic()
    result = {"run_id": RUN_ID, "status": "failed"}
    try:
        with (OUTPUT_ROOT / "train.log").open("x") as stream:
            process = subprocess.Popen(command, cwd="/opt/overlay", env=environment, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            offset = 0
            while True:
                try:
                    code = process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    code = None
                with (OUTPUT_ROOT / "train.log").open() as reader:
                    reader.seek(offset)
                    for line in reader:
                        if "train_progress [" in line:
                            print(line.rstrip(), flush=True)
                    offset = reader.tell()
                if code is not None:
                    break
                if time.monotonic() - started >= 3600:
                    import signal
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                    raise RuntimeError("bounded training wall-time exceeded")
        result["exit_code"] = code
        summary = save_path / "run-summary.json"
        if summary.is_file():
            result["training"] = json.loads(summary.read_text())
        result["status"] = "completed_pending_audit" if code == 0 else "failed"
        print("MP_OPD_PATH_RESULT=" + json.dumps(result), flush=True)
        return result
    finally:
        result["wall_time_seconds"] = time.monotonic() - started
        (OUTPUT_ROOT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        outputs.commit()


# Register paid GPU resources only for an explicit training invocation. CPU
# preparation and regressions remain usable on workspaces without B200 access.
train = None
if os.environ.get("MP_OPD_ENABLE_B200") == "1":
    train = app.function(image=image, gpu="B200", cpu=16, memory=98304, timeout=4200,
        retries=0, single_use_containers=True,
        volumes={"/model-cache": cache, "/runs": outputs},
        secrets=[modal.Secret.from_name("wandb-secret")])(train_impl)


@app.local_entrypoint()
def main():
    if train is None:
        raise RuntimeError("set MP_OPD_ENABLE_B200=1 after B200 billing eligibility is confirmed")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("commit the source overlay before launching training")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    ready = preflight.remote()
    print(train.remote(commit, ready))
