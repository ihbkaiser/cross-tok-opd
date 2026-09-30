"""Cost-bounded Modal launcher for MP-OPD numerical parity diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess

import modal


APP_NAME = "mp-opd-gemma-parity-20260909-r1"
IMAGE = "docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f"
ASSET_VOLUME_NAME = "mp-opd-gemma-parity-assets-v1"
OUTPUT_VOLUME_NAME = "mp-opd-gemma-parity-results-v1"
STUDENT_ID = "google/gemma-2-2b-it"
STUDENT_REVISION = "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
PYTHON = "/opt/venvs/simct-b200/bin/python"
LOCAL_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/opt/overlay")


app = modal.App(APP_NAME)
image = (
    modal.Image.from_registry(IMAGE)
    .entrypoint([])
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_file(
        str(LOCAL_ROOT / "experiments/modal/mp_opd_parity_worker.py"),
        "/opt/overlay/experiments/modal/mp_opd_parity_worker.py",
    )
)
assets = modal.Volume.from_name(ASSET_VOLUME_NAME, create_if_missing=True)
outputs = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


def runtime_environment(*, online: bool) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        PATH=f"/opt/venvs/simct-b200/bin:{environment.get('PATH', '')}",
        HF_HOME="/assets/hf",
        HF_HUB_OFFLINE="0" if online else "1",
        TRANSFORMERS_OFFLINE="0" if online else "1",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
        CUDA_HOME="/usr/local/cuda",
    )
    nvidia = Path("/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia")
    library_dirs = ["/usr/local/cuda/lib64", "/usr/local/nvidia/lib64"]
    library_dirs.extend(str(path) for path in sorted(nvidia.glob("*/lib")))
    if environment.get("LD_LIBRARY_PATH"):
        library_dirs.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(library_dirs)
    return environment


@app.function(image=image, cpu=2, memory=8192, timeout=900, retries=0, volumes={"/assets": assets})
def prepare_student() -> dict[str, str]:
    code = f"""
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer
root = Path('/assets/student')
snapshot_download(
    {STUDENT_ID!r}, revision={STUDENT_REVISION!r}, local_dir=str(root),
    token=__import__('os').environ['HF_TOKEN'],
    allow_patterns=['*.json', '*.safetensors', '*.model', 'merges.txt', 'vocab.json'],
    max_workers=4,
)
config = AutoConfig.from_pretrained(root, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
assert config.model_type == 'gemma2'
ready = {{
    'model_id': {STUDENT_ID!r}, 'revision': {STUDENT_REVISION!r},
    'model_type': config.model_type, 'vocab_size': len(tokenizer),
}}
Path('/assets/ready.json').write_text(json.dumps(ready, sort_keys=True) + '\\n')
print('MP_PARITY_ASSETS_JSON=' + json.dumps(ready, sort_keys=True), flush=True)
"""
    subprocess.run([PYTHON, "-c", code], env=runtime_environment(online=True), check=True)
    assets.commit()
    return json.loads(Path("/assets/ready.json").read_text())


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=720,
    retries=0,
    max_containers=1,
    volumes={"/assets": assets, "/runs": outputs},
)
def parity_probe(run_id: str, backends: str, rl_on_policy_target: str) -> dict:
    ready = json.loads(Path("/assets/ready.json").read_text())
    assert ready["revision"] == STUDENT_REVISION
    output_path = Path("/runs") / run_id / "result.json"
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing result: {output_path}")
    command = [
        PYTHON,
        "/opt/overlay/experiments/modal/mp_opd_parity_worker.py",
        "--model-path",
        "/assets/student",
        "--output",
        str(output_path),
        "--backends",
        backends,
        "--rl-on-policy-target",
        rl_on_policy_target,
    ]
    invocation = {
        "run_id": run_id,
        "image": IMAGE,
        "student": ready,
        "backends": backends.split(","),
        "rl_on_policy_target": rl_on_policy_target,
        "command": command,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (output_path.parent / "invocation.json").write_text(json.dumps(invocation, indent=2) + "\n")
    try:
        subprocess.run(command, env=runtime_environment(online=False), check=True, timeout=660)
        result = json.loads(output_path.read_text())
        return {
            "status": result["status"],
            "environment": result["environment"],
            "comparisons": result["comparisons"],
            "elapsed_seconds": result["elapsed_seconds"],
        }
    finally:
        outputs.commit()


def local_hf_secret() -> modal.Secret:
    values: dict[str, str] = {}
    secret_file = Path("/home/tung/Collaborative-MORL/.secrets/talapas_secrets.env")
    for line in secret_file.read_text().splitlines():
        line = line.removeprefix("export ")
        if line.startswith("HF_TOKEN="):
            values["HF_TOKEN"] = shlex.split(line.split("=", 1)[1])[0]
    if "HF_TOKEN" not in values:
        raise RuntimeError("HF_TOKEN is unavailable")
    return modal.Secret.from_dict(values)


@app.local_entrypoint()
def main(
    stage: str,
    run_id: str = "",
    gpu: str = "A10",
    backends: str = "default",
    rl_on_policy_target: str = "none",
):
    if stage == "prepare":
        result = prepare_student.with_options(secrets=[local_hf_secret()]).remote()
    elif stage == "probe":
        if not run_id:
            raise ValueError("--run-id is required for probe")
        result = parity_probe.with_options(gpu=gpu).remote(
            run_id, backends, rl_on_policy_target
        )
    else:
        raise ValueError("stage must be prepare or probe")
    print("MP_PARITY_MODAL_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
