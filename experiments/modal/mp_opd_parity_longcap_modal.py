"""Modal launcher for the long-cap log-probability parity probe.

Reproduces the `BUG_parity_logprob_4096cap.md` regime on a rented B200 with a
public pinned checkpoint, without touching company data:

  stage=prepare   fetch google/gemma-2-2b-it at the pinned revision into a volume
  stage=probe     run the engine-vs-eager probe for one or more engine configs

Every probe writes its own `result.json` and refuses to overwrite an existing
one, so a repeated invocation cannot silently replace evidence. Full per-token
arrays are stored, so re-analysis never needs a second GPU run.

  MODAL_PROFILE=kieusontung8 uvx --from modal modal run \
    experiments/modal/mp_opd_parity_longcap_modal.py --stage prepare
  MODAL_PROFILE=kieusontung8 uvx --from modal modal run \
    experiments/modal/mp_opd_parity_longcap_modal.py --stage probe \
    --run-id longcap-prod-20261003 --configs prod --max-new-tokens 4096
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import time

import modal

APP_NAME = "simct-parity-longcap-20261003"
IMAGE = ("docker.io/codemaivanngu/simct-b200"
         "@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f")
ASSET_VOLUME_NAME = "simct-parity-longcap-assets-v1"
OUTPUT_VOLUME_NAME = "simct-parity-longcap-results-v1"
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
        str(LOCAL_ROOT / "experiments/modal/mp_opd_parity_longcap_worker.py"),
        "/opt/overlay/experiments/modal/mp_opd_parity_longcap_worker.py",
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
        RAY_USAGE_STATS_ENABLED="0",
        NCCL_CUMEM_HOST_ENABLE="0",
        OMP_NUM_THREADS="4",
    )
    nvidia = Path("/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia")
    library_dirs = ["/usr/local/cuda/lib64", "/usr/local/nvidia/lib64"]
    library_dirs.extend(str(path) for path in sorted(nvidia.glob("*/lib")))
    if environment.get("LD_LIBRARY_PATH"):
        library_dirs.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(library_dirs)
    return environment


@app.function(image=image, cpu=2, memory=8192, timeout=1800, retries=0,
              volumes={"/assets": assets})
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
    allow_patterns=['*.json', '*.safetensors', '*.model', 'tokenizer.json',
                    'tokenizer.model', 'merges.txt', 'vocab.json'],
    max_workers=4,
)
config = AutoConfig.from_pretrained(root, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
assert config.model_type == 'gemma2', config.model_type
assert config.sliding_window, 'expected a sliding window config'
ready = {{
    'model_id': {STUDENT_ID!r}, 'revision': {STUDENT_REVISION!r},
    'model_type': config.model_type, 'vocab_size': len(tokenizer),
    'sliding_window': config.sliding_window,
    'attn_logit_softcapping': getattr(config, 'attn_logit_softcapping', None),
    'layer_types': list(getattr(config, 'layer_types', []) or []),
}}
Path('/assets/ready.json').write_text(json.dumps(ready, sort_keys=True) + '\\n')
print('LONGCAP_ASSETS_JSON=' + json.dumps(ready, sort_keys=True), flush=True)
"""
    subprocess.run([PYTHON, "-c", code], env=runtime_environment(online=True), check=True)
    assets.commit()
    return json.loads(Path("/assets/ready.json").read_text())


@app.function(
    image=image,
    cpu=8,
    memory=98304,
    timeout=7200,
    retries=0,
    max_containers=1,
    volumes={"/assets": assets, "/runs": outputs},
)
def parity_probe(run_id: str, configs: str, max_new_tokens: int, prompt_limit: int,
                 force_cap: int, prompt_style: str, prompt_pad: int) -> dict:
    ready = json.loads(Path("/assets/ready.json").read_text())
    assert ready["revision"] == STUDENT_REVISION, ready
    output_path = Path("/runs") / run_id / "result.json"
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing result: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        PYTHON,
        "/opt/overlay/experiments/modal/mp_opd_parity_longcap_worker.py",
        "--model-path", "/assets/student",
        "--output", str(output_path),
        "--configs", configs,
        "--max-new-tokens", str(max_new_tokens),
        "--prompt-limit", str(prompt_limit),
        "--force-cap", str(force_cap),
        "--prompt-style", prompt_style,
        "--prompt-pad", str(prompt_pad),
    ]
    (output_path.parent / "invocation.json").write_text(json.dumps({
        "run_id": run_id,
        "image": IMAGE,
        "student": ready,
        "configs": [c.strip() for c in configs.split(",") if c.strip()],
        "max_new_tokens": max_new_tokens,
        "prompt_limit": prompt_limit,
        "force_cap": force_cap,
        "prompt_style": prompt_style,
        "prompt_pad": prompt_pad,
        "command": command,
    }, indent=2) + "\n")
    outputs.commit()

    started = time.monotonic()
    process = None
    try:
        with (output_path.parent / "worker.log").open("x") as handle:
            process = subprocess.Popen(command, env=runtime_environment(online=False),
                                       stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            while process.poll() is None:
                outputs.commit()
                if time.monotonic() - started > 6600:
                    raise TimeoutError("long-cap probe time cap")
                time.sleep(20)
        text = (output_path.parent / "worker.log").read_text()
        if process.returncode:
            print("LONGCAP_LOG_TAIL=" + json.dumps(text.strip().splitlines()[-25:]), flush=True)
            raise RuntimeError("worker failed rc=" + str(process.returncode))
        result = json.loads(output_path.read_text())
        return {
            "status": result.get("status"),
            "environment": result.get("environment"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "configs": {
                name: {
                    "status": entry.get("status"),
                    "error": entry.get("error"),
                    "server_args": entry.get("server_args"),
                    "guard_metric": entry.get("detail", {}).get(
                        "guard_metric_decode_vs_hf_eager_temperature"),
                    "prefill_raw_vs_hf_eager_raw": entry.get("comparisons", {}).get(
                        "prefill_raw_vs_hf_eager_raw"),
                    "per_band": entry.get("detail", {}).get("per_band"),
                    "error_by_hf_entropy": entry.get("detail", {}).get("error_by_hf_entropy"),
                }
                for name, entry in (result.get("configs") or {}).items()
            },
        }
    finally:
        if process is not None and process.poll() is None:
            import signal

            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
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
def main(stage: str, run_id: str = "", configs: str = "prod", gpu: str = "B200",
         max_new_tokens: int = 4096, prompt_limit: int = 0, force_cap: int = 0,
         prompt_style: str = "coherent", prompt_pad: int = 0):
    if stage == "prepare":
        result = prepare_student.with_options(secrets=[local_hf_secret()]).remote()
    elif stage == "probe":
        if not run_id:
            raise ValueError("--run-id is required for probe")
        result = parity_probe.with_options(gpu=gpu).remote(
            run_id, configs, max_new_tokens, prompt_limit, force_cap, prompt_style,
            prompt_pad)
    else:
        raise ValueError("stage must be prepare or probe")
    print("LONGCAP_MODAL_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
