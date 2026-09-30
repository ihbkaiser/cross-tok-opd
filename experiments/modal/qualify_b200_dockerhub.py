"""Qualify the published portable SimCT image on one real Modal B200.

Set ``SIMCT_B200_IMAGE`` and ``SIMCT_DOCKER_REGISTRY_SECRET`` in the local
launcher environment. Registry credentials are resolved by Modal and are not
passed to the runtime container.
"""

from __future__ import annotations

import json
import os
import subprocess

import modal

from experiments.modal.b200_docker_contract import (
    normalize_image_ref,
    validate_secret_name,
)


APP_NAME = "vdt-simct-b200-docker-qualification"
IMAGE_REF = normalize_image_ref(os.environ.get("SIMCT_B200_IMAGE", ""))
REGISTRY_SECRET_NAME = validate_secret_name(
    os.environ.get("SIMCT_DOCKER_REGISTRY_SECRET", "")
)

registry_secret = modal.Secret.from_name(REGISTRY_SECRET_NAME)
runtime_image = modal.Image.from_registry(IMAGE_REF, secret=registry_secret).env(
    {
        # Modal imports this module again inside the runtime container. Carry
        # the already validated, non-secret identifiers into that import.
        "SIMCT_B200_IMAGE": IMAGE_REF,
        "SIMCT_DOCKER_REGISTRY_SECRET": REGISTRY_SECRET_NAME,
    }
)
app = modal.App(APP_NAME)


def run_checked(command: list[str], timeout: int = 180) -> str:
    process = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if process.returncode:
        raise RuntimeError(
            f"command failed ({process.returncode}): {command[0]} {command[1]}"
        )
    return process.stdout


def parse_gate(stdout: str) -> dict[str, object]:
    marker = "SIMCT_B200_ENV_JSON="
    payload = next(
        (
            line.removeprefix(marker)
            for line in stdout.splitlines()
            if line.startswith(marker)
        ),
        None,
    )
    if payload is None:
        raise RuntimeError("B200 gate did not emit its machine-readable marker")
    return json.loads(payload)


@app.function(
    image=runtime_image,
    gpu="B200",
    cpu=8,
    memory=32_768,
    ephemeral_disk=524_288,
    timeout=1_800,
    retries=0,
    single_use_containers=True,
)
def qualify() -> dict[str, object]:
    kd_python = "/opt/venvs/simct-b200/bin/python"
    sft_python = "/opt/venvs/simct-b200-sft/bin/python"
    gate = "/opt/simct/experiments/environments/b200_gate.py"

    gates = {
        "simct": parse_gate(run_checked([kd_python, gate])),
        "sft": parse_gate(run_checked([sft_python, gate])),
    }
    package_checks = {
        "simct": run_checked(["uv", "pip", "check", "--python", kd_python]).strip(),
        "sft": run_checked(["uv", "pip", "check", "--python", sft_python]).strip(),
    }
    imports = json.loads(
        run_checked(
            [
                kd_python,
                "-c",
                (
                    "import json, torch, kdflow, transformers; "
                    "from importlib.metadata import version; "
                    "x=torch.randn((1024,1024),device='cuda',dtype=torch.bfloat16); "
                    "y=x@x; torch.cuda.synchronize(); "
                    "assert bool(torch.isfinite(y).all()); "
                    "print(json.dumps({'torch':torch.__version__,"
                    "'transformers':transformers.__version__,"
                    "'sglang':version('sglang'),'flash_attn_4':version('flash-attn-4'),"
                    "'bf16_matmul':'finite'}))"
                ),
            ]
        )
    )
    sft_imports = json.loads(
        run_checked(
            [
                sft_python,
                "-c",
                (
                    "import json, torch, llamafactory; "
                    "from importlib.metadata import version; "
                    "print(json.dumps({'torch':torch.__version__,"
                    "'llamafactory':version('llamafactory')}))"
                ),
            ]
        )
    )
    result = {
        "status": "pass",
        "image": IMAGE_REF,
        "gates": gates,
        "package_checks": package_checks,
        "imports": imports,
        "sft_imports": sft_imports,
        "models_included": False,
        "external_mounts": ["/models", "/data", "/outputs"],
    }
    print("SIMCT_B200_DOCKER_QUALIFICATION_JSON=" + json.dumps(result, sort_keys=True))
    return result


@app.local_entrypoint()
def main() -> None:
    result = qualify.remote()
    print("SIMCT_B200_DOCKER_RESULT_JSON=" + json.dumps(result, sort_keys=True))
