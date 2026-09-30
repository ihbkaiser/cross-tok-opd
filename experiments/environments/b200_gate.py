"""Emit a machine-readable CUDA/Blackwell environment qualification record."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess


def _command_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def main() -> None:
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    capability = torch.cuda.get_device_capability(0)
    if capability != (10, 0):
        raise RuntimeError(f"expected B200 capability (10, 0), got {capability}")

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is required for the portable B200 development image")

    record = {
        "status": "pass",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(capability),
        "gpu_count": torch.cuda.device_count(),
        "nvcc": _command_output([nvcc, "--version"]).splitlines()[-1],
        "driver": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ]
        ).splitlines()[0],
        "arch_contract": {
            "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST"),
            "CUDAARCHS": os.environ.get("CUDAARCHS"),
            "CMAKE_CUDA_ARCHITECTURES": os.environ.get("CMAKE_CUDA_ARCHITECTURES"),
        },
    }
    print("SIMCT_B200_ENV_JSON=" + json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
