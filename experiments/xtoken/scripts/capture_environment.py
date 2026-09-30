"""Selected runtime metadata, with no credential or environment-variable dump."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch


def read_optional(path):
    file = Path(path)
    return file.read_text().strip() if file.is_file() else None


def command(*args):
    return subprocess.check_output(args, cwd=repo, text=True).strip()


repo = Path(os.environ["XTOKEN_REPO"])
out = Path(os.environ["XTOKEN_RUN_DIR"])
packages = {}
for name in ("torch", "torchvision", "transformers", "ray", "numpy", "pytest", "nemo-rl"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
result = {
    "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
    "head": command("git", "rev-parse", "HEAD"),
    "submodules": command("git", "submodule", "status", "--recursive"),
    "uv_lock_sha256": hashlib.sha256((repo / "uv.lock").read_bytes()).hexdigest(),
    "uv": command(os.environ["XTOKEN_UV_BIN"], "--version"),
    "gpu": command("nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"),
    "torch_cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    "cgroup_memory_max": read_optional("/sys/fs/cgroup/memory.max"),
    "cgroup_cpu_max": read_optional("/sys/fs/cgroup/cpu.max"),
    "packages": packages,
    "workspace_persistence": "not_verified",
    "scope": "runtime inventory; not a trainer or model-quality test",
}
(out / "environment.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
