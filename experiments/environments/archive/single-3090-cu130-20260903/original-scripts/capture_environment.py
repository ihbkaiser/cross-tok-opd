"""Capture selected non-secret environment/provenance fields only."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import torch

root = Path("/workspace/xtoken-native")
repo = root / "NeMo-RL"
def command(*args):
    return subprocess.check_output(args, cwd=repo, text=True).strip()

result = {
    "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
    "repo": "https://github.com/NVIDIA-NeMo/RL.git",
    "head": command("git", "rev-parse", "HEAD"),
    "git_status": command("git", "status", "--short"),
    "submodules": command("git", "submodule", "status", "--recursive"),
    "uv_lock_sha256": hashlib.sha256((repo / "uv.lock").read_bytes()).hexdigest(),
    "uv": command("uv", "--version"),
    "gpu": command("nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"),
    "torch_cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(), "bf16_supported": torch.cuda.is_bf16_supported(),
    "cgroup_memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
    "cgroup_cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
    "packages": {name: importlib.metadata.version(name) for name in
                 ("torch", "torchvision", "transformers", "ray", "numpy", "pytest", "nemo-rl")},
    "workspace_is_persistent_volume": False,
    "default_venv_modified": False,
}
(root / "artifacts" / "environment.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
