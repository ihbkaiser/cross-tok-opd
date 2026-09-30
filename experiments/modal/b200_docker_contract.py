"""Pure validation helpers for the B200 Docker build context."""

from __future__ import annotations

from pathlib import Path
import re


LOCAL_ROOT = Path(__file__).resolve().parents[2]

_IMAGE_REF = re.compile(
    r"^(?:docker\.io/)?[a-z0-9]+(?:[._-][a-z0-9]+)*/"
    r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})$"
)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_FORBIDDEN_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "models",
    "checkpoints",
    "outputs",
    "runs",
    "wandb",
    "data",
    "datasets",
    "artifacts",
}
_FORBIDDEN_SUFFIXES = {
    ".safetensors",
    ".gguf",
    ".onnx",
    ".ckpt",
    ".pth",
    ".pt",
    ".bin",
    ".pem",
    ".key",
}


def normalize_image_ref(value: str) -> str:
    if not _IMAGE_REF.fullmatch(value):
        raise ValueError(
            "image must be a tagged Docker Hub reference such as "
            "docker.io/namespace/simct-b200:cu130-<commit>"
        )
    return value if value.startswith("docker.io/") else f"docker.io/{value}"


def validate_secret_name(value: str) -> str:
    if not _SAFE_NAME.fullmatch(value):
        raise ValueError("registry secret name contains unsupported characters")
    return value


def ignore_local_path(path: Path) -> bool:
    candidate = path if path.is_absolute() else LOCAL_ROOT / path
    try:
        relative = candidate.resolve().relative_to(LOCAL_ROOT.resolve())
    except ValueError:
        return True
    lowered_parts = {part.lower() for part in relative.parts}
    if lowered_parts & _FORBIDDEN_PARTS:
        return True
    lowered_name = relative.name.lower()
    return (
        relative.suffix.lower() in _FORBIDDEN_SUFFIXES
        or lowered_name == ".env"
        or lowered_name.startswith(".env.")
        or "secret" in lowered_name
        or "credential" in lowered_name
    )


def audit_local_context() -> dict[str, int]:
    scanned = 0
    forbidden = []
    for path in LOCAL_ROOT.rglob("*"):
        if not path.is_file() or ignore_local_path(path):
            continue
        scanned += 1
        if path.stat().st_size > 100 * 1024 * 1024:
            forbidden.append(str(path.relative_to(LOCAL_ROOT)))
    if forbidden:
        raise RuntimeError(f"unexpected large Docker inputs: {forbidden}")
    return {"scanned_files": scanned, "large_files": 0}
