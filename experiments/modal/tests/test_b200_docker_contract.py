from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments/modal/b200_docker_contract.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("b200_docker_contract", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_docker_hub_reference_is_normalized_and_shell_safe() -> None:
    builder = load_builder()
    assert (
        builder.normalize_image_ref("sontungkieu/simct-b200:cu130-deadbeef")
        == "docker.io/sontungkieu/simct-b200:cu130-deadbeef"
    )
    assert (
        builder.normalize_image_ref("docker.io/sontungkieu/simct-b200:latest")
        == "docker.io/sontungkieu/simct-b200:latest"
    )
    for invalid in (
        "simct-b200:latest",
        "ghcr.io/sontungkieu/simct-b200:latest",
        "sontungkieu/simct-b200",
        "sontungkieu/simct-b200:tag;echo-bad",
    ):
        with pytest.raises(ValueError):
            builder.normalize_image_ref(invalid)


def test_modal_context_filter_excludes_weights_outputs_and_secrets() -> None:
    builder = load_builder()
    excluded = (
        ROOT / "models/student/model.safetensors",
        ROOT / "runs/r1/summary.json",
        ROOT / ".env.private",
        ROOT / "credentials.json",
        ROOT / "misc/checkpoint.pt",
    )
    assert all(builder.ignore_local_path(path) for path in excluded)
    assert not builder.ignore_local_path(ROOT / "kdflow/trainers/mp_opd.py")
    assert builder.audit_local_context()["large_files"] == 0


def test_b200_image_contains_two_locked_envs_but_no_weight_copy() -> None:
    dockerfile = (ROOT / "docker/Dockerfile.b200-cu130").read_text()
    assert "nvidia/cuda:13.0.2-devel-ubuntu22.04" in dockerfile
    assert "UV_PROJECT_ENVIRONMENT=/opt/venvs/simct-b200" in dockerfile
    assert "UV_PROJECT_ENVIRONMENT=/opt/venvs/simct-b200-sft" in dockerfile
    assert dockerfile.count("uv sync") == 2
    assert "--locked" in dockerfile
    assert "COPY models" not in dockerfile
    assert "COPY data" not in dockerfile
    assert "COPY outputs" not in dockerfile
    assert "MODEL_PATH=/models" in dockerfile
    assert "DATA_PATH=/data" in dockerfile
    assert "OUTPUT_PATH=/outputs" in dockerfile
