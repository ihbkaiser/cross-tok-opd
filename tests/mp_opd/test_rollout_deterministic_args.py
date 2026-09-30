"""Rollout deterministic-inference flags: opt-in, default off, no contract drift."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _fixture_env(tmp_path: Path) -> dict:
    for name in ("student", "teacher"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text("{}")
    data = tmp_path / "prompts.parquet"
    data.write_bytes(b"fixture")
    energy = tmp_path / "energy.pt"
    energy.write_bytes(b"energy")
    meta = tmp_path / "meta.parquet"
    meta.write_bytes(b"meta")
    return {
        "MP_PREFLIGHT_ONLY": "1",
        "MP_STUDENT_PATH": str(tmp_path / "student"),
        "MP_TEACHER_PATH": str(tmp_path / "teacher"),
        "MP_DATASET_PATH": str(data),
        "MP_ENERGY_CHECKPOINT": str(energy),
        "MP_META_PATH": str(meta),
        "MP_ALTERNATING": "1",
        "MP_ATTN_IMPLEMENTATION": "eager",
        "MP_OFFLOAD_ADAM_MOMENTS": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }


def _launch(tmp_path: Path, name: str, extra: dict):
    env = dict(os.environ, **_fixture_env(tmp_path), **extra)
    out = tmp_path / name
    result = subprocess.run(
        [sys.executable, str(ROOT / "experiments/runai/run_single_gpu.py"), "soft", "2", str(out)],
        env=env, capture_output=True, text=True,
    )
    return result, out


def test_defaults_keep_production_rollout_contract(tmp_path):
    result, out = _launch(tmp_path, "default", {})
    assert result.returncode == 0, result.stderr
    options = json.loads((out / "launch-config.json").read_text())["options"]
    assert options["rollout_deterministic_inference"] is False
    assert options["rollout_random_seed"] == -1
    assert options["rollout_attention_backend"] == ""
    assert "EFFECTIVE_MP_ROLLOUT_DETERMINISTIC=false" in result.stdout


def test_opt_in_enables_deterministic_inference(tmp_path):
    result, out = _launch(tmp_path, "deterministic", {"MP_ROLLOUT_DETERMINISTIC": "1", "MP_ROLLOUT_SEED": "42"})
    assert result.returncode == 0, result.stderr
    options = json.loads((out / "launch-config.json").read_text())["options"]
    assert options["rollout_deterministic_inference"] is True
    assert options["rollout_random_seed"] == 42
    assert "EFFECTIVE_MP_ROLLOUT_DETERMINISTIC=true" in result.stdout


def test_invalid_flag_fails_closed(tmp_path):
    result, _ = _launch(tmp_path, "invalid", {"MP_ROLLOUT_DETERMINISTIC": "yes"})
    assert result.returncode != 0
    assert "MP_ROLLOUT_DETERMINISTIC" in (result.stderr + result.stdout)


def test_extra_server_args_helper_is_the_only_source_of_serving_flags():
    from kdflow.rollout_serving import build_extra_server_args

    class _Rollout:
        def __init__(self, deterministic, seed, disable_graph=True, backend="", radix=False):
            self.rollout_disable_piecewise_cuda_graph = disable_graph
            self.rollout_deterministic_inference = deterministic
            self.rollout_random_seed = seed
            self.rollout_attention_backend = backend
            self.rollout_disable_radix_cache = radix

    class _Args:
        def __init__(self, rollout):
            self.rollout = rollout

    assert build_extra_server_args(_Args(_Rollout(False, -1))) == {"disable_piecewise_cuda_graph": True}
    assert build_extra_server_args(_Args(_Rollout(True, 42))) == {
        "disable_piecewise_cuda_graph": True,
        "disable_radix_cache": True,
        "enable_deterministic_inference": True,
        "random_seed": 42,
        "attention_backend": "flashinfer",
    }
    triton = build_extra_server_args(_Args(_Rollout(True, 42, backend="triton")))
    assert triton["attention_backend"] == "triton"
    assert triton["disable_radix_cache"] is True
    assert triton["enable_deterministic_inference"] is True
    assert build_extra_server_args(_Args(_Rollout(False, -1, radix=True))) == {
        "disable_piecewise_cuda_graph": True,
        "disable_radix_cache": True,
    }
    explicit = build_extra_server_args(_Args(_Rollout(True, 42, backend="triton")))
    assert explicit["attention_backend"] == "triton"
    backend_only = build_extra_server_args(_Args(_Rollout(False, -1, backend="triton")))
    assert backend_only == {"disable_piecewise_cuda_graph": True, "attention_backend": "triton"}
    opt_out = build_extra_server_args(_Args(_Rollout(True, -1)))
    assert "random_seed" not in opt_out
    assert opt_out["enable_deterministic_inference"] is True
    assert build_extra_server_args(_Args(_Rollout(False, -1, disable_graph=False))) == {}


def test_cli_uses_the_helper_for_the_rollout_group():
    source = (ROOT / "kdflow/cli/train_kd_on_policy.py").read_text()
    assert "from kdflow.rollout_serving import" in source
    assert "serving_args = build_extra_server_args(args)" in source
    assert "extra_server_args=serving_args or None" in source
    assert "assert_serving_contract(" in source