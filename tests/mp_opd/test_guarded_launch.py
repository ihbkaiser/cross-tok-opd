"""Offline argument/launcher checks, without pulling Transformers or Ray."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("guarded_distillation_args", ROOT/"kdflow/arguments/distillation_args.py")
arguments = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = arguments
spec.loader.exec_module(arguments)


@pytest.mark.parametrize("mode", ["align_new", "trust_p"])
def test_argument_modes_and_pure_opd_contract(mode):
    args = arguments.DistillationArguments(kd_algorithm="mp_opd", mp_opd_mode=mode, kd_ratio=1.)
    assert args.mp_opd_mode == mode
    assert args.mp_opd_align_new_min_descent_fraction == .1
    with pytest.raises(ValueError, match="kd_ratio=1"):
        arguments.DistillationArguments(kd_algorithm="mp_opd", mp_opd_mode=mode, kd_ratio=.5)
    with pytest.raises(ValueError, match="exact_head"):
        arguments.DistillationArguments(kd_algorithm="mp_opd", mp_opd_mode=mode, kd_ratio=1., mp_opd_grass_geometry="diag")
    with pytest.raises(ValueError, match="identity"):
        arguments.DistillationArguments(kd_algorithm="mp_opd", mp_opd_mode=mode, kd_ratio=1., mp_opd_credit_transform="forward")


def test_projected_trust_rejects_legacy_scope_and_align_rejects_bad_fraction():
    with pytest.raises(ValueError, match="full optimizer batch"):
        arguments.DistillationArguments(kd_algorithm="mp_opd", mp_opd_mode="trust_p", kd_ratio=1., mp_opd_trust_scope="microbatch")
    for rho in (-.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="descent fraction"):
            arguments.DistillationArguments(kd_algorithm="mp_opd", mp_opd_mode="align_new", kd_ratio=1., mp_opd_align_new_min_descent_fraction=rho)


@pytest.mark.parametrize("mode", ["align_new", "trust_p"])
def test_transformers_argument_parser_accepts_new_mode(mode):
    transformers = pytest.importorskip("transformers")
    parser = transformers.HfArgumentParser(arguments.DistillationArguments)
    args, = parser.parse_args_into_dataclasses([
        "--kd_algorithm", "mp_opd", "--kd_ratio", "1", "--mp_opd_mode", mode,
        "--mp_opd_align_new_min_descent_fraction", "0.25",
    ])
    assert args.mp_opd_mode == mode and args.mp_opd_align_new_min_descent_fraction == .25


@pytest.mark.parametrize("mode", ["align_new", "trust_p"])
def test_actual_python_launcher_preflight_records_new_mode_without_loading_models(tmp_path, mode):
    output = tmp_path/mode
    env = {k: v for k, v in os.environ.items() if not k.startswith("MP_")}
    for role in ("student", "teacher"):
        path = tmp_path/role
        path.mkdir()
        (path/"config.json").write_text("{}")
        env["MP_"+role.upper()+"_PATH"] = str(path)
    dataset = tmp_path/"prompts.parquet"
    dataset.write_bytes(b"offline fixture, not a real dataset")
    env["MP_DATASET_PATH"] = str(dataset)
    env.update(MP_PREFLIGHT_ONLY="1", MP_GRASS_CHUNK_SOURCE="run", MP_GRASS_CHUNK_RUN_LENGTH="2",
               MP_TRUST_SCOPE="batch", MP_ALIGN_NEW_MIN_DESCENT_FRACTION="0.25")
    result = subprocess.run([sys.executable, str(ROOT/"experiments/runai/run_single_gpu.py"), mode, "2", str(output)],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout+result.stderr
    manifest = json.loads((output/"launch-config.json").read_text())
    config = manifest["options"]
    assert config["mp_opd_mode"] == mode
    assert config["train_batch_size"] == 64 and config["micro_train_batch_size"] == 4
    assert config["mp_opd_align_new_min_descent_fraction"] == .25
    assert config["kd_ratio"] == 1.
    if mode == "align_new":
        assert config["mp_opd_grass_chunk_source"] == "run"
        assert config["mp_opd_grass_chunk_run_length"] == 2
    else:
        assert config["mp_opd_trust_scope"] == "batch"
