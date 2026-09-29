"""The fixed-span ladder must stay a different campaign from the alternating one."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/runai"))
import queue_fixed_span_ladder as Q
import queue_full_alternating as F


def _case(tmp_path):
    """A campaign directory whose config passes the source gate, without git or the share."""
    case = tmp_path / "case"
    case.mkdir()
    Q.WRITE(case / "campaign.json", dict(source=str(ROOT), commit=Q.head_commit(), source_dirty="",
        runs=Q.configurations(), recipe=Q.RECIPE, steps=list(Q.STEPS), eval_seeds=[42, 43, 44],
        student="student", teacher="teacher", dataset="dataset"))
    return case


def test_ladder_is_two_spans_by_three_seeds_on_two_slots():
    configs = Q.configurations()
    assert len(configs) == Q.expected_trains() == 6
    assert {c["span"] for c in configs} == {3, 4}
    assert {c["train_seed"] for c in configs} == {42, 43, 44}
    assert len({c["id"] for c in configs}) == 6
    assert {c["slot"] for c in configs if c["span"] == 3} == {6}
    assert {c["slot"] for c in configs if c["span"] == 4} == {7}


def test_recipe_matches_company_v2_fixed():
    assert Q.RECIPE["mode"] == "fixed"
    assert Q.RECIPE["learning_rate"] == 1e-6
    assert Q.RECIPE["micro_train_batch_size"] == 4
    assert Q.RECIPE["train_batch_size"] == 64
    assert Q.RECIPE["num_epochs"] == 2
    assert Q.RECIPE["optimizer_updates"] == 312
    # Full alternating is refused for any mode other than soft (student_actor.py:257).
    assert Q.RECIPE["alternating"] is False
    for config in Q.configurations():
        assert config["student_lr"] == Q.RECIPE["learning_rate"]
        assert config["micro_B"] == 4
        assert config["span"] > 2


def test_eval_cells_cover_every_run_at_every_checkpoint():
    cells = Q.eval_cells()
    assert len(cells) == len(Q.configurations()) * len(Q.STEPS) == 48
    for config in Q.configurations():
        for step in Q.STEPS:
            assert config["id"] + "-step" + str(step) in cells


def test_ladder_ids_never_collide_with_the_alternating_campaign():
    assert not {c["id"] for c in Q.configurations()} & {c["id"] for c in F.configurations()}


def test_train_env_sets_both_span_bounds_and_strips_foreign_mp_flags(tmp_path, monkeypatch):
    case = _case(tmp_path)
    config = Q.configurations()[0]
    monkeypatch.setenv("MP_ALTERNATING", "1")
    monkeypatch.setenv("MP_ENERGY_EVERY", "4")
    monkeypatch.setenv("MP_MAX_SPAN_LENGTH", "2")
    env, root = Q.train_env(case, config)
    # A longer fixed span trips run_single_gpu.py:176 unless the max bound moves with it.
    assert env["MP_FIXED_SPAN_LENGTH"] == str(config["span"]) == env["MP_MAX_SPAN_LENGTH"]
    assert env["MP_SEED"] == str(config["train_seed"])
    assert env["MP_RUN_ROOT"] == str(root)
    assert env["MP_MICRO_TRAIN_BATCH_SIZE"] == "4"
    for flag in ("MP_ALTERNATING", "MP_ENERGY_EVERY", "MP_ENERGY_LR", "MP_META_PATH",
                 "MP_ENERGY_CHECKPOINT", "MP_MAX_LEN"):
        assert flag not in env, flag


def test_train_one_pins_the_slot_and_adopts_the_wrapper_run_directory(tmp_path, monkeypatch):
    case = _case(tmp_path)
    config = Q.configurations()[0]
    seen = {}

    class Result:
        returncode = 0

    real_run = subprocess.run

    def fake_run(argv, **kwargs):
        # Q.subprocess is the real subprocess module, so anything that is not our launcher must
        # keep working for the test runner itself.
        if not (isinstance(argv, list) and any("run_single_gpu.sh" in str(x) for x in argv)):
            return real_run(argv, **kwargs)
        seen["argv"] = argv
        seen["root"] = kwargs["env"]["MP_RUN_ROOT"]
        made = Path(kwargs["env"]["MP_RUN_ROOT"]) / "qwen-gemma-mp_opd-fixed-gpu6-limit312-1-2"
        made.mkdir(parents=True)
        return Result()

    monkeypatch.setattr(Q.subprocess, "run", fake_run)
    run_dir = Q.train_one(case, config)
    assert seen["argv"] == ["bash", str(ROOT / "experiments/runai/run_single_gpu.sh"),
                            str(config["slot"]), "fixed", "312"]
    assert run_dir.name.startswith("qwen-gemma-mp_opd-fixed-gpu6")
    receipt = json.loads((Path(seen["root"]) / "last-exit.json").read_text())
    assert receipt["returncode"] == 0 and receipt["slot"] == config["slot"]
    assert Q.finished(case, config) is True
    assert Q.run_dir_of(case, config) == run_dir


def test_train_starts_both_spans_of_a_seed_together(tmp_path, monkeypatch):
    case = _case(tmp_path)
    started = []

    class Child:
        def wait(self):
            return 0

    real_popen = subprocess.Popen

    def fake_popen(argv, **kwargs):
        if not (isinstance(argv, list) and any(str(x).endswith("train-one") for x in argv)):
            return real_popen(argv, **kwargs)
        started.append(argv[-1])
        return Child()

    monkeypatch.setattr(Q.subprocess, "Popen", fake_popen)
    Q.train(case)
    assert started == [c["id"] for c in Q.configurations()]
    for seed in Q.TRAIN_SEEDS:
        pair = [x for x in started if x.endswith("s" + str(seed))]
        assert pair == ["FIX-fixed3-s" + str(seed), "FIX-fixed4-s" + str(seed)], pair


def test_train_skips_a_run_that_already_finished(tmp_path, monkeypatch):
    case = _case(tmp_path)
    first = Q.configurations()[0]
    root = case / "train" / first["id"]
    root.mkdir(parents=True)
    Q.WRITE(root / "last-exit.json", dict(returncode=0, run_dir=str(root / "run")))
    started = []

    class Child:
        def wait(self):
            return 0

    real_popen = subprocess.Popen

    def fake_popen(argv, **kwargs):
        if not (isinstance(argv, list) and any(str(x).endswith("train-one") for x in argv)):
            return real_popen(argv, **kwargs)
        started.append(argv[-1])
        return Child()

    monkeypatch.setattr(Q.subprocess, "Popen", fake_popen)
    Q.train(case)
    assert first["id"] not in started
    assert len(started) == 5


def test_checked_config_refuses_a_drifting_recipe_or_commit(tmp_path):
    case = _case(tmp_path)
    assert Q.checked_config(case)["runs"] == Q.configurations()
    drifted = json.loads((case / "campaign.json").read_text())
    drifted["recipe"] = dict(drifted["recipe"], learning_rate=1e-3)
    (case / "campaign.json").write_text(json.dumps(drifted))
    with pytest.raises(ValueError, match="recipe"):
        Q.checked_config(case)
    drifted["recipe"] = Q.RECIPE
    drifted["commit"] = "0" * 40
    (case / "campaign.json").write_text(json.dumps(drifted))
    with pytest.raises(ValueError, match="commit"):
        Q.checked_config(case)


def test_train_refuses_an_uninitialised_case(tmp_path):
    with pytest.raises(FileNotFoundError):
        Q.train(tmp_path / "missing")
