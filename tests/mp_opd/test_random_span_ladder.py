"""The random-span ladder: the launched mode is the recipe's mode, and placement stays out of it.

This file used to pin `"fixed"` as the runner's argv mode, which is exactly what made the first
min2max5 stack train one fixed span per arm: run_single_gpu.py takes argv[1] as mp_opd_mode, so
mp_opd_mode was "fixed", mp_opd_fixed_span_length equalled the variant's max span, and
min_span_length (inert outside random mode) was ignored while campaign.json still recorded mode
"random". checked_config() cannot catch that drift because the campaign only stores the recipe,
so the launch is pinned to the recipe here.
"""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/runai"))
import queue_random_span_ladder as Q
import queue_full_alternating as F


def _case(tmp_path):
    """A campaign directory whose config passes the source gate, without git or the share."""
    case = tmp_path / "case"
    case.mkdir()
    Q.WRITE(case / "campaign.json", dict(source=str(ROOT), commit=Q.head_commit(), source_dirty="",
        runs=Q.configurations(), recipe=Q.RECIPE, steps=list(Q.STEPS), eval_seeds=[42, 43, 44],
        student="student", teacher="teacher", dataset="dataset"))
    return case


def _fake_run_ok(seen=None, make_target=False):
    real_run = subprocess.run

    class Result:
        returncode = 0

    def fake_run(argv, **kwargs):
        # Q.subprocess is the real subprocess module, so unrelated calls must keep working.
        if not (isinstance(argv, list) and any("run_single_gpu.py" in str(x) for x in argv)):
            return real_run(argv, **kwargs)
        if seen is not None:
            seen["argv"] = argv
            seen["env"] = kwargs["env"]
        if make_target:
            Path(argv[-1]).mkdir(parents=True, exist_ok=True)
        return Result()

    return fake_run


def test_random_ladder_is_four_bounds_by_three_seeds():
    configs = Q.configurations()
    assert len(configs) == Q.expected_trains() == 12
    assert {c["span"] for c in configs} == {2, 3, 4, 5}
    assert {c["train_seed"] for c in configs} == {42, 43, 44}
    assert len({c["id"] for c in configs}) == 12
    # Placement is not part of the campaign contract, so it must not appear here.
    assert all("slot" not in c for c in configs)


def test_recipe_matches_company_v2_random():
    assert Q.RECIPE["mode"] == "random"
    assert Q.RECIPE["min_span_length"] == 2
    assert Q.RECIPE["source_group"] == "company-v2-random"
    assert Q.RECIPE["learning_rate"] == 1e-6
    assert Q.RECIPE["micro_train_batch_size"] == 4
    assert Q.RECIPE["train_batch_size"] == 64
    assert Q.RECIPE["num_epochs"] == 2
    assert Q.RECIPE["optimizer_updates"] == 312
    assert Q.RECIPE["alternating"] is False
    for config in Q.configurations():
        assert config["span"] >= 2
        assert config["micro_B"] == 4


def test_placement_defaults_and_overrides_without_touching_the_contract(tmp_path, monkeypatch):
    monkeypatch.delenv("MP_LADDER_SLOTS", raising=False)
    assert Q.slots() == {"random2": 4, "random3": 5, "random4": 6, "random5": 7}
    case = _case(tmp_path)
    before = Q.checked_config(case)["runs"]
    monkeypatch.setenv("MP_LADDER_SLOTS", "2,3,4,5")
    assert Q.slots() == {"random2": 2, "random3": 3, "random4": 4, "random5": 5}
    assert Q.checked_config(case)["runs"] == before
    for bad in ("2", "2,3", "2,3,4", "2,2,4,5", "8,3,4,5", "x,3,4,5"):
        monkeypatch.setenv("MP_LADDER_SLOTS", bad)
        with pytest.raises(ValueError):
            Q.slots()


def test_eval_cells_cover_every_run_at_every_checkpoint():
    cells = Q.eval_cells()
    assert len(cells) == len(Q.configurations()) * len(Q.STEPS) == 96
    assert "RND-random5-s43-step312" in cells


def test_eval_label_names_the_span_range_that_is_trained():
    assert Q.eval_mode_label(5) == "random-min2max5"
    assert Q.eval_mode_label(2) == "random-min2max2"
    assert Q.eval_mode_label(3) != "fixed-span3"


def test_runner_mode_is_the_recipes_mode_and_reaches_the_argv(tmp_path, monkeypatch):
    assert Q.RUNNER_MODE == Q.RECIPE["mode"] == "random"
    case = _case(tmp_path)
    seen = {}
    monkeypatch.setattr(Q.subprocess, "run", _fake_run_ok(seen, make_target=True))
    Q.train_one(case, Q.configurations()[-1])
    # run_single_gpu.py reads argv[1] as mp_opd_mode, so this is the only place the launched mode
    # is decided; it has to be the mode the campaign records.
    assert seen["argv"][3] == Q.RUNNER_MODE == "random"
    assert Q.checked_config(case)["recipe"]["mode"] == seen["argv"][3]


def test_ladder_ids_never_collide_with_the_alternating_campaign():
    assert not {c["id"] for c in Q.configurations()} & {c["id"] for c in F.configurations()}
    import queue_fixed_span_ladder as FIX
    assert not {c["id"] for c in Q.configurations()} & {c["id"] for c in FIX.configurations()}


def test_train_env_pins_the_slot_derives_ports_and_keeps_infrastructure(tmp_path, monkeypatch):
    case = _case(tmp_path)
    config = Q.configurations()[0]
    monkeypatch.setenv("MP_LADDER_SLOTS", "1,2,3,4")
    monkeypatch.setenv("MP_ALTERNATING", "1")
    monkeypatch.setenv("MP_RUNTIME_DIR", "/tmp/runtime")
    env, slot = Q.train_env(case, config)
    assert slot == config["span"] - 1 and slot == 1
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["KDFLOW_ROLLOUT_PORT_BASE"] == "16000"
    assert env["KDFLOW_ROUTER_PROMETHEUS_PORT"] == "21000"
    assert env["MP_RAY_TMP"].startswith("/tmp/ar")
    # The draw is randint(min, min(max, remaining)) inside random mode, so the max bound must move
    # with the variant's span and the min bound must stay at the recipe's value.
    assert env["MP_FIXED_SPAN_LENGTH"] == str(config["span"]) == env["MP_MAX_SPAN_LENGTH"]
    assert env["MP_MIN_SPAN_LENGTH"] == str(Q.RECIPE["min_span_length"]) == "2"
    assert env["MP_SEED"] == str(config["train_seed"]) == "42"
    assert env["MP_RESUME"] == "0"
    assert "MP_ALTERNATING" not in env
    assert env["MP_RUNTIME_DIR"] == "/tmp/runtime"


def test_resume_is_requested_only_when_a_checkpoint_exists(tmp_path, monkeypatch):
    case = _case(tmp_path)
    config = Q.configurations()[0]
    assert Q.resumable(case, config) is False
    assert Q.train_env(case, config)[0]["MP_RESUME"] == "0"
    (Q.run_dir(case, config) / "checkpoints").mkdir(parents=True)
    (Q.run_dir(case, config) / "checkpoints" / "latest.json").write_text("{}")
    assert Q.resumable(case, config) is True
    assert Q.train_env(case, config)[0]["MP_RESUME"] == "1"


def test_train_one_passes_an_explicit_output_directory(tmp_path, monkeypatch):
    case = _case(tmp_path)
    config = Q.configurations()[0]
    seen = {}
    monkeypatch.setattr(Q.subprocess, "run", _fake_run_ok(seen, make_target=True))
    target = Q.train_one(case, config)
    assert target == Q.run_dir(case, config)
    assert seen["argv"] == ["bash", str(ROOT / "experiments/runai/python-b200-host.sh"),
                            str(ROOT / "experiments/runai/run_single_gpu.py"), "random", "312",
                            str(target)]
    receipt = json.loads(Q.receipt_path(case, config).read_text())
    assert receipt["returncode"] == 0 and receipt["run_dir"] == str(target)
    assert receipt["resume"] == "0" and receipt["slot"] == 4
    assert Q.finished(case, config) is True
    assert Q.run_dir_of(case, config) == target


def test_train_one_moves_a_husk_aside_instead_of_colliding(tmp_path, monkeypatch):
    case = _case(tmp_path)
    config = Q.configurations()[0]
    target = Q.run_dir(case, config)
    target.mkdir(parents=True)
    (target / "checkpoint").mkdir()
    monkeypatch.setattr(Q.subprocess, "run", _fake_run_ok(make_target=True))
    Q.train_one(case, config)
    husks = list(target.parent.glob(config["id"] + ".abandoned-*"))
    assert len(husks) == 1 and husks[0].is_dir()
    assert target.is_dir() and not (target / "checkpoint").exists()
    assert json.loads(Q.receipt_path(case, config).read_text())["resume"] == "0"


def test_train_one_resumes_into_the_same_directory(tmp_path, monkeypatch):
    case = _case(tmp_path)
    config = Q.configurations()[0]
    (Q.run_dir(case, config) / "checkpoints").mkdir(parents=True)
    (Q.run_dir(case, config) / "checkpoints" / "latest.json").write_text("{}")
    seen = {}
    monkeypatch.setattr(Q.subprocess, "run", _fake_run_ok(seen, make_target=True))
    Q.train_one(case, config)
    assert seen["argv"][-1] == str(Q.run_dir(case, config))
    assert seen["env"]["MP_RESUME"] == "1"
    assert list(Q.run_dir(case, config).parent.glob(config["id"] + ".abandoned-*")) == []


def test_train_starts_every_bound_of_a_seed_together(tmp_path, monkeypatch):
    case = _case(tmp_path)
    started = []
    real_popen = subprocess.Popen

    class Child:
        def wait(self):
            return 0

    def fake_popen(argv, **kwargs):
        if not (isinstance(argv, list) and any(str(x).endswith("train-one") for x in argv)):
            return real_popen(argv, **kwargs)
        started.append(argv[-1])
        return Child()

    monkeypatch.setattr(Q.subprocess, "Popen", fake_popen)
    Q.train(case)
    assert started == [c["id"] for c in Q.configurations()]
    for seed in Q.TRAIN_SEEDS:
        group = [x for x in started if x.endswith("s" + str(seed))]
        assert group == ["RND-random2-s" + str(seed), "RND-random3-s" + str(seed),
                         "RND-random4-s" + str(seed), "RND-random5-s" + str(seed)], group


def test_train_skips_a_run_that_already_finished(tmp_path, monkeypatch):
    case = _case(tmp_path)
    first = Q.configurations()[0]
    Q.WRITE(Q.receipt_path(case, first), dict(returncode=0, run_dir=str(Q.run_dir(case, first))))
    started = []
    real_popen = subprocess.Popen

    class Child:
        def wait(self):
            return 0

    def fake_popen(argv, **kwargs):
        if not (isinstance(argv, list) and any(str(x).endswith("train-one") for x in argv)):
            return real_popen(argv, **kwargs)
        started.append(argv[-1])
        return Child()

    monkeypatch.setattr(Q.subprocess, "Popen", fake_popen)
    Q.train(case)
    assert first["id"] not in started
    assert len(started) == 11


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
