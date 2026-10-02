#!/usr/bin/env python3
"""Random-span ladder: max span 2..5 with the recipe's min span 2, one variant per GPU slot.

Training launches the repo host wrapper with an explicit output directory, so a killed run
can be resumed into the same directory instead of silently creating a second one. Placement
(which GPU) is runtime state, not part of the campaign contract: override it with
MP_LADDER_SLOTS="2,3" without invalidating an existing case.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/runai"))
import queue_full_alternating as F  # noqa: E402
import borrowed_campaign as B  # noqa: E402

READ = F.read
WRITE = F.write
HOST_WRAPPER = ROOT / "experiments/runai/python-b200-host.sh"
RUNNER = ROOT / "experiments/runai/run_single_gpu.py"

VARIANTS = (("random2", 2), ("random3", 3), ("random4", 4), ("random5", 5))
TRAIN_SEEDS = (42, 43, 44)
# Default placement only. The wrapper derives its port bases from the slot, so the two
# concurrent jobs never collide; MP_LADDER_SLOTS overrides the placement per host.
DEFAULT_SLOTS = (4, 5, 6, 7)
STEPS = (40, 80, 120, 156, 200, 240, 280, 312)
UPDATES = 312
PARTITION_SEED = 43
RECIPE = dict(mode="random", learning_rate=1e-6, micro_train_batch_size=4, train_batch_size=64,
              num_epochs=2, optimizer_updates=UPDATES, alternating=False,
              partition_seed=PARTITION_SEED, temperature=0.6, top_p=0.95,
              min_span_length=2,
              generate_max_len=4096, max_len=4096, source_group="company-v2-random")
# The runner takes the algorithm mode as its first positional argument, so derive it from the
# recipe: the launched mode and campaign.json then agree by construction. Launching this case as
# "fixed" silently trains one fixed span (min_span_length is inert outside random mode) while the
# campaign still records mode "random".
RUNNER_MODE = RECIPE["mode"]
# Recipe flags a stale interactive shell must never inject into an immutable run.
# Infrastructure flags (MP_RUNTIME_DIR, MP_RAY_TMP, MP_SHARED_ROOT) stay inherited.
RECIPE_FLAGS = ("MP_ALTERNATING", "MP_ENERGY_CHECKPOINT", "MP_ENERGY_EVERY", "MP_ENERGY_LR",
                 "MP_META_PATH", "MP_META_MICRO_BATCH_SIZE", "MP_MAX_LEN", "MP_PAUSE_AFTER_UPDATES",
                 "MP_CHECKPOINT_STEPS", "MP_SEED", "MP_PARTITION_SEED", "MP_FIXED_SPAN_LENGTH",
                 "MP_MAX_SPAN_LENGTH", "MP_MIN_SPAN_LENGTH", "MP_GBV_BETA", "MP_GBV_GEOMETRY",
                 "MP_MICRO_TRAIN_BATCH_SIZE",
                 "MP_ALGORITHM",
                 "MP_ATTN_IMPLEMENTATION", "MP_PREFLIGHT_ONLY", "MP_RESUME", "MP_RUN_ROOT",
                 "MP_STUDENT_PATH", "MP_TEACHER_PATH", "MP_DATASET_PATH", "MP_SOURCE_COMMIT",
                 "MP_SOURCE_DIRTY")


def slots():
    """GPU slot per variant; placement only, so it stays out of the campaign contract."""
    raw = os.environ.get("MP_LADDER_SLOTS", "")
    if not raw:
        return dict(zip([name for name, _ in VARIANTS], DEFAULT_SLOTS))
    values = [int(x) for x in raw.split(",") if x.strip()]
    if len(values) != len(VARIANTS) or len(set(values)) != len(values):
        raise ValueError("MP_LADDER_SLOTS needs one distinct slot per variant")
    if any(not 0 <= v <= 7 for v in values):
        raise ValueError("MP_LADDER_SLOTS entries must be numeric slots 0..7")
    return dict(zip([name for name, _ in VARIANTS], values))


def configurations():
    return [dict(id="RND-" + name + "-s" + str(seed), variant=name, span=span, mode="random",
                 train_seed=seed, student_updates=UPDATES,
                 micro_B=RECIPE["micro_train_batch_size"], train_B=RECIPE["train_batch_size"],
                 student_lr=RECIPE["learning_rate"])
            for seed in TRAIN_SEEDS for name, span in VARIANTS]


def expected_trains():
    return len(configurations())


def eval_mode_label(span):
    """Plan label naming the span range this arm actually trains, e.g. random-min2max5."""
    return "random-min" + str(RECIPE["min_span_length"]) + "max" + str(span)


def eval_cells():
    return [c["id"] + "-step" + str(step) for c in configurations() for step in STEPS]


def head_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def checked_config(case):
    c = READ(Path(case) / "campaign.json")
    if c["runs"] != configurations():
        raise ValueError("campaign runs do not match this source; refusing to reuse the case")
    if c.get("recipe") != RECIPE:
        raise ValueError("campaign recipe does not match this source")
    if c.get("commit") != head_commit():
        raise ValueError("campaign was created from a different commit")
    return c


def initialize(case, student, teacher, dataset, template_case):
    case = Path(case).resolve()
    if (case / "campaign.json").exists():
        raise ValueError("case already initialised; inspect it before reusing")
    template = READ(Path(template_case) / "eval-template.json")
    (case / "eval").mkdir(parents=True, exist_ok=False)
    WRITE(case / "eval-template.json", template)
    WRITE(case / "campaign.json", dict(
        source=str(ROOT), commit=head_commit(),
        source_dirty=subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip(),
        runs=configurations(), recipe=RECIPE, steps=list(STEPS), eval_seeds=[42, 43, 44],
        student=student, teacher=teacher, dataset=dataset,
        eval_template_source=str(template_case), created=time.time(),
        scope="Random-span ladder at the company-v2-random recipe; min span 2 with max span 2, 3, "
              "4 and 5 (spans 2..N), training seeds 42/43/44"))
    return case


def run_dir(case, config):
    return Path(case) / "train" / config["id"]


def receipt_path(case, config):
    return Path(case) / "train" / (config["id"] + ".last-exit.json")


def resumable(case, config):
    return (run_dir(case, config) / "checkpoints" / "latest.json").is_file()


def train_env(case, config):
    c = checked_config(case)
    env = {k: v for k, v in os.environ.items() if k not in RECIPE_FLAGS}
    slot = slots()[config["variant"]]
    env.update(
        CUDA_VISIBLE_DEVICES=str(slot),
        KDFLOW_ROLLOUT_PORT_BASE=str(15000 + 1000 * slot),
        KDFLOW_ROUTER_PORT_BASE=str(23000 + 1000 * slot),
        KDFLOW_ROUTER_PROMETHEUS_PORT=str(20000 + 1000 * slot),
        MP_STUDENT_PATH=c["student"], MP_TEACHER_PATH=c["teacher"], MP_DATASET_PATH=c["dataset"],
        MP_SEED=str(config["train_seed"]), MP_PARTITION_SEED=str(PARTITION_SEED),
        MP_FIXED_SPAN_LENGTH=str(config["span"]), MP_MAX_SPAN_LENGTH=str(config["span"]),
        MP_MIN_SPAN_LENGTH=str(RECIPE["min_span_length"]),
        MP_ALGORITHM="mp_opd", MP_ATTN_IMPLEMENTATION="eager",
        MP_MICRO_TRAIN_BATCH_SIZE=str(config["micro_B"]),
        MP_PREFLIGHT_ONLY="0",
        MP_RESUME="1" if resumable(case, config) else "0",
        MP_CHECKPOINT_STEPS=",".join(str(s) for s in STEPS),
        # Ray appends a session and socket names below this path; keep it short for AF_UNIX.
        MP_RAY_TMP="/tmp/ar" + str(os.getpid()) + "-" + str(time.time_ns() % 1000000),
    )
    return env, slot


def train_one(case, config):
    """Launch one run into its own directory, resuming when a checkpoint exists."""
    case = Path(case)
    env, slot = train_env(case, config)
    target = run_dir(case, config)
    record = receipt_path(case, config)
    if record.exists() and READ(record)["returncode"] == 0:
        return target
    if target.exists() and not resumable(case, config):
        # No checkpoint means nothing to resume; move the husk aside instead of colliding.
        husk = target.with_name(target.name + ".abandoned-" + str(time.time_ns()))
        target.rename(husk)
        print("ABANDONED", husk, flush=True)
    log = target.parent / (config["id"] + ".attempt-" + str(time.time_ns()) + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    argv = ["bash", str(HOST_WRAPPER), str(RUNNER), RUNNER_MODE, str(UPDATES), str(target)]
    print("RUN_LOG", log, "slot", slot, "resume", env["MP_RESUME"], flush=True)
    with log.open("xb") as handle:
        result = subprocess.run(argv, env=env, stdout=handle, stderr=subprocess.STDOUT)
    WRITE(record, dict(returncode=result.returncode, log=str(log), argv=argv, slot=slot,
        resume=env["MP_RESUME"], run_dir=str(target), finished=time.time()))
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, argv)
    return target


def finished(case, config):
    record = receipt_path(case, config)
    return record.exists() and READ(record)["returncode"] == 0


def run_dir_of(case, config):
    return Path(READ(receipt_path(case, config))["run_dir"])


def train(case, only=None):
    case = Path(case)
    checked_config(case)
    placement = slots()
    print("PLACEMENT", placement, flush=True)
    for seed in TRAIN_SEEDS:
        wanted = [c for c in configurations() if c["train_seed"] == seed]
        if only is not None:
            wanted = [c for c in wanted if c["id"] == only]
        pending = [c for c in wanted if not finished(case, c)]
        for config in wanted:
            if config not in pending:
                print("SKIP", config["id"], "already finished", flush=True)
        if not pending:
            continue
        children = [(config, subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                               "train-one", "--case", str(case), "--id", config["id"]]))
                    for config in pending]
        failed = [config["id"] for config, child in children if child.wait()]
        print("ROUND", seed, "finished", "failed=" + (",".join(failed) if failed else "none"), flush=True)
        if failed:
            raise SystemExit("training failed: " + ",".join(failed))
    print("ALL_TRAINS_DONE", len(configurations()), flush=True)


def eval_plan(case, config, step):
    """Plan one checkpoint with a truthful mode label, then delegate to the shared evaluator."""
    case = Path(case)
    path = case / "eval" / (config["id"] + "-step" + str(step)) / "plan.json"
    if path.exists():
        return F.eval_plan(case, config["id"], step)
    source_dir = run_dir_of(case, config)
    summary = READ(source_dir / "checkpoint" / "run-summary.json")
    if summary.get("status") != "completed" or summary.get("optimizer_updates") != UPDATES:
        raise ValueError("training not complete for " + config["id"])
    E, D, Q = B.eval_modules()
    plan = READ(case / "eval-template.json")
    identity = B.checkpoint_stable(source_dir / "checkpoint" / ("step" + str(step)))
    plan.update(jobs=[dict(id=config["id"] + "-step" + str(step),
                         mode=eval_mode_label(config["span"]), step=step, tier=0,
                         checkpoint=identity)],
                source=D.script_hashes(), hours=168, admit_hours=168)
    WRITE(path, plan)
    WRITE(path.parent / "state.json", dict(
        plan_sha256=E.file_hash(path), started=time.time(), deadline=time.time() + 168 * 3600,
        admit_until=time.time() + 168 * 3600, jobs={}, durations=[],
        qualification=Q.preflight("/usr/bin/python3.12")))
    return path, plan


def report(case):
    case = Path(case)
    checked_config(case)
    rows = []
    for config in configurations():
        for step in STEPS:
            path = case / "eval" / (config["id"] + "-step" + str(step))
            plan = READ(path / "plan.json")
            job_id = plan["jobs"][0]["id"]
            for bench in plan["data"]:
                for seed in plan["seeds"]:
                    metric = READ(path / "cells" / job_id / bench / str(seed) / "metrics.json")
                    rows.append(dict(run=config["id"], span=config["span"],
                                     train_seed=config["train_seed"], step=step,
                                     benchmark=bench, eval_seed=seed, metrics=metric))
            if READ(path / "lcbfix" / "summary.json")["status"] != "completed":
                raise ValueError("LCBfix incomplete for " + str(path))
    WRITE(case / "report.json", dict(status="completed", runs=len(configurations()),
        checkpoints=len(eval_cells()), cells=rows, eval_seeds=[42, 43, 44],
        note="Training seeds and evaluation seeds are separate axes; no imputed failed scores"))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("init")
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--template-case", type=Path, required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--dataset", required=True)
    p = sub.add_parser("plan")
    p = sub.add_parser("train")
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--id")
    p = sub.add_parser("train-one")
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--id", required=True)
    p = sub.add_parser("status")
    p.add_argument("--case", type=Path, required=True)
    p = sub.add_parser("plan-eval")
    p.add_argument("--case", type=Path, required=True)
    p = sub.add_parser("report")
    p.add_argument("--case", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "init":
        print("CASE_READY", initialize(args.case, args.student, args.teacher, args.dataset,
                                       args.template_case))
    elif args.action == "plan":
        for config in configurations():
            print(config["id"], "span", config["span"], "seed", config["train_seed"],
                  "slot", slots()[config["variant"]])
        print("trains", expected_trains(), "eval cells", len(eval_cells()))
    elif args.action == "train-one":
        config = next(c for c in configurations() if c["id"] == args.id)
        print("TRAIN_START", config["id"], "slot", slots()[config["variant"]], flush=True)
        print("TRAIN_DONE", train_one(args.case, config))
    elif args.action == "train":
        train(args.case, args.id)
    elif args.action == "status":
        print("placement", slots())
        for config in configurations():
            record = receipt_path(args.case, config)
            state = READ(record) if record.exists() else None
            print(config["id"], "rc=" + str(state["returncode"]) if state else "not started",
                  "resumable=" + str(resumable(args.case, config)))
    elif args.action == "plan-eval":
        for config in configurations():
            for step in STEPS:
                eval_plan(args.case, config, step)
        print("PLANS_READY", len(eval_cells()))
    elif args.action == "report":
        report(args.case)
        print("REPORT_READY", Path(args.case) / "report.json")


if __name__ == "__main__":
    main()
