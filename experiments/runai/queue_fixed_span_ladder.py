#!/usr/bin/env python3
"""Fixed-span ladder: span 3 and 4 at the company-v2-fixed recipe, one variant per GPU slot.

Reuses the alternating queue for shared machinery (config gate, eval plan, generation, scoring).
Only the training command differs: mode 'fixed' with both MP_FIXED_SPAN_LENGTH and
MP_MAX_SPAN_LENGTH set to the same value, because experiments/runai/run_single_gpu.py refuses a
fixed span longer than the max span (run_single_gpu.py:176).
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

# The ladder. The recipe is the historical company-v2-fixed one, which is what
# 'run_single_gpu.sh <slot> fixed 312' produces from the launcher defaults.
VARIANTS = (("fixed3", 3), ("fixed4", 4))
TRAIN_SEEDS = (42, 43, 44)
# One variant per slot so the two variants of one seed train at the same time. The wrapper
# derives its port bases from the slot, so two concurrent jobs never collide.
SLOTS = {"fixed3": 6, "fixed4": 7}
STEPS = (40, 80, 120, 156, 200, 240, 280, 312)
UPDATES = 312
PARTITION_SEED = 43
RECIPE = dict(mode="fixed", learning_rate=1e-6, micro_train_batch_size=4, train_batch_size=64,
              num_epochs=2, optimizer_updates=UPDATES, alternating=False,
              partition_seed=PARTITION_SEED, temperature=0.6, top_p=0.95,
              generate_max_len=4096, max_len=4096, source_group="company-v2-fixed")


def configurations():
    return [dict(id="FIX-" + name + "-s" + str(seed), variant=name, span=span, mode="fixed",
                 train_seed=seed, slot=SLOTS[name], student_updates=UPDATES,
                 micro_B=RECIPE["micro_train_batch_size"], train_B=RECIPE["train_batch_size"],
                 student_lr=RECIPE["learning_rate"])
            for seed in TRAIN_SEEDS for name, span in VARIANTS]


def expected_trains():
    return len(configurations())


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
        scope="Fixed-span ladder at the company-v2-fixed recipe; span 3 and 4, training seeds 42/43/44"))
    return case


def train_env(case, config):
    c = checked_config(case)
    # A stray MP_* value in the interactive shell must never reach an immutable run.
    env = {k: v for k, v in os.environ.items() if not k.startswith("MP_")}
    root = Path(case) / "train" / config["id"]
    root.mkdir(parents=True, exist_ok=True)
    env.update(
        MP_STUDENT_PATH=c["student"], MP_TEACHER_PATH=c["teacher"], MP_DATASET_PATH=c["dataset"],
        MP_SEED=str(config["train_seed"]), MP_PARTITION_SEED=str(PARTITION_SEED),
        MP_FIXED_SPAN_LENGTH=str(config["span"]), MP_MAX_SPAN_LENGTH=str(config["span"]),
        MP_ALGORITHM="mp_opd", MP_ATTN_IMPLEMENTATION="eager",
        MP_MICRO_TRAIN_BATCH_SIZE=str(config["micro_B"]),
        MP_PREFLIGHT_ONLY="0", MP_RESUME="0", MP_RUN_ROOT=str(root),
    )
    return env, root


def train_one(case, config):
    """Launch one run; the wrapper names its own directory under MP_RUN_ROOT."""
    case = Path(case)
    env, root = train_env(case, config)
    started = [p for p in root.glob("qwen-gemma-*") if p.is_dir()]
    if len(started) > 1:
        raise ValueError("ambiguous run directory under " + str(root))
    log = root / (config["id"] + ".attempt-" + str(time.time_ns()) + ".log")
    argv = ["bash", str(ROOT / "experiments/runai/run_single_gpu.sh"), str(config["slot"]),
            "fixed", str(UPDATES)]
    print("RUN_LOG", log, flush=True)
    with log.open("xb") as handle:
        result = subprocess.run(argv, env=env, stdout=handle, stderr=subprocess.STDOUT)
    runs = [p for p in root.glob("qwen-gemma-*") if p.is_dir()]
    if len(runs) != 1:
        raise ValueError("expected exactly one run directory under " + str(root))
    run_dir = runs[0]
    WRITE(root / "last-exit.json", dict(returncode=result.returncode, log=str(log), argv=argv,
        slot=config["slot"], run_dir=str(run_dir), finished=time.time()))
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, argv)
    return run_dir


def run_dir_of(case, config):
    return Path(READ(Path(case) / "train" / config["id"] / "last-exit.json")["run_dir"])


def finished(case, config):
    marker = Path(case) / "train" / config["id"] / "last-exit.json"
    return marker.exists() and READ(marker)["returncode"] == 0


def train(case, only=None):
    case = Path(case)
    checked_config(case)
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
    run_dir = run_dir_of(case, config)
    summary = READ(run_dir / "checkpoint" / "run-summary.json")
    if summary.get("status") != "completed" or summary.get("optimizer_updates") != UPDATES:
        raise ValueError("training not complete for " + config["id"])
    E, D, Q = B.eval_modules()
    plan = READ(case / "eval-template.json")
    identity = B.checkpoint_stable(run_dir / "checkpoint" / ("step" + str(step)))
    plan.update(jobs=[dict(id=config["id"] + "-step" + str(step),
                         mode="fixed-span" + str(config["span"]), step=step, tier=0,
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
                  "slot", config["slot"])
        print("trains", expected_trains(), "eval cells", len(eval_cells()))
    elif args.action == "train-one":
        config = next(c for c in configurations() if c["id"] == args.id)
        print("TRAIN_START", config["id"], "slot", config["slot"], flush=True)
        print("TRAIN_DONE", train_one(args.case, config))
    elif args.action == "train":
        train(args.case, args.id)
    elif args.action == "status":
        for config in configurations():
            marker = Path(args.case) / "train" / config["id"] / "last-exit.json"
            if marker.exists():
                print(config["id"], "rc=" + str(READ(marker)["returncode"]))
            else:
                print(config["id"], "not started")
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
