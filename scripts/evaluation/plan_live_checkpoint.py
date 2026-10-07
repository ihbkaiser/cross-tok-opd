#!/usr/bin/env python3
"""Build a per-step eval plan from any complete HF checkpoint directory.

`queue_data.prepare` only plans finished 312-update campaign runs, and
`queue_full_alternating.eval_plan` only plans checkpoints of a registered
case run. Neither can plan a checkpoint saved by a run that died, or a
checkpoint whose run is still training. This fills exactly that gap: given a
checkpoint directory on disk, write the `<case>/eval/<run>-step<N>/` plan and
state the existing `worker`/`score-spool` consume unchanged.

What is verified, in order, and why each check exists:

* prepared data files exist with the pinned profile and counts -- the cells
  would otherwise hash against data nobody can reproduce;
* `checkpoint_identity` of the directory (config, tokenizer, weight shards
  all present and hashed) -- an incomplete save must never enter a plan;
* an existing plan/state pair must match byte-for-byte -- resuming against a
  drifted plan would score the wrong weights under the right name;
* `state.json` carries a freshly measured scorer `preflight` qualification,
  exactly like `eval_plan`, so `score-spool` cannot run under a drifting
  grader.

Nothing here starts a GPU server or generates anything. Generation stays where
it is proven: `eval_queue.py worker`, on whatever GPU is free, reading the
plan this writes.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import contract_eval as E
import queue_data as D

# eval_queue is imported lazily inside ensure_state: it needs fcntl, so a
# top-level import would make even the pure plan builder unimportable on
# machines without it (including the Windows box that unit-tests this file).

BENCHES = ("gsm8k", "math500", "mbpp", "live-code-bench-v6")
SEEDS = (42, 43, 44)


def build_plan(case: Path, run: str, step: int, mode: str, checkpoint: Path,
               prepared_root: Path, seeds=SEEDS) -> tuple[Path, dict]:
    """Write plan.json (+state.json) for one checkpoint. Returns (plan_path, plan)."""
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    data = {}
    for benchmark in BENCHES:
        path = prepared_root / f"{benchmark}.json"
        if not path.is_file():
            raise ValueError(f"prepared eval data missing: {path}")
        dataset = json.loads(path.read_text())
        if dataset.get("profile") != D.PROFILE:
            raise ValueError(f"prepared data profile mismatch at {path}")
        if len(dataset.get("items", [])) != E.COUNTS[benchmark]:
            raise ValueError(f"prepared data count mismatch at {path}")
        data[benchmark] = {
            "path": str(path.resolve()),
            "sha256": E.file_hash(path),
            "count": len(dataset["items"]),
            "source": dataset.get("source"),
        }
    identity = E.checkpoint_identity(checkpoint)
    job_id = f"{run}-step{step}"
    job = {"id": job_id, "mode": mode, "step": step, "tier": 0, "checkpoint": identity}
    plan = {
        "schema": "eval-queue-v1",
        "profile": D.PROFILE,
        "seeds": list(seeds),
        "data": data,
        "jobs": [job],
        "protocol": {
            "context_length": 8192, "temperature": 0.6, "top_p": 0.95, "n": 1,
            "caps": E.CAPS,
            "math": "author helpers, not math-verify",
            "mbpp": "author assertion helper 10 seconds/problem",
            "lcb": "pinned official tester; public+private; functional-specific prompt",
            "code_execution": "explicit internal subprocess, NOT namespace/container isolation",
            "limits": "2 GiB address space, 120s CPU/problem, <=120s wall/problem",
            "scope": "3 evaluation seeds, exploratory checkpoint selection, not exact paper replication",
        },
        "source": D.script_hashes(),
        "hours": 168,
        "admit_hours": 168,
    }
    root = case / "eval" / f"{run}-step{step}"
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text())
        if existing != plan:
            raise ValueError(f"eval plan drift at {plan_path}")
    else:
        E.write_new(plan_path, plan)
    return plan_path, plan


def ensure_state(plan_path: Path, plan: dict, score_python: str = "/usr/bin/python3.12") -> Path:
    """Write state.json with a measured qualification, mirroring eval_plan."""
    from eval_queue import atomic_json, locked, preflight, read_state

    root = plan_path.parent
    state_path = root / "state.json"
    with locked(root / "state.lock"):
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get("plan_sha256") != E.file_hash(plan_path):
                raise ValueError(f"eval state drift at {state_path}")
            return state_path
        # read_state creates the skeleton; qualification is measured below.
        state = read_state(root, E.file_hash(plan_path), plan)
        state["qualification"] = preflight(score_python)
        atomic_json(state_path, state)
    return state_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--score-python", default="/usr/bin/python3.12")
    parser.add_argument("--seeds", default="42,43,44")
    args = parser.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    plan_path, plan = build_plan(
        args.case, args.run, args.step, args.mode, args.checkpoint,
        args.prepared_root, seeds=seeds,
    )
    state_path = ensure_state(plan_path, plan, args.score_python)
    print("PLAN_READY=" + str(plan_path), flush=True)
    print("STATE_READY=" + str(state_path), flush=True)
    print("CHECKPOINT_SHA256=" + plan["jobs"][0]["checkpoint"]["sha256"], flush=True)


if __name__ == "__main__":
    main()
