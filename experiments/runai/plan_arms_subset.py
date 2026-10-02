#!/usr/bin/env python3
"""Plan eval for a SUBSET of arms (plan-eval of the ladder plans all 12 and needs every receipt).

    plan_arms_subset.py <case> <arm-id> [arm-id ...]     # env: PYTHONPATH must point at the repo
"""
import os, sys
from pathlib import Path
SRC = os.environ["PYTHONPATH"].split(":")[1]
sys.path.insert(0, SRC); sys.path.insert(0, SRC + "/experiments/runai")
import queue_random_span_ladder as Q
case, want = Path(sys.argv[1]), set(sys.argv[2:])
n = 0
for c in Q.configurations():
    if c["id"] not in want: continue
    for step in Q.STEPS:
        d = case / "eval" / (c["id"] + "-step" + str(step))
        if (d / "plan.json").is_file(): continue
        Q.eval_plan(case, c, step); n += 1
print("plans tao moi:", n)
