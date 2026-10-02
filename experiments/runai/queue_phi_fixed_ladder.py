#!/usr/bin/env python3
"""Phi-4-mini -> Gemma-2-2B fixed-span ladder, ONE ARM PER CASE.

Why one arm per case: the base module places one variant per GPU slot and validates
that the slots are distinct, and the slot is used verbatim as CUDA_VISIBLE_DEVICES.
The phi pair runs on a single B200, so a four-arm case cannot be launched at all.
With exactly one variant the placement is the single-element tuple (0,), and the base
`train` action -- which launches every pending variant of one seed and then waits --
therefore runs exactly one process at a time: the three seeds go through slot 0
sequentially, which is what a one-GPU host can actually do.

Pick the arm through PHI_FIX_ARM; each arm gets its own case (and its own
campaign.json commit/receipt lineage):

    PHI_FIX_ARM=fix2 MP_LADDER_SLOTS=0 python3 queue_phi_fixed_ladder.py init --case ... --student ... --teacher ... --dataset ...
    PHI_FIX_ARM=fix2 MP_LADDER_SLOTS=0 python3 queue_phi_fixed_ladder.py train --case ...

Run ids are stable and pair-scoped: PHI-fix2-s42, PHI-fix2-s43, PHI-fix2-s44, and the
same for fix3, fix4, fix5. Nothing else in the contract changes: span recipe, eight
step milestones, receipts and resume behaviour all come from the base module.
"""
import os

import queue_fixed_span_ladder as base

ARMS = {"fix2": 2, "fix3": 3, "fix4": 4, "fix5": 5}

arm = os.environ.get("PHI_FIX_ARM", "").strip()
if arm not in ARMS:
    raise SystemExit(
        "PHI_FIX_ARM must be one of %s (got %r); one arm per case because the phi "
        "pair runs on a single GPU" % (", ".join(sorted(ARMS)), arm)
    )

span = ARMS[arm]
base.VARIANTS = ((arm, span),)
base.DEFAULT_SLOTS = (0,)
base.ID_PREFIX = "PHI-"
base.SCOPE = ("Phi-4-mini -> Gemma-2-2B-IT fixed-span ladder at the company-v2-fixed "
              "recipe; arm %s (span %d), training seeds 42/43/44, one arm per case "
              "because the pair runs on a single B200" % (arm, span))


if __name__ == "__main__":
    base.main()
