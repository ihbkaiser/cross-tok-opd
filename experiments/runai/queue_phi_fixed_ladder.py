#!/usr/bin/env python3
"""Phi-4-mini -> Gemma-2-2B fixed-span ladder: arms fix2, fix3, fix4, fix5.

Thin wrapper over queue_fixed_span_ladder. The base module owns the contract
(placement, recipe shape, step milestones, receipt/resume behaviour) and keeps one
VARIANTS tuple and one run-id prefix; this pair needs a different arm set, a distinct
id prefix and its own scope text, and those are the only things overridden here.

The second pair of the paper (teacher Phi-4-mini, student Gemma-2-2B-IT) reuses the
same SFT recipe as the qwen pair, so the arm set is what changes, not the recipe:
span 2 (the historical `fixed` arm) through span 5, i.e. the full fixed range that
this pair's plan asks for.

DEFAULT_SLOTS is placement only and stays out of the campaign contract, but slots()
requires exactly one distinct slot per variant in 0..7, so four arms need four slots.

Usage is identical to the base module:
    python3 queue_phi_fixed_ladder.py init --case ... --student ... --teacher ... --dataset ...
    python3 queue_phi_fixed_ladder.py train-one --case ... --id PHI-fix2-s42
"""
import queue_fixed_span_ladder as base

base.VARIANTS = (("fix2", 2), ("fix3", 3), ("fix4", 4), ("fix5", 5))
base.DEFAULT_SLOTS = (4, 5, 6, 7)
base.ID_PREFIX = "PHI-"
base.SCOPE = ("Phi-4-mini -> Gemma-2-2B-IT fixed-span ladder at the company-v2-fixed "
              "recipe; span 2, 3, 4 and 5, training seeds 42/43/44")


if __name__ == "__main__":
    base.main()
