# LCBfix v1

This is a separate evaluation revision, not a new training run. The original
author extractor misses closed `python3` fences. The versioned extractor adds
that tag and otherwise preserves the old first matching block behavior.
Incomplete fences, repetition and algorithmic errors are not repaired.

Run `lcbfix.py --plan OLD/plan.json --out SEPARATE/lcbfix-v1 --workers 4
--internal-code-execution` with the qualified system Python. No GPU is used.
All checkpoints and three evaluation seeds in the source plan are processed.
Unchanged extracted code reuses its old score; changed code is rescored using
the original internal grader and limits. The original source scripts remain
pinned and original journals are read-only. The separate `lcbfix_worker.py`
changes no active evaluation pool or historical scorer.

The output records source plan/data/script hashes, source metric hashes,
per-response provenance and old/new pass counts, gains and losses per cell.
`summary.json` is complete only when all cells finish. A per-output lock prevents
duplicate coordinators; restarting the same command resumes its own score journal.
Infrastructure errors fail the job rather than becoming wrong answers. The
original 120-second outer timeout still counts as failure. Runtime qualification
must match the old pool, and positive/negative python3 smoke cases must pass.

Report this as `lcbfix-v1` alongside the original results. Do not overwrite W&B
historical metrics or describe unchanged-code reuse as fresh execution. No
generation, token-cap change, model-specific extraction or automatic W&B upload
is performed.
