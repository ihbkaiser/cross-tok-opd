# Shared evaluation pool

A pool is one immutable `plan.json` plus its shared directory of state, cells
and journals. Workers on any authorized node mount the same directory and use
the exact source hashes, model/data identities and compatible grading runtime.
Node-local job managers own local GPUs/processes; they must not share SQLite
databases across nodes. Host resource deadlines remain enforced locally.

Generation workers claim checkpoints via a shared exclusive checkpoint lock.
Each GPU defaults to 256 concurrent generation requests and writes durable
responses plus a hash-checked `generation-complete.json` marker per cell.
They never wait for scoring or acquire a GPU from another node.

CPU scoring workers may run on any node, including nodes with no GPU. Each
claims a ready `(checkpoint, benchmark, eval seed)` cell with `score.lock`,
rechecks completion inside the lock, validates the spool contract and scores
only that cell. Multiple scoring workers can process different cells in
parallel. `--score-workers` controls CPU concurrency inside each worker;
provision CPU slots for the sum across workers. Scaling is limited by the
number of ready cells, not individual examples within one cell.

Examples with the same PLAN and pinned source on each node:

```bash
# Generation on an explicitly allocated local GPU; default concurrency is 256.
python3 scripts/evaluation/eval_queue.py worker --plan "$PLAN" --gpu 2 --internal-code-execution
# Scoring elsewhere; no GPU required.
python3 scripts/evaluation/eval_queue.py score-spool --plan "$PLAN" --score-workers 4 --internal-code-execution
```

The shared filesystem must honor cross-node `flock`, already tested on the
current allocation. A shared `scoring.lock` lets new workers coexist while
excluding the historical single exclusive scorer. Worker exit releases claims;
after a crash another worker resumes durable response/score journals. A
scoring error is persisted per cell as `score-error.json`, never silently
retried by every node. After diagnosing it, `retry-scoring --plan "$PLAN"`
archives error markers under cell locks without removing scored examples.

Do not mix combined workers with separated scorers. For existing runs, stop
and verify writers, preserve their artifacts, and use an audited migration
to update the plan/source contract before launching this version. Changing
defaults does not change existing processes or pinned plans. This protocol
does not claim multi-node GPU runtime verification from local tests.

## Seed43 aligned migration

`aligned_eval_pool.py` builds a fresh pool for fixed43/atomic43/random43 at
40,80,120,156,200,240,280,312, skipping and reporting absent checkpoints.
It reads actual training seed/mode from launch configs, hashes the checkpoint
files, and checks seeds/profile/data/protocol and all non-scheduling source
hashes against the historical seed42 evaluation template. A mismatch stops
migration rather than changing the benchmark contract.

It merges existing step200 and step312 response/score journals into a new
directory under exclusive source locks, validates each request/result/hash,
and changes only their plan-contract reference after validation. Fully generated
cells receive a generation-complete marker so generation does not wait for or
read a scorer's live journal. Source artifacts remain intact. The owner's
existing evaluation deadline is retained; borrowed workers keep their own
earlier hard deadline. Completed results are reused, partial journals resume.

For the current allocation, `switch_aligned_pool.py stop` on the borrowed node
cancels only its extra GPU2 eval. Then `owner-start` on the owner node cancels
only old-eval, migrates, and submits generation GPU0 plus a four-worker CPU
scorer. Finally `join` on the borrowed node submits generation GPU2 to the
verified pool. Training, numeric and parity jobs are not cancelled or edited.
Every invocation uses an isolated pinned checkout; do not update live source.
An incomplete migration receipt blocks workers and must be inspected.
