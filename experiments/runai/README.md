# RunAI single-GPU MP-OPD launcher

Evaluation preparation (protocol, known paper/code mismatches, read-only inventory):
[SimCT evaluation preparation](../../docs/simct_eval_preparation_20260909.md).
This does not schedule evaluation or start GPU work.
Implemented runner: [evaluation contracts](../../scripts/evaluation/CONTRACT_README.md)
(`author-code` and explicitly reconstructed `paper-spec`, offline preparation,
five-seed generation, sandboxed scoring and complete-result aggregation).

Adapted from the verified server scripts collected on 2026-09-09.
Run one atomic job on slot 0 and fixed-2 on slot 1. Each job creates its own
Ray runtime/cache, rollout port range and Prometheus listener.

From a Git checkout:

```bash
bash experiments/runai/run_single_gpu.sh 0 atomic 5
bash experiments/runai/run_single_gpu.sh 1 fixed 5
```

Positive limits 1..312 bound executed updates; 5 is a canary, 50 a pilot, and 0
requests the configured full 312 updates. All retain a 312-update scheduler horizon.
Atomic/fixed/random/soft are supported. Soft requires a compatible energy checkpoint.
See [borrowed eight-hour campaign](BORROWED_8H.md) for the separate-host queue,
FP64 partition DP qualification and cooperative checkpoint cutoff.
See [A/B/C contract and oracle guide](../../docs/mp_opd_abc_20260909.md) for seed/span
overrides, CPU preflight, input hashes, baseline algorithm selection and controls.
Microbatch is 4, eager attention, 2 epochs, train batch 64 (16 accumulation steps).
Microbatch 4 is user-requested and has not yet been GPU-validated with MP-OPD;
the prior microbatch-2 canary does not validate its memory use or parity.
Keep runtime libraries outside the checkout:
MP_RUNTIME_DIR defaults to /workspace/storage-shared/nlp/tungks/MP-OPD/work-QA9lLufL.
MP_RUN_ROOT defaults to simct-runs beside the checkout. Model/data defaults
match the original RunAI job; override MP_SHARED_ROOT, MP_STUDENT_PATH,
MP_TEACHER_PATH and MP_DATASET_PATH when paths differ.

The Python wrapper retains CUDA 13, NVRTC include/CPATH and per-job FlashInfer
workspace fixes. The launcher sets PYTHONPATH to this Git checkout and records
source commit and tracked dirty status in launch-config.json.

KDFLOW_ROLLOUT_PORT_BASE and KDFLOW_ROUTER_PROMETHEUS_PORT explicitly isolate
ports. Generic KDFlow callers keep defaults when these variables are absent;
no parsing of CUDA_VISIBLE_DEVICES is forced on multi-GPU or UUID callers.
Use one active run per numeric slot (0..7). Identical slot/port configurations
on the same host are not supported.

The updated MP-OPD temperature/quantile parity checks are retained. The old
server's unconditional finite-outlier bypass is not restored.

Validation: stdlib port regression tests plus Python AST and bash syntax checks.
A GPU canary with the combined Git launcher is still required on the server.
Full server source reconciled against simct-server-source-20260909-041655.tar.gz
(SHA-256 8bda6cea97d65395befa4b13d1e8a29a584db4a105ea9a68cb1740938de62bde).
Atomizer differences are whitespace-only. The newer trainer compatibility fix
and temperature/mean/p99/NaN/Inf parity checks are retained. Historical backup
files are kept in the local evidence archive and not installed as source.
The server's port isolation and launcher/NVRTC changes are integrated here.
No combined GPU training canary has been run after this reconciliation.

Distribution uses the same source commit on GitHub and inside the public HF
Git bundle at https://huggingface.co/codemaivanngu/simct. HF's repository commit
identifies the bundle upload and differs from the source commit recorded in
source-manifest.json. Download bundle and manifest at the same HF revision,
verify SHA-256, then clone into a new directory or pull --ff-only from the bundle.
Keep local model paths/runtime paths in environment variables and outputs
outside the Git checkout. Git may still report conflicts for future source edits;
this integration removes the need to reapply the known server patches manually.


### Separate generation and CPU scoring

`eval_queue.py worker --phase generate` writes durable response journals and a
hash-bound `generation-complete.json` per cell. It never invokes the scorer.
The two GPU workers use `generation-state.json`, independent of scoring failures;
completed legacy metrics are reused. Generation reserves at least 20 GiB of free
disk (`--min-free-gib`) and retains the existing admission/deadline budget.

`eval_queue.py score-spool --plan PLAN --internal-code-execution` uses one
coordinator and at most 16 CPU scorers. It consumes completed generation cells,
reuses saved scores, never launches a model server, and records item IDs plus
subprocess return codes/stdout/stderr on failures. Failed cells are skipped for
that coordinator invocation; other cells continue. Restarting the coordinator
retries unfinished scoring. No infrastructure failure is converted into an
incorrect answer. Existing 120-second timeout semantics remain unchanged.

Scoring starts after each full benchmark/seed cell is generated, not after the
entire checkpoint. Both phases retain the original deadline. A generation job's
completed status does not imply scoring completion; use `summarize` for metrics.
GPU workers release their owned servers at checkpoint transitions and on exit.

The transfer package's `start-separated-eval.sh` validates and forks an existing
25-checkpoint queue, preserves journals and the original clock, then launches two
generation workers and one scoring coordinator. Source the old queue environment
first. It prints the new `queue.env` and log paths. Old queue files remain intact.

The 128-request trial enables up to 128 only for generation phase; combined
workers remain capped at 64. The transfer launcher sets GPU0=128 and GPU1=64.
Migration stops the exact scoring coordinator as well as generation workers,
rebinds generation markers, and preserves completed generation jobs and clocks.
The launcher writes `launched.env` alongside itself for telemetry commands.

The next trial uses GPU0=256 and GPU1=128 generation requests; only generation
accepts up to 256. CPU scoring remains 16 workers with buffer 128. The worker
CLI buffer 256 is validation-only in generation mode: responses spool to disk.
Both phases resume after migration; scorer is briefly stopped for consistent
copying, then restarted alongside generation. Original deadlines remain.

### Append completed evaluation to W&B

`scripts/evaluation/publish_wandb_eval.py --summary FILE --out DIR --publish`
validates the 25-checkpoint aggregate and four exact finished run IDs before
appending eval-only history, a checkpoint table, endpoint summaries and an
evaluation artifact. Existing training history is read back and checked as an
unchanged prefix. Successful runs receive the `evaluated` tag with existing tags
preserved. A source SHA marker prevents re-upload; partial imports fail closed.
No training logs are replayed. Aggregate provenance is not a per-item audit.

`publish_wandb_uncertainty.py --summary FILE --out DIR` appends seed dispersion
to those already-evaluated runs. It uses sample variance (ddof=1), computes macro
per seed before dispersion, and adds spread lines (mean/min/max/mean +/- std)
and seed charts. These are evaluation-seed variability, not confidence intervals.
History-prefix and endpoint readback checks follow each upload.

## Oracle recovery from prepared guide

`recover_oracle.py --original MAIN_WORK --guide GUIDE_WORK --out FRESH_WORK`
uses existing local tokenizers (run through `python-b200-host.sh`) before model
allocation. It preserves every prepared guide row, verifies split identities,
uses explicit 4096 prompt/reference caps with no truncation, and checks context
capacity. It prepares GPU0 primary LR .1 and GPU1 sequential .01/1.0 diagnostics.
Start the returned plan with `python3.12 experiments/runai/campaign.py FRESH_WORK/plan.json`.
The original start/deadline is retained; sensitivity jobs share the remaining
budget, timeouts preserve partial artifacts but are not completed results.
No full training is promoted automatically. CPU endpoint scoring may continue.

## Learned partition pilot

`learned_partition_pilot.py prepare --prior ORACLE_WORK --original MAIN_WORK --work FRESH_WORK`
prepares a GPU1-only DAG: 32-group guide energy learning (45-minute cap), CPU
technical qualification, 5-update soft canary (20-minute cap), then fresh-SFT
50-update soft student pilot (90-minute cap). Run with `campaign.py FRESH_WORK/plan.json`.
The original campaign deadline is retained; jobs that cannot fit are skipped.
GPU0 random training and independent endpoint scoring remain separate.

The energy is the actual 10-feature MPAtomEnergy GRU/span scorer used by the soft
student loss. It learns a first-order select-utility surrogate (normalized by
student token count), not exact Adam hypergradients. Every group is evaluated
before learning from that group's select references; eval does not train or
admit the pilot. Each select-count has an independent network. Dropout is off
via eval mode while gradients are enabled. Energy checkpoints are saved after
every valid group; the student pilot freezes the learned energy. This is a
staged learned-partition experiment, not simultaneous online bilevel training.

`--norm-controls` adds fixed, random, oracle, weighting, learned-partition and
meta-SFT updates rescaled to the atomic gradient norm in the adapter diagnostic;
zero-norm matches are marked invalid. These norm controls do not normalize
full-student Adam updates. Soft pilot checkpoints save every 20 updates and
at completion. Record and compare common update checkpoints against baselines.
The new GPU path needs company canary validation; local tiny-model tests do
not establish B200 execution or efficacy.

Energy GRU eval mode uses the native PyTorch RNN backend, including during
surrogate backward: cuDNN inference-mode RNNs do not support backward. This
keeps dropout disabled consistently in evaluation and energy learning; the
backend flag is scoped to the GRU forward and restored immediately. A CUDA
regression test runs when CUDA is available (otherwise explicitly skipped).

### Recover completed learned energy

`learned_partition_pilot.py recover --prior OLD_WORK --work NEW_WORK` reuses
the completed energy directory and reruns technical qualification, canary, then
student pilot on GPU1. Qualification uses the lightweight algorithm import so
it does not require unrelated X-Token modules. The original campaign clock is
preserved; the pilot budget shrinks to fit after qualification and canary with
a three-minute margin. At least 40 minutes must remain. A timed-out pilot may
produce only partial checkpoints; it is not a completed 50-update result.

## Six-B200 continuation (`six_b200.py`)

This campaign uses four borrowed GPUs for 12 hours and two owner GPUs without
an allocation deadline. Initialize **on tungdd11-sparse-vllm-core-0-0** with
`/usr/bin/python3.12 experiments/runai/six_b200.py init --hours 12`, then run
`/usr/bin/python3.12 experiments/runai/six_b200.py submit` on both nodes, using
the same immutable source checkout. The first init persists the borrowed node's
wall clock in `borrow8-8MgodXcM/six-b200-v1/allocation-clock.json` before setup.
It prints UTC and Vietnam time. Repeating init or submit never extends the clock.
The 12-hour allocation must actually begin at initialization; this cannot extend
an external RunAI allocation. Do not initialize ahead of the reserved window.

| Node / slot | First work | Subsequent work |
| --- | --- | --- |
| Owner 0 | soft seed42, fresh SFT to 312 | shared generation pool |
| Owner 1 | soft seed43, fresh SFT to 312 | shared generation pool |
| Borrowed 0 | random seed43, fresh SFT to 312 | shared generation pool |
| Borrowed 1, 2, 3 | shared generation pool | newly published checkpoints |

Random43 is a full-horizon replay, not a third independent training seed and not
an optimizer-state resume of the old run that stopped at 244. It is skipped if
less than seven hours remain at admission. Its cooperative stop starts with
15 minutes left, with an additional five-minute checkpoint reserve. Owner jobs
have no allocation deadline; each has a 72-hour operational watchdog.
Neither existing queues nor unrelated GPU processes are stopped. New managers
use existing UUID lease files and idle-memory/utilization admission. GPU3 waits
if the prior vLLM process still occupies it. Source, runtime, seed and energy
checks precede work; the known parity gate is not relaxed.

### Experiment contract

Soft uses the same fixed energy checkpoint as the successful 50-update pilot,
eager attention, batch64/microbatch4, 312-step LR horizon and fresh SFT weights.
Two training seeds (42,43) are exploratory replication, not a precise variance
estimate. Compare to the already completed fixed/atomic/SimCT curves using
matched training seed and update count. Do not compare soft50 directly to a
312-step baseline to decide failure. The existing random42-full312 and SimCT43
full312 curves are added automatically; completed historical baseline pools
remain unchanged. No oracle full training is launched: previous adapter-level
headroom is small and not evidence of downstream efficacy.

The shared pool publishes 40/80/120/156/200/240/280/312 only after training has
advanced beyond the save, or a successful final summary plus exitcode exists.
Each checkpoint has an immutable plan and ready record with training seed,
checkpoint hash and source provenance. The publisher's final marker is written
after its last checkpoint scan. Generation claims use cross-node file locks;
scoring uses per-cell locks. Evaluation preserves the previous template's data,
prompts, decoding and seeds42/43/44, with generation concurrency256 and separate
scoring. Four borrowed and two owner coordinators each use four CPU scorers
(24 total subprocess slots); scoring never reserves a GPU. LCBfix writes a
separate result on the owner node after the original 12 cells complete.

Generation uses the minimum of its worker deadline and borrowed allocation end,
stopping admission with ten minutes left. Partial journals can be continued by
owner workers. Score errors stay errors, not zero scores, and do not stop other
cells. `generation-error.json`, `score-error.json` and `lcbfix-error.json` require
inspection; there are no blind automatic retries. Pool metadata has a seven-day
operational lifetime, distinct from the borrowed 12-hour allocation.

Run `six_b200.py status` on either node to read the shared checkpoint/cell counts
and that node's manager state. The shared receipts `six-b200-v1/HOST.json` hold
the local manager path and every submitted job spec. Submission is idempotent.
Local tests cover clock reuse, queue boundaries and checkpoint publication;
they do not establish GPU execution or the remote filesystem lock behavior.

For the original `2d4d3ea` adapter, `repair_six_gen.py recover` replaces only
generation jobs on the current node. Run it on borrowed then owner nodes using
a new checkout; it imports the old pinned source at runtime and supplies the
missing `min_free_gib=20` argument. It archives only matching generation errors
under the checkpoint lock, preserves unrelated errors, and retains job GPU
leases, training dependencies and deadlines. It does not modify the old source,
plans or training jobs. Repeated recovery uses the same replacement IDs.

### Matched seed44 extension

Run `queue_seed44.py submit` on borrowed then owner nodes, from a new checkout.
It retains the original training/evaluator source and original allocation end
as a reporting reference. It creates new local managers with no allocation
deadline, shared GPU lease directories, 64 borrowed / 40 owner CPU slots, and
72-hour per-job operational watchdogs. The 12-hour reference does not terminate
new workers. Existing active training is left alone: random43 retains its
previously captured cooperative and supervisor deadlines; editing the database
cannot remove those in-memory limits. Its full312 completion must be verified.
Borrowed GPUs1/2/3 finish the initial 16-checkpoint gen batch or spend at most
45 further minutes on gen, then switch to fixed2/atomic/SimCT seed44 full312.
The handoff cancels only their named gen adapters; durable response journals
remain resumable. New training has no allocation-based early stop. GPU0 random43
is left running; a bridge waits for its old manager terminal state before gen.

Owner GPU0 runs random44 after soft42; owner GPU1 runs soft44 after soft43.
Both return to shared gen afterwards. A replacement CPU publisher includes all
eight new campaign training runs plus the two existing completed runs; its final
marker waits for the extended producer set. Old scorer coordinators are replaced
with workers using the same evaluator and no allocation cutoff. Queue
terminal statuses are exported so blocked/failed training cannot masquerade as
a successful run or keep the publisher waiting indefinitely. No existing active
training run is canceled. Historical fixed/atomic/SimCT42 and fixed/atomic43
results are reused, not retrained.

Use `queue_seed44.py status`: the extended new pool has at most 80 checkpoint
plans / 960 cells. Existing baseline curves remain outside this pool. Analysis
compares five methods across three training seeds, paired at the eight declared
steps, with three eval seeds each. Random43 full replay replaces the truncated
run in that comparison; it is not counted as an additional independent seed.
Frozen energy is shared across soft seeds; this does not measure energy-training
variance. Report raw per-training-seed results, original LCB and LCBfix separately,
and missing/failed cells. No score-based checkpoint selection or automatic
oracle promotion is performed.

### Unattended scoring recovery

From a new pinned checkout, run `resilient_score.py submit` on each node. It
replaces only `score-seed44-*` coordinators in the seed44 manager with stable
`score-resilient-*` IDs. It leaves training, generation, datasets, responses and
the pinned evaluator hashes unchanged. Concurrent retries own the same cell
`score.lock`; completed metrics are not reopened. Each polling pass can recover
eligible markers left by a coordinator replacement on either host.

SIGTERM scorer failures receive up to three retries per cell; SIGKILL and
SIGSEGV receive one diagnostic retry each, with at most four retries total.
Every reservation is persisted in `score-recovery.json` before retry, so process
restarts do not reset budgets. Retry uses one scoring worker and a ten-second
delay for failures caught in-process, resuming the existing score journal.
Marker recovery archives the original error and starts serially. Unknown errors,
contract/hash mismatches and deadline exceptions are not automatically retried.
Repeated native crashes remain unresolved errors, never incorrect-answer zeros.
Other cells continue. At coordinator completion, `HOST.scoring-attention.json`
lists remaining scoring/LCBfix errors and causes a nonzero job exit. This is an
artifact for review, not an external notification service. LCBfix subprocess
retries and whole-manager/host recovery are outside this adapter's scope.
