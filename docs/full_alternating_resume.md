# Full alternating and on-policy pipeline resume

## User-authorized unattended scheduling

For the subsequently requested direct launch, use `--qualification-policy skip`
instead of `advisory`. There is no qualification job: after the CPU data audit
terminates, training is eligible immediately. Runs record status `skipped` and
remain diagnostic-unqualified. Dependency failures do not cascade; actual
training/runtime/export validity checks still apply. Use a fresh immutable
case with `--extra-gpu-count 4` on both hosts.

The split launcher supports `submit --qualification-policy advisory
--extra-gpu-count 4 --qualification-timeout-seconds 3600` for a **fresh case**.
Use the same flags on both hosts. This user-selected policy waits for terminal
dependencies rather than successful ones: qualification failure/timeout does
not block training, and a failed seed-42 run does not block the next variant.
The qualification job itself still records its real failure; no PASS marker is
invented. GPU admission still waits for the assigned GPU to be available.

Seed 42 uses the owner GPU sequentially. Seed 43 uses extra-host GPUs 0/1/2;
GPU 3 is the additional generation worker. GPU 4 is excluded. Generation
workers on GPUs 0/1/2 may start after their respective training jobs terminate.
Scorers inspect committed exports and retain missing/failed cells as errors.

Each train writes `<run>.qualification.json`, and the launch manifest includes
`qualification_policy` and `qualification_status`. Without a matching PASS,
the run is `unverified` / `diagnostic-unqualified`, even if training completes.
This is permission to attempt the schedule, not evidence of correct resume or
benchmark efficacy. Runtime/data failures can still prevent training; existing
bounded recovery remains in effect. No live jobs are changed by publishing the
source. The default policy remains `required`; policy/GPU-count changes require
a new immutable case rather than editing or relabeling a running campaign.

## Scope and qualification

`kdflow.training_checkpoint` is shared by the on-policy trainer and student
actors, independent of the six-run campaign and the earlier adapter pilot.
HF exports in `checkpoint/stepN` are for evaluation; they are **not** complete
training checkpoints. New recovery transactions are in `checkpoints/`.
Legacy weight-only checkpoints cannot be used for exact continuation.

The production student path still uses full-parameter FSDP2, the original
teacher-forward and SGLang rollout path, and the original real AdamW update.
The new energy update streams the one-step full-parameter AdamW hypergradient
through the same B loss, including prior moments and global clipping. It holds
optimizer moments fixed for this one-step derivative, restores the virtual
student, updates energy, then recomputes detached rates for the real student.
It is not a full-training-trajectory hypergradient or an adapter approximation.

CPU tests compare the streamed derivative with a direct differentiable AdamW
unroll and compare interrupted/resumed training with uninterrupted training.
**Full B64/M16 FSDP2 higher-order autograd, CUDA memory use and SGLang replay
remain target-runtime qualification requirements.** The queue first runs two
full-pipeline canaries: main and every4, each continuous four updates versus
pause after three and resume to four. Weights, optimizer/scheduler/energy/RNG,
data cursor, sampled token IDs and behavior logprobs must match exactly.
Serving request IDs and timing telemetry are excluded. A failure blocks all
six long runs; there is no silent low-rank or approximate fallback.

## Recovery contract

At a completed real optimizer boundary, save student parameters, algorithm
modules (including projectors and energy), all their optimizers, scheduler,
algorithm counters, rank-local Python/NumPy/Torch/CUDA RNG, and driver epoch,
consumed samples, global/update counters and collapse-gate state. Incomplete
gradient accumulation or a virtual meta step is never a committed boundary.
An algorithm with extra mutable Python state must implement
`training_state_dict` and `load_training_state_dict`; module and optimizer
state are discovered independently. Lagged self-distillation teacher resume
is explicitly unsupported until that teacher snapshot is included.

Each rank writes a unique transaction. The driver verifies file checksums,
writes its state and manifest, then atomically publishes `latest.json`.
Two complete recovery snapshots are retained; partial transactions and HF
exports are preserved. Resume verifies checksums, configuration, input hashes,
KDFlow source hashes, runtime versions and topology. It restores the sampler
cursor and explicitly sends restored student weights to new serving engines
before generation. Rollout requests have deterministic seeds indexed by train
seed, rollout step and request index; M selection is similarly stateless.
There is no promise of bitwise equivalence after changing runtime or hardware.
Timing logs remain attempt-local, and interruption may repeat work since the
last committed checkpoint; these are not new independent optimizer updates.

For `run_single_gpu.py`, `MP_RESUME=1` means restore the same output directory
and its latest complete training checkpoint. Missing/corrupt state or source,
data or optimization-config drift fails closed. `MP_PAUSE_AFTER_UPDATES=N`
cooperatively saves and stops without changing the target budget/scheduler.
The default launcher saves every 20 steps plus the explicit evaluation steps.

## Six-run manifest

| Variant | Train seeds, in order | Energy LR | Energy cadence | Student updates |
| --- | --- | --- | --- | --- |
| ALT-main | 42, then 43 in second round | 1e-3 | Every step | 312 |
| ALT-lowLR | 42, then 43 in second round | 1e-4 | Every step | 312 |
| ALT-every4 | 42, then 43 in second round | 1e-3 | 4,8,...,312 | 312 |

Execution order is main42, lowLR42, every4-42, main43, lowLR43, every4-43.
All start from the historical SFT initializer and the same trained energy
weights with fresh energy AdamW moments. Existing frozen runs are reused.
B=64, microbatch=4, accumulation=16; M=16, microbatch=4. Student AdamW uses
LR1e-6, betas(.9,.98), weight decay0, clip1, cosine-with-min-LR scheduler with
312-step horizon and .05 warmup fraction. Max span2, temperature1.
Remaining launch options come from the same pinned historical launcher recipe.

B is `qwen-author/data/prompts.parquet`; M uses teacher references in
`qwen-author/data/selected.parquet`. M must belong to the B corpus and excludes
the current B prompts at each step; roles can change across steps. M selection
is identical across variants for the same train seed and step. References
over the sequence limit are excluded from M with a logged count. These are
teacher references, potentially already used during SFT, not heldout labels.
The `uniform-normalized-group-original-pair-v3` policy groups by normalized
prompt key and deduplicates exact (rendered prompt, reference) pairs. Each step
samples 16 distinct eligible groups uniformly, then one original pair per group
with a separate stateless seed/step/group RNG. Repeated source rows do not give
a group extra weight; input row order does not affect sampling. Normalization
is only a conservative sampling/exclusion key, not a semantic-equivalence
claim: the selected original prompt always retains its own original reference.
Differing references are not assumed contradictory or semantically correct. The
policy and group digest are logged; older source checkpoints cannot silently
resume under this new policy. All six variants use the same policy.
The CPU audit checks nonempty references, membership, expected B cardinality
and normalized exact/containment overlaps with pinned evaluation prompts.
It reports and blocks detected overlaps without silently changing B; this is
not a semantic contamination audit.

## Queue and evaluation

### Two-host plan (current allocation)

Use `experiments/runai/queue_split_alternating.py submit --case CASE --manager MANAGER`
on each confirmed host, with the **same new shared CASE and source checkout**.
The previous single-host queue was not submitted; do not submit both plans.

* `hieplh8-beyond-leakage-1-0-0` GPU0: main42 → lowLR42 → every4-42 → eval.
* `nlp-core-team-0-0` GPU0/1/2: main43 / lowLR43 / every4-43 in parallel,
  each switching to eval after its training succeeds.
* New host GPU3/4: eval workers admitted only after external workloads release
  the GPUs. No process is killed to obtain capacity.

Each host has an audit and its own full resume qualification on GPU0 before
training. Ports and Ray directories are isolated for parallel training.
The manager has no allocation deadline, uses local `/var/tmp` SQLite and
host-local shared GPU leases. An incompatible existing manager is rejected,
not silently reconfigured. Submit is idempotent. Two/four CPU score workers
run on the owner/new host respectively.

HF exports receive an `export-ready/stepN.json` marker **after** the training
state transaction commits. Epoch-end saving never rewrites a published export.
Eval workers on either host claim committed checkpoints using filesystem locks;
generation and scoring overlap training. They share the historical 8 steps ×
4 benchmarks × 3 eval seeds protocol, 48 checkpoints and 576 cells total.
Training seeds remain a separate field from eval seeds.

Each operation preserves a `.done.json` or `.error.json` in `CASE/dispatch`.
Scorer retries remain bounded; an unresolved checkpoint operation is recorded
and skipped so other checkpoints proceed. Errors are not automatically cleared
or reported as zero. Once all producers terminate and work is drained, workers
exit; `report.json` is written only when all 48 scoring operations succeeded.
Missing checkpoints from failed training stay missing, not completed.
`queue_split_alternating.py status --case CASE` refreshes the current host and
shows the other host's timestamped exported state, completed evals and errors.
Remote state may be stale after pod loss; shared storage does not restart pods.

Additional supervision controls (frozen energy + SFT(M)) and random energy
initialization are proposals, not part of this submitted six-run plan.

The current CPU audit also runs the production meta renderer, length filter
and sampler using the local tokenizer before GPU qualification; at least 80
eligible prompt groups are required to leave M16 after excluding B64.
After the pre-training multi-reference failure, use the new source's
`retire-unstarted --case OLD_CASE` on each host, then submit a fresh shared CASE.
Retirement verifies receipt specs, refuses any started training or active
qualification, cancels only that host's old campaign jobs, and waits for them
to stop. It leaves the manager, source, datasets and failure evidence intact.

### Earlier single-host plan

`experiments/runai/queue_full_alternating.py submit --case CASE --manager MANAGER`
is restricted to the confirmed single-B200 pod. It reuses the existing local
manager when unambiguous, otherwise requires `--state`; a new manager uses
local `/var/tmp` SQLite, shared GPU leases and no allocation cutoff. Artifacts
are under shared CASE. Repeating submit is idempotent, not a failed-job reset.

The plan has 105 jobs: audit, GPU qualification, six sequential training jobs,
48 GPU generation jobs, 48 CPU scoring jobs, final report. All six training
runs finish before evaluation starts. Checkpoint steps are
40,80,120,156,200,240,280,312, each with eval seeds42/43/44 and historical
GSM8K, MATH500, MBPP, LCBv6 contracts. CPU scoring overlaps later generation.
LCBfix reuses generations and preserves its separate extraction contract.
`report.json` contains 576 original score cells and separate `train_seed`,
`eval_seed`, variant group and checkpoint-step fields. LCBfix results remain
in each checkpoint's `lcbfix/summary.json`; no failures become zero scores.

Generation and scoring journals are restartable. Scorer signal retries are
bounded and persisted by the existing resilient scorer, falling back to one
CPU worker on retry. Training retries at most twice for direct SIGKILL/SIGTERM
child exits and only with a complete checkpoint. Python/Ray errors, OOM,
failed qualification and exhausted retries stop that dependency chain and
retain evidence. Pod destruction still requires starting the manager on the
replacement pod; shared checkpoints alone do not restart a destroyed process.

Use `queue_full_alternating.py status --case CASE` to read the actual manager.
`queue-receipt.json` identifies its state and job IDs; use the manager's logs
command for the qualification or failed job. Submission is not GPU
qualification, completed training, scientific efficacy, or a W&B upload.
