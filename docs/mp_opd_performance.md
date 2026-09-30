# MP-OPD performance diagnostics

## Company OOM recovery

Alternating updates now resolve Ray/CPU inputs and transfer them per inner
callback, instead of staging the entire B64 (including teacher hidden states)
on GPU. Callbacks are replayed for the first- and second-order passes with the
same normalization and ordering. FULL_META_MEMORY JSON lines report padded
input/teacher-hidden shapes, allocation, reservation and cumulative peak at
forward/backward boundaries. They do not reset the step peak or synchronize
the GPU. This removes eager input staging; it does not establish that the
remaining higher-order attention graph fits a B200. GPU validation is pending.

Current pin: B64/micro1/accumulation64 and M16/meta-micro4. The queue forwards
these immutable configuration fields to the launcher.
Use a new campaign directory; do not resume an old micro4 campaign with this
configuration. The historical synthetic micro4 result below did not establish
capacity for real company rollouts: cd1334e failed in the second-order inner
gradient with 168.29 GiB allocated on real data, and a micro1 run then measured
148.45 GiB peak allocated / 172.53 GiB peak reserved on a 178.35 GiB B200.
A micro2/meta4 recipe was queued for one day and never completed a real-data
run: it is a memory mitigation without capacity evidence, not a verified result.

Production defaults remain full student, B64/micro4/accumulation16 and
M16/meta-micro4. Do not use synthetic capacity results to increase an existing
campaign's microbatch or resume it under a different accumulation contract.

### Microbatch admission today

`run_single_gpu.py` admits the recipes (4,4), (1,1) and (2,4) at any limit; every
other override needs a limit of 1-30. Only the exact (micro1, meta4) soft
alternating full contract prints ADMISSION_PASS. The campaign recipe itself lives
in `queue_full_alternating.configurations()` and `checked_config` rejects any
campaign whose stored runs differ from it, so changing the campaign microbatch
requires a new case, not a flag.

### Microbatch capacity accounting (2026-09-18)

Marginal cost of one more sample in the student microbatch: (168.29 - 148.45) / 3
= 6.61 GiB. A micro2 candidate therefore lands near 155 GiB allocated, and the
allocator holds ~24 GiB above its allocated peak (reserved 172.53 - allocated
148.45), so the reservation would need ~179 GiB on a 178.35 GiB card: short by
about 1 GiB plus fragmentation, not by tens of GiB. The peak is the second-order
graph, so it tracks the longest sample in the microbatch rather than the mean;
two long samples cost more than twice one. Synthetic micro4 (length 1024, no
teacher, no rollout, no partition DP, no company weights) passed at 127.52 GiB,
and the 40.8 GiB gap to the real micro4 failure is the real-data tax.

### Bounded microbatch diagnostic

`experiments/runai/microbatch_diagnostic.py` reuses the production env builder
(`queue_full_alternating.run_command`) and replaces only the microbatch pair, so
an arm runs under the same serving contract, offload flag and wrapper as the
campaign. One GPU at a time; stop on OOM and release the process.

    python3.12 experiments/runai/microbatch_diagnostic.py micro2 2 4 2 plain
    python3.12 experiments/runai/microbatch_diagnostic.py micro2-expandable 2 4 2 expandable

Each arm writes `<case>/diagnostics/<arm>-<stamp>/` with its launcher log,
launch-config.json and an operational-environment sidecar, and prints
DIAG_SUMMARY with the peak, the FULL_META_MEMORY trace count and the run summary.
The expandable arm sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, which
changes memory layout only; it is recorded in the launch manifest and in the
sidecar, never inferred.

A sequence cap needs more than `MP_MAX_LEN`: `init_args` raises `max_len` back to
`prompt_max_len + generate_max_len`, so the launcher rebinds `args.data.max_len`
afterwards and prints `EFFECTIVE_DATA_MAX_LEN`. The first attempt at this measured
nothing - an arm with the cap and an arm without it produced byte-identical step-1
metrics (peak allocated 123.788429 GiB, loss -0.162907, micro-batch `[2, 4096]` at
index 17, the same second-order OOM) because the manifest recorded 2048 while the
trainer ran 4096. Report the effective value, never the requested one.

## Equivalent preprocessing

Span tables retain the same prefix-sum/subtraction operations, executed per
span length rather than per cell. Energy features transfer static metadata in
bulk. Teacher normalization groups equal token counts to preserve scalar-divide
rounding on CUDA. Metric finiteness checks use one host synchronization on the
healthy path and still identify the offending key on failure.

Full-meta releases consumed gradient tuples before subsequent forwards and
optimizer-expression temporaries before hypergradient replay. Existing Adam
moments no longer allocate unused zero-tensor defaults. No update order, LR,
clipping rule, partition rule or trainable parameter set changes.

Measured on B200/Torch 2.11.0+cu130 in Modal app
`ap-wcspuKaPW9RBKpWHJAgunK`, five repetitions after two warmups:

| Operation | Atoms | Original ms | Optimized ms |
|---|---:|---:|---:|
| Energy features | 400 | 41.48 | 0.508 |
| Energy features | 1000 | 103.80 | 1.000 |
| Span tables (L2) | 400 | 37.88 | 0.259 |
| Span tables (L2) | 1000 | 93.66 | 0.255 |

These isolated kernels matched bitwise; feature-to-energy gradients also
matched. These factors must not be reported as end-to-end speedups.

Expected atom rates now vectorize across destination atoms while retaining
each atom's original accumulation order and scalar-promotion rounding.
B200 app `ap-2naRkcVUscLPCyDy8zacah` measured 19.67 -> 0.121 ms (400 atoms)
and 48.91 -> 0.123 ms (1000 atoms). Forward results matched bitwise;
mixed-precision first/second derivatives matched within rtol=1e-5, atol=1e-6.

## Short batch trials

1. Try one or two updates per candidate; stop on OOM and release its process.
2. Measure ten updates for feasible candidates; exclude the first two from
   steady-state statistics. Expand to thirty only if throughput is unstable.
3. Compare step wall time, tokens/second, peak allocated/reserved VRAM and
   actual rollout lengths. A fixed 1024-token probe does not qualify 4096 tokens.
4. Re-run pause/resume checks for the chosen microbatch before a new campaign.

`run_single_gpu.py` accepts `MP_MICRO_TRAIN_BATCH_SIZE` (divisor of 64) and
`MP_META_MICRO_BATCH_SIZE` (divisor of 16) for diagnostic limits 1-30 only.
The immutable campaign launcher strips inherited MP flags and retains its
original recipe. Diagnostic options are recorded in the launch manifest.

`experiments/modal/full_meta_capacity_modal.py` reuses the lhtu05 Gemma asset
volume and pinned runtime. It bounds one container to 1500 seconds and stops
on non-OOM errors. `--steps 10 --batch 64 --micros 4` measures synthetic
student/meta computation; it deliberately excludes teacher, on-policy rollout,
MP partition DP and company SFT weights. It is NOT a matched campaign test.
Results use a unique `--run-id` under `remote_artifacts/`.

## Full-size synthetic results (2026-09-14 UTC)

On lhtu05 B200, at length 1024 and meta-micro4, student micro64 exhausted
memory, micro32 failed cuBLAS allocation, and micro16/8 exhausted memory.
Micro4 passed. Releasing temporary gradients reduced peak allocated memory
from 147.24 to 127.43 GiB in the two-inner-microbatch capacity test; first-step
meta NLL, energy delta and both gradient norms matched the prior result.

App `ap-nMsJ4LCAgMivZGtkiX76sk` completed ten synthetic B64/M16 updates with
student/meta micro4: median of steps 3-10 was 20.122 seconds, range
20.064-20.269 seconds, peak allocated 127.517 GiB, peak reserved 160.279 GiB.
Energy gradients were nonzero on all ten steps. No efficacy inference follows
from synthetic NLL decreasing. This does not establish capacity for longer
actual rollouts, larger meta microbatches, or end-to-end campaign throughput.

Production remains micro4. Larger microbatches are not qualified by these
results. Company input/checkpoint and exact resume qualification remain
necessary before adopting a newly benchmarked batch configuration.
