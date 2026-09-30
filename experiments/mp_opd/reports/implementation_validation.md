# MP-OPD v0 implementation validation

Date: 2026-09-04
Epistemic status: `implementation_validation` plus one `oracle_diagnostic`

## Outcome

MP-OPD is implemented as `kd_algorithm=mp_opd` with selectable `atomic`,
`fixed`, `random`, `oracle`, and `soft` modes. The development worktree is only
an isolation mechanism; the runtime interface does not require branch changes.

The implementation corrects the proposal's unit mismatch. Because `h_i` is an
atom NLL **sum**, span loss is `r_c * sum_i h_i`, not a second token-count
weighted sum. Oracle utility is `-r_c * sum_i z_i`, also without an extra
`w_i`.

## Implemented components

- fail-closed cumulative UTF-8 byte atomizer with explicit EOS masking and
  boundary-safe handling of transient partial-codepoint replacement prefixes;
- detached realized-path teacher/student credit and corrected span loss;
- deterministic atomic/fixed/random full-cover partitions;
- exact float32 `O(nL)` semi-Markov forward/backward and marginals;
- exact prefix-sum utility table and hard max-sum oracle;
- mutation-free adapter-subspace virtual-SGD helpers;
- detached-feature two-layer BiGRU energy model;
- independent energy optimizer/checkpoint/config-hash contract;
- exact first-order energy surrogate;
- KDFlow training-step integration and finite W&B-safe metrics;
- deterministic toy oracle runner and machine-readable result;
- restricted-network B200 installation guide with explicit local model paths;
- bounded 100-update B200 launcher with an exact optimizer-step equation.

## Verification executed

### MP-OPD targeted suite

```text
python -m pytest tests/mp_opd -q
30 passed
```

Coverage includes byte equality and failure modes; real Llama-3.2/Qwen2.5
tokenizer alignment; brute-force partition comparison through `n=8`; marginal
coverage; normalization; signed-credit conservation; L=1 loss/gradient
identity; constant-rate merge invariance; hard-DP argmax; explicit autograd
utility; centered finite difference; detach boundaries; mixed precision;
deterministic random partition; energy checkpoint round trip; shifted labels;
finite atomic backward; zero-gradient exclusion for an invalid size-one
microbatch; JSON-safe metrics; and fail-closed oracle/soft modes.

### Baseline regressions and full repository suite

```text
python -m pytest tests/test_span_ctkd_metrics.py tests/test_xtoken_algorithm.py -q
7 passed, 1 non-failing PyTorch sparse-invariant warning

python -m pytest -q
136 passed, 14 subtests passed, 1 non-failing warning
```

`KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT=1` was used only to avoid importing GPU
runtime dependencies during CPU tests. Default registry behavior is unchanged.
Torch Dynamo was disabled for the baseline CPU test because WSL `/tmp` had no
free space; this changes compilation behavior only, not the tested loss math.

Additional gates:

```text
git diff --check                 PASS
bash -n run_b200_100step.sh      PASS
python -m py_compile ...         PASS
```

## Toy oracle diagnostic

Artifact: `experiments/mp_opd/results/toy_oracle_v0.json`

| Metric | Value |
|---|---:|
| enumerated partitions | 29 |
| atomic partition | six length-one spans |
| oracle partition | `[0,4), [4,6)` |
| predicted oracle advantage | 0.0015626213 |
| actual oracle advantage | 0.0018193321 |
| sign agreement | true |
| Spearman predicted vs actual | 0.9615764 |
| semi-Markov coverage max error | 1.1921e-7 |
| finite-difference directional error | 6.1215e-7 |
| real parameter unchanged | true |

This deliberately constructed analytic adapter fixture has positive local
oracle headroom. It validates gate mechanics and sign conventions only. It is
not evidence that LLM partitions improve validation loss.

## Known limitations and remote decision

The current KDFlow rollout path has no independent stable-ID meta batch M.
Consequently, real-data bilevel `soft` training is not yet end-to-end. `soft`
requires an audited energy checkpoint; random initialization is rejected.
`oracle` requires detached instrumentation scores and cannot silently use
validation/test labels for training.

A 100-update B200 run is therefore scientifically honest only as an `atomic`
systems canary (or later fixed/random control), labelled
`implementation_validation`. It can verify atomization, scoring, backward,
optimizer accounting, telemetry and W&B logging. It cannot validate learned
partition efficacy. Full energy training remains blocked on the real B/M data
loader and stable record IDs, not on the local math/DP implementation.

The unique Modal B200 launcher is staged at
`experiments/modal/mp_opd_b200_100step_modal.py`. It repeats the 640 pinned
unique prompts ten times in a manifest-preserving 6,400-row view, so one epoch
with rollout/train batch 64 yields exactly 100 optimizer updates and only one
final checkpoint. It runs the CUDA-13 B200 gate and MP-OPD tests before
training, has `retries=0`, and records full logs plus W&B identity. At the time
of staging, the Modal budget guard was `WARN` ($11.67103215 reported current,
$15 conservative estimate and $1 reserve; $27.67103215 projected against the
$23.60 warning threshold and $28.025 hard limit). The user explicitly approved
that warning. Attempt r1 then stopped at its pre-training source-integrity gate
because three hashes predated final whitespace cleanup; it performed zero
optimizer updates. The diagnosed r2 launcher pins the committed hashes and a
new app, output-volume, run and W&B identity. It retains `retries=0`.

Attempt r2 passed source integrity but stopped before training when
Transformers 5.6 `AutoTokenizer` entered the Llama `AutoConfig` path and the
`kernels` integration rejected a missing revision/version. Attempt r3 selects
the Llama tokenizer type explicitly and runs tokenizer/data plus MP-OPD
regression checks in a CPU-only Modal preflight before allocating B200. The
scientific training contract is unchanged and r3 still has `retries=0`.

Attempt r4 passed the B200 hardware/runtime gate but stopped before an optimizer
update because FlashAttention 4 does not expose the legacy
`flash_attn.bert_padding` module. Commit `96b9244` replaced only those padding
helpers with value/gradient-preserving PyTorch operations; the attention kernel
and scientific configuration were unchanged.

Attempt r5 then passed the CPU suite and B200 gate, initialized rollout plus the
Qwen3-4B teacher, generated the first 64 student responses, completed all eight
teacher-forward chunks and reached the first student microbatch. It stopped
before optimizer update 1 because the atomizer rejected an intermediate
U+FFFD produced by a partial UTF-8 tokenizer prefix. A CPU diagnostic replay of
the exact 64 saved responses found 46 transient-prefix rejections and 18 valid
samples. After filtering transient prefixes only, 62/64 responses atomized
losslessly. The remaining two are genuine teacher-tokenizer normalization
mismatches: student decode equals the source bytes while teacher decode differs,
so they remain fail-closed and contribute zero gradient with explicit metrics.

Post-fix validation on the pinned Modal CPU image reports:

```text
tests/mp_opd + SpanCTKD/X-Token regressions + ring-attention compatibility
37 passed, 1 skipped, 15 non-failing upstream warnings
exact r5 rollout replay: 62 valid, 2 normalization_mismatch
```
