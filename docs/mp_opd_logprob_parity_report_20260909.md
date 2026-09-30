# MP-OPD behavior/trainer log-probability parity investigation

Date: 2026-09-09 (Asia/Ho_Chi_Minh)

## Conclusion

The dominant MP-OPD parity failure was a probability-semantics bug in KDFlow's
guard. SGLang 0.5.11 output-token logprobs from sampling at temperature
`T = 0.6`, while KDFlow compared them with `log_softmax(student_logits)` at
`T = 1`. The comparison must instead use:

\[
\log \pi_T(y_t \mid y_{<t}, x)
= \log \operatorname{softmax}(z_t / T)_{y_t}.
\]

The experiments also show that SGLang input-token/prefill logprobs are raw
model logprobs by default. Changing the request temperature from `0.6` to `1.0`
changed none of the 1,800 prefill values on either B200 attention backend. This
does not contradict the output-token result: the two SGLang paths expose
different logprob semantics.

After matching temperature, the remaining differences are numerical backend
and precision effects. Their distribution is small enough that a single
maximum-error threshold is brittle. KDFlow now fails on distributional drift
(`mean > 0.1` or `p99 > 0.5`) while retaining max and outlier-rate telemetry.
NaN remains an intentional mask only for synthetic terminal events; infinity
is fatal.

## Original failure and hypotheses

The RunAI Qwen-teacher/Gemma-student canary originally stopped with examples
such as:

```text
behavior/trainer logprob parity failed: mean=0.061313, max=0.828797
behavior/trainer logprob parity failed: mean=0.069395, max=0.995391
```

Replaying two samples showed that dividing HF logits by `0.6` reduced mean
absolute error from approximately `0.07` to `0.009`. A later 64-sample replay
over 30,652 tokens gave mean `0.006778`, p99 `0.1283`, p999 `0.2834`, max
`1.6633`, and only one token above `0.5`. Full-context, prefix-only, BF16, and
FP32 checks did not explain that isolated maximum.

The Modal investigation tested four hypotheses:

1. The mismatch is specific to B200 or its default attention backend.
2. Batched prefill scoring is misaligned with individual scoring.
3. Output and input logprobs use different temperature semantics.
4. SGLang's `rl_on_policy_target="fsdp"` removes the residual mismatch.

## Reproducible setup

The probe reused the pinned image rather than rebuilding the runtime:

```text
docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f
```

Runtime:

- Python 3.12.12
- Torch 2.11.0+cu130
- CUDA 13.0
- Transformers 5.6.0
- SGLang 0.5.11
- `google/gemma-2-2b-it` revision
  `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`
- two prompts, one math and one Python task
- 900 forced output tokens per prompt, 1,800 response tokens total
- temperature `0.6`, top-p `0.95`, `SGLANG_RETURN_ORIGINAL_LOGPROB` unset
- the same fixed token trajectory was rescored by every backend

The probe source is in
[`experiments/modal/mp_opd_parity_worker.py`](../experiments/modal/mp_opd_parity_worker.py)
and [`experiments/modal/mp_opd_parity_modal.py`](../experiments/modal/mp_opd_parity_modal.py).

## Results

### A10 control

The inexpensive A10 run showed that the mismatch was not B200-specific.

| Comparison over 1,800 tokens | Mean abs. | p99 | Max | Count > 0.5 |
|---|---:|---:|---:|---:|
| SGLang decode vs HF eager, both at `T=0.6` | 0.01692 | 0.22868 | 0.37762 | 0 |
| SGLang prefill vs HF eager at `T=0.6` | 0.09877 | 0.65847 | 1.31913 | 38 |
| SGLang decode vs its batched prefill | 0.09899 | 0.60671 | 1.06571 | 35 |
| SGLang batched vs individual prefill | 0.00778 | 0.07060 | 0.11681 | 0 |
| HF eager vs HF SDPA at `T=0.6` | 0.01299 | 0.18747 | 0.36280 | 0 |

The batch-vs-individual difference is much smaller than the
decode-vs-prefill difference, so batching is not the primary cause.

### B200 temperature and backend controls

The B200 run generated once with the default backend, then rescored the same
trajectory with both the default `trtllm_mha` backend and Triton.

| Comparison over 1,800 tokens | Mean abs. | p99 | Max | Count > 0.5 |
|---|---:|---:|---:|---:|
| Default decode vs HF eager at `T=0.6` | 0.01907 | 0.23348 | 0.36290 | 0 |
| Default decode vs HF eager raw (`T=1`) | 0.11103 | 0.68899 | 1.11737 | 40 |
| Default prefill vs HF eager raw | 0.01566 | 0.12616 | 0.23281 | 0 |
| Default prefill vs HF eager at `T=0.6` | 0.11069 | 0.70220 | 1.15070 | 45 |
| Triton prefill vs HF eager raw | 0.01519 | 0.11827 | 0.18330 | 0 |
| Triton prefill vs HF eager at `T=0.6` | 0.11060 | 0.68718 | 1.10777 | 49 |

For both B200 backends:

```text
prefill(T=0.6) vs prefill(T=1.0): mean=0, p99=0, max=0, 1800/1800 exact
batched prefill vs individual prefill: mean=0, p99=0, max=0
```

Cross-backend prefill difference was mean `0.01000`, max `0.17335`, with 11
of 1,800 tokens above `0.1`. Both backends exhibit the same raw-vs-policy
semantics. The bug is therefore not specific to FlashInfer, TRT-LLM MHA,
Triton, or B200.

### SGLang FSDP on-policy control

An additional A10 run set `rl_on_policy_target="fsdp"`. SGLang enabled
deterministic inference and changed its sampling backend from FlashInfer to
PyTorch. Decode-vs-HF policy error was mean `0.01736`, p99 `0.20654`, max
`0.51134`. This did not materially improve aggregate parity over the standard
A10 result (mean `0.01692`), while it changed execution semantics. It is not
enabled by the KDFlow fix.

The two A10 runs sampled different trajectories, so the FSDP comparison is an
aggregate control, not a token-for-token performance comparison.

## Source-level explanation

SGLang 0.5.11's standard sampling path divides output logits by the request
temperature before softmax and returns the selected log probability from that
distribution. Its prefill logprob helper applies temperature only when
`temp_scaled_logprobs` is enabled. A request initializes this field to `False`.
The observed exact invariance of prefill scores to request temperature matches
that implementation:

- [SGLang v0.5.11 sampler output-logprob path](https://github.com/sgl-project/sglang/blob/v0.5.11/python/sglang/srt/layers/sampler.py#L106-L175)
- [SGLang v0.5.11 input-logprob normalization](https://github.com/sgl-project/sglang/blob/v0.5.11/python/sglang/srt/layers/utils/logprob.py#L32-L63)
- [SGLang v0.5.11 request defaults](https://github.com/sgl-project/sglang/blob/v0.5.11/python/sglang/srt/managers/schedule_batch.py#L764-L771)

The production trajectory stores SGLang output-token logprobs. KDFlow was
computing the comparison value with raw `log_softmax(student_logits)`. The
correct value is `log_softmax(student_logits / rollout_temperature)`.

## Implemented fix

[`kdflow/algorithms/mp_opd.py`](../kdflow/algorithms/mp_opd.py) now:

1. validates that rollout temperature is positive and finite;
2. computes trainer policy logprobs at the rollout temperature;
3. masks only NaN behavior values used for synthetic terminal events;
4. rejects behavior infinities and non-finite deltas;
5. logs mean, p99, max, and fraction above `0.5`;
6. aborts on distributional mismatch (`mean > 0.1` or `p99 > 0.5`) instead of
   one isolated finite maximum.

The final broader regression run also exposed an older exact-trajectory fixture
failure introduced by the Gemma terminal-control change. The trainer now checks
for MP-OPD arguments defensively, preserving the generic exact-trajectory path
when a minimal caller has no `args` attribute.

This retains a fail-closed check for systemic errors. The prior RunAI replay
would pass because mean `0.006778` and p99 `0.1283` are small, while its single
`1.6633` maximum remains visible in telemetry. A raw-vs-policy mismatch like
the B200 prefill comparison would still fail because mean is about `0.111` and
p99 is about `0.70`.

## Validation

- Modal CPU, pinned runtime: `37 passed, 1 skipped` in `28.91s` for the focused
  `pytest -q /opt/overlay/tests/mp_opd` run.
- Modal CPU, pinned runtime: `55 passed, 1 skipped` in `43.63s` for `tests/mp_opd`
  plus paper-score, SpanCTKD-metric, trajectory, and exact-trajectory tests.
- Local static checks: Python compilation and `git diff --check` passed.
- B200 probe: completed with exit code 0 and emitted the final result marker.
- A10 standard and FSDP probes: completed with exit code 0 and emitted final
  result markers.
- SGLang emitted a Python `resource_tracker` cleanup KeyError after final
  metrics. The Modal entrypoints still returned success; this is a cleanup
  caveat and does not invalidate the saved results.

## Cost accounting

Profile: `phamvanvuhoan`. Billing period was treated as the default calendar
month, 2026-09-01 through 2026-10-01. Before the B200 run, the refreshed Modal
report showed `$22.3938` used, `$6.6062` remaining to the configured `$29` hard
limit, and `$5.6062` remaining after a separate `$1` reserve.

The final billing snapshot at 2026-09-09 01:19 Asia/Ho_Chi_Minh had posted:

| Work | Cost posted |
|---|---:|
| Asset preparation | $0.00379 |
| Two failed cheap harness checks | $0.04978 |
| Successful A10 baseline | $0.10722 |
| Successful B200 default + Triton probe | $0.67314 |
| Successful A10 FSDP control | $0.09945 |
| Two posted CPU regression runs | $0.00426 |
| Total posted for all task app IDs | **$0.93765** |

The workspace total at that snapshot was `$23.17067`. Billing can still be
adjusted after credits or reservations, so `$0.93765` is the posted resource
cost for the recorded app IDs rather than a final invoice amount. The failed
broader CPU test and its successful rerun had not appeared in the latest
snapshot; they used CPU only and will add a small pending amount.

The B200 app itself cost `$0.67314`, below its approximately `$1.15` timeout
ceiling. Using A10 for hypothesis elimination avoided additional B200 trials.

## Evidence and reproduction

Local evidence is intentionally kept outside Git history under:

```text
/home/tung/simct-b200-portable/remote_artifacts/modal_mp_opd_parity_20260909_r1
```

Important files:

- `a10-default-r3/a10-default-r3/result.json`
- `b200-download/b200-default-triton-r1/result.json`
- `a10-fsdp-download/a10-fsdp-r1/result.json`
- `b200-default-triton-r1.log`
- `mp-opd-tests-r2.log`
- `mp-opd-tests-final-rerun.log`

The probes can be reproduced after selecting a funded Modal profile and passing
the billing guard:

```bash
modal run experiments/modal/mp_opd_parity_modal.py \
  --stage probe --run-id <unique-id> --gpu A10 \
  --backends default

modal run experiments/modal/mp_opd_parity_modal.py \
  --stage probe --run-id <unique-id> --gpu B200 \
  --backends default,triton
```

Each run refuses to overwrite an existing result and records its image digest,
model revision, command, GPU, library versions, server settings, and full token
arrays.

## Limits and next decision

The Modal model is the pinned public base `google/gemma-2-2b-it`, while the
RunAI failure used the project SFT checkpoint with a Qwen teacher. The Modal
experiment isolates the student rollout/trainer probability contract; it does
not reproduce MP-OPD training quality or cross-tokenizer teacher scoring.

The next production action is to apply commits `7b29b14`, `3a42b9e`, and
`e580893` to the RunAI source overlay, then run a five-update canary for atomic
and fixed mode.
Acceptance criteria are:

- at least one optimizer update completes;
- finite loss and gradients;
- no infinity/non-finite parity failure;
- `trajectory_logprob_abs_mean < 0.1`;
- `trajectory_logprob_abs_p99 < 0.5`;
- max and outlier fraction retained for post-run analysis.

A full two-epoch run should start only after both five-update canaries satisfy
those criteria. No further Modal GPU experiment is needed for this parity bug.
