# Phi/Gemma B200 diagnosis, 2026-10-10

Scope: independent sampling/parity probes followed by a two-update paired SimCT
trainer canary. This is not an accuracy experiment or reproduction of the company
Phi-warmup SFT checkpoint. All local model downloads/builds were avoided.

## Reproducibility

- Profile: `kieusontung6`; one B200 per attempt, no automatic retries.
- Runtime: `docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f`.
- Torch 2.11.0+cu130, CUDA 13.0, Transformers 5.6.0, SGLang 0.5.11.
- Student: base `google/gemma-2-2b-it`, revision `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8`.
- Teacher: `microsoft/Phi-4-mini-instruct`, revision `cfbefacb99257ffa30c83adab238a50856ac3083`.
- Sampling: temperature 0.6, top-p 0.95; probes ignore EOS to reach their requested lengths.
- Student default attention/sampling backends: `trtllm_mha` / `flashinfer`.
- Probe deployment: source `73d95802`, app `ap-SOsEf3IHf03tAmo23X1FoW`.
- Canary deployment: source `ca6e96d8`, app `ap-NchTyHXqReQuaqenEdhv5p`.
- Local evidence: `D:/dev/codex/research_vdt/remote_artifacts/phi-gemma-debug-20261010-r1/`
  and `D:/dev/codex/research_vdt/remote_artifacts/phi-gemma-canary-20261010-r2/`.
- Remote volumes use those run names. Retrieve only logs/JSON, not model weights.

## Verified independent probes

All comparisons below use exactly the sampled token IDs and next-token HF scoring.
Mean and p99 are absolute log-probability errors in nats.

| Probe | Response tokens | Mean vs HF eager at T=0.6 | p99 | Exact zeros |
|---|---:|---:|---:|---:|
| Gemma short | 256 | 0.0085136084 | 0.1500618398 | 37 |
| Gemma long | 7680 | 0.0107815762 | 0.1887366647 | 201 |
| Phi short | 128 | 0.0275010517 | 0.3751518735 | 31 |

All three pass mean <=0.1 and p99 <=0.5. Against HF SDPA (the trainer backend),
Phi p99 is 0.4855055958, close to the gate; 128 tokens do not qualify its tail
robustly. Gemma long has one >0.5 outlier out of 7680 tokens, max 0.5100145340;
this does not imply p99 >0.5. Full prompt+response lengths are 3896 and 3877,
so this run does not test the exact 4096 boundary.

### Exact zero is not proof of missing data

At every one of the 269 zero positions across these probes, independent HF
temperature-scaled logprob lies in [-1.1920928245e-7, 0]. This is consistent with
FP32 rounding for an almost certain token, not an unfilled sentinel. The historical
note `PHI_QWEN_SESSION_2026-10-04.md` section 2 infers an engine bug from zero
counts and sampling-vs-greedy behavior. That evidence alone cannot distinguish
missing data from temperature sharpening and numerical rounding. Historical
checkpoints/backends must be rescored before extending this finding to those runs.

Removing zeros still leaves Gemma-long mean 0.0110713337 and p99 0.1892066014.
Only 1.765% of its nonzero decode values are closer to raw HF than to tempered HF;
this probe does not reproduce the mixed raw/tempered decode defect.

### Prefill and decode use different probability scales

SGLang prefill scores match raw HF scores. Gemma-long prefill vs raw HF has mean
0.0092033896 and p99 0.1057193124. Comparing that same prefill with tempered HF
instead produces p99 0.5505894268. That is a scale mismatch in the comparison,
not evidence that decode/trainer parity fails. This does not prove a production
caller uses the wrong scale; callers must be checked separately.

## Harness failure and repair

R1 completed and saved all three probes, then failed on an unnecessary import
through the algorithms registry: `ModuleNotFoundError: xtoken_upstream_token_aligner`.
The original probe image omitted the vendor directory. It performed no training.
`2b1fb0eb` removed the unused import and added the mean gate after R1 had already
uploaded; R1 therefore still executed `73d95802`. The paired canary image includes
the vendor directory and its PYTHONPATH, plus the existing regression tests.

## Paired trainer canary

The canary consumes the saved probes and replays both mean and p99 gates before
starting. CPU preflight: 18 tests passed, 15 warnings, 55.15 seconds.

Settings: native tokenizers (`KDFLOW_TRUST_REMOTE_CODE=0`), exact rollout token IDs,
`span_ctkd` RKL with `span_score_mode=mean_logprob`, batch64/microbatch4, LR5e-7,
two-update stop, max response128/max sequence512. A fixture has 128 rows alternating
two math/code prompts. It checks trainer machinery rather than task learning.
The invocation records every deployed KDFlow Python source hash. The child process
is capped at 900 seconds; the Modal function is capped at 960 seconds.

Verified completion: downloaded `checkpoint/run-summary.json` has status completed,
optimizer_updates=2, session_completed_updates=2, stop_reason=null. Receipt exit0,
691.452437163 seconds including preflight, initialization, training and saving.
Both apps are stopped with zero active tasks, verified from Modal app list.

| Update | Loss | Final optimizer gradient norm | LR actually used | GH masked fraction |
|---|---:|---:|---:|---:|
| 1 | 1.250179 | 13.707727 | 0 (warmup) | 0.121101 |
| 2 | 1.326136 | 14.887672 | 5e-7 | 0.130637 |

No NaN, empty responses or collapse stop observed. Each update processed 8256
student loss tokens, including terminal positions. Span-segment ratios were
0.150840 and 0.151162; teacher candidate mass means 0.868552 and 0.857318.
Peak allocated GPU memory was 45.053638 GiB. These are two optimizer steps,
with only the second using a nonzero LR; loss movement is not efficacy evidence.
The fixture scheduler horizon is 4/warmup1, not the production 312/warmup16.

The deployed canary still prints the old inaccurate zero warning. After verifying
the probes, the local source corrects that warning and the matching explanatory
comments. Zero masking and DPCA fallback behavior remain unchanged. The wording
change was syntax/diff checked; it does not require another GPU experiment.

Reported billing at 2026-10-09 17:37:35 UTC: probe 0.70910521 USD, canary
1.52878112 USD, combined 2.23788633 USD. Earlier snapshots were incomplete;
these are reported usage charges, not a final invoice. Profile cycle Oct1-Nov1,
usage8.96382908, configured budget30/hard27/reserve1. Prelaunch guards passed;
each GPU attempt had estimate2. Dashboard budget controls were not verified.

Conclusion: independent parity and short paired trainer execution pass on these
base assets. No change to loss math, sampling distribution or parity thresholds
was needed for this canary. Historical Phi SFT failure and exact4096/long-training
behavior remain unqualified until their actual assets and failure logs are tested.

Launch from existing WSL Modal CLI (profile flag precedes the file argument):

```bash
export PHI_GEMMA_DEBUG_RUN=phi-gemma-canary-20261010-r2
uvx --offline --from modal modal run --profile kieusontung6 \
  /mnt/d/dev/dsh/research_vai/SimCT/experiments/modal/phi_gemma_debug_20261010.py \
  --stage canary
```

Existing log files refuse duplicate paid attempts. A rerun requires an inspected,
fresh run name. Do not re-run this exact completed or partially completed attempt.
