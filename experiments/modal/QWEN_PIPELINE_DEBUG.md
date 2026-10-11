# Qwen to company Gemma diagnostic

This workflow audits the existing company SFT checkpoint and public pinned Qwen
teacher without changing production algorithms or performing an optimizer update.
Use the existing Modal CLI installation and a profile whose refreshed billing
guard permits the bounded run. Do not install a local GPU environment.

```bash
UV_OFFLINE=1 uvx --offline --from modal modal run --profile kieusontung6 \
  experiments/modal/qwen_pipeline_debug.py --stage prepare
# Optional existing local prompt evidence; no model downloads through the laptop.
QWEN_DEBUG_CORPUS=/absolute/path/prompts.parquet \
  UV_OFFLINE=1 uvx --offline --from modal modal run --profile kieusontung6 \
  experiments/modal/qwen_pipeline_debug.py --stage audit
```

Stages, in dependency order:

- `prepare`: download the pinned Qwen teacher directly into its Modal volume,
  hash all files, compare native tokenizer backends and decoded shared candidates.
- `audit`: optionally compare all rendered prompts with the serialized backend.
- `mapping`: test normalized vocabulary key collisions across repeated calls.
- `probe`: hidden-head / direct SGLang / BF16 eager comparison on four existing
  real-company-SFT continuations. Its eager reference is provisional.
- `isolate`: independently compare TensorRT-LLM MHA and FlashInfer, BF16 eager
  and SDPA, and causal single-prefix scoring.
- `reference`: adjudicate the same exact token IDs with FP32 eager and BF16 SDPA.

Repeated stages refuse to overwrite logs. Inspect an attempt before selecting a
new `QWEN_DEBUG_RUN`. Every GPU function has a timeout, one container, no retries,
and shuts down only its own subprocess group. The prior warmup volume is read by
the worker but never written. Results remain in the run volume; create a local
destination directory before downloading results, and use WSL `/mnt/d/...` paths
for a WSL CLI.

The 2026-10-11 checks found a nondeterministic SimCT shared-space candidate caused
by replacing `Ġ` with `▁` before dictionary construction. A separate decoded
candidate audit found newline/tab mismatches. These findings concern
`SpanCrossTokenizerKD._find_overlap_tokens`; MP-OPD uses a different byte alignment
path. They are not a reproduction of historical accuracy or evidence that every
GRASS run is correct.

For Qwen, BF16 eager was a poor numerical oracle on these inputs: compare against
FP32 eager and SDPA before classifying an SGLang failure. The diagnostic leaves
the discrepancy in raw evidence rather than silently discarding failed comparisons.

Read-only report statistics audit:

```bash
python experiments/modal/audit_campaign_report.py report-payload.json evidence-dir
python -m unittest discover -s experiments/modal -p test_audit_campaign_report.py
```

It computes a macro for each common evaluation seed, then its standard deviation.
Training-seed conflicts inferred from run IDs require launch-config confirmation;
the audit never changes the original report. Partial cells are excluded and
small discrepancies due to historical rounded means are kept separate.

Actual diagnoses, hashes, receipts, and the requested checkpoint priorities live
outside Git in `D:/dev/dsh/research_vai/remote_artifacts/qwen_pipeline_modal_20261011/`.
Post-training checkpoint replay remains pending; warmup tests cannot validate an
RL-trained model's loop behavior.
