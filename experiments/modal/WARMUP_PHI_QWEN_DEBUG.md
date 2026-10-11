# Actual company SFT checkpoint diagnostics

This harness diagnoses the two final checkpoints in
`ihbkaiser/qwen25vl-3b-msd-checkpoints/sft-warmup-both-20261009.zip`, pinned to
revision `d77f01ffaebec859a65041a52dea7a5d11d79312`. It does not train a model or
claim benchmark accuracy. The sixteen common prompts are small diagnostic
fixtures, including the historical MBPP remove-occurrence user prompt.

## Assets and execution

The ZIP is 38,656,399,804 bytes. `warmup_zip_access.py` reads its central
directory with HTTP ranges, then streams only final checkpoint members on
Modal. Both model weights must match the historical launch SHA-256 values.
Selected members must pass ZIP CRC; the entire ZIP hash is **not** recomputed.
No model weights are downloaded locally.

Runtime is pinned by registry digest in `warmup_phi_qwen_debug.py`. The
`kieusontung6` profile already contains the separately pinned Phi teacher in
volume `phi-gemma-debug-20261010-r1`. A different profile needs its own qualified
teacher assets; changing a CLI profile alone does not transfer volumes. Keep
profile-specific billing guards outside the harness, before each paid stage.

Using the existing offline WSL Modal CLI, the entrypoint stages are:

- `--stage prepare`: CPU import into `phi-qwen-warmup-assets-20261011-v1`.
- `--stage diagnose`: B200 paired Phi/Qwen-SFT generation, independent eager
  student scoring, and Phi teacher hidden/head versus direct SGLang scoring.
- `--stage credit`: B200 frozen-prefix atomization and production GRASS-Chunk
  run2 loss gradients, using the qualified teacher scores from `diagnose`.
- `--stage teacher-baseline`: B200 native Phi teacher generation on the same
  fixtures; validates the cached teacher revision from `assets.json`.
- `--stage teacher-readout`: CPU token/text attribution of qualified teacher scores.
- `--stage teacher-tokenization`: initial CPU token path comparison. For finished
  controls use the corrected stripping in `tokenizer-audit`; the initial artifact
  records that native decoded text can retain a finish token.
- `--stage tokenizer-audit`: CPU comparison of AutoTokenizer's loaded backend
  against the pinned `tokenizer.json`, explicit finished-control stripping, and
  `PreTrainedTokenizerFast` as a candidate loader.
- `--stage token-path-scores`: B200 paired scores on identical text. Native
  AutoTokenizer prompt IDs remain fixed; only response tokenization changes.
- `--stage tokenizer-regression`: CPU encode/decode, chat-template, control-ID,
  and explicit `use_fast=True` regression qualification. Production loaders
  are not changed by this diagnostic harness.

Set `WARMUP_DEBUG_RUN` consistently for the stages. Existing worker logs cause
the harness to refuse an automatic duplicate attempt. Inspect the failure
before selecting a different label. Functions have internal wall-time limits,
Modal timeouts, and no automatic retries; any timeout cleanup addresses only
the exact child process group created by the wrapper.

## Results and interpretation

Results go to volume `WARMUP_DEBUG_RUN`; model weights stay in the assets
volume. Pre-create the local download directory, then download known result
files individually with `modal volume get ... --profile kieusontung6`.

`warmup_debug_semantics.py --input-root <downloaded-results>` runs code fixture
assertions in CPU-only Modal Sandboxes, one isolated sandbox per response,
with no secrets or mounted volumes and with networking blocked. Add
`--arm phi_teacher` for teacher-only assertions after downloading its output.
This is independent of AST validity. The math verifier is the existing SFT
data-selection heuristic and can reject a numerically correct free-form
response; inspect raw text instead of calling this metric benchmark accuracy.

The frozen credit probe directly calls the current atomizer and GRASS-Chunk
loss with exact student logits/head inputs. Its GRASS noise EMA starts fresh;
four short prefixes do not reconstruct a batch64 update or the historical
trajectory. A negative stop-logit gradient indicates the local descent
direction of that logit, not the effect of an optimizer update on every model
parameter. The teacher probe qualifies the selected prefixes/API/backend,
not Ray offload or arbitrary 4096-token inputs on a company node.

## Verification

Offline extraction tests:

```powershell
py -3.12 -m unittest discover -s experiments/modal -p test_warmup_zip_access.py -v
```

Preserve raw JSON, complete Modal logs, image digest, receipts, source hashes,
and app IDs beside the analysis in the user-selected persistent evidence
directory. Evidence and checkpoints are not source files and must not be
committed to the source branch.
