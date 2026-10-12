# Bounded checkpoint replay on Modal

This harness imports the two October 11 checkpoint bundles from immutable HF
revision `7a80c71eb4cf86a05ee63e5b64d5f3e960abcd03` into a remote Volume. It
reuses the pinned native B200 runtime, warmup weights and scorer assets. No
models or environments are downloaded onto the local machine.

Before GPU calls, use `modal-gpu-ops` billing guard for the selected profile.
The example profile is `kieusontung6`; CLI calls must specify it explicitly.
Use the already installed Modal CLI, not a fresh local dependency install.

```bash
modal run --profile kieusontung6 experiments/modal/checkpoint_replay_modal.py --stage prepare
modal run --profile kieusontung6 experiments/modal/checkpoint_replay_modal.py --stage replay --arm warmup
modal run --profile kieusontung6 experiments/modal/checkpoint_replay_modal.py --stage remaining
modal run --profile kieusontung6 experiments/modal/checkpoint_replay_modal.py --stage score
```

`CHECKPOINT_REPLAY_RUN` selects a fresh result Volume/app label. A stage refuses
to overwrite an existing log. Import can reuse completed per-checkpoint
receipts; stored nested ZIP downloads resume after transport errors using
bounded HTTP ranges, then validate the entire member hash/CRC. Only verified checkpoints
are eligible for replay. `remaining` runs these eight arms serially on B200:

- GRASS rl2/rl3, seed42, step312.
- SimCT seed44/seed42, step40.
- Temporal/native rl2, seed44, step80/step160.

Each nested ZIP is checked against its supplied SHA256 sidecar and ZIP CRC.
Imported weights/config/tokenizer members receive separate hashes and CRC
checks. The full outer archive hash is **not** recomputed during selective
range import. Path traversal is rejected. Optimizer pickle files are never
loaded. These packages lack launch config/optimizer state, so names alone
do not independently verify the training recipe or checkpoint step.

The matched screen comprises eight code fixtures, eight elementary math
fixtures, six historical failure prompts, and sixteen pinned held-out math
items. Generation uses temperature 0.6, top-p 0.95, top-k disabled, seed42,
response cap4096 and native Gemma user-only prompt rendering. Raw token IDs,
text, stop reason and logprob are retained. This is a diagnostic screen, not
the original full benchmark contract or an accuracy estimate.

Independent Transformers BF16/SDPA scores use exactly the same sampled token
IDs. Shared frozen code/math prefixes test policy changes across weights.
Two greedy requests compare SGLang with Transformers directly. The initial
warmup attempt used two code requests capped at256; RL arms use one code and
one historical math request capped at512. Their greedy counts should not be
treated as a matched cross-checkpoint metric. Sampled generation remains
matched across all arms.

Math regrading runs on CPU with the cached isolated math-verify environment;
per-answer symbolic work is bounded. Generated code runs only in CPU Modal
sandboxes with no network, credentials, mounted Volumes or identity token:

```bash
modal run --profile kieusontung6 experiments/modal/checkpoint_replay_semantics.py --input-root /path/to/downloaded/results
```

Code is graded twice on the same raw output: once with the exact author
extractor pattern (recognizes `python` or an untagged fence), once with a
diagnostic extractor accepting Python version tags such as `python3.8`.
This changes extraction only; generated code and assertions remain identical.
The latter is a counterfactual diagnostic, not an amendment to paper scores.

After all assets are ready, `--stage deep` compares warmup/rl2/rl3 on the same
RL3-generated prefixes at response positions128 and768, including 64-token
target windows and independent greedy continuations. This checks parity
inside actual loops rather than only at their initial healthy tokens.
`--stage deep-temporal` repeats this for temporal/native step160 and warmup
using temporal-generated prefixes. `--stage deep-all` runs both groups serially.
Symbolic verifier timeouts are marked unknown rather than model failures.

Reusable containers reload both asset/result Volumes before each stage to
avoid stale mount snapshots during concurrent CPU import. Use a new
`--attempt retry1` log suffix when retrying a failed stage; do not overwrite
the failure evidence. `--stage resume` skips the completed warmup/rl2/rl3
screen and continues the other six checkpoints.

Download only the result Volume for local analysis. Keep model assets on
Modal. A failure reproduced in both inference backends supports a learned
policy defect; it does not identify which training update caused it.
Inference parity and fixture tests do not qualify a full training run.

## Flat TRUST/ALIGN archive

For `degen-round1-all-20261011.zip`, select importer
`CHECKPOINT_REPLAY_PREPARE=degen_checkpoint_assets.py`, a fresh
`CHECKPOINT_REPLAY_ASSETS` Volume and `CHECKPOINT_REPLAY_RUN` result Volume.
The importer pins HF revision `010c9a418f167c4bae6fb7ee07fa838fc7583bb8`,
validates complete per-checkpoint member sets and transfer CRC, and records
SHA256 for every member. This flat archive has no member SHA sidecars, and
selective import does not recompute its outer LFS hash. It lacks launch config
and optimizer state; names do not alone verify training provenance.

Use `--stage batch --arm trust-r-step160,trust-r-step240,trust-r-step312`
(and the corresponding TRUST-B/ALIGN aliases) to screen explicit checkpoints.
`--stage deep-prefix --arm trust-r-step160,trust-r-step312,warmup
--probe-arm trust-r-step312` compares policies on the same source-policy loop
prefixes. Choose the probe only after confirming capped outputs exist.
No production algorithm, evaluator or runtime dependency changes are needed.

After CPU math grading, deep probes also score three common healthy SFT answer
endpoints (apples/speed/age). Only correctly graded, EOS-terminated warmup outputs
are admitted; terminal control tokens are removed from their copied prefixes.
Each policy's raw EOS/EOT mass is recorded at the identical completed answer.
These endpoint probabilities are HF/SDPA-only conditional probes, not a matched
two-backend generation result or the policy's own stopping frequency.
