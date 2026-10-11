# Run readiness after the Phi / Qwen pipeline audit

Use committed source on `vdt/ops/b200-portable`, the existing B200 runtime,
batch 64 / microbatch 4, and a fresh output directory. A runnable canary is not
an accuracy result or qualification of a complete 312-update trajectory.

Qualification completed on2026-10-11:160 CPU regression tests passed (2 CUDA
tests skipped on CPU);9 launcher/cache checks passed; actual B200 Phi and Qwen
probes each passed four frozen cases through the production SGLang subprocess,
ZMQ hidden-state transport, GRASS run2 loss and SimCT loss gradients. Ray2.58
also executed a real CPU task with persistent session files and local sockets.
This does not test a full batch64 optimizer/rollout sequence or long4096 parity.

| Run | Action before launching | Runtime rebuild? | Priority |
|---|---|---|---|
| Phi→Gemma MP-Atomic + GRASS run2, seed42 | Pull fixes; tokenizer preflight; restart RL from the existing Phi SFT; two-update canary, then evaluate step40 | No | P0 |
| Qwen→Gemma SimCT, seeds42/44 | Pull fixes; record semantic mapping digest; fresh RL from the same Qwen SFT; canary, then evaluate step40 | No | P0 |
| Qwen→Gemma GRASS run2/3/4 and temporal run2/3 | Existing modes; explicitly set source=run and length; temporal start20/end160; retain matched seeds/initialization | No | P1 |
| Qwen→Gemma DPCA / MP-Atomic / corrected GRASS-DP / ALIGN / TRUST | Existing source modes and assets; ordinary node preflight/canary; no new method tuning in this patch | No identified dependency change | P1 controls |
| Replay rl3-seed42-step312 vs rl2-seed42-step312 degeneration | Upload the actual RL checkpoints and manifests first | No identified dependency change | P0 once weights arrive |
| Replay historical SimCT seed44-step40 vs seed42-step40 | Upload those RL checkpoints; historical mapping was not saved, so old coordinate order cannot be reconstructed with certainty | No identified dependency change | P0 once weights arrive |
| GRASS source=xtoken | Include `experiments/modal/vendor` on PYTHONPATH; native chunking does not require a distribution projection | No | Secondary ablation |
| `kd_algorithm=xtoken` | Requires the audited distribution projection and SHA in addition to the vendor module | Asset/support qualification first | Blocked until that gate passes |
| Full alternating soft / host-mask / resume experiments | Keep their separate qualification gates; these frozen pipeline probes do not validate them | Diagnose their own failures | Not admitted by this handoff |

## What changed

1. Both production text tokenizer loaders call one audited loader. Only a GPT2
   tokenizer with a custom serialized Split + ByteLevel pipeline that was replaced
   by its constructor receives the serialized fallback. The fallback validates
   model, normalizer, pre-tokenizer, post-processor, decoder, vocabulary IDs,
   EOS/BOS/PAD, chat template, and fixtures with/without special tokens. Qwen's
   verified AutoTokenizer path is retained. Tokenizer hash/policy are logged.
2. SimCT matches stable emitted UTF-8 strings rather than raw vocabulary spellings.
   Newline/tab/control-character spellings cannot match literal accented symbols.
   Duplicate emissions have a deterministic representative; ambiguous partial
   Unicode/context-dependent emissions are excluded. EOS has an explicit role
   pair. A mapping digest is logged. This changes SimCT's candidate space and
   requires a fresh run; it is not a full cross-tokenizer distribution projection.
3. Every text-only teacher forward uses the input IDs defining its loss mask,
   including legacy SimCT. SGLang cannot re-tokenize these into a different path
   or insert the legacy extra space before EOS. Multimodal legacy behavior stays
   separate and is not qualified here.

For Qwen→Gemma, semantic matching yields97542 content pairs plus EOS, versus
64272 historical raw-key overlaps before terminal handling. For Phi→Gemma it
yields122964 content pairs plus EOS. This corrects the support definition as
well as bad symbol pairs; historical and corrected SimCT are different objectives.
Record the mapping digest and compare fresh matched runs before claiming recovery.

Do not resume an already damaged Phi RL run to test this fix. The audited SFT
cache was valid; these findings do not justify repeating warmup or rebuilding
the whole dataset pipeline. Data-quality improvements are separate experiments.
Do not change Qwen production SGLang to BF16 HF eager: the earlier eager numerical
reference was the source of large scoring discrepancies. Use SDPA or FP32 eager
for independent **Qwen** reference scoring; this is not a Gemma attention change.

## Node checks and commands

The known handoff target is `embed-8b-training-v2-0-1`, GPU3. The node itself has
not been contacted by this qualification. Run from your newly pulled source
checkout. These blocks are manually executed by the user; no company job was
launched by the audit.

**F1 — Qwen preflight**, read-only, before reserving a training run:

```bash
(
set -eu
export PYTHONPATH="$PWD/experiments/modal/vendor:$PWD"
export CUDA_VISIBLE_DEVICES=3
export KDFLOW_TRUST_REMOTE_CODE=0
bash experiments/runai/python-b200-host.sh experiments/runai/pipeline_preflight.py \
  --student /workspace/storage-shared/nlp/tungks/SimCT/runs/qwen-gemma-sft-paper-20260908-045828/checkpoint \
  --teacher /workspace/storage-shared/models/Qwen2.5-7B-Instruct \
  --dataset /workspace/storage-shared/nlp/tungks/SimCT/data/qwen-author/data/prompts.parquet \
  --algorithm span_ctkd --check-gpu
echo '=== END F1 ==='
)
```

**F2 — Phi preflight**, using paths verified from its launch config:

```bash
(
set -eu
export PYTHONPATH="$PWD/experiments/modal/vendor:$PWD"
export CUDA_VISIBLE_DEVICES=3 KDFLOW_TRUST_REMOTE_CODE=0
bash experiments/runai/python-b200-host.sh experiments/runai/pipeline_preflight.py \
  --student /workspace/storage-shared/nlp/tungks/SimCT/runs/phi4-gemma-sft-20261002-131247/checkpoint \
  --teacher /workspace/storage-shared/nlp/tungks/models/Phi-4-mini-instruct \
  --dataset /workspace/storage-shared/nlp/tungks/SimCT/data/phi4-prompts/data/prompts.parquet \
  --algorithm mp_opd --chunk-source run --check-gpu
echo '=== END F2 ==='
)
```

After preflight passes and GPU3 is confirmed unassigned (including sleeping
workers / queue leases), use the existing runner with a fresh run directory.
Set assets explicitly. Preserve full length4096 for the node canary: shortening
responses would not exercise the same memory risk. Do not automatically promote
a canary to a long job solely because loss is finite.

**F3 — Qwen GRASS run2 canary**, batch64/micro4, two optimizer updates:

```bash
(
set -eu
export MP_ALGORITHM=mp_opd MP_SEED=42 MP_RESUME=0 MP_TENSORBOARD=1
export MP_STUDENT_PATH=/workspace/storage-shared/nlp/tungks/SimCT/runs/qwen-gemma-sft-paper-20260908-045828/checkpoint
export MP_TEACHER_PATH=/workspace/storage-shared/models/Qwen2.5-7B-Instruct
export MP_DATASET_PATH=/workspace/storage-shared/nlp/tungks/SimCT/data/qwen-author/data/prompts.parquet
export MP_MICRO_TRAIN_BATCH_SIZE=4 MP_GRASS_CHUNK_SOURCE=run MP_GRASS_CHUNK_RUN_LENGTH=2
export MP_RUN_ROOT=/workspace/storage-shared/nlp/tungks/simct-runs-chunk-rl2-s42
unset MP_RUN_DIR MP_MAX_LEN MP_PREFLIGHT_ONLY
bash experiments/runai/run_single_gpu.sh 3 grass_chunk 2
echo '=== END F3 ==='
)
```

For run3/run4, change `MP_GRASS_CHUNK_RUN_LENGTH` and the run-root `rl<N>`.
For temporal, use mode `grass_chunk_temporal`, run-root
`simct-runs-chunk-temporal-rl<N>-s<seed>`, and explicitly set
`MP_GRASS_CHUNK_TEMPORAL_START_STEP=20`, `MP_GRASS_CHUNK_TEMPORAL_END_STEP=160`.
For SimCT, set `MP_ALGORITHM=span_ctkd`, pass mode `atomic` as the neutral launcher
slot, and use `simct-runs-simct-rl1-s<seed>`. For Phi, replace all three asset
paths with F2 and include `phi` in the run-root. To run a fresh longer job after
the canary, choose a new output directory; do not describe it as a resume.

Collect launch-config, tokenizer-loader policy/hash, SimCT overlap digest,
train log, TB, checkpoint SHA and raw evaluation output. Step40 eval must cover
code and math; premature EOS/repetition/token-cap status are needed alongside
accuracy. The checkpoint request list lives in the persistent audit evidence.

New runner task/JIT caches live under `/workspace/storage-shared/nlp/tungks/tmp/`
(or `MP_TASK_CACHE_ROOT`). Ray logs and session metadata also stay there. Its Unix
IPC endpoints use short `/tmp/rs-*` directories because distributed/FUSE volumes
need not support Unix socket binding. These endpoints are live process resources,
not evidence or reusable cache bytes. The shared cache is retained, never
automatically deleted here. This is not qualification of a full trainer resume.

## If node preflight fails

| Failure | Concrete action |
|---|---|
| Wrong interpreter / Transformers4.56 Gemma loader error / missing packages | Run through the repository `python-b200-host.sh` and `/opt/venvs/simct-b200/bin/python`, rather than `/usr/bin/python` or a different venv. Rerun F1/F2. |
| Missing `runtime-host-libs` | Set `MP_RUNTIME_DIR` to the existing qualified task directory containing those libs. Do not borrow arbitrary CUDA libraries or silently change the dependency stack. |
| Missing NVRTC headers or compiler/header skew | Check the wrapper's `/usr/local/cuda-13.0/include` and venv cu13 include; the wrapper orders toolkit headers first. Restore the qualified payload if incomplete. |
| Runtime payload absent / damaged | Download the existing immutable export below directly on the node, verify SHA256 and inspect targets. Its installer refuses existing paths; an existing environment requires an explicit repair plan, not recursive deletion. |
| Ray `OutOfMemoryError` | Host RAM, CPU backup/plasma and other processes need inspection; it is not evidence of CUDA OOM. Preserve the monitor. This handoff does not establish batch64 host-RAM capacity on the target node. |
| CUDA OOM / logprob parity / SGLang startup failure | Keep the traceback and launch manifest, classify separately. Do not apply the BF16 Qwen eager reference threshold to production SGLang as if it were an exact oracle. |

Existing export (verified through HF API on2026-10-11):

- Repo: `codemaivanngu/simct-b200-portable-cu130` (currently public).
- Revision: `bba4458241d53c3a9629bdb31bcd09b0677d9433`.
- `runtime.tar.gz`:7276682121 bytes;
  SHA256 `ce10659feea7ba24fd956bafb1f54c3a311e6499311ed00cd39af4dae5fe5b9e`.
- Image: `docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f`.
- Keep runtime and new Git source separate: the export's `source.tar` is historical.
  Fetch the new `codemaivanngu/simct` Git bundle instead of reverting to that source.

**F4 — download the existing runtime on the node**, only when needed; no install
or training starts in this block. It avoids a laptop transfer or a new Modal build:

```bash
(
set -eu
REV=bba4458241d53c3a9629bdb31bcd09b0677d9433
DEST=/workspace/storage-shared/nlp/tungks/tmp/runtime-pipeline-fix-$REV
mkdir -p "$DEST"
BASE=https://huggingface.co/datasets/codemaivanngu/simct-b200-portable-cu130/resolve/$REV
for FILE in runtime.tar.gz source.tar manifest.json packages.json targets.txt install.sh python-b200.sh smoke.py SHA256SUMS README.md; do
  test -f "$DEST/$FILE" || curl --fail --location --retry 3 \
    --proxy "${MP_DOWNLOAD_PROXY:-http://10.30.154.118:80}" --noproxy '' \
    "$BASE/$FILE" -o "$DEST/$FILE.part"
  test -f "$DEST/$FILE" || mv "$DEST/$FILE.part" "$DEST/$FILE"
done
(cd "$DEST" && sha256sum -c SHA256SUMS)
printf 'Verified payload: %s\n' "$DEST"
echo '=== END F4 ==='
)
```

A new Modal→HF export is necessary only after a demonstrated dependency/ABI/base
image change. None of the confirmed fixes above requires it. Never reinstall a
runtime while serving/training processes are using its paths.
