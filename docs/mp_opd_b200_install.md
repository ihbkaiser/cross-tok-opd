# MP-OPD on a restricted-network B200 server

This guide extends `experiments/environments/B200_INSTALL.md`. MP-OPD is a
runtime mode in the same SimCT/KDFlow source tree, so the company server does
not need a dedicated MP-OPD branch. Pin one approved commit after the changes
are reviewed and merged into `vdt/experiments/xtoken`.

## 1. Private path overlay

Never put internal pool URLs, credentials, model mount paths, or W&B keys in
Git. Create a private file outside the checkout:

```bash
install -d -m 700 /workspace/simct-private
install -m 600 experiments/mp_opd/b200-100step.env.example \
  /workspace/simct-private/mp-opd.env
${EDITOR:-vi} /workspace/simct-private/mp-opd.env
set -a
. /workspace/simct-private/mp-opd.env
set +a
```

The launcher accepts exact paths. `student_name_or_path` may be a local base or
warm-SFT checkpoint, and `teacher_name_or_path` may be a local model directory.
No directory naming convention is assumed.

## 2. Internal Python pool without dependency confusion

Keep the committed public lock unchanged. Make a private overlay copy on the
company server and add the approved pool as the first index. A representative
private `pyproject.toml` fragment is:

```toml
[[tool.uv.index]]
name = "company"
url = "${COMPANY_PYPI_URL}"
default = true

[tool.uv]
index-strategy = "first-index"
```

Environment interpolation support depends on the company's uv wrapper. If it
does not interpolate TOML, render the URL into the **private copy**, never this
repository. Prefer the internal CA:

```bash
export SSL_CERT_FILE=/etc/company-ca.pem
export UV_NATIVE_TLS=1
```

Do not use `unsafe-best-match` unless the security owner has explicitly
accepted dependency-confusion risk. Do not commit `trusted-host`, credentials,
or a pool-specific lock. If the company requires its own lock, produce and
archive it privately with the source commit and pool snapshot identity.

Install from the checked-in B200 lock as described by the base runbook:

```bash
export UV_NO_MANAGED_PYTHON=1
export UV_CACHE_DIR=/workspace/cache/uv
export UV_PROJECT_ENVIRONMENT=/workspace/venvs/simct-b200
uv sync --project experiments/environments/simct-b200 --locked --offline
```

Remove `--offline` only if the internal pool/cache is expected to serve missing
artifacts. A resolution failure is a missing mirror artifact until diagnosed;
do not silently fall back to public PyPI.

## 3. Verify local model completeness and identity

```bash
for model_dir in "$MP_OPD_STUDENT_PATH" "$MP_OPD_TEACHER_PATH"; do
  test -d "$model_dir"
  test -f "$model_dir/config.json"
  test -f "$model_dir/tokenizer_config.json"
  find "$model_dir" -maxdepth 1 -type f \
    \( -name '*.safetensors' -o -name '*.bin' \) -print -quit | grep -q .
done

install -d -m 700 "$MP_OPD_EVIDENCE_DIR/model-manifests"
for model_dir in "$MP_OPD_STUDENT_PATH" "$MP_OPD_TEACHER_PATH"; do
  find "$model_dir" -maxdepth 1 -type f \
    \( -name '*.json' -o -name '*.model' -o -name '*.safetensors' \) \
    -print0 | sort -z | xargs -0 sha256sum \
    > "$MP_OPD_EVIDENCE_DIR/model-manifests/$(basename "$model_dir").sha256"
done

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
```

If a tokenizer or shard is missing, stop. Do not replace it with a similarly
named public model; that breaks the scientific lineage.

## 4. Preflight before spending B200 time

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
nvcc --version
df -h "$MP_OPD_OUTPUT_ROOT" "$UV_CACHE_DIR"
df -ih "$MP_OPD_OUTPUT_ROOT" "$UV_CACHE_DIR"

export PYTHONPATH="$PWD/experiments/modal/vendor:$PWD"
export KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT=1
python -m pytest tests/mp_opd tests/test_span_ctkd_metrics.py \
  tests/test_xtoken_algorithm.py -q
unset KDFLOW_LIGHTWEIGHT_ALGORITHM_IMPORT
git diff --check
```

Record command output, source commit and `git status --short`. A dirty server
checkout is allowed only if the exact patch hash is recorded.

## 5. Choose a scientifically honest 100-step mode

The current B200 canary may run `atomic`, `fixed`, or `random`. `oracle` is
instrumentation-only. `soft` requires both an audited energy checkpoint and a
real-data B/M integration; do not point it at a random energy file.

For the first bounded systems canary use `atomic`. This tests the new atomizer,
score credit, shifted labels, backward pass, optimizer accounting and metrics
without claiming learned partition gains:

```bash
bash experiments/mp_opd/run_b200_100step.sh
```

To run the fixed matched-capacity control after the atomic canary is terminal:

```bash
MP_OPD_MODE=fixed MP_OPD_FIXED_SPAN_LENGTH=2 \
  bash experiments/mp_opd/run_b200_100step.sh
```

Do not start both runs concurrently on the same output or W&B run ID.

### Live TensorBoard from Jupyter

The B200 launcher enables TensorBoard in parallel with W&B and writes the same
namespaced scalar train metrics plus the 15-second `system/*` GPU telemetry to
`$TENSORBOARD_LOG_DIR`. The default is
`$MP_OPD_OUTPUT_ROOT/tensorboard`. Event files are flushed on every explicit
publication, so completed data remains inspectable after an interrupted job.

From a Jupyter notebook attached to the company server:

```python
%load_ext tensorboard
%tensorboard --logdir /workspace/outputs/mp-opd-b200-100step/tensorboard --port 6006
```

Or start the server from a Jupyter terminal after activating the locked uv
environment:

```bash
tensorboard \
  --logdir "$TENSORBOARD_LOG_DIR" \
  --host 0.0.0.0 \
  --port 6006
```

Use a path visible to the persistent workspace, not `/tmp`. To opt in for any
other KDFlow SFT, off-policy, or on-policy command, add:

```bash
--use_tensorboard True \
--tensorboard_log_dir /workspace/outputs/<run-id>/tensorboard \
--tensorboard_flush_secs 10
```

TensorBoard mirrors metrics that the trainer actually emits; it does not
invent an evaluation split. The current 100-update MP-OPD launcher is a bounded
training/implementation validation and needs a separate held-out evaluation
command before making quality or efficacy claims.

## 6. Terminal evidence gate

A valid 100-step systems result needs all of:

- process exit zero and exactly 100 optimizer updates;
- finite MP-OPD loss and `b/r` metrics;
- finite gradient norm and learning rate;
- nonzero covered student and teacher events;
- atom failure reasons/counts and EOS masking recorded;
- B200 identity, peak memory and token throughput;
- exact source, model, tokenizer, dataset and config hashes;
- native W&B run in `finished` state if W&B is enabled; and
- clean final summary and checkpoint, with no OOM.

This remains `implementation_validation` for atomic/fixed/random unless a
matched scientific comparison and evaluation protocol is completed.

## 7. Modal B200 qualification launcher

`experiments/modal/mp_opd_b200_100step_modal.py` is the reproducible Modal
counterpart of the company-server launcher. It deliberately uses the same
runtime mode (`kd_algorithm=mp_opd`, `mp_opd_mode=atomic`) and does not require
checking out a dedicated branch. Its immutable contract is:

- one B200 with the CUDA 13 native lock;
- pinned Llama-3.2-1B student and Qwen3-4B teacher snapshots already stored in
  the audited source volume;
- 640 pinned unique prompts, each repeated ten times with an explicit
  `repeat_index`, giving 6,400 rows;
- one epoch, rollout/train batch 64 and one sample per prompt, giving exactly
  100 optimizer updates;
- a 100-step cosine scheduler with five warmup updates;
- one final checkpoint, complete local/TensorBoard logs, runtime/source/data
  manifests and native W&B telemetry; and
- `retries=0`, so a failed paid run is diagnosed rather than silently repeated.

Before submitting, run the `modal-gpu-ops` billing snapshot and quota guard for
the explicit profile. A `WARN` or `BLOCKED` result is a hard human gate. After
approval, invoke the unique runner without putting a token or URL in the shell
history:

```bash
modal --profile no1ceboy run \
  experiments/modal/mp_opd_b200_100step_modal.py
```

The launcher re-runs the B200 environment gate and MP-OPD test subset on the
allocated machine before training. A successful process is still marked
`completed_pending_wandb_audit` until the 100-update summary, W&B terminal
state, finite metrics, checkpoint and billing delta have been independently
finalized.
