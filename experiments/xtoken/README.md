# X-Token: isolated native uv setup

**For the two-RTX-3090 / CUDA 12.8 setup, use the newer
[independent native projects and lockfiles](../environments/README.md).**
This page documents the earlier upstream CUDA 13 harness; its historical
limitations below are not the current native-environment status. The two
install paths are alternatives, not steps to mix in one environment.

This experiment lives on `vdt/experiments/xtoken`, based on the SimCT fork's
`main`. It packages the single-RTX-3090 setup harness; it does not port X-Token
into KDFlow, modify Tunix, or implement on-policy distillation.

## Ownership and reproducibility

- Harness: this directory, versioned with SimCT.
- Implementation: [NVIDIA NeMo-RL](https://github.com/NVIDIA-NeMo/RL), downloaded
  into a **separate checkout**, not copied into SimCT or installed into its env.
- Identity: [`upstream.json`](upstream.json) pins the full commit, SHA-256 of
  `uv.lock`, recursive submodule commits, and workload uv version (`0.12.7`).
- Python and ML dependencies: upstream `.python-version`, `pyproject.toml` and
  `uv.lock`; do not merge them with SimCT's Python dependencies or relock here.
- Runtime artifacts: a dedicated absolute task root **outside** this worktree.
  The harness refuses changed source/lock/submodules; it never resets changes.

No venv, downloaded checkout, model weights, HF tokens, dataset, raw run logs,
SSH target, or provider credential is committed. The instance-specific SSH and
credential-transfer scripts from the original setup are intentionally excluded.
Do not pass secrets on command lines. These setup actions strip inherited
token/secret/password/API-key variables and do not authenticate to Hugging Face.
Raw subprocess logs are local diagnostics, not automatically publication-safe:
review/redact them before sharing (including private index/proxy configuration).

## Current evidence and limits

The [2026-09-03 report](reports/2026-09-03-3090-setup.md) records **133 passed,
2 skipped** upstream unit tests and successful synthetic CUDA/loss execution.
It also records four `uv pip check` incompatibilities from upstream overrides.
The base installation is not a clean dependency-health pass.

**Automodel is not yet installed/validated. No full model weights, actual
tokenizer projection, corpus, trainer run, model checkpoint, held-out evaluation,
or multi-GPU scaling result has been validated.** The synthetic P-KL smoke
optimizes logits, not language-model weights. Its VRAM/timing cannot predict
Llama/Qwen training memory or throughput. Passing execution is not evidence of
normalized KL, model improvement, or scientific reproduction.

## Commands (Linux/WSL, on an explicitly authorized machine)

Prerequisites: Git, Bash, Python >=3.10 for the standard-library harness, and
uv `0.12.7` for workloads. CUDA actions need the appropriate NVIDIA driver and
GPU. The original setup used a single RTX 3090; this harness does not install
drivers, modify a default venv, create instances, or stop billed resources.

From the SimCT worktree root, choose a dedicated task directory. On a rented
machine `/workspace/xtoken-native` is one example, not a required machine path:

```bash
export XTOKEN_TASK_ROOT=/workspace/xtoken-native
export XTOKEN_HARNESS="$PWD/experiments/xtoken/scripts/xtoken.py"

# Fetch pinned code/submodules once. Existing checkouts are only verified.
uv run --no-project --python /usr/bin/python3 "$XTOKEN_HARNESS" \
  --root "$XTOKEN_TASK_ROOT" prepare

# Local provenance checks only; no dependency install or GPU work.
uv run --no-project --python /usr/bin/python3 "$XTOKEN_HARNESS" \
  --root "$XTOKEN_TASK_ROOT" check
```

Use `--uv /absolute/path/to/uv` if the workload uv executable is not on PATH.
The launcher uv may differ, but install/test/smoke/capture require the pinned
workload version. No uv upgrade is performed automatically.

Install the upstream **base + build + test** groups (not optional workers):

```bash
uv run --no-project --python /usr/bin/python3 "$XTOKEN_HARNESS" \
  --root "$XTOKEN_TASK_ROOT" --timeout 3600 install-base
```

This includes a CUDA/BF16 import probe and ends with `uv pip check`. A nonzero
exit stays nonzero: inspect `result.json` and `stdout.log`. The four historical
override conflicts are documented in the report, not silently accepted/fixed.
Do not blindly continue or repeatedly reinstall after an unexpected failure.

After reviewing installation diagnostics, individual authorized checks are:

```bash
uv run --no-project --python /usr/bin/python3 "$XTOKEN_HARNESS" \
  --root "$XTOKEN_TASK_ROOT" test-base
uv run --no-project --python /usr/bin/python3 "$XTOKEN_HARNESS" \
  --root "$XTOKEN_TASK_ROOT" gpu-smoke
uv run --no-project --python /usr/bin/python3 "$XTOKEN_HARNESS" \
  --root "$XTOKEN_TASK_ROOT" capture
```

Each action creates a unique `artifacts/<timestamp>-<action>-<suffix>/` with
command metadata, stdout, result/exit status and best-effort GPU telemetry.
Test XML and smoke outputs stay in that run directory. No automatic retries,
overwrite, training continuation or recovery checkout occurs. Timeout exit is
124; Ctrl-C exit is 130, with child-process-group cleanup. Missing GPU telemetry
is recorded; it is not proof of zero utilization or valid performance evidence.

The runner sets task-specific uv/HF caches and disables W&B. Use this entrypoint
consistently rather than activating another venv or relying on inherited cache
settings. Do not run two actions concurrently against the same task root.
After a partial `prepare` failure, preserve its diagnostics and diagnose it;
choose a new task root for an explicitly authorized retry.

## Local harness checks (no CUDA, torch, model or network required)

```bash
uv run --no-project --python /usr/bin/python3 -m unittest discover \
  -s experiments/xtoken/tests -v
bash -n experiments/xtoken/scripts/prepare.sh
bash -n experiments/xtoken/scripts/install_base.sh
bash -n experiments/xtoken/scripts/test_base.sh
git diff --check
```

These validate packaging, provenance gates and process lifecycle only; they are
separate from the historical upstream GPU results.
The [packaging validation report](reports/2026-09-03-harness-validation.md)
records 36 passing offline tests, including a local Git bootstrap fixture.

## Next gate, not yet implemented

Complete the locked Automodel worker backend, then run a bounded single-GPU
end-to-end training smoke with independently pinned model/data/tokenizer and
projection identities, finite gradients, a checkpoint and actual VRAM/timing.
The proposed public `unsloth/Llama-3.2-1B` BF16 substitute has not been downloaded
or proven tensor/tokenizer-identical to the original Meta revision by this
harness. A different initialization must have its own run identity and cannot
be presented as matching the original experiment. Only after real training
validation should we decide on 2x3090 or an OPD extension.

Backing up code does not preserve rented disks. Export reviewed, small
diagnostics before destroying/recycling an instance; never assume persistence.
