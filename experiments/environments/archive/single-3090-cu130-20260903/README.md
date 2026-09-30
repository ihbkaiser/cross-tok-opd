# Recovered single-RTX-3090 setup (historical CUDA 13 variant)

This is the **actual old machine's setup**, recovered on 2026-09-03, not a new
resolution and not the CUDA 12.8 environment for the two-GPU machine. Files in
`source-configs/`, `original-scripts/`, `snapshots/` and `evidence/` are copied
byte-for-byte from the recovery capture. Do not run `uv lock` here or replace
the active `../../xtoken/` or `../../simct/` projects with these files.

## Material passport

- Origin Skill: academic-research-suite; mode: artifact recovery, no new training.
- Verification: off-host archive SHA-256 and all 128 payload files verified;
  selected Git payload separately checksummed by `SHA256SUMS.json`.
- Host: RTX 3090, 24 GiB, driver 580.95.05, compute capability 8.6.
- Actual X-Token Python **3.13.14**, uv **0.12.7**, Torch **2.11.0+cu130**,
  Transformers **5.12.1**, Ray **2.56.1**, NumPy **2.5.1**.
- Base environment: 255 distributions; isolated Automodel worker: 273.
- NeMo-RL: `13a10647ebbf0f940d2b06ea41800b3f2fb46099`, clean source tree.
- Automodel: `1814c6c93a66b9d59d254960ef6a99a64249b671`.
- SimCT harness: `9414f7cb7c3b44cce14a33d00ea3521d16b7ef69`, clean source tree.
- Other exact submodules and source-file hashes: `capture-provenance.json`.

## What is and is not locked

`source-configs/nemo-rl/pyproject.toml` and `uv.lock` are the unchanged upstream
pair. Lock SHA-256:
`95f63521d28a2a4104ff372c5985fe63826ab27d6901b78bada1ab1a89a81bf7`.
Original submodule manifests/locks and source licenses are retained. They have
relative workspace references and are **not standalone uv projects**.
Actual installed packages, direct URLs and editable paths are in `snapshots/`.

The dense HF / PyTorch SDPA worker used the locked upstream `automodel` extra
but omitted `transformer-engine`, `flash-attn`, `deep-ep`, `mamba-ssm`,
`causal-conv1d`, `nv-grouped-gemm`, and `tilelang`. Exact commands are preserved
in `original-scripts/install_native_envs.py` and phase JSON files. This native
subset is **not a clean full NVIDIA environment**. The initial base installer
returned 1 at dependency checking despite completing installation; that failure
remains recorded, not suppressed.

**Old SimCT** only had a Python 3.12.3 venv: its snapshot contains **zero
distributions**. Its original `pyproject.toml` is retained, but no old native
SimCT `uv.lock` existed. Both resolve attempts failed because the installer ran
from NeMo and inherited its uv configuration. This is not the working locked
CUDA 12.8 SimCT environment in `../../simct/`.

## Latest recovered results

- Upstream unit tests and synthetic CUDA/SDPA/P-KL tests passed; actual results
  and JUnit XML are in `evidence/`. Synthetic steps are not real-model training.
- Isolated `DTensorPolicyWorkerV2` import passed after native worker installation.
- Official pinned Llama-3.2-1B / Qwen3-1.7B downloads were verified; model
  revision/shard inventories are retained. Weights are not in Git.
- The frozen 32-row Formal Logic dataset was recovered off-host with its manifest.
- The disconnected projection attempt is now known to have **timed out**:
  exit 124, child -9, 689.604 s elapsed, configured timeout 600 s. No completed
  projection file exists. Do not auto-retry it.
- **No real-model optimizer update or successful SimCT training was produced.**
  This archive is not OPD or a scientific reproduction.

`original-scripts/README.md`, `REPORT.md` and older metadata are historical
snapshots. Their access/install status can be stale; recovered package
snapshots, terminal phase JSON and this verdict take precedence.

## Recovery procedure

1. Verify locally: `python3 verify_archive.py` (no dependencies/network/GPU).
2. On a separately authorized compatible host, restore NeMo and SimCT Git
   checkouts at the exact commits above, with recursive pinned submodules.
   Original scripts assume `/workspace/xtoken-native/NeMo-RL` and
   `/workspace/simct-xtoken-harness`. Compare their manifests with
   `source-configs/`; do not flatten relative submodule paths.
3. Use uv 0.12.7 and Python 3.13.14 for historical X-Token. Review preserved
   base/worker commands, omissions and dependency warnings. Do not automatically
   execute the historical failing SimCT installer.
4. Install into fresh isolated environments, compare packages/direct source
   origins against `snapshots/`, then run appropriate canaries. This is verified
   recovery material, **not a completed rebuild test** or new compute approval.
5. Fetch pinned weights only with separately authorized credentials. Package
   snapshots do not grant access to gated models.

## Off-host backup and deletion scope

Full bounded capture: 129 files including its manifest, stored at
`D:\dev\codex\research_vdt\xtoken-native-3090\backup-single3090-20260903.tar.gz`.
Archive SHA-256:
`87ad13688e8e67e7fa310f6561c2f60abe1624407892977aa22832757652ba8b`.
It includes raw task diagnostics, frozen smoke data and synthetic projections.
Git contains setup and selected small evidence, not raw logs or binary outputs.

Venvs, weights/caches, credentials, SSH keys, the default host environment and
unrelated user files are **not** backed up here. Destroying the old instance
would still remove those. Once the Git commit is pushed and off-host checks
pass, this task's setup no longer depends on its disk; this is not a machine
image. No instance was stopped, restarted or destroyed.

The recovered capture and candidate files passed an exact-value audit for the
designated HF credential and five generic credential patterns (0 findings).
This is not a security claim about unrelated stores or tool output.
