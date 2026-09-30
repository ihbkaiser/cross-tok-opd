# Two independent native CUDA 12.8 environments

For the separate CUDA 13/B200 profile, including company package pools and
pre-mounted internal model paths, see the
[B200 installation runbook](B200_INSTALL.md).

These **checked-in `pyproject.toml` + `uv.lock` pairs** extend the historical
[X-Token harness](../xtoken/README.md). Run them instead of that harness's
CUDA 13 upstream install on a driver-570 RTX 3090 host. The root SimCT
`pyproject.toml` and both algorithms remain unchanged.

The recovered **single-GPU CUDA 13 setup** is preserved separately in the
[historical archive](archive/single-3090-cu130-20260903/README.md), including
original workspace manifests/locks, installed package snapshots and scripts.
It must not be mixed into these CUDA 12.8 projects. Old SimCT was incomplete;
the archive does not invent a successful install or a missing native lockfile.

| Project | Python | PyTorch | Transformers | Scope |
|---|---|---|---|---|
| `xtoken/` | 3.13.15 | 2.11.0+cu128 | 5.12.1 | NeMo dense HF/SDPA, off-policy smoke |
| `simct/` | 3.12.3 | 2.9.1+cu128 | 4.57.1 | KDFlow + SGLang 0.5.9 + FlashAttention 2.8.3 |

Workload uv is pinned to **0.12.7**. Source archives use exact commits and
SHA-256 in the lock. FlashAttention uses the upstream official CPython 3.12,
CUDA 12, Torch 2.9, CXX11-ABI wheel, also hash-locked. No driver changes,
Docker daemon, default-venv changes, cloud rental or credential storage occur.

## Native variant limitations

NeMo's upstream metadata mandates five dependencies that are deliberately
excluded here: `nvidia-nvshmem-cu13`, `nvidia-cudnn-cu13`,
`mooncake-transfer-engine-cuda13`, `tilelang`, and `nixl-cu13`.
Consequently **X-Token `uv pip check` returns nonzero with those five missing
requirements**. This is recorded, not suppressed or called a clean health pass.
Torch supplies its CUDA 12 cuDNN runtime. MoE, NIXL/Mooncake/NVSHMEM data planes,
TE, FP8, context parallelism and optional generation workers are not supported
by this dense smoke environment. Do not use it as a drop-in full NeMo install.

The dense import/CUDA/NCCL canaries have passed; trainer success is a separate
gate. Consult the [current report](reports/2026-09-03-2x3090.md), not a successful
resolver or import message, before claiming training or scientific results.

## Rebuild on an authorized Linux GPU machine

Use a dedicated runtime directory outside the worktree. The example assumes
the branch checkout is `/workspace/SimCT`; replace this with its absolute path.
The actual two-GPU test host has driver 570.181, two 24 GiB RTX 3090 cards over
PCIe (no NVLink), and roughly 171 GiB container RAM. Projection construction
materializes a large CPU tensor: a low-RAM host is not equivalent.
The observed dual envs plus weights occupy about 26 GiB. Prefer >=50 GiB disk;
the tested 32 GiB disk leaves only about 6 GiB, inadequate for optimizer
checkpoints or paper-size output. Check free space before any larger run.

```bash
python3 /workspace/SimCT/experiments/environments/run_phase.py xtoken sync --root /workspace/vdt-runtime
python3 /workspace/SimCT/experiments/environments/run_phase.py simct sync --root /workspace/vdt-runtime
python3 /workspace/SimCT/experiments/environments/prepare_source.py --root /workspace/vdt-runtime
python3 /workspace/SimCT/experiments/environments/run_phase.py xtoken check --root /workspace/vdt-runtime
python3 /workspace/SimCT/experiments/environments/run_phase.py simct check --root /workspace/vdt-runtime
python3 /workspace/SimCT/experiments/environments/run_phase.py xtoken canary --root /workspace/vdt-runtime
python3 /workspace/SimCT/experiments/environments/run_phase.py simct canary --root /workspace/vdt-runtime
```

The expected nonzero X-Token metadata check above requires explicit review;
do not blindly chain through arbitrary dependency failures. Each phase saves
its real exit code, wall time and GPU telemetry in a unique `evidence/` folder.
Independent envs may be synced concurrently; never mutate one env while its
workload runs. `lock` is a deliberate maintainer action, not part of rebuild.

The default install deadline is 1800 seconds. After a diagnosed download timeout
and explicit user approval, a new invocation may reuse the existing cache with
`--sync-timeout-seconds 5400` (90 minutes). This option applies only to `sync`;
it does not extend GPU canary/training deadlines or trigger automatic retries.
Each invocation still retains its own timestamped evidence directory.

The outer deadline is separate from uv's HTTP timeout (default 30 seconds).
After an explicitly approved network-failure retry, pass
`--sync-http-timeout-seconds 300 --sync-timeout-seconds 5400` to `sync`.
The effective `UV_HTTP_TIMEOUT` is recorded in the command evidence. It does not
change the lockfile, dependency versions, GPU canary or training configuration,
and it does not modify any installation already running.

`prepare_source.py` verifies **all** original archive members against an existing
source tree and refuses modification. It does not overwrite/reset the tree.
Runtime bytecode/build extras are not treated as upstream source changes.

## Exact model and smoke setup

`download_models.py --root /workspace/vdt-runtime` receives the authorized HF
token on **stdin**, never argv. Use a protected task-local credential relay;
do not paste a token or commit a relay containing machine/account details.
Only the gated student download receives it. Weights remain outside Git;
the script verifies all shards and records file SHA-256, then exits without
persisting auth. An existing `models-evidence/` is not silently retried.

- Student: Meta Llama-3.2-1B `4e20de362430cd3b72f300e6b0f18e50e7166e08`.
- Teacher: Qwen3-1.7B `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- Data: first 32 Formal Logic rows from NVIDIA's public Specialized v1.1
  dataset, Viewer revision `13fa979be2e7f7e62913eee0ec5e97c8fd6e24af`.
  Both revision and frozen Parquet hash are checked; change is a hard failure.

```bash
python3 /workspace/SimCT/experiments/environments/xtoken_smoke.py data --root /workspace/vdt-runtime
python3 /workspace/SimCT/experiments/environments/xtoken_smoke.py projection --root /workspace/vdt-runtime
python3 /workspace/SimCT/experiments/environments/xtoken_smoke.py config --root /workspace/vdt-runtime
# Execute only with an explicit authorized training attempt:
python3 /workspace/SimCT/experiments/environments/xtoken_smoke.py train --root /workspace/vdt-runtime
```

Smoke follows NVIDIA's functional test: exact Llama-1B/Qwen-1.7B, two GPUs,
BF16, batch 8, microbatch 1, sequence 256, three updates, top-4 tokenizer
projection, original P-KL+CE loss/scheduler. Changes are the pinned 32-row
subset and dense HF/SDPA backend. **This is off-policy, not OPD.** Checkpointing
stays disabled as in the functional test; no quality or persistence claim is
possible from this smoke. Student/teacher revisions are checked before use.

The launcher passes `uv --project` explicitly while keeping the **NeMo source
root as cwd** for upstream prompt files. New Hydra fields use `+`. Registered
Automodel Ray workers reuse the same native venv via the supported
`NEMO_RL_VENV_DIR/<actor-class>/bin/python` lookup, preventing automatic
CUDA 13 worker installation. Model operations are offline after download.

No automatic training retry occurs. A second attempt needs both a distinct
`--train-attempt` number and `--retry-reason`, after diagnosis and authorization.
The runner watches process liveness and enforces a 20-minute training deadline;
it terminates only its own process group. This foreground runner is **not** a
durable service across host failure or lost SSH. Export small diagnostics and
projection before relinquishing a rented machine. Never assume disk persists.

## Diagnosed NCCL cuMem host-memory workaround

On the tested driver-570 PCIe host, Ray-isolated GPU workers hit
`transport/shm.cc:590` / CUDA error 217 (`peer access is not supported`),
followed by illegal-address errors during teacher setup/offload. The same
model's standalone HF and torchrun buffer transfers worked. Disabling P2P
alone did not solve it. **Setting only `NCCL_CUMEM_HOST_ENABLE=0` passed the
two-worker Ray teacher setup/offload probe and the subsequent three-update
training smoke.** This matches
[NVIDIA NCCL issue 1838](https://github.com/NVIDIA/nccl/issues/1838).

Use the opt-in host workaround only after this failure is diagnosed:

```bash
python3 /workspace/SimCT/experiments/environments/xtoken_smoke.py config --root /workspace/vdt-runtime --nccl-cumem-host-fallback
# Select a NEW attempt number and record the diagnosed reason; do not replay an existing marker.
python3 /workspace/SimCT/experiments/environments/xtoken_smoke.py train --root /workspace/vdt-runtime --nccl-cumem-host-fallback --train-attempt 5 --retry-reason diagnosed_nccl_cumem_host_ipc_error217_probe_pass
```

The flag changes NCCL host allocation, not model precision, loss, corpus or
optimizer. It neither disables all shared-memory transport nor changes the
driver. All inherited NCCL/allocator debug settings enter the config digest
and attempt record; changing them requires a fresh config gate. At least
2 GiB free disk is required at launch. `--cuda-launch-blocking` is available
for localization only; its timings must not be reported as normal throughput.

Setup-only diagnostics use `run_cuda_probe.py` with one of `hf`, `nemo`, or
`ray` plus `--root <runtime>` (240-second timeout, offline cached models,
no checkpoints).
Each probe is single-attempt by default. A diagnosed probe-code correction
or controlled transport variant requires a distinct `--probe-id`. The Ray
probe uses an explicit diagnostic worker extension and does not patch NeMo
source or change the training worker. Probe failures and corrections are
listed in the report; these are not optimizer-update evidence.

Verified attempt 5 completed all three updates in 86.36 seconds end-to-end;
its step times were 17.96, 10.33 and 10.17 seconds. This is a short off-policy
engineering smoke, not a quality result, sustained benchmark, OPD or SimCT
reproduction. SimCT training itself has not been run. See the detailed
[host validation report](reports/2026-09-03-2x3090.md).

## Larger 4B / 2048 / global-batch-64 bounded test

`xtoken_target.py` is a separate, single-attempt harness for the explicitly
requested Llama-3.2-1B <- Qwen3-4B workload. It defaults to **three updates**;
the separately authorized `--steps 10` selects ten optimizer updates in a
different run directory. Both retain two GPUs, BF16, native HF/SDPA, P-KL+CE,
and the upstream optimizer/scheduler (250-step warmup, 4750-step cosine horizon).
It is not a full training run, H-KL, OPD, or a paper-quality reproduction.
It does not overwrite `smoke-2gpu-r1` or silently reduce batch/sequence size.

The teacher is pinned to `1cfa9a7208912126459214e8b04321603b3df60c` and all
three public safetensors shards are SHA-256 checked. The student comes from a
licensed cache (or an authorized direct Hub download) with `previous-models.json`;
all student files are re-hashed against that earlier verified manifest. No
credential is persisted or needed for the public teacher.

The new **full projection** executes the same three CLIs and default choices as
the pinned `tools/x_token/build_projection_matrix.sh`: multi-token top-32 with
reverse pass, scale trick and special mappings; `reapply_exact_map`; then
`sort_and_cut_projection_matrix` to runtime top-4. The last step honors the
upstream scale-slot metadata. Only the native `uv --project --locked --no-sync`
launcher and artifact names differ from the wrapper, avoiding its implicit
CUDA-13 environment sync. No embedding model is loaded by this wrapper.
All three artifacts, metadata, shapes, index/weight checks and SHA-256 hashes
are retained under `projection-full/`. They are **not assumed equivalent** to
the lighter functional-test projection. Such equivalence would require separate
artifact and same-input loss/gradient comparisons.

The data is a contiguous, deterministic public Formal Logic prefix at the same
Viewer revision, extended until at least one extra global batch of complete
16,384-character packs exists: 256 packs for three updates, 704 for ten.
The prefix is not synthetically repeated. The CPU gate checks respectively
>=192 or >=640 full-length student and teacher samples at length 2048 and emits
the resolved config. A run-directory protocol marker rejects mixing step counts.

```bash
python3 /workspace/SimCT/experiments/environments/xtoken_target.py models --root /workspace/vdt-runtime
python3 /workspace/SimCT/experiments/environments/xtoken_target.py data --root /workspace/vdt-runtime
python3 /workspace/SimCT/experiments/environments/xtoken_target.py projection --root /workspace/vdt-runtime
python3 /workspace/SimCT/experiments/environments/xtoken_target.py config --root /workspace/vdt-runtime
# Only after resource, source, projection and config gates, with user authorization:
python3 /workspace/SimCT/experiments/environments/xtoken_target.py train --root /workspace/vdt-runtime
```

For the authorized ten-update run, append `--steps 10` to **every** phase above.
Its default directory is `target-4b-2048-b64-full-10steps-r1`. Never replay an
existing phase marker. `run_target_after_setup.py` can execute that exact sequence
once after three explicit, already-running setup result paths become successful.
It does not reinstall, redownload or retry anything on failure, does not wait for
the independent SimCT environment, and refuses GPU/resource or source drift.
The caller supplies the end of the existing setup window as the readiness
deadline; this does not extend any producer's timer. Each actual phase keeps its
own bounded runner. Final process completion still requires separate metric
review; it is not automatically promoted to a validated scientific result.

After the first sequence stopped **before every execution phase** because the
student transfer timed out, an explicitly authorized continuation can use
`--sequence-attempt 2 --continuation-of /workspace/vdt-runtime/sequence-target-4b-2048-b64-full-10steps-r1`.
This uses a fresh `...10steps-r2` directory and the verified result from
`student-http-resume-r4/result.json`; all scientific launch/config/data scripts
and lock hashes must match the predecessor. It cannot replay prior training.

`resume_student_transfer.py --root /workspace/vdt-runtime` implements that
single authorized transfer continuation. It takes the HF token from stdin only,
copies the inactive partial to a separate staging file, requests the missing
HTTP Range once, and requires the full pinned hash before no-overwrite cache
publication. The original partial and failed result remain intact. It has a
hard 1800-second timer and no automatic network retry; do not run it again after
a failed attempt without diagnosing and obtaining new authorization.

### Managed recovery after console disconnection

`run_logged.py` treats a disconnected console as advisory, independently of
durable command logs, `progress.json`, telemetry and terminal `result.json`.
It preserves real child exit codes and hard timeouts; other I/O errors still
fail. This alone is not process persistence: use Supervisor for unattended
execution, not a foreground SSH session.

`managed_target.sh --root /workspace/vdt-runtime` is the explicitly authorized
recovery after the setup-only r2 BrokenPipe failure. Run it through a dedicated
Supervisor program with `autostart=false`, `autorestart=false`, `startretries=0`
and `startsecs=0`, starting the program exactly once. It claims the exclusive
`managed-target-10steps-r3` directory and a runtime lock. An attempted replay
fails before any installation. It reuses verified model results, checks the
predecessor's scientific scripts/lock hashes, and runs one X-Token sync with
the same 5400-second outer limit and HTTP timeout 300. No SimCT retry is included.

Only successful installation invokes `--sequence-attempt 3` with setup-only r2
as its predecessor. All existing model/data/full-projection/config/resource
gates and the ten-update protocol remain unchanged. Logs and results live on
the instance; disconnecting the inspecting SSH client does not control their
lifetime. Supervisor does not restart failed/completed work or auto-start it
after reboot. This does not protect against instance recycle/destroy, nor does
a running install/sequence mean that any optimizer update has completed.

### Authorized TUNA transport fallback

After managed r3 reached its 90-minute download timeout (not BrokenPipe),
`managed_target.sh --root /workspace/vdt-runtime --pypi-mirror` claims a distinct
`managed-target-10steps-mirror-r4` attempt. It requires that exact setup-only
timeout and refuses to run if sequence r3 already exists. The unused sequence
r3 is launched only after successful installation; it still continues from the
original setup-only sequence r2 with unchanged scientific hashes.

`mirror_sync.py` reads the canonical lock and checks TUNA's advertised filename
and SHA-256 for the two incomplete wheels, `nvidia-cublas-cu12` and
`nvidia-cudnn-cu12`. It creates a private installation view under
`<runtime>/mirror-installs/`; only those two wheel URLs change. Registry
identities, dependency graph, all versions/hashes, the original project/lock,
and Torch's official CUDA 12.8 source remain unchanged. Normalizing the two URLs
must reproduce the canonical lock byte-for-byte. The actual install is
`uv sync --frozen --project <view>`, using the existing dedicated X-Token venv
and cache. `uv` verifies full wheel hashes; the index-advertised hash alone is
not evidence of a complete verified download. No global index configuration,
TLS bypass, re-resolution, cache deletion or automatic fallback/retry occurs.

Run this explicitly authorized continuation through its own Supervisor program,
with the same no-restart settings and 5400-second installation deadline. Keep
the original failed results and the transport manifest. This flag is not a
general-purpose switch to alter dependencies or retry training.

Every phase claims an exclusive attempt marker; failure stops the sequence.
The migrated run explicitly retains `NCCL_CUMEM_HOST_ENABLE=0` from the working
host as an operational compatibility setting, not a new-host diagnosis.
Each command is bounded to 30 minutes with GPU/process telemetry. Full-vocabulary
teacher logits can dominate VRAM at this shape: a small-smoke pass does not
guarantee this larger configuration fits two 24-GiB GPUs.

### Autonomous operational continuation after data HTTP502

The user subsequently authorized bounded automatic operational recovery until
the ten-update test succeeds. This does not authorize changing the scientific
protocol, renting a new instance, destroying data, or committing/pushing code.

`continue_target_data.sh --root /workspace/vdt-runtime` is a distinct managed
continuation (`sequence-target-4b-2048-b64-full-10steps-r4`), following only the
diagnosed sequence-r3 data failure. It rehashes prior sources and cached model
files, rechecks dependency exclusions and the two-GPU canary, and never runs an
installer or a model transfer. It uses the same exclusive/shared-lock and
Supervisor no-restart rules as above. Do not launch it again after its directory
has been claimed.

`resumable_target_data.py` preserves the exact dataset revision, contiguous
prefix, 16384-character packing, 704-pack target and original training shape.
It retries only transient HTTP/transport errors, at most eight attempts per
page, with recorded backoff/Retry-After and a 1500-second phase deadline. Revision,
row-index, truncation and hash failures are not retried. Each page gets an atomic
revision/hash sidecar; only such verified cache entries can be reused. Legacy
r3 pages are preserved but fetched again because they lack those sidecars.
Full-projection, configuration and training continue through the unchanged
scientific harness. Any later failure needs diagnosis and a distinct attempt;
the new authorization permits this without asking again for each operational
error, but never permits an unrecorded blind retry or a scientific reduction.

### CPU teacher-logit transport after full-batch CUDA OOM

At global batch 64 / sequence 2048, the pinned upstream teacher retains all
32 local microbatch logits on each GPU before student training. FP32 full-vocab
storage requires about 37.09 GiB per rank, larger than a 3090's VRAM.
`build_cpu_logits_overlay.py` makes a separate, hash-pinned NeMo source copy;
it changes only allocation/rebuild/microbatch transport. The original source
and lock remain untouched. `cpu_logits_transport.py` shares cgroup-accounted
RAM through owner-only memfd mappings and uploads only a selected microbatch.
There is no dtype cast, truncation, batch reduction, or optimizer/loss change.
This increases RAM and PCIe traffic requirements; it is not a speedup claim.

If that exact FP32 buffer itself crosses Ray's node-memory guard,
`cpu_logits_packed_transport.py` may be selected only by the dedicated recorded
continuation. It retains the upper 16 bits of each FP32 word only after checking
that every lower 16-bit word is zero and the value is finite; otherwise it
raises before training. Consumer slices reconstruct the original FP32 bit
pattern. This lossless encoding is applicable to the observed BF16-origin,
upstream-FP32-converted logits, but it is not a general FP32 compressor and
must never round or fall back silently. The mathematical tensor presented to
the loss remains FP32. Canary coverage includes bitwise comparison, test
loss/gradient parity, signed/range values, rejection of a non-encodable value,
and both physical GPUs.

`cpu_logits_attempt.sh --root /workspace/vdt-runtime --retry-same-gpu-canary`
is the unique, gated `cpu-r7` continuation after recorded fcntl-constant and
cross-GPU P2P baseline canary failures. It tests same-GPU CUDA IPC between two
processes on each physical GPU, without requiring unsupported peer access.
It must not be replayed after its directory is
claimed. Run only through its dedicated no-restart Supervisor program. The
worker preserves failed attempts, verifies exact predecessor/artifact hashes,
requires 90 GiB reclaimable cgroup headroom, and runs a two-process transport
canary followed by config parity before one 1800-second training attempt.
The small canary proves tested transport parity, not complete scientific
equivalence or ten-update success. No existing OS/cgroup limits are changed.

The recorded `cpu-r9` continuation also pins and overlays the upstream
`dtensor_policy_worker_v2.py` solely to evict dead CUDA caching-allocator
blocks immediately before `offload_after_refit` moves the live teacher model
to CPU. This addresses the observed allocator failure after lossless packed
logits had accumulated; it does not change any live tensor, dtype, loss,
optimizer, model configuration, or system memory limit. The continuation is
single-shot and must fail closed if its exact `cpu-r8` predecessor signature
or pinned upstream source hashes do not match.

If a predecessor leaves either device with
`gpu_recovery_action=Reboot`, CUDA initialization is considered a hard host
gate: do not replay a canary or training sequence until the instance has been
rebooted and a fresh two-GPU CUDA initialization probe passes. An idle
`nvidia-smi` memory/utilization display alone is not sufficient evidence that
CUDA contexts can be created.

## Local packaging tests (no network/GPU/model download)

```bash
python3 -m unittest discover -s /workspace/SimCT/experiments/environments/tests -v
python3 -m unittest discover -s /workspace/SimCT/experiments/xtoken/tests -v
```
