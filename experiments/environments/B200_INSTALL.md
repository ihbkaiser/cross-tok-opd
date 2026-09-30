# B200 installation and company-network runbook

This runbook turns the checked-in B200 manifests into a reproducible setup on
a Linux x86_64 company server. It covers both an internet-connected host and a
host that can only reach an internal Python package pool and internal model
storage.

The target is environment and short-run qualification:

1. install two isolated, lock-controlled Python environments;
2. prove CUDA 13, Torch 2.11, and Blackwell `sm_100` compatibility;
3. resolve all model and dataset inputs from explicit local paths;
4. run a bounded SFT warm-up and then a bounded SimCT check; and
5. retain configuration, metrics, hashes, and checkpoints outside Git.

Passing these steps establishes portability and execution correctness. It is
not, by itself, a reproduction of the paper's two-epoch/five-seed results.

## 1. Why this guide has a separate restricted-network path

The RLCSD fork previously added a CUDA 13 uv project in commit
[`d50254e`](https://github.com/sontungkieu/RLCSD/commit/d50254ec01fa9fd46a1c54f678c7a2c15a32bdeb).
That project routed packages through several indexes, including an internal
pool, and enabled an insecure host plus `unsafe-best-match`. The temporary
project was removed in commit
[`1b9eec4`](https://github.com/sontungkieu/RLCSD/commit/1b9eec409796a8cbfea3be05a6f5818ee2283a56).

The useful lesson is that the company host may require an internal package
route. This repository deliberately does **not** copy the internal address or
credentials and does not adopt `unsafe-best-match`:

- internal URLs belong in a private runtime overlay, never in the public repo;
- credentials belong in environment variables or the company secret manager;
- `first-index` remains the safe resolver strategy, preventing dependency
  confusion;
- an internal CA is preferred to disabling TLS verification; and
- public and company-pool locks are kept distinct and labelled.

uv documents that indexes referenced by `[tool.uv.sources]` must be declared
inside the project, that `explicit = true` limits an accelerator index to the
packages pinned to it, and that `first-index` is the safe default. See the
[uv package-index documentation](https://docs.astral.sh/uv/concepts/indexes/)
and [PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

## 2. Exact repository inputs

Use branch `vdt/experiments/mp-opd`. Record the commit before installation:

```bash
git clone https://github.com/sontungkieu/SimCT.git /workspace/SimCT
cd /workspace/SimCT
git fetch origin vdt/experiments/mp-opd
git switch --detach origin/vdt/experiments/mp-opd
git rev-parse HEAD | tee /workspace/simct-b200-provenance.commit
git status --short
```

For a controlled company run, replace the moving branch ref with the exact
approved commit. A non-empty `git status --short` is a provenance failure until
the changes are understood and recorded.

The two independent manifests are:

| Purpose | Project | Environment |
|---|---|---|
| SFT/warm start | `experiments/environments/simct-b200-sft` | `/workspace/venvs/simct-b200-sft` |
| KDFlow/SGLang/SimCT | `experiments/environments/simct-b200` | `/workspace/venvs/simct-b200` |

Do not merge these environments. LlamaFactory and SGLang/KDFlow intentionally
have different dependency constraints.

### 2.1 Portable Docker image (both isolated environments, no models)

`docker/Dockerfile.b200-cu130` packages both lock-controlled environments in a
single `linux/amd64` image without merging their dependencies:

- KDFlow/SimCT/MP-OPD: `/opt/venvs/simct-b200` (the default `PATH`);
- SFT/LlamaFactory: `/opt/venvs/simct-b200-sft` (`simct-sft` is a convenience
  link to its CLI);
- source and B200 launchers: `/opt/simct`;
- external model store: `/models`;
- external datasets: `/data`;
- external checkpoints, W&B offline files, TensorBoard events and reports:
  `/outputs`.

The `.dockerignore` rejects model/checkpoint extensions, model/data/output
directories, VCS state, virtual environments and likely credential files.
The Modal build wrapper performs a second context audit and refuses untagged or
non-Docker-Hub image references. These checks reduce accidental inclusion; the
operator must still review `git status` and the final Docker diff before push.

Build locally on an internet-connected Docker host:

```bash
set -euo pipefail
cd /workspace/SimCT
COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --short)"

docker buildx build \
  --platform linux/amd64 \
  --file docker/Dockerfile.b200-cu130 \
  --build-arg "REPO_COMMIT=$COMMIT" \
  --tag "docker.io/DOCKERHUB_NAMESPACE/simct-b200:cu130-${COMMIT:0:12}" \
  --load \
  .
```

The image build verifies both locks with `uv sync --locked`, applies the exact
LlamaFactory Python 3.12 patch, runs `uv pip check` for each environment, and
checks the CUDA 13 Torch and SGLang versions. A real B200 is still required for
the CUDA gate in section 10; building the image is not a GPU qualification.

To build and push on Modal, use the VM Sandbox wrapper. VM Sandboxes are beta
and CPU-only; they provide the real Linux kernel needed for Docker/buildx. The
resulting registry image is then qualified separately on B200.

Create the registry secret once without putting the Docker Hub write token in
shell arguments, Git, the Dockerfile or a long-lived plaintext file:

```bash
set -euo pipefail
umask 077
REGISTRY_ENV="$(mktemp)"
trap 'rm -f "$REGISTRY_ENV"; unset DOCKERHUB_USERNAME DOCKERHUB_TOKEN' EXIT

read -r -p 'Docker Hub username: ' DOCKERHUB_USERNAME
read -r -s -p 'Docker Hub write token: ' DOCKERHUB_TOKEN
printf '\n'
printf 'REGISTRY_USERNAME=%s\nREGISTRY_PASSWORD=%s\n' \
  "$DOCKERHUB_USERNAME" "$DOCKERHUB_TOKEN" >"$REGISTRY_ENV"
chmod 600 "$REGISTRY_ENV"

uvx --from modal==1.5.5 --with python-dotenv modal secret create \
  --profile no1ceboy \
  --from-dotenv "$REGISTRY_ENV" \
  simct-dockerhub-no1ceboy
```

Before starting the build, use the Modal billing snapshot/guard required by the
operator policy. Then push one immutable commit tag (replace the namespace with
the exact Docker Hub account or organization):

```bash
set -euo pipefail
cd /workspace/SimCT
COMMIT="$(git rev-parse HEAD)"
test -z "$(git status --short)"

uvx --from modal==1.5.5 modal run \
  --profile no1ceboy \
  experiments/modal/build_push_b200_dockerhub.py \
  --image "docker.io/DOCKERHUB_NAMESPACE/simct-b200:cu130-${COMMIT:0:12}" \
  --registry-secret simct-dockerhub-no1ceboy \
  --repo-commit "$COMMIT"
```

The wrapper uses BuildKit registry output with eStargz, OCI media types,
provenance and an SBOM, and logs out before terminating the Sandbox. Never use
`latest` as the only retained tag for a scientific run; record the immutable
image digest returned by the registry inspection.

Qualify that exact pushed tag on a real Modal B200 before moving it to the
company host. The registry secret is used by Modal only to pull the image; it
is not mounted into the B200 runtime:

```bash
set -euo pipefail
export SIMCT_B200_IMAGE="docker.io/DOCKERHUB_NAMESPACE/simct-b200:cu130-APPROVED_COMMIT"
export SIMCT_DOCKER_REGISTRY_SECRET="simct-dockerhub-no1ceboy"

uvx --from modal==1.5.5 modal run \
  --profile no1ceboy \
  experiments/modal/qualify_b200_dockerhub.py

unset SIMCT_B200_IMAGE SIMCT_DOCKER_REGISTRY_SECRET
```

The qualification checks both locked environments with `uv pip check`, runs
the CUDA/compute-capability gate from each interpreter, imports KDFlow,
SGLang, FlashAttention 4 and LlamaFactory, and executes a finite BF16 CUDA
matrix multiplication. It deliberately downloads no model and is a systems
qualification, not a training or scientific result.

On the company B200 host, mount internal assets rather than downloading or
baking them into the image:

```bash
docker pull "docker.io/DOCKERHUB_NAMESPACE/simct-b200:cu130-APPROVED_COMMIT"

docker run --rm -it --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e SIMCT_REQUIRE_EXTERNAL_MODELS=1 \
  -v /approved/internal/models:/models:ro \
  -v /approved/internal/datasets:/data:ro \
  -v /approved/simct-runs:/outputs \
  "docker.io/DOCKERHUB_NAMESPACE/simct-b200:cu130-APPROVED_COMMIT"
```

Inside the container, run the real-device gates before SFT or distillation:

```bash
python /opt/simct/experiments/environments/b200_gate.py
/opt/venvs/simct-b200-sft/bin/python \
  /opt/simct/experiments/environments/b200_gate.py
```

Use exact internal subpaths such as `/models/<student-revision>` and
`/models/<teacher-revision>` in the SFT/SimCT configs. Do not replace an absent
internal path with a public model silently.

## 3. Host, disk, and driver preflight

Recommended free space before starting:

- 40 GiB for the two uv environments and cache;
- model size plus one additional copy if the storage system requires staging;
- dataset size; and
- at least two complete checkpoints plus 20% headroom.

For a normal qualification workspace, reserve at least 100 GiB. Full training
may need substantially more. Put uv cache and virtual environments on the same
local filesystem; uv can then hard-link cached artifacts instead of copying
them.

```bash
set -euo pipefail

uname -a
cat /etc/os-release
nvidia-smi
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
nvcc --version
df -h /workspace
df -ih /workspace
free -h
git --version
```

Hard gates:

- NVIDIA B200 appears in `nvidia-smi`;
- the driver supports the CUDA 13 user-space stack;
- CUDA 13 `nvcc` is available, not only the driver runtime;
- Python 3.12 is available locally or through the approved internal toolchain;
- `/workspace` has enough bytes and inodes; and
- the job can write to its output and cache locations.

Do not change the host driver merely because the environment gate fails. Save
the exact gate output first and diagnose driver/runtime/toolkit mismatch.

## 4. Install and pin uv

This branch was locked with `uv==0.12.7`. On an internet-connected machine,
use the official installer or an administrator-provided package, then verify:

```bash
uv --version
test "$(uv --version | awk '{print $2}')" = "0.12.7"
```

On a restricted company host, do not pipe an external installer into the
shell. Ask the platform administrator for the approved `uv` 0.12.7 binary or
mirror it through the internal software-distribution system. Record the binary
hash:

```bash
command -v uv
sha256sum "$(command -v uv)" | tee /workspace/uv-0.12.7.sha256
uv --version
```

If Python 3.12 is already installed by the company image, prevent an accidental
download of a uv-managed interpreter:

```bash
export UV_NO_MANAGED_PYTHON=1
python3.12 --version
```

## 5. Runtime directories and non-secret path configuration

Copy the example file outside the Git worktree and customize only the copy:

```bash
install -d -m 700 /workspace/simct-b200-private
install -m 600 \
  /workspace/SimCT/experiments/environments/b200-company.env.example \
  /workspace/simct-b200-private/run.env
${EDITOR:-vi} /workspace/simct-b200-private/run.env
```

The file contains no credentials. Keep Hugging Face, package-pool, and W&B
tokens in the scheduler/secret manager rather than adding them to `run.env`.

Load the paths without printing their values:

```bash
set -a
. /workspace/simct-b200-private/run.env
set +a
install -d -m 700 \
  "$B200_ENV_ROOT" "$UV_CACHE_DIR" "$HF_HOME" \
  "$SIMCT_OUTPUT_ROOT" "$SIMCT_RUN_EVIDENCE"
```

The existing paper scripts accept a model root through `MODEL_PATH`, a dataset
root through `DATA_PATH`, and an output root through `OUTPUT_PATH`. For an
internal model store, point them at the already-mounted directories:

```bash
MODEL_PATH="$MODEL_PATH" \
DATA_PATH="$DATA_PATH" \
OUTPUT_PATH="$OUTPUT_PATH" \
  bash scripts/ctopd/qwen25_gemma2_span_mix10k_lr5e-7.sh
```

For a custom layout, bypass the assumed subdirectory names and pass exact paths
to KDFlow:

```bash
"$B200_ENV_ROOT/simct-b200/bin/python" -m kdflow.cli.train_kd_on_policy \
  --student_name_or_path "$SIMCT_SFT_CHECKPOINT_PATH" \
  --teacher_name_or_path "$SIMCT_TEACHER_MODEL_PATH" \
  --train_dataset_path "$SIMCT_TRAIN_DATASET_PATH" \
  --save_path "$SIMCT_OUTPUT_ROOT/distillation" \
  --num_nodes 1 \
  --num_gpus_per_node "$SIMCT_GPU_COUNT" \
  --backend fsdp2 \
  --bf16 True
```

That command demonstrates path injection; it is **not** a complete scientific
configuration. Start an actual run from the checked-in experiment script or an
approved, fully recorded config so loss, batch, rollout, scheduler, and stopping
rules are not silently omitted.

## 6. Verify internal models before any GPU run

Local paths avoid Hub access only if every required file is present. At
minimum, inspect each model directory for its configuration, tokenizer, and all
weight shards:

```bash
for model_dir in \
  "$SIMCT_SFT_BASE_MODEL_PATH" \
  "$SIMCT_TEACHER_MODEL_PATH" \
  "$SIMCT_SFT_CHECKPOINT_PATH"
do
  test -d "$model_dir"
  test -f "$model_dir/config.json"
  test -f "$model_dir/tokenizer_config.json"
  find "$model_dir" -maxdepth 1 -type f \
    \( -name '*.safetensors' -o -name '*.bin' \) -print -quit | grep -q .
done
```

Create a manifest in the private evidence directory. This may be slow for large
weights, but it makes the model identity auditable:

```bash
for model_dir in \
  "$SIMCT_SFT_BASE_MODEL_PATH" \
  "$SIMCT_TEACHER_MODEL_PATH" \
  "$SIMCT_SFT_CHECKPOINT_PATH"
do
  manifest="$SIMCT_RUN_EVIDENCE/$(basename "$model_dir").sha256"
  find "$model_dir" -maxdepth 1 -type f \
    \( -name '*.json' -o -name '*.model' -o -name '*.safetensors' \) \
    -print0 | sort -z | xargs -0 sha256sum > "$manifest"
done
```

If the internal mirror provides a revision file or signed manifest, compare it
with the approved model revision too. A directory name alone is not model
lineage evidence.

After verification, force Transformers/Hugging Face into local-only behavior:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
```

An offline failure should be treated as a missing-file or incompatible-config
diagnostic. Do not silently fall back to a public model with the same display
name.

## 7. Installation path A: normal outbound network

Use the committed locks unchanged:

```bash
export UV_PROJECT_ENVIRONMENT="$B200_ENV_ROOT/simct-b200-sft"
uv sync \
  --project /workspace/SimCT/experiments/environments/simct-b200-sft \
  --locked

export UV_PROJECT_ENVIRONMENT="$B200_ENV_ROOT/simct-b200"
uv sync \
  --project /workspace/SimCT/experiments/environments/simct-b200 \
  --locked
```

The SimCT lock contains both PyPI artifact records and an explicit CUDA 13
PyTorch index. Therefore the unchanged `--locked` path needs access to the
hosts recorded in the lock. Changing only `UV_DEFAULT_INDEX` does not make the
public lock an internal-pool lock.

For a slow but working route, extend the read timeout without changing package
versions:

```bash
export UV_HTTP_CONNECT_TIMEOUT=30
export UV_HTTP_TIMEOUT=600
export UV_HTTP_RETRIES=5
```

Record those transport settings in the run evidence. A timeout change is an
operational change, not a scientific configuration change.

## 8. Installation path B: internal package pool only

### 8.1 Required pool contents

Ask the pool administrator to mirror:

- all artifacts selected by both committed `uv.lock` files;
- CUDA 13 (`cu130`) Torch 2.11.0, TorchVision 0.26.0, and TorchAudio 2.11.0;
- source distributions and build dependencies where no wheel is available;
- Python 3.12 / Linux x86_64 wheels; and
- the Git checkout itself or an approved Git mirror.

The fastest audit is to give the administrator the two committed locks. A
successful package-name query is insufficient; the exact wheel for Python
3.12, Linux x86_64, and CUDA 13 must exist.

### 8.2 Create a private manifest overlay

Never edit the checked-in project in place. Make a private copy:

```bash
umask 077
install -d -m 700 /workspace/simct-b200-private/manifests
cp -a /workspace/SimCT/experiments/environments/simct-b200-sft \
  /workspace/simct-b200-private/manifests/
cp -a /workspace/SimCT/experiments/environments/simct-b200 \
  /workspace/simct-b200-private/manifests/
```

In each private `pyproject.toml`:

1. replace only the `pytorch-cu130` URL with the company's CUDA 13 mirror;
2. add a named `company-default` index for the general Python pool;
3. set `default = true` on `company-default`;
4. keep `explicit = true` on `pytorch-cu130`;
5. keep the `[tool.uv.sources]` pins for Torch/TorchVision/TorchAudio; and
6. for the SimCT project, replace the copied relative KDFlow path with the
   absolute checkout path `/workspace/SimCT`.

The resulting private-only blocks should have this shape:

```toml
[tool.uv.sources]
kdflow = { path = "/workspace/SimCT", editable = true }
torch = { index = "pytorch-cu130" }
torchvision = { index = "pytorch-cu130" }
torchaudio = { index = "pytorch-cu130" }

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://PACKAGE_POOL/PYTORCH_CU130_SIMPLE"
explicit = true

[[tool.uv.index]]
name = "company-default"
url = "https://PACKAGE_POOL/PYPI_SIMPLE"
default = true
```

The SFT project has no `kdflow` entry. Do not put a username, password, token,
or real internal hostname into the public checkout.

If authentication is required, use uv's index-name-derived environment
variables:

```bash
export UV_INDEX_COMPANY_DEFAULT_USERNAME="$COMPANY_POOL_USERNAME"
export UV_INDEX_COMPANY_DEFAULT_PASSWORD="$COMPANY_POOL_PASSWORD"
export UV_INDEX_PYTORCH_CU130_USERNAME="$COMPANY_POOL_USERNAME"
export UV_INDEX_PYTORCH_CU130_PASSWORD="$COMPANY_POOL_PASSWORD"
```

Do not run `set -x`, `env`, or `printenv` in that shell. uv documents these
credential variables in its
[HTTP/index authentication guidance](https://docs.astral.sh/uv/concepts/indexes/#authentication).

### 8.3 Resolve a separate internal lock

Start with the copied public lock so uv prefers its already-selected versions,
then resolve against the internal indexes without `--upgrade` or `--refresh`:

```bash
uv lock --project /workspace/simct-b200-private/manifests/simct-b200-sft
uv lock --project /workspace/simct-b200-private/manifests/simct-b200
```

Review version drift before syncing:

```bash
uv tree --project /workspace/SimCT/experiments/environments/simct-b200-sft \
  > /workspace/simct-b200-private/public-sft.tree
uv tree --project /workspace/simct-b200-private/manifests/simct-b200-sft \
  > /workspace/simct-b200-private/internal-sft.tree
diff -u /workspace/simct-b200-private/public-sft.tree \
  /workspace/simct-b200-private/internal-sft.tree

uv tree --project /workspace/SimCT/experiments/environments/simct-b200 \
  > /workspace/simct-b200-private/public-simct.tree
uv tree --project /workspace/simct-b200-private/manifests/simct-b200 \
  > /workspace/simct-b200-private/internal-simct.tree
diff -u /workspace/simct-b200-private/public-simct.tree \
  /workspace/simct-b200-private/internal-simct.tree
```

Expected differences are registry/source locations. Package version changes
must be reviewed explicitly. If the pool lacks an exact version, stop and ask
for that artifact to be mirrored; do not weaken a pin merely to make resolution
finish.

Install from the private locks:

```bash
export UV_PROJECT_ENVIRONMENT="$B200_ENV_ROOT/simct-b200-sft"
uv sync \
  --project /workspace/simct-b200-private/manifests/simct-b200-sft \
  --locked

export UV_PROJECT_ENVIRONMENT="$B200_ENV_ROOT/simct-b200"
uv sync \
  --project /workspace/simct-b200-private/manifests/simct-b200 \
  --locked
```

Archive the private locks and sanitized package trees in the evidence store,
not the public Git repository. They are required to reproduce the company run.

### 8.4 TLS and proxy handling

Preferred order:

1. use an HTTPS package pool with a certificate trusted by the host;
2. install the company CA in the system trust store and set
   `UV_SYSTEM_CERTS=1`;
3. provide a PEM bundle with `SSL_CERT_FILE`; or
4. only as a documented last resort, set `UV_INSECURE_HOST` for the one exact
   approved host.

```bash
export UV_SYSTEM_CERTS=1
# Or, when the company gives you a dedicated PEM bundle:
export SSL_CERT_FILE=/secure/company-ca-bundle.pem
```

Do not set global TLS-disable flags. uv warns that `UV_INSECURE_HOST` bypasses
certificate verification and exposes downloads to man-in-the-middle attacks.
The [uv certificate documentation](https://docs.astral.sh/uv/concepts/authentication/certificates/)
describes system and custom CA handling.

If the network uses an authenticated HTTP proxy, inject `HTTPS_PROXY` and
`NO_PROXY` through the scheduler. Do not save proxy credentials in Git or shell
history.

## 9. Apply the narrow LlamaFactory compatibility patch

After the SFT environment is installed:

```bash
"$B200_ENV_ROOT/simct-b200-sft/bin/python" \
  /workspace/SimCT/experiments/environments/patch_llamafactory_json_path.py
```

The patch is fail-closed: it changes exactly one known LlamaFactory 0.9.5
Python 3.12 parser pattern and stops if upstream source no longer matches.
Never copy a patched site-packages directory from a different environment.

## 10. Environment and B200 gates

```bash
export TORCH_CUDA_ARCH_LIST=10.0
export CUDAARCHS=100
export CMAKE_CUDA_ARCHITECTURES=100

"$B200_ENV_ROOT/simct-b200-sft/bin/python" \
  /workspace/SimCT/experiments/environments/b200_gate.py \
  | tee "$SIMCT_RUN_EVIDENCE/b200-gate-sft.log"

"$B200_ENV_ROOT/simct-b200/bin/python" \
  /workspace/SimCT/experiments/environments/b200_gate.py \
  | tee "$SIMCT_RUN_EVIDENCE/b200-gate-simct.log"
```

Both logs must report the exact pins, B200 compute capability 10.0, CUDA 13,
and a usable GPU. An import pass without a real CUDA tensor operation is not a
complete hardware gate.

Additional checks:

```bash
uv pip check --python "$B200_ENV_ROOT/simct-b200-sft/bin/python"
uv pip check --python "$B200_ENV_ROOT/simct-b200/bin/python"

"$B200_ENV_ROOT/simct-b200/bin/python" - <<'PY'
import torch
x = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print({"finite": bool(torch.isfinite(y).all()),
       "device": torch.cuda.get_device_name(0),
       "capability": torch.cuda.get_device_capability(0)})
PY
```

## 11. Configure SFT with an exact internal model path

LlamaFactory accepts an absolute path in `model_name_or_path`. Generate the
runtime JSON outside the repo so no company mount point is committed:

```bash
"$B200_ENV_ROOT/simct-b200-sft/bin/python" - <<'PY'
import json
import os
from pathlib import Path

required = [
    "SIMCT_SFT_BASE_MODEL_PATH",
    "SIMCT_SFT_DATASET_DIR",
    "SIMCT_OUTPUT_ROOT",
    "WANDB_PROJECT",
]
missing = [name for name in required if not os.environ.get(name)]
if missing:
    raise SystemExit("missing required path/config variables: " + ", ".join(missing))

output_root = Path(os.environ["SIMCT_OUTPUT_ROOT"]).resolve()
config = {
    "model_name_or_path": str(Path(os.environ["SIMCT_SFT_BASE_MODEL_PATH"]).resolve()),
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "full",
    "dataset_dir": str(Path(os.environ["SIMCT_SFT_DATASET_DIR"]).resolve()),
    "dataset": os.environ.get("SIMCT_SFT_DATASET_NAME", "gsm8k_sft_canary"),
    "template": os.environ.get("SIMCT_CHAT_TEMPLATE", "gemma"),
    "cutoff_len": 2048,
    "preprocessing_num_workers": 8,
    "packing": False,
    "output_dir": str(output_root / "sft-checkpoint"),
    "logging_steps": 1,
    "save_strategy": "steps",
    "save_steps": 5,
    "save_total_limit": 1,
    "save_only_model": False,
    "overwrite_output_dir": False,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 16,
    "learning_rate": 2e-6,
    "max_steps": 5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "bf16": True,
    "pure_bf16": True,
    "gradient_checkpointing": True,
    "ddp_timeout": 180000000,
    "report_to": "wandb",
    "run_name": os.environ.get("WANDB_RUN_NAME", "simct-b200-sft-5step-company"),
}
path = Path("/workspace/simct-b200-private/sft-5step.json")
path.write_text(json.dumps(config, indent=2) + "\n")
path.chmod(0o600)
print(path)
PY
```

Review the generated file and launch only after the dataset registry entry is
present in the selected `dataset_dir`:

```bash
"$B200_ENV_ROOT/simct-b200-sft/bin/llamafactory-cli" train \
  /workspace/simct-b200-private/sft-5step.json
```

For a scientific run, replace the five-step qualification values with the
paper-approved SFT config. Do not infer model quality from a five-step loss.

## 12. Configure SimCT with internal teacher and student paths

The checked-in SimCT launch scripts already use `MODEL_PATH`, `DATA_PATH`, and
`OUTPUT_PATH`. When the internal directory layout differs, make a private copy
of the selected script and change only these three arguments:

- `--student_name_or_path "$SIMCT_SFT_CHECKPOINT_PATH"`;
- `--teacher_name_or_path "$SIMCT_TEACHER_MODEL_PATH"`; and
- `--train_dataset_path "$SIMCT_TRAIN_DATASET_PATH"`.

Keep the chosen paper configuration intact: algorithm, KL direction, batch
sizes, sequence limits, rollout sampling, LR/scheduler horizon, seed, precision,
and checkpoint policy. For a ten-update qualification, add an explicit
optimizer-update stop in the private config rather than changing epoch or data
semantics.

Before launch, save a sanitized resolved command/config and hashes:

```bash
sha256sum \
  /workspace/SimCT/experiments/environments/simct-b200/pyproject.toml \
  /workspace/SimCT/experiments/environments/simct-b200/uv.lock \
  > "$SIMCT_RUN_EVIDENCE/simct-environment.sha256"

git -C /workspace/SimCT rev-parse HEAD \
  > "$SIMCT_RUN_EVIDENCE/repo.commit"
nvidia-smi -q > "$SIMCT_RUN_EVIDENCE/nvidia-smi-q.txt"
```

Do not write pool credentials, Hugging Face tokens, W&B keys, or private URLs
to the resolved config.

## 13. W&B and offline-company alternatives

When W&B is reachable, inject `WANDB_API_KEY` through the secret manager and
log at least:

- optimizer update/global step;
- total loss and each loss component;
- learning rate and gradient norm;
- prompt/completion/token counts;
- rollout and teacher latency;
- optimizer-step wall time and throughput;
- GPU memory/utilization and host memory;
- exact config, repository commit, model/dataset hashes; and
- terminal state plus checkpoint identity.

If W&B is blocked, use `WANDB_MODE=offline` and place `WANDB_DIR` under the
durable output root. Sync the resulting run only from an approved connected
machine. An offline directory is not a `finished` remote W&B run until sync and
state verification complete.

## 14. Test and qualification sequence

Run in this order and stop at the first failed gate:

1. repository commit and clean-worktree check;
2. disk/RAM/inode and driver/toolkit preflight;
3. exact uv version check;
4. package resolution and lock review;
5. package metadata check;
6. model/dataset path and hash check;
7. both B200 environment gates;
8. one CUDA BF16 matmul;
9. five-update SFT canary with checkpoint and finite metrics;
10. ten-update SimCT canary from the verified SFT checkpoint; and
11. only then, a longer numerical/scientific qualification.

Never silently retry a training failure. Preserve logs and checkpoints, state
the diagnosed cause, and give a new attempt a new run ID.

## 15. Troubleshooting matrix

| Symptom | Likely cause | Required action |
|---|---|---|
| `Could not resolve host` | DNS/outbound route blocked | use the approved internal pool or proxy; do not change pins |
| TLS/certificate failure | company CA absent from trust store | use `UV_SYSTEM_CERTS=1` or `SSL_CERT_FILE`; avoid disabling verification |
| `401`/`403` from pool | missing/incorrect index credentials | inject `UV_INDEX_<NAME>_*`; never embed credentials in URLs |
| no `cu130` candidate | pool lacks exact CUDA 13 wheel | mirror the exact wheel; do not install a CPU or different CUDA build |
| `uv.lock` wants public hosts | public lock used on internal-only network | create/review a private internal lock as in section 8 |
| package versions changed | internal re-resolution drift | stop, diff trees/locks, and mirror the missing pinned artifacts |
| `nvcc` missing | runtime-only CUDA image | install the approved CUDA 13 development toolkit/image |
| capability is not `(10, 0)` | wrong GPU allocated | stop; this is not the B200 qualification target |
| LlamaFactory JSON `Path` error | compatibility patch not applied | run the exact fail-closed patch once in the SFT environment |
| local model still contacts Hub | incomplete files or non-offline code path | verify manifest and set both offline variables; do not fallback silently |
| OOM on first step | model/batch/optimizer state exceeds device capacity | retain diagnostics; do not reduce scientific config without approval |
| W&B has no steps | training failed before first log or network blocked | inspect local trainer state/logs and W&B offline directory |

## 16. Completion checklist

An installation is complete only when all applicable boxes are true:

- [ ] exact repository commit recorded;
- [ ] public or internal lock origin clearly labelled;
- [ ] no package/model credentials stored in Git, logs, or configs;
- [ ] package versions match the approved lock;
- [ ] both metadata checks pass;
- [ ] both B200 gates pass on the allocated GPU;
- [ ] local model paths and content hashes are recorded;
- [ ] SFT reaches its intended optimizer step and saves a valid checkpoint;
- [ ] SimCT reaches its intended optimizer step with finite metrics;
- [ ] W&B is `finished`, or the offline run is retained and labelled unsynced;
- [ ] terminal logs, resolved config, telemetry, and artifact hashes are saved;
- [ ] no OOM, NaN/Inf, silent fallback, or unrecorded retry occurred; and
- [ ] systems qualification is not mislabeled as paper reproduction.
