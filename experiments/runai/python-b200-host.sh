#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="${MP_RUNTIME_DIR:-/workspace/storage-shared/nlp/tungks/MP-OPD/work-QA9lLufL}"
test -d "$TASK_DIR/runtime-host-libs"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_HOME="/usr/local/cuda-13.0"
export PATH="/opt/venvs/simct-b200/bin:$CUDA_HOME/bin:$PATH"

NVIDIA_LIBS="$(find \
  /opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia \
  -type d -name lib -printf '%p:')"

export LD_LIBRARY_PATH="$TASK_DIR/runtime-host-libs:/opt/simct-portable-libs:$CUDA_HOME/lib64:${NVIDIA_LIBS}${LD_LIBRARY_PATH:-}"
export KDFLOW_TRUST_REMOTE_CODE=0
export TOKENIZERS_PARALLELISM=false

# MP-OPD: NVRTC headers and isolated FlashInfer cache
NVRTC_INCLUDE="/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia/cu13/include"
test -f "$NVRTC_INCLUDE/nvrtc.h"
export CPATH="$NVRTC_INCLUDE${CPATH:+:$CPATH}"
export FLASHINFER_WORKSPACE_BASE="${XDG_CACHE_HOME:-$TASK_DIR/.cache}"

exec /opt/venvs/simct-b200/bin/python "$@"
