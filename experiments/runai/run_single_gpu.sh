#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
GPU="${1:?Usage: run_single_gpu.sh GPU atomic|fixed|random|soft LIMIT_0_TO_312}"
MODE="${2:?Missing mode}"
LIMIT="${3:?Missing update limit}"
[[ "$GPU" =~ ^[0-9]+$ ]] && (( GPU <= 7 )) || { echo "GPU must be a numeric slot 0..7" >&2; exit 2; }
[[ "$MODE" == atomic || "$MODE" == fixed || "$MODE" == random || "$MODE" == soft ]] || { echo "Mode must be atomic, fixed, random or soft" >&2; exit 2; }
[[ "$LIMIT" =~ ^[0-9]+$ ]] && (( 10#$LIMIT <= 312 )) || { echo "Limit must be an integer 0..312 (0 means full)" >&2; exit 2; }

ALGORITHM="${MP_ALGORITHM:-mp_opd}"
[[ "$ALGORITHM" == mp_opd || "$ALGORITHM" == span_ctkd || "$ALGORITHM" == xtoken ]] || { echo "Unsupported MP_ALGORITHM" >&2; exit 2; }
# The generic job manager assigns physical UUIDs. Keep its lease mapping;
# GPU argument is still used for host port separation and output labels.
if [[ -z "${JM_JOB_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
else
  [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != *,* ]] || { echo "Expected one leased GPU" >&2; exit 2; }
fi
export PYTHONPATH="$CODE_ROOT/experiments/modal/vendor:$CODE_ROOT"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export KDFLOW_TRUST_REMOTE_CODE=0 TOKENIZERS_PARALLELISM=false
export RAY_USAGE_STATS_ENABLED=0 NCCL_CUMEM_HOST_ENABLE=0
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export KDFLOW_ROLLOUT_PORT_BASE="$((15000 + 1000 * GPU))"
export KDFLOW_ROUTER_PROMETHEUS_PORT="$((20000 + 1000 * GPU))"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset RAY_ADDRESS RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES

TASK_CACHE="$(mktemp -d "/tmp/mp-${MODE}-XXXXXXXX")"
export XDG_CACHE_HOME="$TASK_CACHE"
export TRITON_CACHE_DIR="$TASK_CACHE/triton"
export TORCH_EXTENSIONS_DIR="$TASK_CACHE/torch"
export MP_RAY_TMP="$TASK_CACHE/ray"
export MP_SOURCE_COMMIT="$(git -C "$CODE_ROOT" rev-parse HEAD)"
export MP_SOURCE_DIRTY="$(git -C "$CODE_ROOT" status --porcelain --untracked-files=no)"
RUN_ROOT="${MP_RUN_ROOT:-$(dirname "$CODE_ROOT")/simct-runs}"
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd -- "$RUN_ROOT" && pwd)"
RUN_DIR="$RUN_ROOT/qwen-gemma-${ALGORITHM}-${MODE}-gpu${GPU}-limit${LIMIT}-$(date +%Y%m%d-%H%M%S)-$$"
echo "RUN_DIR=$RUN_DIR"
cd "$CODE_ROOT"

set +e
bash "$SCRIPT_DIR/python-b200-host.sh" "$SCRIPT_DIR/run_single_gpu.py"   "$MODE" "$LIMIT" "$RUN_DIR" > "${RUN_DIR}.log" 2>&1
RC=$?
printf '%s\n' "$RC" > "${RUN_DIR}.exitcode"
echo "EXIT_CODE=$RC LOG=${RUN_DIR}.log"
exit "$RC"
