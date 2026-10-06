#!/usr/bin/env bash
# DPCA pilot on the mp_opd (gbv / soft-alternating) training path: one atomic
# reference plus one dpca arm, one card per driver.
#
# Purpose: qualify the DPCA objective inside the campaign's own trainer before any
# comparison is drawn. This is implementation_validation, not efficacy: the arm is
# compared against `atomic` on identical data and seed, not against paper numbers.
#
# Placement is per card, so several drivers can share one host:
#   GPU=0 RUNS="atomic dpca" OUT=$SHARE/SimCT/runs/dpca-pilot ...
# They write the same OUT (the reader takes one directory) but a per-card summary and
# a per-(host,gpu) lock. One card holds one training at a time, and an existing run
# directory is never overwritten: inspect the previous log first.
set -euo pipefail

SHARE=${SHARE:-/workspace/storage-shared/nlp/tungks}
SRC=${SRC:-$SHARE/simct-b200-portable-55a7461e}
OUT=${OUT:-$SHARE/SimCT/runs/dpca-pilot}
LOCKS=${LOCKS:-$SHARE/SimCT/runs/dpca-pilot-locks}
GPU=${GPU:-0}
UPDATES=${UPDATES:-40}
SEED=${SEED:-42}
SEEDS=$(printf '%s' "${SEEDS:-$SEED}" | tr ',' ' ')
PARTITION_SEED=${PARTITION_SEED:-43}
# Chunk width. DPCA's semantic prior sums L_T and L_S inside one synchronized chunk,
# and here a chunk IS an atom, so this is the chunk width in student tokens. 1 gives
# per-token chunks and matches the `atomic` reference the arm is read against.
MAX_SPAN_LENGTH=${MAX_SPAN_LENGTH:-1}
MICRO_B=${MICRO_B:-2}
DIAGNOSTICS=${DIAGNOSTICS:-0}
CHECKPOINTS=${CHECKPOINTS:-$UPDATES}
PREFIX=${PREFIX:-dpca-pilot}
RUNS=${RUNS:-"atomic dpca"}
VRAM_IDLE_MIB=${VRAM_IDLE_MIB:-2048}
export MP_SHARED_ROOT=${MP_SHARED_ROOT:-$SHARE/SimCT}

# DPCA paper hyper-parameters. Kept as one place so a run's config is readable from
# the summary without parsing the trainer log.
DPCA_CLIP_LOW=${DPCA_CLIP_LOW:-0.2}
DPCA_CLIP_HIGH=${DPCA_CLIP_HIGH:-0.28}
DPCA_CLIP_C=${DPCA_CLIP_C:-10.0}
DPCA_ADV_CLAMP=${DPCA_ADV_CLAMP:-10.0}
DPCA_AGG=${DPCA_AGG:-token-mean}

# The portable runtime ships a libstdc++ that stops at GLIBCXX_3.4.30, while SGLang
# 0.5.11 JIT kernels need 3.4.32. The repo wrapper puts portable-libs first on
# LD_LIBRARY_PATH, so the loader picks the old one and CUDA-graph capture dies with
# "version GLIBCXX_3.4.32 not found". Preloading the system libstdc++ fixes it
# without touching the shared wrapper, which every other job on every node uses.
# The preflight below refuses to run rather than rediscovering this mid-run.
ABI_LD_PRELOAD=${ABI_LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}

PORT=$((15000 + 1000 * GPU))
RBASE=$((23000 + 1000 * GPU))
PBASE=$((20000 + 1000 * GPU))
PYTHONPATH_VALUE="$SRC/experiments/modal/vendor:$SRC:$SRC/experiments/runai"
SUMMARY="$OUT/pilot-summary-gpu$GPU.txt"
LOCK="$LOCKS/$(hostname)-gpu-$GPU.lock"
mkdir -p "$OUT" "$LOCKS"

spec_for() {  # tag -> mode
  case "$1" in
    atomic) echo "atomic" ;;
    dpca) echo "dpca" ;;
    gbv) echo "gbv" ;;
    *) echo "" ;;
  esac
}

release_lock() { [ -n "${LOCK_HELD:-}" ] && rm -f "$LOCK"; }
trap release_lock EXIT

guard() {
  local used apps pid
  if ! used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' '); then
    echo "GUARD_FAIL khong doc duoc nvidia-smi cho gpu=$GPU"
    exit 3
  fi
  case "$used" in
    '' | *[!0-9]*)
      echo "GUARD_FAIL nvidia-smi tra ve gia tri khong phai so: '$used'"
      exit 3
      ;;
  esac
  if [ "$used" -gt "$VRAM_IDLE_MIB" ]; then
    echo "GUARD_FAIL gpu=$GPU used=${used}MiB: card khong ranh"
    exit 3
  fi
  apps=$(nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  if [ -n "$apps" ]; then
    echo "GUARD_FAIL gpu=$GPU dang co tien trinh: $apps"
    exit 3
  fi
  if [ -e "$LOCK" ]; then
    pid=$(cat "$LOCK" 2>/dev/null || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "GUARD_FAIL lock $LOCK dang giu boi pid=$pid"
      exit 3
    fi
    echo "GUARD_NOTE lock cu (pid=${pid:-?} khong con song), tiep tuc"
    rm -f "$LOCK"
  fi
  echo "GUARD_OK gpu=$GPU used=${used}MiB ports=$PORT/$RBASE/$PBASE lock=$LOCK"
  echo $$ > "$LOCK"
  LOCK_HELD=1
}

preflight_abi() {
  if [ ! -e "$ABI_LD_PRELOAD" ]; then
    echo "PREFLIGHT_FAIL ABI_LD_PRELOAD khong ton tai: $ABI_LD_PRELOAD"
    return 1
  fi
  # `strings` is used rather than glibcxx_ver: S1 on this node class proved the
  # versioned symbols are readable this way, and a missing tool would otherwise turn
  # the guard into a silent pass.
  #
  # The output is captured before it is matched on purpose. Under `pipefail`, a
  # `grep -q` that exits on the first hit leaves `strings` killed by SIGPIPE (141),
  # so the pipeline reports failure for a library that does export the symbol --
  # the guard refused a valid run on this node.
  ABI_SYMS=$(strings -a "$ABI_LD_PRELOAD" 2>/dev/null | grep -o '^GLIBCXX_3\.4\.3[0-9]$' | tr '\n' ' ')
  case " $ABI_SYMS " in
    *" GLIBCXX_3.4.32 "*)
      echo "PREFLIGHT_OK abi=$ABI_LD_PRELOAD provides GLIBCXX_3.4.32"
      ;;
    *)
      echo "PREFLIGHT_FAIL $ABI_LD_PRELOAD khong xuat GLIBCXX_3.4.32; SGLang se fail khi capture cuda graph"
      echo "PREFLIGHT_FAIL symbols=[${ABI_SYMS:-none}]"
      return 1
      ;;
  esac
}

run_one() {  # $1=mode $2=tag $3=seed
  local mode=$1 tag=$2 seed=$3
  local RUN="$OUT/$PREFIX-$tag-s$seed-r1"
  local LOG="$RUN.out"
  if [ -e "$RUN" ] || [ -e "$LOG" ]; then
    echo "REFUSE $tag-s$seed: da co attempt truoc do; xem log roi quyet dinh, khong chay chong"
    return 4
  fi
  echo "RUN_START tag=$tag seed=$seed mode=$mode gpu=$GPU log=$LOG $(date -Is)"
  local rc=0
  (
    cd "$SRC"
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      KDFLOW_ROLLOUT_PORT_BASE="$PORT" \
      KDFLOW_ROUTER_PORT_BASE="$RBASE" \
      KDFLOW_ROUTER_PROMETHEUS_PORT="$PBASE" \
      PYTHONPATH="$PYTHONPATH_VALUE" \
      LD_PRELOAD="$ABI_LD_PRELOAD" \
      MP_ALGORITHM=mp_opd MP_ATTN_IMPLEMENTATION=eager \
      MP_SEED="$seed" MP_PARTITION_SEED="$PARTITION_SEED" \
      MP_MAX_SPAN_LENGTH="$MAX_SPAN_LENGTH" MP_FIXED_SPAN_LENGTH=2 \
      MP_DPCA_CLIP_RATIO_LOW="$DPCA_CLIP_LOW" \
      MP_DPCA_CLIP_RATIO_HIGH="$DPCA_CLIP_HIGH" \
      MP_DPCA_CLIP_RATIO_C="$DPCA_CLIP_C" \
      MP_DPCA_ADV_CLAMP="$DPCA_ADV_CLAMP" \
      MP_DPCA_AGG="$DPCA_AGG" \
      MP_MICRO_TRAIN_BATCH_SIZE="$MICRO_B" \
      MP_OPD_DIAGNOSTICS="$DIAGNOSTICS" \
      MP_CHECKPOINT_STEPS="$CHECKPOINTS" MP_RESUME=0 MP_PREFLIGHT_ONLY=0 \
      MP_SOURCE_COMMIT="$(git rev-parse HEAD)" \
      MP_SOURCE_DIRTY="$(git status --porcelain --untracked-files=no | tr '\n' ';')" \
      MP_RAY_TMP="/tmp/$PREFIX-gpu$GPU-$tag-s$seed" \
      bash experiments/runai/python-b200-host.sh experiments/runai/run_single_gpu.py \
        "$mode" "$UPDATES" "$RUN"
  ) >>"$LOG" 2>&1 || rc=$?
  echo "RUN_EXIT tag=$tag seed=$seed rc=$rc $(date -Is)"
  if [ "$rc" -ne 0 ] || ! grep -q 'RUN_VERIFIED' "$LOG"; then
    echo "RUN_FAIL tag=$tag seed=$seed rc=$rc (khong thay RUN_VERIFIED); dung, khong retry mu"
    return 1
  fi
  echo "RUN_DONE $tag seed=$seed rc=0"
}

# The algorithm registry imports every algorithm, and xtoken needs the vendored
# aligner. Without PYTHONPATH pointing at experiments/modal/vendor this dies in
# seconds -- but only after the guard already claimed the card, which is exactly the
# mistake that cost one attempt on 2026-10-02. Check it before any run starts.
preflight_import() {
  local out
  if ! out=$(cd "$SRC" && env PYTHONPATH="$PYTHONPATH_VALUE" LD_PRELOAD="$ABI_LD_PRELOAD" \
      bash experiments/runai/python-b200-host.sh -c \
      'import kdflow.algorithms; print("IMPORT_OK")' 2>&1); then
    echo "PREFLIGHT_FAIL khong import duoc kdflow.algorithms; 15 dong cuoi:"
    echo "$out" | tail -15
    return 1
  fi
  echo "PREFLIGHT_OK $(echo "$out" | tail -1) pythonpath=$PYTHONPATH_VALUE"
}

preflight_abi
if ! preflight_import; then
  echo "PILOT_STOPPED at preflight (chua chiem GPU cho run nao)" | tee -a "$SUMMARY"
  exit 3
fi
guard

{
  echo "run start $(date -Is) host=$(hostname)"
  echo "src=$SRC gpu=$GPU updates=$UPDATES seeds=$SEEDS partition_seed=$PARTITION_SEED"
  echo "max_span_length=$MAX_SPAN_LENGTH micro_B=$MICRO_B diagnostics=$DIAGNOSTICS"
  echo "dpca clip_low=$DPCA_CLIP_LOW clip_high=$DPCA_CLIP_HIGH clip_c=$DPCA_CLIP_C adv_clamp=$DPCA_ADV_CLAMP agg=$DPCA_AGG"
  echo "prefix=$PREFIX checkpoints=$CHECKPOINTS runs=$RUNS"
  echo "abi_ld_preload=$ABI_LD_PRELOAD"
  echo "pythonpath=$PYTHONPATH_VALUE"
} | tee -a "$SUMMARY"

for seed in $SEEDS; do
  for tag in $RUNS; do
    mode=$(spec_for "$tag")
    if [ -z "$mode" ]; then
      echo "STOPPED unknown tag '$tag'" | tee -a "$SUMMARY"
      exit 2
    fi
    if ! run_one "$mode" "$tag" "$seed"; then
      echo "STOPPED at tag=$tag seed=$seed" | tee -a "$SUMMARY"
      exit 1
    fi
  done
done
echo "ALL_DONE gpu=$GPU seeds=$SEEDS $(date -Is)" | tee -a "$SUMMARY"