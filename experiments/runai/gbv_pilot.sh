#!/usr/bin/env bash
# GBV-Span beta pilot: one atomic reference plus four gbv betas, one card per driver.
#
# Purpose: pick one global beta and freeze it before any primary comparison. The
# selection rule is pre-registered and lives in B200_REAL_RUN.md (kept outside this
# script on purpose): among betas whose mean diag logit-grad cosine is within 0.02 of
# the atomic reference, take the smallest mean retained_dof_fraction. This script only
# runs the pilot and records evidence; it never picks beta and never touches a test set.
#
# Placement is per card, so several drivers can share one host:
#   GPU=0 RUNS="atomic b3p0" OUT=$SH/SimCT/runs/gbv-pilot ...
#   GPU=1 RUNS="b0p1 b1p0"   OUT=$SH/SimCT/runs/gbv-pilot ...
#   GPU=2 RUNS="b0p3"        OUT=$SH/SimCT/runs/gbv-pilot ...
# They write the same OUT (the selector reads one directory) but a per-card summary and
# a per-(host,gpu) lock. One card still holds one training at a time, and an existing
# run directory is never overwritten: inspect the previous log first.
set -euo pipefail

SHARE=${SHARE:-/workspace/storage-shared/nlp/tungks}
SRC=${SRC:-$SHARE/simct-b200-portable-3998aa0}
OUT=${OUT:-$SHARE/SimCT/runs/gbv-pilot}
LOCKS=${LOCKS:-$SHARE/SimCT/runs/gbv-pilot-locks}
GPU=${GPU:-0}
UPDATES=${UPDATES:-40}
SEED=${SEED:-42}
# SEEDS turns the pilot driver into a beta x seed matrix, e.g. SEEDS="42 43 44".
SEEDS=${SEEDS:-$SEED}
PARTITION_SEED=${PARTITION_SEED:-43}
GEOMETRY=${GEOMETRY:-exact_logit}
MICRO_B=${MICRO_B:-2}
# Diagnostics sampling. EVERY=1 costs one extra autograd pass set per micro-batch but no
# extra peak memory, and it buys four times as many samples inside the selection window:
# a 20-update run at EVERY=1 contributes 160 sampled micro-batches to the mean-last-10,
# where a 40-update run at EVERY=4 contributes 40.
DIAG_EVERY=${DIAG_EVERY:-4}
# 1 keeps the mechanism telemetry (distortion/dof/logit-grad) for the whole run; measured
# wall cost at EVERY=1 is inside the 2.1 min/update already observed, so a long run can
# carry it without changing the loss or the update.
DIAGNOSTICS=${DIAGNOSTICS:-1}
# Milestones to checkpoint at; a full 312-update run needs the eight eval steps.
CHECKPOINTS=${CHECKPOINTS:-$UPDATES}
PREFIX=${PREFIX:-gbv-pilot}
RUNS=${RUNS:-"atomic b0p1 b0p3 b1p0 b3p0"}
VRAM_IDLE_MIB=${VRAM_IDLE_MIB:-2048}
export MP_SHARED_ROOT=${MP_SHARED_ROOT:-$SHARE/SimCT}

PORT=$((15000 + 1000 * GPU))
RBASE=$((23000 + 1000 * GPU))
PBASE=$((20000 + 1000 * GPU))
# Same PYTHONPATH the queue wrappers export (schedule_fixed_span5_train.sh,
# run_pending_and_eval.sh, run_single_gpu.sh): the algorithm registry imports every
# algorithm and xtoken needs the vendored aligner, so vendor must be on the path.
PYTHONPATH_VALUE="$SRC/experiments/modal/vendor:$SRC:$SRC/experiments/runai"
SUMMARY="$OUT/pilot-summary-gpu$GPU.txt"
LOCK="$LOCKS/$(hostname)-gpu-$GPU.lock"
mkdir -p "$OUT" "$LOCKS"

spec_for() {  # tag -> "mode beta"
  case "$1" in
    atomic) echo "atomic 1.0" ;;
    b0p1) echo "gbv 0.1" ;;
    b0p3) echo "gbv 0.3" ;;
    b1p0) echo "gbv 1.0" ;;
    b3p0) echo "gbv 3.0" ;;
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
  # Per-card check, not a host-wide pgrep: a driver on another card is legitimate.
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

run_one() {  # $1=mode $2=tag $3=beta $4=seed
  local mode=$1 tag=$2 beta=$3 seed=$4
  local RUN="$OUT/$PREFIX-$tag-s$seed-r1"
  local LOG="$RUN.out"
  if [ -e "$RUN" ] || [ -e "$LOG" ]; then
    echo "REFUSE $tag-s$seed: da co attempt truoc do; xem log roi quyet dinh, khong chay chong"
    return 4
  fi
  echo "RUN_START tag=$tag seed=$seed mode=$mode beta=$beta gpu=$GPU log=$LOG $(date -Is)"
  local rc=0
  (
    cd "$SRC"
    env \
      CUDA_VISIBLE_DEVICES="$GPU" \
      KDFLOW_ROLLOUT_PORT_BASE="$PORT" \
      KDFLOW_ROUTER_PORT_BASE="$RBASE" \
      KDFLOW_ROUTER_PROMETHEUS_PORT="$PBASE" \
      PYTHONPATH="$PYTHONPATH_VALUE" \
      MP_ALGORITHM=mp_opd MP_ATTN_IMPLEMENTATION=eager \
      MP_SEED="$seed" MP_PARTITION_SEED="$PARTITION_SEED" \
      MP_MAX_SPAN_LENGTH=4 MP_FIXED_SPAN_LENGTH=2 \
      MP_GBV_BETA="$beta" MP_GBV_GEOMETRY="$GEOMETRY" \
      MP_MICRO_TRAIN_BATCH_SIZE="$MICRO_B" \
      MP_OPD_DIAGNOSTICS="$DIAGNOSTICS" MP_OPD_DIAGNOSTICS_LOGIT_GRAD="$DIAGNOSTICS" \
      MP_OPD_DIAGNOSTICS_EVERY="$DIAG_EVERY" \
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

# Cheap fail-fast: the algorithm registry imports every algorithm, and xtoken needs the
# vendored aligner. Without PYTHONPATH pointing at experiments/modal/vendor this dies in
# seconds -- but only after the guard already claimed the card, which is exactly the
# mistake that cost one attempt on 2026-10-02. Check it before any run starts.
preflight_import() {
  local out
  if ! out=$(cd "$SRC" && env PYTHONPATH="$PYTHONPATH_VALUE" \
      bash experiments/runai/python-b200-host.sh -c \
      'import kdflow.algorithms; print("IMPORT_OK")' 2>&1); then
    echo "PREFLIGHT_FAIL khong import duoc kdflow.algorithms; 15 dong cuoi:"
    echo "$out" | tail -15
    return 1
  fi
  echo "PREFLIGHT_OK $(echo "$out" | tail -1) pythonpath=$PYTHONPATH_VALUE"
}

guard
if ! preflight_import; then
  echo "PILOT_STOPPED at preflight (chua chiem GPU cho run nao)" | tee -a "$SUMMARY"
  exit 3
fi

{
  echo "run start $(date -Is) host=$(hostname)"
  echo "src=$SRC gpu=$GPU updates=$UPDATES seeds=$SEEDS partition_seed=$PARTITION_SEED"
  echo "geometry=$GEOMETRY micro_B=$MICRO_B diagnostics=$DIAGNOSTICS every=$DIAG_EVERY"
  echo "prefix=$PREFIX checkpoints=$CHECKPOINTS runs=$RUNS"
  echo "pythonpath=$PYTHONPATH_VALUE"
} | tee -a "$SUMMARY"

for seed in $SEEDS; do
  for tag in $RUNS; do
    spec=$(spec_for "$tag")
    if [ -z "$spec" ]; then
      echo "STOPPED unknown tag '$tag'" | tee -a "$SUMMARY"
      exit 2
    fi
    set -- $spec
    if ! run_one "$1" "$tag" "$2" "$seed"; then
      echo "STOPPED at tag=$tag seed=$seed" | tee -a "$SUMMARY"
      exit 1
    fi
  done
done
echo "ALL_DONE gpu=$GPU seeds=$SEEDS $(date -Is)" | tee -a "$SUMMARY"
