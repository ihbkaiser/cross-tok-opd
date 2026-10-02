#!/usr/bin/env bash
# GBV-Span beta pilot: one atomic reference plus four gbv betas, sequential, one card.
#
# Purpose: pick one global beta and freeze it before any primary comparison. The
# selection rule is pre-registered and lives in B200_REAL_RUN.md (kept outside this
# script on purpose): among betas whose mean diag logit-grad cosine is within 0.02 of
# the atomic reference, take the smallest mean retained_dof_fraction. This script only
# runs the pilot and records evidence; it never picks beta and never touches a test set.
#
# Every long run is launched inside this script, which the operator starts with
# `setsid nohup`. One card, one training at a time, no duplicate attempt on an existing
# run directory: a re-run must inspect the previous log first.
set -euo pipefail

SHARE=${SHARE:-/workspace/storage-shared/nlp/tungks}
SRC=${SRC:-$SHARE/simct-b200-portable-10c7c25}
OUT=${OUT:-$SHARE/SimCT/runs/gbv-pilot}
GPU=${GPU:-2}
UPDATES=${UPDATES:-40}
SEED=${SEED:-42}
PARTITION_SEED=${PARTITION_SEED:-43}
GEOMETRY=${GEOMETRY:-exact_logit}
MICRO_B=${MICRO_B:-2}
BETAS=${BETAS:-"0.1 0.3 1.0 3.0"}
export MP_SHARED_ROOT=${MP_SHARED_ROOT:-$SHARE/SimCT}

PORT=$((15000 + 1000 * GPU))
RBASE=$((23000 + 1000 * GPU))
PBASE=$((20000 + 1000 * GPU))
# Same PYTHONPATH the queue wrappers export (schedule_fixed_span5_train.sh,
# run_pending_and_eval.sh, run_single_gpu.sh): the algorithm registry imports every
# algorithm and xtoken needs the vendored aligner, so vendor must be on the path.
PYTHONPATH_VALUE="$SRC/experiments/modal/vendor:$SRC:$SRC/experiments/runai"
mkdir -p "$OUT"

tag_of() {
  case "$1" in
    0.1) echo b0p1 ;;
    0.3) echo b0p3 ;;
    1.0) echo b1p0 ;;
    3.0) echo b3p0 ;;
    *) echo "b$(echo "$1" | tr -d '.')" ;;
  esac
}

guard() {
  local used
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
  if [ "$used" -gt 2048 ]; then
    echo "GUARD_FAIL gpu=$GPU used=${used}MiB: card khong ranh"
    exit 3
  fi
  if pgrep -af 'run_single_gpu.py' >/dev/null 2>&1; then
    echo "GUARD_FAIL da co run_single_gpu.py khac dang chay"
    exit 3
  fi
  if ss -ltn 2>/dev/null | grep -qE ":($PORT|$RBASE|$PBASE)\b"; then
    echo "GUARD_FAIL cong $PORT/$RBASE/$PBASE dang bi chiem"
    exit 3
  fi
  echo "GUARD_OK gpu=$GPU used=${used}MiB ports=$PORT/$RBASE/$PBASE"
}

run_one() {  # $1=mode $2=tag $3=beta
  local mode=$1 tag=$2 beta=$3
  local RUN="$OUT/gbv-pilot-$tag-s$SEED-r1"
  local LOG="$RUN.out"
  if [ -e "$RUN" ] || [ -e "$LOG" ]; then
    echo "REFUSE $tag: da co attempt truoc do; xem log roi quyet dinh, khong chay chong"
    return 4
  fi
  echo "RUN_START tag=$tag mode=$mode beta=$beta log=$LOG $(date -Is)"
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
      MP_SEED="$SEED" MP_PARTITION_SEED="$PARTITION_SEED" \
      MP_MAX_SPAN_LENGTH=4 MP_FIXED_SPAN_LENGTH=2 \
      MP_GBV_BETA="$beta" MP_GBV_GEOMETRY="$GEOMETRY" \
      MP_MICRO_TRAIN_BATCH_SIZE="$MICRO_B" \
      MP_OPD_DIAGNOSTICS=1 MP_OPD_DIAGNOSTICS_LOGIT_GRAD=1 MP_OPD_DIAGNOSTICS_EVERY=4 \
      MP_CHECKPOINT_STEPS="$UPDATES" MP_RESUME=0 MP_PREFLIGHT_ONLY=0 \
      MP_SOURCE_COMMIT="$(git rev-parse HEAD)" \
      MP_SOURCE_DIRTY="$(git status --porcelain --untracked-files=no | tr '\n' ';')" \
      MP_RAY_TMP="/tmp/gbv-pilot-$tag" \
      bash experiments/runai/python-b200-host.sh experiments/runai/run_single_gpu.py \
        "$mode" "$UPDATES" "$RUN"
  ) >>"$LOG" 2>&1 || rc=$?
  echo "RUN_EXIT tag=$tag rc=$rc $(date -Is)"
  if [ "$rc" -ne 0 ] || ! grep -q 'RUN_VERIFIED' "$LOG"; then
    echo "RUN_FAIL tag=$tag rc=$rc (khong thay RUN_VERIFIED); dung pilot, khong retry mu"
    return 1
  fi
  echo "PILOT_RUN_DONE $tag rc=0"
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
  echo "PILOT_STOPPED at preflight (chua chiem GPU cho run nao)" | tee "$OUT/pilot-summary.txt"
  exit 3
fi

{
  echo "pilot start $(date -Is)"
  echo "src=$SRC gpu=$GPU updates=$UPDATES seed=$SEED partition_seed=$PARTITION_SEED"
  echo "geometry=$GEOMETRY micro_B=$MICRO_B betas=$BETAS"
  echo "pythonpath=$PYTHONPATH_VALUE"
} | tee -a "$OUT/pilot-summary.txt"

if ! run_one atomic atomic 1.0; then
  echo "PILOT_STOPPED at atomic" | tee -a "$OUT/pilot-summary.txt"
  exit 1
fi
for b in $BETAS; do
  if ! run_one gbv "$(tag_of "$b")" "$b"; then
    echo "PILOT_STOPPED at beta=$b" | tee -a "$OUT/pilot-summary.txt"
    exit 1
  fi
done
echo "PILOT_DONE $(date -Is)" | tee -a "$OUT/pilot-summary.txt"
