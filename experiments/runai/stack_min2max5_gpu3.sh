#!/usr/bin/env bash
# Stack train min2max5 (RND-random5) 3 seed tren GPU 3, roi plan theo tung arm + add job eval.
#   bash stack_min2max5_gpu3.sh <case>
#   SRC=<clone khac> bash stack_min2max5_gpu3.sh <case>
#
# Guard theo TUNG arm: mot card = mot training, va arm truoc phai nha sach VRAM truoc khi arm
# sau vao. Engine SGLang nha cong khi ngu nhung van giu VRAM, nen chi kiem cong/thoi diem bat
# dau la khong du: seed 43 da chet o buoc khoi tao teacher engine vi seed 42 con giu ~100 GB.
set -uo pipefail
CASE=${1:?usage: bash stack_min2max5_gpu3.sh <case>}
SRC=${SRC:-/workspace/storage-shared/nlp/tungks/simct-b200-portable-549b1ae}
PY=/usr/bin/python3.12
LAD=$SRC/experiments/runai/queue_random_span_ladder.py
# plan_arms_subset.py nam trong repo (khong phu thuoc /tmp bi xoa khi reboot).
PLAN=$SRC/experiments/runai/plan_arms_subset.py
[ -f "$PLAN" ] || PLAN=/tmp/plan_arms.py
EC=/workspace/storage-shared/nlp/tungks/job-manager-eval-core/bin/ec
export EC_ROOT=/workspace/storage-shared/nlp/tungks/eval-core
export PYTHONPATH=$SRC/experiments/modal/vendor:$SRC:$SRC/experiments/runai
ARMS="RND-random5-s42 RND-random5-s43 RND-random5-s44"
SLOTS=4,5,6,3
STEPS=40,80,120,156,200,240,280,312
GPU=3
VRAM_IDLE_MIB=2000
TEARDOWN_WAIT=180   # giay cho arm truoc nha VRAM truoc khi ket luan la con tan du

vram_used() { nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " "; }

# Tien trinh nao dang giu card nay: pgrep thay ca engine dang ngu (nha cong nhung giu VRAM) ma
# nvidia-smi --query-compute-apps bo sot.
card_holders() {
  local p cvd
  for p in $(pgrep -f 'sglang|run_single_gpu.py|vllm' 2>/dev/null); do
    cvd=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')
    case ",${cvd}," in
      *",$GPU,"*) printf '%s cuda=%s %s\n' "$p" "${cvd:-?}" "$(ps -o cmd= -p "$p" 2>/dev/null | cut -c1-90)" ;;
    esac
  done
}

card_clean() { [ "$(vram_used)" -lt "$VRAM_IDLE_MIB" ] && [ -z "$(card_holders)" ]; }

require_clean_card() {   # $1 = moc kiem, chi de doc log
  card_clean && return 0
  echo "  CARD KHONG SACH ($1): VRAM $(vram_used) MiB / nguong $VRAM_IDLE_MIB MiB"
  card_holders | sed 's/^/    /'
  echo "  (in 'ps -o pid,ppid,lstart,cmd -p <PID>' roi kill TUNG PID: con sglang truoc, cha sau)"
  return 1
}

echo "=== 0. guard ==="
[ -d "$SRC" ] || { echo "THIEU SRC: $SRC"; exit 1; }
[ -f "$CASE/campaign.json" ] || { echo "THIEU campaign.json: $CASE"; exit 1; }
[ -f "$PLAN" ] || { echo "THIEU plan_arms (plan_arms_subset.py / /tmp/plan_arms.py)"; exit 1; }
[ -f "$EC" ] || { echo "THIEU ec: $EC"; exit 1; }
[ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ] || { echo "SRC co thay doi chua commit"; exit 1; }
echo "  SRC=$SRC"
git -C "$SRC" log --oneline -1 | sed 's/^/    HEAD /'
require_clean_card "truoc khi bat dau stack" || { echo "DUNG stack: card $GPU dang ban"; exit 2; }
$PY -c 'import json,sys;d=json.load(open(sys.argv[1]));print("  student:",d["student"]);print("  teacher:",d["teacher"]);print("  dataset:",d["dataset"])' "$CASE/campaign.json"
echo "  GPU $GPU: $(vram_used) MiB"

for id in $ARMS; do
  echo "=== GUARD truoc $id $(date -Is)"
  require_clean_card "truoc $id" || { echo "DUNG stack: card $GPU khong sach truoc $id"; exit 3; }

  echo "=== TRAIN $id $(date -Is)"
  env MP_LADDER_SLOTS=$SLOTS MP_OPD_DIAGNOSTICS=1 MP_OPD_DIAGNOSTICS_LOGIT_GRAD=0 MP_OPD_DIAGNOSTICS_EVERY=1 \
      $PY "$LAD" train-one --case "$CASE" --id $id
  rc=$?
  echo "=== TRAIN xong $id rc=$rc $(date -Is)"
  if [ $rc -ne 0 ]; then echo "DUNG stack: $id loi"; break; fi

  echo "=== TEARDOWN check sau $id (cho toi $TEARDOWN_WAIT s)"
  for _ in $(seq 1 $((TEARDOWN_WAIT / 5))); do card_clean && break; sleep 5; done
  require_clean_card "sau $id" || { echo "DUNG stack: $id de lai VRAM/tien trinh tren card $GPU"; exit 4; }

  echo "=== PLAN cho $id"
  $PY "$PLAN" "$CASE" $id
  echo "=== ADD job eval $id"
  python3 "$EC" add --case "$CASE" --run $id --bench all --seeds 42,43,44 --steps $STEPS
done

echo "=== status ==="
python3 "$EC" status | tail -6
echo "=== bat eval gen tren GPU $GPU + cham CPU ==="
setsid nohup python3 "$EC" run --type gen --gpu $GPU >> "$EC_ROOT/run/gen_gpu$GPU.out" 2>&1 < /dev/null &
setsid nohup python3 "$EC" run --type score-cpu   >> "$EC_ROOT/run/score.out" 2>&1 < /dev/null &
sleep 15; tail -3 "$EC_ROOT/run/gen_gpu$GPU.out" 2>/dev/null
echo "=== XONG $(date -Is)"
