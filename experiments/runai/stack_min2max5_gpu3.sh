#!/usr/bin/env bash
# Stack train min2max5 (RND-random5) 3 seed tren GPU 3, roi plan theo tung arm + add job eval.
#   bash stack_min2max5_v2.sh <case>
set -uo pipefail
CASE=${1:?usage: stack_min2max5_v2.sh <case>}
SRC=/workspace/storage-shared/nlp/tungks/simct-b200-portable-549b1ae
PY=/usr/bin/python3.12
LAD=$SRC/experiments/runai/queue_random_span_ladder.py
EC=/workspace/storage-shared/nlp/tungks/job-manager-eval-core/bin/ec
export EC_ROOT=/workspace/storage-shared/nlp/tungks/eval-core
export PYTHONPATH=$SRC/experiments/modal/vendor:$SRC:$SRC/experiments/runai
ARMS="RND-random5-s42 RND-random5-s43 RND-random5-s44"
SLOTS=4,5,6,3
STEPS=40,80,120,156,200,240,280,312
GPU=3

echo "=== 0. guard ==="
[ -f "$CASE/campaign.json" ] || { echo "THIEU campaign.json: $CASE"; exit 1; }
[ -f "$EC" ] || { echo "THIEU ec: $EC"; exit 1; }
FREE=$(nvidia-smi --id=$GPU --query-gpu=memory.used --format=csv,noheader,nounits | tr -d " ")
[ ${FREE:-99999} -lt 2000 ] || { echo "GPU $GPU dang dung ${FREE} MiB"; exit 1; }
$PY -c 'import json,sys;d=json.load(open(sys.argv[1]));print("  student:",d["student"]);print("  teacher:",d["teacher"]);print("  dataset:",d["dataset"])' "$CASE/campaign.json"
echo "  GPU $GPU: ${FREE} MiB"

for id in $ARMS; do
  echo "=== TRAIN $id $(date -Is)"
  env MP_LADDER_SLOTS=$SLOTS MP_OPD_DIAGNOSTICS=1 MP_OPD_DIAGNOSTICS_LOGIT_GRAD=0 MP_OPD_DIAGNOSTICS_EVERY=1 \
      $PY "$LAD" train-one --case "$CASE" --id $id
  rc=$?
  echo "=== TRAIN xong $id rc=$rc $(date -Is)"
  if [ $rc -ne 0 ]; then echo "DUNG stack: $id loi"; break; fi
  echo "=== PLAN cho $id"
  $PY /tmp/plan_arms.py "$CASE" $id
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