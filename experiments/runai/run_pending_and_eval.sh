#!/usr/bin/env bash
# Run the pending ladder runs of one case on one GPU, then evaluate what completed.
#   run_pending_and_eval.sh <case> <gpu> <map> <id...>
# Guard note: a ports-only check is NOT enough -- a sibling run's engine sleeps and releases its
# ports, so a new run can start and then collide when the sleeping engine wakes (hit twice on
# 2026-10-01 at 12:47 and 13:06). Wait for the CARD to be idle: no other training driver on this
# card, card memory near zero, and the slot port family free.
set -uo pipefail
CASE="$1"; GPU="$2"; MAP="$3"; shift 3
IDS="$*"
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$SRC/experiments/modal/vendor:$SRC:$SRC/experiments/runai"
PY=/usr/bin/python3.12
LAD="$SRC/experiments/runai/queue_fixed_span_ladder.py"

card_idle() {
  local p v used
  for p in $(pgrep -f 'run_single_gpu\.py fixed' 2>/dev/null); do
    v=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')
    case ",$v," in *",$GPU,"*) return 1;; esac
  done
  used=$(nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  { [ -n "$used" ] && [ "$used" -lt 2000 ]; } || return 1
  ! ss -ltn 2>/dev/null | grep -qE ':(15000|15001|15002|20000|20001|20002|23000|16000|16001|16002|21000|24000)\b'
}

completed() {
  python3 - "$CASE/train/$1/checkpoint/run-summary.json" <<'PY'
import json,sys,pathlib
try: d=json.loads(pathlib.Path(sys.argv[1]).read_text())
except Exception: raise SystemExit(1)
raise SystemExit(0 if d.get("status")=="completed" and d.get("optimizer_updates")==312 else 1)
PY
}

echo "RUN_PENDING start gpu=$GPU case=$CASE ids=[$IDS] $(date -Is)"
for id in $IDS; do
  if completed "$id"; then echo "SKIP $id (da completed/312)"; continue; fi
  until card_idle; do echo "WAIT_CARD gpu=$GPU $(date -Is)"; sleep 120; done
  echo "START $id map=$MAP $(date -Is)"
  env MP_LADDER_SLOTS="$MAP" MP_OPD_DIAGNOSTICS=1 MP_OPD_DIAGNOSTICS_LOGIT_GRAD=0 \
      MP_OPD_DIAGNOSTICS_EVERY=1 $PY "$LAD" train-one --case "$CASE" --id "$id"
  echo "END $id rc=$? $(date -Is)"
  sleep 30
done
echo "RUN_PENDING eval $(date -Is)"
bash "$CASE/eval_after_card.sh" "$GPU" $IDS
echo "RUN_PENDING xong $(date -Is)"
