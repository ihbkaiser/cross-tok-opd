#!/usr/bin/env bash
# Run the pending ladder runs of one case on one GPU, then evaluate what completed.
#   run_pending_and_eval.sh <case> <gpu> <map> <id...>
# Design note: this lives in the repo (shipped via the HF bundle) so the node never needs
# pasted heredocs. Every step is idempotent and every guard is explicit.
set -uo pipefail
CASE="$1"; GPU="$2"; MAP="$3"; shift 3
IDS="$*"
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$SRC/experiments/modal/vendor:$SRC:$SRC/experiments/runai"
PY=/usr/bin/python3.12
LAD="$SRC/experiments/runai/queue_fixed_span_ladder.py"

ports_free() { ! ss -ltn 2>/dev/null | grep -qE ':(15000|15001|15002|20000|20001|20002|23000|16000|24000|21000)\b'; }

completed() {   # $1 = run id -> 0 if completed/312
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
  until ports_free; do echo "WAIT_PORTS $(date -Is)"; sleep 60; done
  echo "START $id map=$MAP $(date -Is)"
  env MP_LADDER_SLOTS="$MAP" MP_OPD_DIAGNOSTICS=1 MP_OPD_DIAGNOSTICS_LOGIT_GRAD=0 \
      MP_OPD_DIAGNOSTICS_EVERY=1 $PY "$LAD" train-one --case "$CASE" --id "$id"
  echo "END $id rc=$? $(date -Is)"
  sleep 30
done
echo "RUN_PENDING eval $(date -Is)"
bash "$CASE/eval_after_card.sh" "$GPU" $IDS
echo "RUN_PENDING xong $(date -Is)"
