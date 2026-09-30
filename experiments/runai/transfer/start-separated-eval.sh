#!/usr/bin/env bash
set -euo pipefail
TOOLS="$(cd -- "$(dirname -- "$0")" && pwd)"
: "${NEW_PLAN:?Source the current pipeline64-queue.env first}"
: "${EVAL_SOURCE:?Source the current pipeline64-queue.env first}"
OLD_PLAN="$NEW_PLAN"
OLD_SOURCE="$EVAL_SOURCE"
OUT="$(dirname "$(dirname "$OLD_PLAN")")/trial256-$(date -u +%Y%m%d-%H%M%S)-$$"
/usr/bin/python3.12 "$TOOLS/extend-eval-simct.py" --plan "$OLD_PLAN" --source "$OLD_SOURCE" --out "$OUT" --queue-update "$TOOLS/eval_queue.py"
NEW_PLAN="$OUT/plan.json"
EVAL_SOURCE="$OUT/source"
/usr/bin/python3.12 - "$OUT" <<'CHECK'
import json,hashlib,sys
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads((p/'migration.json').read_text())
assert r['status']=='verified'
assert hashlib.sha256((p/'plan.json').read_bytes()).hexdigest()==r['new_plan_sha256']
print('VERIFIED_SAVED_RESULTS:',r['totals'])
CHECK
printf 'export NEW_PLAN=%q
export EVAL_SOURCE=%q
' "$NEW_PLAN" "$EVAL_SOURCE" > "$OUT/queue.env"
for GPU in 0 1; do
  CONCURRENCY=256
  LOG="$OUT/generate-gpu$GPU.log"
  nohup bash -c '
    /usr/bin/python3.12 -u "$1/scripts/evaluation/eval_queue.py" worker --phase generate --plan "$2" --gpu "$3" --concurrency "$5" --score-buffer 256 --internal-code-execution
    rc=$?; printf "%s\n" "$rc" > "$4.exitcode"; exit "$rc"
  ' bash "$EVAL_SOURCE" "$NEW_PLAN" "$GPU" "$LOG" "$CONCURRENCY" > "$LOG" 2>&1 < /dev/null &
  printf 'GENERATION_GPU=%s PID=%s LOG=%s
' "$GPU" "$!" "$LOG"
done
LOG="$OUT/scoring.log"
nohup bash -c '
  /usr/bin/python3.12 -u "$1/scripts/evaluation/eval_queue.py" score-spool --plan "$2" --internal-code-execution
  rc=$?; printf "%s\n" "$rc" > "$3.exitcode"; exit "$rc"
' bash "$EVAL_SOURCE" "$NEW_PLAN" "$LOG" > "$LOG" 2>&1 < /dev/null &
printf 'SCORING_PID=%s LOG=%s
ENV=%s
' "$!" "$LOG" "$OUT/queue.env"

cp "$OUT/queue.env" "$TOOLS/launched.env"
