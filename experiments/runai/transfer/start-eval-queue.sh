#!/usr/bin/env bash
set -euo pipefail
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORK="$(dirname "$SOURCE")"
source "$WORK/queue.env"
test -f "$EVAL_PLAN"
QUEUE="$SOURCE/scripts/evaluation/eval_queue.py"
OUT="$(dirname "$EVAL_PLAN")"
for GPU in 0 1; do
  LOG="$OUT/worker-gpu$GPU-$(date -u +%Y%m%d-%H%M%S)-$$.log"
  nohup bash -c '
    /usr/bin/python3.12 -u "$1" worker --plan "$2" --gpu "$3" --internal-code-execution
    rc=$?
    printf "%s\n" "$rc" > "$4.exitcode"
    exit "$rc"
  ' bash "$QUEUE" "$EVAL_PLAN" "$GPU" "$LOG" > "$LOG" 2>&1 < /dev/null &
  printf 'GPU=%s PID=%s LOG=%s\n' "$GPU" "$!" "$LOG"
done
echo "SUMMARY: /usr/bin/python3.12 $QUEUE summarize --plan $EVAL_PLAN"
echo 'Two workers submitted; occupied GPUs are left waiting. Inspect worker logs for startup failures.'
