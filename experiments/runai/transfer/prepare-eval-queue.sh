#!/usr/bin/env bash
set -euo pipefail
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORK="$(dirname "$SOURCE")"
PYTHON=/usr/bin/python3.12
QUEUE="$SOURCE/scripts/evaluation/eval_queue.py"
test ! -e "$WORK/queue.env" || { echo 'Queue already prepared; use queue.env'; exit 2; }
"$PYTHON" "$QUEUE" preflight --score-python "$PYTHON"
OUT="$WORK/eval-$(date -u +%Y%m%d-%H%M%S)"
"$PYTHON" "$QUEUE" prepare --out "$OUT" --hours 20
printf 'export EVAL_PLAN=%q\n' "$OUT/plan.json" > "$WORK/queue.env"
echo "EVAL_PLAN=$OUT/plan.json"
echo "NEXT: bash $SOURCE/experiments/runai/transfer/start-eval-queue.sh"
echo 'CPU preparation complete; no GPU started.'
