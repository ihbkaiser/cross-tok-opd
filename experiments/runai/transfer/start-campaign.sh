#!/usr/bin/env bash
set -euo pipefail
SOURCE="$(cd "$(dirname "$0")/../../.." && pwd)"
BASE=/workspace/storage-shared/nlp/tungks
OUT="$(mktemp -d "$BASE/campaign20-XXXXXXXX")"
# prepare owns a new child directory; keep all outputs outside the checkout.
PLAN_DIR="$OUT/work"
TEMPLATE="${1:?Pass completed eval plan.json}"
shift
/usr/bin/python3.12 "$SOURCE/experiments/runai/prepare_campaign.py" --out "$PLAN_DIR" --eval-template "$TEMPLATE" "$@"
nohup /usr/bin/python3.12 -u "$SOURCE/experiments/runai/campaign.py" "$PLAN_DIR/plan.json" > "$PLAN_DIR/manager.log" 2>&1 < /dev/null &
printf 'PID=%s\nWORK=%s\nLOG=%s\n' "$!" "$PLAN_DIR" "$PLAN_DIR/manager.log"
