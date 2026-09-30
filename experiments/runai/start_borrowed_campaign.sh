#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORK="${1:?shared work directory}"
MANAGER="${2:?pinned job-manager source directory}"
STARTED="${3:?original borrowed-allocation Unix start}"
EXPECTED_HOST="${4:-tungdd11-sparse-vllm-core-0-0}"
[[ "$(hostname)" == "$EXPECTED_HOST" && "$EXPECTED_HOST" != tungks-0-0 ]] || { echo 'WRONG_HOST'; exit 2; }
[[ ! -f "$WORK/campaign.json" ]] || { echo 'ALREADY_PREPARED: inspect existing state, do not start another queue'; exit 2; }
mkdir -p "$WORK"
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3.12 "$SCRIPT_DIR/borrowed_campaign.py" prepare --work "$WORK" --manager "$MANAGER" --started "$STARTED" --expected-host "$EXPECTED_HOST" 2>&1 | tee "$WORK/prepare.log"
STATE="$(/usr/bin/python3.12 -c 'import json,sys;print(json.load(open(sys.argv[1]))["state"])' "$WORK/campaign.json")"
REMAINING="$(/usr/bin/python3.12 -c 'import json,sys,time;print(max(1,int(json.load(open(sys.argv[1]))["deadline"]-time.time())))' "$WORK/campaign.json")"
cd "$MANAGER"
# Manager exit never resets worker deadlines. Outer timeout bounds telemetry too.
nohup timeout --signal=TERM --kill-after=20 "$REMAINING" /usr/bin/python3.12 -m job_manager --state "$STATE" run > "$WORK/manager.log" 2>&1 < /dev/null &
MANAGER_PID=$!
printf '%s\n' "$MANAGER_PID" > "$WORK/manager-wrapper.pid"
sleep 2
kill -0 "$MANAGER_PID" || { tail -n 30 "$WORK/manager.log"; exit 1; }
/usr/bin/python3.12 -m job_manager --state "$STATE" status
printf '\nSTARTED: %s\nSTATE: %s\nWORK: %s\n' "$STARTED" "$STATE" "$WORK"
