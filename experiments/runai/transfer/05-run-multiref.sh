#!/usr/bin/env bash
set -euo pipefail
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORK="$(dirname "$SOURCE")"
GPU="${1:?Usage: bash 05-run-multiref.sh GPU_SLOT}"
[[ "$GPU" =~ ^[0-7]$ ]] || exit 2
# Coordinate with the eval queue: never take a GPU claimed by an eval worker.
exec 9>"/tmp/simct-eval-gpu${GPU}.lock"
flock -n 9 || { echo 'GPU reserved by eval queue; no probe started'; exit 2; }
USED="$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)"
[[ "$USED" =~ ^[0-9]+$ ]] && (( USED < 1024 )) || { echo 'GPU busy; no probe started'; exit 2; }
test -f "$WORK/data/multiref-mechanics.json"
OUT="$(mktemp -d "$WORK/multiref-gpu${GPU}-XXXXXXXX")"
nohup bash -c '
 export CUDA_VISIBLE_DEVICES="$1" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
 timeout --signal=TERM --kill-after=15s 900s bash "$2/experiments/runai/python-b200-host.sh"    "$2/experiments/mp_opd/real_oracle.py" run    --student /workspace/storage-shared/nlp/tungks/SimCT/runs/qwen-gemma-sft-paper-20260908-045828/checkpoint    --teacher /workspace/storage-shared/models/Qwen2.5-7B-Instruct    --data "$3/data/multiref-mechanics.json" --output "$4/results"    --adapter-module model.layers.25.self_attn.q_proj --device cuda:0    --rank 4 --virtual-lr 0.1 --model-dtype float32 --max-span 4    --select-counts 1 4 8 --max-new-tokens 64 --max-reference-tokens 1024
 rc=$?; printf "%s\n" "$rc" > "$4/exitcode"; exit "$rc"
' bash "$GPU" "$SOURCE" "$WORK" "$OUT" > "$OUT/run.log" 2>&1 < /dev/null &
printf 'PID=%s\nOUT=%s\n' "$!" "$OUT"
printf 'tail -n 60 -f "%s/run.log"\n' "$OUT"
