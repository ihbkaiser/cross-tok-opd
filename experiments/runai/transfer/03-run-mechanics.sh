#!/usr/bin/env bash
set -euo pipefail
PACKAGE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$PACKAGE/workspace.env"
GPU="${1:?Usage: bash 03-run-mechanics.sh GPU_SLOT}"
[[ "$GPU" =~ ^[0-7]$ ]] || { echo 'GPU must be 0..7'; exit 2; }
USED="$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)"
[[ "$USED" =~ ^[0-9]+$ ]] && (( USED < 1024 )) || { echo "GPU $GPU is occupied or status unavailable; no job started"; exit 2; }
STUDENT=/workspace/storage-shared/nlp/tungks/SimCT/runs/qwen-gemma-sft-paper-20260908-045828/checkpoint
TEACHER=/workspace/storage-shared/models/Qwen2.5-7B-Instruct
test -d "$STUDENT"; test -d "$TEACHER"
test -f "$ABC_WORK/data/seen-sft-mechanics-2groups.json"
OUT="$(mktemp -d "$ABC_WORK/mechanics-gpu${GPU}-XXXXXXXX")"
cd "$ABC_WORK/source"
nohup bash -c '
  export CUDA_VISIBLE_DEVICES="$1" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
  bash experiments/runai/python-b200-host.sh experiments/mp_opd/real_oracle.py run \
    --student "$2" --teacher "$3" --data "$4" --output "$5/results" \
    --adapter-module model.layers.25.self_attn.q_proj --device cuda:0 \
    --rank 4 --virtual-lr 0.01 --max-span 4
  rc=$?
  printf "%s\n" "$rc" > "$5/exitcode"
  exit "$rc"
' bash "$GPU" "$STUDENT" "$TEACHER" "$ABC_WORK/data/seen-sft-mechanics-2groups.json" "$OUT" > "$OUT/run.log" 2>&1 < /dev/null &
printf 'PID=%s\nOUT=%s\n' "$!" "$OUT"
printf 'tail -n 30 -f "%s/run.log"\n' "$OUT"
