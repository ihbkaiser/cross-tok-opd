#!/bin/bash
# ============================================================================
# SFT base for the Phi-4-mini -> Gemma-2-2B pair (kdflow trainer).
#
# Trainer is kdflow's own SFT entry (python -m kdflow.cli.train_sft), NOT
# LLaMA-Factory: the company runtime has no `simct-b200-sft` venv, and the
# existing qwen base was produced by this same trainer (its train.log shows
# [sft_trainer.py] / [fsdp_strategy.py] and "Student Model:").
#
# Swap the dataset for the qwen one (run_build_sft_10k_qwen.sh output) to build
# the qwen base with this identical recipe, so the two pairs differ only by teacher.
#
# Flags whose values equal kdflow defaults (lr_scheduler, max_norm, adam_betas,
# weight_decay, seed) are passed explicitly on purpose: they are the paper's
# values, and pinning them keeps the script correct if a default ever changes.
#
# Usage:  bash scripts/sft/run_gemma2_sft_10k_phi4_kdflow.sh            # 1 GPU, accum 32 (paper split, one card)
#         GPUS=4 bash scripts/sft/run_gemma2_sft_10k_phi4_kdflow.sh     # 4 GPUs, accum 8, same effective batch 64
# ============================================================================
set -euo pipefail

SHARED=${SHARED:-/workspace/storage-shared/nlp/tungks}
STUDENT=${STUDENT:-/workspace/storage-shared/models/gemma-2-2b-it}
DATASET=${DATASET:-$SHARED/SimCT/data/sft_warmup_10k_phi-4-mini}
RUN_NAME=${RUN_NAME:-phi4-gemma-sft-$(date +%Y%m%d-%H%M%S)}
SAVE=${SAVE:-$SHARED/SimCT/runs/$RUN_NAME/checkpoint}
# Recipe source of truth: the PAPER (arXiv 2605.07711v2), Table 4 "Warm-Start SFT",
# NOT the repo's scripts/sft/gemma2_sft_warmup_10k_phi4.yaml. The two disagree:
#   paper Table 4 : peak lr 2e-6 | warmup 0.05 | cosine decay | wd 0.0 | 2 epochs |
#                   per-device batch 2 | grad accum 4 | effective batch 64 |
#                   max sequence length 4096 | bf16 | AdamW | hardware 8x H20
#   repo yaml     : cutoff_len 2048 | per_device_train_batch_size 4 (GA 4)  <-- DIFFERENT
# The company qwen base (/workspace/.../runs/qwen-gemma-sft-paper-20260908-045828)
# followed the PAPER, not the yaml; its train.log prints:
#   Num Epochs: 2 | Steps per Epoch: 136 | Per-device Batch Size: 2 | Gradient Accumulation: 32
#   Learning Rate: 2e-06
# i.e. per-device 2 (paper), effective 64 (paper), and accumulation 32 because that run
# used ONE GPU instead of the paper's eight (2 x 32 x 1 = 64 = 2 x 4 x 8).
# max_len 4096 is corroborated by the data: 12 of 8705 qwen-author rows exceed 2048 tokens
# (max 2306) and the trainer logged no max_length filtering, so 2048 is ruled out.
#
# Running GPUS=1 MICRO=2 here reproduces the company qwen base exactly (accum 32);
# kdflow derives accum = train_batch_size / (micro x nodes x gpus) = 64 / (2 x 1 x gpus).
GPUS=${GPUS:-1}
MICRO=${MICRO:-2}
PY=${PY:-/opt/venvs/simct-b200/bin/python}
TORCHRUN=${TORCHRUN:-/opt/venvs/simct-b200/bin/torchrun}

[ -d "$STUDENT" ] || { echo "FAIL: student model not found: $STUDENT"; exit 1; }
[ -d "$DATASET" ] || { echo "FAIL: dataset (datasets.save_to_disk dir) not found: $DATASET"; exit 1; }
[ -x "$TORCHRUN" ] || { echo "FAIL: torchrun not found: $TORCHRUN"; exit 1; }
[ -x "$PY" ] || { echo "FAIL: python not found: $PY"; exit 1; }

# effective batch 64 as in the qwen base: micro x accum x world = 64
ACCUM=$(( 64 / (MICRO * GPUS) ))
[ "$ACCUM" -ge 1 ] || { echo "FAIL: MICRO=$MICRO x GPUS=$GPUS already exceeds 64"; exit 1; }

mkdir -p "$(dirname "$SAVE")"
LOG=${LOG:-$SHARED/SimCT/runs/$RUN_NAME/train.log}
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "aim: effective batch 64 = micro $MICRO x accum $ACCUM x world $GPUS"
echo "student : $STUDENT"
echo "dataset : $DATASET"
echo "save    : $SAVE"
echo "log     : $LOG"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | head -8

OPTS=""
OPTS+=" --num_nodes 1"
OPTS+=" --num_gpus_per_node $GPUS"
OPTS+=" --backend fsdp2"
OPTS+=" --student_name_or_path $STUDENT"
OPTS+=" --train_dataset_path $DATASET"
OPTS+=" --input_key messages"
OPTS+=" --apply_chat_template True"
OPTS+=" --max_len 4096"
OPTS+=" --micro_train_batch_size $MICRO"
OPTS+=" --train_batch_size 64"
OPTS+=" --learning_rate 2e-6"
OPTS+=" --lr_scheduler cosine_with_min_lr"
OPTS+=" --lr_warmup_ratio 0.05"
OPTS+=" --min_lr 0"
OPTS+=" --num_epochs 2"
OPTS+=" --max_norm 1"
OPTS+=" --adam_betas 0.9,0.98"
OPTS+=" --weight_decay 0"
OPTS+=" --gradient_checkpointing True"
OPTS+=" --bf16 True"
OPTS+=" --seed 42"
OPTS+=" --save_path $SAVE"
OPTS+=" --logging_steps 5"
OPTS+=" --use_wandb False"

echo "launching: $TORCHRUN --nproc_per_node=$GPUS -m kdflow.cli.train_sft"
"$TORCHRUN" --nproc_per_node="$GPUS" -m kdflow.cli.train_sft $OPTS 2>&1 | tee "$LOG"
echo "SFT done. Checkpoint: $SAVE"
