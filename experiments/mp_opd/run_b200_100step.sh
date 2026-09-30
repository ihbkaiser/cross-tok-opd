#!/usr/bin/env bash
set -euo pipefail

: "${MP_OPD_STUDENT_PATH:?set exact local student model/checkpoint path}"
: "${MP_OPD_TEACHER_PATH:?set exact local teacher model path}"
: "${MP_OPD_DATASET_PATH:?set exact local training dataset path}"
: "${MP_OPD_OUTPUT_ROOT:?set output root}"
: "${MP_OPD_EVIDENCE_DIR:?set evidence directory}"
: "${MP_OPD_GPU_COUNT:?set B200 GPU count}"

MP_OPD_MODE="${MP_OPD_MODE:-atomic}"
MP_OPD_MAX_SPAN_LENGTH="${MP_OPD_MAX_SPAN_LENGTH:-4}"
MP_OPD_FIXED_SPAN_LENGTH="${MP_OPD_FIXED_SPAN_LENGTH:-2}"
MP_OPD_RANDOM_SEED="${MP_OPD_RANDOM_SEED:-43}"
MP_OPD_PARTITION_TEMPERATURE="${MP_OPD_PARTITION_TEMPERATURE:-1.0}"

case "$MP_OPD_MODE" in
  atomic|fixed|random) ;;
  oracle) echo "oracle mode is instrumentation-only" >&2; exit 2 ;;
  soft)
    : "${MP_OPD_ENERGY_CHECKPOINT:?soft mode requires audited energy checkpoint}"
    ;;
  *) echo "unsupported MP_OPD_MODE: $MP_OPD_MODE" >&2; exit 2 ;;
esac

for path in "$MP_OPD_STUDENT_PATH" "$MP_OPD_TEACHER_PATH" "$MP_OPD_DATASET_PATH"; do
  test -e "$path" || { echo "missing required local path" >&2; exit 2; }
done
install -d -m 700 "$MP_OPD_OUTPUT_ROOT" "$MP_OPD_EVIDENCE_DIR"

git rev-parse HEAD > "$MP_OPD_EVIDENCE_DIR/source.commit"
git status --short > "$MP_OPD_EVIDENCE_DIR/source.status"
python - <<'PY' > "$MP_OPD_EVIDENCE_DIR/runtime.json"
import json, platform, torch
print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}, sort_keys=True))
PY

extra=()
if [[ "$MP_OPD_MODE" == soft ]]; then
  extra+=(--mp_opd_energy_checkpoint "$MP_OPD_ENERGY_CHECKPOINT")
fi

# This is a bounded systems canary. With 6,400 records, rollout batch 64,
# n_samples=1, train batch 64 and one epoch, KDFlow's audited step equation is
# exactly (6400/64) * (64*1/64) = 100 optimizer updates. The scheduler horizon
# is pinned to the same value.
python -m kdflow.cli.train_kd_on_policy \
  --student_name_or_path "$MP_OPD_STUDENT_PATH" \
  --teacher_name_or_path "$MP_OPD_TEACHER_PATH" \
  --train_dataset_path "$MP_OPD_DATASET_PATH" \
  --save_path "$MP_OPD_OUTPUT_ROOT/checkpoint" \
  --ckpt_path "$MP_OPD_OUTPUT_ROOT/checkpoints" \
  --num_nodes 1 \
  --num_gpus_per_node "$MP_OPD_GPU_COUNT" \
  --backend fsdp2 \
  --num_epochs 1 \
  --train_batch_size 64 \
  --micro_train_batch_size 1 \
  --learning_rate 5e-5 \
  --lr_warmup_ratio 0.05 \
  --lr_scheduler cosine_with_min_lr \
  --lr_scheduler_horizon_steps 100 \
  --weight_decay 0.0 \
  --max_norm 1.0 \
  --gradient_checkpointing True \
  --enable_sleep True \
  --bf16 True \
  --full_determinism True \
  --seed 43 \
  --input_key text \
  --apply_chat_template False \
  --max_samples 6400 \
  --prompt_max_len 240 \
  --max_len 2048 \
  --preprocess_num_workers 8 \
  --rollout_num_engines 1 \
  --rollout_tp_size 1 \
  --rollout_mem_fraction_static 0.30 \
  --rollout_batch_size 64 \
  --generate_max_len 1808 \
  --n_samples_per_prompt 1 \
  --temperature 1.0 \
  --top_p 1.0 \
  --teacher_tp_size 1 \
  --teacher_pp_size 1 \
  --teacher_ep_size 1 \
  --teacher_dp_size 1 \
  --teacher_mem_fraction_static 0.35 \
  --teacher_context_length 4096 \
  --teacher_forward_n_batches 8 \
  --kd_algorithm mp_opd \
  --kd_ratio 1.0 \
  --mp_opd_mode "$MP_OPD_MODE" \
  --mp_opd_max_span_length "$MP_OPD_MAX_SPAN_LENGTH" \
  --mp_opd_fixed_span_length "$MP_OPD_FIXED_SPAN_LENGTH" \
  --mp_opd_random_seed "$MP_OPD_RANDOM_SEED" \
  --mp_opd_partition_temperature "$MP_OPD_PARTITION_TEMPERATURE" \
  --save_steps 100 \
  --logging_steps 1 \
  --use_tensorboard True \
  --tensorboard_log_dir "${TENSORBOARD_LOG_DIR:-$MP_OPD_OUTPUT_ROOT/tensorboard}" \
  --tensorboard_flush_secs 10 \
  --use_wandb True \
  --wandb_org "${WANDB_ENTITY:?set WANDB_ENTITY}" \
  --wandb_project "${WANDB_PROJECT:-vdt-simct-tunix-reproduction}" \
  --wandb_run_name "${WANDB_RUN_NAME:-mp-opd-b200-${MP_OPD_MODE}-100step}" \
  --wandb_job_type implementation-validation \
  --wandb_tags "mp-opd,b200,100-update,implementation-validation,${MP_OPD_MODE}" \
  --wandb_mode online \
  --wandb_dir "${WANDB_DIR:-$MP_OPD_OUTPUT_ROOT/wandb}" \
  "${extra[@]}" 2>&1 | tee "$MP_OPD_EVIDENCE_DIR/train.log"
