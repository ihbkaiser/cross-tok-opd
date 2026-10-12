# ALIGN-NEW / TRUST-P: bàn giao agent chạy trên node

Đọc cùng `docs/ALIGN_NEW_TRUST_P.md` và release receipt đi kèm. Mode launcher chính
xác là `align_new`, `trust_p`. Đây là phương pháp thử nghiệm; canary pass không
chứng minh accuracy phục hồi. Người viết tài liệu chưa launch trên node công ty.

## 1. Máy, source và runtime

Node đã được giao: `root@embed-8b-training-v2-0-1:/workspace#`, **GPU3**. Kiểm tra
hostname, UUID, process và quyền sử dụng GPU trước launch; không kill job khác.
Không chạy hai mode đồng thời trên GPU3.

Branch `vdt/ops/b200-portable` được đồng bộ tại `sontungkieu/SimCT`,
`ihbkaiser/cross-tok-opd`; HF source là `codemaivanngu/simct`, file
`simct-b200-portable.bundle`. Lấy **HF revision, source commit và bundle SHA256
từ release receipt cuối**, tải qua proxy đang dùng và kiểm tra hash trước clone.
Clone ở checkout mới; không reset checkout của job đang chạy. Không dùng HF main
thay cho revision đã ghim nếu muốn tái hiện đúng bản đã test.

Runtime đã kiểm chứng:

```text
docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f
/opt/venvs/simct-b200/bin/python
torch=2.11.0+cu130 transformers=5.6.0 sglang=0.5.11 ray=2.58.0
```

Dùng runtime hiện có nếu preflight khớp. Source mới không yêu cầu cài lại
Transformers, build image hoặc tải lại runtime. `MP_RUNTIME_DIR` phải trỏ đúng
payload hiện tại có `runtime-host-libs`; wrapper còn kiểm tra CUDA/NVRTC headers.
Nếu preflight fail, giữ lỗi cụ thể để xử lý môi trường; không pip install đè.

## 2. Chuẩn bị đúng asset và cấu hình

Lấy student, teacher, dataset từ **launch-config đúng cặp/run**. Bắt đầu từ SFT
trong output directory mới, không resume ALIGN/TRUST cũ sang mode mới. Không cần
warmup lại chỉ vì thêm mode. SHA256 model.safetensors của hai SFT đã dùng trên Modal:

| SFT | SHA256 |
|---|---|
| qwen-gemma-sft-paper-20260908-045828 | 26a856f4a6e913f21e961cdcb5c7894f5966733e44f8b4624580de8f7ad439c1 |
| phi4-gemma-sft-20261002-131247 | fb54dee22121b73f21260699b7b2ea64779a04e54dc0cd771168c140bb69414e |

Giữ batch64/micro4, pure OPD (`kd_ratio=1`), identity credit, eager attention,
exact_head FP32, sequence_parallel_size=1. TRUST-P dùng scope=batch. ALIGN-NEW
native dùng source=xtoken, rho=0.1; source=run/run_length=2 là đối chứng riêng.
Không đổi partition giữa hai run rồi quy toàn bộ chênh lệch cho guard mới.

**Block N1 — setup/preflight, chưa train.** Agent điền các biến từ asset/runtime
đã xác minh; đường dẫn dưới đây được nhập tương tác để không dùng nhầm mặc định Qwen
cho Phi. Chạy trong checkout source đã kiểm tra commit theo release receipt.

```bash
(
  set -euo pipefail
  test "$(hostname)" = embed-8b-training-v2-0-1
  nvidia-smi -i 3 --query-gpu=index,uuid,name,memory.total,memory.used --format=csv
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
  read -r -p 'Verified SFT checkpoint directory: ' MP_STUDENT_PATH
  read -r -p 'Verified teacher directory: ' MP_TEACHER_PATH
  read -r -p 'Verified training prompts parquet: ' MP_DATASET_PATH
  read -r -p 'Existing runtime payload directory: ' MP_RUNTIME_DIR
  read -r -p 'Fresh persistent run root: ' MP_RUN_ROOT
  export MP_STUDENT_PATH MP_TEACHER_PATH MP_DATASET_PATH MP_RUNTIME_DIR MP_RUN_ROOT
  export CUDA_VISIBLE_DEVICES=3 MP_MICRO_TRAIN_BATCH_SIZE=4 MP_TRUST_SCOPE=batch
  export MP_GRASS_CHUNK_SOURCE=xtoken MP_ALIGN_NEW_MIN_DESCENT_FRACTION=0.1
  export MP_TENSORBOARD=1 MP_SEED=42
  unset MP_RESUME MP_PREFLIGHT_ONLY MP_MAX_LEN MP_GENERATE_MAX_LEN MP_RUN_DIR
  test -d "$MP_RUNTIME_DIR/runtime-host-libs"
  test -d "$MP_STUDENT_PATH" && test -d "$MP_TEACHER_PATH" && test -f "$MP_DATASET_PATH"
  mkdir -p "$MP_RUN_ROOT"
  bash experiments/runai/python-b200-host.sh experiments/runai/pipeline_preflight.py \
    --student "$MP_STUDENT_PATH" --teacher "$MP_TEACHER_PATH" \
    --dataset "$MP_DATASET_PATH" --algorithm mp_opd --chunk-source xtoken --check-gpu
  for mode in align_new trust_p; do
    MP_PREFLIGHT_ONLY=1 bash experiments/runai/run_single_gpu.sh 3 "$mode" 2
  done
  echo '--- hết Block N1, shell con sống ---'
)
```

Các export trong subshell N1 không tồn tại ở shell cha. Agent ghi lại asset paths
đã kiểm tra vào config riêng của campaign để Block N2 dùng đúng giá trị; không đưa
secrets vào file. Hai preflight tạo directory riêng; đọc launch-config và giữ
bằng chứng, không dùng directory đó làm run train vì launcher yêu cầu directory mới.

## 3. Canary và run dài

**Block N2 — train canary, thực thi sau N1 và xác nhận GPU3 được giao.** Cần export
lại bốn asset/runtime biến và MP_RUN_ROOT từ config đã xác minh. Script cố ý dừng
khi thiếu biến để tránh dùng asset mặc định. Canary short-generation giống Modal:

```bash
(
  set -euo pipefail
  test "$(hostname)" = embed-8b-training-v2-0-1
  : "${MP_STUDENT_PATH:?Set verified SFT path}"
  : "${MP_TEACHER_PATH:?Set verified teacher path}"
  : "${MP_DATASET_PATH:?Set verified prompts path}"
  : "${MP_RUNTIME_DIR:?Set existing runtime path}"
  : "${MP_RUN_ROOT:?Set fresh persistent run root}"
  export MP_STUDENT_PATH MP_TEACHER_PATH MP_DATASET_PATH MP_RUNTIME_DIR MP_RUN_ROOT
  export CUDA_VISIBLE_DEVICES=3 MP_MICRO_TRAIN_BATCH_SIZE=4 MP_TRUST_SCOPE=batch
  export MP_GRASS_CHUNK_SOURCE=xtoken MP_ALIGN_NEW_MIN_DESCENT_FRACTION=0.1
  export MP_TENSORBOARD=1 MP_SEED=42 MP_GENERATE_MAX_LEN=128
  unset MP_RESUME MP_PREFLIGHT_ONLY MP_RUN_DIR MP_MAX_LEN
  for mode in align_new trust_p; do
    bash experiments/runai/run_single_gpu.sh 3 "$mode" 2
  done
  echo '--- hết Block N2, shell con sống ---'
)
```

Kiểm tra mỗi job: exitcode=0; run-summary completed/optimizer_updates=2; launch
config đúng mode, batch64/micro4, context4096/generation128/horizon312; loss và
grad hữu hạn; parity pass; numeric fallback=0 và atom coverage có valid samples.
TRUST-P: `full_batch=1`, `calibration_microbatch_count=16`, `exact_geometry=1`,
`mismatch_scale=1`; nếu anchor tồn tại thì strict_descent_ratio >=1 trong sai số
fp32. Theo dõi strict_scale/retained mismatch norm, không chỉ loss.
ALIGN-NEW: theo dõi beta, guard_active_fraction, no_anchor_fraction và retained
descent; các số là **trung bình per-response head proxy**, không phải full-batch
hoặc full-Transformer guarantee. Logprob=0 riêng lẻ không chứng minh engine bug;
đọc cùng parity và zero rates. Canary chưa kiểm chứng memory cho response4096.

Run dài dùng cùng asset/config, output mới; **unset cả `MP_GENERATE_MAX_LEN` và
`MP_MAX_LEN`** cùng các biến diagnostic/resume/output của canary. Hai knobs độ dài
chỉ được phép cho limit1–30, nên không để lại chúng khi launch312. Chọn một mode,
đừng launch hai mode cùng GPU:

```bash
# Trong subshell có các asset/config export như N2, sau khi node canary pass:
unset MP_GENERATE_MAX_LEN MP_MAX_LEN MP_PREFLIGHT_ONLY MP_RESUME MP_RUN_DIR
export MP_CHECKPOINT_STEPS=40,80,120,160,200,240,280,312
bash experiments/runai/run_single_gpu.sh 3 align_new 312
# Hoặc trust_p 312, trong một run riêng sau khi GPU đã trống.
```

Chấm step40 trước khi tiếp tục dài. Giữ seed/scorer/parser/sampling và benchmark
giống control; lưu raw responses + IDs + stop reasons bên cạnh scores. Không bật
eval-on-checkpoint khi chưa chuẩn bị đúng eval assets. Thu launch-config, full
train log, TensorBoard, run-summary, exitcode, weights/config/tokenizer và hash.
Nếu fail, báo traceback và memory/length thực; không đổi batch hoặc runtime âm thầm.

## 4. Phạm vi kết luận

Kiểm tra autograd/tiny-model và production canary được báo riêng trong đặc tả và
release receipt. Diagnostic fixture 128 prompts lặp không tái hiện distribution
của dataset training. Không dùng 2 updates pass làm bằng chứng chống collapse hoặc
accuracy tăng. Các guarantee toán học hiện tại chỉ áp dụng LM-head output-edge
proxy; tác động qua tied input embedding, full Transformer, Adam và các response
trong ALIGN-NEW vẫn cần đo bằng frozen-prefix/full-gradient và benchmark thực.
