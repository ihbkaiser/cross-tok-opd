# Gói MP-OPD A/B/C — source và lệnh chạy

Source commit và checksum nằm trong source-manifest.json đi kèm.
Branch: vdt/ops/b200-portable. Gói chứa Git bundle đầy đủ, manifest checksum,
script setup/chuẩn bị/canary và bản hướng dẫn chi tiết trong source.
Không chứa model, dataset, secrets hay outputs công ty.

## Dataset SFT đã có trên HF

Đúng: teacher Qwen2.5-7B-Instruct đã sinh dữ liệu trong workflow Modal trước đó.
HF repo: codemaivanngu/simct-author-code-10k-qwen25-7b-instruct
Revision đã kiểm tra: 1d276028899f515328e2074e01c97e1b03a89b5b.
10.000 prompts, 80.000 candidates, 8.705 selected. File selected.parquet có
messages làm prompt và label làm response. Không dùng label của prompts.parquet
thay cho lời giải teacher. Không cần sinh lại dữ liệu để kiểm tra pipeline.
Code answers trong dữ liệu này chỉ qua format checks của tác giả, không phải
đã pass execution tests. Chưa xác định phần nào được giữ ngoài SFT/train.

## 1. Code kéo qua HF bundle

Không cần chuyển ZIP source. Script 00-pull-hf.sh được publish cùng bundle trên
HF repo codemaivanngu/simct. Dùng link revision và checksum của lần phát hành
được gửi kèm trong chat để tải script, kiểm tra SHA256 rồi chạy:

```bash
bash /workspace/storage-shared/nlp/tungks/pull-mp-opd-abc.sh
```

Script resolve HF revision một lần (hoặc nhận HF_REVISION được pin), tải manifest
và bundle từ cùng revision, verify SHA256 rồi clone vào thư mục mới. Không sửa
SimCT-git đang chạy train. Nó in ABC_WORK và lệnh chuẩn bị dữ liệu kế tiếp.
Trong shell hiện tại, đặt PACKAGE_DIR bằng đúng ABC_WORK vừa in, ví dụ:

```bash
PACKAGE_DIR=/workspace/storage-shared/nlp/tungks/mp-opd-abc-XXXXXXXX
```

Dùng giá trị thực tế, không chép nguyên XXXXXXXX. Không tải model/dependency,
không dùng GPU ở bước này. Khi mọi run dùng checkout SimCT-git đã dừng, vẫn có
thể cập nhật checkout đó bằng updater HF cũ:

```bash
bash /workspace/storage-shared/nlp/tungks/update-simct.sh
```

Updater chỉ fast-forward. Không chạy updater trên checkout đang train.

## 2. Chuẩn bị hai nhóm canary bằng selected.parquet đã có

```bash
bash "$PACKAGE_DIR/02-prepare-mechanics.sh"
```

Script kiểm tra file ở SimCT/data/qwen-author/data/selected.parquet.
Nếu không có, tải đúng revision HF vào workspace mới (~11 MB), verify SHA256.
Sau đó dùng /usr/bin/python3.12 cùng datasets5.0.0 có sẵn để chuẩn bị dữ liệu.
Không cài thư viện. Output dùng exclusive-create: không ghi đè khi chạy lại.

Đây là mechanics canary trên dữ liệu có thể đã dùng SFT/train. Chia B/select/eval
khác prompt không xóa được việc student từng học dữ liệu đó. Chỉ dùng để xem
code/gradient/memory/time hoạt động, không tuyên bố held-out generalization.

## 3. Chạy hai nhóm trên một GPU trống (bước này mới dùng GPU)

```bash
nvidia-smi
# Chỉ chọn 1 nếu card 1 thực sự trống:
bash "$PACKAGE_DIR/03-run-mechanics.sh" 1
```

Script từ chối khi GPU đã dùng >=1 GiB. Nó chạy nền, in PID/OUT và lệnh tail log.
Student là SFT checkpoint chung; teacher Qwen hiện có. Chỉ thêm probe B rank4 ở
last q_proj, không sửa checkpoint, không chạy full312, không upload W&B.
virtual-lr0.01 là bước thăm dò adapter, không phải LR train khuyến nghị.
Dừng ở hai nhóm. Gửi exitcode, summary.json và cuối run.log trước khi scale.
Trajectories chứa dữ liệu công ty: giữ trên máy công ty, không upload tự động.

## 4. Bước nghiên cứu sau canary

Cần file reference train/dev có prompts chưa dùng SFT/MP để làm đánh giá độc lập.
Có thể dùng cùng teacher, không cần teacher khác. Chỉ sinh bổ sung cho prompt
mới khi không có response đã giữ lại. CLI có --exclude-prompts để loại tập đã
train/canary và --reference-key @last-assistant cho conversation SFT.
Đọc source/docs/mp_opd_abc_20260909.md để biết các controls và giới hạn.
Không khởi chạy learned soft full training: phần online meta-training D chưa làm.

## Kiểm chứng / trạng thái phát hành

38 tests pass, 1 skipped (thiếu tokenizer fixture); Python/shell syntax pass.
Test toàn luồng dùng model giả lập. Suite cũ cần transformers chưa có local;
chưa thực hiện real-model/GPU canary. Source được phát hành theo workflow GitHub → HF bundle; source-manifest.json
cho biết commit cụ thể. Hướng dẫn tải riêng; code không cần chuyển ZIP thủ công.
Dataset teacher trên HF ở revision trên đã được xác minh riêng.
