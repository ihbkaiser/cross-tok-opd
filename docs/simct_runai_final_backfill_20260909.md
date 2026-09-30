# SimCT 2ep microbatch 4 — log cuối và W&B backfill

Run: qwen-gemma-simct-2ep-mb4-20260908-093304.
W&B ID giữ nguyên: bf-36d2f25d97568e93.

## Bằng chứng nguồn

Archive chứa train.log và checkpoint/run-summary.json. Summary xác nhận span_ctkd, 312 optimizer updates, 312 rollout iterations, completed, stop_reason=null. Log có đủ 312 dòng metrics với step 1–312 liên tục; mọi scalar được parse hữu hạn. Training completed and model saved xuất hiện lúc 2026-09-08 14:50:19 theo đồng hồ log. Thời gian trainer từ summary: 18741.690752 giây (5h12m22s). Không có weights trong archive để kiểm tra nội dung checkpoint.

## Diễn biến

| Chỉ số | TB 20 bước đầu | TB 20 bước cuối |
|---|---:|---:|
| Loss | 1.14508 | 0.76764 |
| Response mean | 475.81 | 1825.65 |
| Response median | 381.78 | 922.08 |
| optimizer_final_grad_norm | 11.654 | 4.951 |

Empty response và collapse_stop bằng 0 trên toàn bộ 312 bước. Không có collapse theo detector nhưng response dài hơn đáng kể; chưa có benchmark để kết luận chất lượng tốt hơn. Loss này khác objective MP-OPD nên không xếp hạng mô hình bằng loss giữa các phương pháp.

Max optimizer_final_grad_norm được log là 71.289978; tên metric không đủ chứng minh giá trị này sau clipping. Peak student allocated 95.845978 GiB, reserved 128.808594 GiB; không phải tổng peak VRAM tất cả actor.

## Quy trình bổ sung lịch sử

Đọc W&B hiện có, xác minh 140 bước đầu khớp tất cả scalar nguồn rồi chỉ append bước 141–312 bằng resume=must. Giữ nguyên ID, tên và namespace metric dạng flat của importer trước đó. Lưu snapshot W&B trước khi sửa trong remote_artifacts/simct-final-20260909, cập nhật source_training_status từ incomplete_snapshot sang completed và ghi summary gốc. Upload log cuối làm artifact.

ETA/Elapsed trong importer cũ chỉ có phần giờ; không dùng làm thời gian giây. Metadata ghi rõ giới hạn này; không sửa lại 140 dòng lịch sử cũ. Các giá trị từ text log đã làm tròn, đặc biệt optimizer_lr_used có thể hiển thị 0 ở warmup dù lr dạng khoa học vẫn khác 0.

## Tái dùng công cụ

Dry-run bằng Python stdlib:

```bash
python3 experiments/modal/append_simct_runai_logs.py --root /path/to/extracted-evidence
```

Cần wandb-before.json đã đọc từ API cùng log và summary. Thêm --upload với interpreter có wandb; công cụ xác minh lại prefix trực tiếp trên W&B trước khi ghi. Credential đọc từ /mnt/c/Users/Tung/_netrc trong WSL, không in hoặc sao chép giá trị. Công cụ dành riêng run ID trên và budget 312 bước; dừng khi thiếu, lệch hoặc trùng step.

Regression tests: 4 trường hợp prefix khớp, metric bị đổi, step trùng/thiếu, metric thiếu.

## Kết quả xác minh sau upload

Đã append 172 bước vào [run W&B cũ](https://wandb.ai/kieusontung8-hanoi-university-of-science-and-technology/vdt-simct-tunix-reproduction/runs/bf-36d2f25d97568e93); state=finished, source_training_status=completed. Đối chiếu tất cả scalar nguồn cho đủ 312 optimizer steps duy nhất: khớp.

API scan_history sau resume trả lặp các dòng cũ giống hệt nhau: lần đầu 452 rows/312 steps (140 lặp), lần kiểm tra sau với page_size khác 353 rows/312 steps (41 lặp). Không có step thiếu hay duplicate mâu thuẫn. Chưa xác định cơ chế pagination/đồng bộ backend; không khẳng định lịch sử vật lý đã hết duplicate. Khi phân tích API export, loại bản sao hoàn toàn theo optimizer_step và báo số dòng thô; không cộng hai lần. Importer không gửi lại 140 bước đầu.

Readback có kiểm tra duplicate giống hệt và từ chối bản sao mâu thuẫn; pre-append vẫn từ chối mọi duplicate để tránh nối vào lịch sử mơ hồ. Tổng 5 regression tests pass.
