# MP-OPD atomic và fixed-2: phân tích hai run RunAI 08/09/2026

## Kết luận
Cả hai run hoàn tất 312 optimizer updates/312 rollout iterations, exit code 0; chuỗi step 1–312 đầy đủ, các scalar đã parse đều hữu hạn. Không có empty response hoặc collapse-stop theo metrics. Đây là kết quả vận hành, chưa có đánh giá math/code để kết luận hiệu quả distillation.

Run dùng Gemma2 từ checkpoint SFT-paper làm student, Qwen2.5-7B-Instruct làm teacher; 10.000 prompts, hai epochs, batch 64, microbatch 2, eager attention, bf16, LR 1e-6, rollout temperature 0.6/top-p 0.95, mỗi run một B200. Hai run dùng bypass parity hữu hạn; không phải run kiểm chứng strict parity.

## So sánh
| Chỉ số | Atomic | Fixed-2 |
|---|---:|---:|
| Optimizer updates | 312 | 312 |
| Thời gian trainer (giờ) | 5.910 | 5.134 |
| Response mean, TB 20 bước đầu → cuối | 499.3 → 578.8 | 477.7 → 606.6 |
| Response median, TB 20 bước đầu → cuối | 391.9 → 439.1 | 393.5 → 440.1 |
| Loss, TB 20 bước đầu → cuối | -0.13628 → -0.09655 | -0.10063 → -0.04844 |
| Mean parity gap, TB các bước | 0.007049 | 0.007243 |
| Max token gap quan sát trong bypass | 25.894341 | 24.941610 |
| Số dòng bypass quan sát | 152 | 192 |
| Token vượt 0.5 trên các dòng bypass | 171 | 204 |
| Samples không atomize được / tổng | 80/19968 (0.401%) | 257/19968 (1.287%) |
| Max optimizer_final_grad_norm được log | 8.463 | 3.002 |
| Peak allocated (GiB, student allocator) | 69.79 | 69.77 |
| Peak reserved (GiB, student allocator) | 112.62 | 107.24 |
| Processed student tokens | 12379998 | 11416250 |
| End-to-end tokens/s, tổng tokens / tổng step time | 583.8 | 620.4 |

## Diễn giải và giới hạn
- Response length tăng vừa phải trong trung bình đầu/cuối; empty_response_fraction, collapse_bad_streak và collapse_stop đều bằng 0 ở tất cả bước. Điều này loại trừ dấu hiệu collapse theo detector được ghi, không chứng minh nội dung đúng.
- Loss âm và khác thang giữa hai partition; code đi kèm gọi hard_partition_loss với current_nll/base_credit/weight. Không diễn giải loss này như KL không âm hoặc dùng trực tiếp để xếp hạng atomic/fixed. Cần benchmark cố định trên checkpoint.
- Atomic có 80 samples không hợp lệ, fixed có 257. Lý do được ghi là unsupported_added_token. Reason metrics xuất hiện thưa và được lấy trung bình trên các microbatch có key, nên không cộng trực tiếp để suy ra tổng samples; tổng trong bảng dùng trajectory_valid_sample_count của từng rollout.
- Trung bình parity nhỏ nhưng đuôi có outlier rất lớn: 25.894341 và 24.941610. Chưa có token ID/context tại outlier nên chưa xác định được lỗi token, model/backend, precision hay logprob. Không đủ căn cứ gọi tất cả là sai số BF16 thông thường.
- Trainer lấy trung bình danh sách metrics. Vì vậy trajectory_logprob_abs_max trên biểu đồ là trung bình các max microbatch, không phải max toàn bước. Báo cáo dùng max từ dòng MP_PARITY_BYPASS để giữ bằng chứng outlier. Chưa có p99 hoặc phân bố delta toàn run; không tái dựng được từ scalar trung bình.
- Các dòng bypass không có global step/token ID; bảng warnings trên W&B dùng warning_index, không gán step suy đoán. Count phản ánh các dòng được lưu trong log, không cam kết số sự kiện tuyệt đối nếu hệ thống logging bỏ dòng.
- Fixed chạy nhanh hơn trong hai run này nhưng có tổng tokens xử lý thấp hơn. Throughput được tính bằng tổng tokens chia tổng step time; chưa phải so sánh matched workload và không suy ra speedup thuật toán tổng quát.
- GPU allocated/reserved là telemetry của allocator được log; resource/gpu_* là snapshot node khi hai run cùng chạy. Không dùng để kết luận mỗi run dùng cả hai GPU hay suy ra utilization trung bình. Microbatch 4 chưa được thử trong hai run này.
- Source provenance được copy sau training; tên backup/code không chứng minh immutable source của tiến trình tại launch. AllArguments trong train.log là bằng chứng cấu hình chạy. Archive không chứa model weights hay rollout JSONL (chỉ danh sách), nên chưa xác minh checkpoint content hoặc replay outlier.

## W&B backfill
Giữ nguyên tên hai run; dùng ID xác định từ SHA-256 của tên run, group qwen-gemma-runai-20260908. Metrics giữ namespace train/* như logger gốc, trục train/global_step, lịch sử 312 bước. Ghi rõ historical_import và parity_diagnostic_bypass. Không giả lập timestamps huấn luyện hoặc telemetry máy import. Upload train.log, summary, analysis và báo cáo làm artifact; cảnh báo bypass thành bảng riêng. Sau upload đọc lại 312 bước, loss và content_length_mean để đối chiếu.

## Việc cần làm để đánh giá chất lượng
Đánh giá SFT baseline cùng hai checkpoint bằng đúng bộ math/code và decoding budget. Nếu điều tra parity tiếp, cần giữ output token IDs, behavior logprobs, model version và logits/replay của các outlier >0.5, ưu tiên outlier khoảng 25. Những run đã hoàn tất vẫn dùng được cho đánh giá chẩn đoán nếu công khai bypass; không gắn nhãn strict parity.

## Nguồn
Archive SHA-256: 18d4f50e99da064781e61c4198a95e240184e01acd6d208e260e3331e46b3b29
- qwen-gemma-mp-atomic-gpu0-limit0-20260908-133550: train.log SHA-256 731cdbd952c5234849f83cefb30989413012faa57d06f2c1c554b430a53f8c0b.
- qwen-gemma-mp-fixed-gpu1-limit0-20260908-133550: train.log SHA-256 13a5b4c4bae73e750807b9e427b6aeaf3891f2f10f8e8396141e69640722ff5d.

## Tái dùng importer

Chạy từ repository bằng interpreter có sẵn. Dry-run chỉ cần Python stdlib:

```bash
python3 experiments/modal/import_mp_opd_runai_logs.py --root /path/to/extracted-export
```

Thêm --upload bằng môi trường có wandb để import. Script đọc credential W&B hiện có tại /mnt/c/Users/Tung/_netrc trong WSL; không in hoặc lưu lại secret. Upload từ chối khi tên run đã tồn tại; cần kiểm tra run/receipt trước khi thử lại sau lỗi mạng. Log format cần content_length_mean và completed_optimizer_updates, expected run 312 steps. Không dùng script này cho run đang chạy hoặc canary.

## Upload đã xác minh

- [qwen-gemma-mp-atomic-gpu0-limit0-20260908-133550](https://wandb.ai/kieusontung8-hanoi-university-of-science-and-technology/vdt-simct-tunix-reproduction/runs/mpbackfill-376343a0f5e6f573): state=finished; đọc lại đủ 312 bước; loss và content_length_mean khớp log cho mọi bước.
- [qwen-gemma-mp-fixed-gpu1-limit0-20260908-133550](https://wandb.ai/kieusontung8-hanoi-university-of-science-and-technology/vdt-simct-tunix-reproduction/runs/mpbackfill-a78dde7b327ddbb7): state=finished; đọc lại đủ 312 bước; loss và content_length_mean khớp log cho mọi bước.

Importer: 4 regression tests pass; parse/validate đủ 624 bước nguồn trước upload.
