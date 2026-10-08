# TRUST-R / TRUST-B — implementation sau audit 2026-10-09

Bản cũ calibrate TRUST-B trên microbatch 4 dù optimizer batch là 64. Đó là một
thuật toán khác định nghĩa full-batch: trung bình nhiều lambda không bằng lambda
của tổng gradient. Báo cáo này thay mô tả cũ; các run trước audit vẫn là bằng
chứng vận hành của bản microbatch, không phải xác nhận TRUST-B full-batch.

## Contract toán học

Atom `one_to_one` thuộc strict S; atom `multi_token` thuộc mismatch M. Ranh giới
khác bị từ chối. Loss mỗi atom là token-sum NLL nhân rate detached `r_i`.

\[
G_S=\sum_{i\in S}r_i\nabla_{W,b}\ell_i,\quad
G_M=\sum_{i\in M}r_i\nabla_{W,b}\ell_i,\qquad
\lambda=\max\left(0,\frac{\langle G_S,G_M\rangle}{\|G_M\|^2}\right).
\]

Calibrate khi có cả S/M và `||G_M||² > eps_g`, mặc định `1e-12`. Nếu không đủ
điều kiện, lambda bằng 0. Không cap lambda ở 1; đại lượng có bound là
`kappa = lambda * ||G_M|| / ||G_S||` khi geometry hợp lệ.

\[
L=L_S+\operatorname{stopgrad}(\lambda)L_M.
\]

TRUST-R dùng S/M của một response. TRUST-B mặc định dùng toàn optimizer batch;
`mp_opd_trust_scope=microbatch` giữ đường pair-Gram cũ như ablation có tên rõ.
Metric `mp_opd_trust_scope=1` chỉ nói calibration unit là batch; phải đọc thêm
`mp_opd_trust_full_batch=1` để xác nhận full optimizer batch.

## Hai lượt forward của TRUST-B

1. Actor gọi `prepare_optimizer_batch` trước backward. Với B64/M4 trên một GPU,
   window phải chứa đúng 16 microbatch và bắt đầu ở optimizer boundary.
2. Lượt đầu chạy train mode trong `no_grad`, giữ nguyên tham số. Ghi RNG trước
   từng microbatch, tính cùng atomization và rates như loss thật.
3. `TrustHeadAccumulator` cộng trực tiếp hai tensor gradient head `[vocab, hidden]`
   FP32, kèm bias nếu head có bias. Không giữ graph hoặc Gram toàn batch. Tích
   có mọi cross-response và cross-microbatch vì norm được lấy sau khi cộng.
4. Với final-logit softcap, delta là `(p-e_y) * (1-(z/softcap)^2)` theo từng
   vocabulary coordinate. TF32 tắt trong proxy; norm và dot reduce bằng FP64.
5. Nếu data parallel, cộng G_S/G_M giữa ranks trước khi calibrate. Sequence
   parallel size khác 1 hiện bị từ chối. Shape head khác giữa ranks cũng bị từ chối.
6. Giải phóng accumulators, replay RNG cho từng microbatch rồi backward với một
   lambda duy nhất. Sau window phục hồi RNG sau lượt đầu: một lần tiến RNG về mặt
   logic. Optimizer cập nhật đúng một lần; checkpoint không chứa graph tạm.

Chung một hệ số chuẩn hóa loss trên window nên hệ số đó triệt tiêu khỏi lambda.
Code không lấy mean các lambda response hay các lambda microbatch.

## Bộ nhớ và guardrails

Hai accumulator FP32 cần `2 * vocab * hidden * 4` bytes. Head Gemma-2-2B
`256000 x 2304` cần khoảng 4.395 GiB cho hai buffer; vocabulary tile giới hạn
transients và số response không tăng dung lượng buffer. Đây chưa phải peak của
training đầy đủ: model, logits, teacher/rollout engines và optimizer cùng dùng GPU.

Full-batch TRUST-B từ chối calibration nếu accumulator vượt nửa free memory,
window thiếu, norm/dot không hợp lệ hoặc input non-finite. Nhánh mới không âm
thầm đổi sang microbatch hay Atomic. CUDA OOM vẫn fatal.

TRUST-R và legacy microbatch giữ policy fallback Atomic có log/counter khi
geometry thất bại trên input hữu hạn. Gram quadratic forms reduce bằng FP64,
norm âm đáng kể và vi phạm Cauchy–Schwarz bị phát hiện. Roundoff nhỏ được đưa
về geometry hợp lệ trước khi tính lambda. Fallback dương vẫn là deviation cần
báo khi đánh giá phương pháp.

## Kiểm chứng và giới hạn

Các test oracle độc lập gồm hand counterexample micro-vs-full, explicit head
autograd với softcap/bias, actor B64/M4 với dropout replay và so cập nhật toàn
bộ tham số. Suite CUDA thêm Gemma-2 BF16 khởi tạo từ config nhỏ và accumulator
đúng kích thước head production. Kết quả và SHA ở
[báo cáo audit / bàn giao](algorithm_audit_fix_20261009.md).

Những test này kiểm tra toán, loss/actor wiring và runtime CUDA. Chúng không đo
quality của model, không thay cho run Qwen-7B → Gemma-2B trên node công ty và
không chứng minh đủ VRAM cho workload dài 4096 khi các engine cùng sống.

Lịch sử quan trọng: run TRUST-B cũ từng OOM ở step 171; pair-Gram giảm peak của
microbatch nhưng không sửa mức aggregate. Không resume checkpoint đó dưới định
nghĩa mới. Bắt đầu trajectory mới từ SFT và giữ nguyên checkpoint/log cũ để audit.
