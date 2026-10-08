# Fleet — SimCT training & eval (cập nhật 2026-10-08)

Nguồn sự thật cho việc gì đang chạy ở đâu. GPU chỉ dùng đúng ô đã ghi;
đổi ô phải sửa file này trước.

## Đang chạy

| Node | GPU | UUID (đầu) | Run | Seed | Eval | Ghi chú |
|---|---|---|---|---|---|---|
| nlp-core-team-0-0 | 5 | `d9124ada` | grass_chunk 312 | 42 | TẮT | R46, TB bật |
| nlp-core-team-0-0 | 6 | `88a158ed` | trust_b 312 | 42 | TẮT | R56b, TB bật |
| nlp-core-team-0-0 | 7 | `2f0fb13c` | eval DPCA | — | worker riêng | Không phải train |
| embed-8b-training-v3-0-0 | 0 | `ce2da021` | grass 312 | 43 | TẮT | R58, TB bật |
| embed-8b-training-v2-0-0 | 0 | `84ea467a` | align 312 | 42 | TẮT | R44 xong exit 0, GPU đã sang người khác |
| embed-8b-training-v2-0-0 | 3 | `eaed83d1` | align 312 (lần 2) | ? | TẮT | wrapper 1731804, src eed10daf, ~5.5h tuổi lúc phát hiện |

## Quy ước đang hiệu lực

- Eval in-process (cờ `MP_EVAL_ON_CKPT`) ĐỎ toàn fleet sau 4 vụ scheduler-crash
  cùng chữ ký. Eval chỉ chạy bằng worker riêng đọc checkpoint đã lưu.
- Guardrail số học ở chế độ warning + atomic fallback (từ `47a0d0bf`):
  parity fail và metric non-finite không giết run; loss non-finite vẫn fatal.
- Mọi run train: TensorBoard bật, micro 4, chunk source `run` (baseline).
- Run mới luôn từ commit có memory guard (`gram_affordable`, từ `affabab6`).

## Đã chết / nghỉ (giữ để khỏi đào lại)

- grass+eval, align+eval (8b): chết step 40, scheduler Triton-crash do eval burst.
- trust_b (nlp-core): sống 171 step metrics khỏe, chết OOM ở step dài bất thường
  (Gram batch 11.73 GB). Đẻ ra memory guard + expandable_segments đã revert.
- grass_chunk (4b): chết lúc init vì RAM host bị job lạ ăn (kill-newest).
- Canary limit-6 (8b/GPU 2): chết cùng chữ ký eval burst.
- Repro SGLang (8b/GPU 0): đã kill, GPU đã trả.

## GPU cho người khác mượn / bận

- 8b/GPU 2: đã cho mượn.
- 8b/GPU 1,4,5,6 + 4b/GPU 0: job của minhpn19 / người khác, không đụng.
