# Fleet — SimCT training & eval (cập nhật 2026-10-08, sau R85)

Nguồn sự thật cho việc gì đang chạy ở đâu. GPU chỉ dùng đúng ô đã ghi;
đổi ô phải sửa file này trước.

## Đang chạy

| Node | GPU | UUID (đầu) | Run | Seed | Eval | Ghi chú |
|---|---|---|---|---|---|---|
| nlp-core-team-0-0 | 6 | `88a158ed` | trust_b 312 | 42 | TẮT | R56b, TB bật (fallback nhiều ở step dài) |
| nlp-core-team-0-0 | 7 | `2f0fb13c` | eval align-s42 | — | worker riêng | E12, 8 plans |
| embed-8b-training-v3-0-0 | 0 | `ce2da021` | grass 312 | 43 | TẮT | R58, TB bật |
| embed-8b-training-v3-1-0-0 | 0 | `e74dd6de` | trust_b 312 pair-Gram | 42 | TẮT | R85, src a024d3b7, canary cho decomposition |
| embed-8b-training-v2-0-0 | 3 | `eaed83d1` | align 312 (lần 2) | 43 | TẮT | wrapper 1731804, src eed10daf |

## Hoàn thành (chờ/kèm eval)

- grass baseline 312 (8b/GPU3, R23): exit 0. Eval 8/8 xong, milestone + dashboard có.
- grass_chunk 312 (nlp/GPU5, R46): exit 0, run campaign hoàn chỉnh đầu tiên của mode mới. Eval 8/8 xong.
- align 312 seed 42 (8b/GPU0, R44): exit 0. Eval 8 plans xong (E10), worker E11/E12.
- trust_r 312 (v3/GPU0, ~01:11): exit 0, run trust đầu tiên về đích. Fallback lẻ tẻ (fraction cuối 0–9%), λ_mean ~0.055. Eval E13.
- DPCA v10 312 (v3, pilot ngoài): rc=0. Eval 80→312 xong (chain GPU 7).

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
