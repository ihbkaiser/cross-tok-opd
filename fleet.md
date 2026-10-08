# Fleet — SimCT training & eval (cập nhật 2026-10-09, sau audit ef49e0ba + fix 7ef62e89)

Nguồn sự thật cho việc gì đang chạy ở đâu. GPU chỉ dùng đúng ô đã ghi;
đổi ô phải sửa file này trước.

## Code hiện tại

- HEAD `7ef62e89` = audit `ef49e0ba` (GRASS pooled-mean, TRUST-B batch-64,
  native chunks, delta-Gram trực tiếp) + fix argv `None` → `"None"`.
- Publish HF `a72f9471`, dual verified. Mọi run/canary mới đi từ đây.
- Chưa review toán đầy đủ: pair-decomposition của tôi (`a024d3b7`) đã bị
  full-batch rewrite của audit thay thế; R85 (pair code) giờ là dữ liệu lịch sử.

## Đang chạy (train, toán mới)

| Node | GPU | UUID (đầu) | Run | Seed | Ghi chú |
|---|---|---|---|---|---|
| nlp-core-team-0-0 | 6 | `88a158ed` | grass 312 mới | 42 | R98, TB bật |
| embed-8b-training-v3-1-0-0 | 0 | `e74dd6de` | trust_b 312 pair-Gram | 42 | R85, code cũ a024d3b7 — dữ liệu lịch sử |
| nlp-core-team-0-0 | 4 | `49d5c3bd` | align 312 mới | 42 | R99 (chờ output) |
| embed-8b-training-v2-0-0 | 3 | `eaed83d1` | trust_r 312 mới | 42 | R100 (chờ output) |

## Canary toán mới (limit 6)

| Mode | Block | Trạng thái |
|---|---|---|
| grass-6 | R92 (nlp6) | ✅ VERIFIED (D≠0, alpha đúng ý audit) |
| trust_r-6 | R96b (ihbkaiser6/gpu3) | ✅ VERIFIED (λ=0.15, calibrate 100%) |
| align-6 | R95c (nlp7) | ✅ VERIFIED |
| grass_chunk-6 | R93c (ihbkaiser6/gpu3) | ⏳ chờ output |
| trust_b-6 | R94c (nlp7) | ⏳ chờ output (code batch-64 mới, rủi ro cao nhất) |

## Hoàn thành (code cũ, kèm eval)

- grass baseline s42 (R23): exit 0. Eval 8/8 + milestone + dashboard.
- grass_chunk s42 (R46): exit 0. Eval 8/8.
- align s42 (R44): exit 0. Eval 8/8.
- align s43 #2 (8b/gpu3, eed10daf): xong (R100 kiểm exitcode lúc launch).
- trust_r s42: exit 0 (fallback 0–9% cuối). Eval E14b chain.
- trust_b R56b s42: exit 0 (fallback dày nửa sau). Eval E15b chain.
- DPCA v10 s42: rc=0. Eval 80→312 xong.
- DPCA rerun s42 (ihbkaiser5): step 300/312 lúc thấy — kiểm exit sau.

## Eval (worker riêng, gen-first)

- grass-baseline / grasschunk / DPCA / align-s42: xong + summaries + union report
  21 groups (`tmp/report_union_20261008.html`).
- trust_r (E14b, 8b/gpu5), trust_b R56b (E15b, nlp/gpu5): đang chạy, gen 67/48.

## Quy ước đang hiệu lực

- Eval in-process ĐỎ toàn fleet (4 vụ scheduler-crash). Worker riêng + gen-first.
- Guardrail số học: warning + atomic fallback loud; loss non-finite fatal; OOM re-raise.
- Mọi run train: TB bật, micro 4, eval TẮT.
- Chunk/align mới: source `xtoken` native (default audit), không projection.
- Block paste: không heredoc lồng nhau; launch nào cũng gate UUID + VRAM + host RAM.
- `numpy` hệ thống là moving part: scoring lệch qual thì update state (verify trước),
  đã có `/tmp/fix_state.py` trên nlp-core.

## Đã chết / nghỉ (giữ để khỏi đào lại)

- grass+eval, align+eval, canary limit-6/10/16 eval-burst: scheduler Triton-crash.
- trust_b step-171 OOM (Gram batch 11.73 GB) → memory guard → pair-Gram → audit rewrite.
- trust_b R47/R49/R51/R53: hidden-None, tiling-gap, TorchMemorySaver (env độc),
  expandable_segments đã revert.
- grass_chunk 4b: OOM host (job lạ). Repro SGLang: đã kill.
- R96 (8b/gpu3): host RAM 96% (job hàng xóm), chuyển canary sang ihbkaiser6.

## GPU người khác / mượn

- 8b/GPU 2: cho mượn. 8b/GPU 1,4,5,6 + 4b/GPU 0: job minhpn19/người khác.
- nlp/GPU 0,1: chưa bao giờ đụng (không có UUID, không gate được).
- ihbkaiser5: cấm dùng (DPCA xong thì thôi). ihbkaiser6 (4 GPUs): sân canary mới.
