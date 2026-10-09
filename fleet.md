# Fleet — SimCT training & eval (cập nhật 2026-10-09, R98-R102b + canary full-board)

Nguồn sự thật cho việc gì đang chạy ở đâu. GPU chỉ dùng đúng ô đã ghi;
đổi ô phải sửa file này trước.

## Hai đời toán — KHÔNG trộn lẫn khi đọc số

**TOÁN CŨ** (mọi commit tới `ccec1966`, gồm pair-Gram `a024d3b7` của tôi):
GRASS distortion ≈ 0 (bug pooled-mean), TRUST-B calibrate theo microbatch-4,
chunk/align chạy fixed-run baseline, guardrails warning+fallback của tôi.
Mọi run hoàn thành + mọi milestone trong union report hiện tại đều thuộc đời này.

**TOÁN MỚI** (từ `ef49e0ba`, + fix argv `7ef62e89`):
GRASS pooled-mean đúng (D≠0), TRUST-B một λ cho batch-64 (streaming 2 lượt),
chunk/align native xtoken, delta-Gram trực tiếp, TRUST-B từ chối calibration lỗi
thay vì fallback atomic. Canary limit-6 + mọi run 312 từ đây đều thuộc đời này.

Quy tắc: so sánh milestone CHỈ trong cùng đời toán. Đặt baseline đời cũ cạnh
run đời mới để kết luận "hơn/thua" là sai phương pháp — khác estimand.
Eval và report của đời mới dùng case/group tên riêng (hậu tố `-new`), không ghi
đè lên groups đời cũ trong union report.

## Đang chạy — TOÁN MỚI (312)

| Node | GPU | UUID (đầu) | Run | Seed | Ghi chú |
|---|---|---|---|---|---|
| nlp-core-team-0-0 | 6 | `88a158ed` | grass 312 | 42 | R98, warmup |
| nlp-core-team-0-0 | 4 | `49d5c3bd` | align 312 | 42 | R99, warmup |
| nlp-core-team-0-0 | 7 | `2f0fb13c` | trust_r 312 | 42 | R100b (chuyển từ 8b-v2 chết RAM host) |
| ihbkaiser6-0-0 | 3 | `c8e6cc11` | grass_chunk 312 | 42 | R101, warmup, native xtoken đầu tiên |
| ihbkaiser6-0-0 | 0 | `93d70d94` | trust_b 312 | 42 | R102b (chờ output) |
| embed-8b-training-v3-1-0-0 | 0 | `e74dd6de` | trust_b 312 pair-Gram | 42 | R85, code cũ a024d3b7 — dữ liệu lịch sử |

## GIÁN ĐOẠN 8b-v2 (node DOWN, bị kill) — thiệt hại và cứu hộ (2026-10-09)

- R103 grass-43 (gpu2): chết step 159/312, không exitcode. Đốt ~3.5 GPU-giờ.
  Cứu được checkpoints 20→160 → milestones 40/80/120/160 eval được.
- R104 chunk-43 (gpu6): chết step 106/312, không exitcode. Đốt ~2.4 GPU-giờ.
  Cứu được checkpoints → milestones 40/80 eval được.
- E14b trust_r eval (gpu5): HOÀN THÀNH trước khi node chết (96/96 gen+scored).
- 8b-v2 cấm mọi ops tới khi SSH sống lại (kể cả đọc). Không relaunch 2 runs trên
  node này; chờ GPU rảnh node khác.

## Canary toán mới (limit 6) — ĐỦ BỘ 5/5 XANH

| Mode | Block | Trạng thái |
|---|---|---|
| grass-6 | R92 (nlp6) | ✅ VERIFIED (D≠0, alpha đúng ý audit) |
| trust_r-6 | R96b (ihbkaiser6/gpu3) | ✅ VERIFIED (λ=0.15, calibrate 100%) |
| align-6 | R95c (nlp7) | ✅ VERIFIED |
| grass_chunk-6 | R93c (ihbkaiser6/gpu3) | ✅ VERIFIED |
| trust_b-6 | R94c (nlp7) | ✅ VERIFIED (full_batch=1, 64/64, 16/16, fallback=0) |

## Hoàn thành — TOÁN CŨ (kèm eval đời cũ, không so với đời mới)

- grass baseline s42 (R23): exit 0. Eval 8/8 + milestone + dashboard.
- grass_chunk s42 (R46): exit 0. Eval 8/8.
- align s42 (R44): exit 0. Eval 8/8.
- align s43 #2 (8b/gpu3, eed10daf): exit 0 full 312.
- trust_r s42: exit 0 (fallback 0–9% cuối). Eval E14b chain (gen 67).
- trust_b R56b s42: exit 0 (fallback dày nửa sau). Eval E15b chain (gen 48).
- DPCA v10 s42: rc=0. Eval 80→312 xong.
- DPCA rerun s42 (ihbkaiser5): step 300/312 lúc thấy — kiểm exit sau.

## Eval — TOÁN CŨ (worker riêng, gen-first; report đời cũ đóng băng ở 21 groups)

- grass-baseline / grasschunk / DPCA / align-s42: xong + summaries + union report
  21 groups (`tmp/report_union_20261008.html`).
- trust_r (E14b): eval XONG 96/96 (hoàn thành trước khi 8b-v2 chết).
- trust_b R56b (E15b, nlp/gpu5): đang chạy.

## Quy ước đang hiệu lực

- Eval in-process ĐỎ toàn fleet (4 vụ scheduler-crash). Worker riêng + gen-first.
- Guardrail số học: warning + atomic fallback loud; loss non-finite fatal; OOM re-raise.
- Mọi run train: TB bật, micro 4, eval TẮT.
- Chunk/align mới: source `xtoken` native (default audit), không projection.
- Block paste: không heredoc lồng nhau; launch nào cũng gate UUID + VRAM + host RAM (>200 GB).
- `numpy` hệ thống là moving part: scoring lệch qual thì update state (verify trước),
  đã có `/tmp/fix_state.py` trên nlp-core.
- 8b-v2 cấm launch train (host RAM ~98% thường trực, Ray OOM 2 lần R96/R100).

## Đã chết / nghỉ (giữ để khỏi đào lại)

- grass+eval, align+eval, canary limit-6/10/16 eval-burst: scheduler Triton-crash.
- trust_b step-171 OOM (Gram batch 11.73 GB) → memory guard → pair-Gram → audit rewrite.
- trust_b R47/R49/R51/R53: hidden-None, tiling-gap, TorchMemorySaver (env độc),
  expandable_segments đã revert.
- R95/R95b align canary: bug argv `str(None)` → fix `opts_to_argv` + regression test.
- grass_chunk 4b: OOM host (job lạ). Repro SGLang: đã kill.
- R96/R100 (8b/gpu3): host RAM 96-98% (job hàng xóm), chuyển sang node khác.

## GPU người khác / mượn / cấm

- 8b/GPU 2: cho mượn. 8b/GPU 1,4,5,6 + 4b/GPU 0: job minhpn19/người khác.
- 8b toàn node: cấm launch train (RAM host), chỉ đọc log.
- **huyhq21-membedding-v1-0-0 = CÙNG HARDWARE với 8b-v2** (GPU 1/2/3 trùng UUID
  GPU 5/6/7 bên 8b-v2; GPU 0 trùng GPU 4). Jobs 2 bên thấy nhau — đặt việc phải
  tính cả 2 tên. Runtime R107 xong (verify 8/8, smoke pass). GPU 3 bận (job khác).
- nlp/GPU 0,1: chưa bao giờ đụng (không có UUID, không gate được).
- ihbkaiser5: cấm dùng. ihbkaiser6 (4 GPUs): sân mới, runtime R97 xong.
