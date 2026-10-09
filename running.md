# Running — trạng thái runs và eval (snapshot 2026-10-09 tối)

File này là ảnh chụp lúc ghi; `fleet.md` giữ vai trò registry GPU sống.
Quy ước đời toán: CŨ (tới `ccec1966`) vs MỚI (từ `ef49e0ba`). Không so milestone khác đời.

## Train — TOÁN MỚI (đang chạy)

| Run | Node/GPU | Seed | Block | Trạng thái cuối thấy |
|---|---|---|---|---|
| grass 312 | nlp-core / 6 | 42 | R98 | warming (chưa thấy step) |
| align 312 | nlp-core / 4 | 42 | R99 | warming, GPU ăn 6 GB |
| trust_r 312 | nlp-core / 7 | 42 | R100b | vừa launch (host 1374 GB trống) |
| grass_chunk 312 retry | ihbkaiser6 / 3 | 42 | R101c | chờ output (bản có fix tiling) |
| trust_b 312 | ihbkaiser6 / 0 | 42 | R102b | chờ output |
| trust_b 312 pair-Gram | v3-1 / 0 | 42 | R85 | **exit 0** (code cũ a024d3b7, dữ liệu lịch sử) |

## Train — TOÁN MỚI (canary limit 6: 5/5 xanh)

grass R92 ✅ · trust_r R96b ✅ · align R95c ✅ · grass_chunk R93c ✅ · trust_b R94c ✅
(chi tiết keys audit trong log từng canary; trust_b: full_batch=1, 64/64, 16/16, fallback=0)

## Train — TOÁN MỚI (seed 43: chết theo node, chờ relaunch)

- grass-43 R103 (8b/gpu2): chết step 159/312, không exitcode. Checkpoints →160 cứu được.
- chunk-43 R104 (8b/gpu6): chết step 106/312, không exitcode. Checkpoints →100 cứu được.

## Train — TOÁN CŨ (xong, exit 0)

grass s42 (R23) · grass_chunk s42 (R46) · align s42 (R44) · align s43 #2 (8b/gpu3) ·
trust_r s42 · trust_b R56b s42 · DPCA v10 s42 · DPCA rerun s42 (step 300 lúc thấy) ·
grass s43 (17:00, toán cũ — KHÔNG phải seed-43 toán mới, đừng lẫn với R103)

## Train — chết (lý do đã rõ, không đào lại)

- R101 chunk-312 mới: OOM gather 38 GB step 112 → fix tiling `5ca5ac34`, retry R101c.
- R96/R100 (8b-v2): host RAM 96–98% (job hàng xóm).
- Eval-burst hook cũ (4 runs), trust_b R47–R53 (hidden/tiling/env), 4b OOM host.
- R44-align? không — exit 0. Likệt kê đủ ở fleet.md mục cũ.

## Eval — xong (gen+score+summary)

- Đời cũ: grass-baseline, grasschunk, DPCA-v10, align-s42 → union report 21 groups.
- Đời cũ: trust_r E14b 96/96 · trust_b R56b E15b 96/96 (summarize? xem bảng dưới).
- Đời mới: grass-new 8/8 · trust_r-new 8/8 · trust_b-new 8/8 → union report 24 groups.

## Eval — đang chạy

- align-new: gen xong 7 plans (thiếu 312, run chưa xong); scoring A1/A2 (E32a).
- chunk-new: gen xong 4 plans; scoring C1/C2 (E32b, ihbkaiser6).
- trust_b-new scorer lẻ (S84a/b + E29a2/b2 + SUP-A/B): đối chiếu `suplog*.log` trước khi
  launch thêm — đã 2 lần launch mù vào plans chưa tồn tại/chưa xong.

## GPUs (trạng thái cuối thấy)

- nlp-core: 4 (R99) · 5 (?) · 6 (R98) · 7 (R100b) bận; 0/1 không đụng; 2/3 chưa có UUID.
- 8b-v2: DOWN (kể cả đọc) — R103/R104 chết theo, E14b xong trước khi chết.
- 8b còn lại: GPU 3 (align #2 xong), GPU 5 (E14 xong) — cần check lại trước khi dùng.
- v3-0-0: ? (R57 trust_r? grass? — chưa xác minh từ S63). v3-1: R85 xong, GPU rảnh?
- ihbkaiser5: cấm. ihbkaiser6: 4 GPUs, runtime R97 xong; gpu3 R101c, gpu0 R102b.

## Nợ mở (theo thứ tự)

1. Output R101c / R102b / R98 / R99 / R100b (steps đầu + metrics mới).
2. E32a/E32b scoring xong → summarize align-new + chunk-new → report groups 25, 26.
3. Seed 43 toán mới (R103/R104 relaunch khi có GPU + node sống).
4. Publish flow `publish_union.py` ra research_vdt: CHƯA duyệt, chưa chạy.
5. W&B: 5 runs cũ đã lên; runs mới chưa upload (đợi milestones đủ).
