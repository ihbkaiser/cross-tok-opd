# Chuẩn bị evaluation SimCT / MP-OPD

Cập nhật triển khai: [contract runner](../scripts/evaluation/CONTRACT_README.md)
đã có prepare/generate/score/summarize và CPU preflight. Phần dưới là audit ban
đầu của legacy evaluator; các sai khác chưa được coi là đã biến mất.
`paper-spec` ghi rõ phần tái dựng MBPP và revision LCB chưa được paper công bố.

Trạng thái: protocol và inventory đã chuẩn bị; chưa có runner đạt đầy đủ paper,
chưa chạy inference/scoring hoặc tải dataset. Không chạy eval_all_monitor.sh trên
node training: script mặc định 8 GPU và có nhánh pkill SGLang rộng.

## Protocol nguồn và convention của lượt tái lập

Nguồn: [SimCT v1 Appendix C](https://arxiv.org/html/2605.07711v1#A3),
[LiveCodeBench official](https://github.com/LiveCodeBench/LiveCodeBench),
[Math-Verify official](https://github.com/huggingface/Math-Verify).

Theo paper: GSM8K, MATH-500, MBPP, LCB-v6; zero-shot, T=0.6, top-p=0.95,
một completion/câu/lượt, năm evaluation repeats. Token cap lần lượt
4096/4096/2048/4096. Math dùng boxed answer và math_verify; code dùng official
evaluators trong sandbox, 10 giây/test, memory cap 128 MB. Báo mean/std năm lượt
và trung bình không trọng số bốn benchmark. Không nhầm pass@1 với pass@5.

Convention đề xuất (không phải thông số tác giả công bố): evaluation seeds
42..46 và sample std ddof=1, lưu thêm population std. Cùng seed list, prompts,
dataset IDs, grader, budget cho mọi checkpoint. Năm eval seeds không đo variance
giữa năm training seeds. SGLang/backend có thể còn nondeterminism: lưu version,
request seed và server flags; không hứa tái lập từng token chỉ bằng seed.

## Ma trận checkpoint

Protocol JSON chứa bốn candidate: SFT baseline, SimCT 2ep, atomic MB4 và fixed-2
MB4 mới. Tổng 4 x 4 x 5 = 80 checkpoint/benchmark/repeat cells.
SimCT100, base và teacher có thể thêm thành nhóm mở rộng; đường dẫn chưa chốt.
Hai run MP lịch sử có finite parity bypass phải nằm nhóm historical riêng.
Không gộp với MP MB4 mới hoặc dùng tên run cũ cho kết quả mới.

Đường dẫn từ handoff chỉ là candidate. Trước eval cần kiểm tra checkpoint HF
đầy đủ (config, tokenizer, toàn bộ shards theo index), source/launch config,
summary completed/312 và exit0 cho full MP. Không chọn theo glob/latest hoặc
tuổi file: có thể vô tình chọn canary/partial checkpoint. Resolve hai MP từ đúng
launch log trong protocol, rồi đóng băng manifest trước inference. Hash weights
sau training, tránh I/O cạnh tranh trên shared storage lúc đang save.

## Sai khác đã kiểm tra ở source f0c2181

| Phần | Source hiện tại | Cần xử lý trước khi gọi chuẩn paper |
|---|---|---|
| GSM8K | Prompt ####, parse float/tolerance | Chốt boxed prompt và math_verify theo paper; ghi rõ mâu thuẫn code/paper |
| MATH-500 | String/numeric/SymPy fallback | Pin math_verify, kiểm tra extraction với gold fixtures |
| Repeats | Không có CLI/request seed | Seed tường minh, 5 output directories độc lập |
| Dataset | DATASETS_DIR rỗng; không revision pin | Data manifest bất biến, split/count/ID/hash |
| MBPP | Custom multiprocessing exec; timeout toàn problem | Chốt variant/harness, sandbox và timeout đúng contract |
| LCB | Custom stdin/stdout runner, không official grader | Official evaluator; hỗ trợ cả functional tasks, kiểm tra private tests |
| Code memory | Không thấy RLIMIT trong evaluator | Áp 128 MB ở sandbox đã kiểm chứng; không gọi subprocess là sandbox |
| Errors | HTTP lỗi chuyển thành output rỗng | Phân biệt infra failure với model incorrect; fail cell nếu thiếu response |
| Resume | Bỏ qua theo file metrics tồn tại | Khớp checkpoint/config/data hash; không reuse kết quả khác seed |
| README CLI | --n, --max_tokens không có trong parser | CLI thực hiện dùng --max_new_tokens; không copy lệnh README cũ |

prepare_lcb_data.py ghi nhầm v6 kết thúc 2024-07-01. Official README định nghĩa
release_v6 gồm 1055 bài May 2023–Apr 2025; 612 bài tới Jul 2024 là release_v3.
Giữ release_v6 và pin HF revision, không dùng release_latest hoặc tự cắt theo
comment cũ. Paper không chỉ rõ MBPP original/sanitized và exact harness commit;
chưa được tự gọi một lựa chọn bất kỳ là tái lập chính xác.

## Inventory an toàn trong lúc train

experiments/runai/eval_inventory.py chỉ đọc metadata/files nhỏ, không import
Torch, không dùng CUDA, không tải/cài gì, không giải nén hay chạy generated code.
Chạy bằng runtime đã có sau khi các file chuẩn bị được chuyển tới node:

```bash
/opt/venvs/simct-b200/bin/python experiments/runai/eval_inventory.py \
  --data-root /absolute/path/to/eval-data
```

Data root có bốn thư mục gsm8k, math500, mbpp, live-code-bench-v6; nếu chưa biết
thì bỏ --data-root. Output chỉ là inventory, không chứng minh eval-ready.
Chưa xác nhận data tồn tại trên công ty; chưa cấp lịch chạy GPU tự động.
Không cập nhật checkout đang dùng bởi training; giữ preparation local cho tới
khi training kết thúc hoặc dùng một checkout eval riêng được xác định rõ.

## Bước triển khai tiếp theo

1. Thu inventory runtime và data hiện có. Chốt revisions/splits và MBPP harness;
   nếu phải tải/cài thì thực hiện trong môi trường remote riêng, không sửa venv
   đang phục vụ training. Giữ proxy công ty, không bypass policy.
2. Tách generation (GPU) khỏi scoring (CPU sandbox không secrets/network/shared
   storage ghi được). Pin prompt text/hash, grader revision, container/runtime.
3. Implement runner với endpoint identity check, per-request seed, strict errors,
   raw responses/finish_reason/token counts và output model/dataset/seed riêng.
4. Kiểm chứng math gold cases và code correct/wrong/timeout/memory fixtures;
   test phục hồi không duplicate/mất câu. CPU fixtures trước GPU smoke.
5. Sau training và khi được giao chạy: smoke một tập nhỏ, rồi đủ 5 repeats.
   Summary lưu pass/fail từng item, N, mean/std, infra failures, truncation;
   chỉ báo overall khi đủ cả bốn benchmark. Không tự upload W&B.

Chỉ sau các bước này mới gọi kết quả là protocol-aligned evaluation. Muốn nói
exact paper reproduction còn phải giải quyết những thông số tác giả chưa pin.
