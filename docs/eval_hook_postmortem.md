# Ghi chú sự cố train/eval (7–8/10/2026)

## 1. Eval in-process giết run — đã về hưu, không bật lại

Triệu chứng (4 lần giống hệt): ~25s sau save checkpoint, scheduler SGLang
văng Triton trong prefill (`alloc_for_extend` → `write_cache_indices`),
router trả 503, run chết ở `wakeup` kế tiếp. Hook bound (`try/except` +
skip-noisily) vẫn không cứu được vì chết là server, không phải hook.

- Các biến thể đã thử và đều chết: chat trực tiếp burst-32, router burst lớn,
  router chunk-16, canary limit-6.
- Server sạch sống mọi burst (repro R30–R32); chỉ server training-state chết.
  Cơ chế chính xác trong SGLang chưa xác định — không đốt thêm run để đoán.
- Quyết định: cờ `MP_EVAL_ON_CKPT` đỏ toàn fleet. Eval chuyển sang worker
  riêng đọc checkpoint đã lưu (`plan_live_checkpoint.py` + `eval_queue worker`
  + `score-spool`). Checkpoint của run chết vẫn eval được.
- Run chết: grass+eval GPU0-8b (step 40), align+eval GPU2-8b (step 40),
  canary limit6 GPU2-8b (step 2), canary chunk-16 GPU0-8b (step 2).

## 2. Launch SGLang trần thiếu CPATH — server chết lúc khởi động

Triệu chứng (ihbkaiser, T9): flashinfer JIT `nvcc fatal: #include <nvrtc.h>,
compilation terminated`, server die trước khi train.
Nguyên nhân: launch bằng `python -m sglang.launch_server` trực tiếp, thiếu
`CPATH=/opt/venvs/simct-b200/lib/python3.12/site-packages/nvidia/cu13/include`
mà wrapper `experiments/runai/python-b200-host.sh` vẫn set (dòng 19-21).
Header có sẵn trong venv, chỉ là `nvcc` không thấy.

Quy tắc: không bao giờ launch SGLang/process train bằng tay trần — luôn đi
qua `python-b200-host.sh` (hoặc export đủ bộ biến của nó). T9b thêm đúng một
dòng CPATH là xanh (server ready 110s, request `stop`, dọn sạch).

## 3. Việc liên quan đang mở (không phải lỗi, nhắc để khỏi quên)

- `torch_memory_saver` từ chối `expandable_segments`: đã revert, giữ mặc định.
- trust_b OOM step dài (Gram batch 11.73 GB): có memory guard + fallback atomic
  từ `affabab6`; OOM vẫn re-raise theo thiết kế.
- Noble-lib/GLIBCXX_3.4.32 (DPCA pilot): lib hệ thống jammy max 3.4.30;
  noble đòi glibc mới hơn host. Chưa có đáp án cuối — nhưng đường train chính
  (run_single_gpu.sh) chưa bao giờ cần symbol đó trên bất kỳ node nào.
