# RTX 3090 native uv — kết quả setup và kiểm tra X-Token

Ngày: 2026-09-03. Thời điểm evidence cuối: khoảng 04:34 UTC / 11:34 Việt Nam.

## Kết luận

**CUDA và loss-kernel smoke PASS; trainer/model thật CHƯA được xác minh.**
Chưa có căn cứ để nâng lên 2×3090 hay so hiệu quả huấn luyện với TPU.

Đã dựng môi trường uv riêng tại `/workspace/xtoken-native/NeMo-RL/.venv`,
không sửa driver, `/venv/main`, dịch vụ có sẵn, Tunix, Kaggle hoặc Modal.
Không commit/push. Checkout NVIDIA sạch và cố định tại
`13a10647ebbf0f940d2b06ea41800b3f2fb46099`; submodule SHA và hash lock nằm trong
[environment.json](artifacts/environment.json).

Máy có 1 RTX 3090 24 GiB, driver 580.95.05, compute capability 8.6;
cgroup khoảng 52.74 GiB RAM và quota CPU tương đương 23.04 CPU.
Môi trường dùng Python 3.13.14, PyTorch 2.11.0+cu130, Transformers 5.12.1,
Ray 2.56.1, theo lock upstream. Không tự chuyển toàn bộ phép tính sang FP32:
smoke SDPA dùng activation BF16, parameter FP32; sparse projection cố ý FP32
theo helper gốc của NVIDIA, kể cả khi nằm trong BF16 autocast.

## Bằng chứng đã chạy

| Kiểm tra | Kết quả | Phạm vi |
|---|---|---|
| `uv sync --locked`, base + build + test | Cài 255 package thành công | Không gồm optional Automodel/Megatron/vLLM/SGLang |
| CUDA import, nhận diện GPU, BF16 | PASS | Runtime phần cứng/phần mềm |
| `uv pip check` | **4 incompatibility; exit 1** | Do override trong lock upstream, xem bên dưới |
| Unit tests X-Token gốc | 124 passed, 2 skipped, 9 warnings; 63.44 s | Hai test TP/CP cần ≥2 GPU được skip |
| Cross-tokenizer loss tests gốc | 9 passed, 76 deselected, 1 warning; 44.96 s | CPU/reference tests |
| Sparse GPU forward/backward so với dense FP32 reference | PASS | Sai số cực đại forward 4.77e-7, backward 2.98e-8 |
| BF16 SDPA + 3 bước AdamW | PASS | Gradient finite/nonzero, parameter thay đổi |
| Large-vocabulary P-KL GPU stress | PASS về execution/autograd | Mỗi shape 3 optimizer steps, dữ liệu tổng hợp |
| Quyền tải Llama-3.2-1B | **DENIED** | HF CLI: `Access denied. This repository requires approval.` |
| Tải config Qwen3-1.7B | PASS | Chỉ config, chưa tải weights |
| Audit exact-value token | 0 findings / 41 files | Script, diagnostics, task HF cache; không scan dependency cache/upstream |

Hai shape P-KL tổng hợp dùng vocab student 128256 và teacher 151936:

- Student `[1,128,128256]`, teacher `[1,160,151936]`: peak allocated 0.561 GiB.
- Student `[1,256,128256]`, teacher `[1,288,151936]`: peak allocated 1.101 GiB.

**Các số VRAM này chỉ là loss + logits + optimizer của harness, KHÔNG gồm
language model.** Chúng không chứng minh Llama/Qwen vừa VRAM, không phải token/s,
không được suy ra tốc độ/chi phí training hoặc khả năng scale 2 GPU.
Sparse projection, chunk alignment và logits đều nhân tạo; không có tokenizer
thật, pretrained weights, corpus, Ray CUDA IPC, checkpoint hoặc đánh giá held-out.

Lưu ý khoa học: giá trị P-KL của stress test là khoảng -0.25 đến -0.26.
Helper upstream lấy trung bình **log-probability** teacher theo chunk mà không
renormalize teacher sau đó; với logits ngẫu nhiên khác nhau theo token, đầu vào
KL không còn là log của một phân phối chuẩn hóa. Vì vậy PASS ở đây chỉ xác nhận
execution/gradient, **không** xác nhận KL không âm, loss calibration hay cải thiện
mô hình. Không sửa công thức upstream trong task setup này.

## Dependency và phần chưa hoàn tất

`uv pip check` báo yêu cầu của Torch khác các phiên bản đã được upstream override:

| Package | Torch yêu cầu | Lock cài |
|---|---|---|
| setuptools | <82 | 83.0.0 |
| nvidia-cudnn-cu13 | 9.19.0.56 | 9.20.0.48 |
| nvidia-nccl-cu13 | 2.28.9 | 2.30.7 |
| nvidia-nvshmem-cu13 | 3.4.5 | 3.7.2 |

Đã đối chiếu `[tool.uv].override-dependencies` ở pyproject gốc, không âm thầm
downgrade/relock. CUDA smoke pass không xóa bỏ cảnh báo metadata này và không
xác minh NCCL đa GPU. Script install có exit 1 **ở bước pip check sau khi cài và
CUDA probe thành công**, không phải OOM hay thất bại tải/cài Torch.

Trainer X-Token DTensor-v2 đăng ký worker bằng `--extra automodel`. Extra đó
**chưa cài**, chưa kiểm tra import/khởi tạo worker hoặc training end-to-end.
HEAD request cho wheel FlashAttention pin chính xác trả HTTP 200; đây chỉ là
bằng chứng file tồn tại, không phải kiểm tra ABI/khả năng chạy trên Ampere.

## Gate Hugging Face và bước tiếp

Đã dùng duy nhất `HF_TOKEN` từ file user chỉ định trong WSL
`/home/tung/Collaborative-MORL/.secrets/talapas_secrets.env`.
Token đi qua stdin SSH, chỉ tồn tại trong memory/environment của tiến trình cần
dùng; không login lưu token, không tạo token file, không sửa source credential.
`hf/token` và `hf/stored_tokens` trong task đều không tồn tại sau kiểm tra.

- Llama revision đã xác định: `4e20de362430cd3b72f300e6b0f18e50e7166e08`.
- Qwen revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- Llama config bị từ chối; không tải weights, không chạy lại và không thay model.

User cần xin/kiểm tra quyền ở
[meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B),
đồng thời bảo đảm token có quyền đọc gated model. Khi gate đó qua, hoàn thiện
Automodel worker environment và chạy một smoke huấn luyện có giới hạn trên
1×3090, với model/data/projection pinned, loss/gradient finite, VRAM/timing thực,
checkpoint và exit status. Chỉ sau đó quyết định 2×3090. Đây chưa phải OPD hoặc
reproduction kết quả SimCT.

## Trạng thái bàn giao

GPU trở lại 1 MiB / utilization 0% sau test; không còn training đang chạy.
Instance **không bị stop/destroy** và vẫn tính tiền theo trạng thái thuê.
Disk cuối khoảng 9.8 GiB dùng, 42 GiB trống. Workspace không có persistent volume:
destroy/recycle sẽ mất môi trường remote. Script, JSON/XML/log và checksum đã được
giữ ở local; không tải model/checkpoint nặng về máy local.

Đã giữ [README](README.md), script tái tạo và [artifact index](artifacts/sha256.json).
Không có silent retry của test/training. Bước dry-run dependency ban đầu im log
dài, đã được kiểm tra và kết thúc với exit 0; cài thực tế là một attempt riêng,
được giữ nguyên log cùng kết quả pip check.

## Material passport

- Origin skill: `academic-research-suite` / experiment-agent; `hf-cli` cho kiểm tra tải config.
- Mode: run, một attempt có timeout/log cho từng phép kiểm tra; không delegate.
- Verification: source-pinned environment và runtime diagnostics; scientific reproduction chưa đạt.
- Scope limitation: synthetic kernel tests, gated Llama access denied, optional worker chưa cài.
- Evidence: machine-readable results và logs trong `artifacts/`; không có tuyên bố training/model quality.
