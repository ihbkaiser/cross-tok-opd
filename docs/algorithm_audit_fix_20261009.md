# GRASS / ALIGN / TRUST: bản sửa và bàn giao GPU 3

Đích do người dùng xác nhận: `root@embed-8b-training-v2-0-1:/workspace#`, GPU 3.
Giữ optimizer batch 64, microbatch 4. Chưa launch training trên node công ty.

## Những lỗi đã sửa

| Lỗi | Sửa / regression phân biệt |
|---|---|
| GRASS-DP dùng từng `r_i` thay pooled mean của candidate, làm D≈0 | Một scalar `rbar_c` trên toàn span. Với r=[0,10], w=[1,1], H=I, sigma²=1: D=50, V=1, alpha=.02, gradient loss=[.1,9.9]. |
| TRUST-B tính lambda cho micro4 thay toàn batch64 | Hai lượt forward, stream tổng G_S/G_M rồi một lambda cho 16 microbatch; oracle autograd kiểm cả lambda và parameter update. |
| Native Chunk/ALIGN bị runner đổi thành fixed-run baseline | Default `MP_GRASS_CHUNK_SOURCE=xtoken`; TokenAligner.align dùng tokenizer strings và không cần asset vocabulary projection không được nó đọc. |
| Noise EMA bỏ quan sát MAD=0 hợp lệ | Quan sát đủ pairs với variance=0 cập nhật EMA và bias-correction counter. |
| Gram của token tự tin mất norm do cancellation FP32 | Dựng delta trực tiếp trước inner product; head bias trừ label và reduce token-to-atom đúng một lần. |
| Chunk cosine dùng geometry khác giữa tử/mẫu | Cùng block mask cho Atomic và shrunk; identity cosine=1. |
| Boundary singleton/cross-chunk zero bị diễn giải sai | Chỉ mark adjacency thực sự trong chunk; geometry ngoài block được đếm undefined. |
| Atom có token chưa align vẫn bị gán chunk | Atom thiếu coverage hoặc có token id âm được cô lập singleton. |
| Singleton noise trace bị đặt 0 | Giữ trace; D/V/cost/alpha của identity span đúng 0. |
| ALIGN sqrt quadratic âm nhỏ trở thành complex | Clamp roundoff về 0, raise ValueError với norm âm đáng kể. |
| TRUST silently clamp norm âm / geometry vi phạm bound | Reduce quadratic FP64, kiểm norm âm và Cauchy–Schwarz. |
| TRUST pair memory guard sau concat | Guard trước allocation `torch.cat`. Full-batch mới không concatenate response logits. |
| Sparse metric keys/order khác giữa ranks làm collective sai | Union schema canonical, một packed reduction có count; test hai process Gloo với key thiếu/đảo thứ tự. |

## Bằng chứng

- Modal CPU: **170 passed, 3 skipped**, PyTorch `2.14.1+cpu`, Python `3.12.10`.
  [App CPU](https://modal.com/apps/kieusontung6/main/ap-x220SwnCJeQoUBEPED18j6).
  Ba skip là một test Transformers và hai test chỉ chạy CUDA.
- Launcher trên Windows, toolchain có sẵn: **4 passed**, không cài dependency.
- Modal CUDA/H100, profile `billionaireproject8`: **173 passed, 0 skipped**,
  Python `3.12.12`, PyTorch `2.11.0+cu130`, CUDA `13.0`.
  [App CUDA](https://modal.com/apps/billionaireproject8/main/ap-vwbEIDhmEjIf3sT6fRGpcE).
  Gemma-2 BF16 B64/M4: lambda `0.003981471993029118`, max gradient error=0,
  max parameter error=0, dropout replay bitwise exact. Không gọi đây là B200 pass.
- B200 trước đó: **172 passed, 1 failed**. Test fail dùng oracle `p-lr*g`
  với hai lần làm tròn BF16; đã sửa oracle dùng cùng SGD `add_(grad, alpha=-lr)`
  và kiểm riêng gradient exact trước khi cập nhật. Bản oracle sửa đã pass H100;
  chưa rerun bản này trên B200.
  [App B200](https://modal.com/apps/kieusontung6/main/ap-FqbPJiMJTBNIvJNpIFn6fY).
  Head production `256000 x 2304` đã pass B200, peak thêm `4.63700008392334` GiB,
  lambda `0.06451612531088799` so oracle `0.06451612903225806`.
- Runtime B200 pin:
  `docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f`.

Gói bàn giao giữ source bytes, SHA-256, app receipts và raw test logs. Run hỏng
cũng được giữ riêng. Profile mặc định `kieusontung8` bị spend limit; CPU/B200 dùng
`kieusontung6`, sau khi người dùng cho phép đổi account đã chạy CUDA bằng
`billionaireproject8` qua billing guard. B200/H100 được đặt theo
[GPU fallback của Modal](https://modal.com/docs/guide/gpu); receipt ghi GPU thực tế.

Commit của task này không ký, theo quyền người dùng cấp sau khi GPG báo
`Unusable secret key`. Không thay cấu hình Git/GPG. Gói ghi commit và SHA từng
file; launcher cũng hỗ trợ source identity `uncommitted-tree:<hash>` nếu một
gói khác chưa có commit, để tránh gắn nhầm hash commit cũ cho code mới.

## Chạy trên node

### Bổ sung: OOM của native GRASS-Chunk ở full run

Run `qwen-gemma-mp_opd-grass_chunk-gpu3-limit312-20261008-201848-22869`
trên source `7ef62e89` gặp OOM sau step 112: gather trong
`grass_chunk_tables` yêu cầu **38.11 GiB**, GPU còn **22.17 GiB**.
Nguyên nhân là tensor FP64 `[num_chunks, max_chunk_length, max_chunk_length]`:
một native chunk dài làm tất cả chunk ngắn bị pad lên cùng kích thước. Chia
nhóm cố định 32 nhưng giữ `max_chunk_length` toàn partition vẫn chưa chặn được lỗi.

Bản vá dùng padding cục bộ cho nhóm tối đa 32 chunk ngắn (dài tối đa 256),
còn chunk dài được reduce bằng view và tile **256 x 256** trên cả hai chiều.
Không cắt/split native partition, không đổi FP64, công thức D/V/alpha hoặc
credit conservation. Thứ tự cộng floating point của nhánh tiled có thể khác.
Một block batched tối đa **16 MiB**, một product tile FP64 tối đa **0.5 MiB**;
các tensor trung gian cùng tồn tại làm peak cao hơn các con số từng tensor này.
Dense Gram và các allocation khác vẫn có chi phí O(n²); đây là giới hạn workspace
của phép reduce, không phải bảo đảm toàn bộ training không bao giờ OOM.

Kiểm chứng source sau vá, cùng runtime digest đã ghi ở trên:

| GPU thực tế | Kết quả | Thời gian pytest | Peak thêm của ca chunk dài |
|---|---|---|---|
| [H100](https://modal.com/apps/billionaireproject8/main/ap-uS9jI9w6dWOQ11vyIipITV) | 178 passed, 0 failed, 0 skipped | 21.96 s | 104887296 bytes |
| [B200](https://modal.com/apps/billionaireproject8/main/ap-0ln8U37IlyZOw6hrw97fYC) | 178 passed, 0 failed, 0 skipped | 13.58 s | 104887296 bytes |

Ca memory dùng n=2560, một chunk dài 2048 và 512 singleton; peak thêm khoảng
100.03 MiB đã bao gồm symmetrization của dense Gram. Regression khác so D/V
với matrix oracle độc lập cho chunk dài 513 xen nhiều chunk ngắn/ragged trên
CPU và CUDA, kiểm finite credit/conservation, và chặn padded 3-D allocation quá lớn.
Bộ test cũng kiểm TRUST-B Gemma-2 BF16 B64/M4: gradient và parameter error đều 0,
dropout replay exact trên cả H100/B200. Đây là test implementation, chưa phải
replay batch làm run cũ chết hoặc full campaign Qwen/Gemma 312 updates.

Raw logs và source SHA-256 được giữ tại
`remote_artifacts/grass_chunk_workspace_20261009/{test_cuda.log,test_b200.log,verification.json}`.
Chưa publish hoặc thay source của run đang chạy trên node công ty.

### Bàn giao hiện có

Giải nén gói source vào một thư mục mới. Không ghi đè clone đang phục vụ run cũ.
`REF` là file `launch-config.json` hoặc `campaign.json` của campaign đúng cặp
Qwen teacher → Gemma SFT; launcher đọc ba đường dẫn asset từ file này.
`NODE_WRAPPER` là wrapper runtime đã chạy được trên chính node v2-0-1. Nếu wrapper
cần `MP_RUNTIME_DIR`, export đúng runtime hiện có của node trước khi launch; không
copy default của node v3. Gói không chứa model/dataset hoặc image mới.

Block H1 — chạy trên node embed-8b-training-v2-0-1 — xem đúng GPU và tiến trình

```bash
(
  nvidia-smi --id=3 --query-gpu=index,uuid,name,memory.used,memory.total --format=csv
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
)
echo "--- het Block H1, shell con song ---"
```

Block H2 — chạy trên node embed-8b-training-v2-0-1 — xem kế hoạch, chưa launch

```bash
(
  SRC='/duong/dan/source-da-giai-nen'
  REF='/duong/dan/campaign-cu/launch-config.json'
  NODE_WRAPPER='/duong/dan/wrapper-runtime-da-qualify-cua-node-v2.sh'
  OUT='/duong/dan/run-moi/trustb-audit-s42-canary2'
  python3 "$SRC/experiments/runai/launch_algorithm_audit_fix.py" \
    --reference "$REF" --runtime-wrapper "$NODE_WRAPPER" \
    --output "$OUT" --mode trust_b --seed 42 --updates 2
)
echo "--- het Block H2, shell con song ---"
```

Block H3 — chạy trên node embed-8b-training-v2-0-1 — launch canary 2 updates đã xem kế hoạch

```bash
(
  SRC='/duong/dan/source-da-giai-nen'
  REF='/duong/dan/campaign-cu/launch-config.json'
  NODE_WRAPPER='/duong/dan/wrapper-runtime-da-qualify-cua-node-v2.sh'
  OUT='/duong/dan/run-moi/trustb-audit-s42-canary2'
  python3 "$SRC/experiments/runai/launch_algorithm_audit_fix.py" \
    --reference "$REF" --runtime-wrapper "$NODE_WRAPPER" \
    --output "$OUT" --mode trust_b --seed 42 --updates 2 --execute
)
echo "--- het Block H3, shell con song ---"
```

Launcher chỉ dùng GPU 3, kiểm UUID/process/VRAM trước launch, không kill, không
đổi card. Nó tạo receipt/log cạnh output: `<OUT>.audit-launch.json` và
`<OUT>.audit-launch.log`. Child exit code được giữ trong receipt. TensorBoard bật,
eval in-process tắt. Không tạo hoặc thay queue eval.

Sau canary, cần `optimizer_updates=1` cho mỗi window, loss/grad hữu hạn,
`mp_opd_trust_full_batch=1`, `calibration_response_count=64` nếu cả 64 response
atomize hợp lệ, `calibration_microbatch_count=16`, fallback=0. Một window có
response không hợp lệ thì response_count thấp hơn; đọc invalid counters.
Với `grass_chunk`, `mp_opd_grass_chunk_source=1` chứng minh native source;
với `align`, kiểm `mp_opd_align_chunk_source=1`.

Để chạy đủ campaign, dùng output khác, `--updates 312`; lựa chọn mode là
`grass`, `grass_chunk`, `align`, `trust_r`, `trust_b`. Giữ seed so sánh và mọi
asset đồng nhất. Không dùng `MP_RESUME=1` hoặc checkpoint trajectory cũ để đại diện
cho thuật toán đã sửa. Launcher từ chối output tồn tại và kiểm SHA source.

## Giới hạn cần đọc đúng

Smoke/test là xác nhận implementation và runtime. Chưa có kết quả quality,
chưa tái lập production Qwen-7B/Gemma-2B cùng SGLang, FSDP2/optimizer và sequence
4096 trên node v2-0-1. Head proxy vẫn là LM-head geometry, không phải gradient
toàn Transformer. GRASS/Chunk/ALIGN/TRUST-R còn policy Atomic fallback có counter;
run có fallback dương cần công bố deviation khi phân tích scientific results.
TRUST-B full-batch mới từ chối calibration lỗi thay vì đổi sang Atomic.
