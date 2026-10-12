# ALIGN-NEW và TRUST-P — đặc tả thử nghiệm v1

Ngày: 2026-10-12. Mode trong launcher: `align_new`, `trust_p`.
Đây là **hai phương pháp thử nghiệm mới**, không phải thay tên các run lịch sử.
`align`, `trust_r`, `trust_b` vẫn giữ quy tắc cập nhật cũ. Chưa có bằng chứng rằng
hai mode mới phục hồi accuracy hoặc giải quyết nguyên nhân duy nhất của collapse.

## 1. Loss và phạm vi của hình học

Với atom byte-synchronized i, giữ định nghĩa OPD hiện hành:

\[
 \ell_i(\theta)=-\sum_{t\in i}\log p_\theta(y_t\mid y_{<t},x),\qquad
 r_i=\frac{L_{T,i}-L_{S,i}}{n_{S,i}},\qquad
 \mathcal L_{\rm atomic}=\sum_i\operatorname{sg}(r_i)\ell_i.
\]

`sg` là stop-gradient; cả rates, partition và quyết định surgery đều detach.
Atom NLL là **token sum**. Vì vậy hệ số r_i được lặp cho từng token trong atom;
không chia thêm cho n_{S,i}. Trainer chia tổng loss bằng một token normalizer
chung của optimizer batch, sau đó gradient accumulation/data parallel dùng hệ số
chung. Những hệ số dương chung không đổi quyết định direction của hai phương pháp.

Đặt G_i là gradient của ell_i qua **output edge của LM-head**, gồm weight và bias
nếu có. Với logits cuối z_t đã softcap và hidden h_t thực sự đi vào head:

\[
 \Delta_t=(p_t-e_{y_t})\odot j_t,\quad
 j_t=1\ \text{(không softcap)},\quad
 j_t=1-(z_t/c)^2\ \text{(softcap }c\tanh(u_t/c)\text{)},
 \quad G_i=\sum_{t\in i}\Delta_t h_t^\top.
\]

Tích vô hướng tính với full vocabulary, fp32 matmul không TF32/autocast, giảm
norm/dot bằng fp64. Head bias thêm vector sum_t Delta_t. Không dùng top-k.

**Giới hạn:** đây không phải gradient toàn Transformer. Với tied embedding nó chỉ
đo contribution qua output edge, không gồm đường input embedding. SGD trên proxy
không chứng minh Adam/preconditioning hay gradient toàn mô hình giữ cùng dấu.
Hai mode v1 yêu cầu `kd_ratio=1`, `exact_head`, sequence parallel size 1.
Không được đặt `MP_OPD_GRASS_EXACT_FP32=0`.

## 2. ALIGN-NEW: kiểm tra tổng update trước khi nhận surgery local

### Vấn đề cần sửa trong phương pháp

ALIGN cũ thực hiện một sweep PCGrad cyclic, dùng reference gốc trong mỗi chunk.
Phép chiếu từng cặp không bảo đảm mọi conflict đều mất sau sweep. Quan trọng hơn,
Gram block-diagonal chỉ mô tả các chunk tách rời, không mô tả tổng gradient thực.
Cosine đo bằng Gram đó có thể dương dù tổng gradient quay ngược Atomic.

Phản ví dụ PSD, mode `run`, length 2:

\[
 G_1=(1,0),\quad G_2=(-1,1),\quad G_3=(-0.4,-1),\quad r=(1,1,1).
\]

Chunk [1,2] và [3]. ALIGN cũ cho e=(2,1.5,1). Metric block cosine = 0.9460503,
nhưng g_0=sum_i r_i G_i=(-0.4,0), g_1=sum_i e_i G_i=(0.1,0.5):
dot(g_0,g_1)=-0.04 và cosine thực=-0.1961161. Một bước -eta*g_1 tăng surrogate
Atomic đã đóng băng rates ở bậc nhất: +0.04*eta. Đây là phản ví dụ toán học và
regression test, không phải chứng minh nó đã xảy ra ở checkpoint lịch sử.

### Quy tắc mới

Giữ nguyên partition và local cyclic PCGrad để tạo **ứng viên** e. Sau đó tính
g_0 và g_1 đầy đủ trên cả response, **bao gồm mọi cross-chunk term**. Chọn hệ số
trộn lớn nhất beta trong [0,1] sao cho vẫn giữ ít nhất rho của Atomic descent:

\[
 q=r+\beta(e-r),\quad g(q)=(1-\beta)g_0+\beta g_1,\qquad
 \langle g_0,g(q)\rangle\ge\rho\|g_0\|^2.
\]

Với a=||g_0||^2, d=<g_0,g_1>:

\[
 \beta=\begin{cases}
 0,&a\le\epsilon_g,\\
 1,&d\ge\rho a,\\
 \min(1,(1-\rho)a/(a-d)),&d<\rho a.
 \end{cases}
\]

Loss mới là sum_i sg(q_i)*ell_i. Khi beta nằm giữa 0 và 1, code dịch beta vào
miền khả thi một khoảng tương đối 1e-6 để giảm lỗi làm tròn coefficient fp32.
Anchor gần zero thì trả Atomic, log `no_anchor_fraction=1`, không tạo direction
mới chỉ vì gradient tham chiếu triệt tiêu.

Mặc định **rho=0.1** (`mp_opd_align_new_min_descent_fraction`;
`MP_ALIGN_NEW_MIN_DESCENT_FRACTION`). Đây là lựa chọn recipe giữ tối thiểu 10%
descent, **không phải optimum suy từ accuracy**. Đặt 0 chỉ giữ non-ascent; đặt 1
giữ toàn bộ first-order descent của Atomic và có thể bác nhiều surgery. Không có
norm matching hoặc norm cap; norm ratio phải được theo dõi, không bị che bằng
rescale. Ở phản ví dụ trên beta xấp xỉ 0.72 và dot mới xấp xỉ 0.016 > 0.

### Chi phí và telemetry

Hình học local vẫn dùng Gram của từng chunk. Guard global chỉ cần ba scalar
a, b=||g_1||^2, d. Code dựng hai update qua từng vocabulary tile 4096, rồi cộng
norm/dot; không giữ global atom Gram hay hai tensor [vocab,hidden] giữa các tile.
Phải trả thêm chi phí matmul full vocabulary mỗi response. Canary B64/M4 với
generation cap128 đã pass trên B200 (mục 6); chưa cam kết tốc độ hay VRAM cho
response4096 token trên production node.

Metric ứng viên cũ đổi namespace thành `mp_opd_align_new_local_*`. Các metric
global chính: `beta`, `guard_active_fraction`, `no_anchor_fraction`,
`candidate_head_update_cosine`, `head_update_cosine`, `head_update_norm_ratio`,
`retained_descent_fraction` dưới prefix `mp_opd_align_new_`. Chúng được trainer
lấy trung bình **theo response**, không phải cosine của tổng optimizer batch.
Nếu thiếu hoặc hỏng geometry, mode mới fail closed; không âm thầm fallback Atomic.

**Chưa sửa:** sự phụ thuộc vào thứ tự sweep, conflict còn lại giữa atom, bias của
head proxy và tương tác giữa các response. Guard bảo vệ tổng của **một response**;
cộng nhiều response không tạo bảo đảm tương tự trên tổng batch.

## 3. TRUST-P: Projected Mismatch Trust

### Motivation và khác biệt với TRUST-R/B

Giữ tập S gồm atom `one_to_one`, M gồm atom `multi_token`. Tokenizer correspondence
không chứng minh supervision S đúng về ngữ nghĩa; S là reference theo giả thuyết,
không phải nhãn ground truth.

TRUST cũ dùng lambda=max(0,<g_S,g_M>/||g_M||^2), rồi update g_S+lambda*g_M.
Vì scale **toàn** g_M, mismatch vuông góc cũng mất khi dot=0; chỉ một thành phần
chống S nhỏ cũng có thể làm toàn bộ M biến mất. Đây là đặc tính công thức cũ,
không phải lỗi implement lambda.

TRUST-P giữ component vuông góc, chiếu M lên halfspace không chống S:

\[
 v^*=\arg\min_v\frac12\|v-g_M\|^2
 \quad\text{s.t.}\quad\langle g_S,v\rangle\ge0,
\]
\[
 g_M^{\rm safe}=g_M-\frac{\min(0,d)}{s}g_S,\quad
 d=\langle g_S,g_M\rangle,\ s=\|g_S\|^2,\qquad
 g_{\rm P}=a g_S+g_M,\quad a=1-\min(0,d)/s.
\]

Loss khả thi mà không cần sửa gradient thủ công:

\[
 \boxed{\mathcal L_{\rm P}=\operatorname{sg}(a)\mathcal L_S+\mathcal L_M.}
\]

M coefficient giữ 1. Điều đó **không** có nghĩa effective mismatch vector giữ
nguyên: correction đi qua việc tăng coefficient của S. Đó là cùng vector g_P
trong proxy, còn trên toàn Transformer chỉ là một phép điều chỉnh loss.

Khi s>epsilon_g và anchor tồn tại:

\[
 \langle g_S,g_P\rangle=s+\max(d,0)\ge s,\quad
 \|g_M^{\rm safe}\|\le\|g_M\|.
\]

Component vuông góc với S không đổi. Projection không khuếch đại norm mismatch,
nhưng **tổng update có thể lớn hơn TRUST cũ**, vì không xóa component vuông góc.
Không giới hạn a tùy ý: clip a có thể làm mất halfspace guarantee. Code làm tròn
a về phía halfspace khả thi trong fp32; a không biểu diễn được thì dừng với lỗi.
Nonexpansiveness trên là định lý cho coefficient số thực: làm tròn fp32 có thể
thêm component parallel nhỏ, cỡ machine epsilon nhân norm S. Khi M rất nhỏ, tỷ
lệ norm tương đối của M có thể lớn dù sai số tuyệt đối nhỏ. Telemetry tính với
**a đã làm tròn**, không báo norm của projection lý tưởng thay cho update thực.
Không có lambda floor, warmup hoặc lịch temporal trong bản v1.

Ví dụ S=(1,0):

| M | TRUST cũ | TRUST-P |
|---|---|---|
| (0,1) | (1,0) | (1,1) |
| (-0.1,1) | (1,0) | (1,1) |
| (-2,0) | (1,0) | (1,0) |
| (1,10) | (1.009901,0.099010) | (2,10) |

Đây là projection halfspace quen thuộc, có quan hệ với gradient surgery; tên
TRUST-P là tên thực nghiệm trong repo, **không phải tuyên bố novelty**.

### Calibration và guards

Mode `trust_p` chỉ có **full optimizer batch**: batch64/micro4 là 16 microbatches.
Lượt no-grad cộng head update S/M, gồm cả cross-response và cross-microbatch term;
finalize mới chọn một a chung. Lượt backward replay RNG để dropout giống lượt
calibration; parameters giữ nguyên giữa hai lượt. Không thay đổi dependency.
Storage giống TRUST-B: hai [vocab,hidden] tensor fp32, bias nếu có; guard allocation
ở nửa VRAM còn trống. Vì vậy không thể hứa không OOM trên production node.

Nếu không có S hoặc s<=1e-12, dùng a=1, tức **Atomic**, và log `no_anchor=1`.
Ở nhánh này không tuyên bố bảo vệ S. Nếu không có M, loss là strict Atomic.
Nếu không có valid atom nào, update zero với coverage count zero. V1 yêu cầu một
loss normalizer chung, dương và hữu hạn giữa các microbatch/rank, đúng contract
trainer hiện tại; cấu hình vi phạm thì dừng. `microbatch` scope bị từ chối.
Không giữ calibration a qua optimizer updates hay lưu nó vào resume checkpoint.

Namespace riêng `mp_opd_trust_p_*`: `strict_scale`, `mismatch_scale`,
`strict_mismatch_dot`, `strict_mismatch_cosine`, `projected`, `no_anchor`,
`mismatch_retained_norm_ratio`, `strict_descent_ratio`, `update_norm_ratio_post_pre`,
coverage counts và `full_batch`. Undefined ratio log zero kèm `update_ratio_defined`;
không đọc zero đó thành suppression. Metric global này được lặp cho mọi microbatch
trong cùng update, nên không bị pha với response lambda của các mode cũ.

## 4. Chạy, resume và thiết kế đánh giá

Sau khi lấy source mới, dùng runtime B200 hiện có. Không cần cài Transformers hay
tải lại runtime để dùng hai mode. **Tạo output directory mới và bắt đầu từ SFT**;
không resume trajectory cũ sang mode mới. Pipeline resume contract ghim mode,
rho, source, config, input hashes và dependency versions; algorithm state của
mode mới cũng kiểm tra method/version khi restore.

Launcher GPU được gọi theo cấu trúc:

```text
bash experiments/runai/run_single_gpu.sh <GPU đã được giao> align_new 2
bash experiments/runai/run_single_gpu.sh <GPU đã được giao> trust_p 2
```

Hai dòng trên là cú pháp tham khảo, chưa được chạy trên node. Asset paths phải lấy
từ launch-config/campaign hiện có bằng `MP_STUDENT_PATH`, `MP_TEACHER_PATH`,
`MP_DATASET_PATH`; run dài dùng limit 312. `align_new` mặc định source `xtoken`,
baseline `run` length2 dùng `MP_GRASS_CHUNK_SOURCE=run`,
`MP_GRASS_CHUNK_RUN_LENGTH=2`. `trust_p` yêu cầu `MP_TRUST_SCOPE=batch`.
Không chạy đồng thời hai job trên cùng GPU. Xác minh GPU ownership trước launch.

Thí nghiệm nên tách ba câu hỏi:

1. **Implementation:** autograd head oracle, phản ví dụ cross-chunk, full-batch
   B64/M4 replay và backward/update oracle. CPU tests không chứng minh B200 capacity.
2. **Proxy fidelity:** lấy cùng frozen prefixes, đo cosine head và full-model
   gradient trước/sau. So sánh phần input của tied embedding và bước Adam thực.
3. **Efficacy:** đối chứng `align`/Atomic với `align_new`; TRUST-R/B/Atomic với
   `trust_p`, cùng SFT, data, seed, sampling, optimizer, checkpoint/scorer contract.
   Chấm code/math và repetition/termination tại step40/80/120/160/200/240/312.

Đối với ALIGN-NEW, ablate rho=0/0.1/1 và giữ source/chunk length cố định để tránh
trộn hiệu ứng partition với guard. Với TRUST-P, kiểm tra chất lượng supervision
S/M và mức orthogonal mismatch thực sự được giữ. Giữ raw responses, launch-config,
train/TB, weights/tokenizer/config và SHA256. Không lấy finite loss, norm hữu hạn,
hoặc canary pass làm bằng chứng accuracy phục hồi.

## 5. Kiểm chứng implementation — 2026-10-12

Source có code đã kiểm tra: `1cbd804d384052ca462f2b7fee476b04c8f96b6e`.
Lượt B200 cuối: **263 passed, 0 failed, 0 skipped**, pytest 19.96 giây,
[Modal app](https://modal.com/apps/lhtu05/main/ap-tnISe96UWB3fpRysvjNFKd).
Đây là bộ kiểm tra audit được chọn, không phải toàn bộ tests của repo.
Raw log và receipt SHA256 nằm tại
`remote_artifacts/align_new_trust_p_20261012/` (persistent, không commit).

Runtime dùng lại nguyên image digest
`docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f`;
Python 3.12.12, Torch 2.11.0+cu130, NVIDIA B200. Không đổi Transformers hoặc
đóng gói lại payload runtime. Lượt đầu có 261 pass, hai fixture preflight thiếu
`CUDA_VISIBLE_DEVICES`; chỉ sửa fixture rồi chạy lại, giữ nguyên guard launcher.

- Geometry head đối chiếu independent autograd; có phản ví dụ cross-chunk,
  projection conflict, zero anchor, rounding, resume và invalid-input contracts.
- Actor B64/M4 được đối chiếu gradient/update oracle; replay dropout hai lượt
  được kiểm tra. Trên B200, tiny Gemma2 thật dùng BF16, softcap 30 và một optimizer
  update cho cả TRUST-B control và TRUST-P: max gradient/parameter error đều 0.
  Fixture ngẫu nhiên TRUST-P có `strict_scale=1`; case opposing được kiểm tra riêng,
  không diễn giải fixture này thành kiểm chứng projection đang kích hoạt trên Gemma2-2B.
- ALIGN-NEW tiled geometry tại vocab 256000, hidden width 2304, 32 token tổng hợp
  dùng thêm 0.3489723205566406 GiB. Đây chỉ là probe head ở kích thước production,
  không phải full-model train batch64/context4096.
- Bộ CPU offline liên quan có 243 pass, 3 skip; Transformers/PEFT thiếu ở interpreter
  CPU local nên chưa collect được toàn bộ repo. B200 audit trên runtime hiện có
  không có test skip.

Chưa chạy train dài Gemma2-2B, chưa đo accuracy, chưa chứng minh head guarantee
chuyển sang full Transformer/Adam, và chưa chứng minh memory capacity tại context4096.
Hai app kiểm tra đã dừng, không còn task. Billing refresh sau test ghi
0.04631946 USD (lượt đầu) + 0.04273889 USD (lượt cuối) = **0.08905835 USD**;
đây là usage do Modal CLI báo tại thời điểm refresh, không phải hóa đơn cuối kỳ.

## 6. Canary production launcher — 2026-10-12

Source đã chạy: `b71017624544c578d111d29f81d777ffb7e09d02`. Dùng **shell launcher
thật**, Ray/FSDP/SGLang, weights SFT Gemma2-2B và teacher Qwen2.5-7B-Instruct hoặc
Phi-4-mini-instruct đã xác minh hash. Không đổi image digest hay dependency.
Mỗi run batch64/micro4, eager, scheduler horizon312, context4096, **generation
cap128**, hai optimizer updates. Cả bốn run exit0 và run-summary completed.

| Cặp / mode | Chunk source | p99 diagnostic update 2 | Trainer peak allocated tối đa (GiB) |
|---|---|---:|---:|
| Qwen / align_new | xtoken | 0.1068090945 | 44.92377090 |
| Qwen / trust_p | không áp dụng | 0.1138979048 | 44.91791153 |
| Phi / align_new | xtoken | 0.1149547026 | 45.05218887 |
| Phi / trust_p | không áp dụng | 0.1214310303 | 45.04665756 |

Số lấy từ TensorBoard scalar fp32. p99 là metric trainer đã aggregate, không phải
quantile toàn bộ token gộp lại; peak là telemetry trainer, không phải capacity
cho response4096. Peak reserved lớn nhất 55.482421875 GiB. Cả 8 updates có parity
failure=0, numeric fallback=0, invalid sample=0, fail-closed sample ratio=0,
64 trajectory samples và 4 valid samples/microbatch. Không có OOM trong các run.

TRUST-P ghi nhận full_batch=1, exact_geometry=1, calibration_response_count=64,
calibration_microbatch_count=16, mismatch_scale=1. **Nhánh projection đã chạy với
weights thật** ở Phi update2: cosine S–M=-0.1209474653, dot=-1515573.75,
||S||²=47471360, strict_scale=1.0319261551, strict_descent_ratio=1.0000001192,
mismatch_retained_norm_ratio=0.9926589131. Các scalar phù hợp công thức trong sai
số fp32. Qwen có dot dương ở cả hai updates nên strict_scale=1, projected=0.

ALIGN-NEW native có beta=1 và guard_active_fraction=0 ở cả hai cặp: canary này
**không tạo case kích hoạt guard mới**. Qwen có chunk nhiều atom khoảng 1.164–1.482%;
Phi chỉ khoảng 0.035–0.057%, với credit_relative_l2_change=0 ở cả hai updates Phi.
Đây là biểu hiện phương pháp native gần Atomic trên fixture này, không phải bằng
chứng guard cải thiện accuracy. Guard conflict đã được kiểm tra bởi phản ví dụ và
independent autograd ở mục 5. Đối chứng Qwen ALIGN-NEW/run2 riêng cũng hoàn thành
hai updates, exit0; không thay native partition bằng run2 âm thầm.

Raw receipts, train logs, launch-config, summaries, TensorBoard, hash manifest và
metric audit persistent tại `remote_artifacts/guarded_production_20261012/`.
Canary dùng 128 prompt rows lặp từ 16 diagnostic cases, **không phải dataset
training gốc**; target updates của fixture là4 do hai epochs, trong khi LR horizon
vẫn312. Chưa chạy dài, chưa chấm accuracy, chưa chứng minh capacity tại trần4096
hoặc guarantee trên full Transformer/Adam. Agent chạy node dùng
[bản bàn giao](ALIGN_NEW_TRUST_P_NODE.md) và release receipt cuối.

Lượt này bổ sung generation cap chỉ dành cho diagnostics và guard nhận diện source
trước khi trả phí. Một invocation CPU ghi nhầm SHA bị chặn; khác biệt CRLF của Git
Windows/WSL được sửa bằng EOL policy **theo từng lệnh**, không sửa Git config toàn
cục. Regression CPU cho launcher/source: 8 pass, 2 skip (thiếu Transformers local);
prepare trên runtime Modal có 19 pass. Không tìm thấy lỗi implementation mới trong
hai thuật toán qua các canary trên. Finite loss hoặc canary pass không chứng minh
nguyên nhân collapse lịch sử đã được giải quyết.
