# TRUST-R / TRUST-B — báo cáo implementation cho tác giả thuật toán

Ngày: 2026-10-08. Người viết: implementer (Muse Spark).
Repo: `sontungkieu/SimCT`, nhánh `vdt/ops/b200-portable`.
Commit chứa bản mới nhất mô tả ở đây: `a024d3b7` (pair-Gram cho TRUST-B).

Tài liệu này ghi **đúng cái đã code**, phân biệt rạch ròi ba lớp:
**(S)** spec gốc bạn đưa, **(I)** quyết định implementation ngoài spec kèm lý do,
**(B)** bug gặp lúc implement và cách sửa. Số section (§) là section trong spec
bạn gửi (toán calibrate, Gram head, audit tương đương).

---

## 1. TRUST theo spec (S) — nhắc lại để đối chiếu

- Phân vùng: strict = `one_to_one`, mismatch = `multi_token` (boundary types của
  SimCT atomizer, không phải do TRUST tự định nghĩa).
- Hệ số hiệu dụng: `q_i = r_i` (OPD atom loss là token-sum thuần nên outer
  coefficient bằng 1, response weight là plain sum), token reduction `a_it = 1`
  (`current_nll` là token sum trên atom).
- Quy tắc calibration: `λ = max(0, ⟨gS, gM⟩ / ‖gM‖²)`, detached (không cho graph
  đi qua quy tắc).
- Hình học: tái dùng **GRASS exact head Gram** nguyên vẹn — cùng code, cùng số.
- Hai mode, khác nhau **chỉ ở mức aggregate**:
  - **TRUST-R**: một λ mỗi response (calibrate trong `_partition_loss`, vốn đã
    per-sample nên không đụng training loop).
  - **TRUST-B**: một λ mỗi micro-batch (sample loop stash responses, calibrate
    một lần sau loop vì Gram cross-response chỉ thấy được ở tầm batch).

## 2. Kiến trúc code (I)

```
kdflow/algorithms/_mp_opd_trust.py      # toán thuần túy, không biết trainer
  strict_mask()                          # boundary type -> bool, fail-closed
  trust_calibrate()                      # (qs,qm,Hss,Hmm,Hsm) -> TrustUnitResult
  trust_response_loss()                  # L_S + stopgrad(λ) L_M
  trust_response_metrics() / trust_batch_metrics()
  gram_affordable()                      # guard bộ nhớ (mục 5.3)
kdflow/algorithms/mp_opd.py
  _trust_r_loss()                        # R: Gram 1 response + calibrate
  _trust_b_loss()                        # B: stash -> pair-Grams -> ráp blocks
  _trust_pair_gram()                     # Gram exact cho 1 cặp response (mục 5.4)
  training_step()                        # stash B, gọi post-loop, bypass averaging
```

Điểm thiết kế quan trọng nhất: `_mp_opd_trust.py` **không import gì từ trainer,
không đọc config, không giữ state**. Mọi quyết định recipe (mode nào, micro bao
nhiêu) nằm ở `mp_opd.py`. Toán trong trust module là pure function trên tensor
+ Python float — test được trên CPU không cần Ray/GPU, và đó cũng là cách toàn
bộ suite Modal chạy.

`TrustUnitResult` mang: `lam, dot, gs_norm2, gm_norm2, cosine, norm_ratio,
kappa, has_strict, has_mismatch, calibrated, update_cosine, update_norm_ratio,
n_strict, n_mismatch`. Trong đó:

- `cosine = dot / (gs·gm + eps)` — đối xứng S/M, dùng đọc hướng.
- `norm_ratio = gm / gs` (mismatch-to-strict) — độ lớn tương đối của phía cần
  hiệu chỉnh.
- `kappa = λ·gm / gs` — norm của phần mismatch **sau** calibration, theo đơn vị
  strict norm. Đây là số trả lời "calibration để lại bao nhiêu", không phải λ.
- `update_cosine_pre_post`, `update_norm_ratio_post_pre` — so update tiền/hậu
  calibration, triển khai theo Gram scalars (`pre2, post2, inner`), không bao
  giờ dựng vector.
- Điều kiện calibrate: có cả hai phía **và** `gm2 > eps_g (1e-12)`. Hai hằng
  `TRUST_EPS_G = TRUST_EPS_NORM = 1e-12` chỉ giữ mẫu số khỏi 0, không bao giờ
  quyết định có calibrate hay λ bằng bao nhiêu (quyết định đó thuộc về hình học).
- Gram diagonal âm nhỏ được clamp về 0 (roundoff của norm bình phương), **không**
  đụng dot product — dot là thứ quyết định λ nên giữ nguyên kể cả dấu.

Loss cuối: mỗi phía là `(rates.detach() * nll).sum()` — đúng construction của
`soft_partition_loss`. Strict giữ nguyên credits, mismatch scale bởi λ detached.
λ là Python float từ lúc sinh ra nên graph không thể lọt qua.

## 3. Các quyết định ngoài spec (I) — đọc kỹ phần này

### 3.1 Không emit percentile keys (median/p10/…/p90)

Metrics reduce per micro-batch bằng **sum-then-mean**, không biểu diễn được
percentile. Emit mean dưới key `_median` là nói dối về statistic. Means và
fractions trên responses của micro-batch là exact as stated (production dùng
`micro_train_batch_size=4`).

(Lưu ý sửa sai: docstring bản đầu ghi micro=1 theo comment cũ trong code; đã
sửa thành 4 sau khi kiểm `launch-config.json` thực tế.)

### 3.2 `mp_opd_trust_scope` là float code, không phải string

Bảng metric chỉ mang finite tensor: `0.0` = response, `1.0` = batch.

### 3.3 Phía thiếu log 0.0, không bao giờ NaN

Sample/unit thiếu một phía log `0.0` cho mọi quantity chưa định nghĩa (trainer
fail-fast trên metric non-finite, mục 5.1). Diễn giải qua các eligibility
fractions (`has_both`, `zero_strict`, `zero_mismatch`) — đó là chìa khóa đọc,
không phải lambda đơn lẻ.

### 3.4 TRUST đòi `exact_head`, từ chối `diag`

`diag` ablation zero mọi cross-atom term → mọi dot về 0 → mọi λ về 0: mode không
bao giờ act được. Fail-closed thay vì no-op lặng lẽ.

### 3.5 Boundary type thứ ba thì raise

Chỉ `one_to_one`/`multi_token` có phía. Loại thứ ba — nếu file vào một phía —
sẽ đổi calibration lặng lẽ, nên raise thay vì đoán.

### 3.6 Span keys và atom keys trùng nhau là định nghĩa, không phải trùng hợp

Một atom là một supervision span ở đây (TRUST không có grouping cao hơn), nên
`strict_span_count == strict_atom_count` v.v. là hệ quả định nghĩa.

### 3.7 B-metrics bypass averaging của loop

Loop huấn luyện average metrics bằng sum-then-mean trên responses. B-values đã
là batch-level nên phải bypass (flag riêng), ngược lại sẽ average hai lần. R
đi đường metrics thường của loop.

## 4. Guardrails số học (I) — chính sách chung của 4 mode hình học

Áp cho `grass | grass_chunk | align | trust_r` (dispatch) và `trust_b`
(post-loop). Ba tầng, thứ tự ưu tiên cứng:

1. **Inputs non-finite → raise ngay, không fallback.** Không loss hữu hạn nào
   sinh ra được, và để lọt sẽ poison GRASS noise EMA (stateful). Diagnosis đi
   kèm trong message.
2. **Mode failure trên inputs finite → atomic fallback + đếm loud.**
   `NUMERIC_FALLBACK` in ra stdout (mode, sample, reason) và
   `mp_opd_numeric_fallback_fraction` đếm fraction steps/samples fallback.
   Loss và gradients của fallback **đồng nhất** với atomic thuần (đã test:
   loss và grad khớp hand combination).
3. **CUDA OOM → re-raise unconditional.** OOM là resource failure; fallback
   quanh nó chỉ đổi chỗ chết (optimizer step ngay sau cũng thiếu RAM) mà che
   mất bệnh tài nguyên. Không có circuit breaker phát minh thêm — visibility
   duy nhất là fallback fraction.

Riêng biệt nhưng liên quan:

- **Parity mean/p99** (so logprob engine vs recompute) là diagnostic-only
  (outputs không vào loss) → mismatch chỉ warn + cờ
  `trajectory_logprob_parity_failed`, không raise.
- **Metric non-finite** (không phải loss) → drop metric đó + warn tên key.
  **`loss`/`kd_loss` non-finite vẫn fatal** — không tồn tại fallback cho train
  trên NaN.
- Shape/config/provenance/identity guards giữ nguyên fatal: chúng báo bug hoặc
  config sai, không phải episode số học.

## 5. Bộ nhớ Gram: OOM step-171, guard, và pair decomposition (I + B)

Đây là phần tốn nhiều máu nhất, ghi đầy đủ để bạn đánh giá đúng.

### 5.1 Vụ OOM (B, production, trust_b)

Run trust_b sống khỏe 171 steps (λ ≈ 0.03–0.12, cosine ≈ ±0.01, has_both ≈ 1.0,
parity_failed = 0, loss finite, grad_norm bình thường), rồi chết ở step có một
response cực dài (4055 atoms, ~4.3k covered tokens). `atom_head_gram` đòi cấp
11.73 GiB một phát trong khi còn 6.34 GiB trống (process đã ôm ~167 GB; 19 GB
khác nằm fragmented-but-reserved).

Nguyên nhân cấu trúc: Gram **batch-level** to theo **tổng** micro-batch, trong
khi Gram response-level (R) chỉ to theo response dài nhất. Với micro=4, B đắt
hơn R khoảng 4× ở cùng độ dài response. Đây là chi phí của cross-response terms
— đúng thứ làm nên B — không phải bug toán.

### 5.2 `gram_affordable` (I)

Trước mỗi Gram call: ước lượng `n_tokens × vocab × 4 bytes × factor 4.0`
(window fp32 + Jacobian + probabilities + temporaries của vocabulary product)
và đòi dưới **nửa** free bytes hiện tại (`torch.cuda.mem_get_info`). Quá thì
raise `ValueError` → rẽ vào atomic fallback có sẵn (mục 4.2). Trên CPU (tests)
luôn affordable. Hệ số 4.0 và ngưỡng 0.5 là ước lượng kỹ thuật, không phải hằng
số vật lý — xem mục 7.3.

Đã thử `expandable_segments` để chống fragmentation và **revert ngay**: SGLang
`torch_memory_saver` từ chối init dưới nó, giết mọi server lúc `load_model`.
Allocator tweaks không đụng nữa.

### 5.3 Pair decomposition (I, commit `a024d3b7`)

Vì guard bắn ở hầu hết step dài (estimates 19–51 GiB/step trong regime
`content_length_mean` 1200–1500), B trên data long-form degrade về atomic đúng
chỗ cần calibration nhất. Fix: `_trust_b_loss` không gọi một Gram cả batch
nữa. Nó gọi `_trust_pair_gram` cho từng cặp response (4 singles + 6 pairs với
micro=4), mỗi call peak bằng **2 response**, rồi ráp đủ 3 blocks `Hss/Hmm/Hsm`
trên trục response-major và gọi **một** `trust_calibrate` duy nhất.

- Toán đồng nhất (cộng đủ mọi cặp; off-diagonal ×2 theo đối xứng; `(a,b)` và
  `(b,a)` của Hsm là hai entries khác nhau, cùng có trong một pair call).
  Không duplicate logic calibrate — `trust_calibrate` vẫn là nơi duy nhất tính
  derived stats.
- Tổng FLOPs ~1.75× bản full-batch (với split đều), peak giảm 2–4×.
- Mỗi pair check affordable riêng; pair quá khổ → cả batch fallback atomic loud
  (giờ là ngoại lệ, không còn là luật).
- Bit-level: thứ tự cộng fp32 khác nhau giữa hai đường → test tương đương dùng
  `abs=1e-3` (chặt hơn mọi claim khoa học dùng số này 1000×).
- Trade-off còn lại xem mục 7.2.

### 5.4 Hai bug wiring bắt bằng launch thật, không bắt bằng unit test (B)

Cả hai đều nằm ở mối nối, test unit gọi loss trực tiếp không thấy:

1. **Hidden slice thiếu flag**: `output_hidden_states` đã bật cho trust nhưng
   nhánh cắt `student_hiddens_flat` vẫn gate mỗi `grass_needs_hidden` → mọi run
   trust nhận `hidden=None`, chết step 0. Fix 1 dòng. Đẻ ra regression test
   `test_trust_needs_flags_are_all_consumed` (assert từng consumption site bằng
   source text).
2. **Tiling gap**: `_trust_b_loss` nối nguyên token rows mỗi sample (kể cả masked
   trailing tokens ngoài atoms) nhưng ranges nối tiếp → gap đúng bằng số trailing
   tokens, Gram fail-closed. Đường single-response không sao vì nó cắt
   `[:covered]`. Fix: mirror `[:covered]`. Đẻ ra test với trailing masked tokens
   (với code cũ nó đỏ).

Bài học quy trình (đã thành test, không thành lời hứa): mọi code đọc output
atomizer phải test với trailing gap + empty side; mọi needs-flag phải có test
tiêu thụ.

## 6. Bằng chứng test

- **Numpy audit** (trước code): κ-bound, λ hand-values, Gram correspondence
  §10.7 — pass trước khi tin implementation.
- **Modal CPU, torch thật**: 70 tests TRUST (calibration/loss/metrics/wiring),
  guardrail suite (parity rewrite, atomic-fallback identity, tiling regression,
  wiring consistency, pair-vs-full equivalence trên stash 3-response), 105+
  tests xanh tổng.
- **Production**: trust_b 171 steps metrics khỏe (trên), fallback prints +
  `mp_opd_numeric_fallback_fraction` hoạt động đúng thiết kế trên step dài.
  Pair decomposition chưa có run production tại thời điểm viết (deploy ở R85).

## 7. Câu hỏi mở cho tác giả (chưa quyết, cần ý bạn)

1. **B-vs-R khi micro=4 có đáng FLOPs không?** B đắt ~1.75× (pair path) với peak
   vẫn gấp đôi R. Nếu milestones cho thấy λ batch ≈ mean λ response trên data
   này, B là mode thừa — giữ R, bỏ B, code nhẹ một nửa.
2. **Atomic-fallback steps có nên loại khỏi milestones không?** Hiện tại
   `fallback_fraction` chỉ ghi nhận; eval vẫn chấm checkpoint như thường (update
   là atomic chuẩn ở step đó). Nếu bạn coi fallback steps là "không phải trust",
   cần quy tắc censor rõ ràng — tôi không tự đặt vì đó là quyết định phương pháp.
3. **Ngưỡng guard (factor 4.0, free 0.5)**: ước lượng tay từ đọc code
   `_window_quantities`. Nếu production cho thấy fallback quá dày hoặc OOM lọt
   (như step-171 lọt qua guard bản đầu), siết/nới bằng số đo, không bằng cảm tính.
4. **Diễn giải κ vs λ**: tôi log cả hai (`lambda` và
   `calibrated_mismatch_norm_ratio` = κ). Nếu chỉ một số được dùng để đọc
   "calibration mạnh hay yếu", nói tôi biết số nào để khỏi log thừa.
5. **Update stats** (`update_cosine_pre_post`, `update_norm_ratio_post_pre`) hiện
   chỉ để đọc. Nếu không ai dùng để quyết định gì sau 1–2 campaign, đề xuất bỏ
   để nhẹ metrics.
6. **Scope code 0.0/1.0**: đủ dùng cho filter TB. Nếu cần thêm scope (ví dụ pair
   decomposition muốn log per-pair diagnostics sau này), mở rộng thành enum sớm
   thay vì nhét thêm float codes.
