# Báo cáo chẩn đoán B200: exact full-meta MP-OPD ở L=4096

**Ngày chốt:** 2026-09-15
**Mục đích:** handoff tự đủ cho mô hình/kỹ sư tiếp theo. Báo cáo phân biệt rõ evidence vận hành, giới hạn của probe và kết luận khoa học chưa thể rút ra.

## 1. Kết luận

Một B200 đơn lẻ không chạy được **exact full hypergradient hiện tại** ở shape mục tiêu nếu giữ nguyên eager path và Adam moments trên GPU: sequence length 4096, inner batch B=64, outer/meta batch M=16, meta microbatch=4. Sau khi thêm cơ chế offload Adam moments theo phase, synthetic full-size harness đã chạy qua hai update liên tiếp ở đúng shape này.

- Eager attention OOM ở inner higher-order VJP cả với microbatch 2 lẫn 1.
- SDPA không có double-backward trong runtime này.
- Flex Attention qua AOTAutograd/torch.compile cũng không hỗ trợ double-backward.
- Reuse VJP storage không tạo giảm peak đo được và đã bị revert.
- Offload `exp_avg`/`exp_avg_sq` sang CPU trong khoảng higher-order VJP đã giảm peak allocated đủ để synthetic harness pass; khả năng này chưa được kiểm tra với teacher/rollout/partition DP thật.

Không gửi lại campaign công ty với cùng exact objective và chỉ thay microbatch/allocator. Cần quyết định phương pháp: giữ exact rồi đổi runtime/tài nguyên; chuyển sang first-order hay implicit estimator có validation riêng; hoặc thay đổi protocol length với phê duyệt khoa học.

## 2. Ranh giới bằng chứng

### Điều đã chạy thật

Các probe chạy B200 thật trên Modal, image được pin:

```text
docker.io/codemaivanngu/simct-b200@sha256:33b2b55874b34447a1395328987b64c63d824a05fa6b737fe5978b22d497b24f
```

Runner tải Gemma2 cache từ Modal Volume `simct-phi-gemma-assets`, dùng BF16/FSDP2 một rank, non-reentrant activation checkpointing, AdamW optimizer state, và gọi production `full_meta_step`. Vì vậy nó kiểm tra trực tiếp lifetime của virtual update và higher-order VJP.

### Điều probe không đại diện

Loss là synthetic token NLL cộng energy differentiable nhỏ. Probe không có teacher thật, rollout/generation, partition DP, data selection, SFT checkpoint công ty hay evaluator downstream.

Do đó:

1. OOM trong probe đủ để chặn exact configuration đó.
2. Pass trong probe chỉ chứng minh mechanics/capacity, không chứng minh campaign pass hay efficacy.
3. Không dùng runtime/throughput của probe làm benchmark campaign.

### Liên hệ log công ty

Hai archive log đã kiểm hash:

| Nguồn | SHA-256 |
|---|---|
| nlp-core-team-0-0 | ae4fe8c9786360b89a8942957498e5ac2f4e34a2b71ec4897648a48a87925254 |
| hieplh8-beyond-leakage-1-0-0 | b197e2356ea27de4d0d8e23e309f36f1407f7e2b16abd3c7f0dac4ad31f3e29d |

Sáu run case `alt-full-stream-20a2eff` đều OOM sau khi source streaming đã chạy ở length 4096. Failure nằm ở `parameter_grad(..., create_graph=True)`; allocated PyTorch xấp xỉ 163--170 GiB, process khoảng 173 GiB trên GPU 178.35 GiB. Reserved-but-free chỉ 1.3--8.8 GiB, nên không có bằng chứng fragmentation là nguyên nhân chính.

## 3. Cấu hình tái lập

Source:

- `experiments/modal/full_meta_capacity.py`
- `experiments/modal/full_meta_capacity_modal.py`
- `kdflow/algorithms/_mp_opd_full_meta.py`

Target: B=64, M=16, L=4096, meta_micro=4, một update. `batch=0` là smoke probe chỉ gồm hai inner microbatches, dùng để kiểm tra backend có hỗ trợ double-backward trước probe B64.

Runner giới hạn một B200, timeout 1500 s, retries=0, max_containers=1. Mỗi micro chạy process riêng để release allocation sau OOM.

## 4. Kết quả Modal

| Run ID | Backend | B / micro / M / L | Status | Peak allocated | Peak reserved | Ý nghĩa |
|---|---|---:|---|---:|---:|---|
| modal-full-meta-b200-realshape-20260915-r1 | eager | 64 / 2 / 16 / 4096 | OOM | 176.239 GiB | 177.313 GiB | Exact path không fit. |
| modal-full-meta-b200-realshape-20260915-r2 | eager + VJP reuse | 64 / 2 / 16 / 4096 | OOM | 176.238 GiB | 177.211 GiB | Không có cải thiện thực tế. |
| modal-full-meta-micro1-4096-20260915-r1 | eager | 64 / 1 / 16 / 4096 | OOM | 166.970 GiB | 176.920 GiB | Chết ở inner second-backward thứ ba. |
| modal-full-meta-sdpa-doubleback-20260915-r1 | SDPA | 4 / 2 / 16 / 4096 | error | 167.118 GiB | 168.385 GiB | Efficient-attention backward không có derivative. |
| modal-full-meta-flex-doubleback-20260915-r2 | Flex | 4 / 2 / 16 / 4096 | error | 120.500 GiB | 131.975 GiB | Đòi tắt donated buffers. |
| modal-full-meta-flex-nodonate-20260915-r1 | Flex, donated buffer off | 4 / 2 / 16 / 4096 | error | 167.102 GiB | 168.318 GiB | AOTAutograd không hỗ trợ double backward. |
| modal-full-meta-offload-20260915-r2 | eager + Adam moment offload | 64 / 1 / 16 / 4096 | **pass** | 147.442 GiB | 175.344 GiB | Exact synthetic full-meta, 1 update. |
| modal-full-meta-offload-steady-20260915-r1 | eager + Adam moment offload | 64 / 1 / 16 / 4096 | **pass** | 147.442 GiB | 176.043 GiB | Exact synthetic full-meta, 2 update liên tiếp. |

Raw result có tại `remote_artifacts/<run-id>/results.json`. Mọi app trong bảng đã terminal; không có job Modal đang chạy lúc chốt.

## 5.1. Patch offload Adam moments và canary B200

Theo hướng dẫn tiếp theo, tôi đã triển khai một patch opt-in ở commit
`e402a1d`:

- thêm config `mp_opd_offload_adam_moments`, mặc định `false`;
- sau khi hoàn tất optimizer VJP và global-clipping VJP, chỉ chuyển
  `exp_avg`/`exp_avg_sq` của student optimizer sang CPU;
- giữ nguyên `step`, parameter groups và mọi state khác;
- restore bằng context `try/finally` trước energy optimizer step/real student
  update; nếu state thiếu hoặc restore lỗi thì fail-closed;
- thêm marker `adam_moments_offloaded` và byte inventory;
- Modal harness nhận flag `--offload-adam-moments`.

Kiểm chứng local: `tests/mp_opd/test_full_meta.py` **12 passed**; factored
hypergradient algebra trên HEAD **36/36 cases pass**. Đây là correctness/CPU
evidence, không thay thế GPU integration.

Canary mới:

| Run | Cấu hình | Kết quả |
|---|---|---|
| `modal-full-meta-offload-20260915-r2` | B200, eager, B64/M16/L4096, micro1, meta4, exact full-meta, offload on | **PASS**, 1 update trong 88.155 s |

Runtime marker ghi `offloaded_bytes=20,914,735,104` (~19.48 GiB). Peak
allocated cuối là **147.442 GiB**, so với **166.970 GiB** của eager micro1
không offload; probe đã đi qua đủ inner second-backward indices 0--63, energy
update và real student update. `CAPACITY_RESULT` có `status=pass`.

Đây là pass của synthetic full-size student/meta capacity harness, chưa phải
pass của teacher thật, rollout, partition DP hay company queue. Raw evidence:
`remote_artifacts/modal-full-meta-offload-20260915-r2/results.json` và
`remote_artifacts/modal-full-meta-offload-20260915-r2.launch.log`.

Probe r1 chỉ fail ở local entrypoint vì thư mục output đã được tạo trước khi
runner tự tạo; không có GPU workload ở r1.

Steady-state follow-up `modal-full-meta-offload-steady-20260915-r1` chạy đủ cả
hai update. Step 1 mất **101.325 s**, step 2 mất **90.806 s**; không có OOM và
đủ inner second-backward indices 0--63 ở cả hai step. Peak allocated vẫn
**147.442 GiB**, peak reserved **176.043 GiB**. Các marker cho thấy
`offloaded_bytes=20,914,735,104` (~19.48 GiB). Metrics synthetic đi từ
`meta_nll=12.957769` sang `12.833384`; đây chỉ là sanity của mechanics/steady
state, không phải efficacy.

Raw evidence: `remote_artifacts/modal-full-meta-offload-steady-20260915-r1/results.json`
và `remote_artifacts/modal-full-meta-offload-steady-20260915-r1.launch.log`.

## 5.2. P0 production wiring audit

Handoff integration được kiểm trên code revision
`787a5896702b80539cb14da2fe2df5526443ca98`; report này được commit riêng
sau khi kiểm tra.

Đường truyền đã kiểm:

```text
campaign.json: offload_adam_moments=true
 -> queue_full_alternating.specs(): MP_OFFLOAD_ADAM_MOMENTS=1
 -> queue_full_alternating.run_command(): MP_OFFLOAD_ADAM_MOMENTS=1
 -> run_single_gpu.py: opts.mp_opd_offload_adam_moments=True
 -> CLI --mp_opd_offload_adam_moments
 -> args.kd.mp_opd_offload_adam_moments
 -> MetaPartitionedOPD.update_energy_full()
 -> full_meta_step(..., offload_adam_moments=True)
```

Launcher fail-closed nếu biến không phải `0/1`, hoặc nếu bật ngoài alternating
soft `mp_opd` production path. Effective config được ghi trong
`launch-config.json` và preflight in marker
`EFFECTIVE_MP_OPD_OFFLOAD_ADAM_MOMENTS=true`.

Kiểm chứng P0:

- `tests/test_runai_contract.py` và `tests/mp_opd/test_full_queue.py` cùng
  `tests/mp_opd/test_full_meta.py`: **21 passed**.
- `py_compile` cho hai launcher và `git diff --check`: **pass**.
- Test propagation xác nhận cả case-level config, job environment và CLI
  manifest; default không có cờ vẫn là `false`.

## 5.3. P1 production integration gate

P1 **chưa được launch** vì thiếu tài sản đúng contract trong Modal. Inventory
read-only của volume được phép `simct-phi-gemma-assets` chỉ có `student/`,
`teacher/`, `prompts.parquet`, `ready.json` và preflight metadata. Các volume
training hiện có chỉ chứa atomic training outputs/checkpoints; không có
`mp_opd_meta_path` hợp lệ và energy checkpoint cho `soft` alternating.

Vì vậy chưa đủ để đi qua teacher + rollout + production atom/credit + energy
network/partition DP + full-meta + weight sync của update thứ hai. Không thay
meta/energy bằng tensor giả, base checkpoint hay checkpoint nội bộ chưa được
phép chuyển. Không có paid integration app nào được tạo; synthetic canary
không bị lặp lại.

Trạng thái gate: `synthetic_pass`; `P0_wiring_pass`;
`integration_pending_blocked_missing_meta_energy_assets`;
`company_not_verified`; `resume_not_qualified_here`.

## 5. Diễn giải kỹ thuật

### Eager path

`full_meta_step` giữ graph inner gradient bằng `create_graph=True` để lấy VJP của virtual AdamW update theo energy parameters. Activation checkpoint giảm một phần forward activation, nhưng không loại bỏ graph và temporary tensor của higher-order backward. Hạ microbatch từ 2 xuống 1 giúp giảm peak allocated khoảng 9.27 GiB nhưng không rút ngắn lifetime tổng thể đủ để fit.

Thử reuse outer-VJP storage ở commit `78e382d` chỉ đổi peak nhỏ hơn 0.001 GiB. Nó bị revert ở `6129c1f`; không phải production fix.

### Backend attention

SDPA có tiềm năng giảm memory của first-order training nhưng runtime trả về:

```text
derivative for aten::_scaled_dot_product_efficient_attention_backward is not implemented
```

Flex Attention ban đầu báo donated buffer không tương thích với `create_graph=True`. Tắt donated buffers dẫn tới blocker cuối cùng:

```text
torch.compile with aot_autograd does not currently support double backward
```

Đây là capability blocker, không phải allocator tuning. Các thay đổi thử Flex đã được revert để runner không quảng cáo option không hợp lệ.

## 6. Git, artifact và kiểm chứng

| Commit | Ý nghĩa | Trạng thái |
|---|---|---|
| 20a2eff | Stream full-meta inner microbatch lên GPU. | Giữ lại. |
| 73f772a | Parameterize eager/SDPA trong capacity harness. | Giữ lại cho diagnosis. |
| 78e382d + 6129c1f | Thử rồi revert reuse VJP storage. | Source đã khôi phục. |
| 001a4c7, 6d03504, ac6e810, a1bf236 | Thử Flex rồi revert toàn bộ. | Source không cho Flex. |
| e402a1d | Thêm opt-in Adam moment offload diagnostic. | Đã test local và chạy trên B200. |
| ce5a7db | Ghi nhận canary offload. | Giữ lại. |
| c411ffa | Ghi nhận steady-state offload probe. | Đã chốt 2 update trên B200. |

Branch: `vdt/ops/b200-portable`. Commit mới local-only, chưa push. `remote_artifacts/` đang untracked và chứa evidence; không stage nó nhầm cùng code.

Đã chạy `py_compile` cho hai Modal runner sau khi revert. Đây là kiểm tra cú pháp, không phải end-to-end campaign test.

## 7. Round evaluation seed 42 đã có

Seed 42 bị vắng trong phần bảng Modal vì bảng đó chỉ nói về capacity của
exact full-meta. Artifact evaluation lịch sử vẫn có đủ round decoding seeds
42/43/44 cho cùng training run và các checkpoint steps 40, 80, 120, 156, 200,
240, 280, 312. Ba seed này là các lần decode/evaluate của **một checkpoint**;
chúng không phải ba independent training seeds.

Ở checkpoint 312, row seed 42 ghi macro **0.313356935995602** (31.3357%):

| Benchmark | seed 42, step 312 |
|---|---:|
| GSM8K | 0.6186504927975739 |
| MATH500 | 0.224 |
| MBPP | 0.334 |
| LiveCodeBench v6 | 0.07677725118483412 |
| Macro | 0.313356935995602 |

Evidence là bảng W&B đã export tại
`remote_artifacts/eval-dispersion-20260910/wandb/run-20260910_180134-mpbackfill-ec7c67c50e92a2e6/files/media/table/eval/dispersion_table_331_839bbdc12866a5107d8f.table.json`.
Đây là evidence evaluation của training path lịch sử; không chứng minh exact
full-meta capacity mới pass và không thay thế training-seed replication.

### Không nhầm với training seed 42 của sơ đồ 4+1 GPU

Sơ đồ 4+1 có hai namespace seed khác nhau:

- `train_seed=42` là queue train trên GPU persistent/owner. Nó chạy tuần tự
  `main42 -> lowLR42 -> every4-42`, rồi mới có thể tạo export/evaluation.
- `eval_seed=42` là một lần decode/scoring của **một checkpoint đã tồn tại**;
  nó nằm cùng `eval_seed=43,44` trong bảng ở trên.

Với case full đã được archive là `alt-full-stream-20a2eff`, sáu full training
runs đều OOM ở `parameter_grad(..., create_graph=True)`. Do đó chain
`train_seed=42` của case này không có training completion/evaluation completion
được xác nhận; scheduler state terminal của gen/score không thay thế
`score.done.json`. Đây là trạng thái cuối cùng có bằng chứng log, không phải
status live của persistent GPU ở thời điểm đọc báo cáo. Để lấy status live,
phải chạy trên owner host `queue_split_alternating.py status --case <CASE>`
với đúng case ID/receipt đang dùng.

## 8. Billing

Billing snapshot 2026-09-15T11:31:06Z (post-P0, no new paid app):

- Tổng tháng: **$22.61386717**
- B200: **$17.98055448**
- Canary offload r2: **$0.22600091**
- Steady offload 2-update: **$0.47653724**

Profile `lhtu05`: workspace budget $30, guard hard limit $28.5, reserve $1. Kỳ billing được gắn `calendar_month_default`; không coi đây là đối soát invoice cuối cùng. Ledger của mỗi app đã được chốt terminal state kèm lý do khi có lỗi.
Sau khi chốt steady probe, guard với estimate 0 trả về **OK**: còn $5.88613283 tới hard limit sau reserve; không có app đang chạy.
Raw billing report: `/home/tung/.codex/state/modal-gpu-ops/raw_billing_reports/lhtu05/2026-09-01_2026-10-01_20260915T113106Z.json`.

## 9. Các nhánh tiếp theo

### A. Giữ exact full hypergradient

Cần runtime/kernel hỗ trợ double backward và đủ memory, hoặc topology phân phối được activation/higher-order graph. Chỉ thêm GPU để shard parameters không đảm bảo activation per-sample giảm; cần B200 canary trên topology chính xác.

### B. First-order hypergradient

Bỏ đạo hàm bậc hai qua virtual update. Có cơ hội fit B200 nhưng là estimator/objective khác. Trước train thật, phải so với exact reference ở tiny model/length: cosine gradient, relative norm/error, loss sau một update và variance theo seed. Không được gọi biến thể này là exact full-meta.

### C. Implicit hoặc truncated estimator

Dùng HVP/linear solve hoặc unroll truncate để giảm graph lifetime. Đây cũng là thay đổi phương pháp; cần contract solver, condition/convergence và validation exact reference. Không giả định HVP tự fit memory.

### D. Đổi length protocol

Hạ length chỉ hợp lệ khi protocol nghiên cứu cho phép. Nó tạo distribution shift, không phải tối ưu kỹ thuật trung tính; phải có control rõ ràng.

## 10. Kế hoạch tối thiểu nên làm

1. Chọn rõ A/B/C/D; tạo config/run ID mới, không ghi đè exact full-meta.
2. Viết regression tiny-model có exact reference cho estimator mới.
3. Chạy B200 capacity canary L=4096 với Gemma/FSDP/optimizer state; pass nghĩa là full virtual update/hypergradient hoàn tất, không chỉ forward hoặc finite loss.
4. Nếu pass, chạy probe nhỏ có teacher, rollout và partition DP; ghi peak memory, source/config SHA, seed, checkpoint lineage.
5. Chỉ sau các gate này mới can thiệp campaign công ty. Không restart/cancel/submit company queue từ báo cáo này.

## 11. Invariant cho mô hình tiếp nhận

- Phân biệt exact full-meta, first-order, implicit và lower-length protocol.
- Completed scheduler step, finite loss hay checkpoint không phải bằng chứng efficacy.
- Static inspection không phải reproduction.
- Không chạy lại B64 eager/micro1 trên company B200: Modal đã đủ evidence OOM cho shape đó.
- Giữ nguyên logs, run IDs, `remote_artifacts/` và thay đổi không liên quan của người dùng.
- Với probe Modal mới: billing guard trước, run ID độc nhất, record terminal state, rồi đọc log/result sau khi app stopped.

### Câu hỏi cần trả lời trước khi code

1. Exact gradient có phải invariant bắt buộc không?
2. Nếu không, first-order hay implicit estimator nào phù hợp claim/paper protocol?
3. Acceptance metric, seed và budget cho estimator mới là gì?
4. Có runtime hoặc allocation multi-GPU nào được phép cho exact path không?
