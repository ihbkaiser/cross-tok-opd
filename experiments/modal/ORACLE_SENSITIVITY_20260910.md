# Oracle numerical sensitivity canary, 2026-09-10

Completed remote execution on NVIDIA A10G, Modal profile phamvanvuhoan.
A100-80GB was rejected before GPU startup because a payment method is required.
One available A10 was used within the authorized $5 / 30-minute GPU envelope.
App: https://modal.com/apps/phamvanvuhoan/main/ap-WuCQV7CJwbmWsEfpkt3w1S
App status verified stopped, zero tasks, exit 0. App lifetime about 45 seconds;
worker model load + ten probes 15.58 seconds. Billing report has no row yet;
reported actual charge is unavailable, not zero. No company GPU or W&B touched.

Source at execution: e0e2962391ebf3aa16e8b13e1f995243d55ddc23.
Reused pinned runtime digest and existing public Gemma cache. No model download,
local dependency installation or runtime rebuild. Torch 2.11.0+cu130, CUDA 13.
Model: google/gemma-2-2b-it, revision 299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8.
This is not the company SFT checkpoint.

## Result

Frozen Gemma with a zero-initialized rank-4 adapter on layer25 q_proj, identical
to the diagnostic adapter arithmetic. One synthetic sentence reference, no
benchmark test data. Controls: skip and meta_sft, LR .001/.01/.1/1, BF16/FP32.
Loss is evaluated on the same sentence used for its gradient, deliberately a
numerical sensitivity check, not generalization or oracle selection evidence.

At LR .01:

| Precision | Gradient norm | Parameter delta norm | Max logit delta | NLL decrease | Predicted decrease |
|---|---:|---:|---:|---:|---:|
| BF16 | .03665 | .0003665 | 0 | 0 | .00001343 |
| FP32 | .03617 | .0003617 | .0004065 | .00001287 | .00001308 |

Skip has exactly zero logit/NLL change in both precisions. BF16 LR1 produces
nonzero logit changes and NLL decrease .001039, demonstrating the virtual
parameter replacement is connected. FP32 LR1 decrease .001308 agrees closely
with predicted .001308. Peak allocated memory about 4.94 GiB BF16 and 9.94 GiB
FP32 on these very short inputs; this is not a memory estimate for long-context
full MP-OPD diagnostics or teacher loading.

This reproduces a low-precision numerical dead zone consistent with the zero
gains seen in the company canary. It does not isolate the precise rounding
operation or prove the company SFT case has the same cause. Comparing full
FP32 and BF16 changes forward and gradient numerics. No inference about the
quality of MP-OPD, teacher credit, atomic/fixed training, or evaluation scores.

Next: qualify sensitivity on the actual SFT checkpoint, then compare all
oracle controls with meaningful virtual updates and independent select/eval
references. Do not increase training LR based on this diagnostic LR sweep.
Do not use the two all-zero company groups as a scientific oracle gate.

## Artifacts and reproduction

Raw result and full logs are retained in remote_artifacts/modal-oracle-sensitivity-20260910.
Remote output volume: mp-opd-gemma-parity-results-v1,
path oracle-sensitivity-a10-20260910-r1/result.json.
The launcher rejects existing result paths, has no automatic retries, caps the
GPU function at 1800 seconds and its child at 1650 seconds. Re-running requires
a new authorized run ID; completed results must not be uploaded as new runs.

Result SHA256: b84849b5c4339fd853f833f2c59e0b54c8ccc8aabae03f49a01de6135261f641
