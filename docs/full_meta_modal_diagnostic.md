# Full-meta FSDP2 diagnostic, 2026-09-14

Modal app `ap-60bn4aIm8chLLBxpTCfiIr` reproduced disconnected optimizer
parameters on NVIDIA A10, Torch 2.11.0+cu130, CUDA 13.0. B200 launch was
rejected by the profile's payment-method requirement. This is a small linear
model autograd test, not company-model training or throughput qualification.

Using the production `full_meta_step`, plain parameters produced a virtual
gradient norm of 0.649764657. FSDP2 parameters returned all `None` from
`autograd.grad(..., allow_unused=True)` and a reported virtual gradient norm
of zero, while ordinary backward populated gradients in both cases.

The helper rejects an entirely disconnected inner/outer gradient. The new
single-rank forward-parameter bridge captures the actual tensors after FSDP
unsharding and converts gradients back to optimizer DTensor layout. During
meta computation, it retains unsharded storage through higher derivatives,
then restores the original reshard policies and removes its hooks. It uses
Torch 2.11 private policy fields only to restore values without guessing.

Modal app `ap-gLrX51mr75afCiwv4QuS57` passed three-update comparisons on L4:
linear, nested nonlinear layers, BF16, and a tiny random Gemma2 with eager
attention and nonreentrant activation checkpointing. The FP32 cases matched
within 1e-5 relative tolerance; BF16 Gemma used 2 percent relative tolerance
and 1e-7 absolute tolerance. Its plain reference explicitly casts embedding
and norm parameters to match FSDP, since autocast alone does not. Virtual
student rollback is checked exactly each update. This is mechanics evidence,
not full-company-model/B64/M16/B200 qualification or downstream efficacy.

The production full-meta caller uses the bridge. Energy/virtual gradients,
energy deltas and optimizer LR now use scientific notation in text logs.
Exact resume alone cannot establish that the energy gradient is correct.

`queue_split_alternating.py retire-disconnected-meta --case OLD_CASE` cancels
only jobs in that host's verified receipt, and only for source commit
`f5d5114e1d7a3c3a29a1901f167968acb5e8b747`. It preserves outputs and marks
them unsuitable for alternating efficacy or fresh-source resume. Submit a
new case after workers stop; do not modify existing checkpoints or bypass
qualification. Other campaign jobs are untouched.

Reproduce with the existing Modal SDK, using an authorized profile:

```bash
META_DEBUG_GPU=A10G modal run experiments/modal/full_meta_debug_modal.py --run-id unique-diagnostic-id
```

The runner reuses a pinned image, downloads no models, and limits the GPU
function to 180 seconds with no function retries. Results are stored beneath
`remote_artifacts/<run-id>/result.json`. Company queues are not modified.
