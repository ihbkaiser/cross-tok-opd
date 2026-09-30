# Native uv / single RTX 3090 X-Token setup

This is an isolated environment and diagnostic smoke-test harness, not a paper
reproduction, not an OPD experiment, and not the retired Tunix branch.

## Locations and provenance

- Local evidence: `D:\dev\codex\research_vdt\xtoken-native-3090`.
- Remote task root: `/workspace/xtoken-native`.
- Upstream checkout: `/workspace/xtoken-native/NeMo-RL`.
- Source: https://github.com/NVIDIA-NeMo/RL
- Pinned commit: `13a10647ebbf0f940d2b06ea41800b3f2fb46099`.
- Python, PyTorch and other dependencies come from that checkout's `uv.lock`.
- The default `/venv/main`, NVIDIA driver, existing services, Kaggle and Modal
  deployments are outside this task and are not changed.

The current Vast instance has **no persistent workspace volume**. Stop/start
preserves its filesystem, but destroy/recycle does not. Preserve code and logs
locally before destroying it. Nothing here authorizes another rental.

## Environment use

Run on the remote host:

```bash
cd /workspace/xtoken-native/NeMo-RL
export UV_NO_CACHE=false
export UV_CACHE_DIR=/workspace/xtoken-native/uv-cache
export UV_LINK_MODE=hardlink
export WANDB_MODE=disabled
uv run --no-sync python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name())'
```

`uv` uses an environment under the task cache, with `.venv` pointing to it.
Keep `UV_CACHE_DIR` consistent. The base install includes the build and test
groups, **not** the optional Automodel, Megatron, vLLM or SGLang workers.
The X-Token DTensor-v2 trainer requires the Automodel worker environment; base
unit/loss tests alone cannot certify that end-to-end backend.

## Scripts and test scope

- `scripts/install_base.sh`: locked base/build/test install and CUDA import probe.
- `scripts/run_logged.py`: one execution, bounded timeout, log and GPU telemetry;
  never retries a command. Uses unique timestamped evidence directories.
- `scripts/test_base.sh`: original upstream X-Token CPU unit tests and selected
  cross-tokenizer loss tests. Multi-GPU tests are expected to skip on one GPU.
- `scripts/gpu_smoke.py`: CUDA BF16 SDPA plus optimizer updates; FP32 sparse
  forward/backward against a dense reference under BF16 autocast; original
  X-Token P-KL loss with synthetic large-vocabulary tensors and optimizer steps.
- `scripts/capture_environment.py`: selected non-secret package/hardware/source
  metadata, never an environment-variable or credential dump.

The synthetic P-KL test optimizes logits, **not a language model**. It uses an
artificial projection and alignment, no pretrained checkpoints, no tokenizer
download, no real corpus, no Ray CUDA IPC, no distributed TP/CP and no held-out
quality evaluation. Its timings and VRAM must not be presented as LM training
throughput or as projected two-GPU performance.

Do not rerun existing tests blindly: inspect `REPORT.md` and existing artifacts
first. The named summary/XML paths represent this setup attempt. For a new
experiment, choose a new output root rather than overwriting evidence.
