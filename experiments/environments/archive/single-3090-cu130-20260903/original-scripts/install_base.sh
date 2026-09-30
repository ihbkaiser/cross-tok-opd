#!/usr/bin/env bash
set -euo pipefail
cd /workspace/xtoken-native/NeMo-RL
test "$(git rev-parse HEAD)" = 13a10647ebbf0f940d2b06ea41800b3f2fb46099
uv sync --locked --no-default-groups --group build --group test --link-mode hardlink
uv run --no-sync python -c 'import sys, torch; print(sys.version); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name()); assert torch.cuda.is_bf16_supported()'
uv pip check
