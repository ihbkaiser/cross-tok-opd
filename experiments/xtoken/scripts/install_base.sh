#!/usr/bin/env bash
set -euo pipefail
: "${XTOKEN_REPO:?use xtoken.py}"
: "${XTOKEN_UV_BIN:?use xtoken.py}"
cd "$XTOKEN_REPO"
"$XTOKEN_UV_BIN" sync --locked --no-default-groups --group build --group test --link-mode hardlink
"$XTOKEN_UV_BIN" run --no-sync python -c 'import sys, torch; print(sys.version); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name()); assert torch.cuda.is_bf16_supported()'
# Preserve exit 1 for the known upstream override conflicts; do not silently relock.
"$XTOKEN_UV_BIN" pip check
