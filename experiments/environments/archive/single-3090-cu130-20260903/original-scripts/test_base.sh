#!/usr/bin/env bash
set -euo pipefail
cd /workspace/xtoken-native/NeMo-RL
test "$(git rev-parse HEAD)" = 13a10647ebbf0f940d2b06ea41800b3f2fb46099
uv run --no-sync python -m pytest -q -o addopts= --timeout=90 \
  tests/unit/algorithms/x_token tests/unit/tools/x_token \
  --junitxml=/workspace/xtoken-native/artifacts/upstream-xtoken-unit.xml
uv run --no-sync python -m pytest -q -o addopts= --timeout=90 \
  tests/unit/algorithms/test_loss_functions.py -k cross_tokenizer \
  --junitxml=/workspace/xtoken-native/artifacts/upstream-xtoken-loss.xml
