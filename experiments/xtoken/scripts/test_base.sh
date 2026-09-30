#!/usr/bin/env bash
set -euo pipefail
: "${XTOKEN_REPO:?use xtoken.py}"
: "${XTOKEN_UV_BIN:?use xtoken.py}"
: "${XTOKEN_RUN_DIR:?use xtoken.py}"
cd "$XTOKEN_REPO"
"$XTOKEN_UV_BIN" run --no-sync python -m pytest -q -o addopts= --timeout=90 \
  tests/unit/algorithms/x_token tests/unit/tools/x_token \
  --junitxml="$XTOKEN_RUN_DIR/upstream-xtoken-unit.xml"
"$XTOKEN_UV_BIN" run --no-sync python -m pytest -q -o addopts= --timeout=90 \
  tests/unit/algorithms/test_loss_functions.py -k cross_tokenizer \
  --junitxml="$XTOKEN_RUN_DIR/upstream-xtoken-loss.xml"
