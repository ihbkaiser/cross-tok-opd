#!/usr/bin/env bash
set -euo pipefail

required_mounts=("${MODEL_PATH:-/models}" "${DATA_PATH:-/data}" "${OUTPUT_PATH:-/outputs}")
for path in "${required_mounts[@]}"; do
  mkdir -p "$path"
done

if [[ "${SIMCT_REQUIRE_EXTERNAL_MODELS:-0}" == "1" ]]; then
  if [[ ! -d "${MODEL_PATH:-/models}" ]] || [[ -z "$(find "${MODEL_PATH:-/models}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "SIMCT_IMAGE_ERROR=MODEL_PATH is empty; mount the approved internal model store" >&2
    exit 64
  fi
fi

exec "$@"
