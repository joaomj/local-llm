#!/usr/bin/env bash
set -euo pipefail

MODEL_CACHE_DIR="${HOME}/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it-qat-GGUF"
MODEL_GLOB="${MODEL_CACHE_DIR}/snapshots"/*/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf

shopt -s nullglob
matches=(${MODEL_GLOB})
shopt -u nullglob

if (( ${#matches[@]} == 0 )); then
  printf 'Missing local GGUF model file. Download unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL before starting the server.\n' >&2
  exit 1
fi

MODEL_PATH="${matches[$((${#matches[@]} - 1))]}"

exec llama-server \
  -m "${MODEL_PATH}" \
  --alias gemma-qat \
  --no-mmproj \
  --reasoning off \
  --temp 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --ctx-size 32768 \
  --port 8080 \
  --tools all \
  --parallel 1
