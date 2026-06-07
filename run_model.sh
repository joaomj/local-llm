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

LOG_DIR="${PWD}/logs"
LOG_KEEP=10
LOG_FILE="${LOG_DIR}/llama-server-$(date +%Y%m%d-%H%M%S).log"

mkdir -p "${LOG_DIR}"

old_logs=()
while IFS= read -r log_path; do
  old_logs+=("${log_path}")
done < <(ls -t "${LOG_DIR}"/llama-server-*.log 2>/dev/null | tail -n +$((LOG_KEEP + 1)))
if (( ${#old_logs[@]} > 0 )); then
  rm -f "${old_logs[@]}"
fi

printf 'Logging llama-server output to %s\n' "${LOG_FILE}" >&2

llama-server \
  -lv 3 \
  -m "${MODEL_PATH}" \
  --alias gemma-qat \
  --no-mmproj \
  --reasoning off \
  --temp 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --ctx-size 131072 \
  --port 8080 \
  --tools all \
  --parallel 1 \
  2>&1 | tee "${LOG_FILE}"
