#!/usr/bin/env bash
set -euo pipefail

llama-server -hf unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL --alias gemma-qat --no-mmproj --reasoning off --temp 0.8 --top-p 0.95 --top-k 64 --ctx-size 32768 --tools all --parallel 1
