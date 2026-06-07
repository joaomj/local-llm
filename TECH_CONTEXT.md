# Technical Context - Local LLM on Mac mini M4 16GB

This file records the current technical state for the local llama.cpp setup.

Last updated: 2026-06-07.

---

## Goal

Run a local model for chat and lightweight classification through llama.cpp and opencode.

Non-goal: replace the primary coding model.

---

## Hardware

| Component | Specification |
|---|---|
| Machine | Mac mini M4 |
| RAM | 16 GB unified memory |
| GPU | Apple M4 10-core GPU |
| Metal-visible memory | About 12,124 MiB observed |
| Memory bandwidth | About 120 GB/s class |

Practical constraint: keep model weights and KV cache comfortably below Metal-visible memory to avoid OOM and swap pressure.

---

## Current Decision

Use Gemma 4 E2B QAT Q4 through llama.cpp:

```text
unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL
```

Rationale:

- Fastest working model measured: 52.7943 generation tok/s.
- Prompt processing measured at 148.5310 tok/s.
- Fits easily on the 16 GB Mac mini.
- Good match for local chat and classification.

Fallback if quality is insufficient:

```text
unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL
```

E4B Q4 measured 32.3460 generation tok/s and 92.4689 prompt tok/s.

---

## Best Server Command

The canonical command is in `run_model.sh`:

```bash
./run_model.sh
```

Expanded command shape:

```bash
llama-server \
  -m ~/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it-qat-GGUF/snapshots/<snapshot>/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf \
  --alias gemma-qat \
  --no-mmproj \
  --reasoning off \
  --temp 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --ctx-size 131072 \
  --port 8080 \
  --tools all \
  --parallel 1
```

Important details:

- `--alias gemma-qat` makes the raw API model name stable.
- `--parallel 1` matches single-user local usage and improved 12B performance during testing.
- `--no-mmproj` keeps the server text-only.
- `--port 8080` matches the opencode provider config.

---

## OpenCode Integration

Global opencode config includes this provider:

```text
llama.cpp/gemma-qat
```

Provider shape follows the official llama.cpp provider docs:

```json
{
  "provider": {
    "llama.cpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama-server (local)",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1"
      },
      "models": {
        "gemma-qat": {
          "name": "Gemma 4 E2B QAT Q4 (local)",
          "limit": {
            "context": 32768,
            "input": 32768,
            "output": 8192
          },
          "temperature": true,
          "tool_call": true
        }
      }
    }
  }
}
```

The provider does not start `llama-server`. Start `./run_model.sh` first, then restart opencode.

---

## Benchmark Summary

| Variant | Model | Gen tok/s | Prompt tok/s | Result |
|---|---|---:|---:|---|
| `e2b-q4-32k` | E2B Q4 | 52.7943 | 148.5310 | Best speed |
| `e4b-q4-32k` | E4B Q4 | 32.3460 | 92.4689 | Quality/speed fallback |
| `12b-q4-32k` | 12B Q4 | 13.5111 | 47.3704 | Quality baseline, not needed for current use |
| `e2b-q2-32k` | E2B Q2 | - | - | Failed on Metal |
| `e4b-q2-32k` | E4B Q2 | - | - | Failed on Metal |

Full cumulative data is in `benchmarks/benchmark_results.md`.

---

## Failed Or Rejected Paths

### Q2 Gemma QAT GGUF

E2B/E4B `UD-Q2_K_XL` failed during llama.cpp Metal warmup:

```text
metal_unsupported_quant_type_35
ggml-metal-device.cpp:901: not implemented
Asserting on type 35
```

Conclusion: do not use Q2 on this current Apple Metal build.

### Gemma 4 26B A4B MoE

Not tested further. The Q4 GGUF is about 14.2 GB, which leaves too little headroom on a 16 GB Mac mini once KV cache, Metal buffers, and macOS memory are included.

### 12B QAT Q4

Works, but it is much slower than E2B/E4B for the current chat/classifier use case.

### MLX paths

Previously evaluated and rejected for this machine because the Gemma 4 unified/multimodal path consumed more memory and did not outperform llama.cpp text-only GGUF.

### Qwen MoE experiments

Previously explored, but not part of the current setup. Context and memory headroom were not attractive for this machine.

---

## Cache State

Only the active E2B Q4 cache is retained locally:

```text
~/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it-qat-GGUF
```

Removed cache artifacts:

- Gemma 4 12B QAT GGUF
- Gemma 4 E4B QAT GGUF
- E2B Q2 blob
- E2B mmproj blob
- Old MLX/Qwen artifacts from earlier exploration

---

## Benchmark Script

Run:

```bash
uv run python scripts/benchmark_gemma_qat.py
```

The script:

- Refuses to take over an existing `llama-server` unless `--kill-existing` is passed.
- Checks port availability.
- Starts one variant at a time.
- Writes server logs and response JSON to `logs/llama-bench/<timestamp>/`.
- Appends cumulative results to `benchmarks/benchmark_results.md`.
- Updates `benchmarks/optimization_memory.md`.
- Updates `run_model.sh` only after a meaningful improvement.

---

## File Inventory

| File | Purpose |
|---|---|
| `README.md` | User-facing quick start and current summary |
| `TECH_CONTEXT.md` | Technical state and rationale |
| `OPTIMIZATION_PROTOCOL.md` | Rules for future optimization experiments |
| `benchmarks/benchmark_results.md` | Cumulative benchmark table |
| `benchmarks/optimization_memory.md` | Current best, failures, and next hypothesis |
| `run_model.sh` | Canonical server startup command |
| `scripts/benchmark_gemma_qat.py` | Benchmark automation script |
| `qat-chat.json` | Historical 12B QAT sample |
| `gemma-q2-chat.json` | Historical 12B Q2 sample |

---

## Next Steps

1. Validate E2B Q4 on real chat and classification prompts.
2. If E2B quality is insufficient, restore/download E4B Q4 and compare quality against speed.
3. Keep Q2 and MoE variants out of the default path unless llama.cpp Metal support or hardware changes.
