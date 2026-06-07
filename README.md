# Local LLM Setup — Gemma 4 12B QAT on Mac mini M4 16GB

> **Final engine:** llama.cpp with `unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL`
> **Hardware:** Mac mini M4, 16 GB unified memory

---

## Quick Start

```bash
# Install
brew install llama.cpp

# Run server (auto-downloads model on first run)
llama-server \
  -hf unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL \
  --no-mmproj \
  --reasoning off \
  --temp 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --ctx-size 32768 \
  --tools all

# Test
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Say hello"}],"max_tokens":50}'
```

---

## What We Tried

| Model | Engine | Size | Gen Speed | Prompt Speed | Result |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B Q2_K_XL | llama.cpp | 11 GB | — | — | ❌ **OOM** |
| Qwen3.6-35B-A3B IP2_XXS | llama.cpp | 10.8 GB | **27.9 tok/s** | — | ✅ Fast MoE, tight ctx |
| Gemma-4-12B Q2_K_XL | llama.cpp | 4.66 GB | 13.44 tok/s | 124.7 tok/s | ⚠️ Fuzzy quality |
| **Gemma-4-12B QAT Q4_K_XL** | **llama.cpp** | **6.72 GB** | **12.08 tok/s** | **132.4 tok/s** | **✅ Best quality** |
| Gemma-4-12B QAT 4-bit | Rapid-MLX | 15.4 GB ws | 6.2 tok/s | — | ❌ Memory panic |
| Gemma-4-12B QAT 4-bit | mlx-vlm | 11.1 GB | 11.7 tok/s | 2.2 tok/s | ⚠️ Same speed, more RAM |
| Gemma-4-12B QAT 4-bit | mlx-lm | — | — | — | ❌ Arch not supported |

**Why not MLX?** Gemma 4's `gemma4_unified` architecture lacks a lean text-only path in the MLX ecosystem. llama.cpp's `--no-mmproj` cleanly drops vision weights, saving ~4 GB vs MLX. On 16 GB, that headroom makes the difference.

---

## Performance

| Metric | Value |
|---|---|
| Generation speed | **12.08 tok/s** |
| Prompt processing | **132.4 tok/s** |
| Model size | 6.72 GB (GGUF UD-Q4_K_XL) |
| Total RAM at load | ~7 GB (text-only, `--no-mmproj`) |
| Context | 32768 tokens |
| Opencode overhead | ~15K system prompt → ~17K usable |

---

## Opencode Integration (Next)

Add this to your `opencode.json`:

```json
{
  "provider": {
    "openai": {
      "api": "http://localhost:8080/v1",
      "models": {
        "default": {
          "name": "gemma-4-12b-it-qat",
          "limit": { "context": 32768, "output": 8192 }
        }
      },
      "options": { "apiKey": "not-needed" }
    }
  }
}
```

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `kIOGPUCommandBufferCallbackErrorOutOfMemory` | Model + KV cache > Metal limit | Lower `--ctx-size` or use smaller quant |
| Sluggish generation | KV cache too large | Reduce `--ctx-size` to 16384 |
| Content is null / think tags | Reasoning enabled | Add `--reasoning off` |
| Multimodal error | Missing projector weights | Add `--no-mmproj` |
| Deprecated flag | `--chat-template-kwargs` removed | Use `--reasoning on/off` |

---

## References

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [unsloth/gemma-4-12B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF)
- [Unsloth Gemma 4 Docs](https://unsloth.ai/docs/models/gemma-4)
