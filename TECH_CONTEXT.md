# Technical Context — Local LLM on Mac mini M4 16GB

> **Single source of truth for all technical decisions, measurements, and configuration.**
> Last updated: 2026-06-06 (Updated: switched back to llama.cpp after exhaustive MLX evaluation)

---

## Hardware Specs

| Component | Specification |
|---|---|
| Machine | Mac mini M4 (2024) |
| RAM | 16 GB unified memory (LPDDR5X) |
| GPU | Apple M4 10-core GPU |
| Neural Engine | 16-core |
| Metal GPU visible | ~12,124 MiB (~11.8 GB) |
| Memory bandwidth | ~120 GB/s |

### Observed System Memory Layout

```
Total RAM:           16 GB
macOS overhead:      ~2–4 GB (wired + compressed)
Available for model: ~12–14 GB
Metal GPU usable:    ~11.8 GB
SWAP total:          2 GB (observed 1.35 GB used during Qwen test)
```

---

## Model Evaluation History

### 1. Qwen3.6-35B-A3B (MoE)

| Variant | Quant | Size | Ctx | Result | Speed |
|---|---|---|---|---|---|
| Q2_K_XL | GGUF | 11 GB | 16K | ❌ OOM (`kIOGPUCommandBufferCallbackErrorOutOfMemory`) | — |
| IP2_XXS | GGUF | 10.8 GB | 16K | ✅ Loaded | 27.9 tok/s |

**Architecture notes:**
- 35B total params, 3B active per token (MoE)
- 262K native context
- 40 layers: 10 Gated Attention + 30 Gated DeltaNet (RNN hybrid)
- **KV cache challenge:** DeltaNet layers use RNN state, not KV cache — but attention layers still need ~2–4 GB for 16K ctx

**Decision:** Removed from plan. MoE speed is good but context room is tight with 15K system prompt overhead.

### 2. Gemma-4-12B-it (Dense) — Final Pick

| Variant | Engine | Format | Size | Ctx | Gen Speed | Prompt Speed | Verdict |
|---|---|---|---|---|---|---|---|
| Q2_K_XL | llama.cpp | GGUF | 4.66 GB | 32K | 13.44 tok/s | 124.7 tok/s | ⚠️ Fuzzy quality, good speed |
| QAT Q4_K_XL | **llama.cpp** | **GGUF** | **6.72 GB** | **32K** | **12.08 tok/s** | **132.4 tok/s** | **✅ Final pick** |
| QAT 4-bit | mlx-vlm | MLX | 11.1 GB | 32K | 11.7 tok/s | 2.2 tok/s | ❌ Same speed, worse memory |
| QAT 4-bit | Rapid-MLX | MLX | 15.4 GB ws | 32K | 6.2 tok/s | — | ❌ Memory panic, reasoning forced |

#### Quality Comparison (Same Prompt)

Prompt: "explain transformers LLMs like i am 12."

**Q2_K_XL:**
- Generated: 686 tokens in 51.0s (13.44 tok/s)
- Tone: Casual, kid-friendly
- Issues: Fuzzy concepts ("the 'T' in ChatGPT", "A in Attention" — these are not correct)
- Structure: Lego analogy, simple metaphors
- Accuracy: ⚠️ Acceptable for chat, not for technical accuracy

**QAT Q4_K_XL:**
- Generated: 907 tokens in 75.1s (12.08 tok/s)
- Tone: Professional but accessible
- Strengths: Precise Self-Attention examples (animal/street co-reference), correct RNN vs Transformer distinction
- Structure: Clear sections, accurate technical content
- Accuracy: ✅ Good for both chat and technical use

**Verdict:** QAT is 32% more verbose but significantly more accurate. For chat + router use, QAT is preferred despite slightly lower speed.

### MLX ML Evaluation (Why it didn't win)

We tested three MLX paths for Gemma 4. None beat llama.cpp on this 16GB machine:

| Path | Issue | Root cause |
|---|---|---|
| `mlx-lm` | ❌ Cannot load | `gemma4_unified` architecture not supported in `mlx_lm.models` |
| `mlx-vlm` | ⚠️ 11.1 GB peak, 11.7 tok/s | Loads unified multimodal architecture even for text-only — no `--no-mmproj` equivalent |
| Rapid-MLX | ❌ 15.4 GB working set, 6.2 tok/s | Forced reasoning parser, memory throttle on 16GB |

**Why MLX lost:**
1. The MLX model (10.3 GB disk) is larger than GGUF QAT (6.72 GB) — different quantization packing
2. `mlx-vlm` loads the full multimodal pipeline; no text-only flag exists
3. 16 GB is borderline — once memory pressure hits 85%+, macOS throttles and performance collapses
4. llama.cpp's Metal backend is mature and highly optimized for GGUF text-only inference

**Takeaway:** MLX is not inherently slower — it just needs comfortable memory headroom. On a 24 GB+ Mac, MLX would likely match or beat llama.cpp. On 16 GB, the tighter fit cancels the architectural advantage.

---

## Engine Comparison

### llama.cpp (✅ Final choice)

**Why it wins on this hardware:**
- GGUF text-only path is lean (`--no-mmproj` drops ~2 GB of vision weights)
- Mature Metal backend with optimized kernels
- `--reasoning off` works correctly (forces content output)
- 6.72 GB model leaves ~9 GB for KV cache + macOS on 16 GB
- 12 tok/s is acceptable for chat

**Optimization possibilities (not yet tested):**
- `--flash-attn` — could reduce memory bandwidth and increase gen speed
- `--ctx-size 16384` — reduces KV cache, freeing bandwidth (from 32K)
- `--threads` tuning — M4 10-core needs balanced CPU/GPU

### MLX Ecosystem (Evaluated, Rejected)

All three MLX paths failed to beat llama.cpp on 16 GB:

| Engine | Status | Why |
|---|---|---|
| `mlx-lm` (0.31.3) | ❌ Cannot load | `gemma4_unified` arch not in `mlx_lm.models` |
| `mlx-vlm` (0.6.2) | ⚠️ 11.1 GB, 11.7 tok/s | Full multimodal pipeline even text-only |
| Rapid-MLX (0.6.80) | ❌ 15.4 GB ws, 6.2 tok/s | Memory throttle + forced reasoning |

**Root cause:** The MLX models from `mlx-community` are `gemma4_unified` (Image-Text-to-Text). Even for text-only inference, the full multimodal pipeline loads. llama.cpp's `--no-mmproj` cleanly skips vision weights — MLX has no equivalent flag.

---

## Configuration

### Python Environment

```bash
# Install llama.cpp (the only engine we need)
brew install llama.cpp
```

### Model Cache Location (llama.cpp)

llama.cpp auto-downloads GGUF models from HuggingFace into its own cache (like `~/.cache/llama.cpp` or alongside the binary). See `brew info llama.cpp` for exact location.

### Model Caches (History)

| Cache | Status | Size |
|---|---|---|
| `models--mlx-community--gemma-4-12B-it-qat-4bit` | ✅ Deleted | ~10.3 GB |
| `models--unsloth--Qwen3.6-35B-A3B-GGUF` | ✅ Already removed | — |
| `models--unsloth--gemma-4-12B-it-qat-GGUF` | ✅ Already removed | — |
| `models--unsloth--gemma-4-12b-it-GGUF` | ✅ Already removed | — |

**Note:** The QAT GGUF model was removed during cleanup and needs re-download (`llama-server` will auto-download on first run).

---

## Context Window Math

The opencode harness injects ~15K tokens of system prompt. With 16K total ctx:

```
Total ctx:     16,384 tokens
System prompt: ~15,000 tokens
User prompt:   ~1,000 tokens
Response:      ~384 tokens  (practically unusable)
```

**Recommended minimum:** 32K ctx for opencode use

| Ctx Size | Usable After System Prompt | Safe on 16GB? |
|---|---|---|
| 16K | ~1K | Yes (tight) |
| 32K | ~17K | Yes (comfortable) |
| 64K | ~49K | Maybe (depends on KV cache) |

For dense 12B models, KV cache at 32K is ~1–2 GB, which fits within the ~12GB available.

---

## Known Issues & Fixes

### 1. Metal OOM (`kIOGPUCommandBufferCallbackErrorOutOfMemory`)

**Trigger:** Model + KV cache exceeds Metal-visible memory (~11.8 GB)

**Observed with:**
- Qwen Q2_K_XL (11 GB model) at 16K ctx
- Qwen IP2_XXS (10.8 GB) worked because it left ~1 GB for KV cache

**Fix:**
- Reduce `--ctx-size`
- Use smaller quant
- Disable vision (`--no-mmproj`)

### 2. Slow Generation (<15 tok/s on 12B)

**Cause:** Memory bandwidth saturation on M4 16GB

**Evidence:**
- Q2 (smaller) was 13.44 tok/s vs QAT (larger) at 12.08 tok/s
- Speed scales inversely with model size on bandwidth-limited hardware
- MLX evaluations confirmed same limitation at 11.7 tok/s

**Fix:**
- Try `--flash-attn` with llama.cpp to reduce memory bandwidth
- Reduce `--ctx-size` (16384 instead of 32768)
- Tune `--threads` for the M4's 10-core layout

### 3. Vision/Multimodal Errors

**Cause:** Model expects mmproj weights but they're not loaded

**Fix:** Always add `--no-mmproj` for text-only use

### 4. Deprecated Flag Warnings

**Old:** `--chat-template-kwargs reason=off`
**New:** `--reasoning off`

---

## Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-06-06 | Removed Qwen from plan | Context too tight (16K cap), OOM at Q2_K_XL |
| 2026-06-06 | Chose QAT Q4 over Q2 | Quality significantly better, acceptable speed tradeoff |
| 2026-06-06 | Skipped llama.cpp optimizations | User wanted direct MLX test |
| 2026-06-06 | Chose Rapid-MLX over mlx-lm | Faster, prompt cache, tool calling, benchmarks look good |
| 2026-06-06 | Chose `gemma-4-12b-qat` alias | Maps to `mlx-community/gemma-4-12B-it-qat-4bit` (best speed/quality tradeoff) |
| 2026-06-06 | Skipped MTP for now | Add complexity only after base validation |
| 2026-06-06 | **Abandoned Rapid-MLX** | Memory pressure (131%), forced reasoning parser, 6 tok/s (slower than llama.cpp) |
| 2026-06-06 | **Tested mlx-vlm** | 11.1 GB peak, 11.7 tok/s — works but no advantage over GGUF |
| 2026-06-06 | **Confirmed mlx-lm cannot load Gemma 4** | `gemma4_unified` architecture not supported in mlx_lm.models |
| 2026-06-06 | **Final: stuck with llama.cpp GGUF** | Text-only, lean, `--reasoning off` works, proven reliable |

---

## Web Sources

| Source | URL | Used For |
|---|---|---|
| llama.cpp | https://github.com/ggerganov/llama.cpp | Main engine |
| unsloth/gemma-4-12B-it-qat-GGUF | https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF | **Final chosen model** |
| unsloth/gemma-4-12b-it-GGUF | https://huggingface.co/unsloth/gemma-4-12b-it-GGUF | GGUF baseline model (non-QAT) |
| Unsloth Gemma 4 Docs | https://unsloth.ai/docs/models/gemma-4 | Quantization info |
| mlx-community/gemma-4-12B-it-qat-4bit | https://huggingface.co/mlx-community/gemma-4-12B-it-qat-4bit | MLX model (evaluated, rejected) |
| Gemma 4 QAT Collection | https://huggingface.co/collections/mlx-community/gemma-4-qat | MLX quants (eval'd, rejected) |
| Rapid-MLX GitHub | https://github.com/raullenchai/Rapid-MLX | MLX engine (eval'd, rejected) |
| Qwen3.6-35B-A3B GGUF | https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF | First model tested (OOM) |

---

## Next Steps

### Phase 1 — Server setup & optimization (before opencode)

1. [ ] Start `llama-server` with the QAT GGUF model (downloads if not cached)
2. [ ] Verify model responds via curl
3. [ ] Benchmark: same 3 prompts, compare tok/s to previous runs
4. [ ] Try `--flash-attn` for speed boost
5. [ ] Evaluate `--ctx-size 16384` vs 32768 tradeoff (context vs speed)
6. [ ] Tune `--threads` for M4 10-core
7. [ ] Lock in optimal config

### Phase 2 — Opencode integration (after optimization)

8. [ ] Configure opencode to point at local llama-server
9. [ ] Test basic chat interaction through opencode
10. [ ] Test tool calling through opencode

### Phase 3 — Evaluation

11. [ ] Assess if 12B QAT is sufficient for chat + router
12. [ ] Consider if 26B-A4B at Q2 could fit (was ~8 GB on disk, may OOM)
13. [ ] Document optimal configuration for opencode

---

## File Inventory

| File | Purpose |
|---|---|
| `README.md` | Human-readable quick start and summary |
| `TECH_CONTEXT.md` | This file — technical single source of truth |
| `qat-chat.json` | QAT Q4_K_XL chat output sample |
| `gemma-q2-chat.json` | Q2_K_XL chat output sample |
| `test_mlx_vlm.py` | ✅ Deleted |
| `.venv/` | ✅ Deleted |
| `server.log` | ✅ Deleted |

---

## Commands Reference

### Check llama.cpp installation
```bash
brew install llama.cpp
llama-server --version
```

### Run Server (auto-downloads model, then serves)
```bash
llama-server \
  -hf unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL \
  --no-mmproj \
  --reasoning off \
  --temp 0.8 \
  --top-p 0.95 \
  --top-k 64 \
  --ctx-size 32768 \
  --tools all
```
First run downloads ~6.72 GB. CTRL+C to stop.

### Test with curl
```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","messages":[{"role":"user","content":"Say hello"}],"max_tokens":50}'
```

### Opencode integration (next step)
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

### Cache Cleanup (done)
```bash
# All MLX artifacts removed:
rm -rf ~/.cache/huggingface/hub/models--mlx-community--gemma-4-12B-it-qat-4bit  # 10 GB
rm -rf .venv
rm test_mlx_vlm.py
rm server.log
```

