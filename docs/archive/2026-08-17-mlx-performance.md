# Performance Report: LFM2.5-8B-A1B on Apple Mac Mini M4

Date: 2026-08-17
Author: local benchmark session
Status: Final

## Executive Summary

Liquid AI's LFM2.5-8B-A1B runs comfortably on a 16GB Mac Mini M4 in the official 8-bit MLX format. The model generates 44-47 tokens per second in live chat. Time to first token is 0.13 seconds for a normal question and about 2.1 seconds when the answer includes live web search results. Peak memory use is 9.3GB, which leaves headroom for the operating system and other applications.

Web search works end to end. The model calls the Exa search API, reads the results, and answers with cited sources.

The single most important recommendation: keep the model at 8-bit precision. Community evidence shows that 4-bit quantization can degrade tool-call reliability for this model. Your 16GB machine can serve the 8-bit version without memory pressure.

## Abstract

We evaluated LiquidAI/LFM2.5-8B-A1B, a Mixture-of-Experts model with 8B total parameters and about 1.5B active parameters per token. We ran it locally on a Mac Mini M4 with 16GB unified memory, using the official 8-bit MLX weights and the mlx-lm server. We measured time to first token, generation speed, prompt processing speed, and memory use. We repeated 10 trials for a plain question and 10 trials for a web-search question, and we report median and p95 values. We also researched the quality impact of 4-bit quantization from public sources.

Results: generation speed is stable at 44-47 tokens/s. Median time to first token is 0.13s for plain questions and 2.11s for answers that follow a web search. Peak memory is 9.31GB. Public evidence suggests 4-bit quantization preserves most general quality but can hurt tool-call reliability, so 8-bit is the safer choice on this hardware.

## Introduction

The goal was to run the LFM2.5-8B-A1B model locally for a chat test with web search, and to measure standard LLM performance metrics. The user's hardware is a Mac Mini M4 with 16GB of unified memory.

Prior context: the user chose the MLX runtime over llama.cpp and Ollama. The user selected Exa as the web search provider.

Scope: local inference only. We did not evaluate answer quality on standard benchmarks. We measured throughput and latency only.

## Methodology

### Setup

- Hardware: Apple M4, 10 cores, 16GB unified memory, macOS 26.6.1
- Model: `LiquidAI/LFM2.5-8B-A1B-MLX-8bit` (official 8-bit MLX quant), 8.4GB on disk
- Runtime: `mlx_lm.server` (mlx-lm 0.31.x), OpenAI-compatible API on port 8080
- Client: custom terminal chat client (`chat.py`) with an Exa `web_search` tool
- Benchmark script: `bench.py`, repeated trials with median and p95 statistics

### Context window configuration

- Model maximum context: 128,000 tokens (from `max_position_embeddings` in `config.json`)
- KV cache: dynamic; the mlx-lm server grows it on demand, no fixed limit flag
- Client response cap: 4,096 tokens per generation (`max_tokens`)
- Largest prompt observed in tests: 3,267 tokens

### Procedure

We ran two scenarios, each repeated 10 times with a fresh conversation:

1. Plain: a reasoning question, rotating among 3 prompt variants
2. Search: a question that requires current facts ("Who won the last men's Wimbledon final and what was the score?"), answered through live Exa search

We captured, per model call: time to first token, generation time, output token count, prompt token count, and cached token count. We also ran the official `mlx_lm.benchmark` tool with a 512-token prompt and 512 generated tokens over 3 trials.

## Results

### Official benchmark (mlx_lm.benchmark)

| Metric | Value |
|---|---|
| Prompt processing (prefill) | 835 tokens/s |
| Generation | 47.6 tokens/s |
| Peak memory | 9.31 GB |

### Live chat: plain question (10 trials)

| Metric | Median | p95 |
|---|---|---|
| Time to first token | 0.125s | 6.84s |
| Generation speed | 45.5 tok/s | 46.4 tok/s |
| Prompt processing | 1,336 tok/s | - |
| Prompt size | 167 tokens | - |
| Output size | 327 tokens | - |

Note: the p95 value comes from one cold-start outlier of 6.84s. The other 9 trials ranged from 0.111s to 0.131s. True p95 without the cold start is about 0.13s.

### Live chat: web search question (10 trials)

| Metric | Median | p95 |
|---|---|---|
| First call, time to first token | 0.135s | 0.162s |
| Answer call, time to first token | 2.11s | 2.23s |
| Generation speed (answer calls) | 44.1 tok/s | 45.7 tok/s |
| Prompt processing (answer calls) | 2,071 tok/s | - |
| Prompt size (answer calls) | 3,267 tokens | - |
| Output size (answer calls) | 259 tokens | - |
| Whole turn, time to first token | 0.135s | 0.15s |
| Whole turn, generation speed | 44.2 tok/s | 44.6 tok/s |

Notes:

- The model performs 1 to 3 search rounds per question. Each round adds about 2 seconds before the final answer.
- The prompt cache is effective. On repeat prompts, 95-99% of prompt tokens were cache hits. This explains the low first-call latency.
- A cold 3,267-token prompt is processed at about 2,070 tokens/s.

### Web search behavior

The search loop worked reliably: the model decided to search, Exa returned results, and the model answered with cited sources. Example answer: "Jannik Sinner won the most recent men's Wimbledon final, defeating Carlos Alcaraz 4-6, 6-4, 6-4, 6-4."

## Discussion

Generation speed is consistent across scenarios (44-47 tok/s). The model is memory-bandwidth-bound in generation, which is expected on this hardware. The 1.5B active parameters make it fast enough for interactive chat.

Time to first token depends strongly on prompt size. A 167-token prompt starts in about 0.13s. A 3,267-token prompt with search results takes about 2.1s. The prefill speed is high, so latency scales with prompt length, not with model slowness.

The 8-bit model uses 9.3GB of memory. This leaves about 6.5GB free on a 16GB machine. No swap activity was observed. A 128K-token context would exceed memory; the tested workloads stayed below 4K tokens, which is safe.

### 4-bit quality evidence

We found no official Liquid AI comparison of 4-bit vs 8-bit quality for this model. Public evidence includes:

1. LocalAI commit `25ecb9f` (June 2026): "The Q4_K_M quant degraded tool-call reliability for LFM2.5-8B-A1B." LocalAI switched its default quant to Q8_0 for this model. Tool calling is the model's core strength, and our chat depends on it.
2. `mlx-community/LFM2.5-8B-A1B-OptiQ-4bit` (sensitivity-mixed 4-bit): MMLU 46.1%, GSM8K 54.6%, IFEval 32.9%, BFCL-V3 24.5%. The eval harness differs from Liquid's official card (IFEval 91.84%, BFCLv3 64.79%), so the numbers are not directly comparable, but the 4-bit scores are visibly weaker.
3. General quantization study (arXiv 2608.08188): at the 8B scale, uniform 4-bit quantization retains about 95% of average performance (example: Qwen3-8B W4 retention 95.9%). Degradation varies by method and task type.
4. Liquid's own guidance: 8-bit is recommended for MLX.

Interpretation: general chat quality at 4-bit is likely close to 8-bit, but tool-call reliability carries a real risk of degradation for this exact model. On 16GB of memory, 8-bit fits comfortably, so the risk is not worth taking. If memory ever becomes tight, 5-bit or 6-bit is a safer middle ground than 4-bit.

## Conclusion

The LFM2.5-8B-A1B model at 8-bit MLX precision runs well on a 16GB Mac Mini M4. It answers plain questions with 45.5 tokens/s and web-search answers at 44.1 tokens/s. Time to first token is about 0.13s for small prompts and 2.1s after processing search results. Peak memory is 9.3GB.

Keep the model at 8-bit precision for this use case. The web-search chat setup is complete and reproducible.

## Next Steps

- Optional: run more trials with `uv run bench.py --trials N` for a larger sample
- Optional: measure 5-bit or 6-bit quants if memory headroom is needed
- Optional: run `mlx_lm.benchmark` on the official 4-bit MLX weights and compare the two locally

## References

- Model card: https://huggingface.co/LiquidAI/LFM2.5-8B-A1B
- Official MLX 8-bit weights: https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-MLX-8bit
- Official MLX 4-bit weights: https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-MLX-4bit
- OptiQ 4-bit eval: https://huggingface.co/mlx-community/LFM2.5-8B-A1B-OptiQ-4bit
- LocalAI quant fix commit: https://github.com/mudler/LocalAI/commit/25ecb9f0158c2d27788f385b4d06fc12dbfd8d0b
- Quantization degradation study: https://arxiv.org/html/2608.08188
- Liquid docs (MLX deployment): https://docs.liquid.ai/deployment/on-device/mlx
- Exa search API: https://exa.ai/docs/reference/search

## Reproducibility

Raw per-call data: `bench_raw.json`
Summarized results: `bench_results.json`
Benchmark script: `bench.py`
Chat client: `chat.py`

Commands:

```bash
mlx_lm.server --model LiquidAI/LFM2.5-8B-A1B-MLX-8bit --host 127.0.0.1 --port 8080
EXA_API_KEY=<key> uv run chat.py
EXA_API_KEY=<key> uv run bench.py --trials 10
```
