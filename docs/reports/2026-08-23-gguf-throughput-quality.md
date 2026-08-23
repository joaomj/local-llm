# GGUF Throughput And Quality Findings

Date: 2026-08-23

Status: Complete with one quality coverage gap

## Executive Summary

Gemma E2B QAT Q2 is the selected throughput model. It passes the `50 tok/s`
gate with `131072` configured context tokens and a `32854`-token occupied
prompt. LFM2.5 1.2B QAD Q4 is faster and scores higher on the local quality
screen.

LFM reaches `117.06 tok/s` for a short request and `101.30 tok/s` with a
`16476`-token occupied prompt at `32768` configured context tokens. It receives
a mean quality score of `4.44/10` across 80 judged answers.

Gemma receives `2.72/10` across 69 judgeable answers. Eleven Gemma answers
repeat until the output limit. Treat this as a generation-quality failure, not
as a missing score that can be ignored.

Recommendation: use LFM when speed and answer quality have priority. Use Gemma
when a much larger context window has priority and the observed repetition
risk is acceptable.

## Abstract

This evaluation compares local GGUF models with a real `llama-server` process
and a local two-turn chat quality set. The throughput gate requires 50
generation tokens per second at 32768 or more configured context tokens. The
quality screen uses 80 questions and the same OpenRouter batch judge for both
models. Gemma E2B QAT Q2 passes the throughput gate and provides the largest
tested context. LFM2.5 1.2B QAD Q4 provides higher speed and a higher judged
quality mean. Gemma quality remains partial because 11 answers do not stop.

## Introduction

The project needs a local model that can support long conversations without
discarding useful answer quality. Earlier low-level `llama-bench` results did
not represent the production server path, so the final throughput screen uses
`llama-server` and a deterministic multi-turn conversation with tool calls.

The quality screen measures local chat behavior. It does not reproduce the
published Liquid AI QAD score, and it does not combine quality and speed into a
single score.

## Methodology

### Throughput

- Runtime: `llama-server`
- Contract: `llama_server_capacity_and_prompt-v3`
- Repetitions: three samples per measurement, reported as medians
- Short request: a small generation request
- Occupied request: a deterministic multi-turn conversation with web-search tool calls
- Speed gate: `50` generation tokens per second
- Minimum configured capacity: `32768` tokens
- Storage floor: 30 GiB and 15 percent free space

The family search follows each manifest's quantization order. It keeps weights
after pass, failure, interruption, or rejection.

### Quality

- Dataset: `benchmarks/mt-bench/question.jsonl`
- Dataset size: 80 questions with two turns each
- Candidate generation temperature: `0.1`
- Judge: `google/gemini-3.7-flash:batch` through OpenRouter
- Judge temperature: `0`
- Score range: integer from 1 to 10

The LFM answer set has 80 valid judgments. Gemma has 69 valid judgments. The
remaining 11 Gemma answers reach the generation limit while repeating content.

## Results

### Throughput Gate

| Candidate | 32K short | 32K occupied | Largest passing configured context | Largest passing occupied prompt | Result |
|---|---:|---:|---:|---:|---|
| Gemma E2B QAT Q4 | 50.64 tok/s | 48.50 tok/s | 32768 | none | Reject |
| Gemma E2B QAT Q2 | 51.68 tok/s | 70.53 tok/s at 8K; 51.07 tok/s at 32K | 131072 | 32854 | Select |
| Gemma E4B QAT Q4 | 33.68 tok/s | 30.54 tok/s | none | none | Reject |
| Gemma E4B QAT Q2 | 34.64 tok/s | 31.30 tok/s | none | none | Reject |
| Qwen3.5 9B quantizations | Below gate | Below gate | none | none | Reject all |
| LFM2.5 1.2B QAD Q4 | 117.06 tok/s | 113.15 tok/s at 8K; 101.30 tok/s at 16K | 32768 tested | 16476 | Pass |

Gemma E2B Q2 passes at 131072 configured context on a short request. Its
occupied-prompt boundary is lower: the last passing occupied prompt is 32854
tokens at 49152 configured context.

### Quality Screen

| Model | Valid judgments | Mean | Range | Generation failures |
|---|---:|---:|---:|---:|
| LFM2.5 1.2B QAD Q4 | 80 | 4.44 | 1 to 10 | 0 |
| Gemma E2B QAT Q2 | 69 | 2.72 | 1 to 10 | 11 |

The Gemma mean uses only judgeable answers. The 11 failed generations are not
included in that mean.

### Operational Findings

- Both retained model weights remain on disk.
- The storage floor stops before a new download. It does not remove weights.
- The LFM server benchmark uses the cache file at
  `~/Library/Caches/local-llm-benchmark/models/`.
- The Gemma weight remains under `.benchmark/throughput-models/`.
- Runtime logs, answers, judgments, and raw summaries remain under ignored
  `.benchmark/` paths.

## Discussion

LFM is the stronger default for interactive use. It is more than twice as fast
as Gemma on the short server test and has the higher judged quality mean.

Gemma E2B Q2 is still useful for long-context work. It reaches the 131K
configured capacity target, while LFM was tested only to 32K in this run.
However, Gemma's repeated non-terminating answers create a clear quality risk.

The quality means are not fully symmetric because Gemma has 11 failed
generations. A complete comparison requires a generation setup that produces
valid stopping behavior for all 80 questions. The current evidence still
favors LFM because it has both the higher mean and complete coverage.

The old MLX report is historical. Its speed, memory, and quality claims use a
different model and runtime. Do not compare those values directly with this
report.

## Conclusion

Select LFM2.5 1.2B QAD Q4 for the default local chat path when speed and answer
quality matter most. Retain Gemma E2B QAT Q2 as the long-context option. Do not
call Gemma's quality result complete until its 11 repeated generations are
resolved.

## Next Steps

1. Keep the LFM and Gemma weights retained for local use.
2. If Gemma remains a candidate, repeat only its 11 failed questions with a
   model-specific stopping strategy and record the result separately.
3. Do not run another quantization search unless a new model family is added.

## References

- Throughput code: `scripts/llama_server_benchmark.py`
- Family search code: `scripts/throughput_context_search.py`
- Quality answer code: `scripts/mt_bench_answers.py`
- Quality judge code: `scripts/mt_bench_batch_judge.py`
- Quality questions: `benchmarks/mt-bench/question.jsonl`
- Throughput summaries: `.benchmark/throughput-server/`
- LFM quality judgments: `.benchmark/lfm-qad-results/lfm2.5-1.2b-instruct-qad-q4_0/judgments.jsonl`
- Gemma quality judgments: `.benchmark/quality-results/gemma-4-e2b-it-qat-ud-q2/judgments-valid.jsonl`
