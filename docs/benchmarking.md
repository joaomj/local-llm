# Benchmarking Guide

This repository compares local GGUF models on Apple Silicon.

The project keeps three result types separate:

- Server throughput
- Local chat quality
- Low-level cache diagnostics

## Repository Layout

| Path | Purpose |
|---|---|
| `scripts/` | Benchmark runners and quality tools |
| `benchmarks/manifests/` | Model and quantization manifests |
| `benchmarks/mt-bench/` | The 80-question, two-turn quality set |
| `benchmarks/reference/` | Published benchmark data used for context |
| `docs/reports/` | Current evaluation reports |
| `docs/archive/` | Historical reports that use older workflows |
| `.benchmark/` | Local model files and resumable runtime data |
| `results/` | Generated comparison output |
| `tests/` | Offline workflow tests |

## Throughput Search

Install the local server and Hugging Face tools:

```bash
brew install llama.cpp
uv sync
```

Review a family without downloading or starting a server:

```bash
uv run --project . python scripts/throughput_context_search.py \
  --dry-run --family gemma4-e2b-it-qat
```

Run a family search:

```bash
uv run --project . python scripts/throughput_context_search.py \
  --family gemma4-e2b-it-qat
```

The search tries quantizations in manifest order. It stops at the first
quantization that meets both gates:

- At least `32768` configured context tokens
- At least `50` generation tokens per second for a short request and an occupied tool conversation

The search records configured capacity and actual prompt length as separate
metrics. It keeps downloaded weights. It stops before a download when free
storage is below 30 GiB or 15 percent.

## Quality Evaluation

Start `llama-server` with a retained GGUF model. Then generate resumable
answers:

```bash
uv run --project . python scripts/mt_bench_answers.py \
  --model-id MODEL_ID \
  --questions benchmarks/mt-bench/question.jsonl \
  --output .benchmark/quality-results/MODEL_ID/answers.jsonl \
  --server-url http://127.0.0.1:18081/v1 \
  --temperature 0.1 \
  --max-tokens 1024
```

Judge saved answers through the OpenRouter Batch API:

```bash
uv run --project . python scripts/mt_bench_batch_judge.py \
  --answers .benchmark/quality-results/MODEL_ID/answers.jsonl \
  --questions benchmarks/mt-bench/question.jsonl \
  --output .benchmark/quality-results/MODEL_ID/judgments.jsonl \
  --dotenv .env \
  --batch-model google/gemini-3.7-flash:batch
```

Set `OPENROUTER_API_KEY` in the environment or in `.env`. Do not commit the
key or generated answer files.

## Verification

Run all offline checks:

```bash
uv run --project . python -m unittest discover -s tests
bash tests/lfm-qad-benchmark-test.sh
uv run --project . ruff format --check .
uv run --project . ruff check .
uv run --project . ty check
```

## Reports

Read the current evaluation in
`docs/reports/2026-08-23-gguf-throughput-quality.md`.

The archived MLX report in `docs/archive/2026-08-17-mlx-performance.md` uses a
different runtime and is not comparable to the current GGUF results.
