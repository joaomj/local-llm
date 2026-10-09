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

## Ornith Evaluation

The staged runner evaluates 9B GGUF and MLX variants. It screens speed before
running paid quality judging. It confirms candidates only after both gates pass.
The baseline is the recorded LFM comparison, not a fresh baseline run.

Install the requirements listed in the README. Put `llama-server` on `PATH`
and configure the judge key as described above.

Validate the environment without downloading models or starting servers:

```bash
uv run --project . python scripts/ornith_orchestrator.py --dry-run
```

Run the workflow while retaining downloaded weights:

```bash
uv run --project . python scripts/ornith_orchestrator.py --keep-losers
```

Warning: Without `--keep-losers`, the runner deletes rejected GGUF weight
folders and rejected MLX model entries from the shared Hugging Face cache.
Other applications can lose access to those cached models. Quality judging
uses a paid remote API. `--dry-run` also checks the judge configuration.

Use `--skip-mlx` or `--skip-gguf` to select one runtime. Use `--report-only`
to generate a report from existing local results without running models or
cleanup. Screening, confirmation, and quality records remain in `.benchmark/`.
Generated reports are written to `docs/reports/`; review them before publishing.

The GGUF screen uses one 64-token sample at 32K configured context. Confirmation
uses three 128-token samples and tests through 65,536 configured context tokens.
The MLX screen uses one trial; confirmation defaults to ten trials. MLX occupied
prompts use a word-count estimate, not an exact tokenizer count. The runner does
not configure or independently verify a 32K MLX server capacity. Do not treat
its MLX context field as proof of capacity or direct GGUF comparability.

### Reuse Weights For Quality

Set `LFM_CHAT_QUALITY_SOURCE_ROOT` to the retained throughput-model directory
to reuse matching GGUF weights. The quality runner creates hard links, so both
directories must be on the same filesystem. It reports a linking error rather
than silently copying a second model. Removing the quality scratch link does
not remove the source weight.

### Server Model Names

When a server model name differs from the benchmark ID, pass
`--request-model SERVER_MODEL` to `scripts/mt_bench_answers.py`. The request uses
that name; saved answer records retain `--model-id`.

### Screening Controls

`LLAMA_SERVER_BENCHMARK_REPETITIONS` accepts values of one or more. The default
remains three. Explicit `LLAMA_SERVER_BENCHMARK_CAPACITIES` no longer adds the
maximum context automatically; the 32K baseline remains mandatory.

## Local Storage And Privacy

Keep weights in `models/`, `.benchmark/`, or a model registry. These local
folders and common weight formats are ignored by Git. Generated results, logs,
secret files, and draft plans are also ignored. Manifests contain model locations,
not weights. Review any report for private paths or data before committing it.

## Verification

Install `ruff` and `ty` separately. Run all offline checks:

```bash
uv run --project . python -m unittest discover -s tests
bash tests/lfm-qad-benchmark-test.sh
ruff format --check .
ruff check .
ty check
```

## Reports

Read the current evaluation in
`docs/reports/2026-08-23-gguf-throughput-quality.md`.

The archived MLX report in `docs/archive/2026-08-17-mlx-performance.md` uses a
different runtime and is not comparable to the current GGUF results.
