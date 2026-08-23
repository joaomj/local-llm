# Local LLM Benchmarks

Compare local GGUF models on Apple Silicon. The repository measures real
`llama-server` throughput, local chat quality, and low-level cache behavior.

## Results

The current findings are in
[`docs/reports/2026-08-23-gguf-throughput-quality.md`](docs/reports/2026-08-23-gguf-throughput-quality.md).

The short result is:

- Select LFM2.5 1.2B QAD Q4 for speed and judged chat quality.
- Retain Gemma E2B QAT Q2 for long-context use.
- Treat Gemma quality as partial because 11 of 80 answers repeat until the output limit.

## Requirements

- macOS on Apple Silicon
- Homebrew
- `uv`
- `llama-server` from `llama.cpp`
- Hugging Face CLI `hf` for model downloads
- `OPENROUTER_API_KEY` for quality judging

Install the local tools:

```bash
brew install llama.cpp
uv sync
```

## Throughput Search

Review a family without downloading models or starting a server:

```bash
uv run --project . python scripts/throughput_context_search.py \
  --dry-run --family gemma4-e2b-it-qat
```

Run the family search:

```bash
uv run --project . python scripts/throughput_context_search.py \
  --family gemma4-e2b-it-qat
```

The search uses the manifest in
`benchmarks/manifests/throughput-models.tsv`. It tries higher-quality
quantizations first. It stops at the first candidate that meets both gates:

- `32768` or more configured context tokens
- `50` or more generation tokens per second for short and occupied requests

The workflow keeps downloaded weights. It stops before a download when free
storage is below 30 GiB or 15 percent. It does not remove weights.

## Quality Test

Start `llama-server` with a retained model. Generate the two-turn answers:

```bash
uv run --project . python scripts/mt_bench_answers.py \
  --model-id MODEL_ID \
  --questions benchmarks/mt-bench/question.jsonl \
  --output .benchmark/quality-results/MODEL_ID/answers.jsonl \
  --server-url http://127.0.0.1:18081/v1 \
  --temperature 0.1 \
  --max-tokens 1024
```

Judge the saved answers:

```bash
uv run --project . python scripts/mt_bench_batch_judge.py \
  --answers .benchmark/quality-results/MODEL_ID/answers.jsonl \
  --questions benchmarks/mt-bench/question.jsonl \
  --output .benchmark/quality-results/MODEL_ID/judgments.jsonl \
  --dotenv .env \
  --batch-model google/gemini-3.7-flash:batch
```

The answer generator resumes existing records. The judge rejects truncated
answers. Keep incomplete results separate from valid scores.

## Verification

Run the offline test suite:

```bash
uv run --project . python -m unittest discover -s tests
bash tests/lfm-qad-benchmark-test.sh
uv run --project . ruff format --check .
uv run --project . ruff check .
uv run --project . ty check
```

## Repository Layout

| Path | Purpose |
|---|---|
| `scripts/` | Benchmark runners and quality tools |
| `benchmarks/manifests/` | Model and quantization manifests |
| `benchmarks/mt-bench/` | The quality question set |
| `benchmarks/reference/` | Published benchmark context |
| `docs/` | Guides, current findings, and historical reports |
| `.benchmark/` | Ignored local weights and runtime data |
| `results/` | Generated reports |
| `tests/` | Offline tests |

The root `chat.py` and `bench.py` files provide the older MLX web-search chat
workflow. See `docs/archive/2026-08-17-mlx-performance.md` for its historical
results. That workflow is separate from the current GGUF evaluation.

See [`docs/benchmarking.md`](docs/benchmarking.md) for the full workflow.
