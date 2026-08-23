# Local LLM: LFM2.5-8B-A1B on Apple Silicon

Run Liquid AI's LFM2.5-8B-A1B locally with web search, and measure its performance. The project targets macOS on Apple Silicon (tested on a Mac Mini M4, 16GB).

The model is a Mixture-of-Experts LLM with 8B total parameters and about 1.5B active parameters per token. It runs in the official 8-bit MLX format.

## What is included

| File | Purpose |
|---|---|
| `chat.py` | Terminal chat client with Exa web search and per-turn metrics |
| `bench.py` | Repeated-trials benchmark with median and p95 metrics |
| `OPENCODE.md` | Use the endpoint as a model provider in opencode |
| `REPORT.md` | Full performance report and 4-bit quality research |

## Requirements

- macOS with Apple Silicon
- 16GB RAM or more
- Homebrew
- `uv` (Python package manager)
- An Exa API key (https://exa.ai) for web search

## Setup

### 1. Install the MLX tools

```bash
uv tool install mlx-lm
uv tool install huggingface_hub
```

### 2. Download the model

The official 8-bit MLX weights live on Hugging Face: https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-MLX-8bit

They download on first server start. To pre-download them:

```bash
hf download LiquidAI/LFM2.5-8B-A1B-MLX-8bit
```

### 3. Set the Exa API key

```bash
export EXA_API_KEY=your-key-here
```

The chat client also reads the key from a `.env` file in the project root (`EXA_API_KEY=...`).

### 4. Start the model server

```bash
mlx_lm.server --model LiquidAI/LFM2.5-8B-A1B-MLX-8bit \
  --host 127.0.0.1 --port 8080 \
  --max-tokens 4096
```

The `--max-tokens` flag is important. The server default is 512. This model reasons before answering, so the default truncates responses.

## Usage

### Chat with web search

```bash
uv run chat.py
```

The model searches the web through the Exa API when the question needs current information.

### Run the benchmark

```bash
uv run bench.py --trials 10
```

The benchmark prints median and p95 for time to first token, generation speed, and prompt processing speed. Results also save to `bench_raw.json`.

### Search server throughput

Use the server-throughput search to compare ordered GGUF quantizations. Install
`llama-server` before a real run:

```bash
brew install llama.cpp
```

Review the Gemma E2B QAT search without a download or server start:

```bash
uv run --project . python scripts/throughput_context_search.py \
  --dry-run --family gemma4-e2b-it-qat
```

Run the historical Gemma E2B QAT Q4 baseline:

```bash
uv run --project . python scripts/throughput_context_search.py \
  --model gemma4-e2b-it-qat-ud-q4-k-xl
```

The result keeps two measurements separate:

- Configured context capacity for a short request
- Actual prompt tokens for a multi-turn tool conversation

Both measurements use a minimum generation speed of `50` tokens per second.
The minimum configured capacity is `32768` tokens. Results are in
`.benchmark/throughput-server/`.

The workflow keeps all downloaded weights. Before each download, it checks free
storage. It stops if free storage is less than 30 GiB or 15 percent. The
operator must then choose which model weights to keep.

`scripts/llama_benchmark.py` remains available for low-level KV-cache
diagnostics. It does not select the production server capacity.

### Compare LFM2.5 QAD checkpoints

The repository also contains a sequential QAD comparison workflow for the four
official LFM2.5 QAD Q4_0 GGUF files. It downloads and processes one model at a
time, then removes only its workflow-owned internal scratch directory. It does
not use Touro or the existing local model cache.

Review the plan without starting downloads or inference:

```bash
bash scripts/lfm-qad-benchmark.sh --dry-run
```

Before a real run, make sure `hf`, `llama-bench`, `llama-server`, `uv`,
`jq`, and `curl` are installed. Put `OPENROUTER_API_KEY` in the environment or
the repository `.env` file. The long-running workflow is:

```bash
bash scripts/lfm-qad-benchmark.sh
```

The same command also builds the final JSON and Markdown comparison report
under `results/lfm-qad/`. Temporary benchmark state, model files, answers, and
judgments stay under the project-local `.benchmark/` directory.

## Use the endpoint in opencode

The server exposes an OpenAI-compatible API on `http://127.0.0.1:8080/v1`. See `OPENCODE.md` for the provider configuration.

## Performance snapshot

Measured on Mac Mini M4 (16GB), official 8-bit MLX weights:

- Generation: 44-47 tokens/s
- Time to first token: 0.13s (small prompt), 2.1s (after web search)
- Peak memory: 9.3GB

See `REPORT.md` for the full report and the 4-bit quantization analysis.
