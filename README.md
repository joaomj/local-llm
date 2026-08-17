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

The official 8-bit MLX weights download on first server start. To pre-download them:

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

## Use the endpoint in opencode

The server exposes an OpenAI-compatible API on `http://127.0.0.1:8080/v1`. See `OPENCODE.md` for the provider configuration.

## Performance snapshot

Measured on Mac Mini M4 (16GB), official 8-bit MLX weights:

- Generation: 44-47 tokens/s
- Time to first token: 0.13s (small prompt), 2.1s (after web search)
- Peak memory: 9.3GB

See `REPORT.md` for the full report and the 4-bit quantization analysis.
