# Local LLM Setup - Gemma 4 E2B QAT on Mac mini M4 16GB

Current local model: `unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL`

Use case: local chat and lightweight classification. This setup is not intended to replace the main coding model.

Hardware: Mac mini M4, 16 GB unified memory.

---

## Quick Start

Install llama.cpp:

```bash
brew install llama.cpp
```

Start the best known local server:

```bash
./run_model.sh
```

The server listens on:

```text
http://127.0.0.1:8080/v1
```

Test it:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-qat","messages":[{"role":"user","content":"Classify this as casual_chat, task_request, bug_report, or unknown: my local model is slow"}],"max_tokens":100}'
```

---

## Run Model

`run_model.sh` currently resolves the locally cached GGUF and runs:

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

Keep that terminal running while using the local model.

---

## OpenCode Provider

The global opencode config has a local `llama.cpp` provider:

```text
llama.cpp/gemma-qat
```

The provider expects `llama-server` at `http://127.0.0.1:8080/v1`. It does not start the server automatically.

After changing opencode config, restart opencode. Running sessions keep using the already-loaded config.

---

## Benchmark Results

| Variant | Model | Gen tok/s | Prompt tok/s | Status |
|---|---|---:|---:|---|
| `e2b-q4-32k` | Gemma 4 E2B QAT Q4 | 52.7943 | 148.5310 | Best speed candidate |
| `e4b-q4-32k` | Gemma 4 E4B QAT Q4 | 32.3460 | 92.4689 | Best quality/speed fallback |
| `12b-q4-32k` | Gemma 4 12B QAT Q4 | 13.5111 | 47.3704 | Quality baseline, too slow for this use case |
| `e2b-q2-32k` | Gemma 4 E2B QAT Q2 | - | - | Failed on Metal |
| `e4b-q2-32k` | Gemma 4 E4B QAT Q2 | - | - | Failed on Metal |

Full history is in `benchmarks/benchmark_results.md`.

---

## Benchmark Automation

Run the benchmark suite with uv:

```bash
uv run python scripts/benchmark_gemma_qat.py
```

Useful commands:

```bash
uv run python scripts/benchmark_gemma_qat.py --only e2b-q4-32k --repeat
uv run python scripts/benchmark_gemma_qat.py --only e4b-q4-32k --repeat
uv run python scripts/benchmark_gemma_qat.py --kill-existing
uv run python scripts/benchmark_gemma_qat.py --startup-timeout 900
```

The script checks `llama-server`, port availability, and existing `llama-server` processes. Logs and response JSON files are written to `logs/llama-bench/<timestamp>/`.

The optimization loop uses:

- `OPTIMIZATION_PROTOCOL.md`
- `benchmarks/benchmark_results.md`
- `benchmarks/optimization_memory.md`
- `run_model.sh`

By default, already-recorded variants are skipped. Use `--repeat` only when intentionally rerunning an experiment.

---

## Current Findings

- E2B Q4 is the fastest viable model for local chat/classification.
- E4B Q4 is the fallback if E2B quality is not enough.
- Q2 variants fail on this Apple Metal llama.cpp build with `metal_unsupported_quant_type_35`.
- The 26B A4B MoE is intentionally excluded because its 14.2 GB GGUF leaves too little headroom on a 16 GB Mac mini.
- Only the E2B Q4 Hugging Face cache is retained locally.

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| Provider fails in opencode | `llama-server` is not running | Start `./run_model.sh` first |
| Model ID error | Server alias mismatch | Use `gemma-qat` for raw API calls and `llama.cpp/gemma-qat` in opencode |
| Port busy | Another server uses 8080 | Stop it or change both `--port` and provider `baseURL` |
| Q2 exits during warmup | Metal backend unsupported quant path | Use Q4 variants |
| Metal OOM | Model + KV cache exceeds available Metal memory | Use E2B Q4 or reduce `--ctx-size` |

---

## References

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [opencode llama.cpp provider docs](https://opencode.ai/docs/providers#llamacpp)
- [Unsloth Gemma 4 QAT collection](https://huggingface.co/collections/unsloth/gemma-4-qat)
- [Gemma 4 E2B QAT GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF)
- [Gemma 4 E4B QAT GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF)
- [Gemma 4 12B QAT GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF)
