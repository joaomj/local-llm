# PLAN-EVAL-MT-BENCH

## Goal

Create two separate evaluations for the four Liquid AI LFM2.5 Q4_0 GGUF
models:

1. Measure chat quality for all four models.
2. Measure hardware performance only for the two highest-quality models.

Quality is the primary decision. A fast model with poor chat quality is not a
useful choice for this project.

Use Python for the workflow orchestrator. Use `llama-server` for production
performance measurements. Use `llama_benchmark.py` only for low-level KV-cache
diagnostics.

## Models

Use the exact entries in `local-llm/scripts/lfm-qad-models.tsv`:

- `LFM2.5-230M-QAD-Q4_0.gguf`, official context 32,768
- `LFM2.5-350M-QAD-Q4_0.gguf`, official context 32,768
- `LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf`, official context 32,768
- `LFM2.5-2.6B-QAD-Q4_0.gguf`, official context 131,072

Process one model at a time. Download each model to a workflow-owned scratch
directory. Keep downloaded weights after each result. Never modify Touro or
`~/Library/Caches/local-llm-benchmark/models`.

## Evaluation Commands

Add `local-llm/scripts/lfm_chat_benchmark.py` as the main entry point.

```bash
uv run python scripts/lfm_chat_benchmark.py quality
uv run python scripts/lfm_chat_benchmark.py quality --dry-run
uv run python scripts/lfm_chat_benchmark.py performance --top 2
uv run python scripts/lfm_chat_benchmark.py all --top 2
```

The `quality` command evaluates all four models. The `performance` command
requires a complete quality report and selects the top two from that report.
The `all` command runs quality first, then performance.

## Quality Evaluation

### Primary Screen: Customized MT-Bench

Use the existing 80-question, two-turn dataset in
`local-llm/benchmarks/mt-bench/question.jsonl`.

Run every model with a fixed 32,768-token server context. This gives the
2.6B model a fair chat comparison without allowing its larger context window
to block quality measurement.

For each model:

1. Download the exact GGUF.
2. Start a local OpenAI-compatible llama.cpp server at 32,768 tokens.
3. Generate and persist all 80 two-turn answers.
4. Submit the saved answers to the OpenRouter Batch API.
5. Retry failed or invalid judgment records.
6. Store the answers, judgments, usage, provider, and failure records.
7. Stop the server and keep the downloaded model weights.

Use the same judge configuration for every model:

- API base: `https://openrouter.ai/api/v1`
- Model: `google/gemini-3.7-flash:batch`
- Temperature: `0`
- Score range: integer `1` through `10`
- Response format: strict JSON Schema
- Credential: `OPENROUTER_API_KEY` or the repository `.env` file
- Retry and spend limits: configurable through environment variables

Use the full MT-Bench mean as the primary quality ranking. Also report scores
by category and show incomplete results as incomplete.

Do not import or report published composite scores from model releases. This
evaluation measures the local customized MT-Bench result only.

### Secondary Confirmation: WildBench v2

Run WildBench v2 only for the two models selected by the MT-Bench ranking.
Use it as a quality confirmation, not as a replacement for the primary
ranking.

WildBench v2 is suitable because it uses challenging tasks from real user
conversations, task-specific checklists, and a larger task set than MT-Bench.
Record the exact WildBench version, judge model, metric, and source data.

Keep WildBench scores separate from MT-Bench scores. Do not average them into a
single score. Do not change the performance top-two selection from the primary
MT-Bench ranking.

## Performance Evaluation

Run performance only for the top two models from the completed MT-Bench quality
report.

Use `local-llm/scripts/llama_server_benchmark.py` to test each selected model.
Start each configured capacity with a real `llama-server` process.

For each context, record:

- Configured context tokens
- Target prompt tokens
- Actual prompt tokens from the server timing data
- Prompt tokens per second
- Generation tokens per second
- Peak memory
- Pass or fail status
- Failure reason

The mandatory performance gate is at least `50` generation tokens per second.
Apply this gate to a short request and an occupied multi-turn tool conversation.
Report the largest configured capacity and occupied prompt as separate values.
Do not select a fast result that fails the quality ranking.

A timeout at a large context is an upper search boundary, not proof that the
model has no useful performance. Continue the context search below the timeout
when the last smaller context passed. Preserve the timeout record.

The performance command must not generate MT-Bench answers or send model output
to OpenRouter.

## Python Orchestrator

Add `local-llm/scripts/lfm_chat_benchmark.py` with these responsibilities:

- Parse `quality`, `performance`, and `all` commands.
- Load the model manifest and validate exact model files.
- Manage workflow-owned GGUF files.
- Start and stop local model servers safely.
- Call the existing answer and batch-judge modules.
- Persist resumable state after every model and stage.
- Select the top two from the completed quality report.
- Call the server performance benchmark for selected models.
- Generate separate quality and performance reports.
- Keep all downloaded model weights.
- Stop before a download when free storage is less than 30 GiB or 15 percent.
- Preserve logs and partial artifacts after failure or interruption.

Keep reusable answer and judgment logic in:

- `local-llm/scripts/mt_bench_answers.py`
- `local-llm/scripts/mt_bench_batch_judge.py`

Use project-local state by default:

- Intermediate artifacts: `.benchmark/lfm-chat-quality/`
- Performance artifacts: `.benchmark/lfm-chat-performance/`
- Final reports: `results/lfm-qad/quality.{json,md}` and
  `results/lfm-qad/performance.{json,md}`

Keep existing partial artifacts intact. Do not reinterpret the old mixed report
as a complete result.

## Reports

Add or update `local-llm/scripts/lfm_qad_report.py` so it writes two reports.

The quality report must contain:

- Four model records
- MT-Bench question count
- Full mean score on the `1` through `10` scale
- Category means
- Judge model and configuration
- Missing or failed judgment count
- MT-Bench ranking
- Selected performance top two
- WildBench v2 confirmation for the selected two
- External benchmark references with source and comparability notes

The performance report must contain:

- Only the selected two models
- All tested configured capacities
- All tested occupied prompts
- Largest passing configured capacity
- Largest passing actual prompt
- Generation speed for each measurement
- Prompt speed
- Peak memory
- Timeout and failure records
- The `>=50 tok/s` gate result

Do not place published composite scores in either report. Do not blend quality
and performance into one score.

## External Chat References

Add `local-llm/benchmarks/mt-bench/reference-scores.json` for context only.
Record the source, date, judge, metric, and model size for each reference.

Use these carefully labeled examples:

- Kanana-2-3B: MT-Bench `7.15`, judged with `gpt-4o-2024-08-06`
- Kanana-2-1.3B: MT-Bench `6.83`, judged with `gpt-4o-2024-08-06`
- Qwen3.5-2B: MT-Bench `6.87`, source model card
- Qwen3-1.7B: MT-Bench `6.98`, source model card
- Qwen2.5-72B-Instruct: MT-Bench `9.35`, Qwen technical report

These references are not directly comparable to the local Gemini-judged run.
Show them as context only. Do not use them to select the top two.

## Implementation Chain

### Step 1: Define the Python workflow contract

- Repository: `local-llm`
- Paths: `local-llm/scripts/lfm_chat_benchmark.py`,
  `local-llm/scripts/lfm-qad-models.tsv`
- Change: Define command parsing, configuration, state paths, and stage names.
- Gate: `uv run python scripts/lfm_chat_benchmark.py --help`
- Pass condition: Help lists `quality`, `performance`, and `all`.
- Stop condition: Stop if command validation cannot distinguish quality from
  performance.

### Step 2: Separate quality execution from performance execution

- Repository: `local-llm`
- Paths: `local-llm/scripts/lfm_chat_benchmark.py`,
  `local-llm/scripts/mt_bench_answers.py`,
  `local-llm/scripts/mt_bench_batch_judge.py`
- Change: Make quality run at 32,768 tokens and make performance read the
  completed MT-Bench ranking before selecting two models.
- Gate: `uv run python scripts/lfm_chat_benchmark.py quality --dry-run`
- Pass condition: The dry run lists four quality models, a 32,768-token context,
  and no performance probe.
- Stop condition: Stop if quality can start performance before all four models
  have complete judgments.

### Step 3: Add WildBench v2 confirmation

- Repository: `local-llm`
- Paths: `local-llm/scripts/lfm_chat_benchmark.py`,
  `local-llm/benchmarks/`
- Change: Add the pinned WildBench v2 data and judge configuration for the two
  MT-Bench leaders. Keep its result separate from the MT-Bench ranking.
- Gate: `uv run python scripts/lfm_chat_benchmark.py quality --dry-run`
- Pass condition: The plan shows WildBench only for the selected two models.
- Stop condition: Stop if WildBench changes the primary top-two selection rule.

### Step 4: Make server performance search independent

- Repository: `local-llm`
- Paths: `local-llm/scripts/llama_server_benchmark.py`,
  `local-llm/scripts/lfm_chat_benchmark.py`
- Change: Measure configured capacity and occupied prompts with `llama-server`.
  Treat a high-capacity failure as an upper boundary.
- Gate: `uv run python -m unittest tests.test_throughput_search`
- Pass condition: Tests pass and both measurements remain separate.
- Stop condition: Stop if a capacity failure discards smaller successful
  measurements.

### Step 5: Build separate reports and references

- Repository: `local-llm`
- Paths: `local-llm/scripts/lfm_qad_report.py`,
  `local-llm/benchmarks/mt-bench/reference-scores.json`
- Change: Write separate quality and performance reports. Remove published
  composite-score inputs from the report path.
- Gate: `uv run python -m unittest tests/test_lfm_qad_report.py`
- Pass condition: Tests verify separate reports, local MT-Bench ranking, and
  context-qualified performance results.
- Stop condition: Stop if either report contains a blended score.

### Step 6: Add black-box workflow tests

- Repository: `local-llm`
- Paths: `local-llm/tests/lfm-qad-benchmark-test.sh`,
  `local-llm/tests/test_lfm_chat_benchmark.py`
- Change: Test dry runs, resume state, top-two selection, weight retention,
  storage limits, incomplete judgment handling, and capacity boundaries.
- Gate: `uv run python -m unittest discover -s tests && bash tests/lfm-qad-benchmark-test.sh`
- Pass condition: All tests pass without downloads, server starts, or API calls.
- Stop condition: Stop if a test can modify Touro or the existing model cache.

## Acceptance Criteria

- The main orchestrator is Python and runs through `uv`.
- Quality and performance use separate commands and state paths.
- Quality evaluates all four models at a fixed 32,768-token context.
- Every completed quality model has 80 answers and 80 valid judgments.
- The primary ranking uses only the local customized MT-Bench mean.
- WildBench v2 runs only for the MT-Bench-selected top two.
- WildBench does not change the performance top-two selection rule.
- Performance evaluates only those two models.
- Performance finds the largest configured capacity at or above 50 generation
  tokens per second, or records that the gate is not met.
- Performance reports the largest actual occupied prompt at the same speed gate.
- Downloaded GGUF files remain after success, rejection, or interruption.
- Answers, judgments, reports, and failure records remain with the weights.
- The API key never appears in logs or result files.
- Touro and the existing local model cache remain unchanged.
- No published composite score is used or reported.
- Dry-run and test commands do not download models or start inference.

## Risks and Boundaries

- External judges receive generated model answers and can incur API cost.
- Judge scores can contain position, length, and model preference bias.
- MT-Bench and WildBench scores use different protocols and remain separate.
- WildBench v2 can cost more than the MT-Bench screen. Run it only for the
  selected two models.
- A 32K quality context does not prove that a model performs well at 131K.
- A long-context timeout does not invalidate smaller passing contexts.
- Failed stages preserve artifacts for inspection and resume.
- The workflow stops before a download if free storage is below its safety limit.
- The operator selects model weights for removal. The workflow does not remove
  them automatically.
- No commit, branch, pull request, or long-running benchmark run is part of this
  plan.

## Verification

Run these checks after implementation:

```bash
uv run python scripts/lfm_chat_benchmark.py quality --dry-run
uv run python scripts/lfm_chat_benchmark.py performance --top 2 --dry-run
uv run python -m unittest discover -s tests
bash tests/lfm-qad-benchmark-test.sh
```

The operator starts the real evaluation only after reviewing both dry-run
outputs and configuring `OPENROUTER_API_KEY`.

## Sources

- Arena-Hard v2: <https://github.com/lmarena/arena-hard-auto>
- WildBench v2: <https://github.com/allenai/WildBench>
- LMArena human preference method: <https://proceedings.mlr.press/v235/chiang24b.html>
- Kanana-2 model card: <https://huggingface.co/kakaocorp/kanana-2-1.3b-instruct>
- Qwen2.5 official report: <https://qwenlm.github.io/blog/qwen2.5-llm/>
