# Gemma QAT Optimization Protocol

Before each experiment, read:

- `README.md`
- `TECH_CONTEXT.md`
- `benchmark_results.md`
- `optimization_memory.md`
- `best_command.sh`
- `scripts/benchmark_gemma_qat.py`

Do not repeat experiments unless the previous result was noisy or incomplete.

Use previous results to choose the next test:

- Start near the current best configuration: Gemma 4 E2B QAT Q4 GGUF, 32K context, `--parallel 1`.
- Change exactly one variable at a time.
- Prefer experiments relevant to local chat and lightweight classification.
- Continue exploring settings that improve generation speed, prompt speed, memory stability, or latency consistency.
- Stop exploring settings that consistently reduce performance or increase memory risk.
- Treat OOM, Metal errors, server crashes, or unusable output quality as failed configurations.
- Treat E2B/E4B `UD-Q2_K_XL` as failed on the current Apple Metal build unless llama.cpp adds support for the observed `metal_unsupported_quant_type_35` path.
- Do not include the 26B A4B MoE in routine tests on this 16 GB machine.

For every new experiment, explain:

- Why it was chosen.
- Which previous result motivates it.
- What improvement is expected.
- What counts as success or failure.

After each experiment:

- Record results in `benchmark_results.md`.
- Update `best_command.sh` if performance improves meaningfully without hurting stability or chat/classifier quality.
- Update `optimization_memory.md` with the best configuration, discovered patterns, failed configurations, and next hypothesis.

Decision rules:

- Prefer stable improvements over tiny one-off gains.
- A speed gain below 2% is not meaningful unless repeated.
- If two configs are within noise, prefer the simpler and safer config.
- For chat/classifier use, prioritize stability, classification quality, generation speed, prompt processing speed, then memory headroom.
