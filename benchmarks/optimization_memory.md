# Optimization Memory

## Best Configuration

Current best remains the previously recorded configuration at 52.7943 tok/s.
No latest run improved generation speed by at least 2%.

## Latest Run Patterns

- `gemma-e2b-q4-49k` (unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL): 53.0002 gen tok/s, 400.7748 prompt tok/s. Hypothesis: Current 49K real-chat baseline with Flash Attention and thinking disabled.
- E4B and Qwen3.5 were tested in the latest comparison, rejected for the current objective, and removed from the active model set.

## Failed Or Lower-Performing Configurations

- No failures in the latest run.

## Next Hypothesis

No unrecorded active variants remain. Use E2B Q4 as the local chat/classifier model. Q2 variants failed on Metal.
