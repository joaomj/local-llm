# Optimization Memory

## Best Configuration

Current best: `e2b-q4-32k` at 52.7943 generation tok/s and 148.5310 prompt tok/s.

Best quality/speed compromise candidate: `e4b-q4-32k` if E2B quality is insufficient.

## Discovered Patterns

- E2B Q4 is the fastest viable model for local chat/classification.
- E4B Q4 is the fallback if E2B quality is insufficient.
- 12B Q4 works but is much slower than needed for this use case.
- `--parallel 1` is the preferred single-user setting.

## Failed Or Lower-Performing Configurations

- `e2b-q2-32k` failed: `metal_unsupported_quant_type_35`. llama.cpp Metal backend does not support this Q2 quant type on this build.
- `e4b-q2-32k` failed: `metal_unsupported_quant_type_35`. llama.cpp Metal backend does not support this Q2 quant type on this build.

## Next Hypothesis

No unrecorded variants remain. Validate E2B Q4 and E4B Q4 on representative chat and classification prompts before choosing the default local model.
