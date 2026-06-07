# Optimization Memory

## Best Configuration

Current best remains the previously recorded configuration at 52.7943 tok/s.
Best speed candidate: `e2b-q4-32k`.

Best quality/speed compromise candidate: `e4b-q4-32k` if E2B quality is insufficient.

## Latest Run Patterns


## Failed Or Lower-Performing Configurations

- `e2b-q2-32k` failed: `metal_unsupported_quant_type_35`. llama.cpp Metal backend does not support this Q2 quant type on this build.
- `e4b-q2-32k` failed: `metal_unsupported_quant_type_35`. llama.cpp Metal backend does not support this Q2 quant type on this build.

## Next Hypothesis

No unrecorded variants remain. Validate E2B Q4 and E4B Q4 on representative chat and classification prompts before choosing the default local model.
