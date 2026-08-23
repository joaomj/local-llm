# Use A Local Server In OpenCode

This guide connects OpenCode to a local OpenAI-compatible server. The current
benchmark workflow uses `llama-server` with retained GGUF weights.

## Start The Server

Set the model path and model ID for the retained model that you want to use:

```bash
export LOCAL_MODEL_PATH="$PWD/.benchmark/throughput-models/gemma-4-E2B-it-qat-UD-Q2_K_XL/gemma-4-E2B-it-qat-UD-Q2_K_XL.gguf"
export LOCAL_MODEL_ID="gemma4-e2b-it-qat-ud-q2-k-xl"
```

Start the server:

```bash
llama-server \
  --model "$LOCAL_MODEL_PATH" \
  --alias "$LOCAL_MODEL_ID" \
  --ctx-size 131072 \
  --flash-attn on \
  --parallel 1 \
  --host 127.0.0.1 \
  --port 8080
```

Check the model ID:

```bash
curl -s http://127.0.0.1:8080/v1/models
```

The alias in the response must match `LOCAL_MODEL_ID`.

## Configure OpenCode

Add this provider block to `~/.config/opencode/opencode.jsonc`:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local llama-server",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1"
      },
      "models": {
        "gemma4-e2b-it-qat-ud-q2-k-xl": {
          "name": "Gemma E2B QAT Q2",
          "limit": {
            "context": 131072,
            "output": 4096
          }
        }
      }
    }
  }
}
```

Replace the model key and name when you start a different retained model.
The key must match the server alias.

Restart OpenCode after a configuration change. Select the model through
`/models`.

## Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| Connection refused | The server is stopped | Start `llama-server` |
| Empty model list | The server is not ready | Check the server log and call `/v1/models` |
| Model not found | The alias differs from the config key | Use the exact alias from `/v1/models` |
| Response stops early | The output limit is too small | Increase the server or OpenCode output limit |

The selected Gemma model passes the throughput gate at 131K configured context
tokens. Its quality evaluation has 11 truncated or repeating answers. See the
current findings report before using it as the default model.
