# Use the local MLX model as an opencode provider

This document explains how to use the local mlx-lm inference endpoint as a model provider in opencode.

## Prerequisites

Start the MLX server before you start opencode:

```bash
mlx_lm.server --model LiquidAI/LFM2.5-8B-A1B-MLX-8bit \
  --host 127.0.0.1 --port 8080 \
  --max-tokens 4096
```

The `--max-tokens` flag matters. The server default is 512 tokens. Without the flag, opencode responses stop after 512 tokens. The model is a reasoning model, so thinking tokens reduce the room for the final answer.

## 1. Add the provider to your config

Add this block to the `provider` section of your opencode config.

Global config (all projects):

```jsonc
~/.config/opencode/opencode.jsonc
```

Project config (this project only):

```jsonc
opencode.jsonc
```

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "mlx": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "MLX (local)",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1"
      },
      "models": {
        "LiquidAI/LFM2.5-8B-A1B-MLX-8bit": {
          "name": "LFM2.5-8B-A1B (local)",
          "limit": {
            "context": 128000,
            "output": 4096
          }
        }
      }
    }
  }
}
```

Field notes:

- `mlx` is the provider ID. It can be any string.
- `npm` must be `@ai-sdk/openai-compatible`. The mlx server speaks the OpenAI API.
- `options.baseURL` must point at the mlx server. Keep the `/v1` suffix.
- The model key must match the `id` returned by `GET /v1/models`. Check with:

  ```bash
  curl -s http://127.0.0.1:8080/v1/models
  ```

- No API key is needed. The mlx server does not require one.

## 2. Restart opencode

opencode loads its config once at startup. Quit opencode and start it again after you save the config.

## 3. Select the model

Run `/models` inside opencode and choose `MLX (local)`.

The full model ID is `mlx/LiquidAI/LFM2.5-8B-A1B-MLX-8bit`.

To make it the default model, add to your config:

```jsonc
"model": "mlx/LiquidAI/LFM2.5-8B-A1B-MLX-8bit"
```

## Behavior notes

- The model is a reasoning model. Its thinking output appears in opencode as reasoning content.
- Tool calling works. opencode sends its tools to the server, and the model calls them.
- Generation speed is 44-47 tokens per second on this hardware. Large tasks run slowly.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Connection refused` or empty model list | Server not running | Start `mlx_lm.server` before opencode |
| Response stops mid-sentence | Server token cap | Start the server with `--max-tokens 4096` |
| Model missing from `/models` | Model key mismatch | Match the model key to the `id` from `/v1/models` |
| Config changes have no effect | opencode did not restart | Quit opencode and start it again |
