"""Terminal chat client for a local MLX model server with Exa web search.

Usage:
    EXA_API_KEY=<key> uv run chat.py

Commands:
    /help     show commands
    /reset    clear conversation history
    /exit     quit
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx
from openai import OpenAI

MODEL = os.environ.get("LLM_MODEL", "LiquidAI/LFM2.5-8B-A1B-MLX-8bit")
SERVER_URL = os.environ.get("LLM_SERVER_URL", "http://127.0.0.1:8080/v1")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4096"))
MAX_SEARCH_ITERATIONS = 4

DIM = "\033[2m"
CYAN = "\033[36m"
RESET = "\033[0m"


def load_exa_key() -> str:
    env_key = os.environ.get("EXA_API_KEY", "")
    if env_key:
        return env_key
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("EXA_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


EXA_API_KEY = load_exa_key()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. Use this when the user asks about "
                "recent events, facts you are unsure about, prices, news, or anything that "
                "may have changed. Returns titles, URLs, and highlighted passages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A concise, specific search query.",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def exa_search(query: str) -> str:
    if not EXA_API_KEY:
        return "ERROR: EXA_API_KEY not set. Set it in the environment or in a .env file."
    payload = {
        "query": query,
        "numResults": 5,
        "type": "auto",
        "contents": {"highlights": True},
    }
    try:
        resp = httpx.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": EXA_API_KEY, "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        return f"ERROR: Exa search failed: {e}"
    data = resp.json()
    results = data.get("results", [])
    if not results:
        return "No results found."
    lines = []
    for r in results:
        lines.append(f"- {r.get('title', 'Untitled')}")
        lines.append(f"  URL: {r.get('url', '')}")
        highlights = r.get("highlights") or []
        if highlights:
            lines.append(f"  {highlights[0][:900]}")
    return "\n".join(lines)


class Metrics:
    def __init__(self) -> None:
        self.calls = []

    def record_call(self, label: str, prompt_tokens: int, completion_tokens: int,
                    ttft_s: float, gen_s: float) -> None:
        gen_s = max(gen_s, 1e-6)
        self.calls.append(
            {
                "label": label,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "ttft_s": round(ttft_s, 3),
                "ttft_ms_per_prompt_token": round(ttft_s * 1000 / max(prompt_tokens, 1), 3),
                "gen_tokens_per_s": round(completion_tokens / gen_s, 2),
            }
        )
        print(
            f"{DIM}metrics [{label}]: ttft={ttft_s:.3f}s | "
            f"gen={completion_tokens / gen_s:.1f} tok/s | "
            f"prompt={prompt_tokens} tok (eval est {prompt_tokens / max(ttft_s, 1e-6):.0f} tok/s){RESET}"
        )

    def summary(self) -> None:
        if not self.calls:
            return
        gen_calls = [c for c in self.calls if c["completion_tokens"] > 0]
        if gen_calls:
            avg_gen = sum(c["gen_tokens_per_s"] for c in gen_calls) / len(gen_calls)
            avg_ttft = sum(c["ttft_s"] for c in gen_calls) / len(gen_calls)
            total_tok = sum(c["completion_tokens"] for c in gen_calls)
            total_s = max(sum(c["ttft_s"] + (c["completion_tokens"] / max(c["gen_tokens_per_s"], 1e-6)) for c in gen_calls), 1e-6)
            print(
                f"{CYAN}session avg: ttft={avg_ttft:.3f}s | "
                f"gen={avg_gen:.1f} tok/s | {total_tok} output tokens this turn{RESET}"
            )


def extra_field(delta, name: str):
    value = getattr(delta, name, None)
    if value is not None:
        return value
    return getattr(delta, "model_extra", {}).get(name)


def stream_turn(client: OpenAI, messages: list, metrics: Metrics) -> None:
    label = "search" if any(m.get("role") == "tool" for m in messages) else "answer"
    t0 = time.monotonic()
    t_first = None
    started_content = False
    first_content_ts = None
    tool_calls = {}
    prompt_tokens = completion_tokens = 0
    print(DIM, end="", flush=True)
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            max_tokens=MAX_TOKENS,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = extra_field(delta, "reasoning")
            if reasoning:
                if t_first is None:
                    t_first = time.monotonic()
                sys.stdout.write(delta.reasoning)
                sys.stdout.flush()
            if delta.content:
                if not started_content:
                    started_content = True
                    first_content_ts = time.monotonic()
                    print(f"{RESET}{CYAN}answer:{RESET}", end="", flush=True)
                sys.stdout.write(delta.content)
                sys.stdout.flush()
            for tc in delta.tool_calls or []:
                entry = tool_calls.setdefault(
                    tc.index or 0,
                    {"id": tc.id or "", "name": "", "arguments": ""},
                )
                if tc.id:
                    entry["id"] = tc.id
                if tc.function and tc.function.name:
                    entry["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    entry["arguments"] += tc.function.arguments
    finally:
        print(RESET, end="", flush=True)
    t_end = time.monotonic()

    if tool_calls:
        for idx, call in sorted(tool_calls.items()):
            print(
                f"\n{CYAN}-> web_search{idx}: {call['name']}({call['arguments']}){RESET}",
                flush=True,
            )
        metrics.record_call(
            label,
            prompt_tokens,
            completion_tokens,
            (t_first - t0) if t_first else (t_end - t0),
            (first_content_ts - t_first) if started_content and t_first else (t_end - t0),
        )
        return list(tool_calls.items())

    metrics.record_call(
        label,
        prompt_tokens,
        completion_tokens,
        (t_first - t0) if t_first else (t_end - t0),
        (t_end - (t_first or t0)),
    )
    return None


def main() -> None:
    if not EXA_API_KEY:
        print("EXA_API_KEY not found. Search will report errors; export the key in a .env file.")
    client = OpenAI(base_url=SERVER_URL, api_key="local")
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with web search. Use the web_search tool for "
                "anything that needs current, factual, or external information, including "
                "events after your training, prices, and news. Cite sources by URL when "
                "you used search results. Keep answers concise. Think before you search."
            ),
        }
    ]
    print("Local LLM chat ready. Type a message, /help, /reset, or /exit.")
    while True:
        try:
            user_input = input("\033[1mYou:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            break
        if user_input == "/reset":
            messages = messages[:1]
            print("History cleared.")
            continue
        if user_input == "/help":
            print("Commands: /reset (clear history), /exit (quit). Search is always on.")
            continue

        messages.append({"role": "user", "content": user_input})
        metrics = Metrics()
        iteration = 0
        while iteration < MAX_SEARCH_ITERATIONS:
            result = stream_turn(client, messages, metrics)
            if result is None:
                break
            assistant_tool_msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": call["arguments"]},
                    }
                    for _, call in result
                ],
            }
            messages.append(assistant_tool_msg)
            for _, call in result:
                args = json.loads(call["arguments"] or "{}")
                query = args.get("query", "")
                print(f"\033[2msearching: {query}...\033[0m", flush=True)
                search_result = exa_search(query)
                messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": search_result}
                )
            iteration += 1
        else:
            print("\033[31mStopped after too many search rounds.\033[0m")
        metrics.summary()
        print()


if __name__ == "__main__":
    main()
