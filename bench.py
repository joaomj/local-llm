"""Repeated live-chat trials with median/p95 metrics.

Usage:
    EXA_API_KEY=<key> uv run bench.py [--trials N]

Scenarios:
  plain   fresh conversation, plain reasoning question (3 rotating prompts)
  search  fresh conversation, web-search question answered with Exa results
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import httpx
from openai import OpenAI

import chat

TRIALS = 10

PLAIN_PROMPTS = [
    "Explain the difference between TCP and UDP in three sentences.",
    "Explain how the HTTPS handshake works in three sentences.",
    "Explain the difference between RAM and ROM in three sentences.",
]

SEARCH_PROMPT = "Who won the last men's Wimbledon final and what was the score? Use web search."


def collect_call(client: OpenAI, messages: list, label: str, scenario: str) -> dict:
    t0 = time.monotonic()
    t_first = None
    t_first_content = None
    tool_calls = {}
    usage = {}

    stream = client.chat.completions.create(
        model=chat.MODEL,
        messages=messages,
        tools=chat.TOOLS,
        max_tokens=chat.MAX_TOKENS,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage.model_dump()
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if chat.extra_field(delta, "reasoning"):
            if t_first is None:
                t_first = time.monotonic()
        if delta.content and t_first_content is None:
            t_first_content = time.monotonic()
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
    t_end = time.monotonic()

    prompt_tokens = usage.get("prompt_tokens", 0) or 0
    completion_tokens = usage.get("completion_tokens", 0) or 0
    cached_tokens = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    ttft = (t_first - t0) if t_first else (t_end - t0)
    gen_s = t_end - (t_first or t0)
    gen_s = max(gen_s, 1e-6)

    return {
        "scenario": scenario,
        "label": label,
        "ttft_s": ttft,
        "gen_s": gen_s,
        "gen_tps": completion_tokens / gen_s,
        "eval_tps": prompt_tokens / max(ttft, 1e-6),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "tool_calls": list(tool_calls.items()),
    }


def run_turn(client: OpenAI, prompt: str, scenario: str, trial_id: int) -> list[dict]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with web search. Use the web_search tool for "
                "anything that needs current, factual, or external information. Keep answers "
                "concise. Think before you search."
            ),
        }
    ]
    messages.append({"role": "user", "content": prompt})
    calls = []
    for iteration in range(chat.MAX_SEARCH_ITERATIONS):
        call = collect_call(
            client,
            messages,
            "first-call" if iteration == 0 else "answer",
            scenario,
        )
        call["trial_id"] = trial_id
        calls.append(call)
        if not call["tool_calls"]:
            break
        assistant_tool_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["arguments"]},
                }
                for _, c in call["tool_calls"]
            ],
        }
        messages.append(assistant_tool_msg)
        for _, c in call["tool_calls"]:
            args = json.loads(c["arguments"] or "{}")
            result = chat.exa_search(args.get("query", ""))
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
    return calls


def summarize(rows: list[dict], scenario: str, label: str) -> dict:
    def q(p, key):
        vals = sorted(r[key] for r in rows)
        if not vals:
            return None
        idx = max(0, min(len(vals) - 1, int(p / 100 * len(vals))))
        return vals[idx]

    return {
        "scenario": scenario,
        "label": label,
        "n": len(rows),
        "ttft_s_median": round(statistics.median(r["ttft_s"] for r in rows), 3) if rows else None,
        "ttft_s_p95": round(q(95, "ttft_s"), 3) if rows else None,
        "gen_tps_median": round(statistics.median(r["gen_tps"] for r in rows), 2) if rows else None,
        "gen_tps_p95": round(q(95, "gen_tps"), 2) if rows else None,
        "eval_tps_median": round(statistics.median(r["eval_tps"] for r in rows), 1) if rows else None,
        "prompt_tokens_median": round(statistics.median(r["prompt_tokens"] for r in rows)) if rows else None,
        "completion_tokens_median": round(statistics.median(r["completion_tokens"] for r in rows)) if rows else None,
        "cached_tokens_median": round(statistics.median(r["cached_tokens"] for r in rows)) if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=TRIALS)
    args = parser.parse_args()

    client = OpenAI(base_url=chat.SERVER_URL, api_key="local")
    all_rows = []
    scenarios = []

    for i in range(args.trials):
        prompt = PLAIN_PROMPTS[i % len(PLAIN_PROMPTS)]
        all_rows.extend(run_turn(client, prompt, "plain", i))
        print(f"trial {i + 1}/{args.trials}: plain done", file=sys.stderr, flush=True)

    for i in range(args.trials):
        all_rows.extend(run_turn(client, SEARCH_PROMPT, "search", i))
        print(f"trial {i + 1}/{args.trials}: search done", file=sys.stderr, flush=True)

    report = []
    for scenario in ("plain", "search"):
        for label in ("first-call", "answer"):
            rows = [r for r in all_rows if r["scenario"] == scenario and r["label"] == label]
            if rows:
                report.append(summarize(rows, scenario, label))
    for scenario in ("plain", "search"):
        turns = []
        for trial_id in range(args.trials):
            rows = [r for r in all_rows if r["scenario"] == scenario and r["trial_id"] == trial_id]
            if not rows:
                continue
            turns.append(
                {
                    "ttft": rows[0]["ttft_s"],
                    "search_rounds": len(rows) - 1,
                    "gen_tps": sum(r["completion_tokens"] for r in rows)
                    / max(sum(r["gen_s"] for r in rows), 1e-6),
                }
            )
        if turns:
            ttft_sorted = sorted(t["ttft"] for t in turns)
            gen_sorted = sorted(t["gen_tps"] for t in turns)
            def p95(vals):
                return vals[int(0.95 * len(vals)) - 1]
            report.append(
                {
                    "scenario": scenario,
                    "label": "whole-turn",
                    "n": len(turns),
                    "ttft_s_median": round(statistics.median(t["ttft"] for t in turns), 3),
                    "ttft_s_p95": round(p95(ttft_sorted), 3),
                    "gen_tps_median": round(statistics.median(t["gen_tps"] for t in turns), 2),
                    "gen_tps_p95": round(p95(gen_sorted), 2),
                    "search_rounds_median": round(statistics.median(t["search_rounds"] for t in turns), 1),
                }
            )

    print(json.dumps(report, indent=2))
    with open("bench_raw.json", "w") as f:
        json.dump(all_rows, f, indent=2)


if __name__ == "__main__":
    main()
