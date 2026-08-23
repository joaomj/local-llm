#!/usr/bin/env python3
"""Build deterministic multi-turn research conversations for benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

SCENARIO_VERSION = "web_research_tools-v1"
RESEARCH_WORDS = [
    "source",
    "evidence",
    "report",
    "analysis",
    "system",
    "performance",
    "context",
    "memory",
    "throughput",
    "latency",
    "benchmark",
    "configuration",
    "observation",
    "comparison",
    "result",
    "limitation",
    "method",
    "hardware",
    "software",
    "request",
    "response",
    "generation",
    "prompt",
    "cache",
    "server",
    "conversation",
    "research",
    "document",
    "finding",
    "reference",
    "measurement",
    "workload",
    "reliability",
    "reproducible",
]


def _research_text(word_count: int) -> str:
    words: list[str] = []
    for index in range(word_count):
        word = RESEARCH_WORDS[index % len(RESEARCH_WORDS)]
        if index % 97 == 0:
            word = f"source-{index // 97}-{word}"
        words.append(word)
    return " ".join(words)


def _content_token_count(
    messages: list[dict[str, Any]], tokenize: Callable[[str], int]
) -> int:
    return tokenize(
        "\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("content")
        )
    )


def build_tool_conversation(
    target_content_tokens: int,
    tokenize: Callable[[str], int],
) -> tuple[list[dict[str, Any]], int]:
    """Return a valid tool transcript near the requested content-token count."""
    if target_content_tokens < 512:
        raise ValueError("target_content_tokens must be at least 512")

    word_count = target_content_tokens
    messages: list[dict[str, Any]] = []
    actual = 0
    for _ in range(6):
        research = _research_text(word_count)
        split_at = len(research) // 2
        first = research[:split_at]
        second = research[split_at:]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research assistant. Use supplied web-search evidence, track "
                    "conflicts, and give a concise final synthesis."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Research how long-context local language-model performance changes with "
                    "real multi-turn tool use."
                ),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "search-1",
                        "type": "function",
                        "function": {
                            "name": "websearch",
                            "arguments": '{"query":"local LLM long context performance"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "search-1",
                "name": "websearch",
                "content": first,
            },
            {
                "role": "assistant",
                "content": "The first sources show several trade-offs. I will search for corroboration.",
            },
            {
                "role": "user",
                "content": "Check additional sources and preserve the detailed evidence for comparison.",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "search-2",
                        "type": "function",
                        "function": {
                            "name": "websearch",
                            "arguments": '{"query":"llama.cpp context throughput Metal"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "search-2",
                "name": "websearch",
                "content": second,
            },
            {
                "role": "assistant",
                "content": "I have compared the two result sets and retained their measurements.",
            },
            {
                "role": "user",
                "content": (
                    "Synthesize the evidence. State the best configuration, measured limits, "
                    "and important caveats."
                ),
            },
        ]
        actual = _content_token_count(messages, tokenize)
        tolerance = max(64, target_content_tokens // 100)
        if abs(actual - target_content_tokens) <= tolerance:
            break
        word_count = max(1, round(word_count * target_content_tokens / max(actual, 1)))
    return messages, actual
