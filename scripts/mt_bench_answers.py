#!/usr/bin/env python3
"""Generate resumable MT-Bench answers from a local OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from mt_bench_judge import is_judgeable_answer
from openai import APIError, OpenAI

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = ROOT / "benchmarks" / "mt-bench" / "question.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "mt-bench"
DEFAULT_SERVER_URL = "http://127.0.0.1:8080/v1"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 32768
DEFAULT_TIMEOUT_SECONDS = 21600.0


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions = []
    seen_ids: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                question = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid question JSON at line {line_number}: {error}"
                ) from error
            question_id = question.get("question_id")
            turns = question.get("turns")
            if not isinstance(question_id, int) or question_id in seen_ids:
                raise ValueError(
                    f"invalid or duplicate question_id at line {line_number}"
                )
            if (
                not isinstance(turns, list)
                or len(turns) != 2
                or not all(isinstance(turn, str) for turn in turns)
            ):
                raise ValueError(
                    f"question {question_id} must contain exactly two text turns"
                )
            seen_ids.add(question_id)
            questions.append(question)
    if not questions:
        raise ValueError(f"question file is empty: {path}")
    return questions


def load_existing_answers(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}

    answers: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                answer = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid answer JSON at line {line_number}: {error}"
                ) from error
            question_id = answer.get("question_id")
            choices = answer.get("choices")
            if (
                not isinstance(question_id, int)
                or not isinstance(choices, list)
                or not choices
            ):
                raise ValueError(f"invalid answer record at line {line_number}")
            turns = choices[0].get("turns") if isinstance(choices[0], dict) else None
            if (
                not isinstance(turns, list)
                or len(turns) != 2
                or not all(isinstance(turn, str) for turn in turns)
            ):
                continue
            if is_judgeable_answer(answer):
                answers[question_id] = answer
    return answers


def dump_answer(handle, answer: dict[str, Any]) -> None:
    handle.write(json.dumps(answer, ensure_ascii=False) + "\n")
    handle.flush()


def usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    if isinstance(usage, dict):
        return usage
    return {
        key: getattr(usage, key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if hasattr(usage, key)
    }


def generate_question(
    client: OpenAI,
    question: dict[str, Any],
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = []
    turns: list[str] = []
    usage: list[dict[str, Any]] = []
    finish_reasons: list[str | None] = []

    for prompt in question["turns"]:
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not response.choices:
            raise RuntimeError(
                f"model returned no choices for question {question['question_id']}"
            )
        choice = response.choices[0]
        content = choice.message.content
        if not isinstance(content, str):
            raise TypeError(
                f"model returned non-text content for question {question['question_id']}"
            )
        turns.append(content.strip())
        messages.append({"role": "assistant", "content": content})
        usage.append(usage_dict(response.usage))
        finish_reasons.append(choice.finish_reason)

    return {
        "question_id": question["question_id"],
        "answer_id": str(uuid.uuid4()),
        "model_id": model_name,
        "choices": [{"index": 0, "turns": turns}],
        "tstamp": time.time(),
        "generation": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "usage": usage,
            "finish_reasons": finish_reasons,
        },
    }


def rewrite_sorted(path: Path, answers: dict[int, dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for question_id in sorted(answers):
            handle.write(json.dumps(answers[question_id], ensure_ascii=False) + "\n")
    temporary.replace(path)


def run(
    questions_path: Path,
    output_path: Path,
    server_url: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> int:
    questions = load_questions(questions_path)
    answers = load_existing_answers(output_path)
    question_ids = {question["question_id"] for question in questions}
    unexpected_ids = set(answers) - question_ids
    if unexpected_ids:
        raise ValueError(
            "answer file contains question IDs not present in question file: "
            + ", ".join(str(question_id) for question_id in sorted(unexpected_ids))
        )

    pending = [
        question for question in questions if question["question_id"] not in answers
    ]
    if not pending:
        print(f"all {len(questions)} questions already completed")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(
        base_url=server_url.rstrip("/"),
        api_key="local",
        timeout=timeout,
        max_retries=0,
    )
    with output_path.open("a", encoding="utf-8") as handle:
        for question in pending:
            question_id = question["question_id"]
            answer = generate_question(
                client,
                question,
                model_name,
                temperature,
                max_tokens,
            )
            answers[question_id] = answer
            dump_answer(handle, answer)
            print(f"completed question {question_id}", flush=True)

    rewrite_sorted(output_path, answers)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id", required=True, help="Model identifier stored in answer records"
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()
    if args.output is None:
        args.output = DEFAULT_OUTPUT_DIR / f"{args.model_id}.jsonl"
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main() -> int:
    try:
        args = parse_args()
        return run(
            questions_path=args.questions,
            output_path=args.output,
            server_url=args.server_url,
            model_name=args.model_id,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    except (APIError, OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
