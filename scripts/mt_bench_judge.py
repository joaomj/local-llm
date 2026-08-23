#!/usr/bin/env python3
"""Judge local MT-Bench answers through OpenRouter with structured output."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = ROOT / "benchmarks" / "mt-bench" / "question.jsonl"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_JUDGE_MODEL = "google/gemini-3.7-flash"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "mt-bench"
JUDGE_SYSTEM_PROMPT = """You are an impartial benchmark judge.

Evaluate the assistant's response to the SECOND user request. Use the first
request and first answer only as conversation context. Consider instruction
following, correctness, relevance, helpfulness, depth, clarity, and the
requested format. Do not reward length by itself. Do not penalize a safe
refusal when the request is unsafe or impossible, but judge whether the
refusal is useful and appropriate.

Return only the requested JSON object. Give an integer score from 1 (very
poor) to 10 (excellent), and a concise rationale that identifies the main
strengths and weaknesses."""
JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 10},
        "rationale": {"type": "string", "minLength": 1},
    },
    "required": ["score", "rationale"],
    "additionalProperties": False,
}


def read_dotenv_value(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            name, separator, value = line.partition("=")
            if separator and name.strip() == key:
                return value.strip().strip('"').strip("'")
    return ""


def load_api_key(dotenv_path: Path) -> str:
    return os.environ.get("OPENROUTER_API_KEY", "") or read_dotenv_value(
        dotenv_path, "OPENROUTER_API_KEY"
    )


def load_questions(path: Path) -> dict[int, dict[str, Any]]:
    questions: dict[int, dict[str, Any]] = {}
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
            if not isinstance(question_id, int) or question_id in questions:
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
            questions[question_id] = question
    if not questions:
        raise ValueError(f"question file is empty: {path}")
    return questions


def is_judgeable_answer(answer: dict[str, Any]) -> bool:
    choices = answer.get("choices")
    turns = choices[0].get("turns") if isinstance(choices, list) and choices else None
    if not isinstance(turns, list) or len(turns) != 2:
        return False
    if not all(isinstance(turn, str) and turn.strip() for turn in turns):
        return False
    generation = answer.get("generation")
    finish_reasons = (
        generation.get("finish_reasons") if isinstance(generation, dict) else None
    )
    return not isinstance(finish_reasons, list) or (
        len(finish_reasons) == 2 and all(reason == "stop" for reason in finish_reasons)
    )


def load_answers(path: Path) -> dict[int, dict[str, Any]]:
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
            if not is_judgeable_answer(answer):
                raise ValueError(
                    f"answer {question_id} is incomplete or truncated and cannot be judged"
                )
            answers[question_id] = answer
    return answers


def load_existing_judgments(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    judgments: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                judgment = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid judgment JSON at line {line_number}: {error}"
                ) from error
            question_id = judgment.get("question_id")
            score = judgment.get("score")
            if (
                not isinstance(question_id, int)
                or isinstance(score, bool)
                or not isinstance(score, int)
                or not 1 <= score <= 10
            ):
                raise ValueError(f"invalid judgment record at line {line_number}")
            judgments[question_id] = judgment
    return judgments


def answer_prompt(question: dict[str, Any], answer: dict[str, Any]) -> str:
    turns = answer["choices"][0]["turns"]
    return (
        "<|The Start of Assistant Conversation|>\n\n"
        f"### User:\n{question['turns'][0]}\n\n"
        f"### Assistant:\n{turns[0]}\n\n"
        f"### User:\n{question['turns'][1]}\n\n"
        f"### Assistant:\n{turns[1]}\n\n"
        "<|The End of Assistant Conversation|>"
    )


def validate_judgment(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"score", "rationale"}:
        raise ValueError("judge response must contain only score and rationale")
    score = value["score"]
    rationale = value["rationale"]
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 10:
        raise ValueError("judge score must be an integer from 1 to 10")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("judge rationale must be non-empty text")
    return {"score": score, "rationale": rationale.strip()}


def response_dump(response: Any) -> dict[str, Any]:
    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        return model_dump(exclude_none=True)
    if isinstance(response, dict):
        return response
    raise ValueError("judge response cannot be serialized")


def response_cost(response_data: dict[str, Any]) -> float | None:
    usage = response_data.get("usage")
    if not isinstance(usage, dict):
        return None
    cost = usage.get("cost")
    return (
        float(cost)
        if isinstance(cost, (int, float)) and not isinstance(cost, bool)
        else None
    )


def response_provider(response_data: dict[str, Any]) -> str | None:
    provider = response_data.get("provider")
    if isinstance(provider, str):
        return provider
    choices = response_data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        provider = choices[0].get("provider")
        if isinstance(provider, str):
            return provider
    return None


def _as_positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def error_details(error: Exception) -> dict[str, Any]:
    response = getattr(error, "response", None)
    status_code = getattr(error, "status_code", None)
    headers: dict[str, str] = {}
    response_headers = getattr(response, "headers", None)
    if response_headers is not None:
        for key in (
            "retry-after",
            "x-ratelimit-limit",
            "x-ratelimit-remaining",
            "x-ratelimit-reset",
        ):
            value = response_headers.get(key)
            if value is not None:
                headers[key] = str(value)

    body = getattr(error, "body", None)
    if not isinstance(body, dict) and response is not None:
        try:
            body = response.json()
        except (AttributeError, ValueError):
            body = None

    retry_after = _as_positive_float(headers.get("retry-after"))
    if retry_after is None and isinstance(body, dict):
        error_body = body.get("error")
        if isinstance(error_body, dict):
            availability = error_body.get("availability")
            if isinstance(availability, dict):
                retry_after = _as_positive_float(availability.get("retry_after"))
        if retry_after is None:
            availability = body.get("availability")
            if isinstance(availability, dict):
                retry_after = _as_positive_float(availability.get("retry_after"))
    if retry_after is None:
        reset = _as_positive_float(headers.get("x-ratelimit-reset"))
        if reset is not None:
            retry_after = (
                max(0.0, reset - time.time()) if reset > time.time() else reset
            )

    safe_body = None
    if isinstance(body, dict):
        safe_body = {"error": body.get("error")} if "error" in body else body
    return {
        "type": type(error).__name__,
        "status_code": status_code,
        "message": str(error),
        "retry_after_seconds": retry_after,
        "rate_limit_headers": headers,
        "body": safe_body,
    }


def retry_delay(error: Exception, attempt: int, base_seconds: float) -> float:
    details = error_details(error)
    hinted = details.get("retry_after_seconds")
    if isinstance(hinted, (int, float)):
        return float(hinted)
    return base_seconds * (2**attempt)


def is_retryable(error: Exception) -> bool:
    if isinstance(error, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(error, APIError):
        status_code = getattr(error, "status_code", None)
        return (
            status_code in {408, 409, 429}
            or isinstance(status_code, int)
            and status_code >= 500
        )
    return isinstance(error, ValueError)


def judge_answer(
    client: OpenAI,
    question: dict[str, Any],
    answer: dict[str, Any],
    judge_model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_base_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": answer_prompt(question, answer)},
    ]
    request_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "mt_bench_judgment",
            "strict": True,
            "schema": JUDGMENT_SCHEMA,
        },
    }

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=request_format,
            )
            if not response.choices:
                raise ValueError("judge returned no choices")
            content = response.choices[0].message.content
            if not isinstance(content, str):
                raise TypeError("judge returned non-text content")
            judgment = validate_judgment(json.loads(content))
            raw_response = response_dump(response)
            return judgment, raw_response, attempt + 1
        except (
            APIConnectionError,
            APIError,
            APITimeoutError,
            TypeError,
            ValueError,
        ) as error:
            last_error = error
            if attempt >= max_retries or not is_retryable(error):
                raise RuntimeError(
                    f"judge request failed after {attempt + 1} attempt(s): {error}"
                ) from error
            time.sleep(retry_delay(error, attempt, retry_base_seconds))
    raise RuntimeError(f"judge request failed: {last_error}")


def run(
    questions_path: Path,
    answers_path: Path,
    output_path: Path,
    base_url: str,
    api_key: str,
    judge_model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_base_seconds: float,
    max_requests: int,
    max_cost_usd: float | None,
    timeout: float,
    request_delay_seconds: float,
) -> int:
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")
    questions = load_questions(questions_path)
    answers = load_answers(answers_path)
    if set(answers) - set(questions):
        unexpected = sorted(set(answers) - set(questions))
        raise ValueError(
            "answer file contains unknown question IDs: "
            + ", ".join(map(str, unexpected))
        )
    judgments = load_existing_judgments(output_path)
    if set(judgments) - set(answers):
        unexpected = sorted(set(judgments) - set(answers))
        raise ValueError(
            "judgment file contains unknown question IDs: "
            + ", ".join(map(str, unexpected))
        )

    pending = [
        question_id for question_id in sorted(answers) if question_id not in judgments
    ]
    if not pending:
        print(f"all {len(answers)} answers already judged")
        return 0
    if len(pending) > max_requests:
        pending = pending[:max_requests]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(
        base_url=base_url.rstrip("/"), api_key=api_key, timeout=timeout, max_retries=0
    )
    error_log_path = output_path.with_name(f"{output_path.stem}.errors.jsonl")
    total_cost = sum(
        float(judgment["cost_usd"])
        for judgment in judgments.values()
        if isinstance(judgment.get("cost_usd"), (int, float))
    )
    with output_path.open("a", encoding="utf-8") as handle:
        for position, question_id in enumerate(pending):
            if position and request_delay_seconds:
                time.sleep(request_delay_seconds)
            try:
                judgment, raw_response, attempts = judge_answer(
                    client,
                    questions[question_id],
                    answers[question_id],
                    judge_model,
                    temperature,
                    max_tokens,
                    max_retries,
                    retry_base_seconds,
                )
            except RuntimeError as error:
                error_record = {
                    "question_id": question_id,
                    "candidate_model_id": answers[question_id].get("model_id"),
                    "judge_model": judge_model,
                    "tstamp": time.time(),
                    **error_details(error.__cause__ or error),
                }
                with error_log_path.open("a", encoding="utf-8") as error_handle:
                    error_handle.write(
                        json.dumps(error_record, ensure_ascii=False) + "\n"
                    )
                raise
            cost = response_cost(raw_response)
            if max_cost_usd is not None and cost is None:
                raise RuntimeError(
                    "judge response did not include usage.cost; cannot enforce --max-cost-usd"
                )
            if (
                cost is not None
                and max_cost_usd is not None
                and total_cost + cost > max_cost_usd
            ):
                raise RuntimeError(
                    f"cost ceiling exceeded before question {question_id}: "
                    f"{total_cost + cost:.6f} > {max_cost_usd:.6f} USD"
                )
            total_cost += cost or 0.0
            record = {
                "question_id": question_id,
                "answer_id": answers[question_id].get("answer_id"),
                "candidate_model_id": answers[question_id].get("model_id"),
                "judge_model": judge_model,
                "score": judgment["score"],
                "rationale": judgment["rationale"],
                "provider": response_provider(raw_response),
                "cost_usd": cost,
                "usage": raw_response.get("usage", {}),
                "attempts": attempts,
                "tstamp": time.time(),
                "raw_response": raw_response,
            }
            judgments[question_id] = record
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"judged question {question_id}: {judgment['score']}/10", flush=True)

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for question_id in sorted(judgments):
            handle.write(json.dumps(judgments[question_id], ensure_ascii=False) + "\n")
    temporary.replace(output_path)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Check credential configuration without judging",
    )
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--base-url", default=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
    )
    parser.add_argument("--dotenv", type=Path, default=ROOT / ".env")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-seconds", type=float, default=5.0)
    parser.add_argument("--request-delay-seconds", type=float, default=2.0)
    parser.add_argument("--max-requests", type=int, default=80)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if not args.check_config and args.answers is None:
        parser.error("--answers is required unless --check-config is used")
    if args.output is None:
        args.output = (
            DEFAULT_OUTPUT_DIR / f"{args.answers.stem}.judgments.jsonl"
            if args.answers is not None
            else None
        )
    if args.temperature < 0 or args.max_tokens <= 0 or args.max_retries < 0:
        parser.error(
            "temperature must be non-negative; max-tokens must be positive; max-retries cannot be negative"
        )
    if (
        args.retry_base_seconds < 0
        or args.request_delay_seconds < 0
        or args.max_requests <= 0
        or args.timeout <= 0
    ):
        parser.error(
            "retry-base-seconds and request-delay-seconds cannot be negative; max-requests and timeout must be positive"
        )
    if args.max_cost_usd is not None and args.max_cost_usd < 0:
        parser.error("max-cost-usd cannot be negative")
    return args


def main() -> int:
    try:
        args = parse_args()
        api_key = load_api_key(args.dotenv)
        if args.check_config:
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY is not set")
            print("OpenRouter credential configuration is present")
            return 0
        return run(
            questions_path=args.questions,
            answers_path=args.answers,
            output_path=args.output,
            base_url=args.base_url,
            api_key=api_key,
            judge_model=args.judge_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            retry_base_seconds=args.retry_base_seconds,
            max_requests=args.max_requests,
            max_cost_usd=args.max_cost_usd,
            timeout=args.timeout,
            request_delay_seconds=args.request_delay_seconds,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
