#!/usr/bin/env python3
"""Judge persisted MT-Bench answers through the OpenRouter Batch API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from mt_bench_judge import (
    DEFAULT_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_QUESTIONS,
    JUDGE_SYSTEM_PROMPT,
    JUDGMENT_SCHEMA,
    answer_prompt,
    load_answers,
    load_api_key,
    load_existing_judgments,
    load_questions,
    response_cost,
    response_provider,
    validate_judgment,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BATCH_MODEL = f"{DEFAULT_JUDGE_MODEL}:batch"
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
ACTIVE_STATUSES = {"validating", "in_progress", "finalizing"}


class BatchAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        headers: dict[str, str] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}


def batch_url(base_url: str, batch_id: str | None = None) -> str:
    parsed = urlsplit(base_url.rstrip("/"))
    path = parsed.path
    for suffix in ("/api/v1", "/v1"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    path = f"{path}/api/beta/batches".rstrip("/")
    if batch_id:
        path += f"/{batch_id}"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def parse_json_bytes(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"raw": raw.decode("utf-8", errors="replace")}


def request_json(
    url: str,
    api_key: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = parse_json_bytes(response.read())
            if not isinstance(body, dict):
                raise BatchAPIError(
                    "batch API returned a non-object response", response.status
                )
            return body
    except HTTPError as error:
        body = parse_json_bytes(error.read())
        message = body.get("error", body) if isinstance(body, dict) else body
        raise BatchAPIError(
            f"batch API returned HTTP {error.code}: {message}",
            error.code,
            {key.lower(): value for key, value in error.headers.items()},
        ) from error
    except (URLError, TimeoutError) as error:
        raise BatchAPIError(f"batch API transport failed: {error}") from error


def retry_delay(error: BatchAPIError, attempt: int, base_seconds: float) -> float:
    value = error.headers.get("retry-after")
    try:
        if value is not None:
            return max(0.0, float(value))
    except ValueError:
        pass
    return base_seconds * (2**attempt)


def api_call(
    url: str,
    api_key: str,
    method: str,
    payload: dict[str, Any] | None,
    timeout: float,
    max_retries: int,
    retry_base_seconds: float,
) -> dict[str, Any]:
    for attempt in range(max_retries + 1):
        try:
            return request_json(url, api_key, method, payload, timeout)
        except BatchAPIError as error:
            retryable = error.status_code is None or error.status_code in {408, 429}
            # Batch records can become visible shortly after a 202 submission.
            retryable = retryable or (method == "GET" and error.status_code == 404)
            retryable = retryable or (
                error.status_code is not None and error.status_code >= 500
            )
            if not retryable or attempt >= max_retries:
                raise
            delay = retry_delay(error, attempt, retry_base_seconds)
            print(
                f"batch API retry in {delay:g}s after attempt {attempt + 1}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError("batch API call did not return")


def custom_id(question_id: int) -> str:
    return f"question-{question_id}"


def question_id_from_custom_id(value: Any) -> int:
    if not isinstance(value, str) or not value.startswith("question-"):
        raise ValueError(f"invalid batch custom_id: {value!r}")
    try:
        return int(value[9:])
    except ValueError as error:
        raise ValueError(f"invalid batch custom_id: {value!r}") from error


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"batch state must be an object: {path}")
    return value


def batch_requests(
    question_ids: list[int],
    questions: dict[int, dict[str, Any]],
    answers: dict[int, dict[str, Any]],
    max_tokens: int,
) -> list[dict[str, Any]]:
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "mt_bench_judgment",
            "strict": True,
            "schema": JUDGMENT_SCHEMA,
        },
    }
    return [
        {
            "custom_id": custom_id(question_id),
            "body": {
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": answer_prompt(
                            questions[question_id], answers[question_id]
                        ),
                    },
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
                "response_format": response_format,
            },
        }
        for question_id in question_ids
    ]


def write_error(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def result_record(
    item: dict[str, Any],
    answers: dict[int, dict[str, Any]],
    questions: dict[int, dict[str, Any]],
    judge_model: str,
    batch_usage: dict[str, Any],
    batch_id: str,
) -> tuple[int, dict[str, Any] | None, str | None]:
    question_id = question_id_from_custom_id(item.get("custom_id"))
    if item.get("error") is not None:
        return question_id, None, f"batch item failed: {item['error']}"
    response = item.get("response")
    if not isinstance(response, dict) or response.get("status_code") != 200:
        return (
            question_id,
            None,
            f"batch item returned no successful response: {response}",
        )
    body = response.get("body")
    if not isinstance(body, dict):
        return question_id, None, "batch item response body is not an object"
    choices = body.get("choices")
    content = (
        choices[0].get("message", {}).get("content")
        if isinstance(choices, list) and choices
        else None
    )
    try:
        judgment = (
            validate_judgment(json.loads(content)) if isinstance(content, str) else None
        )
        if judgment is None:
            raise ValueError("batch item returned non-text content")
    except (json.JSONDecodeError, ValueError) as error:
        return question_id, None, f"invalid batch judgment: {error}"
    record = {
        "question_id": question_id,
        "answer_id": answers[question_id].get("answer_id"),
        "candidate_model_id": answers[question_id].get("model_id"),
        "judge_model": judge_model,
        "score": judgment["score"],
        "rationale": judgment["rationale"],
        "provider": response_provider(body),
        "cost_usd": response_cost(body),
        "usage": body.get("usage", {}),
        "batch_usage": batch_usage,
        "batch_id": batch_id,
        "attempts": 1,
        "tstamp": time.time(),
        "raw_response": body,
    }
    return question_id, record, None


def run(
    questions_path: Path,
    answers_path: Path,
    output_path: Path,
    state_path: Path,
    base_url: str,
    api_key: str,
    batch_model: str,
    max_tokens: int,
    max_batch_retries: int,
    max_api_retries: int,
    retry_base_seconds: float,
    poll_interval_seconds: float,
    timeout: float,
    max_cost_usd: float | None,
) -> int:
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")
    questions = load_questions(questions_path)
    answers = load_answers(answers_path)
    judgments = load_existing_judgments(output_path)
    if set(answers) - set(questions):
        raise ValueError("answer file contains unknown question IDs")
    if set(judgments) - set(answers):
        raise ValueError("judgment file contains unknown question IDs")
    error_path = output_path.with_name(f"{output_path.stem}.errors.jsonl")
    state = load_state(state_path)
    all_pending = sorted(set(answers) - set(judgments))
    if not all_pending:
        print(f"all {len(answers)} answers already judged")
        return 0
    if state and isinstance(state.get("question_ids"), list):
        pending = [
            int(value) for value in state["question_ids"] if int(value) not in judgments
        ]
    else:
        pending = all_pending
    if not pending:
        print(f"all {len(answers)} answers already judged")
        return 0

    total_cost = sum(
        float(value.get("cost_usd", 0))
        for value in judgments.values()
        if isinstance(value.get("cost_usd"), (int, float))
    )
    batch_endpoint = batch_url(base_url)
    max_attempts = max_batch_retries + 1
    while pending:
        active = (
            state and state.get("status") in ACTIVE_STATUSES and state.get("batch_id")
        )
        if active:
            batch_id = str(state["batch_id"])
            attempt = int(state.get("attempt", 1))
        else:
            previous_attempt = int(state.get("attempt", 0)) if state else 0
            attempt = previous_attempt + 1
            if attempt > max_attempts:
                raise RuntimeError(
                    f"batch judging exhausted retries; pending questions: {pending}"
                )
            if state and previous_attempt:
                time.sleep(retry_base_seconds * (2 ** (previous_attempt - 1)))
            payload = {
                "endpoint": "/v1/chat/completions",
                "model": batch_model,
                "requests": batch_requests(pending, questions, answers, max_tokens),
            }
            response = api_call(
                batch_endpoint,
                api_key,
                "POST",
                payload,
                timeout,
                max_api_retries,
                retry_base_seconds,
            )
            batch_id = response.get("id")
            if not isinstance(batch_id, str) or not batch_id:
                raise RuntimeError(f"batch submission did not return an ID: {response}")
            state = {
                "version": 1,
                "batch_id": batch_id,
                "model": batch_model,
                "question_ids": pending,
                "attempt": attempt,
                "status": response.get("status", "validating"),
                "request_counts": response.get("request_counts", {}),
                "submitted_at": time.time(),
            }
            save_state(state_path, state)
            print(
                f"submitted batch {batch_id} with {len(pending)} questions", flush=True
            )

        while True:
            response = api_call(
                batch_url(base_url, batch_id),
                api_key,
                "GET",
                None,
                timeout,
                max_api_retries,
                retry_base_seconds,
            )
            status = response.get("status")
            state.update(
                {
                    "status": status,
                    "request_counts": response.get("request_counts", {}),
                    "polled_at": time.time(),
                }
            )
            save_state(state_path, state)
            print(
                f"batch {batch_id}: {status} {response.get('request_counts', {})}",
                flush=True,
            )
            if status in TERMINAL_STATUSES:
                break
            if status not in ACTIVE_STATUSES:
                raise RuntimeError(f"batch returned unknown status: {status}")
            time.sleep(poll_interval_seconds)

        if status != "completed":
            write_error(
                error_path,
                {
                    "batch_id": batch_id,
                    "status": status,
                    "question_ids": pending,
                    "tstamp": time.time(),
                },
            )
            state["status"] = status
            save_state(state_path, state)
            if attempt >= max_attempts:
                raise RuntimeError(f"batch {batch_id} ended with status {status}")
            state["status"] = "retrying"
            save_state(state_path, state)
            continue

        results = response.get("results")
        if not isinstance(results, list):
            raise TypeError(f"completed batch {batch_id} did not include results")
        batch_usage = response.get("usage", {})
        if not isinstance(batch_usage, dict):
            batch_usage = {}
        batch_cost = batch_usage.get("cost")
        if (
            max_cost_usd is not None
            and isinstance(batch_cost, (int, float))
            and total_cost + float(batch_cost) > max_cost_usd
        ):
            state["status"] = "cost_exceeded"
            save_state(state_path, state)
            raise RuntimeError(
                f"cost ceiling exceeded: {total_cost + float(batch_cost):.6f} > {max_cost_usd:.6f} USD"
            )

        failed: list[int] = []
        with output_path.open("a", encoding="utf-8") as handle:
            for item in results:
                if not isinstance(item, dict):
                    continue
                question_id, record, error = result_record(
                    item, answers, questions, batch_model, batch_usage, batch_id
                )
                if error:
                    failed.append(question_id)
                    write_error(
                        error_path,
                        {
                            "batch_id": batch_id,
                            "question_id": question_id,
                            "error": error,
                            "tstamp": time.time(),
                        },
                    )
                    continue
                if question_id in judgments:
                    continue
                judgments[question_id] = record
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"judged question {question_id}: {record['score']}/10", flush=True
                )
        pending = sorted(set(pending) - set(judgments) | set(failed))
        state.update(
            {
                "status": "completed",
                "failed_question_ids": pending,
                "completed_at": time.time(),
            }
        )
        save_state(state_path, state)
        if not pending:
            return 0
        if attempt >= max_attempts:
            raise RuntimeError(
                f"batch {batch_id} completed with {len(pending)} failed question(s)"
            )
        state["status"] = "retrying"
        save_state(state_path, state)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument(
        "--base-url", default=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
    )
    parser.add_argument("--dotenv", type=Path, default=ROOT / ".env")
    parser.add_argument("--batch-model", default=DEFAULT_BATCH_MODEL)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-batch-retries", type=int, default=2)
    parser.add_argument("--max-api-retries", type=int, default=5)
    parser.add_argument("--retry-base-seconds", type=float, default=5.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-cost-usd", type=float)
    args = parser.parse_args()
    if not args.check_config and args.answers is None:
        parser.error("--answers is required unless --check-config is used")
    if args.output is None and args.answers is not None:
        args.output = args.answers.with_name("judgments.jsonl")
    if args.state is None and args.output is not None:
        args.state = args.output.with_name("judge-batch.json")
    if args.max_tokens <= 0 or args.max_batch_retries < 0 or args.max_api_retries < 0:
        parser.error("max-tokens must be positive; retry counts cannot be negative")
    if (
        args.retry_base_seconds < 0
        or args.poll_interval_seconds < 0
        or args.timeout <= 0
    ):
        parser.error(
            "retry-base-seconds and poll-interval-seconds cannot be negative; timeout must be positive"
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
            state_path=args.state,
            base_url=args.base_url,
            api_key=api_key,
            batch_model=args.batch_model,
            max_tokens=args.max_tokens,
            max_batch_retries=args.max_batch_retries,
            max_api_retries=args.max_api_retries,
            retry_base_seconds=args.retry_base_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
            timeout=args.timeout,
            max_cost_usd=args.max_cost_usd,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
