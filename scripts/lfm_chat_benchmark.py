#!/usr/bin/env python3
"""Run the local 32K customized MT-Bench quality evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "benchmarks" / "manifests" / "lfm-qad-models.tsv"
QUESTIONS = ROOT / "benchmarks" / "mt-bench" / "question.jsonl"
ANSWER_SCRIPT = ROOT / "scripts" / "mt_bench_answers.py"
JUDGE_SCRIPT = ROOT / "scripts" / "mt_bench_batch_judge.py"
DEFAULT_BENCHMARK_ROOT = ROOT / ".benchmark" / "lfm-qad-results"
DEFAULT_SCRATCH_ROOT = ROOT / ".benchmark" / "lfm-qad-models"
DEFAULT_REPORT_ROOT = ROOT / "results" / "lfm-qad"
QUALITY_CONTEXT_TOKENS = 32768
DEFAULT_PORT = 18081
DEFAULT_SERVER_TIMEOUT_SECONDS = 300
DEFAULT_SERVER_REQUEST_TIMEOUT_SECONDS = 21600
DEFAULT_DOWNLOAD_ATTEMPTS = 3
DEFAULT_DOWNLOAD_RETRY_SECONDS = 15
DEFAULT_SERVER_GPU_LAYERS = "auto"
DEFAULT_SERVER_PARALLEL = 1
DEFAULT_ANSWER_MAX_TOKENS = 32768
DEFAULT_ANSWER_TEMPERATURE = 0.1
DEFAULT_ANSWER_TIMEOUT_SECONDS = 21600
DEFAULT_REASONING_MODE = "auto"
DEFAULT_JUDGE_MODEL = "google/gemini-3.7-flash:batch"
DEFAULT_JUDGE_BATCH_RETRIES = 5
DEFAULT_JUDGE_API_RETRIES = 5
DEFAULT_JUDGE_RETRY_BASE_SECONDS = 5.0
DEFAULT_JUDGE_POLL_SECONDS = 30.0
MARKER_NAME = ".lfm-chat-quality-owned"
MARKER_OWNER = "lfm-chat-quality-v1"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    repository: str
    file_name: str
    official_context_tokens: int
    model_directory: str


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{timestamp}] {message}", flush=True)


def fail(message: str) -> None:
    raise RuntimeError(message)


def env_int(name: str, default: int, minimum: int = 0) -> int:
    value = os.environ.get(name, str(default))
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return parsed


def env_gpu_layers(name: str, default: str, minimum: int = 0) -> int | str:
    value = os.environ.get(name, default)
    if value == "auto":
        return value
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer >= {minimum} or auto") from error
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return parsed


def env_float(name: str, default: float, minimum: float = 0.0) -> float:
    value = os.environ.get(name, str(default))
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number >= {minimum}") from error
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return parsed


def config() -> dict[str, Any]:
    return {
        "benchmark_root": Path(
            os.environ.get("LFM_CHAT_QUALITY_ROOT", DEFAULT_BENCHMARK_ROOT)
        ),
        "scratch_root": Path(
            os.environ.get("LFM_CHAT_QUALITY_SCRATCH_ROOT", DEFAULT_SCRATCH_ROOT)
        ),
        "report_root": Path(
            os.environ.get("LFM_CHAT_QUALITY_REPORT_ROOT", DEFAULT_REPORT_ROOT)
        ),
        "hf_bin": os.environ.get("LFM_CHAT_HF_BIN", "hf"),
        "llama_server_bin": os.environ.get("LFM_CHAT_LLAMA_SERVER_BIN", "llama-server"),
        "uv_bin": os.environ.get("LFM_CHAT_UV_BIN", "uv"),
        "server_host": os.environ.get("LFM_CHAT_SERVER_HOST", "127.0.0.1"),
        "server_port": env_int("LFM_CHAT_PORT", DEFAULT_PORT, 1),
        "server_timeout": env_int(
            "LFM_CHAT_SERVER_TIMEOUT_SECONDS", DEFAULT_SERVER_TIMEOUT_SECONDS, 1
        ),
        "server_request_timeout": env_int(
            "LFM_CHAT_SERVER_REQUEST_TIMEOUT_SECONDS",
            DEFAULT_SERVER_REQUEST_TIMEOUT_SECONDS,
            1,
        ),
        "download_attempts": env_int(
            "LFM_CHAT_DOWNLOAD_ATTEMPTS", DEFAULT_DOWNLOAD_ATTEMPTS, 1
        ),
        "download_retry_seconds": env_int(
            "LFM_CHAT_DOWNLOAD_RETRY_SECONDS", DEFAULT_DOWNLOAD_RETRY_SECONDS, 0
        ),
        "server_gpu_layers": env_gpu_layers(
            "LFM_CHAT_SERVER_GPU_LAYERS", DEFAULT_SERVER_GPU_LAYERS, 0
        ),
        "server_parallel": env_int(
            "LFM_CHAT_SERVER_PARALLEL", DEFAULT_SERVER_PARALLEL, 1
        ),
        "answer_max_tokens": env_int(
            "LFM_CHAT_ANSWER_MAX_TOKENS", DEFAULT_ANSWER_MAX_TOKENS, 1
        ),
        "answer_temperature": env_float(
            "LFM_CHAT_ANSWER_TEMPERATURE", DEFAULT_ANSWER_TEMPERATURE
        ),
        "answer_timeout": env_float(
            "LFM_CHAT_ANSWER_TIMEOUT_SECONDS", DEFAULT_ANSWER_TIMEOUT_SECONDS, 1
        ),
        "reasoning_mode": os.environ.get("LFM_CHAT_REASONING", DEFAULT_REASONING_MODE),
        "reasoning_preserve": True,
        "judge_model": os.environ.get("LFM_CHAT_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        "judge_batch_retries": env_int(
            "LFM_CHAT_JUDGE_BATCH_RETRIES", DEFAULT_JUDGE_BATCH_RETRIES, 0
        ),
        "judge_api_retries": env_int(
            "LFM_CHAT_JUDGE_API_RETRIES", DEFAULT_JUDGE_API_RETRIES, 0
        ),
        "judge_retry_base_seconds": env_float(
            "LFM_CHAT_JUDGE_RETRY_BASE_SECONDS", DEFAULT_JUDGE_RETRY_BASE_SECONDS
        ),
        "judge_poll_seconds": env_float(
            "LFM_CHAT_JUDGE_POLL_SECONDS", DEFAULT_JUDGE_POLL_SECONDS
        ),
        "judge_max_cost_usd": os.environ.get("LFM_CHAT_JUDGE_MAX_COST_USD", ""),
    }


def load_manifest(path: Path) -> list[ModelSpec]:
    models: list[ModelSpec] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise ValueError(
                    f"manifest line {line_number} must have five tab-separated fields"
                )
            model_id, repository, file_name, context, model_directory = fields
            if model_id in seen or not re.fullmatch(r"[a-z0-9._-]+", model_id):
                raise ValueError(
                    f"invalid or duplicate model ID on manifest line {line_number}"
                )
            if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repository):
                raise ValueError(f"invalid repository on manifest line {line_number}")
            if not re.fullmatch(r"[A-Za-z0-9_.-]+\.gguf", file_name):
                raise ValueError(
                    f"invalid GGUF filename on manifest line {line_number}"
                )
            if not re.fullmatch(r"[A-Za-z0-9._-]+", model_directory):
                raise ValueError(
                    f"invalid model directory on manifest line {line_number}"
                )
            try:
                context_tokens = int(context)
            except ValueError as error:
                raise ValueError(
                    f"invalid context on manifest line {line_number}"
                ) from error
            if context_tokens < 1024:
                raise ValueError(f"context is too small on manifest line {line_number}")
            models.append(
                ModelSpec(
                    model_id, repository, file_name, context_tokens, model_directory
                )
            )
            seen.add(model_id)
    if not models:
        raise ValueError("manifest must contain at least one model")
    return models


def load_questions(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    questions: list[dict[str, Any]] = []
    ids: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                question = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid question JSON at line {line_number}"
                ) from error
            question_id = question.get("question_id")
            turns = question.get("turns")
            if (
                not isinstance(question_id, int)
                or question_id in ids
                or not isinstance(turns, list)
                or len(turns) != 2
                or not all(isinstance(turn, str) for turn in turns)
            ):
                raise ValueError(f"invalid two-turn question at line {line_number}")
            questions.append(question)
            ids.add(question_id)
    if len(questions) != 80:
        raise ValueError(f"expected 80 questions, found {len(questions)}")
    return questions, sorted(ids)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL in {path} at line {line_number}"
                ) from error
            if not isinstance(value, dict):
                raise TypeError(
                    f"JSONL record in {path} at line {line_number} is not an object"
                )
            records.append(value)
    return records


def answers_complete(path: Path, expected_ids: list[int]) -> bool:
    records = read_jsonl(path)
    if len(records) != len(expected_ids):
        return False
    found: list[int] = []
    for record in records:
        question_id = record.get("question_id")
        choices = record.get("choices")
        turns = (
            choices[0].get("turns") if isinstance(choices, list) and choices else None
        )
        if (
            not isinstance(question_id, int)
            or not isinstance(turns, list)
            or len(turns) != 2
            or not all(isinstance(turn, str) and turn.strip() for turn in turns)
        ):
            return False
        generation = record.get("generation")
        finish_reasons = (
            generation.get("finish_reasons") if isinstance(generation, dict) else None
        )
        if isinstance(finish_reasons, list) and (
            len(finish_reasons) != 2
            or any(reason != "stop" for reason in finish_reasons)
        ):
            return False
        found.append(question_id)
    return sorted(found) == expected_ids


def judgments_complete(path: Path, expected_ids: list[int]) -> bool:
    records = read_jsonl(path)
    if len(records) != len(expected_ids):
        return False
    found: list[int] = []
    for record in records:
        question_id = record.get("question_id")
        score = record.get("score")
        if (
            not isinstance(question_id, int)
            or not isinstance(score, int)
            or isinstance(score, bool)
            or not 1 <= score <= 10
        ):
            return False
        found.append(question_id)
    return sorted(found) == expected_ids


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def invalidate_stale_judgments(answer_path: Path, judgment_path: Path) -> None:
    if not judgment_path.exists():
        return
    answers = {
        record.get("question_id"): record.get("answer_id")
        for record in read_jsonl(answer_path)
        if isinstance(record.get("question_id"), int)
    }
    judgments = read_jsonl(judgment_path)
    current = [
        record
        for record in judgments
        if answers.get(record.get("question_id")) == record.get("answer_id")
    ]
    if len(current) != len(judgments):
        write_jsonl(judgment_path, current)
        log(
            f"JUDGMENT INVALIDATE: removed {len(judgments) - len(current)} stale records"
        )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_state(path: Path, state: dict[str, Any]) -> None:
    write_json(path, {**state, "updated_at": datetime.now(timezone.utc).isoformat()})


def command_available(command: str) -> bool:
    return bool(shutil.which(command))


def process_group_stop(process: subprocess.Popen[Any], force: bool = False) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        return


def run_command(command: Sequence[str]) -> None:
    log("RUN: " + " ".join(str(part) for part in command))
    process = subprocess.Popen(list(command), start_new_session=True)
    try:
        status = process.wait()
    except BaseException:
        process_group_stop(process)
        process.wait()
        raise
    if status != 0:
        raise RuntimeError(f"command failed with exit status {status}: {command[0]}")


def health_check(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


class LocalServer:
    def __init__(
        self,
        config_values: dict[str, Any],
        model: ModelSpec,
        result_dir: Path,
        model_path: Path,
    ):
        self.config = config_values
        self.model = model
        self.result_dir = result_dir
        self.model_path = model_path
        self.process: subprocess.Popen[Any] | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.config['server_host']}:{self.config['server_port']}/v1"

    def start(self) -> None:
        health_url = f"{self.base_url}/health"
        if health_check(health_url):
            fail(f"port {self.config['server_port']} is already serving an endpoint")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.result_dir / "llama-server.log"
        log(
            f"SERVER START: {self.model.model_id} context={QUALITY_CONTEXT_TOKENS} "
            f"gpu_layers={self.config['server_gpu_layers']} parallel={self.config['server_parallel']}"
        )
        with log_path.open("a", encoding="utf-8") as handle:
            self.process = subprocess.Popen(
                [
                    self.config["llama_server_bin"],
                    "--model",
                    str(self.model_path),
                    "--host",
                    self.config["server_host"],
                    "--port",
                    str(self.config["server_port"]),
                    "--ctx-size",
                    str(QUALITY_CONTEXT_TOKENS),
                    "--n-gpu-layers",
                    str(self.config["server_gpu_layers"]),
                    "--parallel",
                    str(self.config["server_parallel"]),
                    "--timeout",
                    str(self.config["server_request_timeout"]),
                    "--reasoning",
                    self.config["reasoning_mode"],
                    "--reasoning-preserve",
                    "--alias",
                    self.model.model_id,
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        for _ in range(self.config["server_timeout"]):
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited before becoming healthy: {log_path}"
                )
            if health_check(health_url):
                log(f"SERVER READY: {self.model.model_id} url={self.base_url}")
                return
            time.sleep(1)
        raise RuntimeError(f"llama-server did not become healthy: {log_path}")

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            self.process = None
            return
        process_group_stop(self.process)
        for _ in range(10):
            if self.process.poll() is not None:
                break
            time.sleep(1)
        if self.process.poll() is None:
            process_group_stop(self.process, force=True)
        self.process.wait()
        self.process = None
        log(f"SERVER STOPPED: {self.model.model_id}")


def marker_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def model_dir_for(config_values: dict[str, Any], model: ModelSpec) -> Path:
    scratch_root = Path(config_values["scratch_root"]).resolve()
    return scratch_root / model.model_directory


def verify_owned_model_dir(
    config_values: dict[str, Any], model: ModelSpec, model_dir: Path
) -> None:
    scratch_root = Path(config_values["scratch_root"]).resolve()
    if model_dir.parent != scratch_root:
        fail(f"model directory is outside the scratch root: {model_dir}")
    if model_dir.is_symlink() or not model_dir.is_dir():
        fail(f"model scratch path is not a normal directory: {model_dir}")
    marker = model_dir / MARKER_NAME
    if not marker.is_file():
        fail(f"model directory has no ownership marker: {model_dir}")
    values = marker_values(marker)
    expected = {
        "owner": MARKER_OWNER,
        "model_id": model.model_id,
        "repo": model.repository,
        "file": model.file_name,
    }
    if any(values.get(key) != value for key, value in expected.items()):
        fail(f"ownership marker mismatch: {marker}")


def prepare_model(config_values: dict[str, Any], model: ModelSpec) -> Path:
    scratch_root = Path(config_values["scratch_root"]).resolve()
    model_dir = model_dir_for(config_values, model)
    scratch_root.mkdir(parents=True, exist_ok=True)
    if str(scratch_root).startswith("/Volumes/Touro"):
        fail(f"refusing to use Touro as scratch storage: {scratch_root}")
    if model_dir.exists():
        verify_owned_model_dir(config_values, model, model_dir)
    else:
        model_dir.mkdir(parents=True)
        (model_dir / MARKER_NAME).write_text(
            "\n".join(
                [
                    f"owner={MARKER_OWNER}",
                    f"model_id={model.model_id}",
                    f"repo={model.repository}",
                    f"file={model.file_name}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    expected_file = model_dir / model.file_name
    if (
        expected_file.is_file()
        and not expected_file.is_symlink()
        and expected_file.stat().st_size > 0
    ):
        log(f"DOWNLOAD SKIP: {model.model_id}")
        return expected_file
    for attempt in range(1, config_values["download_attempts"] + 1):
        log(
            f"DOWNLOAD: {model.model_id} attempt={attempt}/{config_values['download_attempts']}"
        )
        try:
            run_command(
                [
                    config_values["hf_bin"],
                    "download",
                    model.repository,
                    model.file_name,
                    "--repo-type",
                    "model",
                    "--local-dir",
                    str(model_dir),
                ]
            )
        except RuntimeError:
            if attempt == config_values["download_attempts"]:
                raise
            time.sleep(config_values["download_retry_seconds"])
            continue
        if (
            expected_file.is_file()
            and not expected_file.is_symlink()
            and expected_file.stat().st_size > 0
        ):
            log(f"DOWNLOAD COMPLETE: {model.model_id}")
            return expected_file
    fail(f"download did not produce the expected GGUF: {expected_file}")


def cleanup_model(config_values: dict[str, Any], model: ModelSpec) -> None:
    model_dir = model_dir_for(config_values, model)
    if not model_dir.exists():
        return
    verify_owned_model_dir(config_values, model, model_dir)
    expected_file = model_dir / model.file_name
    if expected_file.is_symlink():
        fail(f"refusing to clean a symlinked GGUF: {expected_file}")
    log(f"CLEANUP: removing {model_dir}")
    shutil.rmtree(model_dir)


def finalize_model_scratch(
    config_values: dict[str, Any], model: ModelSpec, succeeded: bool
) -> None:
    if succeeded:
        cleanup_model(config_values, model)
        return
    log(
        "CLEANUP SKIP: retaining model weights after unsuccessful run: "
        f"{model_dir_for(config_values, model)}"
    )


def validate_tools(config_values: dict[str, Any], manifest: Path) -> None:
    for name in ("hf_bin", "llama_server_bin", "uv_bin"):
        command = config_values[name]
        if not command_available(command):
            fail(f"required command was not found: {command}")
    for path in (manifest, QUESTIONS, ANSWER_SCRIPT, JUDGE_SCRIPT):
        if not path.is_file():
            fail(f"required file was not found: {path}")


def validate_openrouter(config_values: dict[str, Any]) -> None:
    command = [
        config_values["uv_bin"],
        "run",
        "--project",
        str(ROOT),
        "python",
        str(JUDGE_SCRIPT),
        "--check-config",
        "--dotenv",
        str(ROOT / ".env"),
    ]
    try:
        subprocess.run(
            command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured in the environment or repository .env file"
        ) from error


def run_answers(
    config_values: dict[str, Any], model: ModelSpec, result_dir: Path
) -> None:
    answer_path = result_dir / "answers.jsonl"
    _, expected_ids = load_questions(QUESTIONS)
    if answers_complete(answer_path, expected_ids):
        log(f"ANSWERS SKIP: {model.model_id} already has 80 answers")
        return
    model_path = model_dir_for(config_values, model) / model.file_name
    server = LocalServer(config_values, model, result_dir, model_path)
    server.start()
    try:
        run_command(
            [
                config_values["uv_bin"],
                "run",
                "--project",
                str(ROOT),
                "python",
                str(ANSWER_SCRIPT),
                "--model-id",
                model.model_id,
                "--questions",
                str(QUESTIONS),
                "--output",
                str(answer_path),
                "--server-url",
                server.base_url,
                "--temperature",
                str(config_values["answer_temperature"]),
                "--max-tokens",
                str(config_values["answer_max_tokens"]),
                "--timeout",
                str(config_values["answer_timeout"]),
            ]
        )
    finally:
        server.stop()
    if not answers_complete(answer_path, expected_ids):
        fail(f"answer generation did not produce 80 complete answers: {answer_path}")
    log(f"ANSWERS COMPLETE: {model.model_id}")


def run_judge(
    config_values: dict[str, Any], model: ModelSpec, result_dir: Path
) -> None:
    answer_path = result_dir / "answers.jsonl"
    judgment_path = result_dir / "judgments.jsonl"
    _, expected_ids = load_questions(QUESTIONS)
    if not answers_complete(answer_path, expected_ids):
        fail(f"refusing to judge incomplete or truncated answers: {answer_path}")
    if judgments_complete(judgment_path, expected_ids):
        log(f"JUDGMENT SKIP: {model.model_id} already has 80 judgments")
        return
    command = [
        config_values["uv_bin"],
        "run",
        "--project",
        str(ROOT),
        "python",
        str(JUDGE_SCRIPT),
        "--answers",
        str(answer_path),
        "--questions",
        str(QUESTIONS),
        "--output",
        str(judgment_path),
        "--batch-model",
        config_values["judge_model"],
        "--max-tokens",
        "1024",
        "--max-batch-retries",
        str(config_values["judge_batch_retries"]),
        "--max-api-retries",
        str(config_values["judge_api_retries"]),
        "--retry-base-seconds",
        str(config_values["judge_retry_base_seconds"]),
        "--poll-interval-seconds",
        str(config_values["judge_poll_seconds"]),
        "--dotenv",
        str(ROOT / ".env"),
    ]
    if config_values["judge_max_cost_usd"]:
        command.extend(["--max-cost-usd", config_values["judge_max_cost_usd"]])
    log(f"JUDGMENT START: {model.model_id} judge={config_values['judge_model']}")
    run_command(command)
    if not judgments_complete(judgment_path, expected_ids):
        fail(f"judgment generation did not produce 80 valid judgments: {judgment_path}")
    log(f"JUDGMENT COMPLETE: {model.model_id}")


def process_model(
    config_values: dict[str, Any], model: ModelSpec, questions: list[dict[str, Any]]
) -> None:
    result_dir = Path(config_values["benchmark_root"]) / model.model_id
    result_dir.mkdir(parents=True, exist_ok=True)
    state_path = result_dir / "state.json"
    state: dict[str, Any] = {"model_id": model.model_id, "status": "starting"}
    succeeded = False
    write_state(state_path, state)
    try:
        _, expected_ids = load_questions(QUESTIONS)
        answers_path = result_dir / "answers.jsonl"
        judgments_path = result_dir / "judgments.jsonl"
        answers_done = answers_complete(answers_path, expected_ids)
        judgments_done = judgments_complete(judgments_path, expected_ids)
        if answers_done and judgments_done:
            state.update(
                {
                    "status": "complete",
                    "message": "existing quality artifacts are complete",
                }
            )
            write_state(state_path, state)
            log(f"MODEL SKIP: {model.model_id} is complete")
            succeeded = True
            return
        if not answers_done:
            state.update({"status": "downloading"})
            write_state(state_path, state)
            prepare_model(config_values, model)
            state.update({"status": "answers"})
            write_state(state_path, state)
            run_answers(config_values, model, result_dir)
            invalidate_stale_judgments(answers_path, judgments_path)
            answers_done = answers_complete(answers_path, expected_ids)
            if not answers_done:
                fail(
                    f"answer generation did not produce 80 judgeable answers: {answers_path}"
                )
            judgments_done = judgments_complete(judgments_path, expected_ids)
        if not judgments_done:
            state.update({"status": "judging"})
            write_state(state_path, state)
            run_judge(config_values, model, result_dir)
        state.update({"status": "complete", "message": "quality evaluation completed"})
        write_state(state_path, state)
        log(f"MODEL COMPLETE: {model.model_id}")
        succeeded = True
    except KeyboardInterrupt:
        state.update(
            {"status": "interrupted", "message": "operator interrupted the model"}
        )
        write_state(state_path, state)
        raise
    except Exception as error:
        state.update({"status": "failed", "message": str(error)})
        write_state(state_path, state)
        raise
    finally:
        finalize_model_scratch(config_values, model, succeeded)


def summarize_model(
    config_values: dict[str, Any],
    model: ModelSpec,
    result_root: Path,
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    result_dir = result_root / model.model_id
    judgments = read_jsonl(result_dir / "judgments.jsonl")
    question_categories = {
        question["question_id"]: question.get("category", "unknown")
        for question in questions
    }
    scores = [
        record["score"] for record in judgments if isinstance(record.get("score"), int)
    ]
    categories: dict[str, list[int]] = {}
    for record in judgments:
        score = record.get("score")
        question_id = record.get("question_id")
        if isinstance(score, int) and isinstance(question_id, int):
            categories.setdefault(
                question_categories.get(question_id, "unknown"), []
            ).append(score)
    return {
        "model_id": model.model_id,
        "official_context_tokens": model.official_context_tokens,
        "quality_context_tokens": QUALITY_CONTEXT_TOKENS,
        "answer_generation": {
            "temperature": config_values["answer_temperature"],
            "max_tokens": config_values["answer_max_tokens"],
            "timeout_seconds": config_values["answer_timeout"],
            "reasoning": config_values["reasoning_mode"],
            "reasoning_preserve": config_values["reasoning_preserve"],
        },
        "answers": len(read_jsonl(result_dir / "answers.jsonl")),
        "judgments": len(judgments),
        "mean_score": round(statistics.fmean(scores), 4) if scores else None,
        "category_means": {
            category: round(statistics.fmean(values), 4)
            for category, values in sorted(categories.items())
        },
        "status": "complete" if len(scores) == 80 else "partial",
        "state_file": str(result_dir / "state.json"),
    }


def write_quality_report(
    config_values: dict[str, Any],
    models: list[ModelSpec],
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    result_root = Path(config_values["benchmark_root"])
    summaries = [
        summarize_model(config_values, model, result_root, questions)
        for model in models
    ]
    ranking = [
        summary["model_id"]
        for summary in sorted(
            summaries,
            key=lambda value: (
                value["mean_score"] is None,
                -(value["mean_score"] or 0),
                value["model_id"],
            ),
        )
    ]
    complete_ranking = [
        summary["model_id"]
        for summary in sorted(
            (summary for summary in summaries if summary["status"] == "complete"),
            key=lambda value: (-(value["mean_score"] or 0), value["model_id"]),
        )
    ]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "Customized MT-Bench",
        "question_count": len(questions),
        "quality_context_tokens": QUALITY_CONTEXT_TOKENS,
        "judge": {
            "model": config_values["judge_model"],
            "temperature": 0,
            "score_range": [1, 10],
            "batch_api": True,
        },
        "models": summaries,
        "rankings": {"mt_bench_mean": ranking},
        "performance_top_two": complete_ranking[:2]
        if len(complete_ranking) >= 2
        else [],
        "note": "This ranking compares these candidates under the same local generation and judge configuration.",
    }
    report_root = Path(config_values["report_root"])
    write_json(report_root / "quality.json", report)
    lines = [
        "# Local Chat Quality",
        "",
        "Customized MT-Bench evaluation at a fixed 32,768-token server context.",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Generation: temperature `{config_values['answer_temperature']}`, max tokens `{config_values['answer_max_tokens']}`, reasoning `{config_values['reasoning_mode']}`",
        f"Judge: `{config_values['judge_model']}`, temperature `0`",
        "",
        "| Rank | Model | Answers | Judgments | Mean (1-10) | Status |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for rank, model_id in enumerate(ranking, 1):
        summary = next(value for value in summaries if value["model_id"] == model_id)
        mean = (
            "n/a" if summary["mean_score"] is None else f"{summary['mean_score']:.2f}"
        )
        lines.append(
            f"| {rank} | `{model_id}` | {summary['answers']} | {summary['judgments']} | {mean} | {summary['status']} |"
        )
    lines.extend(
        ["", "## Category Means", "", "| Model | Category means |", "|---|---|"]
    )
    for summary in summaries:
        category_text = (
            ", ".join(
                f"{category}: {score:.2f}"
                for category, score in summary["category_means"].items()
            )
            or "n/a"
        )
        lines.append(f"| `{summary['model_id']}` | {category_text} |")
    lines.extend(
        [
            "",
            "## Performance Selection",
            "",
            "Top two after complete MT-Bench quality results:",
        ]
    )
    lines.extend(
        f"- `{model_id}`"
        for model_id in report["performance_top_two"] or ["not available yet"]
    )
    markdown_path = report_root / "quality.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


class Lock:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self) -> Lock:  # noqa: PYI034
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            file_descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                handle.write(f"pid={os.getpid()}\n")
        except FileExistsError as error:
            raise RuntimeError(
                f"another quality evaluation is running: {self.path}"
            ) from error
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.path.unlink(missing_ok=True)


def print_dry_run(
    config_values: dict[str, Any], models: list[ModelSpec], requested_model: str | None
) -> None:
    log(
        "DRY RUN: no directories, downloads, servers, inference, or API calls will start."
    )
    for model in models:
        if requested_model and model.model_id != requested_model:
            continue
        log(
            f"PLAN: model={model.model_id} repo={model.repository} file={model.file_name} "
            f"quality_context={QUALITY_CONTEXT_TOKENS} scratch={model_dir_for(config_values, model)}"
        )
    log(
        f"PLAN: generation temperature={config_values['answer_temperature']} "
        f"max_tokens={config_values['answer_max_tokens']} reasoning={config_values['reasoning_mode']} "
        f"reasoning_preserve={config_values['reasoning_preserve']} "
        f"timeout={config_values['answer_timeout']}s"
    )
    log(
        f"PLAN: server_parallel={config_values['server_parallel']} "
        f"judge={config_values['judge_model']} port={config_values['server_port']}"
    )
    log(f"PLAN: report={Path(config_values['report_root']) / 'quality.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    quality = subparsers.add_parser(
        "quality", help="run a 32K MT-Bench quality evaluation"
    )
    quality.add_argument("--dry-run", action="store_true")
    quality.add_argument("--model", help="process one manifest model ID")
    quality.add_argument("--manifest", type=Path, default=MANIFEST)
    quality.add_argument("--benchmark-root", type=Path)
    quality.add_argument("--report-root", type=Path)
    quality.add_argument("--max-tokens", type=int)
    quality.add_argument("--reasoning", choices=("off", "on", "auto"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command != "quality":
        raise RuntimeError(f"unsupported command: {args.command}")
    config_values = config()
    if args.benchmark_root is not None:
        config_values["benchmark_root"] = args.benchmark_root
    if args.report_root is not None:
        config_values["report_root"] = args.report_root
    if args.max_tokens is not None:
        if args.max_tokens <= 0:
            raise ValueError("--max-tokens must be positive")
        config_values["answer_max_tokens"] = args.max_tokens
    if args.reasoning is not None:
        config_values["reasoning_mode"] = args.reasoning
    if config_values["reasoning_mode"] not in {"off", "on", "auto"}:
        raise ValueError("LFM_CHAT_REASONING must be one of: off, on, auto")
    manifest = args.manifest.resolve()
    models = load_manifest(manifest)
    questions, _ = load_questions(QUESTIONS)
    if args.model and args.model not in {model.model_id for model in models}:
        raise ValueError(f"model ID is not in the manifest: {args.model}")
    if args.dry_run:
        print_dry_run(config_values, models, args.model)
        return 0
    validate_tools(config_values, manifest)
    validate_openrouter(config_values)
    benchmark_root = Path(config_values["benchmark_root"]).resolve()
    scratch_root = Path(config_values["scratch_root"]).resolve()
    if benchmark_root == scratch_root:
        fail("quality benchmark root and scratch root must be different")
    benchmark_root.mkdir(parents=True, exist_ok=True)
    selected_models = [
        model for model in models if not args.model or model.model_id == args.model
    ]
    overall_status = 0
    with Lock(benchmark_root / ".lock"):
        for model in selected_models:
            try:
                process_model(config_values, model, questions)
            except KeyboardInterrupt:
                raise
            # Continue the resumable batch when one model fails.
            except Exception as error:  # noqa: BLE001
                overall_status = 1
                log(f"MODEL FAILED: {model.model_id}: {error}")
            finally:
                write_quality_report(config_values, models, questions)
    report = write_quality_report(config_values, models, questions)
    log(f"QUALITY REPORT: {Path(config_values['report_root']) / 'quality.md'}")
    if overall_status:
        log("QUALITY PARTIAL: inspect model state and rerun to resume")
        return 1
    selected_summaries = [
        summary
        for summary in report["models"]
        if not args.model or summary["model_id"] == args.model
    ]
    if any(summary["status"] != "complete" for summary in selected_summaries):
        log("QUALITY PARTIAL: rerun to resume incomplete model artifacts")
        return 1
    log("QUALITY COMPLETE: all selected models have 80 answers and 80 judgments")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
