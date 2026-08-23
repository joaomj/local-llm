#!/usr/bin/env python3
"""Benchmark configured llama-server capacity and occupied chat prompts."""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from benchmark_conversation import SCENARIO_VERSION, build_tool_conversation

CONTRACT = "llama_server_capacity_and_prompt-v1"
DEFAULT_CAPACITIES = (32768, 49152, 65536, 98304, 131072, 150000)
DEFAULT_PROMPT_TARGETS = (8192, 16384, 32768, 49152, 65536, 98304, 128000)


class BenchmarkError(RuntimeError):
    """An operational error that invalidates the benchmark run."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def env_int(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise BenchmarkError(f"{name} must be an integer >= {minimum}") from error
    if value < minimum:
        raise BenchmarkError(f"{name} must be >= {minimum}")
    return value


def env_int_list(name: str, default: tuple[int, ...]) -> list[int]:
    raw = os.environ.get(name)
    if raw is None:
        return list(default)
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as error:
        raise BenchmarkError(
            f"{name} must be a comma-separated integer list"
        ) from error
    if not values or any(value <= 0 for value in values):
        raise BenchmarkError(f"{name} must contain positive integers")
    return values


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class Config:
    server_bin: str
    model_path: Path
    model_id: str
    maximum_context: int
    capacities: list[int]
    prompt_targets: list[int]
    generation_tokens: int
    output_reserve: int
    minimum_tps: int
    benchmark_root: Path
    startup_timeout: int
    request_timeout: int
    shutdown_timeout: int

    @classmethod
    def from_environment(cls) -> Config:
        model_path_raw = os.environ.get("LLAMA_SERVER_BENCHMARK_MODEL_PATH")
        model_id = os.environ.get("LLAMA_SERVER_BENCHMARK_MODEL_ID")
        if not model_path_raw or not model_id:
            raise BenchmarkError(
                "LLAMA_SERVER_BENCHMARK_MODEL_PATH and LLAMA_SERVER_BENCHMARK_MODEL_ID are required"
            )
        maximum = env_int("LLAMA_SERVER_BENCHMARK_MAX_CONTEXT", 131072, 32768)
        capacities = sorted(
            {
                value
                for value in env_int_list(
                    "LLAMA_SERVER_BENCHMARK_CAPACITIES", DEFAULT_CAPACITIES
                )
                if 32768 <= value <= maximum
            }
        )
        if 32768 not in capacities:
            capacities.insert(0, 32768)
        if maximum not in capacities:
            capacities.append(maximum)
        return cls(
            server_bin=os.environ.get("LLAMA_SERVER_BENCHMARK_BIN", "llama-server"),
            model_path=Path(model_path_raw).expanduser(),
            model_id=model_id,
            maximum_context=maximum,
            capacities=capacities,
            prompt_targets=sorted(
                set(
                    env_int_list(
                        "LLAMA_SERVER_BENCHMARK_PROMPT_TARGETS", DEFAULT_PROMPT_TARGETS
                    )
                )
            ),
            generation_tokens=env_int(
                "LLAMA_SERVER_BENCHMARK_GENERATION_TOKENS", 128, 1
            ),
            output_reserve=env_int("LLAMA_SERVER_BENCHMARK_OUTPUT_RESERVE", 2048, 256),
            minimum_tps=env_int("LLAMA_SERVER_BENCHMARK_MIN_GENERATION_TPS", 50, 1),
            benchmark_root=Path(
                os.environ.get(
                    "LLAMA_SERVER_BENCHMARK_ROOT", ".benchmark/throughput-server"
                )
            ).expanduser(),
            startup_timeout=env_int(
                "LLAMA_SERVER_BENCHMARK_STARTUP_TIMEOUT_SECONDS", 600, 1
            ),
            request_timeout=env_int(
                "LLAMA_SERVER_BENCHMARK_REQUEST_TIMEOUT_SECONDS", 3600, 1
            ),
            shutdown_timeout=env_int(
                "LLAMA_SERVER_BENCHMARK_SHUTDOWN_TIMEOUT_SECONDS", 30, 1
            ),
        )


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return int(server_socket.getsockname()[1])


def http_json(
    url: str,
    timeout: int,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise BenchmarkError(f"HTTP {error.code} from {url}: {body}") from error
    except (URLError, TimeoutError) as error:
        raise BenchmarkError(f"request failed for {url}: {error}") from error
    except json.JSONDecodeError as error:
        raise BenchmarkError(f"invalid JSON from {url}") from error
    if not isinstance(parsed, dict):
        raise BenchmarkError(f"non-object JSON from {url}")
    return parsed


def terminate(process: subprocess.Popen[bytes], timeout: int) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=timeout)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def server_command(config: Config, capacity: int, port: int) -> list[str]:
    return [
        config.server_bin,
        "--model",
        str(config.model_path),
        "--alias",
        config.model_id,
        "--no-mmproj",
        "--reasoning",
        "off",
        "--reasoning-budget",
        "0",
        "--ctx-size",
        str(capacity),
        "--flash-attn",
        "on",
        "--parallel",
        "1",
        "--perf",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def wait_until_ready(
    base_url: str, process: subprocess.Popen[bytes], timeout: int
) -> None:
    deadline = time.monotonic() + timeout
    last_error = "server was not ready"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise BenchmarkError(
                f"llama-server exited during startup with code {process.returncode}"
            )
        try:
            http_json(f"{base_url}/v1/models", 2)
            return
        except BenchmarkError as error:
            last_error = str(error)
            time.sleep(0.2)
    raise BenchmarkError(f"llama-server startup timed out: {last_error}")


def tokenize(base_url: str, content: str, timeout: int) -> int:
    response = http_json(f"{base_url}/tokenize", timeout, {"content": content})
    tokens = response.get("tokens")
    if not isinstance(tokens, list):
        raise BenchmarkError("/tokenize response does not contain a token list")
    return len(tokens)


def completion(
    config: Config,
    base_url: str,
    capacity: int,
    scenario: str,
    target_prompt_tokens: int,
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    response = http_json(
        f"{base_url}/v1/chat/completions",
        config.request_timeout,
        {
            "model": config.model_id,
            "messages": messages,
            "max_tokens": config.generation_tokens,
            "temperature": 0,
        },
    )
    elapsed = time.monotonic() - started
    timings = response.get("timings")
    if not isinstance(timings, dict):
        raise BenchmarkError("completion response does not contain timings")
    prompt_n = timings.get("prompt_n")
    predicted_n = timings.get("predicted_n")
    prompt_tps = timings.get("prompt_per_second")
    generation_tps = timings.get("predicted_per_second")
    numeric = (prompt_n, predicted_n, prompt_tps, generation_tps)
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        for value in numeric
    ):
        raise BenchmarkError("completion timings are incomplete")
    qualifies = float(generation_tps) >= config.minimum_tps
    measurement = {
        "measurement_contract": CONTRACT,
        "model_id": config.model_id,
        "scenario": scenario,
        "scenario_version": SCENARIO_VERSION
        if scenario == "tool_conversation"
        else "short-v1",
        "configured_context_tokens": capacity,
        "target_prompt_tokens": target_prompt_tokens,
        "actual_prompt_tokens": int(prompt_n),
        "generated_tokens": int(predicted_n),
        "prompt_tokens_per_second": float(prompt_tps),
        "generation_tokens_per_second": float(generation_tps),
        "wall_seconds": elapsed,
        "qualifies_for_minimum_tps": qualifies,
        "minimum_generation_tokens_per_second": config.minimum_tps,
        "measured_at": utc_now(),
    }
    return measurement, response


def run_capacity(
    config: Config, capacity: int, prompt_targets: list[int]
) -> list[dict[str, Any]]:
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    server_log = config.benchmark_root / f"server-{capacity}.log"
    command = server_command(config, capacity, port)
    log(f"SERVER START: capacity={capacity} command={' '.join(command)}")
    measurements: list[dict[str, Any]] = []
    with server_log.open("wb") as output:
        process = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_until_ready(base_url, process, config.startup_timeout)
            short, response = completion(
                config,
                base_url,
                capacity,
                "short",
                0,
                [
                    {
                        "role": "user",
                        "content": "Summarize the purpose of this benchmark.",
                    }
                ],
            )
            measurements.append(short)
            atomic_json(
                config.benchmark_root / f"response-{capacity}-short.json", response
            )
            log(
                f"MEASURED: capacity={capacity} scenario=short "
                f"actual_prompt={short['actual_prompt_tokens']} "
                f"generation={short['generation_tokens_per_second']} tok/s"
            )
            if not short["qualifies_for_minimum_tps"]:
                return measurements
            for target in prompt_targets:
                if target > capacity - config.output_reserve:
                    continue
                messages, content_tokens = build_tool_conversation(
                    target,
                    lambda text: tokenize(base_url, text, config.request_timeout),
                )
                occupied, response = completion(
                    config,
                    base_url,
                    capacity,
                    "tool_conversation",
                    target,
                    messages,
                )
                occupied["content_tokens_before_chat_template"] = content_tokens
                measurements.append(occupied)
                atomic_json(
                    config.benchmark_root / f"response-{capacity}-prompt-{target}.json",
                    response,
                )
                log(
                    f"MEASURED: capacity={capacity} scenario=tool_conversation "
                    f"target_prompt={target} actual_prompt={occupied['actual_prompt_tokens']} "
                    f"generation={occupied['generation_tokens_per_second']} tok/s"
                )
                if not occupied["qualifies_for_minimum_tps"]:
                    break
        finally:
            terminate(process, config.shutdown_timeout)
            log(f"SERVER STOP: capacity={capacity}")
    return measurements


def summarize(
    config: Config,
    measurements: list[dict[str, Any]],
    capacity_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    passing_short = [
        item
        for item in measurements
        if item["scenario"] == "short" and item["qualifies_for_minimum_tps"]
    ]
    passing_occupied = [
        item
        for item in measurements
        if item["scenario"] == "tool_conversation" and item["qualifies_for_minimum_tps"]
    ]
    baseline_short = any(
        item["configured_context_tokens"] == 32768 for item in passing_short
    )
    baseline_occupied = any(
        item["configured_context_tokens"] == 32768
        and item["target_prompt_tokens"] >= 8192
        for item in passing_occupied
    )
    return {
        "measurement_contract": CONTRACT,
        "model_id": config.model_id,
        "model_path": str(config.model_path),
        "minimum_configured_context_tokens": 32768,
        "minimum_generation_tokens_per_second": config.minimum_tps,
        "largest_passing_configured_context_tokens": max(
            (item["configured_context_tokens"] for item in passing_short), default=0
        ),
        "largest_passing_actual_prompt_tokens": max(
            (item["actual_prompt_tokens"] for item in passing_occupied), default=0
        ),
        "minimum_requirement_met": baseline_short and baseline_occupied,
        "weights_retained": True,
        "measurements": measurements,
        "capacity_failures": capacity_failures,
        "completed_at": utc_now(),
    }


def main() -> int:
    try:
        config = Config.from_environment()
        if not config.model_path.is_file():
            raise BenchmarkError(f"model file is missing: {config.model_path}")
        if not (
            Path(config.server_bin).is_file()
            if "/" in config.server_bin
            else shutil.which(config.server_bin)
        ):
            raise BenchmarkError(f"llama-server was not found: {config.server_bin}")
        config.benchmark_root.mkdir(parents=True, exist_ok=True)
        measurements: list[dict[str, Any]] = []
        capacity_failures: list[dict[str, Any]] = []
        largest_attempted_prompt = 0
        occupied_boundary_reached = False
        for capacity in config.capacities:
            prompt_targets = (
                []
                if occupied_boundary_reached
                else [
                    target
                    for target in config.prompt_targets
                    if target > largest_attempted_prompt
                ]
            )
            try:
                batch = run_capacity(config, capacity, prompt_targets)
            except BenchmarkError as error:
                if not measurements:
                    raise
                capacity_failures.append(
                    {
                        "configured_context_tokens": capacity,
                        "reason": str(error),
                        "measured_at": utc_now(),
                    }
                )
                log(f"CAPACITY BOUNDARY: capacity={capacity} reason={error}")
                break
            measurements.extend(batch)
            short = next(item for item in batch if item["scenario"] == "short")
            if not short["qualifies_for_minimum_tps"]:
                log(f"CAPACITY SPEED BOUNDARY: capacity={capacity}")
                break
            occupied = [
                item for item in batch if item["scenario"] == "tool_conversation"
            ]
            if occupied:
                largest_attempted_prompt = max(
                    item["target_prompt_tokens"] for item in occupied
                )
                occupied_boundary_reached = not occupied[-1][
                    "qualifies_for_minimum_tps"
                ]
            if capacity == 32768 and not any(
                item["target_prompt_tokens"] >= 8192
                and item["qualifies_for_minimum_tps"]
                for item in occupied
            ):
                log("OCCUPIED PROMPT BASELINE FAILED: stopping this candidate")
                break
        summary = summarize(config, measurements, capacity_failures)
        atomic_json(config.benchmark_root / "summary.json", summary)
        log(
            "SUMMARY: "
            f"largest_configured={summary['largest_passing_configured_context_tokens']} "
            f"largest_actual_prompt={summary['largest_passing_actual_prompt_tokens']} "
            f"minimum_requirement_met={summary['minimum_requirement_met']}"
        )
        return 0
    except KeyboardInterrupt:
        log("INTERRUPTED: benchmark stopped by operator")
        return 130
    except (BenchmarkError, OSError, subprocess.SubprocessError) as error:
        log(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
