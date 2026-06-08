#!/usr/bin/env python3
"""Benchmark Gemma QAT llama.cpp server configurations."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ServerDefaults:
    alias: str = "gemma-qat"
    port: int = 8080
    startup_timeout_seconds: int = 600
    request_timeout_seconds: int = 3600
    readiness_interval_seconds: int = 5
    post_start_grace_seconds: int = 2
    poll_timeout_seconds: int = 5
    process_shutdown_timeout_seconds: int = 30
    process_exit_timeout_seconds: int = 30
    process_poll_interval_seconds: int = 1
    post_model_cleanup_grace_seconds: int = 10
    ctx_size: int = 131072
    real_chat_ctx_size: int = 49152
    reduced_ctx_size: int = 16384
    parallel: int = 1
    threads_6: int = 6
    threads_8: int = 8
    disabled_cache_ram: int = 0
    temperature: str = "0.8"
    top_p: str = "0.95"
    top_k: str = "64"
    log_level: int = 4
    reasoning_budget: int = 0
    chat_template_kwargs: str = '{"enable_thinking":false}'
    max_tokens: int = 900
    prompt: str = (
        "Classify this user message as one of: casual_chat, task_request, bug_report, "
        "or unknown. Then briefly explain the label: 'My local model is slow after I "
        "increase the context window.'"
    )
    ready_endpoint: str = "/v1/models"
    completion_endpoint: str = "/v1/chat/completions"
    host: str = "127.0.0.1"
    content_type_header: str = "Content-Type"
    json_content_type: str = "application/json"


@dataclass(frozen=True)
class BenchmarkPaths:
    output_dir: Path = Path("logs") / "llama-bench"
    benchmark_results: Path = Path("benchmarks") / "benchmark_results.md"
    optimization_memory: Path = Path("benchmarks") / "optimization_memory.md"
    run_model: Path = Path("run_model.sh")


@dataclass(frozen=True)
class ModelRepos:
    gemma_12b_q4: str = "unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL"
    gemma_e2b_q4: str = "unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL"
    gemma_e2b_q2: str = "unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q2_K_XL"


@dataclass(frozen=True)
class BenchmarkConfig:
    server: ServerDefaults = field(default_factory=ServerDefaults)
    paths: BenchmarkPaths = field(default_factory=BenchmarkPaths)
    models: ModelRepos = field(default_factory=ModelRepos)
    meaningful_improvement_ratio: float = 1.02
    success_status: str = "ok"
    failed_status: str = "failed"
    benchmark_header_columns: int = 9
    legacy_benchmark_header_columns: int = 8
    markdown_table_prefix: str = "| "
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"
    run_dir_timestamp_format: str = "%Y%m%d-%H%M%S"
    server_log_suffix: str = ".server.log"
    response_suffix: str = ".response.json"


CONFIG = BenchmarkConfig()


@dataclass(frozen=True)
class BenchmarkVariant:
    name: str
    model_repo: str
    ctx_size: int
    flash_attn: str | None = None
    parallel: int | None = None
    threads: int | None = None
    cache_ram: int | None = None
    temperature: str = CONFIG.server.temperature
    top_p: str = CONFIG.server.top_p
    top_k: str = CONFIG.server.top_k
    perf: bool = True
    chat_template_kwargs: str | None = CONFIG.server.chat_template_kwargs
    hypothesis: str = ""
    expected_improvement: str = ""


@dataclass(frozen=True)
class QualityPrompt:
    name: str
    prompt: str
    expected_label: str | None = None
    requires_json: bool = False


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    status: str
    response_path: str | None
    server_log_path: str
    timings: dict[str, Any] | None
    quality: list[dict[str, Any]]
    error: str | None


@dataclass(frozen=True)
class FailureClassification:
    code: str
    message: str


QUALITY_PROMPTS = [
    QualityPrompt(
        name="classification_bug_report",
        prompt=(
            "Classify this user message as one of: casual_chat, task_request, bug_report, "
            "or unknown. Return only JSON with keys label and rationale. Message: "
            "'After I increased the context window, my local model got much slower and sometimes "
            "the server exits during warmup.'"
        ),
        expected_label="bug_report",
        requires_json=True,
    ),
    QualityPrompt(
        name="classification_task_request",
        prompt=(
            "Classify this user message as one of: casual_chat, task_request, bug_report, "
            "or unknown. Return only JSON with keys label and rationale. Message: "
            "'Keep Gemma E2B as the only active local chat model.'"
        ),
        expected_label="task_request",
        requires_json=True,
    ),
    QualityPrompt(
        name="bug_diagnosis",
        prompt=(
            "A llama.cpp server on a 16 GB Mac mini M4 slows down after ctx-size is raised "
            "from 49K to 131K. Explain the most likely cause in five concise bullet points."
        ),
    ),
    QualityPrompt(
        name="tool_json",
        prompt=(
            "Return only valid JSON for a tool call plan with keys action, model, quant, and "
            "should_run_now. Use action='benchmark', model='Gemma-E2B', quant='UD-Q4_K_XL', "
            "and should_run_now=false."
        ),
        requires_json=True,
    ),
    QualityPrompt(
        name="no_thinking_leakage",
        prompt="Answer in one sentence without hidden reasoning: why benchmark one model at a time?",
    ),
]


VARIANTS = {
    "gemma-e2b-q4-49k": BenchmarkVariant(
        "gemma-e2b-q4-49k",
        model_repo=CONFIG.models.gemma_e2b_q4,
        ctx_size=CONFIG.server.real_chat_ctx_size,
        flash_attn="on",
        parallel=CONFIG.server.parallel,
        hypothesis="Current 49K real-chat baseline with Flash Attention and thinking disabled.",
        expected_improvement="Control run for speed, stability, and quality comparison.",
    ),
    "12b-q4-32k": BenchmarkVariant(
        "12b-q4-32k",
        model_repo=CONFIG.models.gemma_12b_q4,
        ctx_size=CONFIG.server.ctx_size,
        parallel=CONFIG.server.parallel,
        hypothesis="Current chat/classifier quality baseline using confirmed 12B QAT GGUF.",
        expected_improvement="Control run for comparing smaller chat/classifier candidates.",
    ),
    "e2b-q4-32k": BenchmarkVariant(
        "e2b-q4-32k",
        model_repo=CONFIG.models.gemma_e2b_q4,
        ctx_size=CONFIG.server.ctx_size,
        parallel=CONFIG.server.parallel,
        hypothesis="E2B Q4 should be much faster and fit easily on 16GB unified memory.",
        expected_improvement="Higher chat/classifier throughput with lower memory use than 12B.",
    ),
    "e2b-q2-32k": BenchmarkVariant(
        "e2b-q2-32k",
        model_repo=CONFIG.models.gemma_e2b_q2,
        ctx_size=CONFIG.server.ctx_size,
        parallel=CONFIG.server.parallel,
        hypothesis="E2B Q2 tests the maximum speed/minimum memory chat/classifier tradeoff.",
        expected_improvement="Fastest candidate, with possible classification accuracy degradation.",
    ),
    "baseline-32k": BenchmarkVariant(
        "baseline-32k",
        model_repo=CONFIG.models.gemma_12b_q4,
        ctx_size=CONFIG.server.ctx_size,
        hypothesis="Legacy original 12B command without explicit --parallel 1.",
        expected_improvement="Historical control run for comparison.",
    ),
    "flash-attn-32k": BenchmarkVariant(
        "flash-attn-32k",
        model_repo=CONFIG.models.gemma_12b_q4,
        ctx_size=CONFIG.server.ctx_size,
        flash_attn="on",
        hypothesis="Test Metal flash attention at the current best 32K context.",
        expected_improvement="Potential memory bandwidth improvement.",
    ),
    "flash-attn-16k": BenchmarkVariant(
        "flash-attn-16k",
        model_repo=CONFIG.models.gemma_12b_q4,
        ctx_size=CONFIG.server.reduced_ctx_size,
        flash_attn="on",
        hypothesis="Test whether smaller context helps with flash attention enabled.",
        expected_improvement="Potential lower KV/cache pressure.",
    ),
    "parallel-1-32k": BenchmarkVariant(
        "parallel-1-32k",
        model_repo=CONFIG.models.gemma_12b_q4,
        ctx_size=CONFIG.server.ctx_size,
        parallel=CONFIG.server.parallel,
        hypothesis="llama.cpp auto-used n_parallel=4, but opencode is single-user.",
        expected_improvement="Lower KV/cache overhead and more consistent single-user latency.",
    ),
    "parallel-1-threads-6-32k": BenchmarkVariant(
        "parallel-1-threads-6-32k",
        model_repo=CONFIG.models.gemma_12b_q4,
        ctx_size=CONFIG.server.ctx_size,
        parallel=CONFIG.server.parallel,
        threads=CONFIG.server.threads_6,
        hypothesis="After --parallel 1, test whether more CPU threads help CPU-side work.",
        expected_improvement="Slightly better prompt or generation throughput on M4.",
    ),
    "parallel-1-threads-8-32k": BenchmarkVariant(
        "parallel-1-threads-8-32k",
        model_repo=CONFIG.models.gemma_12b_q4,
        ctx_size=CONFIG.server.ctx_size,
        parallel=CONFIG.server.parallel,
        threads=CONFIG.server.threads_8,
        hypothesis="Compare heavier CPU thread usage against the 6-thread variant.",
        expected_improvement="Find whether additional M4 cores improve or hurt throughput.",
    ),
    "cache-ram-0-32k": BenchmarkVariant(
        "cache-ram-0-32k",
        model_repo=CONFIG.models.gemma_12b_q4,
        ctx_size=CONFIG.server.ctx_size,
        cache_ram=CONFIG.server.disabled_cache_ram,
        hypothesis="Prompt cache may add memory overhead for local single-user benchmark runs.",
        expected_improvement="Lower memory pressure without harming generation speed.",
    ),
}

COMPARISON_VARIANTS = [
    "gemma-e2b-q4-49k",
]


class BenchmarkError(RuntimeError):
    """Raised when benchmark setup or execution fails."""


def local_url(port: int, endpoint: str) -> str:
    return f"http://{CONFIG.server.host}:{port}{endpoint}"


def classify_server_failure(server_log_path: Path) -> FailureClassification | None:
    if not server_log_path.exists():
        return None
    log_text = server_log_path.read_text(encoding="utf-8", errors="replace")
    if "ggml-metal-device.cpp:901: not implemented" in log_text and "Asserting on type 35" in log_text:
        return FailureClassification(
            code="metal_unsupported_quant_type_35",
            message="llama.cpp Metal backend does not support this Q2 quant type on this build.",
        )
    if "kIOGPUCommandBufferCallbackErrorOutOfMemory" in log_text:
        return FailureClassification(
            code="metal_out_of_memory",
            message="Metal ran out of memory while loading or running the model.",
        )
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Gemma 4 QAT llama.cpp configurations.")
    parser.add_argument(
        "--only",
        choices=sorted(VARIANTS),
        action="append",
        help="Run only the named variant. Can be passed multiple times.",
    )
    parser.add_argument("--port", type=int, default=CONFIG.server.port, help="llama-server port.")
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=CONFIG.server.startup_timeout_seconds,
        help="Seconds to wait for llama-server readiness.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=CONFIG.server.request_timeout_seconds,
        help="Seconds to wait for a chat completion response.",
    )
    parser.add_argument(
        "--kill-existing",
        action="store_true",
        help="Terminate existing llama-server processes before running.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CONFIG.paths.output_dir,
        help="Directory where benchmark run logs are written.",
    )
    parser.add_argument(
        "--repeat",
        action="store_true",
        help="Allow variants already recorded in benchmarks/benchmark_results.md to run again.",
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Run the active 49K Gemma E2B baseline set.",
    )
    return parser.parse_args()


def configure_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("llama_bench")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "benchmark.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    return logger


def run_command(
    command: list[str], timeout: int = CONFIG.server.poll_timeout_seconds
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def combined_output(result: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())


def find_llama_server_processes() -> list[tuple[int, str]]:
    result = run_command(["ps", "-ax", "-o", "pid=,command="])
    processes: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or "llama-server" not in stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        if pid_text.isdigit():
            processes.append((int(pid_text), command))
    return processes


def kill_processes(processes: list[tuple[int, str]], logger: logging.Logger) -> None:
    for pid, command in processes:
        logger.warning("Terminating existing llama-server pid=%s command=%s", pid, command)
        try:
            subprocess.run(
                ["kill", "-TERM", str(pid)],
                check=False,
                timeout=CONFIG.server.poll_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise BenchmarkError(f"Timed out terminating existing llama-server pid={pid}") from exc
    deadline = time.monotonic() + CONFIG.server.process_exit_timeout_seconds
    while time.monotonic() < deadline:
        if not find_llama_server_processes():
            return
        time.sleep(CONFIG.server.process_poll_interval_seconds)
    raise BenchmarkError("Existing llama-server processes did not exit after SIGTERM")


def is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((CONFIG.server.host, port))
        except OSError:
            return False
    return True


def ensure_environment(args: argparse.Namespace, logger: logging.Logger) -> None:
    binary = shutil.which("llama-server")
    if binary is None:
        raise BenchmarkError("llama-server not found in PATH. Install with: brew install llama.cpp")
    logger.info("llama-server binary: %s", binary)

    version = run_command(["llama-server", "--version"])
    if version.returncode != 0:
        raise BenchmarkError(f"llama-server --version failed: {combined_output(version)}")
    logger.info("llama-server version: %s", combined_output(version).replace("\n", " | "))

    processes = find_llama_server_processes()
    if processes and not args.kill_existing:
        details = "; ".join(f"pid={pid} command={command}" for pid, command in processes)
        raise BenchmarkError(
            "Existing llama-server process found. Stop it first or rerun with "
            "`uv run python scripts/benchmark_gemma_qat.py --kill-existing`. "
            + details
        )
    if processes:
        kill_processes(processes, logger)

    if not is_port_available(args.port):
        raise BenchmarkError(f"Port {args.port} is already in use")
    logger.info("Port %s is available", args.port)


def build_server_command(variant: BenchmarkVariant, port: int | None) -> list[str]:
    command = [
        "llama-server",
        "-lv",
        str(CONFIG.server.log_level),
        "-hf",
        variant.model_repo,
        "--alias",
        CONFIG.server.alias,
        "--no-mmproj",
        "--reasoning",
        "off",
        "--reasoning-budget",
        str(CONFIG.server.reasoning_budget),
        "--temp",
        variant.temperature,
        "--top-p",
        variant.top_p,
        "--top-k",
        variant.top_k,
        "--ctx-size",
        str(variant.ctx_size),
    ]
    if variant.perf:
        command.append("--perf")
    if port is not None:
        command.extend(["--port", str(port)])
    command.extend(["--tools", "all"])
    if variant.chat_template_kwargs is not None:
        command.extend(["--chat-template-kwargs", variant.chat_template_kwargs])
    if variant.flash_attn is not None:
        command.extend(["--flash-attn", variant.flash_attn])
    if variant.parallel is not None:
        command.extend(["--parallel", str(variant.parallel)])
    if variant.threads is not None:
        command.extend(["--threads", str(variant.threads)])
    if variant.cache_ram is not None:
        command.extend(["--cache-ram", str(variant.cache_ram)])
    return command


def wait_for_server(port: int, process: subprocess.Popen[bytes], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    url = local_url(port, CONFIG.server.ready_endpoint)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise BenchmarkError(f"llama-server exited early with code {process.returncode}")
        try:
            with urlopen(
                Request(url, method="GET"), timeout=CONFIG.server.poll_timeout_seconds
            ) as response:
                if 200 <= response.status < 300:
                    return
        except (HTTPError, URLError, TimeoutError):
            time.sleep(CONFIG.server.readiness_interval_seconds)
    raise BenchmarkError(f"llama-server was not ready within {timeout} seconds")


def request_completion(port: int, timeout: int, prompt: str) -> dict[str, Any]:
    payload = {
        "model": CONFIG.server.alias,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": CONFIG.server.max_tokens,
    }
    request = Request(
        local_url(port, CONFIG.server.completion_endpoint),
        data=json.dumps(payload).encode("utf-8"),
        headers={CONFIG.server.content_type_header: CONFIG.server.json_content_type},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise BenchmarkError(f"Chat completion failed with HTTP {exc.code}: {error_body}") from exc
    except (URLError, TimeoutError) as exc:
        raise BenchmarkError(f"Chat completion request failed: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BenchmarkError("Chat completion returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise BenchmarkError("Chat completion response was not a JSON object")
    return parsed


def extract_timings(response: dict[str, Any]) -> dict[str, Any] | None:
    timings = response.get("timings")
    return timings if isinstance(timings, dict) else None


def response_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def score_quality(prompt: QualityPrompt, response: dict[str, Any]) -> dict[str, Any]:
    content = response_content(response)
    result: dict[str, Any] = {
        "prompt": prompt.name,
        "has_content": bool(content.strip()),
        "thinking_leaked": "<think>" in content.lower() or "</think>" in content.lower(),
        "json_valid": None,
        "label_correct": None,
    }
    parsed_json: Any = None
    if prompt.requires_json:
        try:
            parsed_json = json.loads(content)
            result["json_valid"] = isinstance(parsed_json, dict)
        except json.JSONDecodeError:
            result["json_valid"] = False
    if prompt.expected_label is not None and isinstance(parsed_json, dict):
        label = parsed_json.get("label")
        result["label_correct"] = label == prompt.expected_label
    return result


def terminate_started_server(process: subprocess.Popen[bytes], logger: logging.Logger) -> None:
    if process.poll() is not None:
        time.sleep(CONFIG.server.post_model_cleanup_grace_seconds)
        return
    logger.info("Stopping llama-server pid=%s", process.pid)
    process.terminate()
    try:
        process.wait(timeout=CONFIG.server.process_shutdown_timeout_seconds)
    except subprocess.TimeoutExpired:
        logger.warning("llama-server did not stop after SIGTERM; sending SIGKILL")
        process.kill()
        process.wait(timeout=CONFIG.server.process_shutdown_timeout_seconds)
    logger.info(
        "Waiting %s seconds for model memory cleanup",
        CONFIG.server.post_model_cleanup_grace_seconds,
    )
    time.sleep(CONFIG.server.post_model_cleanup_grace_seconds)


def run_variant(
    variant: BenchmarkVariant,
    args: argparse.Namespace,
    run_dir: Path,
    logger: logging.Logger,
) -> BenchmarkResult:
    server_log_path = run_dir / f"{variant.name}{CONFIG.server_log_suffix}"
    response_path = run_dir / f"{variant.name}.quality{CONFIG.response_suffix}"
    command = build_server_command(variant, args.port)
    logger.info("Starting variant=%s command=%s", variant.name, " ".join(command))
    with server_log_path.open("wb") as server_log:
        process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT)
        try:
            time.sleep(CONFIG.server.post_start_grace_seconds)
            wait_for_server(args.port, process, args.startup_timeout)
            logger.info("Server ready for variant=%s pid=%s", variant.name, process.pid)
            responses: list[dict[str, Any]] = []
            quality: list[dict[str, Any]] = []
            timings: dict[str, Any] | None = None
            for prompt in QUALITY_PROMPTS:
                response = request_completion(args.port, args.request_timeout, prompt.prompt)
                prompt_response_path = run_dir / f"{variant.name}.{prompt.name}{CONFIG.response_suffix}"
                prompt_response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
                responses.append(response)
                quality.append(score_quality(prompt, response))
                if timings is None:
                    timings = extract_timings(response)
            response_path.write_text(json.dumps(responses, indent=2), encoding="utf-8")
            logger.info(
                "Completed variant=%s first_prompt_timings=%s quality=%s",
                variant.name,
                json.dumps(timings),
                json.dumps(quality),
            )
            return BenchmarkResult(
                variant.name,
                CONFIG.success_status,
                str(response_path),
                str(server_log_path),
                timings,
                quality,
                None,
            )
        except BenchmarkError as exc:
            classification = classify_server_failure(server_log_path)
            error = str(exc)
            if classification is not None:
                error = f"{classification.code}: {classification.message}"
            logger.exception("Variant failed: %s", variant.name)
            return BenchmarkResult(
                variant.name,
                CONFIG.failed_status,
                str(response_path) if response_path.exists() else None,
                str(server_log_path),
                None,
                [],
                error,
            )
        finally:
            terminate_started_server(process, logger)


def timing_value(result: BenchmarkResult, key: str) -> float | None:
    value = None if result.timings is None else result.timings.get(key)
    return float(value) if isinstance(value, int | float) else None


def token_value(result: BenchmarkResult, key: str) -> int | None:
    value = None if result.timings is None else result.timings.get(key)
    return int(value) if isinstance(value, int | float) else None


def quality_score(result: BenchmarkResult) -> str:
    if not result.quality:
        return ""
    checks = 0
    passed = 0
    for prompt_result in result.quality:
        for key in ("has_content", "json_valid", "label_correct"):
            value = prompt_result.get(key)
            if value is None:
                continue
            checks += 1
            if value is True:
                passed += 1
        if prompt_result.get("thinking_leaked") is not None:
            checks += 1
            if prompt_result.get("thinking_leaked") is False:
                passed += 1
    return f"{passed}/{checks}" if checks else ""


def read_recorded_variants() -> set[str]:
    if not CONFIG.paths.benchmark_results.exists():
        return set()
    recorded: set[str] = set()
    for line in CONFIG.paths.benchmark_results.read_text(encoding="utf-8").splitlines():
        if not is_benchmark_row(line):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) >= 2:
            recorded.add(parts[1])
    return recorded


def is_benchmark_row(line: str) -> bool:
    return (
        line.startswith(CONFIG.markdown_table_prefix)
        and not line.startswith("| Timestamp")
        and not line.startswith("|---")
    )


def select_variants(args: argparse.Namespace, logger: logging.Logger) -> list[str]:
    requested = args.only or (COMPARISON_VARIANTS if args.comparison else list(VARIANTS))
    if args.repeat:
        return requested
    recorded = read_recorded_variants()
    selected = [name for name in requested if name not in recorded]
    skipped = [name for name in requested if name in recorded]
    if skipped:
        logger.info("Skipping already-recorded variants: %s", ", ".join(skipped))
    if not selected:
        raise BenchmarkError("No new variants to run. Pass --repeat to rerun recorded experiments.")
    return selected


def unrecorded_variants() -> list[str]:
    recorded = read_recorded_variants()
    return [name for name in VARIANTS if name not in recorded]


def best_recorded_speed() -> float | None:
    if not CONFIG.paths.benchmark_results.exists():
        return None
    best: float | None = None
    for line in CONFIG.paths.benchmark_results.read_text(encoding="utf-8").splitlines():
        if not is_benchmark_row(line):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) >= CONFIG.benchmark_header_columns:
            status = parts[3]
            speed_text = parts[4]
        elif len(parts) >= CONFIG.legacy_benchmark_header_columns:
            status = parts[2]
            speed_text = parts[3]
        else:
            continue
        if status != CONFIG.success_status:
            continue
        try:
            speed = float(speed_text)
        except ValueError:
            logging.getLogger("llama_bench").debug("Skipping row with non-numeric speed: %s", line)
            continue
        best = speed if best is None else max(best, speed)
    return best


def write_summary(results: list[BenchmarkResult], run_dir: Path, logger: logging.Logger) -> None:
    summary_json = run_dir / "summary.json"
    summary_md = run_dir / "summary.md"
    summary_json.write_text(json.dumps([result.__dict__ for result in results], indent=2), encoding="utf-8")
    lines = [
        "# Local Model llama.cpp Benchmark",
        "",
        "| Variant | Model | Status | Gen tok/s | Prompt tok/s | Quality | Error |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for result in results:
        gen_speed = timing_value(result, "predicted_per_second")
        prompt_speed = timing_value(result, "prompt_per_second")
        lines.append(
            f"| {result.name} | {VARIANTS[result.name].model_repo} | {result.status} | "
            f"{gen_speed if gen_speed is not None else ''} | "
            f"{prompt_speed if prompt_speed is not None else ''} | "
            f"{quality_score(result)} | {result.error or ''} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote summary: %s", summary_md)


def append_benchmark_results(
    results: list[BenchmarkResult],
    timestamp: str,
    port: int,
    logger: logging.Logger,
) -> None:
    if not CONFIG.paths.benchmark_results.exists():
        CONFIG.paths.benchmark_results.write_text(
            "# Benchmark Results\n\n"
            "| Timestamp | Variant | Model | Status | Gen tok/s | Prompt tok/s | Prompt tokens | Completion tokens | Command |\n"
            "|---|---|---|---|---:|---:|---:|---:|---|\n",
            encoding="utf-8",
        )
    lines: list[str] = []
    for result in results:
        command = " ".join(build_server_command(VARIANTS[result.name], port))
        gen_speed = timing_value(result, "predicted_per_second")
        prompt_speed = timing_value(result, "prompt_per_second")
        prompt_tokens = token_value(result, "prompt_n")
        completion_tokens = token_value(result, "predicted_n")
        gen_speed_text = f"{gen_speed:.4f}" if gen_speed is not None else ""
        prompt_speed_text = f"{prompt_speed:.4f}" if prompt_speed is not None else ""
        lines.append(
            f"| {timestamp} | {result.name} | {VARIANTS[result.name].model_repo} | "
            f"{result.status} | {gen_speed_text} | {prompt_speed_text} | "
            f"{prompt_tokens or ''} | {completion_tokens or ''} | `{command}` |"
        )
    with CONFIG.paths.benchmark_results.open("a", encoding="utf-8") as benchmark_file:
        benchmark_file.write("\n" + "\n".join(lines) + "\n")
    logger.info("Appended benchmark results: %s", CONFIG.paths.benchmark_results)


def best_successful_result(results: list[BenchmarkResult]) -> BenchmarkResult | None:
    successful = [
        result
        for result in results
        if result.status == CONFIG.success_status
        and timing_value(result, "predicted_per_second") is not None
    ]
    if not successful:
        return None
    return max(successful, key=lambda result: timing_value(result, "predicted_per_second") or 0.0)


def shell_command(command: list[str]) -> str:
    return " \\\n+  ".join(command)


def write_run_model(variant: BenchmarkVariant, logger: logging.Logger) -> None:
    command = " ".join(build_server_command(variant, port=None))
    content = f"#!/usr/bin/env bash\nset -euo pipefail\n\n{command}\n"
    CONFIG.paths.run_model.write_text(content, encoding="utf-8")
    CONFIG.paths.run_model.chmod(0o755)
    logger.info("Updated run model command: %s", CONFIG.paths.run_model)


def update_optimization_memory(
    results: list[BenchmarkResult],
    previous_best: float | None,
    logger: logging.Logger,
) -> None:
    best_run = best_successful_result(results)
    best_speed = timing_value(best_run, "predicted_per_second") if best_run is not None else None
    improved = (
        best_run is not None
        and best_speed is not None
        and previous_best is not None
        and best_speed >= previous_best * CONFIG.meaningful_improvement_ratio
    )
    if improved:
        write_run_model(VARIANTS[best_run.name], logger)

    lines = ["# Optimization Memory", "", "## Best Configuration", ""]
    if improved and best_run is not None and best_speed is not None:
        lines.append(f"Current best updated by this run: `{best_run.name}` at {best_speed:.4f} tok/s.")
    elif previous_best is not None:
        lines.append(f"Current best remains the previously recorded configuration at {previous_best:.4f} tok/s.")
        lines.append("No latest run improved generation speed by at least 2%.")
    else:
        lines.append("No successful baseline has been recorded yet.")

    lines.extend(["", "## Latest Run Patterns", ""])
    for result in results:
        if result.status != CONFIG.success_status:
            continue
        variant = VARIANTS[result.name]
        gen_speed = timing_value(result, "predicted_per_second")
        prompt_speed = timing_value(result, "prompt_per_second")
        lines.append(
            f"- `{result.name}` ({variant.model_repo}): "
            f"{gen_speed:.4f} gen tok/s, {prompt_speed:.4f} prompt tok/s. "
            f"Hypothesis: {variant.hypothesis}"
        )

    lines.extend(["", "## Failed Or Lower-Performing Configurations", ""])
    failed = [result for result in results if result.status != CONFIG.success_status]
    if failed:
        lines.extend(f"- `{result.name}` failed: {result.error}" for result in failed)
    else:
        lines.append("- No failures in the latest run.")

    lines.extend(["", "## Next Hypothesis", ""])
    if improved:
        lines.append(
            "Compare output quality on representative chat and classification prompts before "
            "promoting a speed-only winner."
        )
    elif not unrecorded_variants():
        lines.append(
            "No unrecorded variants remain. Use E2B Q4 as the active local chat model. "
            "Q2 variants failed on Metal."
        )
    else:
        remaining = ", ".join(unrecorded_variants())
        lines.append(
            "Continue with the next unrecorded chat/classifier variant nearest to the current "
            f"fastest stable configuration. Remaining: {remaining}."
        )
    CONFIG.paths.optimization_memory.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Updated optimization memory: %s", CONFIG.paths.optimization_memory)


def main() -> int:
    args = parse_args()
    now = datetime.now()
    timestamp = now.strftime(CONFIG.timestamp_format)
    run_dir = args.output_dir / now.strftime(CONFIG.run_dir_timestamp_format)
    logger = configure_logging(run_dir)
    logger.info("Benchmark run directory: %s", run_dir)

    try:
        ensure_environment(args, logger)
        previous_best = best_recorded_speed()
        names = select_variants(args, logger)
        for name in names:
            variant = VARIANTS[name]
            logger.info("Experiment reason for %s: %s", name, variant.hypothesis)
            logger.info("Expected improvement for %s: %s", name, variant.expected_improvement)
        results = [run_variant(VARIANTS[name], args, run_dir, logger) for name in names]
        write_summary(results, run_dir, logger)
        append_benchmark_results(results, timestamp, args.port, logger)
        update_optimization_memory(results, previous_best, logger)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except (BenchmarkError, OSError, subprocess.SubprocessError) as exc:
        logger.exception("Benchmark failed")
        logger.error("Error: %s", exc)
        return 1

    failed = [result for result in results if result.status != CONFIG.success_status]
    if failed:
        logger.error("%s variant(s) failed", len(failed))
        return 1
    logger.info("All benchmark variants completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
