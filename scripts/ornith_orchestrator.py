#!/usr/bin/env python3
"""Run the staged Ornith test funnel with one command.

The current champion is LFM2.5-1.2B-Instruct-QAD-Q4_0. An Ornith candidate
wins only if a confirmed run beats both champion speed measurements and the
80-question quality mean. The funnel rejects candidates as early as possible
to reduce heat, runtime, memory pressure, and storage use.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
QUALITY_MANIFEST = ROOT / "benchmarks/manifests/ornith-qad-models.tsv"
QUESTIONS = ROOT / "benchmarks/mt-bench/question.jsonl"

SCREEN_ROOT = ROOT / ".benchmark/ornith-screening"
FINAL_ROOT = ROOT / ".benchmark/throughput-server"
GGUF_WEIGHT_ROOT = ROOT / ".benchmark/throughput-models"
GGUF_QUALITY_ROOT = ROOT / ".benchmark/lfm-qad-results"
MLX_ROOT = ROOT / ".benchmark/mlx-results"

CHAMPION_SHORT_TPS = 117.06
CHAMPION_OCCUPIED_TPS = 101.30
CHAMPION_MEAN = 4.44
BASELINE_CONTEXT = 32768
MAXIMUM_CONTEXT = 65536

SCREEN_GENERATION_TOKENS = 64
SCREEN_REPETITIONS = 1
CONFIRM_GENERATION_TOKENS = 128
CONFIRM_REPETITIONS = 3
DEFAULT_COOLDOWN_SECONDS = 120

GGUF_MODELS = [
    ("ornith-9b-q4_k_m", "Ornith-1.5-9B-Q4_K_M"),
    ("ornith-9b-q5_k_m", "Ornith-1.5-9B-Q5_K_M"),
    ("ornith-9b-q6_k", "Ornith-1.5-9B-Q6_K"),
    ("ornith-9b-q8_0", "Ornith-1.5-9B-Q8_0"),
]

# These are the 9B variants that fit a 16 GB Mac. BF16 and 35B are excluded.
MLX_MODELS = [
    ("ornith-9b-mlx-4bit", "ornith-ai/Ornith-1.5-9B-MLX-4bit"),
    ("ornith-9b-mlx-6bit", "ornith-ai/Ornith-1.5-9B-MLX-6bit"),
    ("ornith-9b-mlx-8bit", "ornith-ai/Ornith-1.5-9B-MLX-8bit"),
]


class OrchestratorError(RuntimeError):
    """A recoverable operator-facing workflow failure."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def run(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
    check: bool = True,
) -> int:
    log(f"RUN: {' '.join(command)}")
    result = subprocess.run(command, env=environment, check=False)
    if check and result.returncode != 0:
        raise OrchestratorError(
            f"command failed with code {result.returncode}: {' '.join(command)}"
        )
    return result.returncode


def require_command(name: str, fallback: Path | None = None) -> str:
    found = shutil.which(name)
    if found:
        return found
    if fallback and fallback.is_file():
        return str(fallback)
    raise OrchestratorError(f"required command was not found: {name}")


def storage_check() -> None:
    checked: set[Path] = set()
    for path in (ROOT, Path.home() / ".cache/huggingface"):
        actual = path if path.exists() else ROOT
        actual = actual.resolve()
        if actual in checked:
            continue
        checked.add(actual)
        usage = shutil.disk_usage(actual)
        free_gib = usage.free / 1024**3
        free_percent = usage.free * 100 / usage.total
        log(
            f"STORAGE: path={actual} free_gib={free_gib:.1f} "
            f"free_percent={free_percent:.1f}"
        )
        if free_gib < 30 or free_percent < 15:
            raise OrchestratorError(
                f"storage is below the 30 GiB / 15% safety floor: {actual}"
            )


def cooldown(seconds: int, reason: str) -> None:
    if seconds <= 0:
        return
    log(f"COOLDOWN: {seconds}s after {reason}; all model servers are stopped")
    time.sleep(seconds)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as error:
        raise OrchestratorError(
            f"could not read result file {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise OrchestratorError(f"expected a JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return records
    except OSError as error:
        raise OrchestratorError(
            f"could not read result file {path}: {error}"
        ) from error
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise OrchestratorError(
                f"invalid JSON record in {path}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise OrchestratorError(f"expected JSON objects in {path}")
        records.append(value)
    return records


def speed_metrics(summary: dict[str, Any] | None) -> dict[str, Any]:
    measurements = summary.get("measurements", []) if summary else []
    short = next(
        (
            item
            for item in measurements
            if item.get("scenario") == "short"
            and item.get("configured_context_tokens") == BASELINE_CONTEXT
        ),
        None,
    )
    occupied = next(
        (
            item
            for item in measurements
            if item.get("scenario") == "tool_conversation"
            and item.get("configured_context_tokens") == BASELINE_CONTEXT
            and item.get("target_prompt_tokens", 0) >= 8192
        ),
        None,
    )
    short_tps = short.get("generation_tokens_per_second") if short else None
    occupied_tps = occupied.get("generation_tokens_per_second") if occupied else None
    return {
        "short_tps": short_tps,
        "occupied_tps": occupied_tps,
        "speed_win": (
            isinstance(short_tps, (int, float))
            and isinstance(occupied_tps, (int, float))
            and short_tps > CHAMPION_SHORT_TPS
            and occupied_tps > CHAMPION_OCCUPIED_TPS
        ),
        "maximum_context_tested": max(
            (
                item.get("configured_context_tokens", 0)
                for item in measurements
                if item.get("scenario") == "short"
            ),
            default=0,
        ),
    }


def quality_metrics(model_id: str, runtime: str) -> dict[str, Any]:
    root = GGUF_QUALITY_ROOT if runtime == "GGUF" else MLX_ROOT
    records = load_jsonl(root / model_id / "judgments.jsonl")
    scores = [
        record["score"]
        for record in records
        if isinstance(record.get("score"), int)
        and not isinstance(record.get("score"), bool)
    ]
    mean = statistics.fmean(scores) if scores else None
    complete = len(scores) == 80
    return {
        "judgments": len(scores),
        "mean": mean,
        "complete": complete,
        "quality_win": complete and mean is not None and mean > CHAMPION_MEAN,
    }


def gguf_environment(
    *,
    result_root: Path,
    capacities: str,
    prompt_targets: str,
    generation_tokens: int,
    repetitions: int,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "THROUGHPUT_HF_BIN": require_command("hf", ROOT / ".venv/bin/hf"),
            "LLAMA_SERVER_BENCHMARK_BIN": require_command("llama-server"),
            "THROUGHPUT_BENCHMARK_ROOT": str(result_root),
            "LLAMA_SERVER_BENCHMARK_CAPACITIES": capacities,
            "LLAMA_SERVER_BENCHMARK_PROMPT_TARGETS": prompt_targets,
            "LLAMA_SERVER_BENCHMARK_GENERATION_TOKENS": str(generation_tokens),
            "LLAMA_SERVER_BENCHMARK_REPETITIONS": str(repetitions),
        }
    )
    return environment


def run_gguf_throughput(
    model_id: str,
    *,
    screening: bool,
    cooldown_seconds: int,
) -> dict[str, Any]:
    result_root = SCREEN_ROOT if screening else FINAL_ROOT
    summary_path = result_root / model_id / "summary.json"
    before_mtime = summary_path.stat().st_mtime_ns if summary_path.exists() else None
    environment = gguf_environment(
        result_root=result_root,
        capacities="32768" if screening else "32768,65536",
        prompt_targets="8192" if screening else "8192,16384,32768,49152",
        generation_tokens=(
            SCREEN_GENERATION_TOKENS if screening else CONFIRM_GENERATION_TOKENS
        ),
        repetitions=SCREEN_REPETITIONS if screening else CONFIRM_REPETITIONS,
    )
    phase = "screen" if screening else "confirmation"
    log(f"GGUF {phase}: {model_id}")
    run(
        [
            sys.executable,
            str(ROOT / "scripts/throughput_context_search.py"),
            "--model",
            model_id,
        ],
        environment=environment,
    )
    summary = load_json(summary_path)
    if summary is None:
        raise OrchestratorError(f"GGUF {phase} summary is missing: {summary_path}")
    after_mtime = summary_path.stat().st_mtime_ns
    if before_mtime != after_mtime:
        cooldown(cooldown_seconds, f"GGUF {phase} {model_id}")
    metrics = speed_metrics(summary)
    log(
        f"GGUF {phase} result: model={model_id} "
        f"short={metrics['short_tps']} occupied={metrics['occupied_tps']} "
        f"speed_win={metrics['speed_win']}"
    )
    return metrics


def gguf_quality_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "LFM_CHAT_HF_BIN": require_command("hf", ROOT / ".venv/bin/hf"),
            "LFM_CHAT_LLAMA_SERVER_BIN": require_command("llama-server"),
            "LFM_CHAT_QUALITY_SOURCE_ROOT": str(GGUF_WEIGHT_ROOT),
            "LFM_CHAT_ANSWER_TEMPERATURE": "0.6",
            "LFM_CHAT_JUDGE_MODEL": "google/gemini-3.7-flash:batch",
        }
    )
    return environment


def run_gguf_quality(model_id: str, cooldown_seconds: int) -> dict[str, Any]:
    existing = quality_metrics(model_id, "GGUF")
    if existing["complete"]:
        log(f"GGUF quality resume: {model_id} already has 80 judgments")
        return existing
    environment = gguf_quality_environment()
    log(f"GGUF quality: {model_id}; running only because speed passed")
    run(
        [
            sys.executable,
            str(ROOT / "scripts/lfm_chat_benchmark.py"),
            "quality",
            "--manifest",
            str(QUALITY_MANIFEST),
            "--model",
            model_id,
        ],
        environment=environment,
    )
    metrics = quality_metrics(model_id, "GGUF")
    if not metrics["complete"]:
        raise OrchestratorError(
            f"GGUF quality is incomplete for {model_id}: "
            f"{metrics['judgments']}/80 judgments"
        )
    cooldown(cooldown_seconds, f"GGUF quality {model_id}")
    log(
        f"GGUF quality result: model={model_id} mean={metrics['mean']:.4f} "
        f"quality_win={metrics['quality_win']}"
    )
    return metrics


def stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def mlx_server_command(model: str, max_tokens: int | None = None) -> list[str]:
    entrypoint = shutil.which("mlx_lm.server")
    command = [entrypoint] if entrypoint else [sys.executable, "-m", "mlx_lm.server"]
    command.extend(["--model", model, "--host", "127.0.0.1", "--port", "8080"])
    if max_tokens is not None:
        command.extend(["--max-tokens", str(max_tokens)])
    return command


def wait_for_mlx_server(process: subprocess.Popen[bytes]) -> None:
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise OrchestratorError(
                f"mlx_lm.server exited during startup with code {process.returncode}"
            )
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/v1/models", timeout=2):
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise OrchestratorError("mlx_lm.server startup timed out")


def mlx_bench_metrics(path: Path) -> dict[str, Any]:
    records = load_jsonl(path) if path.suffix == ".jsonl" else []
    if not records:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            value = []
        except (OSError, json.JSONDecodeError) as error:
            raise OrchestratorError(
                f"could not read benchmark file {path}: {error}"
            ) from error
        if not isinstance(value, list) or any(
            not isinstance(row, dict) for row in value
        ):
            raise OrchestratorError(f"expected a JSON array of records in {path}")
        records = value
    short_speeds = [
        record["gen_tps"]
        for record in records
        if record.get("scenario") == "plain"
        and record.get("label") == "first-call"
        and isinstance(record.get("gen_tps"), (int, float))
    ]
    occupied_speeds = [
        record["gen_tps"]
        for record in records
        if record.get("scenario") == "occupied"
        and record.get("label") == "first-call"
        and isinstance(record.get("gen_tps"), (int, float))
    ]
    short_speed = statistics.median(short_speeds) if short_speeds else None
    occupied_speed = statistics.median(occupied_speeds) if occupied_speeds else None
    return {
        "short_tps": short_speed,
        "occupied_tps": occupied_speed,
        "speed_win": (
            short_speed is not None
            and occupied_speed is not None
            and short_speed > CHAMPION_SHORT_TPS
            and occupied_speed > CHAMPION_OCCUPIED_TPS
        ),
        "maximum_context_tested": BASELINE_CONTEXT
        if occupied_speed is not None
        else None,
    }


def run_mlx_throughput(
    model_id: str,
    model: str,
    *,
    screening: bool,
    confirm_trials: int,
    cooldown_seconds: int,
) -> dict[str, Any]:
    phase = "screen" if screening else "confirmation"
    output = MLX_ROOT / model_id / f"{phase}-bench_raw.json"
    existing = mlx_bench_metrics(output)
    if existing["short_tps"] is not None:
        log(f"MLX {phase} resume: {model_id}")
        return existing
    trials = 1 if screening else confirm_trials
    process: subprocess.Popen[bytes] | None = None
    raw = ROOT / "bench_raw.json"
    raw.unlink(missing_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "LLM_MODEL": model,
            "LLM_SERVER_URL": "http://127.0.0.1:8080/v1",
        }
    )
    with output.with_name(f"{phase}-server.log").open("wb") as server_log:
        try:
            log(f"MLX {phase}: {model_id}")
            process = subprocess.Popen(
                mlx_server_command(model),
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            wait_for_mlx_server(process)
            run(
                [
                    sys.executable,
                    str(ROOT / "bench.py"),
                    "--trials",
                    str(trials),
                    "--scenarios",
                    "plain",
                    "occupied",
                ],
                environment=environment,
            )
        finally:
            stop_process(process)
    if not raw.is_file():
        raise OrchestratorError(f"MLX {phase} did not produce {raw}")
    shutil.move(raw, output)
    cooldown(cooldown_seconds, f"MLX {phase} {model_id}")
    metrics = mlx_bench_metrics(output)
    log(
        f"MLX {phase} result: model={model_id} short={metrics['short_tps']} "
        f"speed_win={metrics['speed_win']}"
    )
    return metrics


def run_mlx_quality(
    model_id: str,
    model: str,
    cooldown_seconds: int,
) -> dict[str, Any]:
    existing = quality_metrics(model_id, "MLX")
    if existing["complete"]:
        log(f"MLX quality resume: {model_id} already has 80 judgments")
        return existing
    result_root = MLX_ROOT / model_id
    answers = result_root / "answers.jsonl"
    judgments = result_root / "judgments.jsonl"
    result_root.mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen[bytes] | None = None
    with (result_root / "quality-server.log").open("wb") as server_log:
        try:
            log(f"MLX quality: {model_id}; running only because speed passed")
            process = subprocess.Popen(
                mlx_server_command(model, 4096),
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            wait_for_mlx_server(process)
            run(
                [
                    sys.executable,
                    str(ROOT / "scripts/mt_bench_answers.py"),
                    "--model-id",
                    model_id,
                    "--request-model",
                    model,
                    "--questions",
                    str(QUESTIONS),
                    "--output",
                    str(answers),
                    "--server-url",
                    "http://127.0.0.1:8080/v1",
                    "--temperature",
                    "0.6",
                    "--max-tokens",
                    "4096",
                ]
            )
        finally:
            stop_process(process)
    run(
        [
            sys.executable,
            str(ROOT / "scripts/mt_bench_batch_judge.py"),
            "--answers",
            str(answers),
            "--questions",
            str(QUESTIONS),
            "--output",
            str(judgments),
            "--dotenv",
            str(ROOT / ".env"),
            "--batch-model",
            "google/gemini-3.7-flash:batch",
        ]
    )
    metrics = quality_metrics(model_id, "MLX")
    if not metrics["complete"]:
        raise OrchestratorError(
            f"MLX quality is incomplete for {model_id}: {metrics['judgments']}/80"
        )
    cooldown(cooldown_seconds, f"MLX quality {model_id}")
    return metrics


def gguf_weight_directory(model_directory: str) -> Path:
    return GGUF_WEIGHT_ROOT / model_directory


def mlx_cache_directory(model: str) -> Path:
    return (
        Path.home() / ".cache/huggingface/hub" / f"models--{model.replace('/', '--')}"
    )


def erase_weight(path: Path, model_id: str, keep_losers: bool) -> None:
    if keep_losers:
        log(f"WEIGHT KEEP: {model_id} because --keep-losers is set")
        return
    if not path.exists():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)
    log(f"WEIGHT ERASED: loser={model_id} path={path}")


def gguf_funnel(cooldown_seconds: int, keep_losers: bool) -> str | None:
    for model_id, model_directory in GGUF_MODELS:
        screen = run_gguf_throughput(
            model_id, screening=True, cooldown_seconds=cooldown_seconds
        )
        if not screen["speed_win"]:
            log(
                f"GGUF STOP: {model_id} failed speed; larger quants are expected "
                "to be slower, so quality and larger quants are skipped"
            )
            erase_weight(gguf_weight_directory(model_directory), model_id, keep_losers)
            return None
        quality = run_gguf_quality(model_id, cooldown_seconds)
        if not quality["quality_win"]:
            log(
                f"GGUF CONTINUE: {model_id} passed speed but failed quality; "
                "trying the next quant"
            )
            erase_weight(gguf_weight_directory(model_directory), model_id, keep_losers)
            continue
        confirmation = run_gguf_throughput(
            model_id, screening=False, cooldown_seconds=cooldown_seconds
        )
        if confirmation["speed_win"]:
            log(
                f"GGUF FINALIST: {model_id} confirmed at 32K and tested through "
                f"{MAXIMUM_CONTEXT} context"
            )
            return model_id
        log(f"GGUF STOP: {model_id} failed the three-sample speed confirmation")
        erase_weight(gguf_weight_directory(model_directory), model_id, keep_losers)
        return None
    return None


def mlx_available() -> bool:
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        return False
    return True


def mlx_funnel(
    cooldown_seconds: int,
    confirm_trials: int,
    keep_losers: bool,
) -> str | None:
    if not mlx_available():
        log("MLX SKIP: mlx_lm is not installed")
        return None
    for model_id, model in MLX_MODELS:
        screen = run_mlx_throughput(
            model_id,
            model,
            screening=True,
            confirm_trials=confirm_trials,
            cooldown_seconds=cooldown_seconds,
        )
        if not screen["speed_win"]:
            log(
                f"MLX STOP: {model_id} failed speed; larger quants are expected "
                "to be slower, so quality and larger quants are skipped"
            )
            erase_weight(mlx_cache_directory(model), model_id, keep_losers)
            return None
        quality = run_mlx_quality(model_id, model, cooldown_seconds)
        if not quality["quality_win"]:
            log(
                f"MLX CONTINUE: {model_id} passed speed but failed quality; "
                "trying the next quant"
            )
            erase_weight(mlx_cache_directory(model), model_id, keep_losers)
            continue
        confirmation = run_mlx_throughput(
            model_id,
            model,
            screening=False,
            confirm_trials=confirm_trials,
            cooldown_seconds=cooldown_seconds,
        )
        if confirmation["speed_win"]:
            log(f"MLX FINALIST: {model_id} confirmed")
            return model_id
        log(f"MLX STOP: {model_id} failed speed confirmation")
        erase_weight(mlx_cache_directory(model), model_id, keep_losers)
        return None
    return None


def report_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id, _ in GGUF_MODELS:
        final_summary = load_json(FINAL_ROOT / model_id / "summary.json")
        screen_summary = load_json(SCREEN_ROOT / model_id / "summary.json")
        confirmed = final_summary is not None
        speed = speed_metrics(final_summary or screen_summary)
        quality = quality_metrics(model_id, "GGUF")
        rows.append(
            {
                "id": model_id,
                "runtime": "GGUF",
                **speed,
                **quality,
                "confirmed": confirmed,
                "champion": confirmed and speed["speed_win"] and quality["quality_win"],
            }
        )
    for model_id, _ in MLX_MODELS:
        final_path = MLX_ROOT / model_id / "confirmation-bench_raw.json"
        screen_path = MLX_ROOT / model_id / "screen-bench_raw.json"
        confirmed = final_path.is_file()
        speed = mlx_bench_metrics(final_path if confirmed else screen_path)
        quality = quality_metrics(model_id, "MLX")
        rows.append(
            {
                "id": model_id,
                "runtime": "MLX",
                **speed,
                **quality,
                "confirmed": confirmed,
                "champion": confirmed and speed["speed_win"] and quality["quality_win"],
            }
        )
    return rows


def display_number(value: Any) -> str:
    return "-" if not isinstance(value, (int, float)) else f"{value:.2f}"


def write_report() -> Path:
    rows = report_rows()
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = ROOT / "docs/reports" / f"{date}-ornith-vs-lfm.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    champions = [row for row in rows if row["champion"]]
    lines = [
        f"# Ornith-1.5 vs LFM Champion - {date}",
        "",
        "## Executive Summary",
        "",
        (
            "Champion beaten by: "
            + ", ".join(f"`{row['id']}` ({row['runtime']})" for row in champions)
            if champions
            else "No confirmed Ornith candidate beats LFM on both speed and quality."
        ),
        "",
        (
            "The champion gate requires more than 117.06 tok/s for a short 32K "
            "run, more than 101.30 tok/s for an occupied 32K run, and a mean "
            "above 4.44 over 80 judgments. Screening uses one 64-token sample. "
            "A winner must repeat the speed test with three 128-token samples. "
            "GGUF finalists are also tested at the 65,536-token cap."
        ),
        "",
        "## Results",
        "",
        "| Model | Runtime | Short tok/s | Occupied tok/s | Quality mean | Judgments | Confirmed | 65,536 tested | Champion |",
        "|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['id']}` | {row['runtime']} | "
            f"{display_number(row['short_tps'])} | "
            f"{display_number(row['occupied_tps'])} | "
            f"{display_number(row['mean'])} | {row['judgments']} | "
            f"{'yes' if row['confirmed'] else 'no'} | "
            f"{'yes' if row['maximum_context_tested'] == MAXIMUM_CONTEXT else 'no'} | "
            f"{'YES' if row['champion'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Operational Findings",
            "",
            "- The funnel starts with Q4 or 4-bit and stops a runtime when the smallest quant fails speed.",
            "- Quality runs only after a speed pass. Larger quants run only when a faster quant needs more quality.",
            "- Model servers stop before each cooldown. RAM and GPU memory are released when the process stops.",
            "- Loser weight directories are erased by default. Screening data, confirmations, judgments, and logs remain.",
            f"- Maximum configured context is {MAXIMUM_CONTEXT} tokens.",
            "",
            "## Recommendation",
            "",
            (
                f"Use `{champions[0]['id']}`; it passed both champion gates."
                if champions
                else "Keep LFM2.5-1.2B-Instruct-QAD-Q4_0 as the champion."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"REPORT: {path}")
    return path


def check_judge_configuration() -> None:
    run(
        [
            sys.executable,
            str(ROOT / "scripts/mt_bench_batch_judge.py"),
            "--check-config",
            "--dotenv",
            str(ROOT / ".env"),
        ]
    )


def stage_a(skip_gguf: bool, skip_mlx: bool) -> None:
    log("STAGE A: environment, storage, and manifest checks")
    storage_check()
    if not skip_gguf:
        require_command("hf", ROOT / ".venv/bin/hf")
        require_command("llama-server")
        run(
            [
                sys.executable,
                str(ROOT / "scripts/throughput_context_search.py"),
                "--dry-run",
                "--family",
                "ornith-15-9b",
            ]
        )
        run(
            [
                sys.executable,
                str(ROOT / "scripts/lfm_chat_benchmark.py"),
                "quality",
                "--dry-run",
                "--manifest",
                str(QUALITY_MANIFEST),
            ],
            environment=gguf_quality_environment(),
        )
    if not skip_mlx and not mlx_available():
        log("MLX CHECK: mlx_lm is not installed; MLX will be skipped")
    check_judge_configuration()
    log(
        f"STAGE A COMPLETE: Q4-first funnel, 32K screen, {MAXIMUM_CONTEXT} cap, "
        "BF16 and 35B excluded"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the environment and funnel without downloads or model servers",
    )
    parser.add_argument("--skip-gguf", action="store_true")
    parser.add_argument("--skip-mlx", action="store_true")
    parser.add_argument(
        "--confirm-trials",
        type=int,
        default=10,
        help="MLX confirmation trials after a speed and quality pass",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=int,
        default=DEFAULT_COOLDOWN_SECONDS,
        help="idle time after active local-model phases",
    )
    parser.add_argument(
        "--keep-losers",
        action="store_true",
        help="keep loser weights; result files are always kept",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="build the report from existing results without running tests or cleanup",
    )
    args = parser.parse_args()
    if args.confirm_trials < 1:
        parser.error("--confirm-trials must be at least 1")
    if args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds cannot be negative")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.report_only:
            write_report()
            return 0
        log(
            "START: champion requires both speed and quality; screening rejects "
            "losers before expensive tests"
        )
        stage_a(args.skip_gguf, args.skip_mlx)
        if args.dry_run:
            log("DRY RUN COMPLETE: no downloads, model servers, cleanup, or report")
            return 0
        if not args.skip_gguf:
            gguf_funnel(args.cooldown_seconds, args.keep_losers)
        if not args.skip_mlx:
            mlx_funnel(
                args.cooldown_seconds,
                args.confirm_trials,
                args.keep_losers,
            )
        write_report()
        log("COMPLETE: preserved measurements and erased loser weights")
        return 0
    except KeyboardInterrupt:
        log(
            "INTERRUPTED: servers stopped by their phase handler; partial weights remain for resume"
        )
        return 130
    except (OSError, OrchestratorError, subprocess.SubprocessError) as error:
        log(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
