#!/usr/bin/env python3
"""Benchmark GGUF generation speed at a populated KV-cache depth."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

SCRIPT_VERSION = "3"
MEASUREMENT_METHOD = "generation_at_kv_depth"
ROOT = Path(__file__).resolve().parent
OOM_PATTERN = re.compile(
    r"out of memory|failed to allocate|cannot allocate|memory allocation|killed",
    re.IGNORECASE,
)


class BenchmarkError(RuntimeError):
    """A configuration or operational error that must stop the workflow."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def env_int(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise BenchmarkError(
            f"{name} must be an integer >= {minimum} (got '{raw}')"
        ) from error
    if value < minimum:
        raise BenchmarkError(f"{name} must be an integer >= {minimum} (got '{raw}')")
    return value


def logical_cpus() -> int:
    return os.cpu_count() or 4


@dataclass(frozen=True)
class Config:
    source_root: Path
    local_root: Path
    manifest: Path
    benchmark_root: Path
    llama_bench: str
    max_attempts: int
    retry_delay: int
    repetitions: int
    generation_tokens: int
    minimum_tps: int
    minimum_context: int
    search_step: int
    case_timeout: int
    setup_timeout: int
    fit_target_mib: int
    fit_context_tokens: int
    batch_size: int
    ubatch_size: int
    gpu_layers: int
    cache_type_k: str
    cache_type_v: str
    flash_attention: str
    memory_sample_seconds: int
    cleanup_wait_seconds: int
    stage_timeout: int
    threads: int

    @classmethod
    def from_environment(cls) -> Config:
        home = Path.home()
        gpu_layers = env_int("LLAMA_BENCHMARK_GPU_LAYERS", -1, -1)
        return cls(
            source_root=Path(
                os.environ.get(
                    "LLAMA_BENCHMARK_SOURCE_MODEL_ROOT",
                    os.environ.get(
                        "TOURO_MODEL_DIR", "/Volumes/Touro/local-llm-models"
                    ),
                )
            ).expanduser(),
            local_root=Path(
                os.environ.get(
                    "LLAMA_BENCHMARK_LOCAL_MODEL_ROOT",
                    home / "Library/Caches/local-llm-benchmark/models",
                )
            ).expanduser(),
            manifest=Path(
                os.environ.get(
                    "LLAMA_BENCHMARK_MANIFEST",
                    ROOT.parent
                    / "benchmarks"
                    / "manifests"
                    / "llama-benchmark-models.tsv",
                )
            ).expanduser(),
            benchmark_root=Path(
                os.environ.get(
                    "LLAMA_BENCHMARK_ROOT",
                    home / "Library/Application Support/local-llm-benchmark",
                )
            ).expanduser(),
            llama_bench=os.environ.get("LLAMA_BENCHMARK_BIN", "llama-bench"),
            max_attempts=env_int("LLAMA_BENCHMARK_ATTEMPTS", 2, 1),
            retry_delay=env_int("LLAMA_BENCHMARK_RETRY_DELAY_SECONDS", 10, 0),
            repetitions=env_int("LLAMA_BENCHMARK_REPETITIONS", 3, 1),
            generation_tokens=env_int("LLAMA_BENCHMARK_GENERATION_TOKENS", 128, 1),
            minimum_tps=env_int("LLAMA_BENCHMARK_MIN_GENERATION_TPS", 50, 1),
            minimum_context=env_int("LLAMA_BENCHMARK_MIN_CONTEXT_TOKENS", 512, 1),
            search_step=env_int("LLAMA_BENCHMARK_CONTEXT_SEARCH_STEP", 2048, 256),
            case_timeout=env_int("LLAMA_BENCHMARK_CASE_TIMEOUT_SECONDS", 600, 1),
            setup_timeout=env_int("LLAMA_BENCHMARK_SETUP_TIMEOUT_SECONDS", 30, 1),
            fit_target_mib=env_int("LLAMA_BENCHMARK_FIT_TARGET_MIB", 2048, 1),
            fit_context_tokens=env_int("LLAMA_BENCHMARK_FIT_CONTEXT_TOKENS", 4096, 1),
            batch_size=env_int("LLAMA_BENCHMARK_BATCH_SIZE", 512, 1),
            ubatch_size=env_int("LLAMA_BENCHMARK_UBATCH_SIZE", 256, 1),
            gpu_layers=gpu_layers,
            cache_type_k=os.environ.get("LLAMA_BENCHMARK_CACHE_TYPE_K", "q8_0"),
            cache_type_v=os.environ.get("LLAMA_BENCHMARK_CACHE_TYPE_V", "q8_0"),
            flash_attention=os.environ.get("LLAMA_BENCHMARK_FLASH_ATTN", "auto"),
            memory_sample_seconds=env_int(
                "LLAMA_BENCHMARK_MEMORY_SAMPLE_SECONDS", 1, 1
            ),
            cleanup_wait_seconds=env_int("LLAMA_BENCHMARK_CLEANUP_WAIT_SECONDS", 3, 0),
            stage_timeout=env_int("LLAMA_BENCHMARK_STAGE_TIMEOUT_SECONDS", 3600, 1),
            threads=env_int("LLAMA_BENCHMARK_THREADS", logical_cpus(), 1),
        )


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    relative_path: Path
    maximum_context: int


@dataclass(frozen=True)
class CaseOutcome:
    status: str
    qualifies: int
    generation_tps: float | None
    reason: str


class Logger:
    def __init__(self) -> None:
        self.path: Path | None = None
        self._lock = threading.Lock()

    def attach(self, path: Path) -> None:
        self.path = path

    def log(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        with self._lock:
            print(line, flush=True)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")

    def raw(self, line: str) -> None:
        with self._lock:
            print(line, end="", file=sys.stderr, flush=True)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)


def atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=False) + "\n")


def state_text(values: dict[str, object]) -> str:
    return "".join(
        f"{key}={str(value).replace(chr(9), ' ').replace(chr(10), ' ')}\n"
        for key, value in values.items()
    )


def read_state(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def command_exists(command: str) -> bool:
    return (
        Path(command).is_file() if "/" in command else shutil.which(command) is not None
    )


def process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_process_group(process: subprocess.Popen[str], logger: Logger) -> None:
    if process.poll() is not None:
        return
    logger.log(f"CLEANUP: terminating benchmark process {process.pid}.")
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=2)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


class BenchmarkRunner:
    def __init__(
        self, config: Config, args: argparse.Namespace, logger: Logger
    ) -> None:
        self.config = config
        self.args = args
        self.logger = logger
        self.run_id = (
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
        )
        self.state_dir = config.benchmark_root / "state"
        self.results_dir = config.benchmark_root / "results"
        self.raw_dir = config.benchmark_root / "raw"
        self.runs_dir = config.benchmark_root / "runs"
        self.lock_dir = self.state_dir / ".lock"
        self.log_file = self.runs_dir / f"{self.run_id}.log"
        self.run_state_file = self.state_dir / "run.state"
        self.lock_acquired = False
        self.run_status = "starting"
        self.current_case: tuple[str, str, int] | None = None
        self.active_process: subprocess.Popen[str] | None = None

    def measurement_config(self) -> dict[str, object]:
        return {
            "generation_tokens": self.config.generation_tokens,
            "minimum_generation_tps": self.config.minimum_tps,
            "repetitions": self.config.repetitions,
            "fit_target_mib": self.config.fit_target_mib,
            "fit_context_tokens": self.config.fit_context_tokens,
            "batch_size": self.config.batch_size,
            "ubatch_size": self.config.ubatch_size,
            "threads": self.config.threads,
            "gpu_layers": self.config.gpu_layers,
            "cache_type_k": self.config.cache_type_k,
            "cache_type_v": self.config.cache_type_v,
            "flash_attention": self.config.flash_attention,
        }

    def load_manifest(self) -> list[ModelSpec]:
        if not self.config.manifest.is_file():
            raise BenchmarkError(
                f"model manifest is not readable: {self.config.manifest}"
            )
        models: list[ModelSpec] = []
        with self.config.manifest.open(encoding="utf-8", newline="") as handle:
            for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
                if not row or not row[0] or row[0].startswith("#"):
                    continue
                if len(row) != 3:
                    raise BenchmarkError(
                        f"manifest line {line_number} must contain three columns"
                    )
                model_id, relative_raw, maximum_raw = row
                if not re.fullmatch(r"[a-z0-9._-]+", model_id):
                    raise BenchmarkError(f"invalid model ID in manifest: {model_id}")
                relative_path = Path(relative_raw)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise BenchmarkError(
                        f"manifest path must be relative and cannot contain '..': {relative_raw}"
                    )
                try:
                    maximum = int(maximum_raw)
                except ValueError as error:
                    raise BenchmarkError(
                        f"invalid maximum context for {model_id}"
                    ) from error
                if (
                    maximum < self.config.minimum_context
                    or maximum <= self.config.generation_tokens
                ):
                    raise BenchmarkError(
                        f"maximum context for {model_id} must be at least "
                        f"{self.config.minimum_context} and exceed generated tokens"
                    )
                source = self.config.source_root / relative_path
                if not source.is_file():
                    raise BenchmarkError(f"model file is missing: {source}")
                models.append(ModelSpec(model_id, relative_path, maximum))
        if not models:
            raise BenchmarkError("model manifest contains no models")
        if self.args.model and self.args.model not in {
            model.model_id for model in models
        }:
            raise BenchmarkError(f"model ID is not in the manifest: {self.args.model}")
        return models

    def selected(self, models: list[ModelSpec]) -> list[ModelSpec]:
        return [
            model
            for model in models
            if not self.args.model or model.model_id == self.args.model
        ]

    def context_ladder(self, maximum: int) -> list[int]:
        contexts: list[int] = []
        context = self.config.minimum_context
        while context < maximum:
            contexts.append(context)
            context *= 2
        contexts.append(maximum)
        return contexts

    def create_directories(self) -> None:
        directories = (
            self.state_dir,
            self.results_dir,
            self.raw_dir,
            self.runs_dir,
            self.config.local_root,
        )
        started = time.monotonic()
        for path in directories:
            path.mkdir(parents=True, exist_ok=True)
        if time.monotonic() - started > self.config.setup_timeout:
            raise BenchmarkError("local benchmark directory setup exceeded its timeout")
        self.log_file.touch()
        self.logger.attach(self.log_file)

    def acquire_lock(self) -> None:
        try:
            self.lock_dir.mkdir()
        except FileExistsError:
            existing_pid = 0
            pid_file = self.lock_dir / "pid"
            if pid_file.is_file():
                try:
                    existing_pid = int(pid_file.read_text(encoding="utf-8").strip())
                except ValueError:
                    existing_pid = 0
            if existing_pid and process_is_alive(existing_pid):
                raise BenchmarkError(
                    f"another benchmark is running (pid {existing_pid})"
                )
            self.logger.log("Removing stale benchmark lock.")
            shutil.rmtree(self.lock_dir)
            self.lock_dir.mkdir()
        atomic_write_text(self.lock_dir / "pid", f"{os.getpid()}\n")
        atomic_write_text(self.lock_dir / "run_id", f"{self.run_id}\n")
        self.lock_acquired = True

    def release_lock(self) -> None:
        if self.lock_acquired:
            shutil.rmtree(self.lock_dir, ignore_errors=True)
            self.lock_acquired = False

    def write_run_state(self, status: str, message: str) -> None:
        atomic_write_text(
            self.run_state_file,
            state_text(
                {
                    "version": SCRIPT_VERSION,
                    "run_id": self.run_id,
                    "status": status,
                    "model_root": self.config.local_root,
                    "source_model_root": self.config.source_root,
                    "local_model_root": self.config.local_root,
                    "benchmark_root": self.config.benchmark_root,
                    "log_file": self.log_file,
                    "updated_at": utc_now(),
                    "message": message,
                }
            ),
        )
        self.run_status = status

    def write_case_state(
        self,
        case_key: str,
        model_id: str,
        context: int,
        status: str,
        attempt: int,
        exit_code: int,
        message: str,
    ) -> None:
        atomic_write_text(
            self.state_dir / f"{case_key}.state",
            state_text(
                {
                    "version": SCRIPT_VERSION,
                    "run_id": self.run_id,
                    "case": case_key,
                    "model_id": model_id,
                    "context_tokens": context,
                    "status": status,
                    "attempt": attempt,
                    "exit_code": exit_code,
                    "updated_at": utc_now(),
                    "message": message,
                }
            ),
        )

    def stage_model(self, model: ModelSpec) -> Path:
        source = self.config.source_root / model.relative_path
        destination = self.config.local_root / model.relative_path
        if source == destination:
            self.logger.log(
                f"STAGE SKIP: {model.model_id} source and local model paths are identical"
            )
            return destination
        if (
            destination.is_file()
            and destination.stat().st_size == source.stat().st_size
        ):
            self.logger.log(
                f"STAGE SKIP: {model.model_id} local copy is complete "
                f"({destination.stat().st_size} bytes)"
            )
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f"{destination.name}.part.{os.getpid()}")
        temporary.unlink(missing_ok=True)
        self.logger.log(f"STAGE: copying {model.model_id} from source to local storage")
        try:
            subprocess.run(
                ["cp", "-p", str(source), str(temporary)],
                check=True,
                timeout=self.config.stage_timeout,
            )
        except subprocess.TimeoutExpired as error:
            temporary.unlink(missing_ok=True)
            raise BenchmarkError(
                f"staging {model.model_id} exceeded {self.config.stage_timeout}s"
            ) from error
        except subprocess.CalledProcessError as error:
            temporary.unlink(missing_ok=True)
            raise BenchmarkError(
                f"could not stage {model.model_id} from {source}"
            ) from error
        if temporary.stat().st_size != source.stat().st_size:
            temporary.unlink(missing_ok=True)
            raise BenchmarkError(f"staged model size mismatch for {model.model_id}")
        temporary.replace(destination)
        self.logger.log(
            f"STAGE COMPLETE: {model.model_id} local_size={destination.stat().st_size} bytes "
            f"path={destination}"
        )
        return destination

    def capture_memory(self, path: Path) -> None:
        content = f"timestamp={utc_now()}\n"
        if shutil.which("memory_pressure"):
            result = subprocess.run(
                ["memory_pressure", "-Q"], capture_output=True, text=True, check=False
            )
            content += result.stdout + result.stderr
        path.write_text(content, encoding="utf-8")

    def benchmark_command(self, model_path: Path, context: int) -> list[str]:
        depth = context - self.config.generation_tokens
        return [
            self.config.llama_bench,
            "--model",
            str(model_path),
            "--n-prompt",
            "0",
            "--n-gen",
            str(self.config.generation_tokens),
            "--n-depth",
            str(depth),
            "--repetitions",
            str(self.config.repetitions),
            "--output",
            "json",
            "--progress",
            "--fit-target",
            str(self.config.fit_target_mib),
            "--fit-ctx",
            str(self.config.fit_context_tokens),
            "--batch-size",
            str(self.config.batch_size),
            "--ubatch-size",
            str(self.config.ubatch_size),
            "--threads",
            str(self.config.threads),
            "--n-gpu-layers",
            str(self.config.gpu_layers),
            "--cache-type-k",
            self.config.cache_type_k,
            "--cache-type-v",
            self.config.cache_type_v,
            "--flash-attn",
            self.config.flash_attention,
        ]

    def peak_rss_kib(self, pid: int) -> int:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            return int(result.stdout.strip())
        except ValueError:
            return 0

    def stream_stderr(self, stream: TextIO, destination: TextIO) -> None:
        for line in stream:
            destination.write(line)
            destination.flush()
            self.logger.raw(line)

    def run_process(
        self, command: list[str], stdout_path: Path, stderr_path: Path
    ) -> tuple[int, int]:
        peak_rss = 0
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_handle,
            stderr_path.open("w", encoding="utf-8") as stderr_handle,
        ):
            process = subprocess.Popen(
                command,
                stdout=stdout_handle,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            self.active_process = process
            assert process.stderr is not None
            stderr_thread = threading.Thread(
                target=self.stream_stderr,
                args=(process.stderr, stderr_handle),
                daemon=True,
            )
            stderr_thread.start()
            deadline = time.monotonic() + self.config.case_timeout
            while process.poll() is None:
                peak_rss = max(peak_rss, self.peak_rss_kib(process.pid))
                if time.monotonic() >= deadline:
                    self.logger.log(
                        f"TIMEOUT: stopping benchmark process after {self.config.case_timeout}s."
                    )
                    terminate_process_group(process, self.logger)
                    stderr_thread.join()
                    self.active_process = None
                    return 124, peak_rss
                time.sleep(self.config.memory_sample_seconds)
            stderr_thread.join()
            self.active_process = None
            return process.returncode, peak_rss

    def parse_output(
        self, path: Path, context: int
    ) -> tuple[float, float, list[float]] | None:
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(records, list):
            return None
        depth = context - self.config.generation_tokens
        matches = [
            record
            for record in records
            if isinstance(record, dict)
            and record.get("n_prompt") == 0
            and record.get("n_depth") == depth
            and record.get("n_gen") == self.config.generation_tokens
        ]
        if len(matches) != 1:
            return None
        record = matches[0]
        average = record.get("avg_ts")
        stddev = record.get("stddev_ts")
        samples = record.get("samples_ts")
        if (
            not isinstance(average, (int, float))
            or isinstance(average, bool)
            or not isinstance(stddev, (int, float))
            or isinstance(stddev, bool)
            or not isinstance(samples, list)
            or not samples
            or any(
                not isinstance(value, (int, float)) or isinstance(value, bool)
                for value in samples
            )
        ):
            return None
        return float(average), float(stddev), [float(value) for value in samples]

    def write_case_result(
        self,
        model: ModelSpec,
        model_path: Path,
        context: int,
        status: str,
        attempt: int,
        exit_code: int,
        qualifies: int,
        reason: str,
        peak_rss: int,
        stdout_path: Path,
        stderr_path: Path,
        measurement: tuple[float, float, list[float]] | None,
    ) -> None:
        average, stddev, samples = measurement or (None, None, None)
        case_key = f"{model.model_id}-ctx-{context}"
        atomic_write_json(
            self.results_dir / f"{case_key}.json",
            {
                "version": SCRIPT_VERSION,
                "run_id": self.run_id,
                "case": case_key,
                "model_id": model.model_id,
                "model_path": str(model_path),
                "backend": "llama.cpp",
                "measurement_method": MEASUREMENT_METHOD,
                "measurement_config": self.measurement_config(),
                "status": status,
                "reason": reason,
                "requested_context_tokens": context,
                "actual_context_tokens": context if measurement else None,
                "generation_tokens_per_second": average,
                "generation_stddev_tokens_per_second": stddev,
                "generation_samples_tokens_per_second": samples,
                "kv_depth_tokens": context - self.config.generation_tokens,
                "generated_tokens": self.config.generation_tokens,
                "prompt_tokens_per_second": None,
                "qualifies_for_50_tps": qualifies,
                "attempt": attempt,
                "exit_code": exit_code,
                "peak_rss_kib": peak_rss,
                "raw_stdout": str(stdout_path),
                "raw_stderr": str(stderr_path),
                "updated_at": utc_now(),
            },
        )

    def existing_outcome(self, case_key: str) -> CaseOutcome | None:
        path = self.results_dir / f"{case_key}.json"
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            result.get("version") != SCRIPT_VERSION
            or result.get("measurement_method") != MEASUREMENT_METHOD
            or result.get("measurement_config") != self.measurement_config()
        ):
            self.logger.log(
                f"STALE: {case_key} uses an obsolete measurement contract; rerunning"
            )
            return None
        status = result.get("status")
        if status in {"complete", "oom"} or (
            status in {"failed", "interrupted"} and not self.args.retry_failed
        ):
            outcome = CaseOutcome(
                str(status),
                int(result.get("qualifies_for_50_tps", 0)),
                result.get("generation_tokens_per_second"),
                str(result.get("reason", "")),
            )
            self.logger.log(
                f"SKIP: {case_key} status={outcome.status} "
                f"generation_tps={outcome.generation_tps if outcome.generation_tps is not None else 'n/a'} "
                f"reason={outcome.reason or 'none'}"
            )
            return outcome
        return None

    def run_case(self, model: ModelSpec, model_path: Path, context: int) -> CaseOutcome:
        case_key = f"{model.model_id}-ctx-{context}"
        self.current_case = (case_key, model.model_id, context)
        existing = self.existing_outcome(case_key)
        if existing is not None:
            return existing
        depth = context - self.config.generation_tokens
        for attempt in range(1, self.config.max_attempts + 1):
            stdout_path = self.raw_dir / f"{case_key}.attempt-{attempt}.stdout.json"
            stderr_path = self.raw_dir / f"{case_key}.attempt-{attempt}.stderr.log"
            before_memory = (
                self.raw_dir / f"{case_key}.attempt-{attempt}.memory-before.log"
            )
            after_memory = (
                self.raw_dir / f"{case_key}.attempt-{attempt}.memory-after.log"
            )
            self.capture_memory(before_memory)
            self.write_case_state(
                case_key,
                model.model_id,
                context,
                "running",
                attempt,
                0,
                "benchmark started",
            )
            command = self.benchmark_command(model_path, context)
            self.logger.log(
                f"START: {case_key} model={model.model_id} total_context={context} "
                f"kv_depth={depth} generated={self.config.generation_tokens} "
                f"threshold={self.config.minimum_tps}"
            )
            self.logger.log("COMMAND: " + " ".join(command))
            exit_code, peak_rss = self.run_process(command, stdout_path, stderr_path)
            self.logger.log(
                f"CLEANUP: waiting {self.config.cleanup_wait_seconds}s for Metal buffers "
                "and mapped pages to be reclaimed."
            )
            time.sleep(self.config.cleanup_wait_seconds)
            self.capture_memory(after_memory)
            measurement = (
                self.parse_output(stdout_path, context) if exit_code == 0 else None
            )
            if measurement is not None:
                qualifies = int(measurement[0] >= self.config.minimum_tps)
                reason = (
                    "generation speed meets threshold"
                    if qualifies
                    else "generation speed is below threshold"
                )
                self.write_case_result(
                    model,
                    model_path,
                    context,
                    "complete",
                    attempt,
                    exit_code,
                    qualifies,
                    reason,
                    peak_rss,
                    stdout_path,
                    stderr_path,
                    measurement,
                )
                self.write_case_state(
                    case_key,
                    model.model_id,
                    context,
                    "complete",
                    attempt,
                    exit_code,
                    reason,
                )
                self.logger.log(
                    f"COMPLETE: {case_key} generation_avg={measurement[0]} tok/s "
                    f"generation_stddev={measurement[1]} samples={measurement[2]} "
                    f"total_context={context} kv_depth={depth} peak_rss={peak_rss} KiB "
                    f"qualifies={qualifies}"
                )
                return CaseOutcome("complete", qualifies, measurement[0], reason)
            output = ""
            for path in (stdout_path, stderr_path):
                try:
                    output += path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    pass
            if exit_code in {9, 137} or OOM_PATTERN.search(output):
                reason = "llama.cpp reported an allocation or process memory failure"
                self.write_case_result(
                    model,
                    model_path,
                    context,
                    "oom",
                    attempt,
                    exit_code,
                    0,
                    reason,
                    peak_rss,
                    stdout_path,
                    stderr_path,
                    None,
                )
                self.write_case_state(
                    case_key, model.model_id, context, "oom", attempt, exit_code, reason
                )
                self.logger.log(
                    f"OOM: {case_key}; larger contexts for this model will not be attempted."
                )
                return CaseOutcome("oom", 0, None, reason)
            reason = f"llama-bench failed with exit status {exit_code} or invalid JSON output"
            if attempt < self.config.max_attempts:
                self.write_case_state(
                    case_key,
                    model.model_id,
                    context,
                    "retrying",
                    attempt,
                    exit_code,
                    reason,
                )
                self.logger.log(f"RETRY: {case_key} after {self.config.retry_delay}s")
                time.sleep(self.config.retry_delay)
                continue
            self.write_case_result(
                model,
                model_path,
                context,
                "failed",
                attempt,
                exit_code,
                0,
                reason,
                peak_rss,
                stdout_path,
                stderr_path,
                None,
            )
            self.write_case_state(
                case_key, model.model_id, context, "failed", attempt, exit_code, reason
            )
            self.logger.log(f"FAILED: {case_key}")
            return CaseOutcome("failed", 0, None, reason)
        raise AssertionError("attempt loop did not return")

    def probe_model(self, model: ModelSpec) -> bool:
        model_path = self.stage_model(model)
        self.logger.log(
            f"MODEL: {model.model_id} max_context={model.maximum_context} path={model_path}"
        )
        last_pass = 0
        first_failure = 0
        first_failure_status = ""
        hard_failure = False
        for context in self.context_ladder(model.maximum_context):
            outcome = self.run_case(model, model_path, context)
            self.current_case = None
            hard_failure = hard_failure or outcome.status in {"failed", "interrupted"}
            if outcome.status == "complete" and outcome.qualifies:
                last_pass = context
                self.logger.log(
                    f"PASS: {model.model_id} context={context} generation={outcome.generation_tps} tok/s"
                )
                continue
            first_failure = context
            first_failure_status = outcome.status
            self.logger.log(
                f"STOP COARSE SEARCH: {model.model_id} context={context} "
                f"status={outcome.status} reason={outcome.reason}"
            )
            break
        if (
            last_pass > 0
            and first_failure > last_pass
            and first_failure_status in {"complete", "oom"}
        ):
            low, high = last_pass, first_failure
            self.logger.log(
                f"FINE SEARCH: {model.model_id} between {low} and {high} "
                f"with step {self.config.search_step}"
            )
            while high - low > self.config.search_step:
                middle = ((low + high) // 2 // 256) * 256
                middle = max(middle, low + 256)
                if middle >= high:
                    break
                outcome = self.run_case(model, model_path, middle)
                self.current_case = None
                hard_failure = hard_failure or outcome.status in {
                    "failed",
                    "interrupted",
                }
                if outcome.status == "complete" and outcome.qualifies:
                    low = middle
                    self.logger.log(
                        f"PASS FINE: {model.model_id} context={middle} "
                        f"generation={outcome.generation_tps} tok/s"
                    )
                else:
                    high = middle
                    self.logger.log(
                        f"FAIL FINE: {model.model_id} context={middle} "
                        f"status={outcome.status} reason={outcome.reason}"
                    )
            last_pass = low
        atomic_write_text(
            self.state_dir / f"{model.model_id}.summary",
            state_text(
                {
                    "model_id": model.model_id,
                    "maximum_context_tokens": model.maximum_context,
                    "largest_passing_context_tokens": last_pass,
                    "first_failing_context_tokens": first_failure,
                    "updated_at": utc_now(),
                }
            ),
        )
        if last_pass:
            self.logger.log(
                f"MODEL SUMMARY: {model.model_id} largest_passing_context={last_pass}"
            )
        else:
            self.logger.log(
                f"MODEL SUMMARY: {model.model_id} did not meet the "
                f"{self.config.minimum_tps} tok/s threshold at the minimum context"
            )
        return not hard_failure

    def write_summary(self) -> None:
        records: list[dict[str, Any]] = []
        for path in self.results_dir.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                isinstance(record, dict)
                and record.get("version") == SCRIPT_VERSION
                and record.get("measurement_method") == MEASUREMENT_METHOD
            ):
                records.append(record)
        if records:
            records.sort(
                key=lambda record: (
                    record.get("model_id", ""),
                    record.get("requested_context_tokens", 0),
                )
            )
            atomic_write_json(self.config.benchmark_root / "summary.json", records)

    def run(self) -> int:
        models = self.load_manifest()
        selected = self.selected(models)
        self.logger.log(f"Models loaded: {len(models)}")
        for model in selected:
            contexts = ",".join(
                str(value) for value in self.context_ladder(model.maximum_context)
            )
            self.logger.log(f"PLAN: {model.model_id} contexts={contexts},")
        if self.args.dry_run:
            self.logger.log(
                "DRY RUN COMPLETE: no model was loaded and no benchmark process was started."
            )
            return 0
        self.create_directories()
        self.acquire_lock()
        self.write_run_state("running", "benchmark started")
        self.logger.log(f"Run id: {self.run_id}")
        self.logger.log(f"Persistent log: {self.log_file}")
        if self.args.stage_only:
            for model in selected:
                self.stage_model(model)
            self.write_run_state(
                "complete", "selected models staged locally; no inference was started"
            )
            self.logger.log(
                f"STAGE SUMMARY: selected models are ready under {self.config.local_root}"
            )
            return 0
        all_succeeded = True
        for model in selected:
            all_succeeded = self.probe_model(model) and all_succeeded
        self.write_summary()
        if all_succeeded:
            self.write_run_state("complete", "all selected models were benchmarked")
            self.logger.log(
                f"SUMMARY: benchmark completed. Results: {self.config.benchmark_root / 'summary.json'}"
            )
            return 0
        self.write_run_state(
            "partial", "one or more selected models had benchmark failures"
        )
        self.logger.log(
            f"SUMMARY: benchmark completed with failures. Results: "
            f"{self.config.benchmark_root / 'summary.json'}"
        )
        return 1

    def close(self, status: int) -> None:
        if self.active_process is not None:
            terminate_process_group(self.active_process, self.logger)
        if status and self.current_case and self.state_dir.is_dir():
            case_key, model_id, context = self.current_case
            self.write_case_state(
                case_key,
                model_id,
                context,
                "interrupted",
                0,
                status,
                "benchmark process ended before case completion",
            )
        if status and self.run_status == "running":
            self.write_run_state(
                "interrupted", f"benchmark stopped with exit status {status}"
            )
        self.release_lock()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GGUF generation at populated KV-cache depths."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logger = Logger()
    runner: BenchmarkRunner | None = None
    status = 2
    try:
        args = parse_args(argv if argv is not None else sys.argv[1:])
        config = Config.from_environment()
        logger.log(f"Starting llama.cpp benchmark orchestrator {SCRIPT_VERSION}.")
        logger.log(f"Source model root: {config.source_root}")
        logger.log(f"Local model cache: {config.local_root}")
        logger.log(f"Benchmark root: {config.benchmark_root}")
        logger.log(f"Manifest: {config.manifest}")
        logger.log("No inference process has started.")
        if not config.source_root.is_dir():
            raise BenchmarkError(
                f"source model root is not mounted: {config.source_root}"
            )
        if (args.dry_run or not args.stage_only) and not command_exists(
            config.llama_bench
        ):
            raise BenchmarkError(f"llama-bench was not found: {config.llama_bench}")
        if args.dry_run or not args.stage_only:
            result = subprocess.run(
                [config.llama_bench, "--help"],
                capture_output=True,
                check=False,
                timeout=30,
            )
            if result.returncode != 0:
                raise BenchmarkError("llama-bench could not start")
        runner = BenchmarkRunner(config, args, logger)
        status = runner.run()
    except KeyboardInterrupt:
        logger.log("Signal received; stopping the active operation.")
        status = 130
    except BenchmarkError as error:
        logger.log(f"ERROR: {error}")
        status = 2
    finally:
        if runner is not None:
            runner.close(status)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
