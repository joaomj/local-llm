#!/usr/bin/env python3
"""Download and benchmark ordered model-family candidates with llama-server."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = "llama_server_capacity_and_prompt-v1"
DEFAULT_MIN_FREE_BYTES = 30 * 1024**3
DEFAULT_MIN_FREE_PERCENT = 15


class SearchError(RuntimeError):
    """An operational error that must stop the model search."""


class StorageFloorError(SearchError):
    """The next download would violate the storage safety policy."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def env_int(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise SearchError(f"{name} must be an integer >= {minimum}") from error
    if value < minimum:
        raise SearchError(f"{name} must be >= {minimum}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def command_exists(command: str) -> bool:
    if "/" in command:
        return Path(command).is_file() and os.access(command, os.X_OK)
    return shutil.which(command) is not None


def existing_path(path: Path) -> Path:
    current = path.expanduser().resolve()
    while not current.exists():
        if current == current.parent:
            raise SearchError(f"no existing parent for storage path: {path}")
        current = current.parent
    return current


@dataclass(frozen=True)
class Candidate:
    model_id: str
    family: str
    repo: str
    filename: str
    maximum_context_tokens: int
    model_directory: str


@dataclass(frozen=True)
class Config:
    manifest: Path
    benchmark_root: Path
    scratch_root: Path
    hf_bin: str
    server_bin: str
    server_runner: Path
    download_attempts: int
    download_retry_seconds: int
    minimum_free_bytes: int
    minimum_free_percent: int

    @classmethod
    def from_environment(cls) -> Config:
        return cls(
            manifest=Path(
                os.environ.get(
                    "THROUGHPUT_MANIFEST", ROOT / "scripts" / "throughput-models.tsv"
                )
            ).expanduser(),
            benchmark_root=Path(
                os.environ.get(
                    "THROUGHPUT_BENCHMARK_ROOT",
                    ROOT / ".benchmark" / "throughput-server",
                )
            ).expanduser(),
            scratch_root=Path(
                os.environ.get(
                    "THROUGHPUT_SCRATCH_ROOT",
                    ROOT / ".benchmark" / "throughput-models",
                )
            ).expanduser(),
            hf_bin=os.environ.get("THROUGHPUT_HF_BIN", "hf"),
            server_bin=os.environ.get("LLAMA_SERVER_BENCHMARK_BIN", "llama-server"),
            server_runner=Path(
                os.environ.get(
                    "THROUGHPUT_LLAMA_SERVER_BENCHMARK",
                    ROOT / "scripts" / "llama_server_benchmark.py",
                )
            ).expanduser(),
            download_attempts=env_int("THROUGHPUT_DOWNLOAD_ATTEMPTS", 3, 1),
            download_retry_seconds=env_int("THROUGHPUT_DOWNLOAD_RETRY_SECONDS", 15, 0),
            minimum_free_bytes=env_int(
                "THROUGHPUT_MIN_FREE_BYTES", DEFAULT_MIN_FREE_BYTES, 0
            ),
            minimum_free_percent=env_int(
                "THROUGHPUT_MIN_FREE_PERCENT", DEFAULT_MIN_FREE_PERCENT, 0
            ),
        )


def load_manifest(path: Path) -> list[Candidate]:
    if not path.is_file():
        raise SearchError(f"manifest is missing: {path}")
    candidates: list[Candidate] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, fields in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if not fields or not fields[0] or fields[0].startswith("#"):
                continue
            if len(fields) != 6:
                raise SearchError(
                    f"manifest line {line_number} must contain six columns"
                )
            model_id, family, repo, filename, maximum_raw, model_directory = fields
            if not re.fullmatch(r"[a-z0-9._-]+", model_id):
                raise SearchError(f"invalid model ID: {model_id}")
            if not re.fullmatch(r"[a-z0-9._-]+", family):
                raise SearchError(f"invalid family ID: {family}")
            if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repo):
                raise SearchError(f"invalid repository for {model_id}")
            if not re.fullmatch(r"[A-Za-z0-9_.-]+\.gguf", filename):
                raise SearchError(f"invalid GGUF filename for {model_id}")
            if not re.fullmatch(r"[A-Za-z0-9._-]+", model_directory):
                raise SearchError(f"invalid model directory for {model_id}")
            try:
                maximum_context = int(maximum_raw)
            except ValueError as error:
                raise SearchError(f"invalid maximum context for {model_id}") from error
            if maximum_context < 32768:
                raise SearchError(f"maximum context for {model_id} must be >= 32768")
            candidates.append(
                Candidate(
                    model_id,
                    family,
                    repo,
                    filename,
                    maximum_context,
                    model_directory,
                )
            )
    if not candidates:
        raise SearchError("manifest contains no models")
    return candidates


def storage_status(path: Path) -> tuple[int, float]:
    usage = shutil.disk_usage(existing_path(path))
    return usage.free, usage.free * 100 / usage.total


def enforce_storage_floor(config: Config, candidate: Candidate) -> None:
    free_bytes, free_percent = storage_status(config.scratch_root)
    log(
        f"STORAGE: candidate={candidate.model_id} free_gib={free_bytes / 1024**3:.1f} "
        f"free_percent={free_percent:.1f}"
    )
    if (
        free_bytes < config.minimum_free_bytes
        or free_percent < config.minimum_free_percent
    ):
        raise StorageFloorError(
            "storage safety floor reached before download; no weights were removed. "
            "Choose which downloaded models to keep before the search continues"
        )


def download_model(config: Config, candidate: Candidate, model_path: Path) -> None:
    if model_path.is_file() and model_path.stat().st_size > 0:
        log(f"DOWNLOAD SKIP: {candidate.model_id} weights already exist")
        return
    enforce_storage_floor(config, candidate)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        config.hf_bin,
        "download",
        candidate.repo,
        candidate.filename,
        "--local-dir",
        str(model_path.parent),
    ]
    for attempt in range(1, config.download_attempts + 1):
        log(
            f"DOWNLOAD: {candidate.model_id} attempt={attempt}/{config.download_attempts}"
        )
        result = subprocess.run(command, check=False)
        if (
            result.returncode == 0
            and model_path.is_file()
            and model_path.stat().st_size > 0
        ):
            log(f"DOWNLOAD COMPLETE: {candidate.model_id}")
            return
        if attempt < config.download_attempts:
            time.sleep(config.download_retry_seconds)
    raise SearchError(
        f"download did not produce {model_path}; retaining scratch directory"
    )


def completion_key(
    config: Config, candidate: Candidate, model_path: Path
) -> dict[str, Any]:
    stat = model_path.stat()
    return {
        "measurement_contract": CONTRACT,
        "model_id": candidate.model_id,
        "model_size": stat.st_size,
        "model_mtime_ns": stat.st_mtime_ns,
        "maximum_context_tokens": candidate.maximum_context_tokens,
        "capacities": os.environ.get("LLAMA_SERVER_BENCHMARK_CAPACITIES", ""),
        "prompt_targets": os.environ.get("LLAMA_SERVER_BENCHMARK_PROMPT_TARGETS", ""),
        "generation_tokens": os.environ.get(
            "LLAMA_SERVER_BENCHMARK_GENERATION_TOKENS", ""
        ),
        "minimum_generation_tps": os.environ.get(
            "LLAMA_SERVER_BENCHMARK_MIN_GENERATION_TPS", "50"
        ),
    }


def valid_summary(path: Path, candidate: Candidate) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SearchError(
            f"invalid server benchmark summary for {candidate.model_id}"
        ) from error
    if (
        not isinstance(value, dict)
        or value.get("measurement_contract") != CONTRACT
        or value.get("model_id") != candidate.model_id
        or not isinstance(value.get("minimum_requirement_met"), bool)
        or not isinstance(value.get("largest_passing_configured_context_tokens"), int)
        or not isinstance(value.get("largest_passing_actual_prompt_tokens"), int)
        or value.get("weights_retained") is not True
    ):
        raise SearchError(
            f"server benchmark summary has the wrong contract for {candidate.model_id}"
        )
    return value


def run_benchmark(
    config: Config, candidate: Candidate, model_path: Path
) -> dict[str, Any]:
    result_root = config.benchmark_root / candidate.model_id
    result_root.mkdir(parents=True, exist_ok=True)
    summary_path = result_root / "summary.json"
    marker_path = result_root / "completion.json"
    key = completion_key(config, candidate, model_path)
    if marker_path.is_file() and summary_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            marker = None
        if marker == key:
            log(f"BENCHMARK RESUME: {candidate.model_id} exact configuration completed")
            return valid_summary(summary_path, candidate)

    environment = os.environ.copy()
    environment.update(
        {
            "LLAMA_SERVER_BENCHMARK_BIN": config.server_bin,
            "LLAMA_SERVER_BENCHMARK_MODEL_PATH": str(model_path),
            "LLAMA_SERVER_BENCHMARK_MODEL_ID": candidate.model_id,
            "LLAMA_SERVER_BENCHMARK_MAX_CONTEXT": str(candidate.maximum_context_tokens),
            "LLAMA_SERVER_BENCHMARK_ROOT": str(result_root),
        }
    )
    command = (
        [sys.executable, str(config.server_runner)]
        if config.server_runner.suffix == ".py"
        else [str(config.server_runner)]
    )
    log(f"BENCHMARK START: {candidate.model_id}")
    result = subprocess.run(command, env=environment, check=False)
    if result.returncode != 0:
        raise SearchError(
            f"server benchmark failed for {candidate.model_id} with code {result.returncode}"
        )
    summary = valid_summary(summary_path, candidate)
    atomic_json(marker_path, key)
    log(
        f"BENCHMARK RESULT: {candidate.model_id} "
        f"configured={summary['largest_passing_configured_context_tokens']} "
        f"occupied_prompt={summary['largest_passing_actual_prompt_tokens']} "
        f"minimum_met={summary['minimum_requirement_met']}"
    )
    return summary


def acquire_lock(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".search.lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise SearchError(f"another search owns lock: {lock}") from error
    (lock / "pid").write_text(f"{os.getpid()}\n", encoding="ascii")
    return lock


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the highest-quality family quant that meets the server gate."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--family")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    lock: Path | None = None
    try:
        args = parse_args(argv if argv is not None else sys.argv[1:])
        config = Config.from_environment()
        candidates = load_manifest(config.manifest)
        model_ids = {candidate.model_id for candidate in candidates}
        family_ids = {candidate.family for candidate in candidates}
        if args.model and args.model not in model_ids:
            raise SearchError(f"model ID is not in the manifest: {args.model}")
        if args.family and args.family not in family_ids:
            raise SearchError(f"family ID is not in the manifest: {args.family}")
        candidates = [
            candidate
            for candidate in candidates
            if (not args.model or candidate.model_id == args.model)
            and (not args.family or candidate.family == args.family)
        ]
        if args.dry_run:
            for candidate in candidates:
                log(
                    f"DRY RUN: family={candidate.family} candidate={candidate.model_id} "
                    f"maximum_context={candidate.maximum_context_tokens}"
                )
            return 0
        if not command_exists(config.hf_bin):
            raise SearchError(f"Hugging Face CLI was not found: {config.hf_bin}")
        if not command_exists(config.server_bin):
            raise SearchError(f"llama-server was not found: {config.server_bin}")
        if not config.server_runner.is_file():
            raise SearchError(
                f"server benchmark runner is missing: {config.server_runner}"
            )

        lock = acquire_lock(config.benchmark_root)
        selected: dict[str, str] = {}
        summaries: list[dict[str, Any]] = []
        for candidate in candidates:
            if not args.model and candidate.family in selected:
                log(
                    f"QUANT SKIP: {candidate.model_id}; family={candidate.family} "
                    f"already selected={selected[candidate.family]}"
                )
                continue
            model_path = (
                config.scratch_root / candidate.model_directory / candidate.filename
            )
            download_model(config, candidate, model_path)
            summary = run_benchmark(config, candidate, model_path)
            summaries.append(summary)
            if summary["minimum_requirement_met"]:
                selected[candidate.family] = candidate.model_id
                log(
                    f"FAMILY SELECTED: family={candidate.family} candidate={candidate.model_id}"
                )
            else:
                log(
                    f"QUANT DESCEND: family={candidate.family} candidate={candidate.model_id} "
                    "did not meet the 32K capacity and occupied-prompt speed gate"
                )
        atomic_json(
            config.benchmark_root / "summary.json",
            {
                "measurement_contract": CONTRACT,
                "selected_by_family": selected,
                "weights_retained": True,
                "candidates": summaries,
                "completed_at": utc_now(),
            },
        )
        return 0
    except StorageFloorError as error:
        log(f"SEARCH STOPPED: {error}")
        return 2
    except KeyboardInterrupt:
        log("SEARCH STOPPED: interrupted; downloaded weights were retained")
        return 130
    except (SearchError, OSError, subprocess.SubprocessError) as error:
        log(f"SEARCH STOPPED: {error}; downloaded weights were retained")
        return 1
    finally:
        if lock is not None:
            shutil.rmtree(lock, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
