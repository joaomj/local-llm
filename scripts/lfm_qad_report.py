#!/usr/bin/env python3
"""Build separate performance and quality reports for the LFM QAD run."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "benchmarks" / "manifests" / "lfm-qad-models.tsv"
DEFAULT_PUBLISHED = ROOT / "benchmarks" / "reference" / "lfm-qad-published.json"
DEFAULT_RESULTS = ROOT / ".benchmark" / "lfm-qad-results"
DEFAULT_REPORT_DIR = ROOT / "results" / "lfm-qad"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    models = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise ValueError(
                    f"manifest line {line_number} must have five tab-separated fields"
                )
            model_id, repo, file_name, context, model_directory = fields
            models.append(
                {
                    "model_id": model_id,
                    "repository": repo,
                    "file": file_name,
                    "official_context_tokens": int(context),
                    "model_directory": model_directory,
                }
            )
    if len(models) != 4:
        raise ValueError(f"expected four models in manifest, found {len(models)}")
    return models


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL in {path} at line {line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise TypeError(
                    f"JSONL record in {path} at line {line_number} is not an object"
                )
            records.append(value)
    return records


def performance_summary(path: Path, minimum_tps: float) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "missing",
            "minimum_generation_tps": minimum_tps,
            "largest_passing_context_tokens": None,
            "cases": 0,
        }
    records = load_json(path)
    if not isinstance(records, list):
        raise TypeError(f"performance summary is not an array: {path}")
    passing = []
    for record in records:
        if not isinstance(record, dict):
            continue
        context = record.get("requested_context_tokens")
        tps = record.get("generation_tokens_per_second")
        qualifies = record.get("qualifies_for_50_tps")
        if (
            isinstance(context, int)
            and isinstance(tps, (int, float))
            and not isinstance(tps, bool)
            and tps >= minimum_tps
            and qualifies is not False
        ):
            passing.append(context)
    return {
        "status": "complete" if records else "empty",
        "minimum_generation_tps": minimum_tps,
        "largest_passing_context_tokens": max(passing) if passing else None,
        "cases": len(records),
        "passing_cases": len(passing),
    }


def chat_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "questions": 0, "mean_score": None}
    records = load_jsonl(path)
    scores = [record.get("score") for record in records]
    if not all(
        isinstance(score, int) and not isinstance(score, bool) and 1 <= score <= 10
        for score in scores
    ):
        raise ValueError(f"invalid score in judgment file: {path}")
    if not scores:
        return {"status": "empty", "questions": 0, "mean_score": None}
    judge_models = sorted(
        {record.get("judge_model") for record in records if record.get("judge_model")}
    )
    providers = sorted(
        {record.get("provider") for record in records if record.get("provider")}
    )
    costs = [record.get("cost_usd") for record in records]
    numeric_costs = [
        cost
        for cost in costs
        if isinstance(cost, (int, float)) and not isinstance(cost, bool)
    ]
    return {
        "status": "complete" if len(records) == 80 else "partial",
        "questions": len(records),
        "mean_score": round(statistics.fmean(scores), 4),
        "min_score": min(scores),
        "max_score": max(scores),
        "judge_models": judge_models,
        "providers": providers,
        "cost_usd": round(sum(numeric_costs), 8),
    }


def model_report(
    model: dict[str, Any],
    results_root: Path,
    published: dict[str, Any],
    minimum_tps: float,
) -> dict[str, Any]:
    model_id = model["model_id"]
    result_dir = results_root / model_id
    published_model = published.get("models", {}).get(model_id)
    if not isinstance(published_model, dict):
        raise TypeError(f"published metrics do not contain model: {model_id}")
    return {
        **model,
        "performance": performance_summary(
            result_dir / "performance" / "summary.json", minimum_tps
        ),
        "published_quality": published_model,
        "mt_bench": chat_summary(result_dir / "judgments.jsonl"),
    }


def sort_ids(
    models: list[dict[str, Any]], key: str, descending: bool = True
) -> list[str]:
    present = [model for model in models if model[key] is not None]
    absent = [model for model in models if model[key] is None]
    present.sort(
        key=lambda model: (
            (-model[key], model["model_id"])
            if descending
            else (model[key], model["model_id"])
        )
    )
    return [model["model_id"] for model in present + absent]


def build_report(
    manifest_path: Path,
    published_path: Path,
    results_root: Path,
    minimum_tps: float,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    published = load_json(published_path)
    if not isinstance(published, dict):
        raise TypeError(f"published metrics must be an object: {published_path}")
    models = [
        model_report(model, results_root, published, minimum_tps) for model in manifest
    ]
    for model in models:
        model["performance"] = {
            **model["performance"],
            "gate_passed": model["performance"]["largest_passing_context_tokens"]
            is not None,
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "LFM2.5 QAD Q4_0",
        "sources": published.get("source", {}),
        "judge_note": "MT-Bench scores compare these candidates under the same OpenRouter judge and are not historical GPT-4 MT-Bench scores.",
        "models": models,
        "rankings": {
            "local_context_at_or_above_50_tps": sort_ids(
                [
                    {
                        **model,
                        "value": model["performance"]["largest_passing_context_tokens"],
                    }
                    for model in models
                ],
                "value",
            ),
            "published_qad_mean": sort_ids(
                [
                    {**model, "value": model["published_quality"].get("qad_mean_score")}
                    for model in models
                ],
                "value",
            ),
            "mt_bench_mean": sort_ids(
                [
                    {**model, "value": model["mt_bench"].get("mean_score")}
                    for model in models
                ],
                "value",
            ),
        },
    }


def display(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# LFM2.5 QAD Q4_0 Comparison",
        "",
        "This report keeps local performance, published quality, and judged chat quality separate. It does not use a blended score.",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Local Performance",
        "",
        "| Model | Official context | Largest context at >=50 tok/s | Gate | Cases |",
        "|---|---:|---:|---|---:|",
    ]
    for model in report["models"]:
        performance = model["performance"]
        lines.append(
            f"| `{model['model_id']}` | {model['official_context_tokens']} | "
            f"{display(performance['largest_passing_context_tokens'])} | "
            f"{'pass' if performance['gate_passed'] else 'not met'} | {performance['cases']} |"
        )
    lines.extend(
        [
            "",
            "## Published Liquid AI Quality",
            "",
            "| Model | QAD mean | BF16 score retained | Math benchmark |",
            "|---|---:|---:|---|",
        ]
    )
    for model in report["models"]:
        quality = model["published_quality"]
        lines.append(
            f"| `{model['model_id']}` | {display(quality.get('qad_mean_score'))} | "
            f"{display(quality.get('bf16_retained_percent'))}% | {quality.get('math_benchmark', 'n/a')} |"
        )
    lines.extend(
        [
            "",
            "## MT-Bench Judged Chat",
            "",
            "Judge: `google/gemini-3.7-flash` through OpenRouter, temperature 0.",
            "",
            "| Model | Questions | Mean score | Range | Cost (USD) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in report["models"]:
        chat = model["mt_bench"]
        score_range = (
            f"{chat.get('min_score')}–{chat.get('max_score')}"
            if chat.get("min_score") is not None
            else "n/a"
        )
        lines.append(
            f"| `{model['model_id']}` | {chat['questions']} | {display(chat['mean_score'])} | "
            f"{score_range} | {display(chat.get('cost_usd'))} |"
        )
    lines.extend(
        [
            "",
            "## Ranking Order",
            "",
            "- Local context at or above 50 tok/s: "
            + ", ".join(
                f"`{model_id}`"
                for model_id in report["rankings"]["local_context_at_or_above_50_tps"]
            ),
            "- Published QAD mean: "
            + ", ".join(
                f"`{model_id}`" for model_id in report["rankings"]["published_qad_mean"]
            ),
            "- MT-Bench mean: "
            + ", ".join(
                f"`{model_id}`" for model_id in report["rankings"]["mt_bench_mean"]
            ),
            "",
            "Source: " + str(report["sources"].get("url", "not recorded")),
        ]
    )
    return "\n".join(lines) + "\n"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--published", type=Path, default=DEFAULT_PUBLISHED)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--output-json", type=Path, default=DEFAULT_REPORT_DIR / "comparison.json"
    )
    parser.add_argument(
        "--output-markdown", type=Path, default=DEFAULT_REPORT_DIR / "comparison.md"
    )
    parser.add_argument("--minimum-tps", type=float, default=50.0)
    args = parser.parse_args()
    if args.minimum_tps <= 0:
        parser.error("--minimum-tps must be positive")
    return args


def main() -> int:
    try:
        args = parse_args()
        report = build_report(
            args.manifest, args.published, args.results_root, args.minimum_tps
        )
        write_text(
            args.output_json, json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
        write_text(args.output_markdown, markdown_report(report))
        print(f"wrote {args.output_json}")
        print(f"wrote {args.output_markdown}")
        return 0
    except (OSError, ValueError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
