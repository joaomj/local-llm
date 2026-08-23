from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "lfm_qad_report.py"


class LfmQadReportCliTest(unittest.TestCase):
    def test_report_keeps_performance_and_quality_dimensions_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "models.tsv"
            published = root / "published.json"
            results = root / "results"
            output_json = root / "report.json"
            output_markdown = root / "report.md"
            manifest.write_text(
                "# id\trepo\tfile\tcontext\tdirectory\n"
                "model-a\tOrg/A\ta.gguf\t32768\tA\n"
                "model-b\tOrg/B\tb.gguf\t131072\tB\n",
                encoding="utf-8",
            )
            published.write_text(
                json.dumps(
                    {
                        "source": {"url": "https://example.test/source"},
                        "models": {
                            "model-a": {
                                "qad_mean_score": 10.0,
                                "bf16_retained_percent": 99.0,
                                "math_benchmark": "GSM8K",
                            },
                            "model-b": {
                                "qad_mean_score": 20.0,
                                "bf16_retained_percent": 98.0,
                                "math_benchmark": "AIME25",
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            for model_id, score in (("model-a", 7), ("model-b", 9)):
                result_dir = results / model_id
                (result_dir / "performance").mkdir(parents=True)
                (result_dir / "performance" / "summary.json").write_text(
                    json.dumps(
                        [
                            {
                                "requested_context_tokens": 512,
                                "generation_tokens_per_second": 60,
                                "qualifies_for_50_tps": True,
                            },
                            {
                                "requested_context_tokens": 1024,
                                "generation_tokens_per_second": 40,
                                "qualifies_for_50_tps": False,
                            },
                        ]
                    ),
                    encoding="utf-8",
                )
                (result_dir / "judgments.jsonl").write_text(
                    json.dumps(
                        {
                            "question_id": 1,
                            "score": score,
                            "judge_model": "google/gemini-3.7-flash",
                            "provider": "test-provider",
                            "cost_usd": 0.001,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            # Use a two-model fixture only to keep the CLI test small. The real manifest remains four models.
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--manifest",
                    str(manifest),
                    "--published",
                    str(published),
                    "--results-root",
                    str(results),
                    "--output-json",
                    str(output_json),
                    "--output-markdown",
                    str(output_markdown),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1)

            # The production manifest requires four models, so verify the useful
            # report seam with the real manifest and a complete synthetic result
            # set below instead of weakening that contract.
            published_real = root / "published-real.json"
            published_real.write_text(
                (
                    ROOT / "benchmarks" / "reference" / "lfm-qad-published.json"
                ).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            real_results = root / "real-results"
            for model_id in (
                "lfm2.5-230m-qad-q4_0",
                "lfm2.5-350m-qad-q4_0",
                "lfm2.5-1.2b-instruct-qad-q4_0",
                "lfm2.5-2.6b-qad-q4_0",
            ):
                result_dir = real_results / model_id
                (result_dir / "performance").mkdir(parents=True)
                (result_dir / "performance" / "summary.json").write_text(
                    json.dumps(
                        [
                            {
                                "requested_context_tokens": 32768,
                                "generation_tokens_per_second": 55,
                                "qualifies_for_50_tps": True,
                            }
                        ]
                    ),
                    encoding="utf-8",
                )
                (result_dir / "judgments.jsonl").write_text(
                    json.dumps(
                        {
                            "question_id": 1,
                            "score": 8,
                            "judge_model": "google/gemini-3.7-flash",
                            "provider": "test-provider",
                            "cost_usd": 0.001,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--results-root",
                    str(real_results),
                    "--output-json",
                    str(output_json),
                    "--output-markdown",
                    str(output_markdown),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(len(report["models"]), 4)
            self.assertEqual(
                report["models"][0]["performance"]["largest_passing_context_tokens"],
                32768,
            )
            self.assertEqual(report["models"][0]["mt_bench"]["mean_score"], 8.0)
            markdown = output_markdown.read_text(encoding="utf-8")
            self.assertIn("## Local Performance", markdown)
            self.assertIn("## Published Liquid AI Quality", markdown)
            self.assertIn("## MT-Bench Judged Chat", markdown)
            self.assertIn("does not use a blended score", markdown)


if __name__ == "__main__":
    unittest.main()
