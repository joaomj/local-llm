from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.lfm_chat_benchmark import (
    MARKER_NAME,
    MARKER_OWNER,
    ModelSpec,
    finalize_model_scratch,
    invalidate_stale_judgments,
    prepare_model,
    write_quality_report,
)


class LfmChatBenchmarkTest(unittest.TestCase):
    def test_reuses_throughput_weight_without_duplicate_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "throughput"
            scratch_root = root / "quality"
            model = ModelSpec("model-a", "Org/Repo", "model.gguf", 32768, "model")
            source_file = source_root / model.model_directory / model.file_name
            source_file.parent.mkdir(parents=True)
            source_file.write_bytes(b"weights")
            config = {
                "scratch_root": scratch_root,
                "source_root": source_root,
                "download_attempts": 1,
            }

            prepared = prepare_model(config, model)

            self.assertEqual(prepared.read_bytes(), b"weights")
            self.assertEqual(prepared.stat().st_ino, source_file.stat().st_ino)
            finalize_model_scratch(config, model, succeeded=True)
            self.assertTrue(source_file.exists())

    def test_retains_model_scratch_until_run_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scratch_root = Path(directory)
            model = ModelSpec("model-a", "Org/Repo", "model.gguf", 32768, "model")
            model_dir = scratch_root / model.model_directory
            model_dir.mkdir()
            (model_dir / MARKER_NAME).write_text(
                "\n".join(
                    (
                        f"owner={MARKER_OWNER}",
                        f"model_id={model.model_id}",
                        f"repo={model.repository}",
                        f"file={model.file_name}",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            (model_dir / model.file_name).write_bytes(b"weights")
            config = {"scratch_root": scratch_root}

            finalize_model_scratch(config, model, succeeded=False)
            self.assertTrue((model_dir / model.file_name).exists())

            finalize_model_scratch(config, model, succeeded=True)
            self.assertFalse(model_dir.exists())

    def test_writes_quality_report_with_generation_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_root = root / "results"
            result_dir = result_root / "model-a"
            result_dir.mkdir(parents=True)
            (result_dir / "answers.jsonl").write_text(
                json.dumps({"question_id": 1}) + "\n", encoding="utf-8"
            )
            (result_dir / "judgments.jsonl").write_text(
                json.dumps({"question_id": 1, "score": 8}) + "\n", encoding="utf-8"
            )
            config = {
                "benchmark_root": result_root,
                "report_root": root / "report",
                "answer_temperature": 0.1,
                "answer_max_tokens": 32768,
                "answer_timeout": 21600.0,
                "reasoning_mode": "auto",
                "reasoning_preserve": True,
                "judge_model": "test-judge",
            }
            model = ModelSpec("model-a", "Org/Repo", "model.gguf", 32768, "model")

            report = write_quality_report(
                config, [model], [{"question_id": 1, "category": "general"}]
            )

            self.assertEqual(
                report["models"][0]["answer_generation"]["reasoning"], "auto"
            )
            self.assertTrue((root / "report" / "quality.md").exists())

    def test_invalidates_judgments_for_regenerated_answers_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answers = root / "answers.jsonl"
            judgments = root / "judgments.jsonl"
            answers.write_text(
                "\n".join(
                    json.dumps({"question_id": question_id, "answer_id": answer_id})
                    for question_id, answer_id in (
                        (1, "new-answer"),
                        (2, "same-answer"),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            judgments.write_text(
                "\n".join(
                    json.dumps(
                        {"question_id": question_id, "answer_id": answer_id, "score": 8}
                    )
                    for question_id, answer_id in (
                        (1, "old-answer"),
                        (2, "same-answer"),
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            invalidate_stale_judgments(answers, judgments)

            remaining = [
                json.loads(line)
                for line in judgments.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                remaining, [{"question_id": 2, "answer_id": "same-answer", "score": 8}]
            )


if __name__ == "__main__":
    unittest.main()
