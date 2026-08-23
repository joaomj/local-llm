from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "mt_bench_judge.py"


class JudgeHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict]] = []
    fail_first = False
    always_fail = False

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        self.requests.append(payload)
        if self.always_fail:
            body = b'{"error":{"message":"rate limited"}}'
            self.send_response(429)
            self.send_header("Retry-After", "0.01")
            self.send_header("X-RateLimit-Remaining", "0")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.fail_first and len(self.requests) == 1:
            body = b'{"error":{"message":"temporary failure"}}'
            self.send_response(503)
            self.send_header("Retry-After", "0.01")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        body = {
            "id": "chatcmpl-judge",
            "object": "chat.completion",
            "created": 0,
            "model": payload["model"],
            "provider": "test-provider",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '{"score": 8, "rationale": "Useful answer."}',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 8,
                "total_tokens": 28,
                "cost": 0.001,
            },
        }
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:
        return


class MtBenchJudgeCliTest(unittest.TestCase):
    def setUp(self) -> None:
        JudgeHandler.requests = []
        JudgeHandler.fail_first = True
        JudgeHandler.always_fail = False
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), JudgeHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_check_config_reads_dotenv_without_printing_credential(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dotenv = Path(directory) / ".env"
            dotenv.write_text(
                "OPENROUTER_API_KEY=test-dotenv-secret\n", encoding="utf-8"
            )
            environment = os.environ.copy()
            environment.pop("OPENROUTER_API_KEY", None)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--check-config",
                    "--dotenv",
                    str(dotenv),
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("credential configuration is present", result.stdout)
            self.assertNotIn("test-dotenv-secret", result.stdout + result.stderr)

    def test_retries_and_persists_structured_judgment_without_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            questions = directory_path / "questions.jsonl"
            answers = directory_path / "answers.jsonl"
            judgments = directory_path / "judgments.jsonl"
            questions.write_text(
                json.dumps(
                    {
                        "question_id": 1,
                        "category": "general",
                        "turns": ["first question", "second question"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            answers.write_text(
                json.dumps(
                    {
                        "question_id": 1,
                        "answer_id": "answer-1",
                        "model_id": "candidate-model",
                        "choices": [
                            {"index": 0, "turns": ["first answer", "second answer"]}
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["OPENROUTER_API_KEY"] = "test-secret-that-must-not-escape"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--answers",
                    str(answers),
                    "--questions",
                    str(questions),
                    "--output",
                    str(judgments),
                    "--base-url",
                    f"http://127.0.0.1:{self.server.server_port}/v1",
                    "--judge-model",
                    "google/gemini-3.7-flash",
                    "--retry-base-seconds",
                    "0.01",
                    "--max-cost-usd",
                    "0.01",
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(JudgeHandler.requests), 2)
            self.assertEqual(
                JudgeHandler.requests[-1]["model"], "google/gemini-3.7-flash"
            )
            self.assertEqual(JudgeHandler.requests[-1]["temperature"], 0)
            self.assertEqual(
                JudgeHandler.requests[-1]["response_format"]["json_schema"]["strict"],
                True,
            )
            record = json.loads(judgments.read_text(encoding="utf-8").strip())
            self.assertEqual(record["score"], 8)
            self.assertEqual(record["provider"], "test-provider")
            self.assertEqual(record["usage"]["cost"], 0.001)
            self.assertEqual(record["attempts"], 2)
            self.assertNotIn(
                "test-secret-that-must-not-escape",
                judgments.read_text(encoding="utf-8"),
            )
            self.assertNotIn(
                "test-secret-that-must-not-escape", result.stdout + result.stderr
            )

    def test_refuses_to_send_incomplete_answer_to_judge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            questions = directory_path / "questions.jsonl"
            answers = directory_path / "answers.jsonl"
            judgments = directory_path / "judgments.jsonl"
            questions.write_text(
                json.dumps({"question_id": 1, "turns": ["first", "second"]}) + "\n",
                encoding="utf-8",
            )
            answers.write_text(
                json.dumps(
                    {
                        "question_id": 1,
                        "answer_id": "answer-1",
                        "model_id": "candidate-model",
                        "choices": [{"index": 0, "turns": ["first answer", ""]}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["OPENROUTER_API_KEY"] = "test-secret-that-must-not-escape"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--answers",
                    str(answers),
                    "--questions",
                    str(questions),
                    "--output",
                    str(judgments),
                    "--base-url",
                    f"http://127.0.0.1:{self.server.server_port}/v1",
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(JudgeHandler.requests, [])
            self.assertIn("incomplete or truncated", result.stderr)

    def test_rate_limit_failure_is_logged_for_resume_diagnosis(self) -> None:
        JudgeHandler.fail_first = False
        JudgeHandler.always_fail = True
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            questions = directory_path / "questions.jsonl"
            answers = directory_path / "answers.jsonl"
            judgments = directory_path / "judgments.jsonl"
            questions.write_text(
                json.dumps({"question_id": 1, "turns": ["first", "second"]}) + "\n",
                encoding="utf-8",
            )
            answers.write_text(
                json.dumps(
                    {
                        "question_id": 1,
                        "answer_id": "answer-1",
                        "model_id": "candidate-model",
                        "choices": [
                            {"index": 0, "turns": ["first answer", "second answer"]}
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["OPENROUTER_API_KEY"] = "test-secret-that-must-not-escape"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--answers",
                    str(answers),
                    "--questions",
                    str(questions),
                    "--output",
                    str(judgments),
                    "--base-url",
                    f"http://127.0.0.1:{self.server.server_port}/v1",
                    "--max-retries",
                    "0",
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            error_log = judgments.with_name("judgments.errors.jsonl")
            error_record = json.loads(error_log.read_text(encoding="utf-8").strip())
            self.assertEqual(error_record["status_code"], 429)
            self.assertEqual(error_record["rate_limit_headers"]["retry-after"], "0.01")
            self.assertNotIn(
                "test-secret-that-must-not-escape",
                error_log.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
