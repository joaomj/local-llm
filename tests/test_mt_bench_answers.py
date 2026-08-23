from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "mt_bench_answers.py"


class CompletionHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict]] = []
    response_delay = 0.0

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        self.requests.append(payload)
        time.sleep(self.response_delay)
        body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "test answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except BrokenPipeError:
            pass

    def log_message(self, format: str, *args) -> None:
        return


class MtBenchAnswersCliTest(unittest.TestCase):
    def setUp(self) -> None:
        CompletionHandler.requests = []
        CompletionHandler.response_delay = 0.0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), CompletionHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_regenerates_incomplete_answer_and_generates_two_turn_answers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            questions = directory_path / "questions.jsonl"
            output = directory_path / "answers.jsonl"
            questions.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "question_id": 1,
                                "category": "general",
                                "turns": ["one", "two"],
                            }
                        ),
                        json.dumps(
                            {
                                "question_id": 2,
                                "category": "general",
                                "turns": ["three", "four"],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output.write_text(
                json.dumps(
                    {
                        "question_id": 1,
                        "answer_id": "existing",
                        "model_id": "test-model",
                        "choices": [{"index": 0, "turns": ["old one", ""]}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--model-id",
                    "test-model",
                    "--questions",
                    str(questions),
                    "--output",
                    str(output),
                    "--server-url",
                    f"http://127.0.0.1:{self.server.server_port}/v1",
                    "--max-tokens",
                    "16",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(CompletionHandler.requests), 4)
            self.assertEqual(
                [
                    message["role"]
                    for message in CompletionHandler.requests[-1]["messages"]
                ],
                ["user", "assistant", "user"],
            )
            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([record["question_id"] for record in records], [1, 2])
            self.assertEqual(
                records[1]["choices"][0]["turns"], ["test answer", "test answer"]
            )
            self.assertEqual(records[1]["generation"]["usage"][0]["total_tokens"], 5)

    def test_timeout_does_not_retry_generation(self) -> None:
        CompletionHandler.response_delay = 0.2
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            questions = directory_path / "questions.jsonl"
            output = directory_path / "answers.jsonl"
            questions.write_text(
                json.dumps({"question_id": 1, "turns": ["one", "two"]}) + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--model-id",
                    "test-model",
                    "--questions",
                    str(questions),
                    "--output",
                    str(output),
                    "--server-url",
                    f"http://127.0.0.1:{self.server.server_port}/v1",
                    "--timeout",
                    "0.05",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(len(CompletionHandler.requests), 1)
            self.assertIn("Request timed out", result.stderr)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
