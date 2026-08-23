from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "llama_server_benchmark.py"


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


class LlamaServerBenchmarkCliTest(unittest.TestCase):
    def test_records_capacity_and_real_tool_conversation_prompt_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            model.write_bytes(b"model")
            invocations = root / "server-invocations.jsonl"
            fake_server = root / "llama-server"
            write_executable(
                fake_server,
                r"""#!/usr/bin/env python3
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--ctx-size", type=int, required=True)
args, unknown = parser.parse_known_args()
with open(os.environ["FAKE_SERVER_INVOCATIONS"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"capacity": args.ctx_size, "arguments": unknown}) + "\n")

class Handler(BaseHTTPRequestHandler):
    prompt_counts = {}

    def send_json(self, status, value):
        encoded = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path == "/v1/models":
            self.send_json(200, {"data": [{"id": "test-model"}]})
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        if self.path == "/tokenize":
            self.send_json(200, {"tokens": payload.get("content", "").split()})
            return
        if self.path != "/v1/chat/completions":
            self.send_json(404, {"error": "not found"})
            return
        messages = payload["messages"]
        actual_prompt_n = sum(len(str(message.get("content", "")).split()) for message in messages) + 32
        if actual_prompt_n > 1000:
            roles = {message["role"] for message in messages}
            has_tool_call = any(message.get("tool_calls") for message in messages)
            if "tool" not in roles or not has_tool_call:
                self.send_json(400, {"error": "missing tool transcript"})
                return
        prompt_key = "occupied" if actual_prompt_n > 1000 else "short"
        prompt_count = self.prompt_counts.get(prompt_key, 0)
        self.prompt_counts[prompt_key] = prompt_count + 1
        evaluated_prompt_n = actual_prompt_n if prompt_count == 0 else 1
        speed = 55.0 if actual_prompt_n <= 9000 else 45.0
        predicted_n = payload["max_tokens"]
        self.send_json(200, {
            "choices": [{"message": {"role": "assistant", "content": "benchmark response"}}],
            "usage": {
                "prompt_tokens": actual_prompt_n,
                "completion_tokens": predicted_n,
                "prompt_tokens_details": {"cached_tokens": actual_prompt_n - evaluated_prompt_n},
            },
            "timings": {
                "prompt_n": evaluated_prompt_n,
                "prompt_per_second": 200.0,
                "predicted_n": predicted_n,
                "predicted_per_second": speed,
            },
        })

    def log_message(self, *_args):
        pass

ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
""",
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "FAKE_SERVER_INVOCATIONS": str(invocations),
                    "LLAMA_SERVER_BENCHMARK_BIN": str(fake_server),
                    "LLAMA_SERVER_BENCHMARK_MODEL_PATH": str(model),
                    "LLAMA_SERVER_BENCHMARK_MODEL_ID": "test-model",
                    "LLAMA_SERVER_BENCHMARK_MAX_CONTEXT": "49152",
                    "LLAMA_SERVER_BENCHMARK_CAPACITIES": "32768,49152",
                    "LLAMA_SERVER_BENCHMARK_PROMPT_TARGETS": "8192,16384",
                    "LLAMA_SERVER_BENCHMARK_GENERATION_TOKENS": "32",
                    "LLAMA_SERVER_BENCHMARK_REPETITIONS": "3",
                    "LLAMA_SERVER_BENCHMARK_ROOT": str(root / "results"),
                    "LLAMA_SERVER_BENCHMARK_STARTUP_TIMEOUT_SECONDS": "5",
                    "LLAMA_SERVER_BENCHMARK_REQUEST_TIMEOUT_SECONDS": "5",
                    "LLAMA_SERVER_BENCHMARK_SHUTDOWN_TIMEOUT_SECONDS": "2",
                }
            )

            result = subprocess.run(
                [sys.executable, str(RUNNER)],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(
                (root / "results" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["measurement_contract"], "llama_server_capacity_and_prompt-v3"
            )
            self.assertEqual(
                summary["largest_passing_configured_context_tokens"], 49152
            )
            self.assertGreaterEqual(
                summary["largest_passing_actual_prompt_tokens"], 8192
            )
            self.assertLess(summary["largest_passing_actual_prompt_tokens"], 16384)
            self.assertTrue(summary["minimum_requirement_met"])
            occupied = [
                item
                for item in summary["measurements"]
                if item["scenario"] == "tool_conversation"
            ]
            self.assertTrue(occupied)
            self.assertTrue(
                all(item["configured_context_tokens"] >= 32768 for item in occupied)
            )
            self.assertTrue(all(item["actual_prompt_tokens"] > 0 for item in occupied))
            self.assertTrue(
                all(
                    item["actual_prompt_tokens"] > item["evaluated_prompt_tokens"]
                    for item in occupied
                )
            )
            self.assertTrue(
                all(
                    len(item["generation_samples_tokens_per_second"]) == 3
                    for item in summary["measurements"]
                )
            )
            server_runs = [
                json.loads(line)
                for line in invocations.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([item["capacity"] for item in server_runs], [32768, 49152])
            self.assertTrue(
                all("--flash-attn" in item["arguments"] for item in server_runs)
            )
            self.assertTrue(
                all("--parallel" in item["arguments"] for item in server_runs)
            )

            environment.update(
                {
                    "LLAMA_SERVER_BENCHMARK_MAX_CONTEXT": "32768",
                    "LLAMA_SERVER_BENCHMARK_CAPACITIES": "32768",
                    "LLAMA_SERVER_BENCHMARK_PROMPT_TARGETS": "8192",
                    "LLAMA_SERVER_BENCHMARK_MIN_GENERATION_TPS": "56",
                    "LLAMA_SERVER_BENCHMARK_ROOT": str(root / "failing-results"),
                }
            )
            failing_result = subprocess.run(
                [sys.executable, str(RUNNER)],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )

            self.assertEqual(failing_result.returncode, 0, failing_result.stderr)
            failing_summary = json.loads(
                (root / "failing-results" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertFalse(failing_summary["minimum_requirement_met"])
            self.assertEqual(
                [item["scenario"] for item in failing_summary["measurements"]],
                ["short", "tool_conversation"],
            )


if __name__ == "__main__":
    unittest.main()
