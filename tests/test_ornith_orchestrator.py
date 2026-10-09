from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.ornith_orchestrator import mlx_bench_metrics, speed_metrics


class OrnithOrchestratorTest(unittest.TestCase):
    def test_gguf_speed_gate_requires_short_and_occupied_wins(self) -> None:
        summary = {
            "measurements": [
                {
                    "scenario": "short",
                    "configured_context_tokens": 32768,
                    "generation_tokens_per_second": 118.0,
                },
                {
                    "scenario": "tool_conversation",
                    "configured_context_tokens": 32768,
                    "target_prompt_tokens": 8192,
                    "generation_tokens_per_second": 101.0,
                },
            ]
        }

        self.assertFalse(speed_metrics(summary)["speed_win"])
        summary["measurements"][1]["generation_tokens_per_second"] = 102.0
        self.assertTrue(speed_metrics(summary)["speed_win"])

    def test_mlx_speed_gate_requires_short_and_occupied_wins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bench_raw.json"
            rows = [
                {"scenario": "plain", "label": "first-call", "gen_tps": 118.0},
                {
                    "scenario": "occupied",
                    "label": "first-call",
                    "gen_tps": 101.0,
                },
            ]
            path.write_text(json.dumps(rows), encoding="utf-8")

            self.assertFalse(mlx_bench_metrics(path)["speed_win"])
            rows[1]["gen_tps"] = 102.0
            path.write_text(json.dumps(rows), encoding="utf-8")
            self.assertTrue(mlx_bench_metrics(path)["speed_win"])


if __name__ == "__main__":
    unittest.main()
