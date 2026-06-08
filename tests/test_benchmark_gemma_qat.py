from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_gemma_qat.py"


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_gemma_qat", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load benchmark module")
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_gemma_qat"] = module
    spec.loader.exec_module(module)
    return module


class BenchmarkConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_benchmark_module()

    def test_active_comparison_only_runs_e2b_baseline(self) -> None:
        variants = self.module.VARIANTS

        self.assertEqual(["gemma-e2b-q4-49k"], self.module.COMPARISON_VARIANTS)
        self.assertIn("gemma-e2b-q4-49k", variants)
        self.assertNotIn("gemma-e4b-q4-49k", variants)
        self.assertNotIn("qwen35-9b-ud-q4-49k", variants)
        self.assertNotIn("qwen35-9b-ud-q5-49k", variants)

        command = self.module.build_server_command(variants["gemma-e2b-q4-49k"], 8080)

        self.assertIn("unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL", command)
        self.assertIn("49152", command)
        self.assertIn('{"enable_thinking":false}', command)

    def test_server_command_uses_proven_no_thinking_flash_attention_flags(self) -> None:
        variant = self.module.VARIANTS["gemma-e2b-q4-49k"]
        command = self.module.build_server_command(variant, 8080)

        expected_flags = [
            "--perf",
            "--flash-attn",
            "on",
            "--reasoning",
            "off",
            "--reasoning-budget",
            "0",
            "--chat-template-kwargs",
            '{"enable_thinking":false}',
            "--parallel",
            "1",
        ]
        for flag in expected_flags:
            self.assertIn(flag, command)

    def test_quality_suite_contains_structured_checks(self) -> None:
        prompts = self.module.QUALITY_PROMPTS

        self.assertGreaterEqual(len(prompts), 5)
        self.assertTrue(any(prompt.expected_label == "bug_report" for prompt in prompts))
        self.assertTrue(any(prompt.requires_json for prompt in prompts))


if __name__ == "__main__":
    unittest.main()
