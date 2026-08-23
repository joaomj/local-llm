from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOW_LEVEL = ROOT / "scripts" / "llama_benchmark.py"
WRAPPER = ROOT / "scripts" / "throughput_context_search.py"


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


class ThroughputSearchCliTest(unittest.TestCase):
    def test_manifest_starts_with_historical_gemma_qat_quant_order(self) -> None:
        rows = [
            line.split("\t")
            for line in (ROOT / "scripts" / "throughput-models.tsv")
            .read_text(encoding="utf-8")
            .splitlines()
            if line and not line.startswith("#")
        ]

        self.assertEqual(
            [row[0] for row in rows[:4]],
            [
                "gemma4-e2b-it-qat-ud-q4-k-xl",
                "gemma4-e2b-it-qat-ud-q2-k-xl",
                "gemma4-e4b-it-qat-ud-q4-k-xl",
                "gemma4-e4b-it-qat-ud-q2-k-xl",
            ],
        )
        self.assertEqual(rows[0][2], "unsloth/gemma-4-E2B-it-qat-GGUF")
        self.assertEqual(rows[0][3], "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf")

    def test_low_level_times_generation_after_exact_kv_depth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            (models / "model.gguf").write_bytes(b"model")
            manifest = root / "models.tsv"
            manifest.write_text(
                "# id\trelative_model_path\tmaximum_context_tokens\n"
                "test-model\tmodel.gguf\t1024\n",
                encoding="utf-8",
            )
            args_log = root / "args.log"
            fake_bench = root / "llama-bench"
            write_executable(
                fake_bench,
                """#!/usr/bin/env bash
set -eu
if [[ "${1:-}" == "--help" ]]; then exit 0; fi
printf '%s\n' "$*" >> "$FAKE_ARGS_LOG"
prompt=-1
depth=-1
generated=-1
while (( $# > 0 )); do
  case "$1" in
    --n-prompt) prompt="$2"; shift ;;
    --n-depth) depth="$2"; shift ;;
    --n-gen) generated="$2"; shift ;;
  esac
  shift
done
if (( depth == 384 )); then speed=60; else speed=40; fi
printf '[{"n_prompt":%s,"n_depth":%s,"n_gen":%s,"avg_ts":%s,"stddev_ts":1,"samples_ts":[%s,%s,%s]}]\n' \
  "$prompt" "$depth" "$generated" "$speed" "$((speed - 1))" "$speed" "$((speed + 1))"
""",
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "FAKE_ARGS_LOG": str(args_log),
                    "LLAMA_BENCHMARK_BIN": str(fake_bench),
                    "LLAMA_BENCHMARK_SOURCE_MODEL_ROOT": str(models),
                    "LLAMA_BENCHMARK_LOCAL_MODEL_ROOT": str(models),
                    "LLAMA_BENCHMARK_MANIFEST": str(manifest),
                    "LLAMA_BENCHMARK_ROOT": str(root / "results"),
                    "LLAMA_BENCHMARK_ATTEMPTS": "1",
                    "LLAMA_BENCHMARK_REPETITIONS": "3",
                    "LLAMA_BENCHMARK_CLEANUP_WAIT_SECONDS": "0",
                    "LLAMA_BENCHMARK_CASE_TIMEOUT_SECONDS": "20",
                }
            )

            result = subprocess.run(
                [sys.executable, str(LOW_LEVEL)],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            arguments = args_log.read_text(encoding="utf-8")
            self.assertIn("--n-prompt 0 --n-gen 128 --n-depth 384", arguments)
            self.assertIn("--n-prompt 0 --n-gen 128 --n-depth 896", arguments)
            records = json.loads(
                (root / "results" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(records[0]["measurement_method"], "generation_at_kv_depth")
            self.assertEqual(records[0]["kv_depth_tokens"], 384)
            self.assertEqual(records[0]["actual_context_tokens"], 512)
            self.assertEqual(
                records[0]["generation_samples_tokens_per_second"], [59, 60, 61]
            )
            self.assertEqual(records[1]["qualifies_for_50_tps"], 0)
            self.assertEqual(records[0]["measurement_config"]["cache_type_k"], "q8_0")

            environment["LLAMA_BENCHMARK_CACHE_TYPE_K"] = "q4_0"
            rerun = subprocess.run(
                [sys.executable, str(LOW_LEVEL)],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(rerun.returncode, 0, rerun.stderr)
            self.assertEqual(len(args_log.read_text(encoding="utf-8").splitlines()), 4)
            updated = json.loads(
                (root / "results" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(updated[0]["measurement_config"]["cache_type_k"], "q4_0")

    def test_wrapper_descends_quant_then_stops_family_after_first_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "models.tsv"
            manifest.write_text(
                "# id\tfamily\trepo\tfile\tmaximum_context_tokens\tmodel_directory\n"
                "a-high\tfamily-a\tOrg/Repo\ta-high.gguf\t49152\ta-high\n"
                "a-low\tfamily-a\tOrg/Repo\ta-low.gguf\t49152\ta-low\n"
                "a-lower\tfamily-a\tOrg/Repo\ta-lower.gguf\t49152\ta-lower\n"
                "b-high\tfamily-b\tOrg/Repo\tb-high.gguf\t49152\tb-high\n",
                encoding="utf-8",
            )
            bin_dir = root / "bin"
            bin_dir.mkdir()
            write_executable(bin_dir / "llama-server", "#!/usr/bin/env bash\nexit 0\n")
            write_executable(
                bin_dir / "hf",
                """#!/usr/bin/env bash
set -eu
file="$3"
shift 3
while (( $# > 0 )); do
  if [[ "$1" == "--local-dir" ]]; then local_dir="$2"; break; fi
  shift
done
mkdir -p "$local_dir"
printf 'weights' > "$local_dir/$file"
""",
            )
            fake_low_level = root / "fake-server-runner.sh"
            write_executable(
                fake_low_level,
                """#!/usr/bin/env bash
set -eu
id="$LLAMA_SERVER_BENCHMARK_MODEL_ID"
printf '%s\n' "$id" >> "$FAKE_CALL_LOG"
mkdir -p "$LLAMA_SERVER_BENCHMARK_ROOT"
if [[ "$id" == "a-high" ]]; then met=false; configured=0; occupied=0; else met=true; configured=49152; occupied=8192; fi
printf '{"measurement_contract":"llama_server_capacity_and_prompt-v1","model_id":"%s","minimum_requirement_met":%s,"largest_passing_configured_context_tokens":%s,"largest_passing_actual_prompt_tokens":%s,"weights_retained":true,"measurements":[]}\n' "$id" "$met" "$configured" "$occupied" > "$LLAMA_SERVER_BENCHMARK_ROOT/summary.json"
""",
            )
            call_log = root / "calls.log"
            environment = os.environ.copy()
            environment.update(
                {
                    "FAKE_CALL_LOG": str(call_log),
                    "THROUGHPUT_MANIFEST": str(manifest),
                    "THROUGHPUT_BENCHMARK_ROOT": str(root / "results"),
                    "THROUGHPUT_SCRATCH_ROOT": str(root / "models"),
                    "THROUGHPUT_HF_BIN": str(bin_dir / "hf"),
                    "THROUGHPUT_LLAMA_SERVER_BENCHMARK": str(fake_low_level),
                    "LLAMA_SERVER_BENCHMARK_BIN": str(bin_dir / "llama-server"),
                }
            )

            result = subprocess.run(
                [sys.executable, str(WRAPPER)],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                call_log.read_text(encoding="utf-8").splitlines(),
                ["a-high", "a-low", "b-high"],
            )
            self.assertIn(
                "FAMILY SELECTED: family=family-a candidate=a-low", result.stdout
            )
            self.assertIn("QUANT SKIP: a-lower", result.stdout)
            self.assertTrue((root / "models" / "a-high" / "a-high.gguf").exists())
            self.assertTrue((root / "models" / "a-low" / "a-low.gguf").exists())

    def test_wrapper_stops_and_retains_weights_on_operational_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "models.tsv"
            manifest.write_text(
                "# id\tfamily\trepo\tfile\tmaximum_context_tokens\tmodel_directory\n"
                "first\tfamily-a\tOrg/Repo\tfirst.gguf\t49152\tfirst\n"
                "second\tfamily-b\tOrg/Repo\tsecond.gguf\t49152\tsecond\n",
                encoding="utf-8",
            )
            bin_dir = root / "bin"
            bin_dir.mkdir()
            write_executable(bin_dir / "llama-server", "#!/usr/bin/env bash\nexit 0\n")
            write_executable(
                bin_dir / "hf",
                """#!/usr/bin/env bash
set -eu
file="$3"
shift 3
while (( $# > 0 )); do
  if [[ "$1" == "--local-dir" ]]; then local_dir="$2"; break; fi
  shift
done
mkdir -p "$local_dir"
printf 'weights' > "$local_dir/$file"
""",
            )
            fake_low_level = root / "fake-server-runner.sh"
            write_executable(fake_low_level, "#!/usr/bin/env bash\nexit 1\n")
            environment = os.environ.copy()
            environment.update(
                {
                    "THROUGHPUT_MANIFEST": str(manifest),
                    "THROUGHPUT_BENCHMARK_ROOT": str(root / "results"),
                    "THROUGHPUT_SCRATCH_ROOT": str(root / "models"),
                    "THROUGHPUT_HF_BIN": str(bin_dir / "hf"),
                    "THROUGHPUT_LLAMA_SERVER_BENCHMARK": str(fake_low_level),
                    "LLAMA_SERVER_BENCHMARK_BIN": str(bin_dir / "llama-server"),
                }
            )

            result = subprocess.run(
                [sys.executable, str(WRAPPER)],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            self.assertTrue((root / "models" / "first" / "first.gguf").exists())
            self.assertFalse((root / "models" / "second").exists())
            self.assertIn("SEARCH STOPPED", result.stdout)

    def test_wrapper_stops_before_download_when_storage_is_unsafe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "models.tsv"
            manifest.write_text(
                "# id\tfamily\trepo\tfile\tmaximum_context_tokens\tmodel_directory\n"
                "first\tfamily-a\tOrg/Repo\tfirst.gguf\t49152\tfirst\n",
                encoding="utf-8",
            )
            bin_dir = root / "bin"
            bin_dir.mkdir()
            hf_log = root / "hf.log"
            write_executable(
                bin_dir / "hf",
                f"#!/usr/bin/env bash\nprintf called > {hf_log}\n",
            )
            write_executable(bin_dir / "llama-server", "#!/usr/bin/env bash\nexit 0\n")
            fake_runner = root / "runner.sh"
            write_executable(fake_runner, "#!/usr/bin/env bash\nexit 0\n")
            environment = os.environ.copy()
            environment.update(
                {
                    "THROUGHPUT_MANIFEST": str(manifest),
                    "THROUGHPUT_BENCHMARK_ROOT": str(root / "results"),
                    "THROUGHPUT_SCRATCH_ROOT": str(root / "models"),
                    "THROUGHPUT_HF_BIN": str(bin_dir / "hf"),
                    "THROUGHPUT_LLAMA_SERVER_BENCHMARK": str(fake_runner),
                    "LLAMA_SERVER_BENCHMARK_BIN": str(bin_dir / "llama-server"),
                    "THROUGHPUT_MIN_FREE_BYTES": str(10**18),
                }
            )

            result = subprocess.run(
                [sys.executable, str(WRAPPER)],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("storage safety floor", result.stdout)
            self.assertFalse(hf_log.exists())
            self.assertFalse((root / "models" / "first").exists())


if __name__ == "__main__":
    unittest.main()
