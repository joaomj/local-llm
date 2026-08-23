#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT="$ROOT_DIR/scripts/lfm-qad-benchmark.sh"
TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/lfm-qad-test.XXXXXX")"
trap 'rm -rf "$TEMP_ROOT"' EXIT

output="$TEMP_ROOT/dry-run.out"
if ! LFM_QAD_BENCHMARK_ROOT="$TEMP_ROOT/benchmark" \
  LFM_QAD_SCRATCH_ROOT="$TEMP_ROOT/scratch" \
  bash "$SCRIPT" --dry-run > "$output"; then
  printf '%s\n' "dry-run failed" >&2
  exit 1
fi

for expected in \
  'lfm2.5-230m-qad-q4_0.*context=32768' \
  'lfm2.5-350m-qad-q4_0.*context=32768' \
  'lfm2.5-1.2b-instruct-qad-q4_0.*context=32768' \
  'lfm2.5-2.6b-qad-q4_0.*context=131072'; do
  if ! grep -Eq "$expected" "$output"; then
    printf '%s\n' "missing dry-run plan: $expected" >&2
    exit 1
  fi
done

if [[ -e "$TEMP_ROOT/benchmark" || -e "$TEMP_ROOT/scratch" ]]; then
  printf '%s\n' "dry-run created local state or scratch storage" >&2
  exit 1
fi

single_output="$TEMP_ROOT/single.out"
if ! LFM_QAD_BENCHMARK_ROOT="$TEMP_ROOT/single-benchmark" \
  LFM_QAD_SCRATCH_ROOT="$TEMP_ROOT/single-scratch" \
  bash "$SCRIPT" --dry-run --model lfm2.5-2.6b-qad-q4_0 > "$single_output"; then
  printf '%s\n' "single-model dry-run failed" >&2
  exit 1
fi

if [[ "$(grep -c '^\[.*\] PLAN: model=' "$single_output")" -ne 1 ]]; then
  printf '%s\n' "single-model dry-run planned an unexpected number of models" >&2
  exit 1
fi
grep -Eq 'lfm2.5-2.6b-qad-q4_0.*context=131072' "$single_output"
printf '%s\n' "LFM QAD dry-run checks passed"
