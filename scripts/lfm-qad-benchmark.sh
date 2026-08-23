#!/usr/bin/env bash

# Run the LFM QAD benchmark one GGUF at a time.

if [[ -z "${BASH_VERSION:-}" ]]; then
  printf '%s\n' "ERROR: run this script with Bash: ./scripts/lfm-qad-benchmark.sh" >&2
  exit 2
fi

set -Eeuo pipefail
IFS=$'\n\t'
umask 022

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_VERSION="1"
readonly ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
readonly DEFAULT_MANIFEST="$ROOT_DIR/benchmarks/manifests/lfm-qad-models.tsv"
readonly DEFAULT_QUESTIONS="$ROOT_DIR/benchmarks/mt-bench/question.jsonl"
readonly DEFAULT_BENCHMARK_ROOT="$ROOT_DIR/.benchmark"
readonly DEFAULT_SCRATCH_ROOT="$DEFAULT_BENCHMARK_ROOT/lfm-qad-models"
readonly DEFAULT_HF_BIN="hf"
readonly DEFAULT_LLAMA_BENCH_BIN="llama-bench"
readonly DEFAULT_LLAMA_SERVER_BIN="llama-server"
readonly DEFAULT_UV_BIN="uv"
readonly DEFAULT_PORT="18080"
readonly DEFAULT_SERVER_HOST="127.0.0.1"
readonly DEFAULT_SERVER_TIMEOUT_SECONDS="300"
readonly DEFAULT_DOWNLOAD_ATTEMPTS="3"
readonly DEFAULT_DOWNLOAD_RETRY_SECONDS="15"
readonly DEFAULT_SERVER_GPU_LAYERS="99"
readonly DEFAULT_ANSWER_MAX_TOKENS="1024"
readonly DEFAULT_JUDGE_MAX_COST_USD=""
readonly DEFAULT_JUDGE_BATCH_RETRIES="2"
readonly DEFAULT_JUDGE_API_RETRIES="5"
readonly DEFAULT_JUDGE_RETRY_BASE_SECONDS="5"
readonly DEFAULT_JUDGE_POLL_SECONDS="30"
readonly DEFAULT_REPORT_ROOT="$ROOT_DIR/results/lfm-qad"

MANIFEST="${LFM_QAD_MANIFEST:-$DEFAULT_MANIFEST}"
QUESTIONS="${LFM_QAD_QUESTIONS:-$DEFAULT_QUESTIONS}"
BENCHMARK_ROOT="${LFM_QAD_BENCHMARK_ROOT:-$DEFAULT_BENCHMARK_ROOT}"
BENCHMARK_ROOT="${BENCHMARK_ROOT%/}"
SCRATCH_ROOT="${LFM_QAD_SCRATCH_ROOT:-$DEFAULT_SCRATCH_ROOT}"
SCRATCH_ROOT="${SCRATCH_ROOT%/}"
REPORT_ROOT="${LFM_QAD_REPORT_ROOT:-$DEFAULT_REPORT_ROOT}"
REPORT_ROOT="${REPORT_ROOT%/}"
HF_BIN="${LFM_QAD_HF_BIN:-$DEFAULT_HF_BIN}"
LLAMA_BENCH_BIN="${LFM_QAD_LLAMA_BENCH_BIN:-$DEFAULT_LLAMA_BENCH_BIN}"
LLAMA_SERVER_BIN="${LFM_QAD_LLAMA_SERVER_BIN:-$DEFAULT_LLAMA_SERVER_BIN}"
UV_BIN="${LFM_QAD_UV_BIN:-$DEFAULT_UV_BIN}"
SERVER_HOST="${LFM_QAD_SERVER_HOST:-$DEFAULT_SERVER_HOST}"
SERVER_PORT="${LFM_QAD_PORT:-$DEFAULT_PORT}"
SERVER_TIMEOUT_SECONDS="${LFM_QAD_SERVER_TIMEOUT_SECONDS:-$DEFAULT_SERVER_TIMEOUT_SECONDS}"
DOWNLOAD_ATTEMPTS="${LFM_QAD_DOWNLOAD_ATTEMPTS:-$DEFAULT_DOWNLOAD_ATTEMPTS}"
DOWNLOAD_RETRY_SECONDS="${LFM_QAD_DOWNLOAD_RETRY_SECONDS:-$DEFAULT_DOWNLOAD_RETRY_SECONDS}"
SERVER_GPU_LAYERS="${LFM_QAD_SERVER_GPU_LAYERS:-$DEFAULT_SERVER_GPU_LAYERS}"
ANSWER_MAX_TOKENS="${LFM_QAD_ANSWER_MAX_TOKENS:-$DEFAULT_ANSWER_MAX_TOKENS}"
JUDGE_MAX_COST_USD="${LFM_QAD_JUDGE_MAX_COST_USD:-$DEFAULT_JUDGE_MAX_COST_USD}"
JUDGE_BATCH_RETRIES="${LFM_QAD_JUDGE_BATCH_RETRIES:-$DEFAULT_JUDGE_BATCH_RETRIES}"
JUDGE_API_RETRIES="${LFM_QAD_JUDGE_API_RETRIES:-$DEFAULT_JUDGE_API_RETRIES}"
JUDGE_RETRY_BASE_SECONDS="${LFM_QAD_JUDGE_RETRY_BASE_SECONDS:-$DEFAULT_JUDGE_RETRY_BASE_SECONDS}"
JUDGE_POLL_SECONDS="${LFM_QAD_JUDGE_POLL_SECONDS:-$DEFAULT_JUDGE_POLL_SECONDS}"
JUDGE_MODEL="${LFM_QAD_JUDGE_MODEL:-google/gemini-3.7-flash:batch}"

STATE_ROOT="$BENCHMARK_ROOT/lfm-qad-state"
RESULTS_ROOT="$BENCHMARK_ROOT/lfm-qad-results"
MANIFEST_ROOT="$STATE_ROOT/manifests"
LOCK_DIR="$STATE_ROOT/.lock"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_STATE_FILE="$STATE_ROOT/run.state"
LOG_DIR="$STATE_ROOT/runs"
LOG_FILE="$LOG_DIR/$RUN_ID.log"

LLAMA_BENCH_SCRIPT="$ROOT_DIR/scripts/llama_benchmark.py"
ANSWER_SCRIPT="$ROOT_DIR/scripts/mt_bench_answers.py"
JUDGE_SCRIPT="$ROOT_DIR/scripts/mt_bench_batch_judge.py"
REPORT_SCRIPT="$ROOT_DIR/scripts/lfm_qad_report.py"

DRY_RUN=0
REQUESTED_MODEL=""
LOCK_ACQUIRED=0
RUN_STATUS="starting"
CURRENT_MODEL_ID=""
SERVER_PID=""

MODEL_IDS=()
MODEL_REPOS=()
MODEL_FILES=()
MODEL_MAX_CONTEXTS=()
MODEL_DIRS=()
EXPECTED_QUESTION_IDS_JSON=""

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [options]

Downloads, benchmarks, chats, judges, reports, and removes one LFM QAD GGUF at a time.

Options:
  --dry-run              Validate manifests and print the planned workflow.
  --model ID             Process only one manifest model.
  --help                 Show this help.

Environment overrides:
  LFM_QAD_MANIFEST             Model manifest (default: $DEFAULT_MANIFEST)
  LFM_QAD_QUESTIONS            MT-Bench questions (default: $DEFAULT_QUESTIONS)
  LFM_QAD_BENCHMARK_ROOT       Persistent state and results root (default: $DEFAULT_BENCHMARK_ROOT)
  LFM_QAD_SCRATCH_ROOT         Owned temporary model root (default: $DEFAULT_SCRATCH_ROOT)
  LFM_QAD_REPORT_ROOT           Final report directory (default: $DEFAULT_REPORT_ROOT)
  LFM_QAD_HF_BIN               Hugging Face CLI (default: $DEFAULT_HF_BIN)
  LFM_QAD_LLAMA_BENCH_BIN     llama-bench command (default: $DEFAULT_LLAMA_BENCH_BIN)
  LFM_QAD_LLAMA_SERVER_BIN    llama-server command (default: $DEFAULT_LLAMA_SERVER_BIN)
  LFM_QAD_UV_BIN              uv command (default: $DEFAULT_UV_BIN)
  LFM_QAD_PORT                Local server port (default: $DEFAULT_PORT)
  LFM_QAD_JUDGE_MAX_COST_USD  Optional OpenRouter spend ceiling
  LFM_QAD_JUDGE_BATCH_RETRIES  Retries for failed batch items (default: $DEFAULT_JUDGE_BATCH_RETRIES)
  LFM_QAD_JUDGE_API_RETRIES     Retries for batch API calls (default: $DEFAULT_JUDGE_API_RETRIES)
  LFM_QAD_JUDGE_RETRY_BASE_SECONDS  Initial retry delay (default: $DEFAULT_JUDGE_RETRY_BASE_SECONDS)
  LFM_QAD_JUDGE_POLL_SECONDS    Batch polling interval (default: $DEFAULT_JUDGE_POLL_SECONDS)

The real run requires hf, llama-bench, llama-server, curl, jq, and uv. Dry-run
does not start any of them. The API key is read by the judge from the
OPENROUTER_API_KEY environment variable or the repository .env file.
The final JSON and Markdown reports are written under:
  $REPORT_ROOT
EOF
}

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

fatal() {
  log "ERROR: $*"
  exit 2
}

validate_integer() {
  local name="$1"
  local value="$2"
  local minimum="$3"
  [[ "$value" =~ ^[0-9]+$ ]] || fatal "$name must be an integer >= $minimum (got '$value')"
  (( value >= minimum )) || fatal "$name must be >= $minimum (got '$value')"
}

parse_arguments() {
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --dry-run)
        DRY_RUN=1
        ;;
      --model)
        shift
        [[ "$#" -gt 0 ]] || fatal "--model requires an ID"
        REQUESTED_MODEL="$1"
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        usage >&2
        fatal "unknown argument: $1"
        ;;
    esac
    shift
  done
}

write_state() {
  local path="$1"
  local status="$2"
  local message="$3"
  local temporary="$path.tmp.$$"
  mkdir -p "$(dirname "$path")"
  {
    printf 'version=%s\n' "$SCRIPT_VERSION"
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'status=%s\n' "$status"
    printf 'updated_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'message=%s\n' "$message"
  } > "$temporary"
  mv -f "$temporary" "$path"
}

write_model_state() {
  local model_id="$1"
  local status="$2"
  local message="$3"
  write_state "$STATE_ROOT/models/$model_id.state" "$status" "$message"
}

release_lock() {
  if [[ "$LOCK_ACQUIRED" -ne 1 ]]; then
    return 0
  fi
  rm -f "$LOCK_DIR/pid" "$LOCK_DIR/run_id"
  rmdir "$LOCK_DIR" 2>/dev/null || true
  LOCK_ACQUIRED=0
}

acquire_lock() {
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" > "$LOCK_DIR/pid"
    printf '%s\n' "$RUN_ID" > "$LOCK_DIR/run_id"
    LOCK_ACQUIRED=1
    return 0
  fi

  local existing_pid=""
  if [[ -r "$LOCK_DIR/pid" ]]; then
    read -r existing_pid < "$LOCK_DIR/pid" || existing_pid=""
  fi
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" >/dev/null 2>&1; then
    fatal "another LFM QAD benchmark is running (pid $existing_pid)"
  fi
  rm -f "$LOCK_DIR/pid" "$LOCK_DIR/run_id"
  rmdir "$LOCK_DIR" 2>/dev/null || fatal "could not remove stale lock at $LOCK_DIR"
  acquire_lock
}

stop_server() {
  if [[ ! "$SERVER_PID" =~ ^[0-9]+$ ]]; then
    SERVER_PID=""
    return 0
  fi
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    log "Stopping llama-server pid=$SERVER_PID"
    kill -TERM "$SERVER_PID" >/dev/null 2>&1 || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "$SERVER_PID" >/dev/null 2>&1 || break
      sleep 1
    done
    if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      kill -KILL "$SERVER_PID" >/dev/null 2>&1 || true
    fi
  fi
  wait "$SERVER_PID" >/dev/null 2>&1 || true
  SERVER_PID=""
}

on_exit() {
  local exit_code="$?"
  stop_server || true
  if [[ "$RUN_STATUS" == "running" && "$exit_code" -ne 0 && -d "$STATE_ROOT" ]]; then
    write_state "$RUN_STATE_FILE" "interrupted" "benchmark stopped with exit status $exit_code" || true
  fi
  release_lock
}

handle_signal() {
  local signal_code="$1"
  stop_server || true
  if [[ -n "$CURRENT_MODEL_ID" && -d "$STATE_ROOT" ]]; then
    write_model_state "$CURRENT_MODEL_ID" "interrupted" "operator interrupted the benchmark" || true
  fi
  if [[ -d "$STATE_ROOT" ]]; then
    write_state "$RUN_STATE_FILE" "interrupted" "operator interrupted the benchmark" || true
  fi
  exit "$signal_code"
}

marker_path() {
  printf '%s/.lfm-qad-owned\n' "$1"
}

marker_value() {
  local marker="$1"
  local wanted_key="$2"
  local key=""
  local value=""
  [[ -r "$marker" ]] || return 1
  while IFS='=' read -r key value; do
    if [[ "$key" == "$wanted_key" ]]; then
      printf '%s\n' "$value"
      return 0
    fi
  done < "$marker"
  return 1
}

is_owned_model_dir() {
  local model_dir="$1"
  local model_id="$2"
  local repo="$3"
  local file="$4"
  local marker
  marker="$(marker_path "$model_dir")"
  [[ -d "$model_dir" && ! -L "$model_dir" && -f "$marker" ]] || return 1
  [[ "$(marker_value "$marker" owner || true)" == "lfm-qad-benchmark-v1" ]] || return 1
  [[ "$(marker_value "$marker" model_id || true)" == "$model_id" ]] || return 1
  [[ "$(marker_value "$marker" repo || true)" == "$repo" ]] || return 1
  [[ "$(marker_value "$marker" file || true)" == "$file" ]] || return 1
}

cleanup_owned_model() {
  local model_dir="$1"
  local model_id="$2"
  local repo="$3"
  local file="$4"
  local expected_file="$model_dir/$file"
  local marker
  marker="$(marker_path "$model_dir")"
  case "$model_dir" in
    "$SCRATCH_ROOT"/*) ;;
    *) fatal "refusing to clean a model outside the scratch root: $model_dir" ;;
  esac
  is_owned_model_dir "$model_dir" "$model_id" "$repo" "$file" || fatal "ownership marker mismatch; refusing to clean $model_dir"
  [[ ! -L "$expected_file" ]] || fatal "refusing to clean a model directory with a symlinked GGUF: $expected_file"
  log "CLEANUP: removing workflow-owned model directory $model_dir"
  rm -rf "$model_dir"
}

load_manifest() {
  local model_id=""
  local repo=""
  local file=""
  local max_context=""
  local model_dir=""
  local extra=""
  [[ -r "$MANIFEST" ]] || fatal "model manifest is not readable: $MANIFEST"
  while IFS=$'\t' read -r model_id repo file max_context model_dir extra; do
    [[ -z "$model_id" || "$model_id" == \#* ]] && continue
    [[ -z "$extra" ]] || fatal "too many columns in model manifest for $model_id"
    [[ "$model_id" =~ ^[a-z0-9._-]+$ ]] || fatal "invalid model ID: $model_id"
    [[ "$repo" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] || fatal "invalid Hugging Face repository: $repo"
    [[ "$file" =~ ^[A-Za-z0-9_.-]+\.gguf$ ]] || fatal "invalid GGUF filename: $file"
    [[ "$model_dir" =~ ^[A-Za-z0-9._-]+$ ]] || fatal "invalid model directory: $model_dir"
    validate_integer "maximum context for $model_id" "$max_context" 1024
    if (( ${#MODEL_IDS[@]} > 0 )); then
      for existing_id in "${MODEL_IDS[@]}"; do
        [[ "$existing_id" != "$model_id" ]] || fatal "duplicate model ID: $model_id"
      done
    fi
    MODEL_IDS+=("$model_id")
    MODEL_REPOS+=("$repo")
    MODEL_FILES+=("$file")
    MODEL_MAX_CONTEXTS+=("$max_context")
    MODEL_DIRS+=("$model_dir")
  done < "$MANIFEST"
  (( ${#MODEL_IDS[@]} == 4 )) || fatal "expected exactly four LFM QAD models; found ${#MODEL_IDS[@]}"
  if [[ -n "$REQUESTED_MODEL" ]]; then
    local found=0
    for model_id in "${MODEL_IDS[@]}"; do
      [[ "$model_id" != "$REQUESTED_MODEL" ]] || found=1
    done
    (( found == 1 )) || fatal "model ID is not in the manifest: $REQUESTED_MODEL"
  fi
}

validate_questions() {
  [[ -r "$QUESTIONS" ]] || fatal "MT-Bench question file is not readable: $QUESTIONS"
  jq -e -s 'length == 80 and all(.[]; (.turns | length) == 2)' "$QUESTIONS" >/dev/null \
    || fatal "MT-Bench question file must contain 80 two-turn records: $QUESTIONS"
  EXPECTED_QUESTION_IDS_JSON="$(jq -Rsc '[split("\n")[] | select(length > 0) | fromjson.question_id] | sort' "$QUESTIONS")"
}

validate_configuration() {
  validate_integer "LFM_QAD_PORT" "$SERVER_PORT" 1
  (( SERVER_PORT <= 65535 )) || fatal "LFM_QAD_PORT must be <= 65535"
  validate_integer "LFM_QAD_SERVER_TIMEOUT_SECONDS" "$SERVER_TIMEOUT_SECONDS" 1
  validate_integer "LFM_QAD_DOWNLOAD_ATTEMPTS" "$DOWNLOAD_ATTEMPTS" 1
  validate_integer "LFM_QAD_DOWNLOAD_RETRY_SECONDS" "$DOWNLOAD_RETRY_SECONDS" 0
  validate_integer "LFM_QAD_SERVER_GPU_LAYERS" "$SERVER_GPU_LAYERS" 0
  validate_integer "LFM_QAD_ANSWER_MAX_TOKENS" "$ANSWER_MAX_TOKENS" 1
  validate_integer "LFM_QAD_JUDGE_BATCH_RETRIES" "$JUDGE_BATCH_RETRIES" 0
  validate_integer "LFM_QAD_JUDGE_API_RETRIES" "$JUDGE_API_RETRIES" 0
  [[ "$JUDGE_RETRY_BASE_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || fatal "LFM_QAD_JUDGE_RETRY_BASE_SECONDS must be a non-negative number"
  [[ "$JUDGE_POLL_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || fatal "LFM_QAD_JUDGE_POLL_SECONDS must be a non-negative number"
  [[ -z "$JUDGE_MAX_COST_USD" || "$JUDGE_MAX_COST_USD" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    || fatal "LFM_QAD_JUDGE_MAX_COST_USD must be a non-negative number"
  [[ "$SCRATCH_ROOT" != "/Volumes/Touro" && "$SCRATCH_ROOT" != /Volumes/Touro/* ]] \
    || fatal "LFM_QAD_SCRATCH_ROOT must not point to Touro"
  [[ "$SCRATCH_ROOT" != "$BENCHMARK_ROOT" ]] || fatal "scratch root must be separate from the benchmark root"
}

validate_tools() {
  command -v "$HF_BIN" >/dev/null 2>&1 || fatal "Hugging Face CLI was not found: $HF_BIN"
  command -v "$LLAMA_BENCH_BIN" >/dev/null 2>&1 || fatal "llama-bench was not found: $LLAMA_BENCH_BIN"
  command -v "$LLAMA_SERVER_BIN" >/dev/null 2>&1 || fatal "llama-server was not found: $LLAMA_SERVER_BIN"
  command -v "$UV_BIN" >/dev/null 2>&1 || fatal "uv was not found: $UV_BIN"
  command -v curl >/dev/null 2>&1 || fatal "curl is required"
  command -v jq >/dev/null 2>&1 || fatal "jq is required"
  [[ -x "$LLAMA_BENCH_SCRIPT" ]] || fatal "llama benchmark script is not executable: $LLAMA_BENCH_SCRIPT"
  [[ -r "$ANSWER_SCRIPT" && -r "$JUDGE_SCRIPT" ]] || fatal "benchmark Python scripts are missing"
  [[ -r "$REPORT_SCRIPT" ]] || fatal "report script is missing: $REPORT_SCRIPT"
}

validate_openrouter_config() {
  if ! "$UV_BIN" run --project "$ROOT_DIR" python "$JUDGE_SCRIPT" \
    --check-config --dotenv "$ROOT_DIR/.env" >/dev/null; then
    fatal "OPENROUTER_API_KEY is not configured in the environment or repository .env file"
  fi
}

print_plan() {
  log "DRY RUN: no directories, downloads, servers, or inference will start."
  for index in "${!MODEL_IDS[@]}"; do
    if [[ -n "$REQUESTED_MODEL" && "${MODEL_IDS[$index]}" != "$REQUESTED_MODEL" ]]; then
      continue
    fi
    log "PLAN: model=${MODEL_IDS[$index]} repo=${MODEL_REPOS[$index]} file=${MODEL_FILES[$index]} context=${MODEL_MAX_CONTEXTS[$index]} scratch=$SCRATCH_ROOT/${MODEL_DIRS[$index]}"
  done
  log "PLAN: local server=http://$SERVER_HOST:$SERVER_PORT/v1 judge=$JUDGE_MODEL"
  log "PLAN: report=$REPORT_ROOT"
}

prepare_model() {
  local index="$1"
  local model_id="${MODEL_IDS[$index]}"
  local repo="${MODEL_REPOS[$index]}"
  local file="${MODEL_FILES[$index]}"
  local model_dir="$SCRATCH_ROOT/${MODEL_DIRS[$index]}"
  local expected_file="$model_dir/$file"
  local marker
  local attempt=0
  local status=0
  marker="$(marker_path "$model_dir")"

  if [[ -e "$model_dir" && ! -d "$model_dir" ]]; then
    fatal "model scratch path is not a directory: $model_dir"
  fi
  if [[ -d "$model_dir" ]]; then
    is_owned_model_dir "$model_dir" "$model_id" "$repo" "$file" \
      || fatal "existing scratch directory is not owned by this workflow: $model_dir"
  else
    mkdir -p "$model_dir"
    {
      printf 'owner=lfm-qad-benchmark-v1\n'
      printf 'model_id=%s\n' "$model_id"
      printf 'repo=%s\n' "$repo"
      printf 'file=%s\n' "$file"
    } > "$marker"
  fi

  if [[ -s "$expected_file" ]]; then
    log "DOWNLOAD SKIP: $model_id already has a local GGUF"
    return 0
  fi

  while (( attempt < DOWNLOAD_ATTEMPTS )); do
    attempt=$((attempt + 1))
    log "DOWNLOAD: $model_id attempt=$attempt/$DOWNLOAD_ATTEMPTS repo=$repo file=$file"
    set +e
    "$HF_BIN" download "$repo" "$file" --repo-type model --local-dir "$model_dir"
    status=$?
    set -e
    if [[ "$status" -eq 0 && -s "$expected_file" ]]; then
      log "DOWNLOAD COMPLETE: $model_id"
      return 0
    fi
    if [[ "$status" -eq 130 || "$status" -eq 143 ]]; then
      return "$status"
    fi
    log "DOWNLOAD FAILED: $model_id exit=$status"
    if (( attempt < DOWNLOAD_ATTEMPTS )); then
      sleep "$DOWNLOAD_RETRY_SECONDS"
    fi
  done
  return 1
}

write_llama_manifest() {
  local index="$1"
  local path="$MANIFEST_ROOT/${MODEL_IDS[$index]}.tsv"
  mkdir -p "$MANIFEST_ROOT"
  printf '# id\trelative_model_path\tmaximum_context_tokens\n%s\t%s\t%s\n' \
    "${MODEL_IDS[$index]}" "${MODEL_FILES[$index]}" "${MODEL_MAX_CONTEXTS[$index]}" > "$path"
  printf '%s\n' "$path"
}

performance_complete() {
  local result_dir="$1"
  [[ -s "$result_dir/performance.done" && -s "$result_dir/performance/summary.json" ]]
}

run_performance() {
  local index="$1"
  local model_id="${MODEL_IDS[$index]}"
  local model_dir="$SCRATCH_ROOT/${MODEL_DIRS[$index]}"
  local result_dir="$RESULTS_ROOT/$model_id"
  local manifest_path
  mkdir -p "$result_dir"
  if performance_complete "$result_dir"; then
    log "PERFORMANCE SKIP: $model_id already completed"
    return 0
  fi
  manifest_path="$(write_llama_manifest "$index")"
  log "PERFORMANCE START: $model_id"
  if env \
    LLAMA_BENCHMARK_SOURCE_MODEL_ROOT="$model_dir" \
    LLAMA_BENCHMARK_LOCAL_MODEL_ROOT="$model_dir" \
    LLAMA_BENCHMARK_MANIFEST="$manifest_path" \
    LLAMA_BENCHMARK_ROOT="$result_dir/performance" \
    LLAMA_BENCHMARK_BIN="$LLAMA_BENCH_BIN" \
    uv run --project "$ROOT_DIR" python "$LLAMA_BENCH_SCRIPT" --model "$model_id"; then
    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$result_dir/performance.done"
    log "PERFORMANCE COMPLETE: $model_id"
    return 0
  fi
  log "PERFORMANCE FAILED: $model_id"
  return 1
}

jsonl_complete() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  jq -e -s --argjson expected "$EXPECTED_QUESTION_IDS_JSON" \
    'length == 80 and ((map(.question_id) | sort) == $expected)' "$path" >/dev/null
}

server_url() {
  printf 'http://%s:%s/v1\n' "$SERVER_HOST" "$SERVER_PORT"
}

start_server() {
  local index="$1"
  local model_id="${MODEL_IDS[$index]}"
  local model_path="$SCRATCH_ROOT/${MODEL_DIRS[$index]}/${MODEL_FILES[$index]}"
  local result_dir="$RESULTS_ROOT/$model_id"
  local url
  local attempt=0
  url="$(server_url)"
  if curl -fsS "$url/health" >/dev/null 2>&1; then
    fatal "port $SERVER_PORT is already serving an endpoint; refusing to use an unknown server"
  fi
  log "SERVER START: $model_id port=$SERVER_PORT context=${MODEL_MAX_CONTEXTS[$index]}"
  "$LLAMA_SERVER_BIN" \
    --model "$model_path" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --ctx-size "${MODEL_MAX_CONTEXTS[$index]}" \
    --n-gpu-layers "$SERVER_GPU_LAYERS" \
    --alias "$model_id" \
    > "$result_dir/llama-server.log" 2>&1 &
  SERVER_PID="$!"
  while (( attempt < SERVER_TIMEOUT_SECONDS )); do
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      wait "$SERVER_PID" >/dev/null 2>&1 || true
      SERVER_PID=""
      log "SERVER FAILED: $model_id exited before becoming healthy"
      return 1
    fi
    if curl -fsS "$url/health" >/dev/null 2>&1; then
      log "SERVER READY: $model_id url=$url"
      return 0
    fi
    attempt=$((attempt + 1))
    sleep 1
  done
  log "SERVER TIMEOUT: $model_id after ${SERVER_TIMEOUT_SECONDS}s"
  stop_server
  return 1
}

run_answers() {
  local index="$1"
  local model_id="${MODEL_IDS[$index]}"
  local result_dir="$RESULTS_ROOT/$model_id"
  local answer_path="$result_dir/answers.jsonl"
  if jsonl_complete "$answer_path"; then
    log "ANSWERS SKIP: $model_id already has 80 answers"
    return 0
  fi
  start_server "$index" || return 1
  log "ANSWERS START: $model_id"
  if "$UV_BIN" run --project "$ROOT_DIR" python "$ANSWER_SCRIPT" \
    --model-id "$model_id" \
    --questions "$QUESTIONS" \
    --output "$answer_path" \
    --server-url "$(server_url)" \
    --max-tokens "$ANSWER_MAX_TOKENS"; then
    stop_server
  else
    local status="$?"
    stop_server
    log "ANSWERS FAILED: $model_id exit=$status"
    return "$status"
  fi
  jsonl_complete "$answer_path" || { log "ANSWERS INCOMPLETE: $model_id"; return 1; }
  log "ANSWERS COMPLETE: $model_id"
}

run_judge() {
  local index="$1"
  local model_id="${MODEL_IDS[$index]}"
  local result_dir="$RESULTS_ROOT/$model_id"
  local answer_path="$result_dir/answers.jsonl"
  local judgment_path="$result_dir/judgments.jsonl"
  local judge_args=()
  if [[ -s "$judgment_path" ]] && jq -e -s --arg judge "$JUDGE_MODEL" \
    'any(.[]; .judge_model != $judge)' "$judgment_path" >/dev/null; then
    log "JUDGMENT RESET: $model_id moving judgments from a different judge to preserve batch consistency"
    mv "$judgment_path" "$result_dir/judgments.sync.jsonl"
    if [[ -f "$result_dir/judgments.errors.jsonl" ]]; then
      mv "$result_dir/judgments.errors.jsonl" "$result_dir/judgments.sync.errors.jsonl"
    fi
  fi
  if jsonl_complete "$judgment_path"; then
    log "JUDGMENT SKIP: $model_id already has 80 judgments"
    return 0
  fi
  judge_args+=(--answers "$answer_path" --questions "$QUESTIONS" --output "$judgment_path")
  judge_args+=(--batch-model "$JUDGE_MODEL")
  judge_args+=(--max-batch-retries "$JUDGE_BATCH_RETRIES" --max-api-retries "$JUDGE_API_RETRIES")
  judge_args+=(--retry-base-seconds "$JUDGE_RETRY_BASE_SECONDS" --poll-interval-seconds "$JUDGE_POLL_SECONDS")
  judge_args+=(--dotenv "$ROOT_DIR/.env")
  if [[ -n "$JUDGE_MAX_COST_USD" ]]; then
    judge_args+=(--max-cost-usd "$JUDGE_MAX_COST_USD")
  fi
  log "JUDGMENT START: $model_id judge=$JUDGE_MODEL"
  if "$UV_BIN" run --project "$ROOT_DIR" python "$JUDGE_SCRIPT" "${judge_args[@]}"; then
    :
  else
    local status="$?"
    log "JUDGMENT FAILED: $model_id exit=$status"
    return "$status"
  fi
  jsonl_complete "$judgment_path" || { log "JUDGMENT INCOMPLETE: $model_id"; return 1; }
  log "JUDGMENT COMPLETE: $model_id"
}

model_complete() {
  local model_id="$1"
  local result_dir="$RESULTS_ROOT/$model_id"
  performance_complete "$result_dir" \
    && jsonl_complete "$result_dir/answers.jsonl" \
    && jsonl_complete "$result_dir/judgments.jsonl"
}

run_report() {
  mkdir -p "$REPORT_ROOT"
  log "REPORT START: combining local performance, published quality, and MT-Bench judgments"
  if "$UV_BIN" run --project "$ROOT_DIR" python "$REPORT_SCRIPT" \
    --manifest "$MANIFEST" \
    --published "$ROOT_DIR/benchmarks/reference/lfm-qad-published.json" \
    --results-root "$RESULTS_ROOT" \
    --output-json "$REPORT_ROOT/comparison.json" \
    --output-markdown "$REPORT_ROOT/comparison.md"; then
    log "REPORT COMPLETE: $REPORT_ROOT/comparison.md"
    return 0
  fi
  log "REPORT FAILED"
  return 1
}

process_model() {
  local index="$1"
  local model_id="${MODEL_IDS[$index]}"
  local repo="${MODEL_REPOS[$index]}"
  local file="${MODEL_FILES[$index]}"
  local model_dir="$SCRATCH_ROOT/${MODEL_DIRS[$index]}"
  CURRENT_MODEL_ID="$model_id"
  mkdir -p "$RESULTS_ROOT/$model_id"
  if model_complete "$model_id"; then
    log "MODEL SKIP: $model_id is complete"
    if [[ -d "$model_dir" ]]; then
      is_owned_model_dir "$model_dir" "$model_id" "$repo" "$file" \
        || fatal "completed model has an unsafe scratch directory: $model_dir"
      cleanup_owned_model "$model_dir" "$model_id" "$repo" "$file"
    fi
    write_model_state "$model_id" "complete" "existing results are complete"
    CURRENT_MODEL_ID=""
    return 0
  fi

  write_model_state "$model_id" "starting" "preparing one workflow-owned model"
  if ! prepare_model "$index"; then
    write_model_state "$model_id" "failed" "model download failed"
    cleanup_owned_model "$model_dir" "$model_id" "$repo" "$file"
    CURRENT_MODEL_ID=""
    return 1
  fi
  if ! run_performance "$index" || ! run_answers "$index" || ! run_judge "$index"; then
    stop_server
    write_model_state "$model_id" "failed" "one benchmark stage failed; artifacts were preserved"
    cleanup_owned_model "$model_dir" "$model_id" "$repo" "$file"
    CURRENT_MODEL_ID=""
    return 1
  fi
  stop_server
  write_state "$RESULTS_ROOT/$model_id/complete" "complete" "all model stages completed"
  write_model_state "$model_id" "complete" "performance, answers, and judgments completed"
  cleanup_owned_model "$model_dir" "$model_id" "$repo" "$file"
  CURRENT_MODEL_ID=""
  log "MODEL COMPLETE: $model_id; model directory removed"
}

parse_arguments "$@"
validate_configuration
load_manifest
validate_questions

if [[ "$DRY_RUN" -eq 1 ]]; then
  print_plan
  exit 0
fi

validate_tools
validate_openrouter_config
[[ ! -L "$SCRATCH_ROOT" ]] || fatal "scratch root must not be a symlink: $SCRATCH_ROOT"
mkdir -p "$STATE_ROOT/models" "$RESULTS_ROOT" "$MANIFEST_ROOT" "$LOG_DIR" "$SCRATCH_ROOT"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1
if [[ "$SCRATCH_ROOT" == "$DEFAULT_SCRATCH_ROOT" ]]; then
  log "Scratch root is internal application storage: $SCRATCH_ROOT"
fi

trap 'on_exit' EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM
acquire_lock
write_state "$RUN_STATE_FILE" "running" "benchmark started"
RUN_STATUS="running"
log "Starting sequential LFM QAD benchmark run=$RUN_ID"
log "Persistent root: $BENCHMARK_ROOT"
log "Owned scratch root: $SCRATCH_ROOT"
log "Touro and the existing local model cache will not be used"

overall_status=0
for index in "${!MODEL_IDS[@]}"; do
  if [[ -n "$REQUESTED_MODEL" && "${MODEL_IDS[$index]}" != "$REQUESTED_MODEL" ]]; then
    continue
  fi
  if process_model "$index"; then
    :
  else
    overall_status=1
  fi
done

if [[ "$overall_status" -eq 0 ]]; then
  if run_report; then
    write_state "$RUN_STATE_FILE" "complete" "all selected models and the final report completed"
    RUN_STATUS="complete"
    log "BENCHMARK COMPLETE: results are under $RESULTS_ROOT; report is under $REPORT_ROOT"
  else
    write_state "$RUN_STATE_FILE" "partial" "models completed but the final report failed"
    RUN_STATUS="partial"
    exit 1
  fi
else
  write_state "$RUN_STATE_FILE" "partial" "one or more selected models failed"
  RUN_STATUS="partial"
  log "BENCHMARK PARTIAL: inspect per-model state and results under $RESULTS_ROOT"
  exit 1
fi
