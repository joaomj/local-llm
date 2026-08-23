#!/usr/bin/env bash

# Download the requested GGUF files to a dedicated directory on Touro.
#
# The Hugging Face CLI keeps resumable download metadata under each local
# directory. This wrapper adds persistent per-file state, logs, retries, and a
# lock so an interrupted run can be started again safely.

if [ -z "${BASH_VERSION:-}" ]; then
  printf '%s\n' "ERROR: run this script with Bash: ./scripts/download_models.sh" >&2
  exit 2
fi

set -Eeuo pipefail
IFS=$'\n\t'
umask 022

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_VERSION="1"
readonly DEFAULT_TOURO_MOUNT="/Volumes/Touro"
readonly DEFAULT_MODEL_DIR_NAME="local-llm-models"
readonly DEFAULT_MAX_ATTEMPTS="5"
readonly DEFAULT_RETRY_DELAY_SECONDS="15"
readonly DEFAULT_TOURO_OPERATION_TIMEOUT_SECONDS="30"

TOURO_MOUNT="${TOURO_MOUNT:-$DEFAULT_TOURO_MOUNT}"
TOURO_MOUNT="${TOURO_MOUNT%/}"
MODEL_ROOT="${TOURO_MODEL_DIR:-$TOURO_MOUNT/$DEFAULT_MODEL_DIR_NAME}"
HF_BIN="${HF_BIN:-hf}"
MAX_ATTEMPTS="${HF_DOWNLOAD_ATTEMPTS:-$DEFAULT_MAX_ATTEMPTS}"
RETRY_DELAY_SECONDS="${HF_RETRY_DELAY_SECONDS:-$DEFAULT_RETRY_DELAY_SECONDS}"
RECHECK_COMPLETE="${RECHECK_COMPLETE:-0}"
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
TOURO_OPERATION_TIMEOUT_SECONDS="${TOURO_OPERATION_TIMEOUT_SECONDS:-$DEFAULT_TOURO_OPERATION_TIMEOUT_SECONDS}"
export HF_HUB_DOWNLOAD_TIMEOUT HF_HUB_ETAG_TIMEOUT

# Keep these as separate files. Downloading the complete Qwen repository would
# fetch hundreds of gigabytes of alternative quantizations and auxiliary files.
MODEL_IDS=(
  "lfm2.5-230m-qad-q4_0"
  "lfm2.5-1.2b-instruct-qad-q4_0"
  "qwen3.8-27b-ud-iq1_m"
  "qwen3.8-27b-ud-iq2_s"
  "qwen3.8-27b-ud-q2_k_xl"
)
MODEL_REPOS=(
  "LiquidAI/LFM2.5-230M-GGUF"
  "LiquidAI/LFM2.5-1.2B-Instruct-GGUF"
  "unsloth/Qwen3.8-27B-GGUF"
  "unsloth/Qwen3.8-27B-GGUF"
  "unsloth/Qwen3.8-27B-GGUF"
)
MODEL_FILES=(
  "LFM2.5-230M-QAD-Q4_0.gguf"
  "LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf"
  "Qwen3.8-27B-UD-IQ1_M.gguf"
  "Qwen3.8-27B-UD-IQ2_S.gguf"
  "Qwen3.8-27B-UD-Q2_K_XL.gguf"
)
MODEL_DIRS=(
  "LFM2.5-230M-QAD-Q4_0"
  "LFM2.5-1.2B-Instruct-QAD-Q4_0"
  "Qwen3.8-27B-UD-IQ1_M"
  "Qwen3.8-27B-UD-IQ2_S"
  "Qwen3.8-27B-UD-Q2_K_XL"
)

STATE_DIR="$MODEL_ROOT/.download-state"
LOCK_DIR="$STATE_DIR/.lock"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
LOG_FILE=""
RUN_STATE_FILE="$STATE_DIR/run.state"
RUN_STATUS="starting"
LOCK_ACQUIRED=0
CURRENT_SLUG=""
CURRENT_STATUS=""
CURRENT_ATTEMPT=0
CURRENT_REPO=""
CURRENT_FILE=""
CURRENT_DESTINATION=""
COMPLETED_COUNT=0
SKIPPED_COUNT=0
FAILED_COUNT=0

usage() {
  cat <<EOF
Usage: $SCRIPT_NAME [--recheck-complete]

Downloads these five files to:
  $MODEL_ROOT

The script is safe to rerun. Partial downloads remain in place and hf resumes
them. Completed files are skipped when their saved size matches their state.

Options:
  --recheck-complete  Ask hf to inspect completed files again instead of using
                      the saved completion state as a local fast path.

Environment overrides:
  TOURO_MOUNT             Mounted drive path (default: /Volumes/Touro)
  TOURO_MODEL_DIR         Destination under TOURO_MOUNT
  HF_BIN                  Hugging Face CLI command (default: hf)
  HF_DOWNLOAD_ATTEMPTS    Attempts per file (default: 5)
  HF_RETRY_DELAY_SECONDS  Initial retry delay (default: 15 seconds)
  RECHECK_COMPLETE        Set to 1 to recheck completed files
  HF_HUB_DOWNLOAD_TIMEOUT Hugging Face download timeout (default: 60)
  HF_HUB_ETAG_TIMEOUT     Hugging Face metadata timeout (default: 30)
  TOURO_OPERATION_TIMEOUT_SECONDS
                          Timeout for initial Touro filesystem operations
                          (default: 30)

Logs and state are stored on Touro under:
  $STATE_DIR
EOF
}

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

fatal() {
  log "ERROR: $*"
  exit 2
}

run_with_timeout() {
  local seconds="$1"
  shift
  local child_pid=""
  local elapsed=0
  local status=0

  "$@" &
  child_pid="$!"

  while kill -0 "$child_pid" >/dev/null 2>&1; do
    if (( elapsed >= seconds )); then
      kill -TERM "$child_pid" >/dev/null 2>&1 || true
      sleep 1
      kill -KILL "$child_pid" >/dev/null 2>&1 || true
      wait "$child_pid" >/dev/null 2>&1 || true
      return 124
    fi
    sleep 1
    ((elapsed += 1))
  done

  wait "$child_pid" >/dev/null 2>&1 || status="$?"
  return "$status"
}

write_probe_file() {
  local path="$1"
  local value="$2"
  printf '%s\n' "$value" > "$path"
}

sanitize_state_value() {
  local value="$1"
  value="${value//$'\t'/ }"
  value="${value//$'\n'/ }"
  printf '%s' "$value"
}

file_size() {
  local path="$1"
  if stat -f %z "$path" >/dev/null 2>&1; then
    stat -f %z "$path"
  else
    stat -c %s "$path"
  fi
}

state_value() {
  local state_file="$1"
  local wanted_key="$2"
  local key=""
  local value=""

  [[ -r "$state_file" ]] || return 1
  while IFS='=' read -r key value; do
    if [[ "$key" == "$wanted_key" ]]; then
      printf '%s\n' "$value"
      return 0
    fi
  done < "$state_file"
  return 1
}

clear_current_model() {
  CURRENT_SLUG=""
  CURRENT_STATUS=""
  CURRENT_ATTEMPT=0
  CURRENT_REPO=""
  CURRENT_FILE=""
  CURRENT_DESTINATION=""
}

write_model_state() {
  local slug="$1"
  local repo="$2"
  local file="$3"
  local destination="$4"
  local status="$5"
  local attempt="$6"
  local exit_code="$7"
  local size_bytes="$8"
  local message="$9"
  local state_file="$STATE_DIR/$slug.state"
  local temporary_file="$STATE_DIR/.$slug.state.tmp.$$"

  message="$(sanitize_state_value "$message")"
  {
    printf 'version=%s\n' "$SCRIPT_VERSION"
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'model_id=%s\n' "$slug"
    printf 'repo=%s\n' "$repo"
    printf 'file=%s\n' "$file"
    printf 'destination=%s\n' "$destination"
    printf 'status=%s\n' "$status"
    printf 'attempt=%s\n' "$attempt"
    printf 'exit_code=%s\n' "$exit_code"
    printf 'size_bytes=%s\n' "$size_bytes"
    printf 'updated_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'message=%s\n' "$message"
  } > "$temporary_file"
  mv -f "$temporary_file" "$state_file"

  if [[ "$slug" == "$CURRENT_SLUG" ]]; then
    CURRENT_STATUS="$status"
    CURRENT_ATTEMPT="$attempt"
  fi
}

write_run_state() {
  local status="$1"
  local message="$2"
  local temporary_file="$STATE_DIR/.run.state.tmp.$$"

  message="$(sanitize_state_value "$message")"
  {
    printf 'version=%s\n' "$SCRIPT_VERSION"
    printf 'run_id=%s\n' "$RUN_ID"
    printf 'status=%s\n' "$status"
    printf 'model_root=%s\n' "$MODEL_ROOT"
    printf 'log_file=%s\n' "$LOG_FILE"
    printf 'updated_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'message=%s\n' "$message"
  } > "$temporary_file"
  mv -f "$temporary_file" "$RUN_STATE_FILE"
  RUN_STATUS="$status"
}

release_lock() {
  if [[ "$LOCK_ACQUIRED" -ne 1 ]]; then
    return 0
  fi

  rm -f "$LOCK_DIR/pid" "$LOCK_DIR/started_at" "$LOCK_DIR/host"
  if ! rmdir "$LOCK_DIR" 2>/dev/null; then
    log "WARNING: Could not remove lock directory: $LOCK_DIR"
  fi
  LOCK_ACQUIRED=0
}

on_exit() {
  local exit_code=$?

  if [[ -n "$CURRENT_SLUG" && "$CURRENT_STATUS" != "interrupted" && "$CURRENT_STATUS" != "complete" ]]; then
    write_model_state \
      "$CURRENT_SLUG" \
      "$CURRENT_REPO" \
      "$CURRENT_FILE" \
      "$CURRENT_DESTINATION" \
      "failed" \
      "$CURRENT_ATTEMPT" \
      "$exit_code" \
      "0" \
      "process exited before the current model completed"
  fi

  if [[ "$RUN_STATUS" == "running" || "$RUN_STATUS" == "starting" ]]; then
    if [[ "$exit_code" -eq 0 ]]; then
      write_run_state "complete" "all model files completed"
    else
      write_run_state "failed" "process exited with status $exit_code"
    fi
  fi

  release_lock
}

handle_signal() {
  local signal_name="$1"
  local signal_code="$2"

  log "Received $signal_name. Partial files and state are preserved for resume."
  if [[ -n "$CURRENT_SLUG" ]]; then
    write_model_state \
      "$CURRENT_SLUG" \
      "$CURRENT_REPO" \
      "$CURRENT_FILE" \
      "$CURRENT_DESTINATION" \
      "interrupted" \
      "$CURRENT_ATTEMPT" \
      "$signal_code" \
      "0" \
      "interrupted by $signal_name"
  fi
  write_run_state "interrupted" "interrupted by $signal_name"
  exit "$signal_code"
}

acquire_lock() {
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    LOCK_ACQUIRED=1
    printf '%s\n' "$$" > "$LOCK_DIR/pid"
    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK_DIR/started_at"
    hostname > "$LOCK_DIR/host"
    return 0
  fi

  local owner_pid=""
  local owner_command=""
  if [[ -r "$LOCK_DIR/pid" ]]; then
    read -r owner_pid < "$LOCK_DIR/pid" || owner_pid=""
  fi

  if [[ "$owner_pid" =~ ^[0-9]+$ ]] && kill -0 "$owner_pid" 2>/dev/null; then
    owner_command="$(ps -p "$owner_pid" -o command= 2>/dev/null || true)"
    if [[ "$owner_command" == *"$SCRIPT_NAME"* ]]; then
      fatal "another downloader is running (pid $owner_pid). Wait for it to finish."
    fi
    fatal "lock exists at $LOCK_DIR and its PID is in use by another process; inspect it before removing the lock."
  fi

  if [[ -z "$owner_pid" ]]; then
    fatal "lock exists at $LOCK_DIR but has no readable PID; inspect it before removing the lock."
  fi

  log "Removing stale downloader lock for exited pid $owner_pid."
  rm -f "$LOCK_DIR/pid" "$LOCK_DIR/started_at" "$LOCK_DIR/host"
  if ! rmdir "$LOCK_DIR" 2>/dev/null; then
    fatal "could not remove stale lock at $LOCK_DIR; another process may have acquired it"
  fi
  acquire_lock
}

retry_delay() {
  local attempt="$1"
  local delay="$RETRY_DELAY_SECONDS"
  local multiplier=1

  if [[ "$attempt" -gt 1 ]]; then
    multiplier=2
  fi
  if [[ "$attempt" -gt 2 ]]; then
    multiplier=4
  fi
  if [[ "$attempt" -gt 3 ]]; then
    multiplier=8
  fi
  if [[ "$attempt" -gt 4 ]]; then
    multiplier=16
  fi

  delay=$((delay * multiplier))
  if [[ "$delay" -gt 300 ]]; then
    delay=300
  fi
  printf '%s\n' "$delay"
}

download_model() {
  local index="$1"
  local slug="${MODEL_IDS[$index]}"
  local repo="${MODEL_REPOS[$index]}"
  local file="${MODEL_FILES[$index]}"
  local directory="$MODEL_ROOT/${MODEL_DIRS[$index]}"
  local expected_file="$directory/$file"
  local model_state_file="$STATE_DIR/$slug.state"
  local prior_status=""
  local prior_size=""
  local current_size="0"
  local attempt=0
  local status=0
  local delay=0

  CURRENT_SLUG="$slug"
  CURRENT_STATUS="starting"
  CURRENT_ATTEMPT=0
  CURRENT_REPO="$repo"
  CURRENT_FILE="$file"
  CURRENT_DESTINATION="$expected_file"

  if [[ "$RECHECK_COMPLETE" != "1" ]]; then
    prior_status="$(state_value "$model_state_file" status 2>/dev/null || true)"
    prior_size="$(state_value "$model_state_file" size_bytes 2>/dev/null || true)"
    if [[ "$prior_status" == "complete" && -s "$expected_file" && "$prior_size" =~ ^[0-9]+$ ]]; then
      current_size="$(file_size "$expected_file")"
      if [[ "$current_size" == "$prior_size" ]]; then
        log "SKIP: $file is already complete ($current_size bytes)."
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        clear_current_model
        return 0
      fi
      log "The saved size for $file does not match the local file; hf will recheck it."
    fi
  fi

  if ! mkdir -p "$directory"; then
    write_model_state "$slug" "$repo" "$file" "$expected_file" "failed" "0" "1" "0" "could not create destination directory"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    clear_current_model
    return 1
  fi

  log "START: $repo/$file"
  log "DESTINATION: $expected_file"
  write_model_state "$slug" "$repo" "$file" "$expected_file" "running" "0" "0" "0" "starting download"

  while [[ "$attempt" -lt "$MAX_ATTEMPTS" ]]; do
    attempt=$((attempt + 1))
    CURRENT_ATTEMPT="$attempt"
    write_model_state "$slug" "$repo" "$file" "$expected_file" "downloading" "$attempt" "0" "0" "hf download attempt $attempt of $MAX_ATTEMPTS"
    log "DOWNLOAD: attempt $attempt/$MAX_ATTEMPTS"
    log "COMMAND: $HF_BIN download $repo $file --local-dir $directory"

    set +e
    "$HF_BIN" download "$repo" "$file" --repo-type model --local-dir "$directory"
    status=$?
    set -e

    if [[ "$status" -eq 0 && -s "$expected_file" ]]; then
      current_size="$(file_size "$expected_file")"
      write_model_state "$slug" "$repo" "$file" "$expected_file" "complete" "$attempt" "0" "$current_size" "download completed"
      log "COMPLETE: $file ($current_size bytes)"
      COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
      clear_current_model
      return 0
    fi

    if [[ "$status" -eq 130 || "$status" -eq 143 ]]; then
      write_model_state "$slug" "$repo" "$file" "$expected_file" "interrupted" "$attempt" "$status" "0" "hf was interrupted"
      log "INTERRUPTED: $file"
      write_run_state "interrupted" "download interrupted while processing $file"
      exit "$status"
    fi

    if [[ "$status" -eq 0 ]]; then
      status=66
      log "ERROR: hf reported success but the expected file is missing or empty: $expected_file"
    else
      log "ERROR: hf failed for $file with exit status $status"
    fi

    if [[ "$attempt" -lt "$MAX_ATTEMPTS" ]]; then
      delay="$(retry_delay "$attempt")"
      write_model_state "$slug" "$repo" "$file" "$expected_file" "retrying" "$attempt" "$status" "0" "retrying after exit status $status"
      log "RETRY: waiting ${delay}s before attempt $((attempt + 1))"
      sleep "$delay"
    fi
  done

  write_model_state "$slug" "$repo" "$file" "$expected_file" "failed" "$attempt" "$status" "0" "download failed after $MAX_ATTEMPTS attempts"
  log "FAILED: $file after $MAX_ATTEMPTS attempts"
  FAILED_COUNT=$((FAILED_COUNT + 1))
  clear_current_model
  return 1
}

parse_arguments() {
  while [[ "$#" -gt 0 ]]; do
    case "$1" in
      --recheck-complete)
        RECHECK_COMPLETE=1
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

parse_arguments "$@"

log "Starting downloader $SCRIPT_VERSION. This run has not started model downloads."
log "Checking Touro mount: $TOURO_MOUNT"

if [[ ! "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  fatal "HF_DOWNLOAD_ATTEMPTS must be a positive integer"
fi
if [[ ! "$RETRY_DELAY_SECONDS" =~ ^[0-9]+$ ]]; then
  fatal "HF_RETRY_DELAY_SECONDS must be a non-negative integer"
fi
if [[ ! "$TOURO_OPERATION_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  fatal "TOURO_OPERATION_TIMEOUT_SECONDS must be a positive integer"
fi
if [[ "$RECHECK_COMPLETE" != "0" && "$RECHECK_COMPLETE" != "1" ]]; then
  fatal "RECHECK_COMPLETE must be 0 or 1"
fi

if [[ ! -d "$TOURO_MOUNT" ]]; then
  fatal "Touro is not mounted at $TOURO_MOUNT"
fi
log "Touro mount path is present. Verifying the volume."
if [[ "$MODEL_ROOT" != "$TOURO_MOUNT"/* ]]; then
  fatal "TOURO_MODEL_DIR must be inside the Touro mount: $TOURO_MOUNT"
fi
if command -v diskutil >/dev/null 2>&1 && ! diskutil info "$TOURO_MOUNT" >/dev/null 2>&1; then
  fatal "diskutil could not verify the mounted volume at $TOURO_MOUNT"
fi
log "Checking Hugging Face CLI: $HF_BIN"
if ! command -v "$HF_BIN" >/dev/null 2>&1; then
  fatal "Hugging Face CLI '$HF_BIN' was not found. Install it first, then rerun this script."
fi
if ! "$HF_BIN" download --help >/dev/null 2>&1; then
  fatal "Hugging Face CLI '$HF_BIN' does not support the download command"
fi

log "Creating model and persistent state directories on Touro."
if run_with_timeout "$TOURO_OPERATION_TIMEOUT_SECONDS" mkdir -p "$MODEL_ROOT" "$STATE_DIR"; then
  :
else
  status="$?"
  if [[ "$status" -eq 124 ]]; then
    fatal "Touro filesystem did not create the directories within ${TOURO_OPERATION_TIMEOUT_SECONDS}s; eject and reconnect Touro, then rerun"
  fi
  fatal "could not create the model or state directories on Touro"
fi

write_probe="$STATE_DIR/.write-test.$$"
log "Checking Touro write access."
if run_with_timeout "$TOURO_OPERATION_TIMEOUT_SECONDS" write_probe_file "$write_probe" "$RUN_ID"; then
  :
else
  status="$?"
  if [[ "$status" -eq 124 ]]; then
    fatal "Touro filesystem did not complete the write test within ${TOURO_OPERATION_TIMEOUT_SECONDS}s; eject and reconnect Touro, then rerun"
  fi
  fatal "Touro is not writable at $STATE_DIR"
fi
rm -f "$write_probe"

LOG_FILE="$STATE_DIR/download-$RUN_ID.log"
log "Creating persistent log: $LOG_FILE"
if run_with_timeout "$TOURO_OPERATION_TIMEOUT_SECONDS" sh -c ': > "$1"' sh "$LOG_FILE"; then
  :
else
  status="$?"
  if [[ "$status" -eq 124 ]]; then
    fatal "Touro filesystem did not create the log within ${TOURO_OPERATION_TIMEOUT_SECONDS}s; eject and reconnect Touro, then rerun"
  fi
  fatal "could not create the log file at $LOG_FILE"
fi

# Tee keeps the live Hugging Face progress output visible and persists it.
exec > >(tee -a "$LOG_FILE") 2>&1

trap 'on_exit' EXIT
trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM

acquire_lock
write_run_state "running" "starting model downloads"

log "Run id: $RUN_ID"
log "Touro mount: $TOURO_MOUNT"
log "Model root: $MODEL_ROOT"
log "Log file: $LOG_FILE"
log "The existing Touro contents will not be modified outside this model root."
log "Qwen IQ1_M, IQ2_S, and Q2_K_XL are disk downloads; their runtime fit on 16GB RAM is a separate constraint."

overall_status=0
for index in "${!MODEL_IDS[@]}"; do
  if ! download_model "$index"; then
    overall_status=1
  fi
done

log "SUMMARY: completed=$COMPLETED_COUNT skipped=$SKIPPED_COUNT failed=$FAILED_COUNT"
if [[ "$overall_status" -ne 0 ]]; then
  write_run_state "failed" "one or more model downloads failed"
  exit 1
fi

write_run_state "complete" "all requested model files are present"
log "All requested model files are present. Rerun this script to resume or verify state."
