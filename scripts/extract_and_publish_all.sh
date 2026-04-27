#!/usr/bin/env bash
set -uo pipefail

VINDEX_DIR="$HOME/.cache/larql/local"
LOG_DIR="$(dirname "$0")/extract_logs"
mkdir -p "$LOG_DIR"

LARQL="larql"
FAILED_PUBLISHES=()

run_model() {
  local model_id="$1"
  local name="$2"
  local repo="$3"
  local out="$VINDEX_DIR/${name}.vindex"
  local log="$LOG_DIR/${name}.log"

  echo "=== [$(date '+%H:%M:%S')] START extract: $name ===" | tee -a "$log"

  if [ -d "$out" ]; then
    echo "  Vindex already exists at $out — resuming" | tee -a "$log"
  fi

  if ! $LARQL extract \
    --quant q4k \
    --resume \
    --output "$out" \
    "$model_id" \
    2>&1 | tee -a "$log"; then
    echo "=== [$(date '+%H:%M:%S')] ERROR extract: $name ===" | tee -a "$log"
    return
  fi

  echo "=== [$(date '+%H:%M:%S')] DONE extract: $name ===" | tee -a "$log"

  publish_model "$name" "$repo" "$out" "$log"
}

publish_model() {
  local name="$1"
  local repo="$2"
  local out="$3"
  local log="$4"

  echo "=== [$(date '+%H:%M:%S')] START publish: $repo ===" | tee -a "$log"

  if $LARQL publish \
    --repo "$repo" \
    "$out" \
    2>&1 | tee -a "$log"; then
    echo "=== [$(date '+%H:%M:%S')] DONE publish: $repo ===" | tee -a "$log"
  else
    echo "=== [$(date '+%H:%M:%S')] FAILED publish: $repo — will retry ===" | tee -a "$log"
    FAILED_PUBLISHES+=("$name|$repo|$out|$log")
  fi
  echo "" | tee -a "$log"
}

# Mistral 7B
run_model "mistralai/Mistral-7B-v0.1"           "mistral-7b-v0.1-q4k"           "chrishayuk/mistral-7b-v0.1-vindex"
run_model "mistralai/Mistral-7B-Instruct-v0.3"  "mistral-7b-instruct-v0.3-q4k"  "chrishayuk/mistral-7b-instruct-v0.3-vindex"

# Llama 2 7B
run_model "meta-llama/Llama-2-7b-hf"            "llama2-7b-q4k"                 "chrishayuk/llama-2-7b-vindex"
run_model "meta-llama/Llama-2-7b-chat-hf"       "llama2-7b-chat-q4k"            "chrishayuk/llama-2-7b-chat-vindex"

# Llama 3 8B
run_model "meta-llama/Meta-Llama-3-8B"          "llama3-8b-q4k"                 "chrishayuk/llama-3-8b-vindex"
run_model "meta-llama/Meta-Llama-3-8B-Instruct" "llama3-8b-instruct-q4k"        "chrishayuk/llama-3-8b-instruct-vindex"

# Llama 3.2
run_model "meta-llama/Llama-3.2-1B"             "llama3.2-1b-q4k"               "chrishayuk/llama-3.2-1b-vindex"
run_model "meta-llama/Llama-3.2-1B-Instruct"    "llama3.2-1b-instruct-q4k"      "chrishayuk/llama-3.2-1b-instruct-vindex"
run_model "meta-llama/Llama-3.2-3B"             "llama3.2-3b-q4k"               "chrishayuk/llama-3.2-3b-vindex"
run_model "meta-llama/Llama-3.2-3B-Instruct"    "llama3.2-3b-instruct-q4k"      "chrishayuk/llama-3.2-3b-instruct-vindex"

# Retry any failed publishes
if [ ${#FAILED_PUBLISHES[@]} -gt 0 ]; then
  echo "=== Retrying ${#FAILED_PUBLISHES[@]} failed publish(es) ==="
  for entry in "${FAILED_PUBLISHES[@]}"; do
    IFS='|' read -r name repo out log <<< "$entry"
    publish_model "$name" "$repo" "$out" "$log"
  done
fi

echo "=== ALL DONE $(date) ==="
