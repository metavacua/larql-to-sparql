#!/usr/bin/env bash
# Thin entrypoint → run_matrix.py (the real runner). Kept so the workflow and
# README keep a stable `run_matrix.sh <level> <vindex> <corpus> <out>` API while
# the capture logic lives in Python, where UTF-8 handling, codepoint-safe
# truncation, full-stream files, and shell-free subprocess orchestration are
# correct by construction. All configuration is via env (see run_matrix.py):
#   LARQL_BIN  MODEL_ID  TMPROOT  WRAP  CELL_TIMEOUT
set -uo pipefail
exec python3 "$(dirname "$0")/run_matrix.py" "$@"
