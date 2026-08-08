#!/usr/bin/env bash
# Thin entrypoint → run_matrix.py (the real runner). Kept so the workflow and
# README keep a stable `run_matrix.sh <level> <vindex> <corpus> <out>` API while
# the capture logic lives in Python, where UTF-8 handling, full-stream files,
# and shell-free subprocess orchestration are correct by construction.
# (It used to say "codepoint-safe truncation" too. Nothing is truncated any
# more — the captures are verbatim, which is the point.)
# All configuration is via env (see run_matrix.py):
#   LARQL_BIN  MODEL_ID  TMPROOT  WRAP  CELL_TIMEOUT
# Trailing flags pass through, e.g. --driver repl-pipe (default lql).
set -uo pipefail
exec python3 "$(dirname "$0")/run_matrix.py" "$@"
