#!/usr/bin/env bash
# Per-crate wasm32v1-none gating pipeline.
# Usage: tools/pipeline.sh <crate-base>   e.g. tools/pipeline.sh larql-kv
set -euo pipefail
BASE="$1"
CRATE="${BASE}-wasm32v1-none"
WASM="crates/larql-wasm/Cargo.toml"
TOOLS="crates/larql-wasm/tools"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

echo "── pipeline: $CRATE ──"
# normalize FIRST to repair any earlier buggy-insert breakage, THEN gate.
python3 "$TOOLS/normalize_prelude.py" "$CRATE"      | tail -1
python3 "$TOOLS/autogate_fn.py" "$CRATE"            | tail -1
python3 "$TOOLS/gate_cascade.py" "$CRATE" --max 15  | tail -3
python3 "$TOOLS/dedupe_gates.py" "$CRATE"           | tail -1
python3 "$TOOLS/normalize_prelude.py" "$CRATE"      | tail -1

w=$(cargo check --manifest-path "$WASM" -p "$CRATE" --target wasm32v1-none --lib 2>&1 | grep -c '^error\[' || true)
n=$(cargo check --manifest-path "$WASM" -p "$CRATE" --lib 2>&1 | grep -c '^error\[' || true)
echo "RESULT $CRATE: wasm32=$w native=$n"
