#!/usr/bin/env bash
# GT10 — Grid benchmark regression gate (ADR-0012 §Layer 3).
#
# Usage:
#   ./scripts/bench-grid-regress.sh [MODEL_SHORTHAND] [BASELINE_FILE]
#
# Required env vars:
#   LARQL_BENCH_VINDEX   — path to the vindex directory
#   LARQL_BENCH_FFN_URL  — URL of a running larql-server --ffn-only shard
#
# Optional env vars:
#   LARQL_TOK_PER_S_THRESHOLD  — fail if tok/s drops below baseline × (1 - this). Default: 0.05
#   LARQL_P99_THRESHOLD        — fail if p99 rises above baseline × (1 + this). Default: 0.10
#
# Example:
#   LARQL_BENCH_VINDEX=output/gemma3-4b-q4k.vindex \
#   LARQL_BENCH_FFN_URL=http://localhost:8080 \
#   ./scripts/bench-grid-regress.sh gemma3-4b-q4k

set -euo pipefail

MODEL="${1:-${LARQL_BENCH_VINDEX:-}}"
BASELINE_NAME="${2:-}"

if [ -z "${LARQL_BENCH_VINDEX:-}" ] && [ -z "${MODEL}" ]; then
    echo "error: set LARQL_BENCH_VINDEX or pass model path as first arg" >&2
    exit 1
fi

VINDEX="${LARQL_BENCH_VINDEX:-${MODEL}}"
MODEL_KEY="$(basename "${VINDEX}" .vindex)"

if [ -z "${BASELINE_NAME}" ]; then
    BASELINE="${BASH_SOURCE[0]%/*}/../bench/baselines/grid-${MODEL_KEY}.json"
else
    BASELINE="${BASELINE_NAME}"
fi

if [ -z "${LARQL_BENCH_FFN_URL:-}" ]; then
    echo "error: LARQL_BENCH_FFN_URL is not set (needs a running --ffn-only server)" >&2
    exit 1
fi

CURRENT="$(mktemp /tmp/larql-bench-XXXXXX.json)"
trap 'rm -f "${CURRENT}"' EXIT

echo "[bench-grid-regress] running bench against ${LARQL_BENCH_FFN_URL}..."
./target/release/larql bench "${VINDEX}" \
    --ffn "${LARQL_BENCH_FFN_URL}" \
    --wire f32,f16 \
    --tokens 30 \
    --warmup 3 \
    --output json \
    --output-file "${CURRENT}"

if [ ! -f "${BASELINE}" ]; then
    echo "[bench-grid-regress] no baseline found at ${BASELINE} — saving current run as baseline"
    mkdir -p "$(dirname "${BASELINE}")"
    cp "${CURRENT}" "${BASELINE}"
    echo "[bench-grid-regress] baseline saved. Re-run to compare."
    exit 0
fi

echo "[bench-grid-regress] comparing against baseline ${BASELINE}..."
python3 "$(dirname "${BASH_SOURCE[0]}")/bench_compare.py" \
    --baseline "${BASELINE}" \
    --current  "${CURRENT}" \
    --tok-per-s-threshold "${LARQL_TOK_PER_S_THRESHOLD:-0.05}" \
    --p99-threshold       "${LARQL_P99_THRESHOLD:-0.10}"
