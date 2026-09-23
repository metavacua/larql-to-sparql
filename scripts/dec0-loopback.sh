#!/usr/bin/env bash
# DEC-0 arm M — loopback batch curve (docs/dec-funnel.md §3).
#
# Runs the full DEC-0 measurement on one machine:
#   1. launches a loopback expert server (larql-server --ffn-only),
#   2. anchor arm: e2e single-stream `larql bench --ffn` (streaming + batch
#      dispatch, wire sweep) — the re-baseline against the known numbers,
#   3. capture arm: 64-prompt residual pool via `larql dec-bench capture`,
#   4. replay arm: batch × wire × dispatch sweep via `larql dec-bench replay`,
#      emitting the dec/* pulse JSONL + full JSON run record,
#   5. prints the C1 verdict input (step p50 vs batch sub-linearity).
#
# Usage:
#   LARQL_BENCH_VINDEX=output/gemma4-26b-a4b-q4k.vindex ./scripts/dec0-loopback.sh
#
# Optional env vars:
#   DEC0_PORT       — expert server port            (default: 8080)
#   DEC0_OUT_DIR    — output directory              (default: bench/dec0)
#   DEC0_PROMPTS    — prompts file                  (default: bench/dec0/prompts.txt)
#   DEC0_STEPS      — capture/replay steps          (default: 16)
#   DEC0_PROMPT_N   — prompts to capture            (default: 64)
#   DEC0_REPEATS    — replay repeats per point      (default: 3)
#   DEC0_BATCHES    — replay batch list             (default: 1,8,16,32,64)
#   DEC0_WIRES      — replay wire list              (default: f32,f16,i8,q8k)
#   DEC0_SKIP_ANCHOR / DEC0_SKIP_CAPTURE — set to 1 to skip a phase (reuse
#      an existing pool with DEC0_SKIP_CAPTURE=1).
#
# Bench discipline (thermal-artifacts memory): run on AC power with
# indexers quiet; suspect any regression measured on a hot machine.

set -euo pipefail

VINDEX="${LARQL_BENCH_VINDEX:?set LARQL_BENCH_VINDEX to the vindex directory}"
PORT="${DEC0_PORT:-8080}"
OUT_DIR="${DEC0_OUT_DIR:-bench/dec0}"
PROMPTS="${DEC0_PROMPTS:-bench/dec0/prompts.txt}"
STEPS="${DEC0_STEPS:-16}"
PROMPT_N="${DEC0_PROMPT_N:-64}"
REPEATS="${DEC0_REPEATS:-3}"
BATCHES="${DEC0_BATCHES:-1,8,16,32,64}"
WIRES="${DEC0_WIRES:-f32,f16,i8,q8k}"
URL="http://127.0.0.1:${PORT}"
MODEL_KEY="$(basename "${VINDEX}" .vindex)"
STAMP="$(date +%Y%m%d-%H%M%S)"

mkdir -p "${OUT_DIR}"

echo "[dec0] building release binaries…"
cargo build --release -p larql-cli -p larql-server

echo "[dec0] launching expert server (${URL}, --ffn-only)…"
./target/release/larql-server "${VINDEX}" --ffn-only --port "${PORT}" \
    >"${OUT_DIR}/server-${STAMP}.log" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

for i in $(seq 1 120); do
    if curl -sf "${URL}/v1/health" >/dev/null 2>&1; then break; fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[dec0] server died during startup — see ${OUT_DIR}/server-${STAMP}.log" >&2
        exit 1
    fi
    sleep 1
    if [ "$i" = 120 ]; then
        echo "[dec0] server did not become healthy in 120s" >&2
        exit 1
    fi
done
echo "[dec0] server healthy."

# ── 1. Anchor arm: e2e single-stream (existing bench path, unchanged) ────────
if [ "${DEC0_SKIP_ANCHOR:-0}" != "1" ]; then
    for dispatch in streaming batch; do
        echo "[dec0] anchor arm: --ffn-dispatch ${dispatch}…"
        ./target/release/larql bench "${VINDEX}" \
            --ffn "${URL}" \
            --ffn-dispatch "${dispatch}" \
            --metal \
            --wire f32,f16,i8 \
            --tokens 50 \
            --warmup 3 \
            --output json \
            --output-file "${OUT_DIR}/anchor_${MODEL_KEY}_${dispatch}_${STAMP}.json"
    done
fi

# ── 2. Capture arm: residual pool ────────────────────────────────────────────
POOL="${OUT_DIR}/residuals-${MODEL_KEY}"
if [ "${DEC0_SKIP_CAPTURE:-0}" != "1" ]; then
    echo "[dec0] capturing ${PROMPT_N}-prompt residual pool (${STEPS} steps)…"
    ./target/release/larql dec-bench capture "${VINDEX}" \
        --ffn "${URL}" \
        --prompts-file "${PROMPTS}" \
        --num-prompts "${PROMPT_N}" \
        --steps "${STEPS}" \
        --metal \
        --out "${POOL}"
fi

# ── 3. Replay arm: the batch curve ───────────────────────────────────────────
PULSE="${OUT_DIR}/dec0_${MODEL_KEY}_pulse_${STAMP}.jsonl"
RECORD="${OUT_DIR}/dec0_${MODEL_KEY}_replay_${STAMP}.json"
echo "[dec0] replay sweep: batch {${BATCHES}} × wire {${WIRES}} × dispatch {streaming,batch}…"
./target/release/larql dec-bench replay \
    --ffn "${URL}" \
    --capture "${POOL}" \
    --batch "${BATCHES}" \
    --wire "${WIRES}" \
    --dispatch streaming,batch \
    --repeats "${REPEATS}" \
    --net-rtt-ms 0.05 \
    --net-gbps 0 \
    --output-file "${RECORD}" \
    --pulse-file "${PULSE}"

echo
echo "[dec0] done."
echo "  run record : ${RECORD}"
echo "  pulse      : ${PULSE}"
echo
echo "[dec0] C1 check — dec/step_ms_p50 vs batch (sub-linear through 32 = pass):"
python3 - "$PULSE" <<'EOF'
import json, sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1])]
by_arm = defaultdict(list)
for r in rows:
    by_arm[(r["dec/wire_format"], r["dec/dispatch_mode"])].append(r)
for (wire, dispatch), pts in sorted(by_arm.items()):
    pts.sort(key=lambda r: r["dec/batch"])
    base = next((p for p in pts if p["dec/batch"] == 1), None)
    print(f"  {wire:>4} / {dispatch:<9}", end="")
    for p in pts:
        b, ms = p["dec/batch"], p["dec/step_ms_p50"]
        scale = f" (×{ms / base['dec/step_ms_p50']:.1f} @ B{b})" if base and b > 1 else ""
        print(f"  B{b}: {ms:.1f}ms{scale}", end="")
    print()
print("  sub-linear = ×factor at B32 well below 32 (bandwidth amortising).")
EOF
