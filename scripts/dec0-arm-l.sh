#!/usr/bin/env bash
# DEC-0 arm L — loopback batch curve, Linux/Colab (docs/dec-funnel.md §3).
#
# Sibling of scripts/dec0-loopback.sh (arm M). Differs in three ways:
#   1. No Metal — this box has no GPU attention path pre-G-ladder, and
#      dec-bench replay never invokes attention anyway (it replays captured
#      residuals straight at the expert/FFN server).
#   2. No capture — arm L's whole point is host-portability: it replays the
#      *same* 64-prompt pool captured on the Mac (arm M), fetched from
#      wherever it was staged, not a fresh Linux-side capture.
#   3. No e2e single-stream anchor by default — that needs the FULL vindex
#      (attention + embed + FFN weights, ~tens of GB) plus a working CPU
#      attention decode path. The expert-server-only vindex slice
#      (`hf://chrishayuk/gemma-4-26b-a4b-it-vindex-expert-server`) is enough
#      for the replay sweep, which is the claim-bearing measurement (batch
#      *shape*, not absolute tok/s — dec-funnel.md §3 DEC-0). Set
#      DEC0L_RUN_ANCHOR=1 with DEC0L_ANCHOR_VINDEX pointing at a full vindex
#      to also run the anchor arm.
#
# Usage:
#   POOL_DENSE_URL=https://.../residuals-gemma4-26b-a4b-q4k.tar.gz \
#   POOL_ROUTED_URL=https://.../residuals-gemma4-26b-a4b-q4k-routed.tar.gz \
#   ./scripts/dec0-arm-l.sh
#
# Optional env vars:
#   DEC0L_VINDEX         — expert-server vindex source (default: the
#                           published HF expert-server slice; `larql-server`
#                           auto-downloads hf:// on first launch)
#   DEC0L_PORT            (default: 8080)
#   DEC0L_OUT_DIR          (default: bench/dec0)
#   DEC0L_POOL_DIR         (default: bench/dec0, pools land in
#                           $DEC0L_POOL_DIR/residuals-gemma4-26b-a4b-q4k[-routed])
#   POOL_DENSE_URL        — required unless the dense pool dir already exists
#   POOL_ROUTED_URL       — required unless the routed pool dir already exists
#   DEC0L_STEPS            (default: 16, must match the captured pool)
#   DEC0L_REPEATS          (default: 3)
#   DEC0L_BATCHES           (default: 1,8,16,32,64)
#   DEC0L_DENSE_WIRES       (default: f32,f16,i8,q8k)
#   DEC0L_ROUTED_WIRES      (default: f32,q8k — the experts endpoint's only
#                           supported wires, per `larql dec-bench replay --help`)
#   DEC0L_RUN_ANCHOR=1     — also run the e2e single-stream anchor arm
#   DEC0L_ANCHOR_VINDEX    — full vindex for the anchor arm (required if
#                           DEC0L_RUN_ANCHOR=1)
#
# Binaries (ADR-0026 — GPU hosts never build from source):
#   DEC0L_LARQL_VERSION    — release tag to fetch (default: v0.1.0)
#   DEC0L_BIN_DIR          — where fetched binaries land
#                           (default: $DEC0L_OUT_DIR/bin; reused if present)
#   DEC0L_LARQL_BIN        — skip acquisition, use this `larql` binary
#   DEC0L_SERVER_BIN       — skip acquisition, use this `larql-server` binary
#   DEC0L_ALLOW_SOURCE_BUILD=1
#                         — permit `cargo build` even on a GPU host. Only if
#                           you accept burning the GPU allocation on ~20-40
#                           minutes of CPU-only compilation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORT="${DEC0L_PORT:-8080}"
OUT_DIR="${DEC0L_OUT_DIR:-bench/dec0}"
POOL_DIR="${DEC0L_POOL_DIR:-bench/dec0}"
VINDEX="${DEC0L_VINDEX:-hf://chrishayuk/gemma-4-26b-a4b-it-vindex-expert-server}"
STEPS="${DEC0L_STEPS:-16}"
REPEATS="${DEC0L_REPEATS:-3}"
BATCHES="${DEC0L_BATCHES:-1,8,16,32,64}"
DENSE_WIRES="${DEC0L_DENSE_WIRES:-f32,f16,i8,q8k}"
ROUTED_WIRES="${DEC0L_ROUTED_WIRES:-f32,q8k}"
URL="http://127.0.0.1:${PORT}"
STAMP="$(date +%Y%m%d-%H%M%S)"

DENSE_POOL="${POOL_DIR}/residuals-gemma4-26b-a4b-q4k"
ROUTED_POOL="${POOL_DIR}/residuals-gemma4-26b-a4b-q4k-routed"

mkdir -p "${OUT_DIR}" "${POOL_DIR}"

fetch_pool() {
    local dir="$1" url_var="$2"
    if [ -f "${dir}/manifest.json" ]; then
        echo "[dec0-arm-l] pool already present: ${dir}"
        return
    fi
    local url="${!url_var:-}"
    if [ -z "${url}" ]; then
        echo "[dec0-arm-l] ${dir} missing and ${url_var} not set — set it to a" \
             "fetchable tar.gz of the arm-M pool (dec-funnel.md: pools are" \
             "host-portable, capture is model-free)." >&2
        exit 1
    fi
    echo "[dec0-arm-l] fetching pool: ${url}"
    local tmp
    tmp="$(mktemp)"
    curl -fsSL "${url}" -o "${tmp}"
    mkdir -p "${dir}"
    tar -xzf "${tmp}" -C "${dir}" --strip-components=1
    rm -f "${tmp}"
}

fetch_pool "${DENSE_POOL}" POOL_DENSE_URL
fetch_pool "${ROUTED_POOL}" POOL_ROUTED_URL

# ── Binary acquisition (ADR-0026) — shared with the other DEC drivers ───────
LARQL_RELEASE_VERSION="${DEC0L_LARQL_VERSION:-v0.1.0}"
LARQL_BIN_DIR="${DEC0L_BIN_DIR:-${OUT_DIR}/bin}"
LARQL_BIN="${DEC0L_LARQL_BIN:-}"
LARQL_SERVER_BIN="${DEC0L_SERVER_BIN:-}"
LARQL_ALLOW_SOURCE_BUILD="${DEC0L_ALLOW_SOURCE_BUILD:-0}"
LARQL_LOG_PREFIX="dec0-arm-l"
# shellcheck source=lib/larql-binaries.sh
. "${SCRIPT_DIR}/lib/larql-binaries.sh"
larql_acquire_binaries
SERVER_BIN="${LARQL_SERVER_BIN}"

echo "[dec0-arm-l] launching expert server (${URL}, --ffn-only, vindex=${VINDEX})…"
echo "[dec0-arm-l] first launch downloads the vindex from HuggingFace if not cached — allow a few minutes."
"${SERVER_BIN}" "${VINDEX}" --ffn-only --port "${PORT}" \
    >"${OUT_DIR}/server-arml-${STAMP}.log" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

for i in $(seq 1 600); do
    if curl -sf "${URL}/v1/health" >/dev/null 2>&1; then break; fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[dec0-arm-l] server died during startup — see ${OUT_DIR}/server-arml-${STAMP}.log" >&2
        exit 1
    fi
    sleep 1
    if [ "$i" = 600 ]; then
        echo "[dec0-arm-l] server did not become healthy in 600s (cold HF download can be slow)" >&2
        exit 1
    fi
done
echo "[dec0-arm-l] server healthy."

# ── Optional anchor arm (off by default — see header) ───────────────────────
if [ "${DEC0L_RUN_ANCHOR:-0}" = "1" ]; then
    ANCHOR_VINDEX="${DEC0L_ANCHOR_VINDEX:?set DEC0L_ANCHOR_VINDEX for the anchor arm}"
    for dispatch in streaming batch; do
        echo "[dec0-arm-l] anchor arm: --ffn-dispatch ${dispatch}…"
        "${LARQL_BIN}" bench "${ANCHOR_VINDEX}" \
            --ffn "${URL}" \
            --ffn-dispatch "${dispatch}" \
            --wire f32,f16,i8 \
            --tokens 50 \
            --warmup 3 \
            --output json \
            --output-file "${OUT_DIR}/anchor_arml_${dispatch}_${STAMP}.json"
    done
fi

# ── Dense replay: walk-ffn endpoint ──────────────────────────────────────────
DENSE_PULSE="${OUT_DIR}/dec0_arml_dense_pulse_${STAMP}.jsonl"
DENSE_RECORD="${OUT_DIR}/dec0_arml_dense_replay_${STAMP}.json"
echo "[dec0-arm-l] dense replay sweep: batch {${BATCHES}} × wire {${DENSE_WIRES}} × dispatch {streaming,batch}…"
"${LARQL_BIN}" dec-bench replay \
    --ffn "${URL}" \
    --capture "${DENSE_POOL}" \
    --endpoint walk-ffn \
    --batch "${BATCHES}" \
    --wire "${DENSE_WIRES}" \
    --dispatch streaming,batch \
    --repeats "${REPEATS}" \
    --steps "${STEPS}" \
    --net-rtt-ms 0.05 \
    --net-gbps 0 \
    --output-file "${DENSE_RECORD}" \
    --pulse-file "${DENSE_PULSE}"

# ── Routed replay: experts endpoint (needs the --routing sidecars) ──────────
ROUTED_PULSE="${OUT_DIR}/dec0_arml_routed_pulse_${STAMP}.jsonl"
ROUTED_RECORD="${OUT_DIR}/dec0_arml_routed_replay_${STAMP}.json"
echo "[dec0-arm-l] routed replay sweep: batch {${BATCHES}} × wire {${ROUTED_WIRES}} × dispatch {streaming,batch}…"
"${LARQL_BIN}" dec-bench replay \
    --ffn "${URL}" \
    --capture "${ROUTED_POOL}" \
    --endpoint experts \
    --batch "${BATCHES}" \
    --wire "${ROUTED_WIRES}" \
    --dispatch streaming,batch \
    --repeats "${REPEATS}" \
    --steps "${STEPS}" \
    --net-rtt-ms 0.05 \
    --net-gbps 0 \
    --output-file "${ROUTED_RECORD}" \
    --pulse-file "${ROUTED_PULSE}"

echo
echo "[dec0-arm-l] done."
echo "  dense  run record : ${DENSE_RECORD}"
echo "  dense  pulse       : ${DENSE_PULSE}"
echo "  routed run record : ${ROUTED_RECORD}"
echo "  routed pulse       : ${ROUTED_PULSE}"
echo
echo "  C1 verdict input: step p50 vs batch, both pools — compare against the"
echo "  arm-M numbers in docs/dec-funnel.md §3 DEC-0. Kill condition: either"
echo "  pool saturates below batch 16 on this box while arm M does not."
echo
