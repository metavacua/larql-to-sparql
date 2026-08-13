#!/usr/bin/env bash
# DEC-0.5 — x86 expert-tier kernel gate (docs/dec-funnel.md v0.5 §3).
#
# Runs ON the rented x86 box (Ubuntu, ≥128GB RAM). Turnkey: clone, fetch
# the expert-server vindex slice + replay pools from HF, run the per-core
# kernel bench + the serving-level replay points, print the gate inputs.
# Designed to fit inside a ≤3h lease with time to spare.
#
# Binaries: `larql`/`larql-server` are FETCHED from the tagged release
# (ADR-0026), not built. The criterion kernel bench is the one thing still
# compiled here — it is the measurement object, not a shipped artifact —
# and it builds `larql-compute` alone, a much smaller graph than the CLI
# and server the fetch removed from the critical path.
#
# Prereqs on the box: git, curl, build-essential, pkg-config, libssl-dev,
# python3-pip (for huggingface_hub). Rust installed by this script (still
# needed for the bench).
#
# Usage:
#   HF_TOKEN=hf_... ./scripts/dec0p5-x86.sh [COMMIT]
#
# Optional env vars:
#   DEC0P5_LARQL_VERSION   — release tag to fetch (default: v0.1.0)
#   DEC0P5_BIN_DIR         — where fetched binaries land (default: $OUT/bin)
#   DEC0P5_LARQL_BIN / DEC0P5_SERVER_BIN
#                          — use these binaries, skip acquisition
#   DEC0P5_ALLOW_SOURCE_BUILD=1
#                          — permit building larql itself on a GPU host
#
# Outputs under ./dec0p5-out/: kernel bench log, replay run records +
# pulses (dense q8k B8 + routed experts-ml-q8k B8), kernel_class line,
# lscpu + core count — everything the registry run record needs.

set -euo pipefail

COMMIT="${1:-main}"
OUT="$PWD/dec0p5-out"
URL=http://127.0.0.1:8080
mkdir -p "$OUT"

echo "[dec0p5] host: $(uname -mo); cores: $(nproc)"
lscpu > "$OUT/lscpu.txt" 2>/dev/null || true

if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    . "$HOME/.cargo/env"
fi

if [ ! -d larql ]; then git clone --depth 50 https://github.com/chrishayuk/larql.git; fi
cd larql && git fetch --depth 50 origin "$COMMIT" && git checkout "$COMMIT"
git rev-parse HEAD > "$OUT/commit.txt"

pip3 install -q huggingface_hub
python3 - << 'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download("chrishayuk/gemma-4-26b-a4b-it-vindex-expert-server",
                  local_dir="vindex-expert-server")
snapshot_download("chrishayuk/dec0-residual-pools", repo_type="dataset",
                  local_dir="pools")
PYEOF

# ── Serving binaries (ADR-0026) ─────────────────────────────────────────
# Fetched, not built: `larql` + `larql-server` are shipped artifacts, and
# this box may carry an incidental GPU (the funnel's infra table provisions
# "cheapest attached" for DEC-0.5 because some marketplace tiers require
# one). The kernel bench below is the exception — see its note.
LARQL_RELEASE_VERSION="${DEC0P5_LARQL_VERSION:-v0.1.0}"
LARQL_BIN_DIR="${DEC0P5_BIN_DIR:-$OUT/bin}"
LARQL_BIN="${DEC0P5_LARQL_BIN:-}"
LARQL_SERVER_BIN="${DEC0P5_SERVER_BIN:-}"
LARQL_ALLOW_SOURCE_BUILD="${DEC0P5_ALLOW_SOURCE_BUILD:-0}"
LARQL_LOG_PREFIX="dec0p5"
# cwd is the cloned repo root at this point, so the helper resolves here.
# shellcheck source=lib/larql-binaries.sh
. scripts/lib/larql-binaries.sh
larql_acquire_binaries

# ── 1. Per-core kernel bench (single-thread GiB/s; AVX2-or-scalar on x86,
#      the startup kernel_class line pins which) ─────────────────────────
# This one MUST compile from source and is deliberately exempt from the
# fetch-don't-build rule: a criterion bench target is not a shipped binary,
# and the kernel it times is the object DEC-0.5 exists to measure. It builds
# `larql-compute` only — a far smaller graph than larql-cli + larql-server,
# which is precisely what the fetch above removed from the critical path.
echo "[dec0p5] kernel bench (single-thread, compiled from source — the measurement object)…"
cargo bench -p larql-compute --bench q4k_q8k_matvec 2>&1 | tee "$OUT/kernel-bench.log" | grep -E "GiB|time:|thrpt" | head -20

# ── 2. Serving-level points (loopback on this box) ──────────────────────
"${LARQL_SERVER_BIN}" vindex-expert-server --ffn-only --port 8080 \
    > "$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill "$SRV" 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
    curl -sf $URL/v1/health >/dev/null 2>&1 && break
    kill -0 "$SRV" 2>/dev/null || { echo "server died — see $OUT/server.log"; exit 1; }
    sleep 1
done
grep -E "kernel class|decode options" "$OUT/server.log" | tee "$OUT/kernel-class.txt"

echo "[dec0p5] dense q8k replay point (B ∈ {1,8}, batch dispatch)…"
"${LARQL_BIN}" dec-bench replay --ffn $URL --capture pools/dense \
    --endpoint walk-ffn --wire q8k --batch 1,8 --dispatch batch --repeats 3 \
    --net-rtt-ms 0.05 --net-gbps 0 \
    --output-file "$OUT/dense-q8k.json" --pulse-file "$OUT/dense-q8k.jsonl"

echo "[dec0p5] routed experts-ml-q8k replay point (B ∈ {1,8}, batch dispatch)…"
"${LARQL_BIN}" dec-bench replay --ffn $URL --capture pools/routed \
    --endpoint experts --wire q8k --batch 1,8 --dispatch batch --repeats 3 \
    --net-rtt-ms 0.05 --net-gbps 0 \
    --output-file "$OUT/routed-q8k.json" --pulse-file "$OUT/routed-q8k.jsonl"

echo
echo "[dec0p5] done. Gate inputs in $OUT:"
echo "  kernel-bench.log (single-thread GiB/s vs Mac baseline)"
echo "  kernel-class.txt (which kernel arm actually served)"
echo "  {dense,routed}-q8k.jsonl (step p50 per point; divide by cores for per-core)"
grep -h "step_ms_p50" "$OUT"/*.jsonl 2>/dev/null | head -8 || true
