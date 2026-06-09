#!/usr/bin/env bash
# Bench plan: larql DSv4-Flash vs llama.cpp PR #23122, same box, same model.
#
# PURPOSE (the "re-litigate the thesis" task): produce real numbers on
# whether larql's CPU-FFN + GPU-attention hybrid is competitive with
# llama.cpp on DeepSeek-V4-Flash. larql currently can't beat llama.cpp on
# a measured metric — this script gathers the data to confirm or refute
# that, which should gate further DSv4 investment.
#
# COST: loads the 172 GB GGUF (~161 GB+ RSS resident) and runs minutes of
# decode per config; pegs this shared box for a while. Run when the
# machine is free. Nothing here is wired into CI.
#
# Usage:  scripts/bench-dsv4-vs-llamacpp.sh [PROMPT_TOKENS] [GEN_TOKENS]
set -euo pipefail

GGUF="${DSV4_GGUF:-/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf}"
PROMPT_TOKENS="${1:-512}"
GEN_TOKENS="${2:-128}"
LLAMA_DIR="${LLAMA_CPP_DIR:-/tank/ai/llama.cpp-pr23122}"   # PR #23122 checkout
OUT="${OUT_DIR:-/tank/ai/tmp/dsv4-bench}"
mkdir -p "$OUT"

echo "== config =="
echo "  GGUF          : $GGUF"
echo "  prompt/gen    : $PROMPT_TOKENS / $GEN_TOKENS tokens"
echo "  llama.cpp     : $LLAMA_DIR"
echo "  out           : $OUT"
[ -f "$GGUF" ] || { echo "FATAL: GGUF not found at $GGUF"; exit 1; }

# ── 1. Hardware/thermal context (so numbers are comparable run-to-run) ──
echo "== machine ==" | tee "$OUT/machine.txt"
{ nproc; free -g | head -2; (nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"); } | tee -a "$OUT/machine.txt"

# ── 2. llama.cpp baseline (PR #23122 — the DSv4-Flash reference impl) ──
# Build once:
#   cd "$LLAMA_DIR" && cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target llama-bench llama-cli
# llama-bench reports prompt (pp) + generation (tg) tok/s directly.
echo "== llama.cpp llama-bench =="
"$LLAMA_DIR/build/bin/llama-bench" \
    -m "$GGUF" \
    -p "$PROMPT_TOKENS" -n "$GEN_TOKENS" \
    -ngl 99 \
    -r 3 \
    -o md | tee "$OUT/llamacpp.md"
# Capture: pp tok/s (prefill), tg tok/s (decode). Decode tg is the headline
# number to beat — DSv4 decode is weight-bandwidth-bound.

# ── 3. larql DSv4-Flash decode ──
# larql has no DSv4 server/CLI path yet (the HTTP server is vindex-only and
# rejects deepseek_v4), so timing goes through the resident-generate test
# harness in larql-inference. The #[ignore] test below should print
# `prefill_ms`, `decode_ms_per_tok` (add a one-line eprintln if missing):
#
#   cargo test --release -p larql-inference --features cuda \
#       --test <dsv4 decode bench test> -- --ignored --nocapture
#
# Recommended: add a dedicated #[ignore] bench test `dsv4_decode_timing`
# that loads resident layers once (load_dsv4_resident_layers) + head, then
# times dsv4_resident_generate_with_prefix_cache over GEN_TOKENS from a
# PROMPT_TOKENS prompt, printing ms/tok for both CPU-only and
# (LARQL_DSV4_GPU=1) hybrid backends. Run both:
echo "== larql DSv4 decode (CPU FFN, GPU attention hybrid) =="
echo "  TODO: cargo test --release -p larql-inference --features cuda \\"
echo "        dsv4_decode_timing -- --ignored --nocapture  2>&1 | tee $OUT/larql-hybrid.txt"
echo "  TODO: LARQL_DSV4_GPU=0 ... | tee $OUT/larql-cpu.txt   # CPU-only baseline"

# ── 4. Compare ──
# The decision metric: larql decode tok/s  vs  llama.cpp tg tok/s.
#  - If larql is within ~1.5x of llama.cpp: the hybrid is viable; invest.
#  - If larql is >3x slower with no clear path to parity: the DSv4 work
#    needs a different justification (research / vindex semantic features),
#    since llama.cpp PR #23122 already runs this model.
# Also record peak RSS (the >VRAM driving-goal claim) via /usr/bin/time -v.
echo "== compare: extract tg tok/s from $OUT/llamacpp.md and ms/tok from larql, compute ratio =="
echo "Done. Raw outputs in $OUT/."
