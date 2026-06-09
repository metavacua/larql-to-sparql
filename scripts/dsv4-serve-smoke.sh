#!/usr/bin/env bash
# §6.5 — produce a real DSv4-Flash vindex and smoke-test serving one token
# through /v1/chat/completions. This is the end-to-end validation the code
# arc (extraction → serving reader → hp reconstruct → chat route) has NOT
# yet had: it needs the 172 GB GGUF, ~440 GB RAM for the resident
# reconstruction, and ~161 GB of output. RUN WHEN THE BOX IS FREE — not
# wired into CI, not run unattended.
#
# Prereqs:
#   - DeepSeek-V4-Flash GGUF on disk (DSV4_GGUF below)
#   - its HuggingFace tokenizer.json (DSV4_TOKENIZER below) — the GGUF
#     stores tokenizer data as metadata KVs, not an HF file
set -euo pipefail

GGUF="${DSV4_GGUF:-/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf}"
TOKENIZER="${DSV4_TOKENIZER:-/tank/ai/deepseek-ai/DeepSeek-V4-Flash/tokenizer.json}"
VINDEX="${DSV4_VINDEX:-/tank/ai/vindex/deepseek-v4-flash}"
PORT="${PORT:-8080}"

echo "== §6.5 DSv4 serve smoke =="
echo "  GGUF      : $GGUF"
echo "  tokenizer : $TOKENIZER"
echo "  vindex    : $VINDEX  (port $PORT)"
[ -f "$GGUF" ] || { echo "FATAL: GGUF not found"; exit 1; }
[ -f "$TOKENIZER" ] || { echo "FATAL: tokenizer.json not found (needed so the server can tokenise)"; exit 1; }

# ── 1. Produce the server-conforming DSv4 vindex (~161 GB out) ──
# Uses the dedicated DSv4 extraction path (build_dsv4_vindex), NOT the
# generic gguf-to-vindex (which rejects deepseek_v4). Emits the per-blob
# weight files + index.json (VindexConfig) + embeddings.bin + tokenizer.json.
if [ ! -f "$VINDEX/index.json" ]; then
    echo "== building DSv4 vindex (this is the long, RAM-heavy step) =="
    cargo run --release -p larql-cli -- convert gguf-to-dsv4-vindex \
        "$GGUF" --output "$VINDEX" --tokenizer "$TOKENIZER"
else
    echo "== vindex already present at $VINDEX — skipping build =="
fi

# ── 2. Serve it ──
# The server auto-detects family=deepseek_v4 from index.json and routes
# /v1/chat/completions to run_dsv4_chat_completion (resident generate).
echo "== starting larql server (background) =="
cargo run --release -p larql-cli -- serve "$VINDEX" --port "$PORT" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
# Wait for readiness (the first chat request also triggers the ~440 GB
# resident reconstruction, cached for the server's lifetime).
for i in $(seq 1 60); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break
    sleep 2
done

# ── 3. Smoke one chat completion ──
echo "== POST /v1/chat/completions =="
curl -sS "http://localhost:$PORT/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"In one sentence, what is a transformer?"}],"max_tokens":32}' \
    | tee "$VINDEX/../dsv4-smoke-response.json"
echo
echo "== PASS if the response above contains coherent generated text =="
# This is the first DSv4 token generated through HTTP. If it's garbled,
# the bug is in the resident forward / hp reconstruction, not the wiring.
