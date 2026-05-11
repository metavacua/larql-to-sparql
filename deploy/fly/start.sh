#!/bin/bash
set -e

VINDEX_DIR="${VINDEX_PATH:-/data/vindex}"
HF_REPO="${HF_REPO:-chrishayuk/gemma-4-26b-a4b-it-vindex-expert-server}"

# Verify the vindex is complete: index.json + embeddings + interleaved FFN + 30 layer files
LAYER_COUNT=$(ls "$VINDEX_DIR/layers/"*.weights 2>/dev/null | wc -l)
HAS_EMBED=$([ -f "$VINDEX_DIR/embeddings.bin" ] && echo yes || echo no)
HAS_FFN=$([ -f "$VINDEX_DIR/interleaved_q4k.bin" ] && echo yes || echo no)
if [ ! -f "$VINDEX_DIR/index.json" ] || [ "$HAS_EMBED" = "no" ] || [ "$HAS_FFN" = "no" ] || [ "$LAYER_COUNT" -lt 30 ]; then
  echo "Vindex incomplete (layers=$LAYER_COUNT/30 embed=$HAS_EMBED ffn=$HAS_FFN) — re-downloading..."
  rm -rf "$VINDEX_DIR"
  mkdir -p "$VINDEX_DIR"
  HF_HUB_ENABLE_HF_TRANSFER=1 python3 - <<PYEOF
import os, sys
from huggingface_hub import snapshot_download

repo_id = os.environ.get("HF_REPO", "chrishayuk/gemma-4-26b-a4b-it-vindex-expert-server")
token   = os.environ.get("HF_TOKEN") or None
dest    = os.environ.get("VINDEX_PATH", "/data/vindex")

print(f"Downloading {repo_id} → {dest}", flush=True)
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=dest,
    token=token,
    ignore_patterns=["*.md", ".gitattributes"],
)
print("Download complete.", flush=True)
PYEOF
  echo "Vindex ready at $VINDEX_DIR"
fi

echo "Starting larql-server from $VINDEX_DIR"
echo "  EXPERTS: ${EXPERTS:-all}"
echo "  LAYERS:  ${LAYERS:-all}"

EXTRA_ARGS=""
[ -n "$EXPERTS" ] && EXTRA_ARGS="$EXTRA_ARGS --experts $EXPERTS"
[ -n "$LAYERS"  ] && EXTRA_ARGS="$EXTRA_ARGS --layers $LAYERS"

# --warmup-walk-ffn pre-faults the owned expert pages into RAM at startup.
# This prevents mmap thrashing: pages for the owned shard are hot before the
# first request; pages for other shards are never touched (--experts filter).
# On performance-8x (16 GB), each 64-expert shard needs ~8 GB → 8 GB headroom.
[ "${WARMUP:-0}" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --warmup-walk-ffn"
[ -n "$GRPC_PORT"  ] && EXTRA_ARGS="$EXTRA_ARGS --grpc-port $GRPC_PORT"

exec larql-server "$VINDEX_DIR" --port "${PORT:-8080}" --host 0.0.0.0 $EXTRA_ARGS
