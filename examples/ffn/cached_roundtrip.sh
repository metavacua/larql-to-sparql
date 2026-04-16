#!/bin/bash
# CachedFfn — calibrate, save, load, verify bit-identical

MODEL="google/gemma-3-4b-it"
CACHE_DIR="output/ffn_caches"

echo "=== CachedFfn: calibrate + save ==="
cargo run --release -- graph-walk -m "$MODEL" \
  --prompts "The capital of France is,The capital of Germany is,Who wrote Hamlet" \
  --save "$CACHE_DIR" --compare

echo ""
echo "=== CachedFfn: load + verify ==="
cargo run --release -- graph-walk -m "$MODEL" \
  --prompts "The capital of France is" \
  --load "$CACHE_DIR/prompt_0.ffncache" --compare

echo ""
echo "=== Cache file sizes ==="
ls -lh "$CACHE_DIR"/*.ffncache 2>/dev/null
