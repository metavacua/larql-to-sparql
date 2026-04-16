#!/bin/bash
# FFN Bench — compare all backends on one layer

MODEL="google/gemma-3-4b-it"
GATE_INDEX="output/gemma3-4b.gate-index.jsonl"

echo "=== FFN Backend Comparison (layer 20) ==="
echo "Dense vs Sparse vs Cached vs Entity-routed vs Clustered"
echo ""

cargo run --release -- ffn-bench -m "$MODEL" \
  -l 20 --iterations 50 \
  -k 64,256,1024,4096,8192 \
  --gate-index "$GATE_INDEX" \
  --clusters 128 --top-c-values 1,2,4,8
