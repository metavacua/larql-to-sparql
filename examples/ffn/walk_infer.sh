#!/bin/bash
# WalkFfn — Dense FFN + vindex interpretability trace
# This is the INFER path: exact answers with visible feature activations.

VINDEX="output/gemma3-4b.vindex"
MODEL="google/gemma-3-4b-it"

echo "=== Walk with prediction (attention + vindex FFN) ==="
echo "Bit-identical to dense, with feature trace"
echo ""

cargo run --release -- walk --index "$VINDEX" -m "$MODEL" \
  -p "The capital of France is" \
  -k 10 --predict --compare -v

echo ""
echo "=== Compare walk vs dense ==="
cargo run --release -- walk --index "$VINDEX" -m "$MODEL" \
  -p "The capital of France is" \
  -k 10 --predict --compare
