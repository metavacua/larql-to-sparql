#!/bin/bash
# WeightFfn — Dense ground truth inference
# This is the standard model prediction path.

MODEL="google/gemma-3-4b-it"

echo "=== Dense FFN (WeightFfn) ==="
cargo run --release -- predict "$MODEL" \
  -p "The capital of France is"

echo ""
echo "=== Multiple prompts ==="
for prompt in "The capital of France is" "The capital of Germany is" "Who wrote Hamlet"; do
  cargo run --release -- predict "$MODEL" -p "$prompt" 2>&1 | grep "1\."
done
