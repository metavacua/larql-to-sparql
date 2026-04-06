#!/bin/bash
# SparseFfn — sweep K to find accuracy/speed tradeoff

VINDEX="output/gemma3-4b.vindex"
MODEL="google/gemma-3-4b-it"
PROMPTS="The capital of France is,The capital of Germany is,Who wrote Hamlet,What is 2+2,The president of Brazil is"

echo "=== SparseFfn K sweep (attention + sparse FFN) ==="
cargo run --release -- vindex-bench --index "$VINDEX" -m "$MODEL" \
  --prompts "$PROMPTS" \
  -k 100,500,1000,2000,4000,8092
