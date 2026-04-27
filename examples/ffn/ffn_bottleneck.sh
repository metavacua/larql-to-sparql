#!/bin/bash
# FFN Bottleneck Analysis — where does FFN time go?

MODEL="google/gemma-3-4b-it"

echo "=== FFN Bottleneck (layer 20) ==="
cargo run --release -- ffn-bottleneck -m "$MODEL" -l 20 --iterations 50

echo ""
echo "=== Attention Bottleneck (layer 20) ==="
cargo run --release -- attn-bottleneck -m "$MODEL" -l 20 --iterations 50
