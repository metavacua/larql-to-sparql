# FFN Backend Examples

Shell scripts demonstrating each FFN backend. All require a built vindex:

```bash
# Build vindex first (one-time, ~10 min)
cargo run --release -- extract-index google/gemma-3-4b-it -o output/gemma3-4b.vindex --include-weights
```

## Production

- `weight_dense.sh` — Dense FFN (ground truth)
- `walk_infer.sh` — WalkFfn: dense FFN + vindex trace (INFER path)
- `sparse_sweep.sh` — SparseFfn at various K values

## Experimental

- `cached_roundtrip.sh` — CachedFfn: calibrate, save, load, verify bit-identical
- `ffn_bottleneck.sh` — Bottleneck analysis: where FFN time goes
- `ffn_bench.sh` — Compare all backends on one layer
