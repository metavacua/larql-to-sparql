# ADR-012: Encoder Merging — Reduce Per-Layer Metal Encoders

**Status**: Accepted (partial benefit)  
**Date**: 2026-04-08  
**Context**: The decode pipeline used 7-12 separate Metal compute command encoders per layer. The hypothesis was that reducing encoder count would save ~8ms of dispatch overhead for a 34-layer model.

## Decision

Merge the per-layer dispatch structure from ~10 encoders to 3-4 per layer:

- **Encoder A**: Input norm → QKV projection → RoPE → V-norm (all pre-attention)
- **Encoder B**: KV cache append → KV attend (separate for cache write barrier)
- **Encoder C**: O projection → residual+norm+Q8 → FFN → post-FFN residual → layer scalar

Also refactored `kv_cache.rs` to expose `encode_kv_append` and `encode_kv_attend` functions that dispatch into an existing encoder, enabling the merge.

## Measured Impact

```
Before merging:  17.8ms / 56 tok/s  (21 layers)
After merging:   17.5ms / 57 tok/s  (21 layers)
Improvement:     ~0.3ms (1.7%)
```

## Why the Impact Was Small

**The encoder creation overhead was already negligible** — 0.05ms for 238 empty encoders (0.0002ms each). The hypothesis that encoder boundaries caused 8ms of overhead was wrong.

The actual cost breakdown (34 layers):
- FFN compute: 13.0ms (35.8%) — at hardware bandwidth limit
- KV cache attend: 10.5ms (28.9%) — real GPU work
- Norm compute: 10.6ms (29.0%) — real GPU work (reading all elements for RMS)
- QKV projection: 1.3ms (3.4%)
- Encoder overhead: 0.05ms (0.0%) — **not the bottleneck**

The "dispatch overhead" seen in component profiling (0.155ms/layer for rms_norm) is the **actual GPU compute time** for the norm operation, not Metal API overhead. RMSNorm reads all hidden_size elements for mean-of-squares, which takes ~0.15ms at memory bandwidth.

## Consequences

- **Good**: Cleaner code structure — clear A/B/C encoder phases per layer
- **Good**: Fewer synchronization points (potential for future GPU overlap)
- **Good**: KV cache operations now composable via encode_kv_append/encode_kv_attend
- **Lesson**: Metal encoder creation is nearly free. The real optimization path is **reducing the number of layers computed** (template caching: 34 → 8 layers = 4.25x speedup).

## Updated Path to Ollama Parity

Encoder merging is NOT the path. The correct approach:

1. **Cache template layers L0-12** (2.6x speedup, compute only 8 entity-dependent layers)
2. **Specialize single-query attention** (potential KV cache attend optimization)
3. **Fused layer kernel** (single dispatch per layer — reduces 10+ dispatches to 1)

Option 1 alone achieves 149 tok/s, exceeding Ollama's 97 tok/s.
