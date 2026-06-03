# ADR-012: Encoder Merging + Decode Pipeline Optimization

**Status**: Complete  
**Date**: 2026-04-08 (updated with full session results)  
**Context**: The decode pipeline used 7-12 separate Metal compute command encoders per layer, plus per-layer command buffer commit+wait. Investigated all dispatch overhead reduction approaches.

## Decision

Final architecture: **single command buffer** for all 34 layers, **single encoder per layer**, **no explicit memory barriers** (Apple Silicon serialises compute dispatches within an encoder).

## What Worked (29.2ms → ~12.9ms = 2.3x faster)

| Optimization | Savings | Notes |
|-------------|---------|-------|
| Q4_KF FFN routing (q4kf_proj) | ~8ms | llama.cpp-exact kernel for FFN gate/up/down |
| Q4_K matvec rewrite (uint4, nr0=2) | ~3ms | Vectorized loads, multi-row |
| Q4_K format for FFN (skip Q8) | ~4.5ms | residual_norm instead of residual_norm_q8 |
| Fused gate+up (q4k_ffn_gate_up) | ~1ms | Single dispatch, shared input |
| Batched RoPE + V-norm | ~0.5ms | 16 per-head dispatches → 3 batched |
| SIMD KV attention | ~1ms | simd_max/simd_sum, 3 barriers (was 6) |

## What Didn't Work

| Approach | Result | Why |
|----------|--------|-----|
| Encoder merging (4 → 1) | ~0ms | Metal encoder creation is ~0.0002ms each |
| Single cmd buffer (34 → 1 wait) | ~0ms | wait_until_completed returns instantly when GPU done |
| Memory barriers | ~0ms | Apple Silicon serialises within encoder anyway |
| 2-sub-block unrolling | **Slower** | Register pressure, poor tail at K=2560 (40 pairs / 32 lanes) |
| Fused GEGLU+down | **32x slower** | exp() recomputed per output row (26M vs 10K calls) |

## Lesson

**Metal dispatch overhead is negligible on Apple Silicon.** The real bottleneck was kernel execution speed (Q4_0 vs Q4_K vs Q4_KF format) and the number of memory reads per output element. The llama.cpp-exact inner loop with register-cached input was the unlock.

## Result

```
Before:   29.2ms / 34 tok/s (34 layers) = 2.84x Ollama
Mid:      18.3ms / 55 tok/s (34 layers) = 1.79x Ollama  (kernel opts + buffer prealloc)
Final:     8.5ms / 117 tok/s (34 layers) = 0.83x Ollama  (+ cooperative norm fix, ADR-014)
```

The cooperative norm fix (ADR-014) was the single biggest win: ~10ms saved by fixing
O(N²) reads in all norm kernels. See ADR-014 for details.
