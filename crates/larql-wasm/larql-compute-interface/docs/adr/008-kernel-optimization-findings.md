# ADR-008: Q4_K Kernel Optimization Findings

**Status**: Accepted (findings recorded)  
**Date**: 2026-04-07  
**Context**: Extensive kernel optimization to close the gap with llama.cpp's Q4_K Metal kernel.

## Approaches Tested

| # | Approach | Result | Origin | Finding |
|---|----------|--------|--------|---------|
| 1 | Fused Q4_K QKV | 1.78x vs Q8 | LARQL original | Fusing Q+K+V eliminates dispatch overhead |
| 2 | Half-precision inner loop | No improvement | LARQL experiment | GPU is not ALU-throughput-bound at this size |
| 3 | Integer Q8 inner loop | No improvement | Inspired by Q4_0 v4 | Q8 quantize overhead = integer multiply savings |
| 4 | Pre-baked half scales (Q4_KF) | No improvement | LARQL experiment | Scale decode is <10% of per-sub-block ALU |
| 5 | Sub-block lane assignment | +3% | LARQL original | 80 subs / 32 lanes = better utilization |
| 6 | simd_shuffle input broadcast | Battery only | Inspired by llama.cpp | Plugged in: parallel lanes > register broadcast |
| 7 | Struct-aligned reads | Marginal | LARQL experiment | Compiler already coalesces byte reads |
| 8 | Pre-loaded 128B register array | Slower | LARQL experiment | 32 × uint32 causes register spilling |
| 9 | 2 sub-blocks per lane (ILP) | Marginal | Inspired by llama.cpp's 2-block technique | Compiler handles ILP adequately |
| 10 | Direct device reads (no TG memory) | Same speed | Inspired by llama.cpp | L2 cache handles both patterns equally |
| 11 | llama.cpp exact kernel port | Same speed | llama.cpp (MIT) | Same inner loop = same speed on same hardware |
| 12 | GGUF 144-byte format | Same speed | GGUF spec | 4-byte layout difference doesn't affect perf |

## Key Discovery

**All kernel variants converge to 0.50ms/layer for Q4_K QKV projection.** The raw kernel (no pipeline) achieves 0.044ms/layer — **6.9x faster than Ollama's entire layer.**

The bottleneck is NOT the kernel. It's the **per-layer dispatch overhead**: 7 compute encoder creations per layer, each adding ~0.15ms of GPU idle time for small operations (rms_norm, residual_add).

## Component Profiling (34 layers, isolated)

| Component | ms/layer | % of total |
|-----------|----------|------------|
| FFN (gate+up+geglu+down) | 0.379 | 36% |
| KV cache (append+attend) | 0.308 | 29% |
| Norms (2× rms_norm) | 0.309 | 29% |
| QKV fused | 0.037 | 3% |
| O projection | 0.024 | 2% |
| Residual add | 0.010 | 1% |

## Recommendation

Merge all per-layer operations into a single compute encoder per layer (or fewer). This eliminates ~204 unnecessary encoder creations and their associated GPU idle time.
