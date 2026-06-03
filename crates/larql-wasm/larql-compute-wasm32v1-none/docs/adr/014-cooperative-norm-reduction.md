# ADR-014: Cooperative SIMD Norm Reduction

**Status**: Accepted  
**Date**: 2026-04-09  
**Context**: After closing the matmul gap to Ollama, profiling showed 10.1ms of the 18.3ms decode was in "KV attend + norms + residual" — 55% of total time. The dispatch floor test showed 340 trivial dispatches take only 0.7ms, so 9.4ms was actual kernel execution in element-wise norms.

## Root Cause

All norm kernels (rms_norm, rms_norm_q8, residual_norm, residual_norm_q8) computed sum_sq by having **every thread read ALL elements**:

```metal
// OLD: O(N^2) reads — each of 2560 threads reads all 2560 elements
float sum_sq = 0.0f;
for (uint i = 0; i < len; i++) {
    sum_sq += x[i] * x[i];
}
```

For hidden=2560: 2560 threads x 2560 reads = **6.5M reads per norm dispatch**.
With 2-3 norms per layer x 34 layers = **~660M redundant reads per decode step**.

## Decision

Replace all per-thread full-vector reads with cooperative SIMD reduction:

```metal
// NEW: O(N) reads — each thread sums a stripe, then SIMD + TG reduce
float partial = 0.0f;
for (uint i = tid; i < len; i += tg_sz) {
    partial += x[i] * x[i];
}
float sg_sum = simd_sum(partial);
threadgroup float tg_p[8];
if (lane == 0) tg_p[sg_id] = sg_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
float sum_sq = tg_p[0];
for (uint i = 1; i < n_sg; i++) sum_sq += tg_p[i];
```

Total reads: 2560 (each element read once) + 1 threadgroup barrier.

## Measured Impact

```
Before: 18.3ms / 55 tok/s (34 layers)
After:   8.5ms / 117 tok/s (34 layers)
Savings: ~10ms (the single biggest optimization in the entire session)
vs Ollama: 1.79x → 0.83x (from 79% slower to 17% FASTER)
```

## Affected Kernels

| Kernel | File | What changed |
|--------|------|-------------|
| `rms_norm` | residual_inject.rs | stripe + simd_sum + tg_reduce |
| `rms_norm_q8` | fused_ops.rs | same + simd_max for Q8 block max |
| `residual_norm` | fused_ops.rs | same (residual add + norm) |
| `residual_norm_q8` | fused_ops.rs | same (residual add + norm + Q8) |

## Consequences

- **Good**: 3.4x speedup, exceeds Ollama without caching
- **Good**: All norm variants now consistent in reduction strategy
- **Trade-off**: One threadgroup barrier per norm (was zero, but each thread did 2560x more work)
- **Lesson**: O(N^2) patterns in "small" kernels can dominate total runtime at scale. The norm looked trivial in isolation (~0.15ms per dispatch in old profiling) but that was the *redundant* cost — 2560 threads each doing 2560 reads masked the true O(N^2) nature.
