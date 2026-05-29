# ADR-002: Q4_0 Matvec v4 as Production Kernel

**Status**: Accepted  
**Date**: 2026-04-05  
**Context**: 5 kernel variants (v1-v5) were developed iteratively. Need to pick one for production.

## Decision

v4 (`q4_matvec_v4`) is the production kernel for Q4_0 FFN operations.

## Benchmark (M3 Max, [10240, 2560] = 14.7MB Q4_0)

| Variant | Time | Bandwidth | Technique |
|---------|------|-----------|-----------|
| v1 | 0.48ms | 31 GB/s | Simdgroup + threadgroup shared memory |
| v2 | 0.36ms | 41 GB/s | 4 rows per thread, f32 input |
| v3 | 0.66ms | 22 GB/s | 8 rows unrolled (register spilling) |
| **v4** | **0.26ms** | **57 GB/s** | **uint32 wide loads + simdgroup** |
| v5 | 0.26ms | 57 GB/s | 256 rows/TG, no simd (same speed) |

## Key Techniques in v4

1. Load Q4 nibbles as 4 × uint32 (16 bytes per load, 8 nibbles per word)
2. Extract nibbles via bitshift: `int((w >> shift) & 0xFu) - 8`
3. Q8 input in threadgroup shared memory (1 byte/element, 4x less than f32)
4. Integer multiply-accumulate: `isum += nibble * q8_val` (no float until final scale)
5. simd_sum for cross-lane reduction

## Consequences

- v1-v3, v5 kept in shader inventory for benchmarking comparisons
- All variants compiled in `all_shaders()` but only v4 dispatched in production
- v4 technique (uint32 wide loads + integer arithmetic) informed the Q4_K kernel design
