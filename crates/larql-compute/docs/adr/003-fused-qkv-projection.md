# ADR-003: Fused Q+K+V Projection Kernel

**Status**: Accepted  
**Date**: 2026-04-06  
**Context**: Separate Q, K, V dispatches incur 3x encoder overhead. Need single-dispatch QKV.

## Decision

Two fused QKV kernels: `q8_qkv_proj` (Q8 weights) and `q4k_qkv_proj` (Q4_K weights).

Both use the same architecture: rows 0..q_rows → Q output, q_rows..q_rows+k_rows → K output, rest → V output. One threadgroup handles 8 rows across all three projections.

## Origin

- **q8_qkv_proj**: Original LARQL design. Simdgroup Q8×Q8 integer dot product with shared Q8 input in threadgroup memory. 2.2x faster than 3 separate dispatches.
- **q4k_qkv_proj**: Original LARQL design. Same fused pattern for Q4_K data with sub-block lane assignment (80 sub-blocks / 32 lanes = 83% utilization).
- **q4kf_qkv_proj**: Inspired by llama.cpp's `kernel_mul_mv_q4_K_f32` (MIT license). Register-based input loading, quarter-block lane decomposition (ix=lane/8, iq=it/4, ir=it%4), uint16_t nibble extraction with bit masking, `FOR_UNROLL` pragma. Adapted for our fused QKV dispatch pattern and our 148-byte Q4_K block layout (vs GGUF's 144-byte layout).

## Benchmark (M3 Max, QKV = 5120 rows × 2560 hidden, 21 layers)

| Kernel | attn/21L | Speedup vs Q8 | Notes |
|--------|----------|---------------|-------|
| Q8 separate (3 dispatches) | 18.4ms | 1.0x | Baseline |
| Q8 fused (1 dispatch) | 10.2ms | 1.8x | Our design |
| Q4_K fused | 10.3ms | 1.78x | Our design, smaller data |

## Consequences

- Q4_K QKV at 1.2ms for 34 layers = 0.037ms/layer — faster than Ollama's entire layer
- The kernel is NOT the bottleneck (3.4% of total decode time)
- Bottleneck is elsewhere: FFN (36%), KV cache (29%), dispatch overhead (29%)
