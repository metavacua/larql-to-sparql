# ADR-006: Dual-Path Decode (Q4_K vs Q8 Auto-Detection)

**Status**: Accepted  
**Date**: 2026-04-06  
**Context**: Attention weights can be Q4_K (Ollama strategy) or Q8_0 (higher precision). Different kernels needed.

## Decision

`decode_token` auto-detects weight format from `FullPipelineLayer.wq.format` and selects the optimal pipeline:

**Q4_K path** (when `format == Q4_K | Q6_K | Q4_KF`):
1. rms_norm → f32 output (no Q8 quantize)
2. Fused Q4_K QKV (one dispatch, f32 input)
3. KV cache append+attend
4. Q4_K O projection (f32 input)
5. Fused residual+norm+Q8
6. Q4_0 FFN (gate+up+GEGLU+down+residual, one encoder)

**Q8 path** (when `format == Q8_0`):
1. Fused rms_norm+Q8 quantize
2. Fused Q8 QKV (one dispatch, Q8 input)
3. KV cache append+attend
4. Q8 O projection (Q8 quantize + Q8 matvec)
5. Fused residual+norm+Q8
6. Q4_0 FFN (same as above)

## Origin

Original LARQL design. Q4_K path eliminates the Q8 quantization step, saving one dispatch per layer. Q8 path uses the established fused Q8 QKV kernel.

## Benchmark

| Path | Decode/21L | tok/s | Notes |
|------|-----------|-------|-------|
| Q4_K | 16.9ms | 59 | Skips Q8 quantize, smaller data |
| Q8 | 24.3ms | 41 | Higher precision, larger data |
| Speedup | 1.44x | | |

## Consequences

- Caller doesn't choose path — format field on FullPipelineLayer drives selection
- Both paths share: KV cache, FFN, residuals, norms
- Q4_K path is default when Ollama-format weights available from vindex
