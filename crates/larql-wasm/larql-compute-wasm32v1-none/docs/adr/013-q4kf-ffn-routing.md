# ADR-013: Q4_KF FFN Routing — llama.cpp Kernel for FFN

**Status**: Accepted  
**Date**: 2026-04-08  
**Context**: FFN was the dominant cost (~36% of decode time). The FFN path used Q4_0 kernels with Q8-quantized input, requiring an extra quantization step and using a less optimized inner loop.

## Decision

Route FFN (gate, up, down projections) through format-specific kernels based on the weight format:

1. **Q4_KF (GGUF)**: Use `q4kf_proj` — the llama.cpp-exact kernel with register-cached input, uint16 nibble masking, native half reads, multi-row (nr0=2)
2. **Q4_K**: Use `q4k_ffn_gate_up` (fused) + `q4k_matvec` (down) — uint4 vectorized loads, sub-block striping, nr0=2
3. **Q4_0**: Keep legacy path — residual_norm_q8 + Q4_0 matvec with Q8 input

The Q4_K and Q4_KF paths skip Q8 quantization entirely, using `residual_norm` (f32 output) instead of `residual_norm_q8`. A separate `residual_add` computes `h_post_attn = h + o` for the post-FFN residual.

## Implementation

In `decode.rs`, the FFN section checks `layer.gate.format`:

```
if Q4_KF → q4kf_proj pipeline (gate, up, down)
else if Q4_K/Q6_K → q4k_ffn_gate_up (gate+up fused) + q4k_matvec (down)
else → Q4_0 legacy path
```

## Measured Impact

```
Before (Q4_0 FFN): 29.2ms / 34 tok/s (34 layers) = 2.84x Ollama
After (Q4_K FFN):  24.7ms / 40 tok/s              = 2.37x Ollama
After (Q4_KF FFN): ~12.9ms / ~77 tok/s            = ~1.25x Ollama
```

The Q4_KF kernel's register-cached input pattern (yl[16]/yh[16] arrays loaded once, reused across rows) and uint16 nibble masking (no bit shifts for lower nibble, `1.f/256.f` scaling for upper) account for the majority of the improvement.

## Final Numbers (after all optimizations including ADR-014 norm fix)

```
Before (Q4_0 FFN): 29.2ms / 34 tok/s (34L) = 2.84x Ollama
After (Q4_KF FFN):  8.5ms / 117 tok/s (34L) = 0.83x Ollama (17% faster)
```

Also added `q4kf_ffn_gate_up` kernel (2026-04-09): fused gate+up for Q4_KF format
with llama.cpp inner loop, eliminating one dispatch per layer.

## Consequences

- **Good**: Ollama exceeded at 34 layers without caching
- **Good**: Format detection is automatic via FullPipelineLayer.gate.format
- **Good**: Legacy Q4_0 path preserved as fallback
- **Trade-off**: Q4_KF weights are in GGUF 144-byte format (not our 148-byte format), so the vindex loader must emit GGUF-compatible blocks
