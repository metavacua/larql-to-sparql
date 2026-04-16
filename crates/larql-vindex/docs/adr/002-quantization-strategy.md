# ADR-002: Ollama-Compatible Quantization Strategy

**Status**: Accepted  
**Date**: 2026-04  
**Context**: Need to match Ollama's precision/performance for fair comparison.

## Decision

Match Ollama's Q4_K_M quantization strategy:

| Component | Format | Block Size | Bytes/256 vals | Origin |
|-----------|--------|------------|----------------|--------|
| Attention Q/K/O | Q4_K | 256 | 148 | GGUF standard |
| Attention V | Q6_K | 256 | 210 | GGUF standard |
| FFN gate/up | Q4_0 | 32 | 18 | GGUF standard |
| FFN down | Q4_0 | 32 | 18 | GGUF standard |

## Storage Architecture

Vindex stores raw quantized bytes. Compute kernels handle dequantization at inference time.

```
Vindex (storage):
  attn_weights_q4k.bin  → raw Q4_K/Q6_K bytes
  interleaved_q4.bin    → raw Q4_0 bytes (gate|up|down packed)
  manifest.json         → per-layer format tags ("Q4_K", "Q6_K")

Compute (inference):
  q4k_qkv_proj shader  → reads Q4_K bytes, dequants, dot product
  q4_matvec_v4 shader  → reads Q4_0 bytes, integer inner loop
```

## Our Q4_K vs GGUF Q4_K

| Field | Our Layout (148B) | GGUF Layout (144B) |
|-------|-------------------|-------------------|
| d, dmin | 2+2 bytes (ushort f16) | 2+2 bytes (half) |
| Scales | 12 bytes (8×6-bit) | 12 bytes (scales+mins packed) |
| Mins | 4 bytes (8×4-bit) | (packed into scales) |
| Nibbles | 128 bytes | 128 bytes |

Our format separates scales and mins for simpler code. GGUF packs both into 12 bytes for 4 fewer bytes per block. Both produce equivalent results. `quantize_q4_k_gguf()` in larql-compute can produce the GGUF format.

## Consequences

- Fair Ollama comparison (same quantization, same precision)
- Vindex is format-agnostic (stores bytes, compute interprets)
- Three parallel storage paths: f32 (fallback), Q8 (high precision), Q4_K/Q6_K (production)
