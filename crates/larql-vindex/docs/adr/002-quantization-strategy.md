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
| FFN gate/up | Q4_K | 256 | 148 | GGUF standard |
| FFN down | Q6_K | 256 | 210 | GGUF standard |

The legacy `interleaved_q4.bin` (Q4_0, 32-value blocks, 18 bytes) path
is kept for backwards compatibility with older vindexes and specific
compute benchmarks, but new extractions default to the Q4_K/Q6_K
layout that matches Ollama's Q4_K_M exactly.

## Storage Architecture

Vindex stores raw quantized bytes. Compute kernels handle dequantization at inference time.

```
Vindex (storage):
  attn_weights_q4k.bin            → raw Q4_K (Q/K/O) + Q6_K (V) bytes
  attn_weights_q4k_manifest.json  → per-tensor {key, shape, format, offset, length}
  interleaved_q4k.bin             → raw Q4_K (gate/up) + Q6_K (down) bytes
  interleaved_q4k_manifest.json   → per-tensor layout for the FFN pack

  interleaved_q4.bin              → legacy Q4_0 bytes (still supported)

Compute (inference):
  q4k_qkv_proj shader   → reads Q4_K bytes, dequants, dot product
  q4k_ffn_* shaders     → reads Q4_K/Q6_K FFN bytes
  q4_matvec_v4 shader   → reads legacy Q4_0 bytes, integer inner loop
```

## Dispatch

`VindexConfig.quant: QuantFormat` (`none` / `q4k`) tags the vindex at
write time; loaders branch on this field rather than sniffing
filenames. The CLI surfaces this as `larql extract-index --quant q4k`,
which runs the streaming extract path that skips the f32 intermediate
entirely — quantisation happens in one pass straight from the
bf16/f16 safetensors shards.

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
