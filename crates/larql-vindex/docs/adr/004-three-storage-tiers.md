# ADR-004: Three-Tier Weight Storage (f32, Q8, Q4_K)

**Status**: Accepted  
**Date**: 2026-04  
**Context**: Different use cases need different precision/size trade-offs.

## Decision

Support three parallel attention weight storage formats:

| Tier | Format | Size/Layer | Use Case |
|------|--------|-----------|----------|
| **f32** | `attn_weights.bin` | ~50MB | CPU BLAS fallback, extraction |
| **Q8** | `attn_weights_q8.bin` | ~13MB | High precision GPU |
| **Q4_K** | `attn_weights_q4k.bin` | ~7.6MB | Production (Ollama-compatible) |

## Loading Priority

```
predict_honest() in larql-inference:
  1. Check for Q4_K attention → use q4k_qkv_proj shader (fastest)
  2. Fall back to Q8 → use q8_qkv_proj shader
  3. Fall back to f32 → use CPU BLAS
```

Each format has its own manifest JSON tracking per-layer offsets and format tags.

## Build Pipeline

Two paths depending on precision target:

**Multi-tier (f32 → Q8 / Q4_K)** — builds every tier and lets the
inference path pick the best available:

```
safetensors (original) → f32 extraction → Q8 quantize → Q4_K quantize
                                           ↓               ↓
                                    attn_weights_q8.bin  attn_weights_q4k.bin
```

Build tools: `build_attn_q8`, `build_attn_q4`, `build_q4k_weights`.

**Streaming Q4_K (new)** — single-pass extraction that skips the f32
intermediate and quantises straight from bf16/f16 safetensors:

```
safetensors (original) → quantise-in-stream → attn_weights_q4k.bin +
                                              interleaved_q4k.bin +
                                              manifests
```

Invoke via `larql extract-index --quant q4k`. Materialises all of
attention, FFN, norms, and `lm_head` in one pass; implies
`--level all`.

## Dispatch

`VindexConfig.quant: QuantFormat` is written to `index.json` at build
time (`none` for float vindexes, `q4k` for the streaming Q4_K path).
Loaders branch on this field — `load_model_weights` errors cleanly
when called on a Q4_K vindex and points the caller at
`VectorIndex::load_attn_q4k` / `load_interleaved_q4k`.

## Consequences

- User controls precision/size trade-off at build time
- Inference auto-selects best available format (multi-tier path)
- Streaming Q4_K avoids multi-GB f32 intermediate on disk
- All formats share the same vindex directory
- Manifest JSON enables format mixing (Q4_K for Q/K/O, Q6_K for V)
- `config.quant` dispatch avoids silent "file not found" errors on
  cross-path loader calls
