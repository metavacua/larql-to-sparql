# ADR-004: predict_honest — Production Inference Pipeline

**Status**: Accepted  
**Date**: 2026-04  
**Context**: Need a single inference entry point that combines cached layers, GPU decode, vindex logits, and per-layer architecture parameterization.

## Decision

`predict_honest()` is the production inference path. It:

1. Runs cached layers (L0-12) via `CachedLayerGraph` (~5ms)
2. Builds `FullPipelineLayer` structs from vindex weights with per-layer arch params
3. Dispatches to `backend.decode_token()` (GPU Metal) or `backend.full_pipeline_q4()` (fallback)
4. Finalizes via `finalize_logits()` — vindex lm_head KNN instead of dense matmul

### Per-Layer Architecture Parameterization

Each `FullPipelineLayer` carries 18+ fields from `ModelArchitecture`:
- `head_dim`, `num_q_heads`, `num_kv_heads` — per-layer (Gemma 4 variable)
- `rope_base` — per-layer (Gemma 3/4 dual bases)
- `attn_scale` — per-layer (Gemma 4: 1.0 with QK-norm)
- `rotary_dim` — per-layer (Gemma 4: 25% on global layers)
- `activation`, `norm_type`, `ffn_type` — per-layer
- `eps` — from `arch.norm_eps()`

No model-type strings or hardcoded constants in the pipeline.

## Consequences

- **Good**: Single entry point for all model architectures (Gemma 2/3/4, Llama, StarCoder2, etc.)
- **Good**: GPU and CPU paths produce identical results
- **Good**: Vindex logits replace 27ms dense matmul with ~1ms KNN
- **Trade-off**: Large function (~200 lines). Accepted because it encapsulates the full pipeline with no external state.
