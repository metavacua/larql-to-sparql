# ADR-011: Per-Layer Architecture Parameters on FullPipelineLayer

**Status**: Accepted  
**Date**: 2026-04-07  
**Context**: The compute pipeline used global function arguments for head_dim, num_q_heads, num_kv_heads, rope_base, and attention scale. This made it impossible to support models with per-layer variation like Gemma 4 (different head_dim for sliding vs global layers) or Gemma 3 (different rope_base per layer).

## Decision

Move all architecture-dependent parameters from global function arguments to per-layer fields on `FullPipelineLayer`. The pipeline reads these per-layer in its inner loop.

### Fields added to FullPipelineLayer (18 new)

| Field | Type | Purpose |
|-------|------|---------|
| `eps` | f32 | Norm epsilon (was hardcoded 1e-6) |
| `attn_scale` | f32 | Attention scale (was 1/sqrt(head_dim)) |
| `head_dim` | usize | Head dimension (was global arg) |
| `num_q_heads` | usize | Q heads (was global arg) |
| `num_kv_heads` | usize | KV heads (was global arg) |
| `rope_base` | f32 | RoPE theta (was global arg) |
| `rotary_dim` | usize | Partial RoPE dims (new) |
| `sliding_window` | usize | Window size (new) |
| `has_v_norm` | bool | V-norm flag (new) |
| `layer_scalar` | f32 | Per-layer scalar (new) |
| `norm_type` | NormType | RMSNorm vs LayerNorm (new) |
| `ffn_type` | FfnType | Gated vs Standard (new) |
| `activation` | Activation | SiLU vs GeluTanh (replaces use_gelu_tanh) |
| `qk_norm_offset` | f32 | QK norm weight offset (new) |
| `input_norm_bias` | Option | LayerNorm bias (new) |
| `post_attn_norm_bias` | Option | LayerNorm bias (new) |
| `ffn_up_bias` | Option | FFN bias (new) |
| `ffn_down_bias` | Option | FFN bias (new) |

### New enums

- `NormType { RmsNorm, LayerNorm }` — replaces implicit RMSNorm-only
- `FfnType { Gated, Standard }` — replaces implicit gated-only
- `Activation { Silu, GeluTanh }` — replaces `use_gelu_tanh: bool`

### New shaders (7 kernels)

- `silu` / `gelu_tanh` — standalone activations for non-gated FFN
- `layer_norm` / `layer_norm_no_bias` — standard LayerNorm
- `v_norm` — parameter-free RMSNorm on V states
- `scale_vector` — per-layer scalar multiplier
- `rope_at_pos` modified to accept `rotary_dim` parameter

## Consequences

- **Good**: Gemma 4 can now run correctly (per-layer head_dim, rope_base, attention scale, V-norm, layer scalar, partial RoPE).
- **Good**: StarCoder2 can now run (LayerNorm, non-gated FFN, bias).
- **Good**: No hardcoded model assumptions remain in the compute path.
- **Good**: Backward compatible — existing code sets defaults matching previous behavior.
- **Trade-off**: `FullPipelineLayer` struct is larger (18 new fields). Accepted because these are all scalar/pointer fields with negligible memory impact, and the alternative (per-model branching code) is worse.
- **Trade-off**: Global function args on `decode_token`/`prefill_q4` still exist but are now unused for the per-layer values. Kept for buffer sizing and API compatibility.

## Structural change

`FullPipelineLayer` and all pipeline types extracted from `lib.rs` into `pipeline.rs` for modularity. `lib.rs` re-exports from `pipeline.rs`.
