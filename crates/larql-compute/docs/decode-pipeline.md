# Decode Pipeline — larql-compute

How `decode_token` processes one token through all layers with KV cache.

## Overview

```
Input: x[hidden] (embedded token)
Output: h[hidden] (final hidden state for logit projection)

Per layer (single encoder, ~10 dispatches):
  1. Input norm
  2. Fused QKV projection (Q4_K or Q4_KF)
  3. Batched RoPE (all Q heads + all K heads = 2 dispatches)
  4. Batched V-norm (optional, Gemma 4)
  5. KV cache append + attend (SIMD reductions)
  6. O projection
  7. Residual + norm (f32 for Q4_K/Q4_KF, +Q8 for Q4_0)
  8. FFN: fused gate+up (or separate) + GEGLU + down
  9. Post-FFN residual + optional layer scalar
```

## Dual-Path Architecture

Weights are either Q4_K (Ollama strategy, smaller) or Q8_0 (higher precision).
`decode_token` auto-detects from `FullPipelineLayer.wq.format`.

### Q4_KF Path (fastest — llama.cpp-exact kernel)

```
h_buf [f32]
  → rms_norm → norm_f32 [f32]
  → q4kf_qkv_proj (fused, GGUF format) → Q, K, V [f32]
  → rope_at_pos_batched (Q heads) + rope_at_pos_batched (K heads)
  → v_norm_batched (optional, Gemma 4)
  → kv_cache_append + kv_attention (simd_max/simd_sum)
  → q4kf_proj (O projection)
  → residual_norm → ffn_norm_out [f32], residual_add → h_post_attn [f32]
  → q4kf_proj (gate) + q4kf_proj (up) → geglu → q4kf_proj (down)
  → residual_add → h_buf [f32] for next layer
```

Advantages: llama.cpp-exact inner loop, register-cached input, native half reads, uint16 nibble masking. ~1.25x Ollama.

### Q4_K Path

```
h_buf [f32]
  → rms_norm → norm_f32 [f32]
  → q4k_qkv_proj (fused) → Q, K, V [f32]
  → rope_at_pos_batched + kv_cache_append + kv_attention
  → q4k_proj (O projection)
  → residual_norm → ffn_norm_out [f32], residual_add → h_post_attn [f32]
  → q4k_ffn_gate_up (fused, one dispatch) → geglu → q4k_matvec (down)
  → residual_add → h_buf [f32] for next layer
```

Advantages: Fused gate+up (one dispatch), uint4 loads, 8 rows/TG, multi-row (nr0=2). ~2.0x Ollama.

### Q8 Path

```
h_buf [f32]
  → rms_norm_q8 (fused) → q8_buf [int8], q8s_buf [f32]
  → q8_qkv_proj (fused) → Q, K, V [f32]
  → kv_cache_append → kv_attention → attn_out [f32]
  → quantize_q8 → q8_attn [int8]
  → q8_matvec (O proj) → o_out [f32]
  → residual_norm_q8 (fused) → FFN path (same as Q4_K)
```

Advantages: Higher precision QKV. Established path with integer inner loop.

## Metal Dispatch Structure

Single Metal command buffer for all layers. One encoder per layer, no explicit memory barriers
(Apple Silicon serialises compute dispatches within an encoder).

Current dispatch count per layer: ~10
- Input norm (1)
- Fused QKV projection (1)
- Batched RoPE Q + K (2)
- Batched V-norm (0 or 1)
- KV append + attend (2)
- O projection (1)
- Residual + norm (1)
- FFN: gate+up fused or separate + GEGLU + down (2–3)
- Post-FFN residual (1)

Total for 34 layers: ~340 dispatches in 34 encoders, 1 command buffer, 1 commit+wait.

## KV Cache

```rust
pub struct KVCache {
    pub layers: Vec<LayerKVCache>,
}

pub struct LayerKVCache {
    pub k_cache: Buffer,    // [max_seq, num_kv_heads, head_dim] f32
    pub v_cache: Buffer,    // same
    pub current_len: usize, // tokens cached so far
    pub max_seq: usize,     // capacity (default 4096)
}
```

- Populated during prefill via `populate_kv_layer` (CPU → GPU copy)
- Extended during decode via `kv_cache_append` shader
- `kv_attention` shader attends Q against all cached K/V (positions 0..current_len)

## Prefill Pipeline (seq > 1)

`prefill_q4` in `metal/prefill.rs` handles multi-token prefill on GPU:
- Per-position Q4_K projection dispatch within one command buffer
- Fused attention with skip_rope and rotary_dim flags (partial RoPE for Gemma 4)
- KV cache populated via CPU `prefill_with_kv` after GPU forward pass

## Performance (M3 Max, Gemma 3 4B, 2026-04-09)

| Path | Time | tok/s | vs Ollama |
|------|------|-------|-----------|
| **Q4_KF decode (34L)** | **8.5ms** | **117** | **0.83x (17% faster)** |
| Q4_K decode (21L) | 11.6ms | 86 | 1.13x |
| Q8 decode (21L) | 19.3ms | 52 | — |
| Ollama (34L) | 10.3ms | 98 | 1.0x |

### Component Breakdown (34 layers)

| Component | Time | Per-Layer | % |
|-----------|------|-----------|---|
| FFN (gate+up+GEGLU+down) | 6.1ms | 0.179ms | 33% |
| QKV projection | 1.3ms | 0.037ms | 7% |
| O projection | 0.8ms | 0.024ms | 5% |
| KV attend + norms + residual | 0.5ms | 0.015ms | 3% |

### Key: Cooperative SIMD Norms

All norm kernels (rms_norm, residual_norm, residual_norm_q8) use cooperative SIMD
reduction for sum_sq. Each thread computes a partial sum over a stripe of elements,
then simd_sum + threadgroup reduction produces the global result. This is O(N) reads
vs the previous O(N²) where every thread redundantly read all elements.
