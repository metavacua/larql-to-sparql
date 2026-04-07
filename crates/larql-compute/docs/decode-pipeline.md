# Decode Pipeline — larql-compute

How `decode_token` processes one token through all layers with KV cache.

## Overview

```
Input: x[hidden] (embedded token)
Output: h[hidden] (final hidden state for logit projection)

Per layer (in one Metal command buffer):
  1. Input norm
  2. QKV projection (fused)
  3. KV cache append + attend
  4. O projection
  5. Residual + pre-FFN norm + Q8 quantize (fused)
  6. FFN: gate + up + GEGLU + down + residual (one encoder)
```

## Dual-Path Architecture

Weights are either Q4_K (Ollama strategy, smaller) or Q8_0 (higher precision).
`decode_token` auto-detects from `FullPipelineLayer.wq.format`.

### Q4_K Path

```
h_buf [f32]
  → rms_norm → norm_f32 [f32]
  → q4k_qkv_proj (fused) → Q[q_dim], K[kv_dim], V[kv_dim] [f32]
  → kv_cache_append → cache updated
  → kv_attention → attn_out [f32]
  → q4k_proj → o_out [f32]
  → residual_norm_q8 (fused) → h_post_attn [f32], ffn_q8 [int8], ffn_q8s [f32]
  → q4_matvec (gate, up) → geglu → q4_f32_matvec (down) → residual_add
  → h_buf [f32] for next layer
```

Advantages: No Q8 quantization of input (saves one dispatch). Q4_K data is 1.73x smaller than Q8.

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

One Metal command buffer per `decode_token` call. All layers encoded before `commit()`.

Current encoder count per layer:
- Q4_K path: 5 encoders (norm+QKV merged, KV append, KV attend, O proj, residual+FFN+residual merged)
- Q8 path: 6 encoders (norm+Q8, QKV, KV append, KV attend, O quant+proj, residual+FFN+residual)

Total for 34 layers: 170-204 encoders in one command buffer.

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
- Fused attention with skip_rope flag
- KV cache populated via CPU `prefill_with_kv` after GPU forward pass

## Performance (M3 Max, 21 layers)

| Path | Time | tok/s |
|------|------|-------|
| Q4_K decode | 16.9ms | 59 |
| Q8 decode | 24.3ms | 41 |
| Ollama (34 layers) | 10.3ms | 97 |
