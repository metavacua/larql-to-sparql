# Compute Integration — larql-vindex × larql-compute

How vindex stores data and compute consumes it.

## Principle

**Vindex stores raw quantized bytes. Compute dequants + multiplies at inference time.**

Vindex never interprets quantized data — it provides byte slices and format tags. The compute backend handles all format-specific logic (Q4_K dequantization, nibble extraction, scale unpacking).

## Data Flow

```
Build time (larql-vindex):
  safetensors → f32 extraction → quantize → store in vindex
                                    ↓
                              Q4_K / Q6_K / Q4_0 / Q8_0 raw bytes

Inference time (larql-compute reads from vindex):
  vindex.attn_q4k_layer_data(layer) → [(&[u8], "Q4_K"), (&[u8], "Q4_K"), (&[u8], "Q6_K"), (&[u8], "Q4_K")]
                                          Q proj       K proj        V proj        O proj
                                            ↓
  backend.full_pipeline_q4(layers, ...) → Metal Q4_K shader → f32 output
```

## API Surface

### Vindex → Compute (data providers)

| Vindex Method | Returns | Used By |
|---------------|---------|---------|
| `gate_vectors(layer)` | `&[f32]` view from mmap | `backend.matmul_transb()` for KNN |
| `gate_q4_data(layer)` | `&[u8]` Q4_0 bytes | `backend.q4_matvec()` for Q4 KNN |
| `attn_q4k_layer_data(layer)` | `[(&[u8], &str); 4]` | `FullPipelineLayer` construction |
| `attn_q8_layer_data(layer)` | `[(&[u8], &[f32]); 4]` | `FullPipelineLayer` construction |
| `interleaved_q4k_mmap_ref()` | `&[u8]` entire mmap | FFN Q4_K/Q6_K weight slicing (preferred) |
| `interleaved_q4_mmap_ref()` | `&[u8]` entire mmap | FFN Q4_0 weight slicing (fallback) |
| `lm_head_q4_data()` | `&[u8]` Q4_0 bytes | `backend.q4_matvec()` for logits |
| `down_layer_matrix(layer)` | `ArrayView2<f32>` | Walk FFN, zero-copy |
| `up_layer_matrix(layer)` | `ArrayView2<f32>` | Walk FFN, zero-copy |

### Compute → Vindex (format contracts)

| Compute Shader | Expects From Vindex | Block Size |
|----------------|-------------------|------------|
| `q4k_qkv_proj` | Q4_K bytes (148B blocks) | 256 values |
| `q6k_matvec` | Q6_K bytes (210B blocks) | 256 values |
| `q4_matvec_v4` | Q4_0 bytes (18B blocks) | 32 values |
| `q8_qkv_proj` | Q8_0 int8 + f32 scales | 32 values |
| `fused_attention` | f32 Q/K/V (from projection output) | — |

## FullPipelineLayer Construction

The inference crate assembles `FullPipelineLayer` from vindex data:

```rust
// In predict_honest (larql-inference):
let [q, k, v, o] = index.attn_q4k_layer_data(layer).unwrap();
let layer = FullPipelineLayer {
    wq: QuantWeight { data: q.0, scales: None, format: to_format(q.1) },
    wk: QuantWeight { data: k.0, scales: None, format: to_format(k.1) },
    // ... format tag drives kernel selection in decode_token

    // Per-layer architecture params (from ModelArchitecture trait):
    eps: arch.norm_eps(),
    attn_scale: arch.attention_scale_for_layer(layer) as f32,
    head_dim: arch.head_dim_for_layer(layer),
    num_q_heads: arch.num_q_heads_for_layer(layer),
    num_kv_heads: arch.num_kv_heads_for_layer(layer),
    rope_base: arch.rope_base_for_layer(layer) as f32,
    rotary_dim: (arch.rotary_fraction_for_layer(layer) * arch.head_dim_for_layer(layer) as f64) as usize,
    sliding_window: arch.sliding_window_size().unwrap_or(0),
    has_v_norm: arch.has_v_norm(),
    layer_scalar: 0.0,  // loaded from weight if arch.layer_scalar_key(layer) exists
    norm_type: NormType::RmsNorm,  // or LayerNorm for StarCoder2
    ffn_type: FfnType::Gated,     // or Standard for StarCoder2
    activation: Activation::Silu,  // or GeluTanh for Gemma
    // ...
};
```

The `QuantFormat` tag (Q4_K, Q6_K, Q8_0, Q4_0) determines which Metal shader the compute backend dispatches.
All per-layer architecture params (head_dim, rope_base, attention scale, etc.) come from the `ModelArchitecture` trait — the compute backend reads these per-layer in its inner loop.

## Format Decision Flow

```
Vindex build: user chooses quantization level
  → build_q4k_weights: creates attn_weights_q4k.bin
  → build_attn_q8: creates attn_weights_q8.bin

Inference: auto-selects best available (attention)
  if has_q4k → Q4_K path (q4kf_qkv_proj shader, skip Q8 quantize)
  elif has_q8 → Q8 path (q8_qkv_proj shader, fused Q8 input)
  else → f32 path (CPU BLAS matmul_transb)

Inference: auto-selects best available (FFN)
  if has_interleaved_q4k → Q4_K FFN (QuantFormat::Q4_K on FullPipelineLayer)
  elif has_interleaved_q4 → Q4_0 FFN (QuantFormat::Q4_0, fallback)
  else → CPU walk FFN (f32 sparse)
```
