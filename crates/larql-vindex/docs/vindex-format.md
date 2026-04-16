# Vindex File Format Specification

A vindex is a directory containing a transformer model's weights reorganized for queryability. The model IS the database.

## Directory Layout

```
model.vindex/
├── index.json                 Config, layer bands, provenance, checksums
├── tokenizer.json             Tokenizer configuration
│
├── gate_vectors.bin           W_gate per layer (f32 or f16, KNN index)
├── gate_vectors_q4.bin        W_gate Q4_0 quantized (7x smaller)
├── embeddings.bin             W_embed matrix
├── down_meta.bin              Per-feature output metadata (binary, ~5.8KB)
│
├── attn_weights.bin           Q, K, V, O per layer (f32/f16)
├── attn_weights_q8.bin        Q8_0 quantized attention (optional)
├── attn_weights_q4k.bin       Q4_K/Q6_K Ollama-compatible (optional)
├── weight_manifest.json       Weight file offsets
├── attn_weights_q8_manifest.json
├── attn_weights_q4k_manifest.json
│
├── up_weights.bin             W_up per layer (FFN up-projection)
├── down_weights.bin           W_down per layer (FFN down-projection)
├── down_features.bin          Feature-major down vectors (zero-copy slice)
├── up_features.bin            Feature-major up vectors
├── norms.bin                  LayerNorm/RMSNorm parameters
├── lm_head.bin                Output projection
├── lm_head_q4.bin             Q4_0 output projection (optional)
│
├── interleaved.bin            gate|up|down packed per layer (f32, optional)
├── interleaved_q4.bin         Q4_0 quantized interleaved (optional)
├── interleaved_q4k.bin        Q4_K/Q6_K interleaved (optional)
│
├── router_weights.bin         MoE router (optional, for MoE models)
├── relation_clusters.json     Discovered relation types (optional)
└── feature_labels.json        Probe-confirmed labels (optional)
```

## Extract Levels

| Level | Files Loaded | Size (Gemma 4B) | Operations Supported |
|-------|-------------|-----------------|---------------------|
| **Browse** | gate + embed + down_meta | ~3 GB | WALK, DESCRIBE, SELECT |
| **Inference** | + attention weights | ~6 GB | INFER |
| **All** | + up, down, norms, lm_head | ~8.5 GB | COMPILE |

## index.json Schema

```json
{
  "version": 2,
  "model_family": "gemma",
  "model_name": "gemma-3-4b",
  "num_layers": 34,
  "hidden_size": 2560,
  "intermediate_size": 10240,
  "num_features_per_layer": 10240,
  "storage_dtype": "f16",
  "layer_bands": {
    "syntax": [0, 12],
    "knowledge": [13, 27],
    "output": [28, 33]
  },
  "model_config": {
    "model_type": "gemma3",
    "head_dim": 256,
    "num_q_heads": 8,
    "num_kv_heads": 4,
    "rope_base": 1000000.0,
    "sliding_window": 1024,
    "global_head_dim": null,
    "num_global_kv_heads": null,
    "partial_rotary_factor": null,
    "sliding_window_pattern": null,
    "attention_k_eq_v": false,
    "num_kv_shared_layers": null
  },
  "checksums": {
    "gate_vectors.bin": "sha256:...",
    "embeddings.bin": "sha256:..."
  }
}
```

For Gemma 4, the `model_config` includes per-layer geometry:

```json
{
  "model_config": {
    "model_type": "gemma4_text",
    "head_dim": 256,
    "num_q_heads": 16,
    "num_kv_heads": 8,
    "rope_base": 1000000.0,
    "sliding_window": 1024,
    "global_head_dim": 512,
    "num_global_kv_heads": 4,
    "partial_rotary_factor": 0.25,
    "sliding_window_pattern": 6,
    "attention_k_eq_v": true,
    "num_kv_shared_layers": 20,
    "per_layer_embed_dim": 256,
    "rope_local_base": 10000.0
  }
}
```

All Gemma 4 fields are optional — existing vindexes without them load correctly
with defaults (standard behavior for pre-Gemma-4 models).

## Binary down_meta Format

```
Header (16 bytes):
  magic: u32 = 0x444D4554 ("DMET")
  version: u32 = 1
  num_layers: u32
  top_k: u32

Per layer:
  num_features: u32
  Per feature:
    token_id: u32
    c_score: f32
    top_k × (token_id: u32, logit: f32)
```

Total: ~5.8 KB for 100K features with top_k=10 (vs 160 MB JSONL).

## Q4_K Attention Manifest

```json
[
  {
    "layer": 0,
    "q": { "offset": 0, "length": 3788800, "format": "Q4_K" },
    "k": { "offset": 3788800, "length": 1894400, "format": "Q4_K" },
    "v": { "offset": 5683200, "length": 2520000, "format": "Q6_K" },
    "o": { "offset": 8203200, "length": 3788800, "format": "Q4_K" }
  }
]
```

## Interleaved Layout

Gate, up, and down weights packed contiguously per layer to reduce TLB thrashing:

```
Layer 0: [gate_vectors][up_vectors][down_vectors]
Layer 1: [gate_vectors][up_vectors][down_vectors]
...
```

Q4_0 interleaved: 18 bytes per 32 values, 3 matrices per layer.
Q4_K interleaved: 148 bytes per 256 values, with Q6_K for down.
