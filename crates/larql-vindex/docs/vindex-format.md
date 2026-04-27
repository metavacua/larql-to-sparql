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
├── interleaved_q4k_manifest.json  Per-tensor offsets for interleaved_q4k.bin
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

`attn_weights_q4k_manifest.json` — flat list of 4 entries per layer
(Q, K, V, O in that order), layer-major. V carries `Q6_K`, the rest
`Q4_K`. The `key` matches the original safetensors tensor name.

```json
[
  {
    "key": "model.layers.0.self_attn.q_proj.weight",
    "shape": [3584, 3584],
    "format": "Q4_K",
    "offset": 0,
    "length": 3788800
  },
  {
    "key": "model.layers.0.self_attn.k_proj.weight",
    "shape": [1792, 3584],
    "format": "Q4_K",
    "offset": 3788800,
    "length": 1894400
  },
  {
    "key": "model.layers.0.self_attn.v_proj.weight",
    "shape": [1792, 3584],
    "format": "Q6_K",
    "offset": 5683200,
    "length": 2520000
  },
  {
    "key": "model.layers.0.self_attn.o_proj.weight",
    "shape": [3584, 3584],
    "format": "Q4_K",
    "offset": 8203200,
    "length": 3788800
  }
]
```

**V-shares-K fallback** (Gemma 4 31B global layers). When the source
has no `v_proj` AND `arch.v_shares_k(layer)` returns true, the writer
falls back to K's bytes and stores them in the V slot — still tagged
`Q6_K`, still with `key` = the V tensor name, so downstream 4-per-layer
indexing stays valid.

## Q4_K Interleaved (FFN) Manifest

`interleaved_q4k_manifest.json` — symmetric to the attention manifest.
3 entries per layer (gate, up, down) in that order, layer-major. Down
carries `Q6_K`, gate and up carry `Q4_K`.

```json
[
  {
    "key": "model.layers.0.mlp.gate_proj.weight",
    "shape": [14336, 3584],
    "format": "Q4_K",
    "offset": 0,
    "length": 29692928
  },
  {
    "key": "model.layers.0.mlp.up_proj.weight",
    "shape": [14336, 3584],
    "format": "Q4_K",
    "offset": 29692928,
    "length": 29692928
  },
  {
    "key": "model.layers.0.mlp.down_proj.weight",
    "shape": [3584, 14336],
    "format": "Q6_K",
    "offset": 59385856,
    "length": 42164480
  }
]
```

Padding: each tensor is zero-padded to the next multiple of 256 f32
elements before quantisation (Q4_K/Q6_K super-blocks require
`len % 256 == 0`). Readers must multiply their expected element count
by the block overhead to compute raw byte sizes.

## Interleaved Layout

Gate, up, and down weights packed contiguously per layer to reduce TLB thrashing:

```
Layer 0: [gate_vectors][up_vectors][down_vectors]
Layer 1: [gate_vectors][up_vectors][down_vectors]
...
```

Q4_0 interleaved: 18 bytes per 32 values, 3 matrices per layer.
Q4_K interleaved: 148 bytes per 256 values, with Q6_K for down.

## index.json `quant` field

`VindexConfig.quant` tags the weight storage format so loaders can
dispatch without sniffing filenames:

| `quant` | Weight files | Manifest |
|---------|---|---|
| `"none"` | `attn_weights.bin`, `interleaved.bin` (optional) | `weight_manifest.json` (per-tensor offsets) |
| `"q4k"` | `attn_weights_q4k.bin`, `interleaved_q4k.bin` | `attn_weights_q4k_manifest.json` + `interleaved_q4k_manifest.json` |

Writers set this field alongside `has_model_weights = true`; cold
loaders should branch on `quant` before opening any `.bin` file.
