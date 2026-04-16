# Weight Loading Pipeline

## Overview

`larql-models` loads model weights from safetensors and GGUF formats into a canonical `ModelWeights` struct. All format-specific concerns (dtype conversion, prefix stripping, GGUF dequantization, HuggingFace cache resolution) are handled here.

## Entry Points

```
load_model_dir(path)     → auto-detect format, load ModelWeights
  ├── safetensors/       → safetensors::load_model_dir
  ├── *.gguf             → gguf::load_gguf
  └── error              → ModelError::NotADirectory

resolve_model_path(name) → resolve HF cache path to model directory
```

## Safetensors Pipeline

### 1. Resolve Path

```
Input: "google/gemma-3-4b" or "/path/to/model"
  ↓
Check if directory exists directly
  ↓ (if not)
Search ~/.cache/huggingface/hub/models--{org}--{name}/snapshots/
  ↓
Return resolved Path
```

### 2. Detect Architecture

```
Read config.json → serde_json::Value
  ↓
parse_model_config() → ModelConfig
  ↓
Match model_type → Box<dyn ModelArchitecture>
```

Config parsing handles:
- Top-level config (Llama, Qwen, etc.)
- Nested `text_config` (multimodal Gemma 3/4)
- Fallback defaults per model family

### 3. Load Tensors

```
Glob *.safetensors files (sorted for deterministic order)
  ↓
For each shard:
  mmap the file → &[u8]
  Parse safetensors header (JSON index)
  ↓
  For each tensor:
    Strip key prefix (e.g., "model." → "")
    Read raw bytes from mmap region
    Convert dtype:
      f32 → use directly
      f16 → quant::half::decode_f16
      bf16 → quant::half::decode_bf16
      other → ModelError::UnsupportedDtype
    ↓
    Reshape to Array2<f32> (2D: [rows, cols])
    Convert to ArcArray2<f32> (shared ownership)
    Insert into HashMap<String, WeightArray>
```

### 4. Extract Special Tensors

```
embed = tensors.remove("embed_tokens.weight")
  ↓ (if missing)
embed = tensors.remove(arch.embed_key())

lm_head = tensors.remove("lm_head.weight")
  ↓ (if missing, tie_word_embeddings)
lm_head = embed.clone()

1D tensors → vectors HashMap (norm weights, biases)
2D tensors → tensors HashMap (projections)
```

### 5. Prefix Stripping

Each architecture specifies prefixes to strip via `key_prefixes_to_strip()`:

| Architecture | Prefixes | Example |
|-------------|----------|---------|
| Llama/Qwen/etc. | `["model."]` | `model.layers.0.` → `layers.0.` |
| Gemma 3 | `["language_model.model.", "model."]` | multimodal wrapper |
| Gemma 4 | `["model.language_model.model.", "model.language_model.", ...]` | deeper nesting |

Stripping is tried in order; first match wins.

## GGUF Pipeline

### 1. Parse Header

```
Read magic (0x46554747 = "GGUF")
Read version (3)
Read tensor_count, metadata_count
  ↓
Parse metadata key-value pairs:
  general.architecture → model_type
  *.block_count → num_layers
  *.embedding_length → hidden_size
  *.feed_forward_length → intermediate_size
  ... (all config fields)
```

### 2. Build Config

GGUF metadata keys map to config.json fields:

| GGUF key | ModelConfig field |
|----------|-----------------|
| `{arch}.block_count` | `num_layers` |
| `{arch}.embedding_length` | `hidden_size` |
| `{arch}.feed_forward_length` | `intermediate_size` |
| `{arch}.attention.head_count` | `num_q_heads` |
| `{arch}.attention.head_count_kv` | `num_kv_heads` |
| `{arch}.rope.freq_base` | `rope_base` |

### 3. Load Tensors

```
For each tensor descriptor:
  Read name, shape, dtype, offset
  Seek to data offset
  ↓
  Match dtype:
    F32 → read directly
    F16 → quant::half::decode_f16
    BF16 → quant::half::decode_bf16
    Q4_0 → quant::ggml::dequantize (block decode)
    Q4_1 → quant::ggml::dequantize
    Q8_0 → quant::ggml::dequantize
    other → ModelError::UnsupportedDtype
  ↓
  Strip GGUF key prefix ("blk.N." → "layers.N.")
  Reshape + insert into tensors
```

### 4. Key Translation

GGUF uses different key patterns than safetensors:

| GGUF key | Safetensors equivalent |
|----------|----------------------|
| `blk.0.attn_q.weight` | `layers.0.self_attn.q_proj.weight` |
| `blk.0.ffn_gate.weight` | `layers.0.mlp.gate_proj.weight` |
| `token_embd.weight` | `embed_tokens.weight` |
| `output_norm.weight` | `norm.weight` |

## ModelWeights Struct

```rust
pub struct ModelWeights {
    pub tensors: HashMap<String, WeightArray>,  // 2D weight matrices
    pub vectors: HashMap<String, Vec<f32>>,     // 1D vectors (norms, biases)
    pub embed: WeightArray,                      // Embedding matrix
    pub lm_head: WeightArray,                    // Output projection
    pub arch: Box<dyn ModelArchitecture>,         // Detected architecture
    // Cached config values for hot-path access:
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub rope_base: f64,
}
```

### drop_ffn_weights

Removes FFN tensors from memory for walk-only mode. Matches patterns:
- `gate_proj`, `up_proj`, `down_proj` (dense models)
- `ffn_gate`, `ffn_up`, `ffn_down` (GGUF key format)
- `mlp.experts`, `block_sparse_moe.experts` (MoE per-expert)
- `packed_gate_up_blocks`, `packed_down_blocks` (GPT-OSS MXFP4)

Typical savings: ~13GB for a 4B model (~80% of total weights are FFN).
