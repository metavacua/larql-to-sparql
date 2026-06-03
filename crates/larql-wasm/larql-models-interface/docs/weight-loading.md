# Weight Loading Pipeline

## Overview

`larql-models` loads model weights from safetensors and GGUF formats into a canonical `ModelWeights` struct. All format-specific concerns (dtype conversion, prefix stripping, GGUF dequantization, HuggingFace cache resolution) are handled here.

## Entry Points

```
load_model_dir(path)                   → auto-detect format, load all tensors
load_model_dir_validated(path)         → validate architecture before loading tensors
load_model_dir_walk_only(path)         → skip FFN tensors at parse/dequant time (no heap spike)
load_model_dir_walk_only_validated(path)
load_model_dir_filtered(path, skip_fn) → skip any tensors matching predicate
load_model_dir_filtered_validated(path, skip_fn)
  ├── *.safetensors/     → loading::safetensors
  ├── *.gguf             → loading::gguf::load_gguf_filtered
  └── error              → ModelError::{NotADirectory, NoSafetensors}

resolve_model_path(name) → resolve HF cache path to model directory
```

Use validated entry points for inference, extraction, and long-running servers.
Use permissive entry points for inspection/conversion tools that need to report
or repair incomplete configs.

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
  ↓
Validated entry points call arch.validate()
```

Config parsing handles:
- Top-level config (Llama, Qwen, etc.)
- Nested `text_config` (multimodal Gemma 3/4)
- Fallback defaults per model family

Detection is intentionally permissive so tooling can inspect partial configs.
Validated entry points call `arch.validate()` to fail fast on invalid dimensions,
head geometry, RoPE values, per-layer metadata, KV sharing, or MoE routing. The
validation implementation and diagnostic field constants live in `validation.rs`.

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
    If tensor is a packed BF16 expert block:
      store a retained mmap byte range instead of copying to heap
      skip f32 conversion
    Convert dtype:
      f32 → use directly
      f16 → quant::half::decode_f16
      bf16 → quant::half::decode_bf16
      other → collected into ModelWeights::skipped_tensors (not fatal)
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

Absent optional GGUF metadata is omitted from the synthesized config so the
same architecture defaults and loader fallbacks used by safetensors configs
still apply. For example, a Llama GGUF without `{arch}.rope.freq_base` gets the
standard 10,000 RoPE base instead of an explicit zero, and missing vocab size
can still fall back to tokenizer metadata.

### 3. Load Tensors

```
For each tensor descriptor:
  Read name, shape, dtype, offset
  Normalize key ("blk.N." → "layers.N.", etc.)
  Apply optional skip predicate before reading/dequantizing data
  Seek to data offset
  ↓
  Match dtype:
    F32 → read directly
    F16 → quant::half::decode_f16
    BF16 → quant::half::decode_bf16
    Q4_0 → quant::ggml::dequantize (block decode)
    Q4_1 → quant::ggml::dequantize
    Q5_0 → quant::ggml::dequantize
    Q5_1 → quant::ggml::dequantize
    Q8_0 → quant::ggml::dequantize
    Q4_K → quant::ggml::dequantize
    Q6_K → quant::ggml::dequantize
    other → ModelError::UnsupportedDtype
  ↓
  Reshape GGUF `[cols, rows]` dimensions into standard `[rows, cols]`
  row-major ndarray matrices and insert into tensors
```

`load_gguf_filtered` applies the predicate after key normalization and before
data-size calculation and dequantization. This is what keeps walk-only GGUF
loads from expanding FFN tensors into f32.

### 4. Key Translation

GGUF uses different key patterns than safetensors:

| GGUF key | Safetensors equivalent |
|----------|----------------------|
| `blk.0.attn_q.weight` | `layers.0.self_attn.q_proj.weight` |
| `blk.0.attn_qkv.weight` | `layers.0.self_attn.qkv_proj.weight` (split into q/k/v post-load) |
| `blk.0.ffn_gate.weight` | `layers.0.mlp.gate_proj.weight` |
| `token_embd.weight` | `embed_tokens.weight` |
| `position_embd.weight` | `wpe.weight` (lookup via `arch.position_embed_key()`) |
| `output_norm.weight` | `norm.weight` |

The replacement table is centralized in `loading/gguf.rs`; add new GGUF key
forms there rather than scattering ad-hoc rewrites through loading code.

### 5. Canonical-orientation pass

After key translation, `load_gguf_filtered_with_validation` runs three
post-load normalisation passes so all downstream consumers see a single
shape:

- `orient_ffn_tensors` — for each layer, transposes `ffn_gate`, `ffn_up`,
  `ffn_down`, MoE per-expert variants, and shared-expert variants when their
  loaded shape matches the inverse of the canonical Linear layout (gate/up:
  `(intermediate, hidden)`, down: `(hidden, intermediate)`).
- `orient_attention_tensors` — same idea for `q_proj`/`k_proj`/`v_proj`/
  `o_proj` plus the fused `qkv_proj` `(q + 2*kv, hidden)` shape. No-op when
  rows == cols (orientation can't be inferred from shape alone).
- `split_fused_qkv` — for architectures that declare `fused_qkv_key`, slices
  the fused `qkv_proj` rows into per-projection `q_proj`/`k_proj`/`v_proj`
  tensors at the trait's standard keys, plus matching biases from the fused
  bias vector. After this pass downstream code never sees `qkv_proj`.

These passes are driven by trait methods + `ModelConfig` dimensions — no
family-specific branching. Architectures that don't need them are no-ops.

The motivation is real-world non-canonical GGUFs: some GPT-2 converters
don't transpose Conv1D weights, leaving FFN tensors shaped `(intermediate,
hidden)` instead of the Linear `(hidden, intermediate)`. The orient passes
detect and fix this; the split pass unfuses GPT-2's `c_attn` matrix into
the per-projection layout the rest of the stack already speaks.

## ModelWeights Struct

```rust
pub struct ModelWeights {
    pub tensors: HashMap<String, WeightArray>,   // 2D weight matrices
    pub vectors: HashMap<String, Vec<f32>>,      // 1D vectors (norms, biases)
    pub raw_bytes: HashMap<String, Vec<u8>>,     // Small packed-byte fallback/test tensors
    pub skipped_tensors: Vec<(String, String)>,  // (key, dtype) for unsupported dtypes
    pub packed_mmaps: HashMap<String, Mmap>,     // Retained memory-mapped packed files
    pub packed_byte_ranges: HashMap<String, (String, usize, usize)>, // key → (file, offset, len)
    pub embed: WeightArray,                       // Embedding matrix [vocab, hidden]
    pub lm_head: WeightArray,                     // Output projection (may be tied to embed)
    pub position_embed: Option<WeightArray>,       // Learned wpe (GPT-2); None for rotary models
    pub arch: Box<dyn ModelArchitecture>,          // Detected architecture
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

### Memory management methods

| Method | Frees | Use case |
|--------|-------|----------|
| `drop_ffn_weights()` | gate/up/down projections, packed expert blocks | Walk-only inference (vindex-backed FFN) |
| `drop_attn_weights()` | Q/K/V/O projections, QK norms | Server-side FFN-only deployment |
| `drop_lm_head()` | Output projection matrix | Server that doesn't compute logits |
| `drop_embed()` | Input embedding matrix | Server that receives residuals, not tokens |

All return freed bytes. Typical savings for a 4B model:
- `drop_ffn_weights`: ~13 GB (~80% of parameters)
- `drop_attn_weights`: ~1 GB
- `drop_lm_head` / `drop_embed`: ~2.7 GB each

Packed byte tensors are read through `ModelWeights::get_packed_bytes()`, which
checks retained mmap ranges first and falls back to `raw_bytes`. Gemma 4 A4B
packed BF16 expert tensors are kept in mmap ranges during safetensors loading
so loading does not clone multi-GB expert blocks into heap memory.

Pattern matching for `drop_ffn_weights`:
- `gate_proj`, `up_proj`, `down_proj` (dense models)
- `mlp.c_fc`, `mlp.c_proj` (StarCoder2)
- `ffn_gate`, `ffn_up`, `ffn_down` (GGUF key format)
- `mlp.experts`, `block_sparse_moe.experts` (MoE per-expert)
- `packed_gate_up_blocks`, `packed_down_blocks` (GPT-OSS MXFP4)

Loader string constants are centralized in code:
- `weights.rs` owns shared FFN/attention classifiers and packed expert key fragments.
- `loading/safetensors.rs` owns safetensors/GGUF extension names, HF cache path fragments, and GPT-OSS MXFP4 suffix/key helpers.
- `loading/gguf.rs` owns GGUF metadata suffixes and the GGUF-to-HF key replacement table.

### skipped_tensors

Tensors with unsupported dtypes (I64 attention masks, U8 token type IDs, etc.) are collected here rather than causing a load failure. Each entry is `(tensor_key, dtype_string)`. Check after loading to detect unexpected format gaps:

```rust
let weights = load_model_dir_validated(path)?;
for (key, dtype) in &weights.skipped_tensors {
    if !["I64", "I32", "U8"].iter().any(|&d| dtype.contains(d)) {
        eprintln!("unexpected skipped tensor: {key} ({dtype})");
    }
}
```
