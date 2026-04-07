# ModelArchitecture Trait — Design and Extension Guide

## Overview

`ModelArchitecture` is the core abstraction in `larql-models`. It has 82 methods that describe *what a model is* — tensor key patterns, norm behavior, activation functions, scaling — without any compute dependencies.

Every model family (Gemma, Llama, DeepSeek, ...) implements this trait. The rest of LARQL (inference, compute, vindex) only interacts with models through this trait.

## Design Principles

### 1. Zero compute dependency

The trait lives in a crate with no math dependencies beyond `ndarray` (for `ArcArray2` in `ModelWeights`). It never imports BLAS, Metal, or any compute library. This ensures architecture definitions can be used anywhere — CLI tools, web servers, analysis scripts — without pulling in heavy dependencies.

### 2. Sensible defaults for everything

Every trait method has a default implementation. This means:
- Adding a new architecture only requires overriding what's different
- Adding a new method to the trait doesn't break existing architectures
- The `GenericArch` fallback works with zero overrides

```rust
// A minimal architecture implementation:
impl ModelArchitecture for MyArch {
    fn family(&self) -> &str { "my_model" }
    fn config(&self) -> &ModelConfig { &self.config }
    // Everything else uses defaults (RMSNorm, SiLU, Gated FFN, no MoE, ...)
}
```

### 3. Per-layer methods

Methods that can vary by layer take a `layer: usize` parameter. This is critical for architectures like Gemma 4 where head_dim, KV heads, and rotary fraction differ between sliding and global attention layers.

```rust
// Base trait provides constant defaults:
fn head_dim_for_layer(&self, _layer: usize) -> usize {
    self.config().head_dim
}

// Gemma 4 overrides per layer:
fn head_dim_for_layer(&self, layer: usize) -> usize {
    if self.is_global_layer(layer) {
        self.config.global_head_dim.unwrap_or(self.config.head_dim)
    } else {
        self.config.head_dim
    }
}
```

### 4. String keys, not enums

Tensor keys are returned as `String`, not an enum. This keeps the trait open to new tensor patterns without modifying central types. Component names (`ffn_down`, `attn_ov`, ...) are `&str` constants for the same reason.

## Method Categories

### Tensor Keys (~20 methods)

Map layer indices to safetensors key strings. The trait provides HuggingFace-standard defaults (`layers.{N}.self_attn.q_proj.weight`). Architectures override when their keys differ.

Key pattern methods:
- `layer_prefix(layer)` → `"layers.5."` (base for all layer keys)
- `key_prefixes_to_strip()` → `&["model."]` (stripped during loading)
- `embed_key()`, `final_norm_key()` → embedding and final norm
- `attn_q_key(layer)`, `attn_k_key(layer)`, ... → attention projections
- `ffn_gate_key(layer)`, `ffn_up_key(layer)`, `ffn_down_key(layer)` → FFN
- `input_layernorm_key(layer)`, `post_attention_layernorm_key(layer)` → norms

### Attention Geometry (~15 methods)

Control how attention is computed at each layer:
- `head_dim_for_layer(layer)` — head dimension (Gemma 4: 256 sliding, 512 global)
- `num_kv_heads_for_layer(layer)` — KV head count
- `num_q_heads_for_layer(layer)` — Q head count
- `rotary_fraction_for_layer(layer)` — fraction of dims with RoPE (Gemma 4: 0.25 global)
- `rope_base_for_layer(layer)` — RoPE theta (Gemma 3/4: 10k sliding, 1M global)
- `attention_scale_for_layer(layer)` — 1/sqrt(head_dim) or 1.0 for QK-norm models
- `is_sliding_window_layer(layer)` — sliding vs full attention
- `v_shares_k(layer)` — K=V sharing (Gemma 4)
- `kv_shared_source_layer(layer)` — cross-layer KV reuse

### Normalization (~8 methods)

- `norm_type()` — RMSNorm vs LayerNorm
- `norm_weight_offset()` — 0.0 (Llama, Gemma 4) or 1.0 (Gemma 2/3: weight = 1 + learned)
- `qk_norm_weight_offset()` — same for QK norms (Gemma 2/3: 1.0, Gemma 4: 0.0)
- `has_post_norms()` — 4 norms per layer (Gemma 2/3/4) vs 2 (Llama)
- `attn_q_norm_key(layer)`, `attn_k_norm_key(layer)` — QK norm weights (None if unused)

### MoE (~12 methods)

- `is_moe()`, `num_experts()`, `num_experts_per_token()`, `num_shared_experts()`
- `expert_format()` — `PerExpert` (Mixtral) vs `PackedMxfp4` (GPT-OSS)
- `moe_router_key(layer)` — router weight tensor
- `expert_ffn_{gate,up,down}_key(layer, expert_id)` — per-expert keys
- `packed_{gate_up,down}_{blocks,scales}_key(layer)` — packed MXFP4 keys
- `shared_expert_{gate,up,down}_key(layer)` — always-active experts (DeepSeek)

### MLA (~6 methods)

Multi-head Latent Attention (DeepSeek):
- `uses_mla()` — flag
- `kv_lora_rank()`, `q_lora_rank()` — compression dimensions
- `mla_kv_a_key(layer)`, `mla_kv_b_key(layer)` — KV compress/decompress
- `mla_q_a_key(layer)`, `mla_q_b_key(layer)` — Q compress/decompress

## Adding a New Architecture

1. **Create the file**: `src/architectures/my_model.rs`
2. **Implement the struct**:

```rust
use crate::config::{ModelArchitecture, ModelConfig};

pub struct MyModelArch {
    config: ModelConfig,
}

impl MyModelArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for MyModelArch {
    fn family(&self) -> &str { "my_model" }
    fn config(&self) -> &ModelConfig { &self.config }
    
    // Override only what differs from defaults:
    fn norm_type(&self) -> NormType { NormType::LayerNorm }
    fn activation(&self) -> Activation { Activation::Gelu }
    fn ffn_type(&self) -> FfnType { FfnType::Standard }
}
```

3. **Register in `architectures/mod.rs`**: `pub mod my_model;`
4. **Add detection in `detect.rs`**: Match on `model_type` string
5. **Re-export in `lib.rs`**: `pub use architectures::my_model::MyModelArch;`
6. **Add tests in `detect.rs`**: Config parsing + key pattern assertions

## Precomputation Pattern (Gemma 4)

For architectures with complex per-layer behavior, precompute lookup tables in `from_config` rather than computing per call:

```rust
pub struct Gemma4Arch {
    config: ModelConfig,
    global_layers: Vec<bool>,      // precomputed: which layers are full attention
    kv_sources: Vec<Option<usize>>, // precomputed: KV sharing sources
}
```

This keeps trait methods O(1) — a simple vector index rather than conditional logic.
