# larql-models

Model architecture definitions for LARQL — traits, config parsing, tensor key mappings, weight loading, and quantization formats.

## What it does

Describes *what a model is* without performing any computation. Every model architecture implements the `ModelArchitecture` trait, which tells the rest of LARQL how to find tensors, apply norms, scale embeddings, handle RoPE, and route MoE experts — all without importing a single math crate.

## Supported Architectures

| Architecture | Model Type | Key Features |
|-------------|-----------|--------------|
| **Gemma 4** | `gemma4*` | Per-layer head_dim, partial RoPE, K=V sharing, V-norm, PLE, KV layer sharing, layer scalars |
| **Gemma 3** | `gemma3*` | QK-norm, 4 norms/layer, sliding window (every 6th full), dual RoPE bases |
| **Gemma 2** | `gemma2`, `gemma` | QK-norm, attn/final logit softcapping, 4 norms/layer |
| **Llama** | `llama*` | GQA, RoPE scaling (linear, YaRN, llama3) |
| **Mistral** | `mistral` | GQA, sliding window |
| **Mixtral** | `mixtral` | MoE (PerExpert), block_sparse_moe routing |
| **Qwen** | `qwen*` | Attention bias, GQA |
| **DeepSeek** | `deepseek*` | MoE + MLA (multi-head latent attention), KV/Q compression |
| **GPT-OSS** | `gpt_oss` | MoE (PackedMxfp4), fused expert blocks, e8m0 scales |
| **Granite** | `granite*` | Embedding/residual/attention/logits multipliers |
| **StarCoder2** | `starcoder2` | LayerNorm, GELU, FFN bias, attention bias |
| **Generic** | (fallback) | Safe defaults for unknown model_type values |

## Architecture Detection

```rust
use larql_models::{detect_architecture, detect_from_json, ModelArchitecture};

// From a model directory (reads config.json)
let arch = detect_architecture(Path::new("/path/to/model"))?;

// From parsed JSON (multimodal text_config handled automatically)
let arch = detect_from_json(&config_json);

println!("{} — {} layers, head_dim={}", 
    arch.family(), arch.config().num_layers, arch.config().head_dim);
```

Detection handles both top-level and nested `text_config` (multimodal models like Gemma 3/4).

## ModelArchitecture Trait

The trait has 82 methods organized into categories:

| Category | Methods | Purpose |
|----------|---------|---------|
| **Tensor keys** | `attn_q_key`, `ffn_gate_key`, `embed_key`, ... | Where to find weights in safetensors |
| **Norms** | `norm_type`, `norm_weight_offset`, `has_post_norms` | How to normalize hidden states |
| **Attention** | `attention_scale`, `is_sliding_window_layer`, `rope_base_for_layer` | Attention geometry per layer |
| **Per-layer geometry** | `head_dim_for_layer`, `num_kv_heads_for_layer`, `rotary_fraction_for_layer` | Gemma 4 variable attention |
| **FFN** | `activation`, `ffn_type` | Gated vs standard, SiLU vs GELU |
| **MoE** | `is_moe`, `num_experts`, `expert_format`, `moe_router_key` | Expert routing |
| **MLA** | `uses_mla`, `kv_lora_rank`, `mla_kv_a_key` | DeepSeek compressed attention |
| **Scaling** | `embed_scale`, `residual_multiplier`, `logits_scaling` | Granite-style multipliers |
| **Softcapping** | `attn_logit_softcapping`, `final_logit_softcapping` | Gemma 2 score clamping |
| **PLE** | `has_per_layer_embeddings`, `per_layer_embed_key` | Gemma 4 per-layer embeddings |
| **KV sharing** | `kv_shared_source_layer`, `v_shares_k` | Cross-layer KV reuse, K=V |

Every method has a sensible default. New architectures only override what differs.

## Weight Loading

```rust
use larql_models::load_model_dir;

// Auto-detects format: safetensors or GGUF
let weights = load_model_dir("/path/to/model")?;

// Access tensors
let q_proj = &weights.tensors["layers.0.self_attn.q_proj.weight"];
let embed = &weights.embed;  // Embedding matrix
let lm_head = &weights.lm_head;  // Output projection (may be tied to embed)

// Architecture is attached
println!("{}", weights.arch.family());

// Walk-only mode: drop FFN weights to save ~13GB
let freed = weights.drop_ffn_weights();
```

### Supported Formats

| Format | Source | Handling |
|--------|--------|----------|
| **Safetensors** | HuggingFace | mmap + dtype conversion (f16/bf16 → f32), prefix stripping |
| **GGUF** | llama.cpp | Parse + dequantize (Q4_0, Q4_1, Q8_0, F16, BF16 → f32) |

### HuggingFace Cache Resolution

`resolve_model_path()` finds models in `~/.cache/huggingface/hub/` by name (e.g., `google/gemma-3-4b` → snapshot directory).

## Quantization Formats

| Module | Formats | Purpose |
|--------|---------|---------|
| `quant::half` | f16, bf16 | IEEE 754 half-precision encode/decode |
| `quant::ggml` | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 | GGML block quantization (32-element blocks) |
| `quant::mxfp4` | MXFP4 + e8m0 | Microscaling 4-bit (GPT-OSS/OpenAI packed experts) |

These handle data format encoding/decoding only. Compute operations (GPU matvec, shader dispatch) are in `larql-compute`.

## Vector Interchange

Shared types for NDJSON vector records extracted from model weights:

```rust
use larql_models::{VectorRecord, ALL_COMPONENTS};

// Components: ffn_down, ffn_gate, ffn_up, attn_ov, attn_qk, embeddings
for comp in ALL_COMPONENTS {
    println!("{comp}");
}
```

Component names are strings, not enums — the engine is generic with no domain-specific types baked in.

## Architecture

```
src/
  lib.rs              Re-exports: ModelArchitecture, ModelConfig, detect, load, quant
  config.rs           ModelArchitecture trait + enums (NormType, Activation, FfnType, ExpertFormat)
  detect.rs           Auto-detection from config.json (parse_model_config + dispatch)
  weights.rs          ModelWeights struct (HashMap<String, ArcArray2<f32>>)
  vectors.rs          VectorRecord, VectorFileHeader, component constants

  architectures/
    mod.rs            Module declarations (12 architectures)
    gemma4.rs         Gemma 4 (per-layer geometry, PLE, KV sharing, V-norm)
    gemma3.rs         Gemma 3 (QK-norm, sliding window, dual RoPE)
    gemma2.rs         Gemma 2 (softcapping, QK-norm)
    llama.rs          Llama (GQA, RoPE scaling)
    mistral.rs        Mistral (sliding window)
    mixtral.rs        Mixtral (MoE, PerExpert)
    qwen.rs           Qwen (attention bias)
    deepseek.rs       DeepSeek (MoE + MLA)
    gpt_oss.rs        GPT-OSS (PackedMxfp4 experts)
    granite.rs        Granite (scaling multipliers)
    starcoder2.rs     StarCoder2 (LayerNorm, bias)
    generic.rs        Fallback for unknown models

  loading/
    mod.rs            Format routing (safetensors vs GGUF)
    safetensors.rs    mmap + dtype conversion + HF cache resolution
    gguf.rs           GGUF parser + dequantization

  quant/
    mod.rs            Module declarations
    half.rs           f16/bf16 encode/decode
    ggml.rs           Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 block quantization
    mxfp4.rs          MXFP4 + e8m0 scale dequantization

tests/
  test_architectures.rs  Integration tests (58): all 12 architectures, MoE, MLA, bias, scaling, quant

examples/
  architecture_demo.rs   Guided tour: detection, keys, sliding window, MoE, quant formats
  demo_loading.rs        Load model from disk, inspect tensors and architecture
  demo_tensor_keys.rs    Compare tensor key patterns across all 12 architectures
```

## Tests

```bash
cargo test -p larql-models
```

169 tests (111 unit + 58 integration) covering all 12 architectures: detection, tensor key patterns, MoE expert formats (PerExpert vs PackedMxfp4), MLA compression keys, Gemma 2 softcapping + QK norm offsets, Gemma 3 sliding window + dual RoPE, Gemma 4 per-layer geometry (head_dim, KV heads, partial RoPE, KV sharing, PLE, V-norm, K=V), Qwen attention bias, StarCoder2 bias + LayerNorm + non-gated FFN, DeepSeek shared experts + MLA, Granite scaling multipliers, generic fallback defaults, quantization round-trips (Q4_0, Q8_0), malformed-input rejection across every GGML dequantizer + MXFP4 + truncated GGUF files, and `drop_ffn_weights`.

## Examples

### Demos

```bash
# Architecture detection — all 12 architectures, tensor keys, sliding window, MoE, quant
cargo run -p larql-models --example architecture_demo

# Load a real model and inspect its structure
cargo run -p larql-models --example demo_loading -- /path/to/model

# Compare tensor key patterns across architectures
cargo run -p larql-models --example demo_tensor_keys
```

## Documentation

| Doc | Content |
|-----|---------|
| [ROADMAP.md](ROADMAP.md) | Planned architectures, trait extensions, loading improvements |
| [docs/adr/](docs/adr/) | 6 architectural decision records (trait design, component names, config parsing, prefix stripping, Gemma 4 layers, norm offsets) |
| [docs/architecture-trait.md](docs/architecture-trait.md) | ModelArchitecture trait design and extension guide |
| [docs/weight-loading.md](docs/weight-loading.md) | Loading pipeline: formats, dtype conversion, prefix stripping |
| [docs/quantization-formats.md](docs/quantization-formats.md) | GGML, MXFP4, f16 format specifications |

## Design Principles

1. **Zero compute** — this crate describes models, it never runs them
2. **Trait-driven** — every architecture implements `ModelArchitecture`, callers are generic
3. **Sensible defaults** — new architectures only override what differs from the base
4. **String components** — no domain-specific enums (component names are `&str`)
5. **Format-agnostic** — safetensors and GGUF produce the same `ModelWeights`
6. **Multimodal-aware** — config parsing handles nested `text_config` automatically

## License

Apache-2.0
