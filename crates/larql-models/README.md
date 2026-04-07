# larql-models

Model architecture definitions, weight loading, config parsing, tensor key mappings, and quantization formats.

This crate knows *what models are* and *how to load them* — architecture traits, weight key patterns, safetensors/GGUF loading, MXFP4 dequantization — without any compute or inference dependencies.

## Usage

```rust
use larql_models::*;

// Load model weights from safetensors, GGUF, or MLX
let weights = load_model_dir("google/gemma-3-4b-it")?;
println!("{} layers, hidden={}", weights.num_layers, weights.hidden_size);

// Or from a GGUF file
let weights = load_gguf("model.gguf")?;

// Detect architecture from config.json
let arch = detect_architecture(&model_dir)?;
println!("Family: {}, layers: {}", arch.family(), arch.config().num_layers);

// Get tensor keys
let gate_key = arch.ffn_gate_key(0);     // "layers.0.mlp.gate_proj.weight"
let q_key = arch.attn_q_key(0);          // "layers.0.self_attn.q_proj.weight"

// MoE models
if arch.is_moe() {
    match arch.expert_format() {
        ExpertFormat::PerExpert => {
            // Mixtral: per-expert tensors
            let key = arch.expert_ffn_gate_key(0, 3);
        }
        ExpertFormat::PackedMxfp4 => {
            // GPT-OSS: packed MXFP4 tensors
            let key = arch.packed_gate_up_blocks_key(0);
        }
    }
}

// Quantization
use larql_models::quant::{half, ggml, mxfp4};
let f16_bytes = half::encode_f16(&[1.0, 2.0, 3.0]);
let f32_back = half::decode_f16(&f16_bytes);
```

## Supported Architectures

| Family | Model Type | FFN | Expert Format |
|--------|-----------|-----|---------------|
| Gemma 4 | `gemma4` | Gated (GeGLU) | Dense (31B) / MoE (A4B) |
| Gemma 3 | `gemma3` | Gated (GeGLU) | Dense |
| Gemma 2 | `gemma2` | Gated (GeGLU) | Dense |
| Llama | `llama` | Gated (SiLU) | Dense |
| Mistral | `mistral` | Gated (SiLU) | Dense |
| Mixtral | `mixtral` | MoE | PerExpert |
| Qwen | `qwen2` | Gated (SiLU) | Dense |
| DeepSeek | `deepseek_v2` | MoE | PerExpert |
| GPT-OSS | `gpt_oss` | MoE | PackedMxfp4 |
| Granite | `granite` | Gated | Dense |
| StarCoder2 | `starcoder2` | Gated | Dense |
| GPT-2 | (generic) | Dense (GELU) | Dense |

## Model Loading

Loads model weights from multiple formats into the canonical `ModelWeights` struct:

| Format | Function | Source |
|--------|----------|--------|
| safetensors | `load_model_dir()` | HuggingFace, local directory |
| MLX | `load_model_dir()` | Apple MLX (safetensors in `weights/` subdir) |
| GGUF | `load_gguf()` | llama.cpp (dequantized to f32) |

Auto-detects format from file extensions. Resolves HuggingFace model IDs from cache. Handles multi-shard safetensors.

## Quantization Formats

Data format encoding and decoding. Compute operations (matvec, vecmat) are in `larql-compute`.

```
quant/
├── half.rs      f16/bf16 ↔ f32 (encode + decode)
├── ggml.rs      GGML block quantization:
│                  - Dequantize: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 (all implemented)
│                  - Quantize: Q4_0, Q8_0 (f32 → packed bytes)
│                  - Format metadata: tensor_data_size, type_name
└── mxfp4.rs     MXFP4 microscaling (e8m0 scales + 4-bit values)
```

> **Note:** Q4/Q8 matrix-vector operations (matvec, vecmat, NEON kernels) have moved
> to `larql-compute`. This crate only handles data format encoding/decoding.

## Crate Structure

```
larql-models/src/
├── lib.rs              Re-exports
├── config.rs           ModelArchitecture trait, ModelConfig, ExpertFormat
├── detect.rs           Auto-detect from config.json, ModelError
├── weights.rs          ModelWeights struct
├── vectors.rs          VectorRecord, TopKEntry
├── loading/            Model weight loading
│   ├── safetensors.rs  safetensors/MLX → ModelWeights
│   └── gguf.rs         GGUF → ModelWeights
├── quant/              Quantization/dequantization
│   ├── half.rs         f16/bf16
│   ├── ggml.rs         GGML block quantization
│   └── mxfp4.rs        MXFP4 (GPT-OSS)
└── architectures/
    ├── llama.rs        Llama 2/3
    ├── gemma4.rs       Gemma 4 (per-layer head_dim, K=V, partial RoPE, layer_scalar)
    ├── gemma3.rs       Gemma 3
    ├── gemma2.rs       Gemma 2
    ├── mistral.rs      Mistral
    ├── mixtral.rs      Mixtral (MoE, PerExpert)
    ├── gpt_oss.rs      GPT-OSS (MoE, PackedMxfp4)
    ├── qwen.rs         Qwen 2/2.5
    ├── deepseek.rs     DeepSeek V2/V3
    ├── granite.rs      Granite
    ├── starcoder2.rs   StarCoder2
    └── generic.rs      Fallback
```

## Testing

```bash
cargo test -p larql-models                              # 88 tests
cargo run -p larql-models --example architecture_demo   # Architecture showcase
```

## License

Apache-2.0
