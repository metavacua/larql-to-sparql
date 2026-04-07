# Roadmap — larql-models

## Current: 12 architectures, 130 tests, safetensors + GGUF loading

## P0: Complete Gemma 4 Support

### Wire v_shares_k into inference forward pass
**Impact**: Correct K=V handling without runtime tensor probing  
**Effort**: Low  
**Status**: Trait method done (returns `config.attention_k_eq_v`), inference wiring pending

Currently the inference crate detects K=V by checking for missing v_proj tensors at runtime. Now that `v_shares_k()` exposes the config flag, the forward pass should use it directly.

### Validate PLE (per-layer embeddings) end-to-end
**Impact**: Correct Gemma 4 E2B inference  
**Effort**: Medium  
**Status**: Keys and config parsed, forward pass not yet wired

PLE adds a gated embedding lookup per layer. Keys (`per_layer_embed_key`, `per_layer_input_gate_key`, `per_layer_projection_key`, `post_per_layer_input_norm_key`) are all implemented. Need to wire into inference and verify against HuggingFace reference outputs.

### KV layer sharing in inference
**Impact**: Memory savings for Gemma 4 (20 shared layers = 20 fewer KV caches)  
**Effort**: Medium  
**Status**: `kv_shared_source_layer()` returns correct sources, KV cache not yet shared

## P1: Architecture Coverage

### Phi-3 / Phi-4
**Effort**: Low  
**Status**: Not started

Similar to Llama with some attention differences (partial RoPE, SuRoPE). Most trait defaults apply.

### Command R / Cohere
**Effort**: Medium  
**Status**: Not started

Different attention key pattern, different norm placement.

### Mamba / state-space models
**Effort**: Large  
**Status**: Research

Would require extending the trait beyond transformer assumptions (no attention keys, no KV cache). May warrant a separate trait hierarchy.

## P2: Loading Improvements

### Streaming safetensors loading
**Effort**: Medium  
**Status**: Not started

Current loader reads all shards into memory. For 70B+ models, streaming with per-layer loading would reduce peak memory. Already have mmap infrastructure — extend to lazy loading with `Arc<Mmap>` references.

### GGUF quantized inference (skip dequant)
**Effort**: Large  
**Status**: Not started

Currently GGUF tensors are dequantized to f32 during loading. For Q4_K/Q6_K formats, keep data in quantized form and pass directly to `larql-compute` Q4_K shaders. Requires a `QuantizedWeights` variant alongside `ModelWeights`.

### MLX npz/safetensors hybrid
**Effort**: Low  
**Status**: Partial (MLX safetensors work, npz not yet)

Apple MLX models sometimes use `.npz` format. Add npz parsing alongside safetensors.

## P3: Trait Evolution

### Per-layer FFN type
**Effort**: Low  
**Status**: Not started

Some models (e.g., future MoE variants) may have different FFN types per layer (dense for early layers, MoE for later). Add `ffn_type_for_layer(layer)` method.

### Attention pattern abstraction
**Effort**: Medium  
**Status**: Research

Current sliding window is boolean per layer. Future models may have more complex patterns (local + global hybrid, dilated attention, prefix caching hints). Consider a richer `AttentionPattern` enum.

### Config validation
**Effort**: Low  
**Status**: Not started

Add a `validate()` method to `ModelArchitecture` that checks for inconsistencies (e.g., head_dim doesn't divide hidden_size, num_experts set but not num_experts_per_token). Currently these fail silently at inference time.

## Completed

| Item | Date | Impact |
|------|------|--------|
| ModelArchitecture trait | 2026-03 | Foundation — 80+ methods with defaults |
| Gemma 2/3 support | 2026-03 | QK-norm, softcapping, sliding window |
| Llama/Mistral/Qwen/DeepSeek | 2026-03 | Core architecture coverage |
| Mixtral MoE (PerExpert) | 2026-03 | Expert key patterns |
| GPT-OSS (PackedMxfp4) | 2026-03 | MXFP4 dequantization, packed expert keys |
| Granite (scaling multipliers) | 2026-03 | Embedding/residual/attention/logits scaling |
| StarCoder2 | 2026-03 | LayerNorm, bias, GELU |
| GGUF loading | 2026-03 | Q4_0/Q4_1/Q8_0/F16/BF16 dequantization |
| Safetensors mmap + HF cache | 2026-03 | Zero-copy loading, cache resolution |
| drop_ffn_weights | 2026-04 | Walk-only mode saves ~13GB |
| Gemma 4 architecture | 2026-04 | Per-layer geometry, PLE, KV sharing, V-norm, layer scalars |
| Gemma 4 31B + E2B configs | 2026-04 | Both variants tested with real config.json |
| Gemma4Arch re-export | 2026-04-07 | Public API complete |
| v_shares_k from config | 2026-04-07 | Uses attention_k_eq_v flag instead of hardcoded false |
| Gemma 3 qk_norm_weight_offset | 2026-04-07 | Was missing (Gemma 2 had it, Gemma 3 didn't) |
| Full test coverage (130 tests) | 2026-04-07 | All 12 architectures tested: Gemma 2/3/4, Llama, Mistral, Mixtral, Qwen, DeepSeek, GPT-OSS, Granite, StarCoder2, Generic |
| Clippy clean (zero warnings) | 2026-04-07 | lib + examples + tests all pass `-D warnings` |
| Documentation suite | 2026-04-07 | README, ROADMAP, PERFORMANCE, 3 docs, 6 ADRs |
| Example suite (3 demos) | 2026-04-07 | architecture_demo (all 12), demo_tensor_keys (all 12), demo_loading |
