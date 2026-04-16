# Performance — larql-models

This crate is not compute-bound — it describes models and loads weights. Performance characteristics are about loading speed and memory.

## Weight Loading (M3 Max, NVMe SSD)

| Model | Format | Shards | Tensors | Load Time | Peak RAM | Notes |
|-------|--------|--------|---------|-----------|----------|-------|
| Gemma 3 4B | safetensors | 2 | ~270 | ~2s | ~16.6GB | f16 → f32 conversion |
| Gemma 3 4B | safetensors (mmap) | 2 | ~270 | ~0.8s | ~8.3GB | Zero-copy where possible |
| Llama 3 8B | safetensors | 4 | ~290 | ~4s | ~32GB | f16 → f32 |
| Gemma 3 4B | GGUF Q4_K | 1 | ~270 | ~3s | ~16.6GB | Dequant Q4_K → f32 |

### Where Time Goes

| Phase | % of Load | Notes |
|-------|-----------|-------|
| mmap file(s) | 5% | OS page cache makes repeated loads fast |
| Parse safetensors index | 1% | JSON header with tensor offsets |
| dtype conversion (f16→f32) | 70% | Vectorized but still touches every byte |
| Prefix stripping + key mapping | 1% | String operations on ~270 keys |
| Architecture detection | <1% | JSON parse + match |
| GGUF dequantization | 80% | Block-by-block decode (when using GGUF) |

### Memory: drop_ffn_weights

Walk-only mode drops FFN tensors after loading:

| Model | Before | After | Freed | Savings |
|-------|--------|-------|-------|---------|
| Gemma 3 4B (f32) | 16.6GB | 3.5GB | 13.1GB | 79% |
| Llama 3 8B (f32) | 32GB | 6.5GB | 25.5GB | 80% |

FFN weights (gate + up + down projections) are ~80% of total model weight. When using vindex walk mode, these are served from mmap'd index files instead.

## Architecture Detection

Detection is essentially instant — JSON parse + string match:

```
detect_from_json: <1μs (no I/O)
detect_architecture: ~50μs (read config.json + parse + detect)
```

## Config Parsing

`parse_model_config` handles ~30 fields from config.json. All fields use `.as_u64()` / `.as_f64()` with defaults — no validation overhead, no allocations beyond the final `ModelConfig` struct.

Gemma 4 adds precomputed vectors in `from_config`:
- `global_layers: Vec<bool>` — O(num_layers) allocation, computed once
- `kv_sources: Vec<Option<usize>>` — O(num_layers), computed once

These avoid per-call branching in hot-path trait methods like `head_dim_for_layer()`.

## Quantization Format Performance

Encode/decode throughput (single-threaded, M3 Max):

| Format | Operation | Throughput | Notes |
|--------|-----------|------------|-------|
| f16 | encode (f32→f16) | ~2 GB/s | Bit manipulation, no SIMD |
| f16 | decode (f16→f32) | ~2 GB/s | Bit manipulation |
| bf16 | decode (bf16→f32) | ~2 GB/s | Shift + mask |
| Q4_0 | dequantize (32-block) | ~500 MB/s | Scale × nibble lookup |
| Q8_0 | dequantize (32-block) | ~800 MB/s | Scale × int8, simpler |
| MXFP4 | dequantize (32-block) | ~400 MB/s | e8m0 scale decode + 4-bit lookup |

These are data format operations only. For compute-path quantized operations (GPU matvec at 57 GB/s), see `larql-compute/PERFORMANCE.md`.
