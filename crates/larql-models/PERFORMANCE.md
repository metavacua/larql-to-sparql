# Performance — larql-models

This crate is not compute-bound — it describes models and loads weights. Performance characteristics are about loading speed and memory.

## Benchmark Suite

Run the crate-local Criterion suite with:

```bash
cargo bench -p larql-models --bench models
```

The suite is intentionally crate-local and does not require external model
downloads. It currently prints these groups:

- `config_detection/*` — permissive and validated detection for Llama, Gemma 4, and GPT-OSS configs
- `config_validation/*` — standalone `ModelArchitecture::validate()` cost for those same configs
- `tensor_keys/*` — hot tensor-key generation across all Gemma 4 layers
- `tensor_classification/*` — FFN/non-FFN key classification
- `quant_decode/*` — GGML Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, and Q6_K dequantization
- `weight_loading/*` — validated loading of an in-benchmark synthetic safetensors model

Current local baseline from 2026-04-26:

| Benchmark | Median |
|-----------|--------|
| `config_detection/detect/llama` | ~590 ns |
| `config_detection/detect_validated/llama` | ~605 ns |
| `config_detection/detect/gemma4` | ~2.48 µs |
| `config_detection/detect_validated/gemma4` | ~2.58 µs |
| `config_detection/detect/gpt_oss` | ~583 ns |
| `config_detection/detect_validated/gpt_oss` | ~609 ns |
| `config_validation/llama` | ~24 ns |
| `config_validation/gemma4` | ~149 ns |
| `config_validation/gpt_oss` | ~23 ns |
| `tensor_keys/gemma4_all_layer_hot_keys` | ~24.3 µs |
| `tensor_classification/is_ffn_tensor_key_set` | ~6.15 µs |
| `weight_loading/load_synthetic_safetensors_validated` | ~156 µs |

| Quant Decode | Median | Throughput |
|--------------|--------|------------|
| `quant_decode/q4_0` | ~4.43 µs | ~1.85 Gelem/s |
| `quant_decode/q4_1` | ~4.22 µs | ~1.94 Gelem/s |
| `quant_decode/q5_0` | ~5.09 µs | ~1.61 Gelem/s |
| `quant_decode/q5_1` | ~5.37 µs | ~1.53 Gelem/s |
| `quant_decode/q8_0` | ~3.76 µs | ~2.18 Gelem/s |
| `quant_decode/q4_k` | ~2.40 µs | ~3.42 Gelem/s |
| `quant_decode/q6_k` | ~6.51 µs | ~1.26 Gelem/s |

Validation itself is concrete and small: roughly 23-24 ns for Llama/GPT-OSS and
149 ns for Gemma 4 in the standalone benchmark. End-to-end validated detection
adds roughly +15 ns for Llama, +100 ns for Gemma 4, and +26 ns for GPT-OSS in
this baseline. That keeps validated APIs appropriate for inference/extraction
boundaries while leaving permissive APIs available for inspection tools.

## Weight Loading

The full-model rows below are representative M3 Max / NVMe measurements or
planning baselines, not CI assertions. Re-measure on target hardware before
using them as capacity limits.

| Model | Format | Shards | Tensors | Load Time | Peak RAM | Notes |
|-------|--------|--------|---------|-----------|----------|-------|
| Gemma 3 4B | safetensors | 2 | ~270 | ~2s | ~16.6GB | f16 → f32 scalar decode |
| Gemma 3 4B | safetensors (mmap) | 2 | ~270 | ~0.8s | ~8.3GB | Zero-copy where possible |
| Llama 3 8B | safetensors | 4 | ~290 | ~4s | ~32GB | Planning baseline; re-measure |
| Gemma 3 4B | GGUF Q4_K | 1 | ~270 | ~3s | ~16.6GB | Dequant Q4_K → f32 |

### Where Time Goes

Safetensors and GGUF use different hot paths, so percentages should be read
per format rather than added together.

Safetensors load path:

| Phase | % of Load | Notes |
|-------|-----------|-------|
| mmap file(s) | 5% | OS page cache makes repeated loads fast |
| Parse safetensors index | 1% | JSON header with tensor offsets |
| dtype conversion (f16/bf16→f32) | 70% | Scalar bit decode today; still touches every retained element |
| Prefix stripping + key mapping | 1% | String operations on ~270 keys |
| Architecture detection | <1% | JSON parse + match |
| Config validation | <1% | O(num_layers); ~24 ns for Llama, ~149 ns for Gemma 4, ~23 ns for GPT-OSS |
| `skipped_tensors` collection | <1% | Recorded during the same tensor scan; no extra pass |
| Other runtime overhead | ~22% | Allocation, HashMap insertion, tensor bookkeeping, OS variance |

GGUF load path:

| Phase | % of Load | Notes |
|-------|-----------|-------|
| mmap file | 5% | OS page cache makes repeated loads fast |
| Parse GGUF metadata/index | 5% | Metadata, tensor descriptors, key normalization |
| Prefix stripping + key mapping | 1% | String operations on normalized tensor names |
| Architecture detection | <1% | Derived config JSON + match |
| Config validation | <1% | Same validated API path as safetensors |
| GGUF dequantization | 80% | Block-by-block decode (when using GGUF) |
| Other runtime overhead | ~9% | Allocation, tensor bookkeeping, format routing, OS variance |

### Memory: Walk-only filtering and drop_ffn_weights

Walk-only mode skips FFN tensors during loading where possible. Safetensors keys
are filtered before dtype conversion, GGUF keys are normalized and filtered
before dequantization, and GPT-OSS packed MXFP4 experts are not expanded when
their generated expert keys are filtered. `drop_ffn_weights()` remains available
for already-loaded `ModelWeights`. Gemma 4 A4B packed BF16 expert blocks are
kept as retained mmap byte ranges instead of heap-cloned raw bytes, and
`drop_ffn_weights()` releases their ranges and any unreferenced packed mmaps.

| Model | Before | After | Freed | Savings |
|-------|--------|-------|-------|---------|
| Gemma 3 4B (f32) | 16.6GB | 3.5GB | 13.1GB | 79% |
| Llama 3 8B (f32) | 32GB | 6.5GB | 25.5GB | 80% |

FFN weights (gate + up + down projections) are ~80% of total model weight. When using vindex walk mode, these are served from mmap'd index files instead.

Other memory controls:

| Operation | Use case | Expected impact |
|-----------|----------|-----------------|
| `drop_attn_weights()` | Server-side split where attention is not needed locally | Removes Q/K/V/O and attention norms |
| `drop_lm_head()` | Browse/walk workloads that do not produce logits | Removes output projection when untied |
| `drop_embed()` | Post-extraction workflows that no longer need token embeddings | Removes embedding matrix |

MoE and MLA notes:

- DeepSeek MLA is mostly architecture metadata and key mapping in this crate; loading still follows the same safetensors/GGUF tensor paths.
- Per-expert MoE tensors are ordinary tensors unless a model packs experts into a custom format.
- GPT-OSS packed MXFP4 experts are predicate-aware: walk-only filtering avoids expanding packed gate/up/down experts into f32 when the generated expert keys are filtered out.
- Gemma 4 A4B packed BF16 experts stay mmap-backed and are served through `ModelWeights::get_packed_bytes()`.

## Architecture Detection

Detection is essentially instant — JSON parse + string match:

```
detect_from_json: ~0.6µs for Llama/GPT-OSS, ~2.5µs for Gemma 4 (no I/O)
detect_from_json_validated: ~0.6µs for Llama/GPT-OSS, ~2.6µs for Gemma 4
validate: ~24ns for Llama, ~149ns for Gemma 4, ~23ns for GPT-OSS
detect_architecture: ~50µs estimate (read config.json + parse + detect)
```

## Config Parsing

`parse_model_config` handles ~30 fields from config.json. All fields use `.as_u64()` / `.as_f64()` with defaults and detection remains permissive. `ModelArchitecture::validate()` is an explicit O(num_layers) caller check, not part of detection by default.

Gemma 4 adds precomputed vectors in `from_config`:
- `global_layers: Vec<bool>` — O(num_layers) allocation, computed once
- `kv_sources: Vec<Option<usize>>` — O(num_layers), computed once

These avoid per-call branching in hot-path trait methods like `head_dim_for_layer()`.

## Quantization Format Performance

Encode/decode throughput (single-threaded). The first table is the current
Criterion baseline where available; supported formats without a Criterion row
are still covered by tests but should not be treated as benchmarked yet.

| Format | Operation | Throughput | Notes |
|--------|-----------|------------|-------|
| Q4_0 | dequantize (32-block) | ~1.85 Gelem/s | Criterion `quant_decode/q4_0` |
| Q4_1 | dequantize (32-block) | ~1.94 Gelem/s | Criterion `quant_decode/q4_1` |
| Q5_0 | dequantize (32-block) | ~1.61 Gelem/s | Criterion `quant_decode/q5_0` |
| Q5_1 | dequantize (32-block) | ~1.53 Gelem/s | Criterion `quant_decode/q5_1` |
| Q8_0 | dequantize (32-block) | ~2.18 Gelem/s | Criterion `quant_decode/q8_0` |
| Q4_K | dequantize (256-block) | ~3.42 Gelem/s | Criterion `quant_decode/q4_k` |
| Q6_K | dequantize (256-block) | ~1.26 Gelem/s | Criterion `quant_decode/q6_k` |

Q4_K is faster than Q8_0 in this dequant-only benchmark even though it has more
scale/min logic. The benchmark uses nonzero deterministic K-quant blocks; the
likely reason is input byte traffic: Q4_K reads 144 bytes per 256 output
elements, while Q8_0 reads 272 bytes for the same 256 outputs. This benchmark
does not measure fused K-quant row-dot or scaled-add paths.

Supported formats not yet in the Criterion suite:

| Format | Operation | Current coverage | Notes |
|--------|-----------|------------------|-------|
| f16 | encode/decode | Unit tests | Scalar bit manipulation |
| bf16 | encode/decode | Unit tests | Shift + mask |
| MXFP4 | dequantize (32-element groups) | Unit tests | One e8m0 scale per 32 values; GPT-OSS packed experts |

These are data format operations only. For compute-path quantized operations (GPU matvec at 57 GB/s), see `larql-compute/PERFORMANCE.md`.
