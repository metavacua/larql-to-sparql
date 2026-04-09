# larql-compute

Hardware-accelerated compute backends for LARQL. CPU (BLAS + NEON Q4), Metal GPU, and future CUDA.

## What it does

Provides a `ComputeBackend` trait that abstracts all hardware-specific matrix operations. Every LARQL crate (inference, vindex) uses this trait — the caller never knows whether the operation runs on CPU or GPU.

## Backends

| Backend | Feature flag | f32 matmul | Quantized ops | Pipeline |
|---------|-------------|------------|---------------|---------|
| **CPU** | (always) | BLAS (Accelerate AMX) | C kernel (ARM vdotq_s32) | Sequential |
| **Metal** | `--features metal` | Tiled shaders | Simdgroup Q4/Q4_K/Q6_K/Q8 | One command buffer |
| **CUDA** | (planned) | — | — | — |

## Performance vs Ollama (M3 Max, Gemma 3 4B)

```
LARQL Q4_KF (34 layers):       8.5ms/token = 117 tok/s (decode, KV cached)
Ollama gemma3:4b:              10.3ms/token =  98 tok/s (decode, 34 layers)
vs Ollama:                     0.83x (17% FASTER)
```

### Key Optimizations (2026-04-08 — 2026-04-09)

| Optimization | Savings | Technique |
|-------------|---------|-----------|
| **Cooperative SIMD norms** | **~10ms** | **O(N²)→O(N) reads in rms_norm / residual_norm** |
| Q4_KF FFN routing | ~8ms | llama.cpp-exact kernel (q4kf_proj) for FFN |
| Q4_K matvec rewrite | ~3ms | uint4 loads, 8 rows/TG, multi-row (nr0=2) |
| Buffer pre-allocation | ~2ms | Eliminate 550 Metal allocs per decode |
| Fused gate+up kernels | ~1ms | q4k_ffn_gate_up + q4kf_ffn_gate_up |
| Batched RoPE/V-norm | ~0.5ms | 16 per-head dispatches → 3 batched |
| SIMD KV attention | ~1ms | simd_max/simd_sum, fewer barriers |

### Architecture

Single command buffer + single global encoder for all 34 layers. Pre-allocated scratch
buffers. Format-aware FFN: Q4_KF routes through llama.cpp kernel, Q4_K through fused
gate+up, Q4_0 through legacy Q8 path. All norms use cooperative SIMD reduction.

## Shaders (~48 Metal kernels)

| Category | Kernels | Notes |
|----------|---------|-------|
| f32 matmul | sgemm, sgemm_transb | Tiled 32×32 |
| Q4_0 matvec | v1, v2, v3, **v4** (prod), v5, sparse | v4: uint32 wide loads, 61 GB/s |
| Q4_K/Q6_K | **q4k_matvec** (uint4, nr0=2), q4k_qkv_proj, **q4kf_qkv_proj/q4kf_proj**, q6k_matvec | llama.cpp-exact kernel for Q4_KF |
| Q4_K fused FFN | **q4k_ffn_gate_up**, q4k_geglu_silu_down, q4k_geglu_gelu_tanh_down | Fused gate+up, shared input |
| Q8 | q8_matvec, q8_qkv_proj, q8_proj_rope | Fused QKV, simdgroup reduction |
| Attention | fused_attention (RoPE+GQA+softcap), causal, **kv_attention** (simd), kv_append | SIMD reductions, float4 dot |
| Normalization | rms_norm, layer_norm (2), **v_norm**, **v_norm_batched** | Batched V-norm (1 dispatch) |
| Activation | geglu_silu, geglu_gelu_tanh, silu, gelu_tanh | Gated + standalone |
| Element-wise | residual_add, residual_inject, scale_vector, quantize_q8 | |
| RoPE | rope_apply, rope_at_pos, **rope_at_pos_batched** | Batched all heads (1 dispatch) |
| Fused ops | rms_norm_q8, residual_norm, residual_norm_q8 | Multi-op fusion |
| Experimental | turboquant_encode/decode, graph_walk_knn | |

## Safe Buffer Access

All Metal buffer reads go through one audited function with null/size checks:

```rust
// Replaces 13 previous unsafe { from_raw_parts } sites
pub fn read_buffer_f32(buf: &metal::Buffer, len: usize) -> Vec<f32>
```

## Quick Start

```rust
use larql_compute::{ComputeBackend, default_backend};

let backend = default_backend();
println!("Using: {} ({})", backend.name(), backend.device_info());

// f32 matmul
let c = backend.matmul_transb(a.view(), b.view());

// Q4_K matvec (Ollama-compatible format)
let scores = backend.q4k_matvec(&q4k_data, &x, rows, hidden);

// KV-cached decode (one token through all layers)
let h = backend.decode_token(&layers, &x, hidden, inter, q_dim, kv_dim,
    num_q_heads, num_kv_heads, head_dim, rope_base);

// GPU prefill (seq>1, populates KV cache)
let h = backend.prefill_q4(&layers, &x, hidden, inter, q_dim, kv_dim,
    seq_len, num_q_heads, num_kv_heads, head_dim, rope_base, qk_norm, softcap);
```

## Architecture

```
src/
  lib.rs              Re-exports from pipeline.rs and backend.rs
  pipeline.rs         QuantFormat, QuantWeight, NormType, FfnType, Activation, FullPipelineLayer
  backend.rs          ComputeBackend trait (15 methods)

  cpu/
    mod.rs            CpuBackend (BLAS f32 + C Q4 + Q4_K/Q6_K reference)
    ops/              f32_matmul, q4_matvec, q4_vecmat, q4k_matvec, q6k_matvec,
                      q4_common (quantizers: Q4_0, Q4_K, Q4_KF, Q6_K, GGUF Q4_K),
                      q8_matvec, vector, attention, geglu

  metal/              (feature-gated: --features metal)
    mod.rs            MetalBackend (30 pipeline states, KV cache)
    trait_impl.rs     ComputeBackend dispatch (Q4_K/Q8 dual-path)
    decode.rs         KV-cached decode (norm→QKV→attend→O→FFN per layer)
    prefill.rs        GPU prefill for seq>1
    buffers.rs        GPU buffer cache + read_buffer_f32
    shaders/          44 Metal kernels across 32 shader files
    ops/              GPU dispatch helpers

  csrc/q4_dot.c       ARM NEON Q4 kernel
```

## Tests

```bash
# CPU only (38 tests)
cargo test -p larql-compute

# CPU + Metal (83 tests)
cargo test -p larql-compute --features metal
```

83 tests covering: quantization round-trips, cross-backend correctness (Metal vs CPU with tolerance), shader compilation, fused attention, partial RoPE, KV cache, pipeline output verification, standalone activations (SiLU, GELU-tanh), LayerNorm (with/without bias), V-norm, scale_vector, per-layer eps verification.

## Examples

### Demos

```bash
# Architecture overview — guided tour of all major design decisions
cargo run --release --features metal -p larql-compute --example demo_architecture

# Basic usage — backend detection, matmul, Q4 dispatch
cargo run --release --features metal -p larql-compute --example demo_basic
```

### Benchmarks: Compare (us vs Ollama)

```bash
cargo run --release --features metal -p larql-compute --example compare_decode     # Q4_K vs Q8, KV cached
cargo run --release --features metal -p larql-compute --example compare_generation  # Prefill + decode
cargo run --release --features metal -p larql-compute --example compare_pipeline    # Attention + FFN breakdown
cargo run --release --features metal -p larql-compute --example compare_formats     # Q4_KF vs Q4_K vs GGUF
```

### Benchmarks: Profile (bottleneck analysis)

```bash
cargo run --release --features metal -p larql-compute --example profile_components   # Every op isolated over 34 layers
cargo run --release --features metal -p larql-compute --example profile_operations   # CPU vs Metal per-operation
cargo run --release --features metal -p larql-compute --example profile_kernels      # Q4 v1-v5, sparse, attention
cargo run --release --features metal -p larql-compute --example profile_raw_dispatch # Pure kernel, zero overhead
cargo run --release --features metal -p larql-compute --example profile_new_kernels  # New model-agnostic kernels
cargo run --release --features metal -p larql-compute --example profile_kv_cache     # Attention vs cache length
cargo run --release --features metal -p larql-compute --example profile_bandwidth    # Raw memory throughput
```

### Benchmarks: Best Run

```bash
cargo run --release --features metal -p larql-compute --example best_pipeline       # Full pipeline, 1 cmd buffer
cargo run --release --features metal -p larql-compute --example best_multi_layer     # Multi-layer batch
```

## Documentation

| Doc | Content |
|-----|---------|
| [PERFORMANCE.md](PERFORMANCE.md) | Benchmark data, component profiling, optimization history |
| [ROADMAP.md](ROADMAP.md) | Planned optimizations, performance targets |
| [docs/adr/](docs/adr/) | 12 architectural decision records (design choices, algorithm origins, per-layer params, encoder merging) |
| [docs/shaders.md](docs/shaders.md) | All 44 Metal kernels with origin, performance, parameters |
| [docs/quantization-formats.md](docs/quantization-formats.md) | Q4_0, Q4_K, Q4_KF, Q6_K, Q8_0 format specs |
| [docs/decode-pipeline.md](docs/decode-pipeline.md) | Decode data flow, dual-path architecture, KV cache |

## Design Principles

1. **Trait-based dispatch** — callers use `ComputeBackend` exclusively
2. **One file per kernel** — 32 shader files, each containing related kernels
3. **Zero-copy mmap** — `newBufferWithBytesNoCopy` for weight buffers
4. **Safe by default** — `read_buffer_f32` with bounds checking
5. **Feature-gated** — Metal with `--features metal`, CPU always available
6. **Auto-calibration** — benchmarks CPU vs GPU at startup for routing threshold
7. **Dual-path decode** — auto-detects Q4_K vs Q8 weights, uses optimal pipeline
8. **GGUF-compatible** — Q4_K/Q6_K formats match Ollama's quantization

## License

Apache-2.0
