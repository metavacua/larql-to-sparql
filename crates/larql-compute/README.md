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
Ollama gemma3:4b:       9.7–10.3ms/token = 97–103 tok/s (decode, 34 layers)
LARQL Q4_K (21 layers):     16.9ms/token =  59 tok/s (decode, KV cached)
LARQL Q4_K (34 layers):     27.3ms/token =  37 tok/s (decode, KV cached)
LARQL Q8   (21 layers):     24.3ms/token =  41 tok/s (decode, KV cached)
```

### Component Breakdown (34 layers, isolated — `profile_components`)

| Component | Total | Per-Layer | % | Notes |
|-----------|-------|-----------|---|-------|
| Q4 FFN (gate+up+geglu+down) | 13.0ms | 0.382ms | 36% | Dominant cost |
| KV cache append+attend | 10.5ms | 0.308ms | 29% | kv_attention shader |
| Norms (2×) | 10.5ms | 0.309ms | 29% | Dispatch overhead, not compute |
| **Q4_K QKV fused** | **1.3ms** | **0.037ms** | **3.5%** | **Fast — not the bottleneck** |
| Q4_K O projection | 0.8ms | 0.024ms | 2% | |
| Residual add | 0.3ms | 0.010ms | 1% | |

### Raw Kernel Speed (`profile_raw_dispatch`)

| Kernel | Time | Per-Layer | vs Ollama |
|--------|------|-----------|-----------|
| Q4_K QKV (34L, 1 cmd) | 1.6ms | 0.048ms | **6.3x faster than Ollama's entire layer** |
| Q4_0 v4 matvec [10240,2560] | 0.26ms | — | 57 GB/s production FFN kernel |
| Q8 fused QKV (1 dispatch) | 0.51ms | — | 2.5x vs separate dispatches |

### Path to Parity

The kernel is already faster than Ollama. The gap is in per-dispatch overhead (5–7 encoder creations per layer × 34 layers) and FFN cost. Two paths to close:

1. **Merge dispatches**: norm+QKV+attend+O+FFN in 1 encoder per layer → save ~8ms
2. **Cache L0-12**: compute only 8 entity-dependent layers → 59 × 21/8 = **155 tok/s**

## Shaders (28 Metal kernels)

| Category | Kernels | Production |
|----------|---------|------------|
| f32 matmul | sgemm, sgemm_transb | Tiled 32×32 |
| Q4_0 matvec | v1, v2, v3, **v4** (prod), v5, sparse | v4: uint32 wide loads, 61 GB/s |
| Q4_K/Q6_K | q4k_matvec, q4k_qkv_proj, q4kf_qkv_proj, q6k_matvec | Fused QKV, sub-block lanes |
| Q8 | q8_matvec, q8_qkv_proj, q8_proj_rope | Fused QKV, simdgroup reduction |
| Attention | fused_attention (RoPE+GQA+softcap), causal, kv_attention, kv_append | skip_rope flag for prefill |
| Element-wise | geglu, rms_norm, residual_add, residual_inject, rope, quantize_q8 | |
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
  lib.rs              QuantFormat, QuantWeight, FullPipelineLayer, re-exports
  backend.rs          ComputeBackend trait (15 methods)

  cpu/
    mod.rs            CpuBackend (BLAS f32 + C Q4 + Q4_K/Q6_K reference)
    ops/              f32_matmul, q4_matvec, q4_vecmat, q4k_matvec, q6k_matvec,
                      q4_common (quantizers: Q4_0, Q4_K, Q4_KF, Q6_K, GGUF Q4_K),
                      q8_matvec, vector, attention, geglu

  metal/              (feature-gated: --features metal)
    mod.rs            MetalBackend (28 pipeline states, KV cache)
    trait_impl.rs     ComputeBackend dispatch (Q4_K/Q8 dual-path)
    decode.rs         KV-cached decode (norm→QKV→attend→O→FFN per layer)
    prefill.rs        GPU prefill for seq>1
    buffers.rs        GPU buffer cache + read_buffer_f32
    shaders/          28 Metal kernels (one file each)
    ops/              GPU dispatch helpers

  csrc/q4_dot.c       ARM NEON Q4 kernel
```

## Tests

```bash
# CPU only (38 tests)
cargo test -p larql-compute

# CPU + Metal (74 tests)
cargo test -p larql-compute --features metal
```

74 tests covering: quantization round-trips, cross-backend correctness (Metal vs CPU with tolerance), shader compilation, fused attention, KV cache, pipeline output verification.

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
| [docs/adr/](docs/adr/) | 8 architectural decision records (design choices, algorithm origins) |
| [docs/shaders.md](docs/shaders.md) | All 28 Metal kernels with origin, performance, parameters |
| [docs/quantization-formats.md](docs/quantization-formats.md) | Q4_0, Q4_K, Q4_KF, Q6_K, Q8_0 format specs |
| [docs/decode-pipeline.md](docs/decode-pipeline.md) | Decode data flow, dual-path architecture, KV cache |

## Design Principles

1. **Trait-based dispatch** — callers use `ComputeBackend` exclusively
2. **One file per kernel** — 28 shaders, each in its own file
3. **Zero-copy mmap** — `newBufferWithBytesNoCopy` for weight buffers
4. **Safe by default** — `read_buffer_f32` with bounds checking
5. **Feature-gated** — Metal with `--features metal`, CPU always available
6. **Auto-calibration** — benchmarks CPU vs GPU at startup for routing threshold
7. **Dual-path decode** — auto-detects Q4_K vs Q8 weights, uses optimal pipeline
8. **GGUF-compatible** — Q4_K/Q6_K formats match Ollama's quantization

## License

Apache-2.0
