# larql-compute

Hardware-accelerated compute backends for LARQL. CPU (BLAS + NEON Q4), Metal GPU, and future CUDA.

## What it does

Provides a `ComputeBackend` trait that abstracts all hardware-specific matrix operations. Every LARQL crate (inference, vindex) uses this trait — the caller never knows whether the operation runs on CPU or GPU.

## Backends

| Backend | Feature flag | f32 matmul | Q4 fused ops | Multi-layer pipeline |
|---------|-------------|------------|--------------|---------------------|
| **CPU** | (always) | BLAS (Accelerate AMX) | C kernel (ARM vdotq_s32) | Sequential |
| **Metal** | `--features metal` | Tiled compute shaders | Simdgroup Q4×Q8 (v4 kernel) | One command buffer |
| **CUDA** | (planned) | — | — | — |

## Performance vs Ollama (M3 Max, Gemma 3 4B)

Ollama gemma3:4b Q4_K_M: **10.1ms/token = 99 tok/s** (decode, warm)

```
Operation                      CPU         Metal       Ollama      Notes
─────────────────────────────  ──────────  ──────────  ──────────  ──────
Q4 matvec [10240,2560]         0.93ms      0.57ms      ~0.4ms     FFN projection (1.4x)
Q4 pair batch (6 pos)          11.42ms     1.54ms      —          gate+up fused (7x)
Q4 vecmat [10240,2560]         1.30ms      1.68ms      —          down projection
Multi-layer Q4 FFN (21L)       —           8.4ms       ~5ms       one cmd buffer (1.7x)
Full pipeline (21L, norms)     —           25.9ms      ~10ms      attn+FFN+norms+residuals
f32 logits [262K,2560]         24.0ms      28.4ms      —          f32 BLAS
Q4 logits [262K,2560]          —           0.57ms      ~1ms       FASTER than Ollama
f32 attn proj [2560²]          0.68ms      1.11ms      —          CPU BLAS wins (small)
Gate KNN [10240,2560]          0.91ms      0.90ms      —          vindex scoring
```

**Component parity with Ollama.** The remaining gap is pipeline integration:
- Ollama's full pipeline is tighter (C++ vs Rust dispatch overhead)
- Ollama uses Q4_K_M/Q6_K (group scaling) vs our Q4_0/Q8_0 (per-block scaling)
- Our Q4 logits are **faster** than Ollama (0.57ms vs ~1ms)

## Shaders (22 Metal kernels, all compiled and tested)

| Shader | Purpose | Status |
|--------|---------|--------|
| `sgemm` | f32 tiled matmul C=A×B | production |
| `sgemm_transb` | f32 tiled matmul C=A×B^T | production |
| `q4_matvec` | Q4×Q8 simdgroup (v1) | production |
| `q4_matvec_v2` | Q4×f32, 4 rows/thread | experimental |
| `q4_matvec_v3` | Q4×f32, 8 rows unrolled | experimental |
| `q4_matvec_v4` | Q4×Q8 uint32 wide loads (**production**, 0.69ms) | production |
| `q4_matvec_v5` | Q4×Q8, 256 rows/TG | experimental |
| `q4_vecmat` | Q4 scatter-accumulate | production |
| `q4_f32_matvec` | Q4×f32 for transposed down | production |
| `q4_sparse_matvec` | Sparse Q4 by index (walk) | production |
| `q8_matvec` | Q8×Q8 (V projection) | production |
| `geglu_silu` | Element-wise SiLU gate | production |
| `quantize_q8` | f32→Q8 (layer chaining) | production |
| `residual_copy/add/rms_norm` | Buffer ops | production |
| `causal_attention` | Basic causal (seq≤64) | production |
| `kv_attention` | KV-cached GQA decode | production |
| `kv_cache_append` | Append K/V to cache | production |
| `rope_apply` | Rotary position embeddings | **new**, tested |
| `fused_attention` | Full GQA: RoPE+QK-norm+softcap+causal | **new**, tested |

## Quick start

```rust
use larql_compute::{ComputeBackend, default_backend, cpu_backend};

// Auto-detect best backend (Metal if available, else CPU)
let backend = default_backend();
println!("Using: {} ({})", backend.name(), backend.device_info());

// f32 matmul
let c = backend.matmul_transb(a.view(), b.view());

// Q4 fused operations
if backend.has_q4() {
    let scores = backend.q4_matvec(&q4_data, &q8_x, &q8_scales, rows, hidden);
}

// Multi-layer Q4 FFN (one command buffer, 8.6ms for 21 layers)
let result = backend.multi_layer_q4_ffn(&layers_q4, &x, inter, hidden);

// Full pipeline: attention + FFN for all layers (10.5ms for 21 layers)
let result = backend.full_pipeline_q4(&layers, &x, hidden, inter, q_dim, kv_dim);

// Vector operations (BLAS-backed)
use larql_compute::{dot, norm, cosine};
let similarity = cosine(&a_vec.view(), &b_vec.view());

// Q4 quantization utility
use larql_compute::cpu::q4::quantize_q4_0;
let q4_data = quantize_q4_0(&f32_weights);
```

## Architecture

```
src/
  lib.rs                    — crate root, ComputeBackend trait, factory functions
  backend.rs                — trait definition + helper functions

  cpu/
    mod.rs                  — CpuBackend struct + trait impl
    ops/
      f32_matmul.rs         — BLAS sgemm/sgemm_transb       (3 tests)
      q4_matvec.rs          — C kernel Q4×Q8 matvec          (2 tests)
      q4_vecmat.rs          — C kernel Q4 vecmat             (2 tests)
      q4_common.rs          — Q4/Q8 quantize, C FFI decls    (7 tests)
      q8_matvec.rs          — Q8 matvec + weight quantizer   (2 tests)
      vector.rs             — dot, norm, cosine similarity    (6 tests)
      geglu.rs              — SiLU gate activation            (3 tests)
      attention.rs          — Causal attention (fused QKV)    (3 tests)

  metal/                    (feature-gated: --features metal)
    mod.rs                  — MetalBackend struct + trait impl
    shaders/                — 20 Metal Shading Language kernels (one file each)
    ops/                    — GPU dispatch modules (one file each)
    buffers.rs              — GPU buffer cache (zero-copy mmap)
    calibrate.rs            — CPU vs GPU auto-calibration
    f32_ops.rs              — f32 dispatch with GPU/CPU routing

  csrc/
    q4_dot.c                — C kernel: ARM vdotq_s32 + scalar fallback
```

## Tests

```bash
# CPU tests (28 unit + 6 integration + 2 doc = 36 tests)
cargo test -p larql-compute

# CPU + Metal tests (64 tests)
cargo test -p larql-compute --features metal
```

Test coverage:
- Q4 quantize: output size, zero input, round-trip accuracy, alignment, end-to-end matvec
- Q4 matvec: CPU kernel, Metal v4, zero input, small matrix, Metal vs CPU
- Q4 vecmat: CPU kernel, Metal, zero activation
- Q4 sparse: Metal sparse matches dense at selected indices
- Q8 matvec: CPU kernel, Q8 vs f32 cosine > 0.999, Metal nonzero
- Vector ops: dot product, norm, cosine similarity (identical, orthogonal, opposite)
- GEGLU: SiLU basic, Metal vs CPU cross-validation
- Residual: Metal add correctness
- Attention: single token, causal mask, output shape
- RoPE: Metal shader matches CPU reference implementation
- Fused attention: single token GQA with RoPE, finite nonzero output
- Batch: Metal pair_batch matches individual calls
- Multi-layer: 21-layer pipeline produces output (zero-copy)
- Shader compilation: all 20 kernel functions exist
- Buffer cache: pointer reuse, zero-copy mmap
- Trait dispatch: Metal implements ComputeBackend correctly

## Benchmarks

```bash
# All operations at representative sizes (CPU + Metal side by side)
cargo run --release -p larql-compute --features metal --example bench_full

# Full 21-layer pipeline (all Q4, one submission) — compare with Ollama
cargo run --release -p larql-compute --features metal --example bench_full_pipeline

# Kernel variant comparison (v1-v5 + sparse)
cargo run --release -p larql-compute --features metal --example bench_kernel_variants

# Q4 attention projections (single + 21-layer)
cargo run --release -p larql-compute --features metal --example bench_q4_attention

# Token generation with KV cache
cargo run --release -p larql-compute --features metal --example bench_generation

# Raw memory bandwidth test
cargo run --release -p larql-compute --example bench_bandwidth -- <file>

# Verify all 20 shaders compile
cargo run --release -p larql-compute --features metal --example test_shader_compile

# Criterion statistical benchmarks
cargo bench -p larql-compute --bench matmul
```

## Design principles

1. **One file per operation** — every shader and dispatch function lives in its own file
2. **Trait-based dispatch** — callers use `ComputeBackend` exclusively
3. **Zero-copy for mmap** — `newBufferWithBytesNoCopy` on Apple Silicon unified memory
4. **Cached vs transient** — weight buffers cached by pointer, input/output allocated fresh
5. **Feature-gated** — Metal with `--features metal`, CPU always available
6. **Auto-calibration** — Metal benchmarks CPU vs GPU at startup
7. **Batch API** — multi-layer pipeline encodes all ops in one command buffer
8. **Shared utilities** — `quantize_q4_0` and `quantize_to_q8` public, no duplication
9. **Mixed precision** — Q4 for projections + FFN, Q8 for V, f32 for attention scores

## License

Apache-2.0
