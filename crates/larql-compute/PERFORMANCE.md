# Performance Tracking — larql-compute

Machine: M3 Max, macOS, Gemma 3 4B (34 layers, hidden=2560, inter=10240, vocab=262K)

## Current State (2026-04-19)

### Synthetic (compare_ollama, random weights, M3 Max)
```
LARQL Q4_KF decode (34 layers, KV cache):   8.5ms = 117 tok/s  ← synthetic ceiling
Ollama gemma3:4b (34 layers):              10.3ms =  98 tok/s
vs Ollama (synthetic):                     0.83x (17% FASTER)
```

### Real vindex (larql bench, gemma3-4b-q4k-v2.vindex, M3 Max, 2026-04-19)
```
Prompt: "The capital of France is" (5 tokens)

  prefill (warm, after KV cache pre-alloc): 67.7ms
  decode (50 tok, 3 warmup discarded):      15.6ms = 64.1 tok/s
  lm_head (Q4_0 synthesized):               2.0ms  (was 4.3ms f16 gemv)
  GPU forward (34 layers):                 14.1ms  (86% of decode)

vs Ollama gemma3:4b:                       ~100 tok/s  (1.56× gap)

Per-stage:
  embed       0.002ms  (0.0%)
  GPU fwd    14.1ms   (86.3%)
  final_norm  0.007ms  (0.0%)
  lm_head     2.0ms   (13.6%)
  detok       0.008ms  (0.1%)
```

### Optimizations applied (2026-04-08 — 2026-04-19)

1. Single command buffer + single global encoder for all 34 layers
2. Batched RoPE + V-norm shaders (16 dispatches → 3 per layer)
3. Q4_K format for FFN (skip Q8 quantize, use q4k_matvec)
4. Fused gate+up kernels (q4k_ffn_gate_up, q4kf_ffn_gate_up)
5. Q4_K matvec rewrite: uint4 loads, 8 rows/TG, multi-row (nr0=2)
6. Q4_KF (GGUF) FFN routing through q4kf_proj (llama.cpp-exact kernel)
7. KV attention: simd_max/simd_sum, float4 Q·K, 1024-entry threadgroup scores
8. Pre-allocated scratch buffers (eliminated ~550 per-decode Metal allocations)
9. **Cooperative SIMD norm reduction** — O(N) reads instead of O(N²). Saved ~10ms.
   All norm kernels (rms_norm, residual_norm, residual_norm_q8) previously had each
   thread redundantly reading ALL elements. Now: stripe + simd_sum + threadgroup reduce.
10. **Q4_0 lm_head synthesis** — synthesized from f16 embeddings at load time. Avoids
    5.6 GB heap clone; lm_head path 4.3ms → 2.0ms (2.2× faster).
11. **KV cache kept on reset** — `reset_kv_cache` now resets `current_len` only; stops
    reallocating ~1.1 GB of GPU buffers on every new prompt.
12. **q4_matvec ROWS_PER_TG=32** — TG memory 9 KB → 2.88 KB (K=2560 exact fit), concurrent
    TGs per core 3 → 11, wave count 273 → ~18.
13. **q6k_matvec ROWS_PER_TG=4** — doubles TG count (320 → 640) for better DRAM utilisation
    on the 2560-row down projection.

## Component Profiling (34 layers, isolated, one command buffer each)

| Component | Total | Per-Layer | % of 36ms | Notes |
|-----------|-------|-----------|-----------|-------|
| **Q4 FFN (gate+up+geglu+down)** | **13.0ms** | **0.382ms** | **35.8%** | Dominant cost. Q4_0 v4 kernel. |
| **KV cache append+attend** | **10.5ms** | **0.308ms** | **28.9%** | kv_attention shader |
| rms_norm | 5.3ms | 0.155ms | 14.5% | Dispatch overhead dominates |
| residual+norm+Q8 fused | 5.2ms | 0.154ms | 14.4% | Fused kernel, still dispatch-bound |
| **Q4_K QKV fused** | **1.3ms** | **0.037ms** | **3.5%** | Fast — NOT the bottleneck |
| Q4_K O projection | 0.8ms | 0.024ms | 2.2% | Small matrix |
| residual add | 0.3ms | 0.010ms | 0.9% | Trivial |
| Empty encoder overhead | 0.05ms | — | 0.0% | Metal API cost is negligible |

**Key finding**: The Q4_K QKV kernel is blazing fast (1.24ms for 34 layers). The bottleneck
is FFN (35.6%) and KV cache (28.9%), plus norm dispatch overhead (29%).

**Next optimization target**: Merge all per-layer operations into fewer compute encoders.
Each `new_compute_command_encoder()` + `end_encoding()` cycle adds ~0.15ms of GPU idle time
for element-wise ops like rms_norm (which finish in microseconds of GPU compute but pay
full dispatch overhead).

## Full Operation Benchmark (M3 Max, latest run 2026-04-07)

| Operation | CPU | Metal | Notes |
|-----------|-----|-------|-------|
| f32 matmul [6,2560]×[2560,2560]^T | 0.69ms | 0.73ms | Attention Q/O proj |
| f32 matmul [6,2560]×[10240,2560]^T | 1.91ms | 1.93ms | FFN gate/up |
| f32 matmul [1,2560]×[262K,2560]^T | 24.7ms | 28.4ms | Logits (CPU wins) |
| Q4_0 matvec [10240,2560] | 1.00ms | 0.69ms | FFN projection |
| Q4_0 vecmat [10240,2560] | 1.35ms | 1.84ms | Down proj (CPU wins) |
| Q4_0 pair batch (6 pos) | 11.6ms | 1.58ms | 7.3x GPU speedup |
| Q4_0 v4 matvec [10240,2560] | — | 0.26ms | 57 GB/s, production |
| Q4_K matvec (via q4k_matvec) | — | ~0.20ms | Standalone Q4_K |
| Q8 fused QKV (1 dispatch) | — | 0.51ms | 2.5x vs separate |
| Q8 fused QKV (21L) | — | 10.6ms | 0.50ms/layer |
| Q4_K fused QKV (34L, 1 cmd) | — | 1.63ms | 0.048ms/layer |
| Multi-layer Q4 FFN (21L, 1 cmd) | — | 8.4ms | Production |
| Full pipeline (21L, attn+FFN) | — | 18.7ms | Q4_K attn + Q4_0 FFN |
| KV cache attend (T=10, 21L) | — | 0.81ms | Sweet spot |
| Full layer (attn+FFN, seq=1) | — | 1.64ms | Per-layer |
| f32 BLAS gemv (warm) | 0.91ms | — | 116 GB/s |
| GEGLU (10240 elements) | 0.015ms | — | Trivial |
| Quantize to Q8 (2560 elements) | 0.002ms | — | Trivial |

## New Kernel Benchmarks (model-agnostic alignment, 2026-04-07)

Isolated dispatch timing (M3 Max). Each kernel dispatched individually — in a fused pipeline, these share
one command buffer and add effectively zero latency.

| Kernel | Time | vs Baseline | Notes |
|--------|------|-------------|-------|
| SiLU standalone (10240) | 305µs | — | Dispatch-dominated |
| GELU-tanh standalone (10240) | 189µs | — | Dispatch-dominated |
| GEGLU SiLU (gated, 10240) | 194µs | — | Comparable to standalone |
| RMSNorm (2560) | 687µs | baseline | Standard norm |
| LayerNorm with bias (2560) | 686µs | 1.00x RMSNorm | No penalty |
| LayerNorm no bias (2560) | 499µs | 0.73x RMSNorm | 27% faster |
| V-norm (256, 1 head) | 181µs | — | Parameter-free RMSNorm |
| V-norm (256, 4 heads) | 723µs | — | Per-head dispatch |
| scale_vector (2560) | 163µs | — | Element-wise multiply |
| Full RoPE (256 dims) | 151µs | baseline | Standard rotation |
| Partial RoPE (64 dims) | 149µs | ~same | Dispatch-dominated at this size |

**Key finding**: All new kernels are dispatch-overhead-dominated. The actual GPU compute is <1µs for element-wise ops.
In the fused decode pipeline, V-norm, layer_scalar, partial RoPE, and LayerNorm add negligible overhead because they share the command buffer with the existing dispatches.

## Ollama Reference

```
gemma3:4b Q4_K_M, Metal GPU:
  Prefill (warm):  15ms / 14 tokens = 925 tok/s
  Decode:          9.7–10.3ms/token = 97–103 tok/s
  RAM:             3.3 GB
  Layers:          34
  Per-layer:       0.303ms (entire layer including QKV + attend + FFN + norms)
```

## Raw Kernel Speed (pure GPU, no pipeline overhead)

| Kernel | Size | Time | Bandwidth | Notes |
|--------|------|------|-----------|-------|
| Q4_K QKV fused (34L, 1 cmd) | 5120 rows × 2560 | 1.63ms | 0.048ms/layer | **6.3x faster than Ollama's entire layer** |
| Q4_K QKV fused (1 dispatch) | 5120 rows × 2560 | 0.30ms | 25.3 GB/s | Single dispatch overhead |
| Q4_0 v4 matvec [10240,2560] | 14.7 MB | 0.26ms | 57 GB/s | Production FFN kernel |
| Q4_0 v4 Q proj [2560,2560] | 7.3 MB | 0.28ms | 53 GB/s | Attention projection |
| Q8 fused QKV (21L, 1 cmd) | 13.1 MB/layer | 10.2ms | 0.49ms/layer | |
| Q8 fused QKV (1 dispatch) | Q+K+V | 0.48ms | — | 2.5x vs 3 separate |
| f32 BLAS gemv [10240,2560] | 105 MB | 0.91ms | 116 GB/s | CPU Accelerate |
| Memory bandwidth (BLAS warm) | 105 MB | 0.91ms | 116 GB/s | M3 Max single-core |
| Memory bandwidth (mmap warm) | 3.6 GB | 3.8ms | 938 GB/s | Unified memory peak |

## Kernel Optimization Journey

### Q4_K QKV Projection (5120 rows × 2560 hidden)

| Variant | attn/21L | Decode | vs Q8 | Technique |
|---------|----------|--------|-------|-----------|
| Q8 fused (baseline) | 18.7ms | 24.6ms | 1.0x | Q8×Q8 integer dot, shared memory |
| Q4_K fused | 10.7ms | 17.5ms | 1.75x | Q4_K struct, uint4 loads, separated dot/xsum |
| + sub-block lanes | 10.4ms | 17.3ms | 1.80x | 80 subs / 32 lanes = 83% utilization |
| + direct device reads | 10.4ms | 17.2ms | 1.80x | No threadgroup memory for input |
| + llama.cpp architecture | 10.4ms | 17.1ms | 1.80x | Register input, 2 rows/sg, quarter-block lanes |
| + GGUF format kernel | 10.4ms | 17.0ms | 1.80x | Exact llama.cpp inner loop |

**Conclusion**: All Q4_K kernel variants converge to ~10.4ms/21L. The inner loop is at
the hardware's limit for this dispatch pattern. The 1.80x speedup vs Q8 comes from smaller
data (7.6MB vs 13.1MB per layer) and eliminating Q8 quantization overhead.

### Approaches Tested and Measured

| Approach | Result | Why |
|----------|--------|-----|
| Half-precision inner loop | No improvement | Not ALU-throughput-bound |
| Integer Q8 inner loop (on-the-fly quantize) | No improvement | Q8 quantization overhead = savings |
| Pre-baked scales (Q4_KF format) | No improvement | Scale decode is <10% of ALU |
| 2 sub-blocks per lane (ILP) | Marginal | Compiler already does this |
| Pre-loaded 128-byte register array | Slower | Register spilling (32 × uint32) |
| simd_shuffle input broadcast | Helps on battery only | Plugged in: parallelism wins |
| Struct-aligned reads (block_q4_K*) | Marginal | Compiler already coalesces |
| Merged norm+QKV encoder | Marginal | Metal encoder overhead is ~0ms |
| llama.cpp exact kernel port | Same speed | Same inner loop = same speed |

## Shader Inventory (44 kernels, all compiled and tested)

| Shader | Type | Status | Notes |
|--------|------|--------|-------|
| sgemm / sgemm_transb | f32 matmul | Production | 32×32 tiled, shared memory |
| q4_matvec v1 | Q4×Q8 | Legacy | Simdgroup + threadgroup |
| q4_matvec v2 | Q4×f32 | Experimental | 4-row variant |
| q4_matvec v3 | Q4×Q8 | Experimental | 8-row unrolled |
| **q4_matvec v4** | Q4×Q8 | **Production** | uint32 wide loads, 61 GB/s |
| q4_matvec v5 | Q4×Q8 | Experimental | 256-row, no simd |
| q4_vecmat | f32×Q4 | Production | Scatter-accumulate |
| q4_f32_matvec | Q4×f32 | Production | Down projection |
| q4_sparse_matvec | Q4×Q8 | Production | Index-based subset |
| **q4k_matvec** | Q4_K×f32 | **Production** | uint4 loads, 8 rows/TG, multi-row (nr0=2) |
| **q4k_qkv_proj** | Q4_K×f32 | **Production** | Fused QKV, sub-block lanes |
| q4kf_qkv_proj | Q4_K×f32 | Production | llama.cpp-exact kernel (GGUF format) |
| q4k_proj / q4kf_proj | Q4_K×f32 | Production | O projection / standalone matvec |
| **q4k_ffn_gate_up** | Q4_K×f32 | **Production** | Fused gate+up, one dispatch, shared input |
| q4k_geglu_silu_down | Q4_K×f32 | Experimental | Fused GEGLU+down (unused — exp() per row too costly) |
| q4k_geglu_gelu_tanh_down | Q4_K×f32 | Experimental | Fused GELU+down (unused — same issue) |
| q6k_matvec | Q6_K×f32 | Production | V projection |
| q8_matvec | Q8×Q8 | Production | Attention projections |
| q8_qkv_proj | Q8×Q8 | Production | Fused QKV (Q8 path) |
| q8_proj_rope | Q8×Q8 | Production | O projection with RoPE |
| geglu_silu | Element-wise | Production | SiLU activation |
| quantize_q8 | f32→Q8 | Production | On-the-fly quantization |
| rms_norm | Element-wise | Production | With configurable offset |
| residual_add | Element-wise | Production | a + b |
| residual_inject | Element-wise | Production | Buffer copy |
| rope_apply | Element-wise | Production | Split-half RoPE, partial rotary_dim |
| fused_attention | GQA | Production | RoPE + partial rotary + QK-norm + softcap + causal |
| causal_attention | Basic | Production | Simple causal (benchmarks) |
| kv_attention | GQA | Production | KV-cached decode |
| kv_cache_append | Buffer | Production | K/V cache update |
| fused_ops (rms_norm_q8, residual_norm, residual_norm_q8) | Fused | Production | Multi-op fusion |
| **silu** | Activation | **Production** | Standalone SiLU (non-gated FFN) |
| **gelu_tanh** | Activation | **Production** | Standalone GELU-tanh (non-gated FFN) |
| **layer_norm** | Normalization | **Production** | Standard LayerNorm with bias (StarCoder2) |
| **layer_norm_no_bias** | Normalization | **Production** | LayerNorm without bias |
| **v_norm** | Normalization | **Production** | Parameter-free RMSNorm on V (Gemma 4) |
| **v_norm_batched** | Normalization | **Production** | All KV heads in one dispatch |
| **rope_at_pos_batched** | Element-wise | **Production** | All Q/K heads in one dispatch |
| **scale_vector** | Element-wise | **Production** | Per-layer scalar multiplier (Gemma 4) |
| turboquant_encode/decode | Experimental | New | WHT + 4-bit quantization |
| graph_walk_knn | Experimental | New | GPU-accelerated gate KNN |

## Test Summary

```
CPU unit tests:      30
Metal shader tests:  46 (compilation + correctness + cross-backend + partial RoPE + new kernels)
Correctness tests:    6 (CPU vs ndarray)
Doc tests:            2
Bench tests:          2
Total:               83 tests (with --features metal), all passing
Warnings:             0
```

### New Shader Tests (model-agnostic compute alignment)

| Test | Verifies |
|------|----------|
| silu_standalone_matches_cpu | SiLU activation without gate multiply |
| gelu_tanh_standalone_matches_cpu | GELU-tanh activation without gate multiply |
| layer_norm_matches_cpu | Standard LayerNorm with bias |
| layer_norm_no_bias_matches_cpu | LayerNorm without bias |
| v_norm_matches_cpu | Parameter-free RMSNorm (Gemma 4 V-norm) |
| scale_vector_matches_cpu | Per-layer scalar multiplier |
| rms_norm_with_different_eps | Verifies eps is parameterized (not hardcoded) |
| new_kernel_functions_exist | All 7 new kernels compile and link |

### Cross-Backend Tests (Metal vs CPU)

| Test | Tolerance | Status |
|------|-----------|--------|
| q4k_matvec_matches_cpu | 0.5 | ✓ |
| q6k_matvec_matches_cpu | 0.3 | ✓ |
| q8_matvec_metal_matches_cpu_ref | 3.0 | ✓ |
| multi_position_q4k_matches_individual | 0.5 | ✓ |
| full_pipeline_seq1_produces_nonzero | — | ✓ |
| sgemm_matches_cpu | 0.1 | ✓ |
| sgemm_transb_matches_cpu | 0.1 | ✓ |
| q4_matvec_matches_cpu | 0.01 | ✓ |
| fused_attention_matches_cpu | 0.1 | ✓ |
| geglu_matches_cpu | 1e-4 | ✓ |
| rms_norm_matches_cpu | 1e-5 | ✓ |

## Safe Buffer Access

All Metal buffer reads go through a single audited function:

```rust
pub fn read_buffer_f32(buf: &metal::Buffer, len: usize) -> Vec<f32>
```

- Null pointer assertion
- Size bounds check
- Immediately copies to Vec (no dangling references)
- Replaces 13 previous `unsafe { from_raw_parts }` call sites

## Architecture

```
larql-compute/
  src/
    lib.rs            QuantFormat, QuantWeight, FullPipelineLayer, re-exports
    backend.rs        ComputeBackend trait (matmul, q4, q4k, q6k, kv, prefill)
    cpu/
      mod.rs          CpuBackend impl
      ops/            f32_matmul, q4_matvec, q4_vecmat, q4k_matvec, q6k_matvec,
                      q4_common (Q4/Q4_K/Q6_K/Q4_KF quantizers), q8_matvec,
                      vector, attention, geglu
    metal/
      mod.rs          MetalBackend struct + pipeline construction
      trait_impl.rs   ComputeBackend impl (dispatches to ops/)
      buffers.rs      GPU buffer cache + read_buffer_f32
      f32_ops.rs      Tiled f32 matmul with GPU/CPU auto-routing
      calibrate.rs    CPU vs GPU crossover threshold
      decode.rs       KV-cached decode pipeline (Q4_K + Q8 dual-path)
      prefill.rs      GPU prefill for seq>1
      pipeline.rs     Legacy full pipeline + multi-layer FFN batch
      direct_ops.rs   Q4 direct dispatch for benchmarks
      shaders/        ~30 Metal shader files (~48 kernels)
      ops/            GPU dispatch helpers (q4_matvec, q4_vecmat, q4_batched,
                      q4_f32_matvec, kv_cache, full_pipeline, full_layer)
  csrc/
    q4_dot.c          ARM NEON Q4 dot product kernel
  tests/
    test_correctness.rs    CPU functional tests (6)
    test_metal_shaders.rs  Metal shader tests (46)
  examples/
    23 organized: 3 demo_, 4 compare_, 10 profile_, 2 best_, 2 test_, 1 arch, 1 tool
  benches/
    matmul.rs         Criterion benchmark
```

## What LARQL Has That Ollama Doesn't

| Feature | Ollama | LARQL |
|---------|--------|-------|
| Editable knowledge | no | yes (vindex patches) |
| Inspectable features | no | yes (gate KNN, walk trace) |
| Adaptive residency | no | yes (pin/evict with memory budget) |
| Template caching | no | yes (0ms for L0-12, proven at 0.999 cosine) |
| GPU prefill pipeline | yes | yes (new: prefill_q4 with KV cache population) |
| Model-aware pipeline | limited | yes (architecture traits drive norms/RoPE/softcap) |
| 70B in 4.9GB | 40GB needed | yes (vindex walk, 88x RAM reduction) |
| Cross-backend tests | no | yes (Metal vs CPU with tolerance) |
| Safe buffer reads | n/a | yes (read_buffer_f32 with bounds checking) |

## Historical Progress

```
Date        Milestone                                    Time      tok/s
2026-04-05  Dense f32 baseline                           534ms     1.9
2026-04-05  + vindex logits KNN                          308ms     3.2
2026-04-05  + cache 13 template layers                   218ms     4.6
2026-04-05  + zero-copy mmap→Metal FFN                    88ms    11.3
2026-04-05  + full Q4 pipeline (approx attn)              13ms    77.7
2026-04-06  + fused_attention shader                     25.9ms    39
2026-04-06  + fused Q8 QKV (1 dispatch for Q+K+V)       18.5ms    54
2026-04-06  + Q4_K fused QKV                             19.2ms    52 (pipeline)
2026-04-06  + Q4_K decode with KV cache                  17.5ms    57
2026-04-07  + sub-block lanes + merged encoders          17.0ms    59
2026-04-07  + GGUF kernel architecture                   17.0ms    59
2026-04-07  Component profiling → FFN is 36% of cost      —        —
2026-04-08  + Q4_K FFN (skip Q8, use q4k_matvec)        24.7ms    40  (34L)
2026-04-08  + fused gate+up kernel                       21.4ms    47  (34L)
2026-04-08  + q4k_matvec uint4 + 8 rows/TG              21.4ms    47  (34L)
2026-04-08  + multi-row nr0=2                            20.8ms    48  (34L)
2026-04-08  + Q4_KF (GGUF) FFN via q4kf_proj            20.5ms    49  (34L)
2026-04-08  + SIMD KV attention reductions               20.5ms    49  (34L)
2026-04-09  + pre-allocated scratch buffers               18.3ms    55  (34L)
2026-04-09  + fused Q4_KF gate+up (q4kf_ffn_gate_up)     18.3ms    55  (34L)
2026-04-09  + cooperative SIMD norm (O(N²)→O(N))           8.5ms   117  (34L, synthetic) ← exceeds Ollama synthetic
2026-04-09  vs Ollama (synthetic): 2.84x → 0.83x (17% faster)
2026-04-18  Real vindex wired (bench_cmd), base ~55 tok/s  15.8ms    63  (34L, real)
2026-04-19  + Q4_0 lm_head synthesis (4.3ms → 2.0ms)      15.6ms    64  (34L, real)
2026-04-19  + KV cache kept on reset (prefill 323ms→68ms)  67.7ms    64  (prefill warm)
2026-04-19  + q4_matvec ROWS_PER_TG=32, TG mem 9KB→2.9KB    —        —
2026-04-19  + q6k_matvec ROWS_PER_TG=4 (320→640 TGs)         —        —
2026-04-19  vs Ollama (real): 1.56x gap (64 vs ~100 tok/s)
```

## Path to Ollama Parity — EXCEEDED (2026-04-09)

Ollama exceeded at 34 layers without caching: 8.5ms / 117 tok/s vs 10.3ms / 98 tok/s.

The final breakthrough: all norm kernels (rms_norm, residual_norm, residual_norm_q8) had
O(N²) memory reads — each of 2560 threads read ALL 2560 elements for sum_sq. Fixing to
cooperative SIMD reduction (stripe + simd_sum + threadgroup reduce) saved ~10ms.

### What worked
| Optimization | Savings | Technique |
|-------------|---------|-----------|
| **Cooperative SIMD norms** | **~10ms** | **O(N²)→O(N) reads. THE fix.** |
| Q4_KF FFN routing | ~8ms | llama.cpp kernel for FFN gate/up/down |
| Q4_K matvec rewrite | ~3ms | uint4 loads, 8 rows/TG, nr0=2 |
| Q4_K format for FFN | ~4.5ms | Skip Q8 quantize step |
| Buffer pre-allocation | ~2ms | Eliminate 550 Metal buffer allocs per decode |
| Fused gate+up kernels | ~1ms | Single dispatch, shared input read |
| Batched RoPE/V-norm | ~0.5ms | 16 dispatches → 3 per layer |
| SIMD KV attention | ~1ms | simd_max/simd_sum, fewer barriers |

### What didn't work
| Approach | Result | Why |
|----------|--------|-----|
| Dispatch merging (single cmd buffer) | ~0ms | Apple Silicon dispatch overhead negligible |
| Memory barriers removal | ~0ms | Dispatches already serialise within encoder |
| 2-sub-block unrolling | Slower | Register pressure, poor tail utilization at K=2560 |
| Fused GEGLU+down kernel | 32x slower | exp() recomputed per output row (26M calls vs 10K) |

### With caching (future)
```
117 tok/s → current (34 layers, all computed, Q4_KF)
~500 tok/s → cache L0-12, compute 8 layers only
              117 × (34/8) ≈ 497 tok/s (theoretical)
```
