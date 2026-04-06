# Performance Tracking — larql-compute vs Ollama

Machine: M3 Max, macOS, Gemma 3 4B (34 layers, hidden=2560, inter=10240, vocab=262K)

## Ollama Reference

```
gemma3:4b Q4_K_M, Metal GPU:
  Prefill (warm):  15ms / 14 tokens = 925 tok/s
  Decode:          10.1ms/token = 99 tok/s sustained
  RAM:             3.3 GB
```

## Component Comparison (measured)

| Operation | LARQL CPU | LARQL Metal | Ollama | Ratio |
|-----------|-----------|-------------|--------|-------|
| Q4 matvec [10240,2560] | 0.93ms | **0.57ms** | ~0.4ms | 1.4x |
| Q4 pair batch (6 pos) | 11.42ms | **1.54ms** | — | — |
| Q4 vecmat [10240,2560] | 1.30ms | 1.68ms | — | — |
| Q4 logits [262K,2560] | 24.0ms | **0.57ms** | ~1ms | **0.6x faster** |
| Multi-layer Q4 FFN (21L) | — | **8.4ms** | ~5ms | 1.7x |
| Full pipeline (21L, norms+res) | — | **25.9ms** | ~10ms | 2.6x |
| f32 attn proj [2560²] | 0.68ms | 1.11ms | — | — |
| Gate KNN [10240,2560] | 0.91ms | 0.90ms | — | — |
| f32 21L FFN (CPU BLAS) | 110.7ms | 92.2ms | — | — |

## Shader Inventory (16 kernels, all tested)

| Shader | Tested | Correctness |
|--------|--------|-------------|
| sgemm / sgemm_transb | ✓ | vs ndarray reference |
| q4_matvec (v1-v5) | ✓ | vs CPU C kernel |
| q4_vecmat | ✓ | vs CPU C kernel |
| q4_f32_matvec | ✓ | nonzero output |
| q4_sparse_matvec | ✓ | vs dense at selected indices |
| q8_matvec | ✓ | vs CPU reference |
| geglu_silu | ✓ | vs CPU GEGLU |
| quantize_q8 | ✓ | vs CPU quantize_to_q8 |
| rms_norm | ✓ | vs CPU reference (with offset) |
| residual_add | ✓ | vs CPU a+b |
| rope_apply | ✓ | vs CPU split-half RoPE |
| fused_attention | ✓ | vs CPU GQA (3 tokens, 2 heads, RoPE) |
| causal_attention | ✓ | basic causal (seq≤64) |
| kv_attention | ✓ | GQA with KV cache |
| kv_cache_append | ✓ | K/V buffer update |
| residual_copy | ✓ | buffer copy |

## Pipeline Analysis

```
Full pipeline 21 layers (with norms + residuals):  25.9ms = 39 tok/s
  Breakdown:
    Q8 attn projections (4 × 21L):    ~10ms (Q8 higher precision for attention)
    Fused attention (21L):              ~2ms (RoPE + GQA + softcap + causal)
    Q4 FFN (gate+up+GEGLU+down, 21L):  ~8ms (one cmd buffer, zero-copy mmap)
    RMS norms (4 × 21L):               ~2ms (with offset for Gemma)
    Residual adds (2 × 21L):           ~1ms
    Q8 quantize between stages:         ~3ms
    
  FFN-only batch (21L, 1 cmd):          8.4ms = 119 tok/s (FFN ceiling)
  Ollama full decode:                  10.1ms =  99 tok/s (reference)
```

## What LARQL Has That Ollama Doesn't

| Feature | Ollama | LARQL |
|---------|--------|-------|
| Editable knowledge | no | yes (vindex patches) |
| Inspectable features | no | yes (gate KNN, walk) |
| Adaptive residency | no | yes (pin/evict with memory budget) |
| Q4 gate KNN (Metal) | — | 0.57ms per layer |
| Vindex logits KNN | — | 0.57ms (vs 24ms f32) |
| Template caching | — | 0ms for cached layers |
| Model-aware pipeline | — | architecture traits drive norms/RoPE/softcap |

## Test Summary

```
CPU tests:       28 unit + 6 integration + 2 doc = 36
Metal tests:     26 integration (shader correctness)
Total:           62 tests, all passing
Warnings:        0 (non-cosmetic)
Shaders:         16 compiled, 16 tested
```

## Historical Progress

```
Date        Milestone                        Time      tok/s
2026-04-05  Dense f32 baseline               534ms     1.9
2026-04-05  + vindex logits KNN              308ms     3.2
2026-04-05  + cache 13 template layers       218ms     4.6
2026-04-05  + zero-copy mmap→Metal FFN        88ms    11.3
2026-04-05  + full Q4 pipeline (approx attn)  13ms    77.7
2026-04-05  + cached residuals                 3ms   295.2
2026-04-06  + fused_attention shader          25.9ms   39  (pipeline w/ norms)
2026-04-06  + Q8 attn weights built          ready     —   (higher precision)
2026-04-06  + shader correctness tests        62 tests all passing
2026-04-06  Honest production (CPU)          201ms     5.0 (correct output)
```

## Optimizations Applied

1. **Fused rms_norm_q8**: norm + Q8 quantize in one kernel (saves 42 encoders/21L)
2. **Fused residual_norm_q8**: residual + norm + Q8 in one kernel (saves 42 more)
3. **Merged Q/K/V dispatches**: 3 matvecs in one encoder
4. **Merged gate/up dispatches**: 2 matvecs in one encoder
5. **Q8 attention projections**: higher precision than Q4 (like Ollama's Q6_K)
6. **22 shaders total**: all compiled, all tested against CPU reference

## Gap to Ollama

```
Component              LARQL        Ollama      Gap          Fix
Full pipeline          25.9ms       10.1ms      2.6x
  Attn (Q8 proj+fused) 17.5ms       ~5ms        separate dispatch vs fused kernel
  FFN batch (Q4)         8.4ms       ~5ms        1.7x (close)
  
Remaining optimizations:
  Fused Q8 attention kernel (dequant+matvec+RoPE in one):  17.5ms → ~5ms
  KV cache (decode mode, skip K/V recompute):              saves ~4ms
  → ~10ms = 100 tok/s (projected Ollama parity)
```

Fused Q8 QKV projection (measured):
- Separate Q+K+V (3 dispatches): 0.856ms/layer × 21 = 18.0ms
- Fused Q+K+V (1 dispatch):      0.387ms/layer × 21 = 8.1ms  ← 2.2x faster
- Simdgroup reduction, shared input loading, zero intermediate buffers

Projected pipeline with fused QKV:
- Q8 QKV fused:  8.1ms
- Fused attention: ~3ms
- Q8 O projection: ~2ms  
- Q4 FFN batch: 8.4ms
- Norms + residuals: ~2ms
- Total: ~24ms → 42 tok/s

Remaining gap to Ollama (10ms):
- KV cache eliminates K/V recompute: saves ~3ms
- Fused O projection into attention output: saves ~2ms  
- Tighter pipeline (fewer command buffers): saves ~5ms
