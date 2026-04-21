# Performance — larql-vindex

Machine: M3 Max, macOS. All numbers from fresh runs (2026-04-07).

## Core Operations (synthetic, 1024 features × 256 hidden, 8 layers)

| Operation | Time | Notes |
|-----------|------|-------|
| Gate KNN (1 layer) | 0.026ms | BLAS matmul_transb |
| Walk (8 layers, top-10) | 0.240ms | Per-feature down vector lookup |
| Feature lookup | 267ns | HashMap O(1) |
| Save gates (8 MB) | 1.4ms | Binary write |
| Save down_meta (8K records) | 0.5ms | 384 KB binary |
| Load vindex (mmap) | 1.1ms | Zero heap for gates |
| Mutate (meta + gate) | 952ns | In-memory overlay |
| Checksum (SHA256) | 19.9ms | Integrity verification |
| Build (8K features) | 2.7ms | In-memory index construction |

## Production-Scale KNN (f32 BLAS, per-layer)

| Model | Features | Hidden | Gate MB | KNN/layer | Walk 14L |
|-------|----------|--------|---------|-----------|----------|
| Gemma 3 4B | 10,240 | 2,560 | 100 MB | 2.75ms | 38.4ms |
| Llama 3 8B | 14,336 | 4,096 | 224 MB | 20.8ms | 332.8ms |
| Llama 3 70B | 28,672 | 8,192 | 896 MB | 107.0ms | 5137ms |
| Mixtral 8x22B (1 expert) | 16,384 | 6,144 | 384 MB | 35.4ms | 1134ms |

## MoE Scaling

| Config | Total Features | Gate MB | KNN/layer |
|--------|---------------|---------|-----------|
| Dense (10240) | 10,240 | 100 MB | 2.89ms |
| 8 experts × 2048 | 16,384 | 160 MB | 11.4ms |
| 16 experts × 2048 | 32,768 | 320 MB | 28.4ms |
| 64 experts × 2048 | 131,072 | 1,280 MB | 162ms |

## Q4 Gate KNN (with larql-compute Metal backend)

| Model | f32 BLAS | Q4 Metal | Gate Q4 MB | Speedup |
|-------|----------|----------|-----------|---------|
| Gemma 3 4B | 2.68ms | 0.96ms | 14.1 MB | 2.8x |

## HNSW vs Brute-Force (dim=2560, M3 Max)

| Features | Brute | HNSW | Build | Winner |
|----------|-------|------|-------|--------|
| 1,024 | 0.190ms | 0.152ms | 4.3ms | HNSW |
| 4,096 | 3.34ms | 2.66ms | 30.7ms | HNSW |
| 10,240 | 2.95ms | 1.70ms | 46.7ms | HNSW |
| 28,672 | 19.7ms | 15.5ms | 158ms | HNSW |

## Mmap Zero-Copy Verified

```
gate_vectors.bin: 544 MB on disk (34 layers × 4096 features × 1024 hidden)
RSS after load:   delta = 544.8 MB (100% — mmap maps entire file)
Gate heap:        0 bytes (zero-copy confirmed)
Query L13:        0.301ms, RSS delta 0 MB (page already in)
Walk L14-27:      2.8ms, RSS delta 0.2 MB (41% of layers paged)
Page fault:       0.064ms overhead on first cold access
```

## Memory Projections (f16 storage, mmap)

| Model | Full RAM | Vindex RAM | Ratio |
|-------|----------|------------|-------|
| Gemma 3 4B | 7 GB | 1.3 GB | 5x |
| Llama 3 8B | 15 GB | 2.2 GB | 7x |
| Llama 3 70B | 130 GB | 4.9 GB | 27x |
| Llama 3 405B | 754 GB | 8.6 GB | 88x |
| Mixtral 8x22B | 263 GB | 4.8 GB | 55x |
| DeepSeek V3 | 1,250 GB | 10.9 GB | 115x |
| Kimi-K2 | 1,863 GB | 10.9 GB | 171x |

**A 1T model in 10.9 GB on a laptop.**

## LM Head Dispatch (2026-04-19)

`lm_head_knn_backend` tries three paths in order:

| Path | Trigger | Latency | Notes |
|------|---------|---------|-------|
| Q4_0 matvec (mmap) | `lm_head_q4.bin` present | ~1ms | Explicit Q4 file |
| Q4_0 matvec (synth) | tied-embed model + no Q4 file | ~2ms | Synthesized at load from f16 embeddings |
| f16 gemv | tied-embed, Metal available | ~4ms | Avoids 5.6 GB f32 clone |
| f32 BLAS fallback | all else | ~25ms | CPU only |

**Synthesis**: `synthesize_lm_head_q4()` converts the `embeddings.bin` f16 mmap to Q4_0 in RAM
at load time (one-time ~2s on Gemma 3 4B, then amortized). Reduces lm_head from 4.3ms to 2.0ms
on M3 Max (2.2× speedup). The synthesized bytes are `vocab × (hidden/32 × 18)` = ~377 MB.

## Connection to larql-compute

Vindex stores raw quantized bytes. Compute kernels dequant + multiply at inference.

| Vindex Operation | Compute Method | Format |
|------------------|---------------|--------|
| Gate KNN (f32) | `matmul_transb` | f32 BLAS |
| Gate KNN (Q4) | `q4_matvec` | Q4_0 |
| LM head KNN (mmap) | `q4_matvec` | Q4_0 |
| LM head KNN (synth) | `q4_matvec` | Q4_0 synthesized at load |
| LM head KNN (f16) | `f16_gemv` | f16 mmap |
| K-means clustering | `matmul_transb` | f32 BLAS |
| HNSW projection | `matmul` | f32 BLAS |
| MoE routing | `matmul` | f32 BLAS |
| Attention weights | stored as Q4_K/Q6_K/Q8 | Read by compute pipeline |
| FFN interleaved | stored as Q4_0 | Read by compute pipeline |

**Build pipeline uses `larql_compute::cpu::ops::q4_common` quantizers as single source of truth.** This ensures the format matches what Metal/CPU kernels expect.

## Format Alignment (verified 2026-04-07)

```
Vindex Storage              → Inference Wiring         → Compute Kernel
──────────────                ────────────────           ──────────────
attn_weights_q4k.bin (Q4_K) → QuantFormat::Q4_K       → q4k_qkv_proj shader    ✅
attn_weights_q4k.bin (Q6_K) → QuantFormat::Q6_K       → q6k_matvec shader      ✅
interleaved_q4k.bin  (Q4_K) → QuantFormat::Q4_K       → Q4_K FFN dispatch      ✅ NEW
interleaved_q4.bin   (Q4_0) → QuantFormat::Q4_0       → q4_matvec_v4 (fallback)✅
lm_head_q4.bin       (Q4_0) → q4_matvec                                        ✅
embeddings.bin (f16) → synthesize_lm_head_q4() → Q4_0 in RAM → q4_matvec     ✅ NEW
gate_vectors_q4.bin  (Q4_0) → q4_matvec                                        ✅

Inference auto-selects: Q4_K FFN preferred → Q4_0 fallback
Quantizers: all from larql_compute (single source of truth, ADR-008)
```
