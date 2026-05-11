# Performance — larql-vindex

Machine: M3 Max, macOS. Tables below split by audit date — older
sections preserved for diff continuity. The 2026-04-25 audit added
end-to-end Q4K decode numbers (was synthetic-only) plus a confirmed
mmap residency map.

## Perf round-4 (2026-04-25): four shipped wins

End-to-end decode is **86.7 % GPU forward** (lives in `larql-compute`/
`larql-metal`, not vindex). Vindex itself is a thin mmap shim during
real Metal decode. The round-4 audit found four measurable
vindex-side wins; all are shipped, all measured by criterion benches.

### W1. `top_k_from_scores` → bounded min-heap

Replaced the `Vec<(usize, f32)>::select_nth_unstable_by` of size N
with a `BinaryHeap` of capacity K. Allocation drops from O(N) to
O(K) — for Gemma 4B walks (K=10, N=10240), 5.4 MB → 16 KB per token.

| Bench | Before | After | Δ |
|---|---|---|---|
| `gate_knn 4096×512` | 425 µs | 352 µs | **-18 %** |
| `walk 14L×4096×512` | 5.79 ms | 2.20 ms | **-62 %** |
| `gate_knn 10240×2560` | 2.66 ms | 2.65 ms | flat (BLAS dominates) |

`cargo bench -p larql-vindex --bench vindex_ops -- gate_knn_per_layer`

### W2. Feature-major Q4_K down (`down_features_q4k.bin`)

Down-proj is stored `[hidden, intermediate]` on disk, so per-feature
decode requires gathering across `hidden` separate rows. The legacy
path (`q4k_ffn_layer` cache) amortises by dequantising the whole
layer + transposing once. The W2 fix emits a feature-major file at
extract time so per-feature decode is a single row dequant.

| K (active features) | Cache+transpose | Feature-major | Speedup |
|---|---|---|---|
| 100 (sparse) | 77.6 ms | **31.8 µs** | **2440×** |
| 1024 (medium) | 81.7 ms | **325 µs** | **251×** |
| 10240 (full) | 82.9 ms | **3.24 ms** | **25×** |

Numbers are *first-access* — the cache amortises across many calls
to the same layer, so the gap narrows on warm cache. For grid/MoE
shards (each shard touches each layer once or twice; cache never
amortises) feature-major is the operating regime.

Opt-in at extract: `--feature-major-down` on `larql extract-index`
or `larql convert quantize q4k`. Adds ~14 MB / layer to disk on
Gemma 4B; eliminates the ~840 MB heap cache ceiling.

`cargo bench -p larql-vindex --bench q4k_cache -- q4k_down_cache_vs_feature_major`

### W3. Parallel HNSW warmup across layers

`warmup_hnsw_all_layers()` rayon-shards layer builds. Per-layer HNSW
build itself stays serial (algorithm requires it). Side-fix:
`get_or_build_hnsw` no longer holds the cache lock during the ~76 ms
per-layer build, so concurrent KNN on different layers no longer
blocks (matters for grid shards with parallel layer-range routing).

| Bench | Serial | Parallel | Speedup |
|---|---|---|---|
| dense-8L (10240×2560) | 395 ms | 109 ms | **3.6×** |
| moe-4L (32768×2560) | 785 ms | 276 ms | **2.8×** |

Estimated 34-layer Gemma 4B HNSW warmup: ~2.6 s serial → ~700 ms
parallel. Sub-linear in cores because the search-level inner loop is
memory-bound — bounding BLAS to 1 thread inside the rayon pool was
investigated and *slightly hurt* (109 → 113 ms), so no further wins
from BLAS-tuning.

`cargo bench -p larql-vindex --bench hnsw_decode -- hnsw_warmup`

### P2. Parallel batch top-K for prefill

`gate_knn_batch` now `par_iter`s the per-position top-K extraction
when `seq_len ≥ 16`. Decode (seq_len=1) takes the same serial path
as before; prefill paths get the parallel speedup.

| seq_len | Serial (RAYON=1) | Parallel | Δ |
|---|---|---|---|
| 1 (decode) | 2.78 ms | 2.73 ms | flat (below threshold) |
| 64 | 5.42 ms | 5.05 ms | -7 % |
| 256 (typical prefill) | 11.31 ms | 8.56 ms | **-24 %** |

`cargo bench -p larql-vindex --bench vindex_ops -- gate_knn_batch`

## CPU vs GPU comparison (2026-04-26, M3 Max)

Side-by-side at production gate-matrix shapes. Same operation, same
inputs, both backends. CPU goes through Apple Accelerate (BLAS);
Metal goes through `larql-compute`'s shaders (`f32_gemv_force` for
decode, `matmul_transb` MPS path for prefill, `q4_matvec` for the
Q4-decode hot path).

| Op | Shape | CPU (Accelerate) | Metal | Speedup |
|---|---|---|---|---|
| f32 gemv (decode) | gemma-3-4b 10240×2560 | 2.09 ms | **525 µs** | **4.0×** |
| f32 gemv (decode) | llama-3-8b 14336×4096 | 3.08 ms | **878 µs** | **3.5×** |
| f32 matmul (seq64 prefill) | gemma-3-4b 10240×2560 | 4.06 ms | **3.11 ms** | **1.3×** |
| f32 matmul (seq64 prefill) | llama-3-8b 14336×4096 | 9.63 ms | **5.55 ms** | **1.7×** |
| Q4 matvec (decode, production hot path) | gemma-3-4b 10240×2560 | 1.17 ms | **496 µs** | **2.4×** |
| Q4 matvec (decode, production hot path) | llama-3-8b 14336×4096 | 2.86 ms | **850 µs** | **3.4×** |

Notes:
- **Metal wins everywhere on single-position decode** — the Apple
  Silicon GPU's bandwidth advantage compounds with the dispatch
  cost being amortised across many large matvec calls per token.
- **Prefill speedup is smaller** because Accelerate's GEMM is already
  near memory-bandwidth-bound at seq_len=64 — the GPU still wins
  but by a smaller margin.
- **Q4 decode is the production path for `larql-inference`** —
  `q4k_matmul_transb` streams Q4_K bytes from mmap straight into
  Metal shaders. The 2.4–3.4× margin matches the older
  Q4-Metal-vs-f32-BLAS numbers in the "Q4 Gate KNN" table below
  but with newer kernels (Metal Q4 Gemma 4B was 0.96 ms in
  2026-04-19; now 496 µs — a further 1.9× from kernel tuning).
- Scaling bench is **CPU-only**. The dedicated `vindex_scaling.rs`
  bench measures CPU through the full `gate_knn` pipeline; this
  bench measures the raw compute kernel both ways.

`cargo bench -p larql-vindex --features metal --bench cpu_vs_gpu`

## End-to-end decode (2026-05-02, real Q4K Gemma 3 4B)

`larql bench output/gemma3-4b-q4k-v2.vindex --tokens 30 --warmup 8 --backends metal`
with all five 2026-05 dispatch fusions default-on (qk_norm_rope,
kv_append_attend, post_attn_residual_norm_store, post_ffn_norm_residual_add)
plus the lm_head v5 stride-32 correctness fix:

| Backend | tok/s | ms/tok | GPU fwd | lm_head | Peak footprint |
|---------|-------|--------|---------|---------|----------------|
| metal   | **72–75** | 13.5–13.9 | 11.5–12.0 ms (79%) | 2.9–3.0 ms (20%) | 6.59 GB |
| cpu     |   0.4 | 2787 | 2777 ms | — | 3.70 GB |

The 72–75 tok/s reading is the **honest** number — it incorporates the
lm_head v5 correctness fix (the model now emits "Paris" rather than
gibberish; the fix added ~0.7 ms to lm_head). Pre-fix benches showing
78–80 tok/s ran on incorrect output and are not comparable. Cumulative
2026-05 fusion saving: -0.99 ms GPU forward vs. unfused baseline.

GPU forward is now 79% of decode (down from 86.7% pre-lm_head-fix);
kernel-compute work and the lm_head matvec are roughly equal levers.
Path-to-80 documented in `crates/larql-inference/ROADMAP.md` G-3.

## mmap residency (live decode pid, vmmap)

Real Q4K Gemma 3 4B during decode:

```
File                              VSIZE   RSDNT   madvise
gate_vectors.bin            1.7 GB     0 K   RANDOM       ← pure demand-paged
down_meta.bin                29 M    544 K   RANDOM       ← only touched layers paged
embeddings.bin              1.3 G    1.3 G   SEQ+WILLNEED ← prefaulted
interleaved_q4k.bin         1.6 G    1.6 G   RANDOM (warmed by decode)
attn_weights_q4k.bin       309 M    309 M   SEQ+WILLNEED
heap (MALLOC_LARGE)          3.0 G   3.0 G   ← KV cache + GPU intermediates
                             ─────
Physical footprint            3.1 G   (peak 3.4 G)
```

The 3.0 GB MALLOC_LARGE is **not** the Q4K dequant cache — confirmed
by `larql bench -v` reporting `q4k_ffn_cache after larql-metal: 0
populated slots, 0.0 MB`. The Metal full-K fast path streams Q4_K
bytes through `q4k_matmul_transb` and bypasses the dequant cache
entirely. The cache only fires on the CPU per-position fallback (where
it's a 30× win because one 614 ms layer-dequant is amortised across
many feature reads).

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
