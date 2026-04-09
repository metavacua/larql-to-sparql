# Metal Shader Reference — larql-compute

~48 Metal Shading Language kernels across ~30 shader files in `src/metal/shaders/`.
All compiled into a single Metal library via `all_shaders()`.

## f32 Matrix Multiply

### sgemm.rs — `sgemm`
**C = A × B** (row-major). 32×32 tiled with threadgroup shared memory.
Grid: `(ceil(N/32), ceil(M/32), 1)`, TG: `(32, 32, 1)`.

### sgemm_transb.rs — `sgemm_transb`
**C = A × B^T**. Same tiling strategy. Used for all projection matmuls (Q/K/V/O, FFN gate/up).

## Q4_0 Quantized Matvec (4-bit, 18 bytes per 32 values)

### q4_matvec.rs — `q4_matvec` (v1)
Simdgroup + threadgroup shared memory for Q8 input. Baseline implementation.
Origin: LARQL original.

### q4_matvec_v2.rs — `q4_matvec_v2`
4 rows per thread, f32 input. Experimental variant.

### q4_matvec_v3.rs — `q4_matvec_v3`
8 rows unrolled. Slower due to register spilling. Experimental.

### q4_matvec_v4.rs — `q4_matvec_v4` (PRODUCTION)
**The fast Q4_0 kernel.** uint32 wide loads (4 bytes → 8 nibbles), Q8 input in threadgroup memory, integer multiply-accumulate, simd_sum reduction. 57-61 GB/s on M3 Max.
Origin: LARQL original, iterative optimization from v1-v3.

```
Performance: 0.26ms for [10240, 2560] = 14.7MB (57 GB/s)
Technique: NIBBLE(w, shift) macro extracts nibbles via bitshift
Grid: 8 rows per TG, 256 threads (8 simdgroups × 32 lanes)
```

### q4_matvec_v5.rs — `q4_matvec_v5`
256 rows per TG, no simd. Same speed as v4. Experimental.

### q4_vecmat.rs — `q4_vecmat`
**out[K] = activation[N] @ Q4[N,K]**. Scatter-accumulate pattern (one thread per output element). Used for down projection alternatives.

### q4_f32_matvec.rs — `q4_f32_matvec`
**out[N] = Q4[N,K] @ f32_x[K]**. Takes f32 input directly (no Q8 quantization). Used for down projection with transposed weights and GEGLU activation output.

### q4_sparse_matvec.rs — `q4_sparse_matvec`
Sparse Q4 matvec by index — only computes selected rows. Used by vindex walk architecture for feature-selective FFN.

## Q4_K Quantized (Ollama-compatible, 148 bytes per 256 values)

### q4k_matvec.rs — `q4k_matvec` (PRODUCTION for Q4_K FFN down)
Standalone Q4_K matvec. uint4 vectorized loads, sub-block striped across lanes, multi-row processing (nr0=2: each simdgroup computes 2 output rows, amortizing input reads via L1 cache). 8 rows per TG (4 simdgroups × 2 rows).
Origin: LARQL original, rewritten 2026-04-08 with uint4+nr0 optimizations.

### q4k_qkv_proj.rs — `q4k_qkv_proj` (PRODUCTION for Q4_K attention)
**Fused Q+K+V projection.** Single dispatch for all three attention projections. Rows 0..q_rows → Q, q_rows..+k_rows → K, rest → V. Sub-block lane assignment (80 sub-blocks / 32 lanes = 83% utilization). No threadgroup memory for input — direct device reads.
Origin: LARQL original. Sub-block iteration is novel; direct device reads inspired by llama.cpp.

```
Performance: 1.5ms for 34 layers (0.045ms/layer) — 6.7x faster than Ollama's entire layer
Grid: 8 rows per TG, 256 threads
```

### q4kf_qkv_proj.rs — `q4kf_qkv_proj`, `q4kf_proj` (PRODUCTION for Q4_KF attention + FFN)
Fused Q+K+V and standalone projection using llama.cpp's kernel_mul_mv_q4_K_f32 architecture. Register-based input (yl[16], yh[16]), quarter-block lane decomposition (ix=lane/8), uint16_t nibble extraction, FOR_UNROLL pragma. Operates on GGUF 144-byte blocks.
`q4kf_proj` is also used for **FFN gate, up, and down** when weights are Q4_KF format — this is the key kernel for Ollama parity.
Origin: Ported from llama.cpp (MIT license), adapted for fused QKV and standalone projection.

```
Performance: Same as q4k_qkv_proj for QKV; FFN routing enables ~1.25x Ollama
Grid: 4 rows per TG (2 SG × 2 rows/SG), 64 threads
```

### q4kf_ffn_gate_up.rs — `q4kf_ffn_gate_up` (PRODUCTION for Q4_KF FFN)
Fused gate+up for GGUF format. Same llama.cpp inner loop as q4kf_proj, with both matrices dispatched in one call. Layout: first half of threadgroups → gate, second half → up.
Origin: LARQL original (2026-04-09).

### q4k_ffn_gate_up.rs — `q4k_ffn_gate_up` (PRODUCTION for Q4_K FFN)
Fused gate+up projection — two Q4_K matvecs sharing the same input vector in one dispatch. Threadgroups 0..N handle gate rows, N..2N handle up rows. Same uint4+sub-block inner loop as q4k_matvec.
Origin: LARQL original (2026-04-08).

### q4k_geglu_down.rs — `q4k_geglu_silu_down`, `q4k_geglu_gelu_tanh_down`
Experimental fused GEGLU activation + Q4_K down projection. Computes SiLU(gate)×up on-the-fly during the down matmul. **Not used in production** — recomputing exp() per output row (26M calls for hidden=2560 × inter=10240) is 32x slower than separate GEGLU + matmul.
Origin: LARQL original (2026-04-08). Kept for documentation of the failed experiment.

## Q6_K Quantized (210 bytes per 256 values)

### q6k_matvec.rs — `q6k_matvec`
6-bit quantization with 16 sub-block int8 scales. Used for V projection and FFN down (higher precision than Q4_K). 4 rows per TG.
Origin: LARQL original, matching GGUF Q6_K dequantization formula.

## Q8 Quantized (int8 + per-block f32 scales)

### q8_matvec.rs — `q8_matvec`
Q8 weight × Q8 input. Used for attention projections in the Q8 decode path.
Origin: LARQL original.

### q8_attn_proj.rs — `q8_qkv_proj`, `q8_proj_rope`
**Fused Q8 Q+K+V projection.** Same row-dispatch pattern as q4k_qkv_proj but for Q8 weights. Threadgroup shared Q8 input, integer dot product, simd_sum. 2.5x faster than 3 separate dispatches.
`q8_proj_rope`: Single Q8 projection (for O projection).
Origin: LARQL original.

```
Performance: 0.48ms single dispatch, 10.2ms for 21 layers fused
Grid: 8 rows per TG, 256 threads
```

## Attention

### fused_attention.rs — `fused_attention` (PRODUCTION)
**Full GQA attention in one kernel.** RoPE (split-half, partial rotation) + QK-norm (optional, Gemma3/4) + GQA (grouped query) + softcap (optional, Gemma2/4) + causal mask + softmax + V weighted sum.

One threadgroup per (query_head, query_position). `skip_rope` flag (buffer 12) allows caller to pre-apply RoPE for prefill KV cache population. `rotary_dim` (buffer 13) controls partial rotation — only the first `rotary_dim` dimensions of each head get RoPE; the rest pass through unchanged (Gemma 4 global layers use 25% rotation). Pass 0 for full rotation.
Origin: LARQL original. See [ADR-007](adr/007-fused-attention-skip-rope.md) and [ADR-010](adr/010-partial-rope-rotary-dim.md).

```
Input: Q[seq, num_q*hd], K[seq, num_kv*hd], V[seq, num_kv*hd]
Output: out[seq, num_q*hd]
Buffers: 0-3 data, 4-8 dims/scale, 9 rope_base, 10 use_qk_norm, 11 softcap, 12 skip_rope, 13 rotary_dim
Causal: scores limited to positions 0..=qi
Threadgroup: float tg_scores[4096] (max seq_len)
```

### causal_attention.rs — `causal_attention`
Basic causal attention (seq≤64). Used by full_layer benchmark. Simpler than fused_attention (no RoPE, no GQA, no softcap).

### kv_attention.rs — `kv_attention` (optimized 2026-04-08)
KV-cached decode attention. One query attends against full cached K/V. One threadgroup per query head. Optimizations:
- **simd_max/simd_sum** for softmax reductions (eliminates serial loops, 3 barriers instead of 6)
- **float4 vectorized** Q·K dot products
- Sliding window support (window_size > 0)

### rope.rs — `rope_at_pos`, `rope_at_pos_batched`, `rope_apply`
Standalone RoPE (split-half pairing) with partial rotation support.
- `rope_at_pos`: Single head, used by prefill and decode_hybrid
- `rope_at_pos_batched`: **All heads in one 2D dispatch** (grid: rotary_dim/2 × num_heads). Reduces 12 per-head dispatches to 2 (one for Q heads, one for K heads).
- `rope_apply`: Multi-position variant for prefill
`rotary_dim` controls partial rotation. See [ADR-010](adr/010-partial-rope-rotary-dim.md).

## Element-wise Operations

### geglu.rs — `geglu_silu`
**silu(gate) × up** activation. One thread per element. Used between FFN gate/up and down projections.

### quantize_q8.rs — `quantize_q8`
f32 → Q8_0 on-the-fly quantization. Per-block abs-max scaling. Used to quantize layer outputs for next layer's Q8 input.

### residual_inject.rs — `residual_copy`, `residual_add`
Buffer copy and element-wise addition for residual connections.

## Fused Multi-Op Kernels

### fused_ops.rs — `rms_norm_q8`, `residual_norm`, `residual_norm_q8`

All norm kernels use **cooperative SIMD reduction** for sum_sq computation:
each thread sums a stripe of elements (stride = tg_sz), then `simd_sum` reduces
within each simdgroup, and a threadgroup reduction combines across simdgroups.
This is O(N) reads vs the previous O(N²) where every thread redundantly read all
elements. **This single fix saved ~10ms for 34 layers** (the biggest optimization).

- `rms_norm_q8`: Fused norm + Q8 quantize. Uses `simd_max` for block-max.
- `residual_norm`: Fused residual add + norm.
- `residual_norm_q8`: Fused residual add + norm + Q8 quantize.
Origin: LARQL original. SIMD cooperative reduction added 2026-04-09.

### residual_inject.rs — `residual_copy`, `residual_add`, `scale_vector`, `rms_norm`

- `rms_norm`: Standalone RMS norm with cooperative SIMD reduction.
- `residual_add`: Element-wise a + b.
- `residual_copy`: Buffer copy.
- `scale_vector`: Per-layer scalar multiplier (Gemma 4).

## KV Cache

### kv_attention.rs — `kv_cache_append`
Appends new K/V token to the cache buffer at the current position. One thread per element.

## Experimental

### turboquant_encode.rs — `turboquant_encode_4bit`
WHT (Walsh-Hadamard Transform) + Lloyd-Max 4-bit quantization for KV cache compression. One threadgroup per vector.

### turboquant_decode.rs — `turboquant_decode_4bit`
Inverse: unpack indices → centroid lookup → inverse WHT → rescale.

### graph_walk_knn.rs
GPU-accelerated gate KNN for vindex walk architecture.

### activation.rs — `silu`, `gelu_tanh`
Standalone activation functions for non-gated FFN (StarCoder2, GPT-2). Unlike GEGLU, these apply activation to a single buffer without gate multiplication. `out[i] = activation(input[i])`.

### layer_norm.rs — `layer_norm`, `layer_norm_no_bias`
Standard LayerNorm: `out = (x - mean) / sqrt(var + eps) * (weight + offset) + bias`. Used by StarCoder2, GPT-2, BERT. `layer_norm_no_bias` variant omits the bias term.

### v_norm.rs — `v_norm`, `v_norm_batched`
Parameter-free RMSNorm applied to V states before attention (Gemma 4). `out = x / sqrt(mean(x²) + eps)`.
- `v_norm`: Single head
- `v_norm_batched`: **All KV heads in one 2D dispatch** (grid: head_dim × num_heads). Reduces 4 per-head dispatches to 1.

### residual_inject.rs (updated) — `scale_vector`
Per-layer scalar multiplier: `out = input * scalar`. Used by Gemma 4's learned layer scalars. Added alongside existing `residual_copy` and `residual_add`.

## Common Header (common.rs)

Included by all shaders:
- `decode_f16_metal(ushort)` — f16 bit pattern → f32 conversion
- `struct block_q4_K` — 148-byte Q4_K superblock layout
- `struct block_q4_K_gguf` — 144-byte GGUF-compatible layout
- `struct block_q4_kf` — 160-byte pre-baked half scales layout
