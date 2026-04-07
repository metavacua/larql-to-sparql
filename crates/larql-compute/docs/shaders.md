# Metal Shader Reference — larql-compute

28 Metal Shading Language kernels. One file per kernel in `src/metal/shaders/`.
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

### q4k_matvec.rs — `q4k_matvec`
Standalone Q4_K matvec. Reads f16 d/dmin header, unpacks 6-bit scales and 4-bit mins, processes 8 sub-blocks of 32 values. 4 rows per TG.
Origin: LARQL original, matching GGUF Q4_K dequantization formula.

### q4k_qkv_proj.rs — `q4k_qkv_proj` (PRODUCTION for Q4_K attention)
**Fused Q+K+V projection.** Single dispatch for all three attention projections. Rows 0..q_rows → Q, q_rows..+k_rows → K, rest → V. Sub-block lane assignment (80 sub-blocks / 32 lanes = 83% utilization). No threadgroup memory for input — direct device reads.
Origin: LARQL original. Sub-block iteration is novel; direct device reads inspired by llama.cpp.

```
Performance: 1.5ms for 34 layers (0.045ms/layer) — 6.7x faster than Ollama's entire layer
Grid: 8 rows per TG, 256 threads
```

### q4kf_qkv_proj.rs — `q4kf_qkv_proj`
Fused Q+K+V using llama.cpp's kernel_mul_mv_q4_K_f32 architecture. Register-based input (yl[16], yh[16]), quarter-block lane decomposition (ix=lane/8), uint16_t nibble extraction, FOR_UNROLL pragma. Operates on our 148-byte Q4_K blocks.
Origin: Ported from llama.cpp (MIT license), adapted for fused QKV and our block layout.

```
Performance: Same as q4k_qkv_proj (both converge to same speed)
Grid: 4 rows per TG (2 SG × 2 rows/SG), 128 threads
```

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
**Full GQA attention in one kernel.** RoPE (split-half) + QK-norm (optional, Gemma3) + GQA (grouped query) + softcap (optional, Gemma2) + causal mask + softmax + V weighted sum.

One threadgroup per (query_head, query_position). `skip_rope` flag (buffer 12) allows caller to pre-apply RoPE for prefill KV cache population.
Origin: LARQL original.

```
Input: Q[seq, num_q*hd], K[seq, num_kv*hd], V[seq, num_kv*hd]
Output: out[seq, num_q*hd]
Causal: scores limited to positions 0..=qi
Threadgroup: float tg_scores[4096] (max seq_len)
```

### causal_attention.rs — `causal_attention`
Basic causal attention (seq≤64). Used by full_layer benchmark. Simpler than fused_attention (no RoPE, no GQA, no softcap).

### kv_attention.rs — `kv_attention`
KV-cached decode attention. One query attends against full cached K/V (all previous positions). One threadgroup per query head.

### rope.rs — `rope_apply`
Standalone RoPE (split-half pairing). Applies position-dependent rotation to [seq_len, dim] in-place. Used by prefill pipeline to pre-RoPE K for KV cache population.

## Element-wise Operations

### geglu.rs — `geglu_silu`
**silu(gate) × up** activation. One thread per element. Used between FFN gate/up and down projections.

### quantize_q8.rs — `quantize_q8`
f32 → Q8_0 on-the-fly quantization. Per-block abs-max scaling. Used to quantize layer outputs for next layer's Q8 input.

### residual_inject.rs — `residual_copy`, `residual_add`
Buffer copy and element-wise addition for residual connections.

## Fused Multi-Op Kernels

### fused_ops.rs — `rms_norm`, `rms_norm_q8`, `residual_norm`, `residual_norm_q8`
- `rms_norm`: RMS normalization with configurable weight offset (0.0 for Llama, 1.0 for Gemma).
- `rms_norm_q8`: Fused norm + Q8 quantize (saves one dispatch per layer).
- `residual_norm`: Fused residual add + norm.
- `residual_norm_q8`: Fused residual add + norm + Q8 quantize (saves two dispatches per layer).
Origin: LARQL original. Fusion motivated by component profiling showing dispatch overhead dominates for small ops.

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

## Common Header (common.rs)

Included by all shaders:
- `decode_f16_metal(ushort)` — f16 bit pattern → f32 conversion
- `struct block_q4_K` — 148-byte Q4_K superblock layout
- `struct block_q4_K_gguf` — 144-byte GGUF-compatible layout
- `struct block_q4_kf` — 160-byte pre-baked half scales layout
