# Roadmap — larql-compute

## Current: 117 tok/s (34L, Q4_KF) | Ollama: 98 tok/s | **17% FASTER**

## P0: Exceed Ollama — DONE (2026-04-09)

### ✅ Full kernel + norm optimization
**Status**: Complete — 17% faster than Ollama

8.5ms / 117 tok/s vs Ollama 10.3ms / 98 tok/s. Key changes:
- Cooperative SIMD norm reduction (O(N²)→O(N)) — saved ~10ms alone
- Q4_KF (GGUF) FFN through llama.cpp-exact q4kf_proj kernel
- Fused gate+up kernels (q4k_ffn_gate_up + q4kf_ffn_gate_up)
- Q4_K matvec rewrite: uint4, 8 rows/TG, multi-row (nr0=2)
- Pre-allocated scratch buffers (550 allocs → 20)
- Batched RoPE + V-norm, SIMD KV attention
- Single cmd buffer + single global encoder

Previous: 29.2ms / 34 tok/s (2.84x Ollama).

### ✅ Dispatch merging
**Status**: Complete (but negligible impact — Apple Silicon dispatch overhead is ~0ms)

### Wire cached layers into decode path
**Impact**: ~4x speedup (compute 8 layers instead of 34)  
**Effort**: Low  
**Status**: Not started (infrastructure ready in larql-inference)

L0-12 are template-fixed (0.999 cosine similarity). At 0.25ms/layer × 8 layers = 2ms → ~500 tok/s.

### ✅ Optimize KV cache attend kernel
**Status**: Complete — simd_max/simd_sum reductions, float4 Q·K dot products, 1024-entry scores.

### ✅ Fix O(N²) norm kernels
**Status**: Complete — cooperative SIMD reduction in all norms. Saved ~10ms (the single biggest win).

## P0.5: Gemma 4 26B A4B correctness (in progress)

### ✅ CPU MoE decode interleave — DONE (2026-04-20)
GPU dense FFN + CPU MoE per layer. Layer scalar correctly applied to
the FFN+MoE delta only (`h_post_attn + scalar*(dense+moe)`). First
coherent Gemma 4 26B output confirmed (Paris, Berlin, oxygen).

### Batched MoE prefill
**Effort**: Medium
**Status**: Workaround shipped (token-by-token decode loop in `prefill_q4`)

Current workaround is correct but serialises `seq_len` decode calls —
O(seq_len × num_layers) GPU command buffers for a prompt. The real fix
is a batched prefill that processes all positions in a single pass:
for each layer, dispatch GPU dense FFN over all positions, then CPU MoE
over all positions, then proceed to next layer. Requires restructuring
`dispatch_full_pipeline` to accept a per-layer CPU callback.

### Fix `dispatch_full_pipeline` layer_scalar
**Effort**: Low
**Status**: Not started — current models (Gemma 3 4B) not affected

`dispatch_full_pipeline` applies `layer_scalar` to `h_bufs[l+1]`
(full residual = `h_post_attn + ffn_delta`) instead of just the FFN
delta. Correct formula: `h_post_attn + scalar * ffn_delta`.

Fix: pass `(scale_pipeline, scalar)` into
`residual::encode_post_ffn`, apply scalar to the normed FFN buffer
before the residual add. Call sites: `full_pipeline.rs:844`,
`tests/test_metal_shaders.rs:2696,2748` — add `None` for non-scaling.

Not urgent: Gemma 3 4B has `layer_scalar = 0.0` (no scaling); Gemma 4
26B is all-MoE and bypasses `dispatch_full_pipeline` via the new
decode-loop prefill.

## P1: Production Hardening

### CUDA backend
**Effort**: Large  
**Status**: Trait ready, no implementation

ComputeBackend trait supports it. Need: CUDA buffer management, kernel ports for Q4_K/Q8 matvec, fused attention, KV cache.

### Streaming prefill
**Effort**: Medium  
**Status**: Prefill pipeline exists but uses CPU for KV cache population

The `prefill_q4` GPU pipeline runs the forward pass. KV cache is populated via CPU `prefill_with_kv` afterward. Integrate KV cache writes into the GPU pipeline to eliminate the CPU roundtrip.

### Dynamic KV cache sizing
**Effort**: Low  
**Status**: Fixed at 4096 max_seq

Current KV cache allocates for 4096 tokens at creation. Need dynamic growth or configurable max_seq for long-context inference.

## P2: Research

### Q4_K FFN pipeline (end-to-end) — DONE
**Effort**: Medium  
**Status**: ✅ Complete (2026-04-07)

Vindex loader (`load_interleaved_q4k`), inference wiring (`predict_honest` prefers Q4_K FFN), and format tag propagation through `FullPipelineLayer` all wired. When `interleaved_q4k.bin` exists, Q4_K format flows through to compute shader dispatch.

### simdgroup_multiply_accumulate for tiled matmul
**Effort**: Large  
**Status**: Research

Apple Silicon has dedicated matrix hardware. For batch inference (seq>1), tiled Q4_K matmul using simdgroup_matrix operations could significantly speed up prefill. Not useful for seq=1 decode (matvec, not matmul).

### Fused layer kernel
**Effort**: Large  
**Status**: Research

Single kernel per layer: norm → QKV → attention → O → residual → norm → FFN → residual. Eliminates ALL inter-op dispatch overhead. Requires careful register management and threadgroup synchronization.

## Completed

| Item | Date | Impact |
|------|------|--------|
| ComputeBackend trait | 2026-04-03 | Foundation |
| Q4_0 v1-v5 kernels | 2026-04-05 | v4 at 61 GB/s |
| Multi-layer FFN batch | 2026-04-05 | 8.4ms/21L |
| Fused attention (RoPE+GQA+softcap) | 2026-04-06 | Correct output |
| Q8 fused QKV | 2026-04-06 | 2.2x vs separate |
| Full pipeline (attn+FFN, 1 cmd) | 2026-04-06 | 18.5ms/21L |
| Safe buffer reads | 2026-04-06 | 13 unsafe sites → 1 |
| CPU Q4_K/Q6_K reference | 2026-04-06 | Cross-backend tests |
| Cross-backend tests (11 tests) | 2026-04-06 | Metal vs CPU verified |
| Q4_K fused QKV | 2026-04-06 | 1.78x vs Q8 |
| Dual-path decode (Q4_K/Q8 auto) | 2026-04-06 | 59 tok/s |
| GPU prefill pipeline | 2026-04-06 | seq>1 on GPU |
| skip_rope flag | 2026-04-06 | Prefill KV cache |
| Sub-block lane assignment | 2026-04-07 | 83% utilization |
| llama.cpp kernel architecture port | 2026-04-07 | Register-based input |
| Component profiling | 2026-04-07 | Found real bottleneck |
| Zero warnings | 2026-04-07 | Clean build |
| ADR documentation | 2026-04-07 | 8 decisions recorded |
| Partial RoPE (rotary_dim) | 2026-04-07 | rope_apply + fused_attention, ADR-010 |
| Gemma 4 architecture support | 2026-04-07 | Per-layer head_dim, KV heads, K=V, layer_scalar |
| Shader documentation | 2026-04-07 | docs/shaders.md — all 44 kernels |
| Quantization format docs | 2026-04-07 | docs/quantization-formats.md |
| Decode pipeline docs | 2026-04-07 | docs/decode-pipeline.md |
| Example reorganization | 2026-04-07 | 25 examples: demo_, compare_, profile_, best_, test_ |
| PERFORMANCE.md refresh | 2026-04-07 | All numbers from fresh benchmark runs |
| ROADMAP.md | 2026-04-07 | P0/P1/P2 targets documented |
| Per-layer architecture params (ADR-011) | 2026-04-07 | 18 fields on FullPipelineLayer: eps, attn_scale, head_dim, num_q/kv_heads, rope_base, rotary_dim, sliding_window, v_norm, layer_scalar, norm_type, ffn_type, activation, biases |
| pipeline.rs extraction | 2026-04-07 | FullPipelineLayer + types moved from lib.rs to pipeline.rs |
| 7 new shader kernels | 2026-04-07 | silu, gelu_tanh, layer_norm (2), v_norm, scale_vector, rope_at_pos partial |
| Model-agnostic compute | 2026-04-07 | No hardcoded model assumptions — all behavior parameterized per-layer |
| Single cmd buffer decode | 2026-04-08 | All 34 layers in one cmd, single encoder per layer |
| Batched RoPE/V-norm | 2026-04-08 | rope_at_pos_batched, v_norm_batched — 16 dispatches → 3 |
| Q4_K FFN format routing | 2026-04-08 | Q4_K weights use q4k_matvec, skip Q8 quantize |
| Fused gate+up kernel | 2026-04-08 | q4k_ffn_gate_up — single dispatch, shared input |
| Q4_K matvec rewrite | 2026-04-08 | uint4 loads, 8 rows/TG, sub-block striping, nr0=2 |
| Q4_KF FFN routing | 2026-04-08 | llama.cpp-exact q4kf_proj for FFN gate/up/down |
| SIMD KV attention | 2026-04-08 | simd_max/simd_sum, float4 dot, 3 barriers (was 6) |
| Ollama parity | 2026-04-08 | 2.84x → ~1.25x at 34 layers, no caching |
| Q4_KF fused gate+up | 2026-04-09 | q4kf_ffn_gate_up — llama.cpp inner loop, shared input |
| Pre-allocated scratch buffers | 2026-04-09 | 550 allocs → 20, saved ~2ms |
| Single global encoder | 2026-04-09 | One encoder for all 34 layers (no per-layer create/end) |
| **Cooperative SIMD norms** | **2026-04-09** | **O(N²)→O(N) in rms_norm/residual_norm — saved ~10ms** |
| **Ollama EXCEEDED** | **2026-04-09** | **8.5ms / 117 tok/s = 0.83x Ollama (17% faster)** |
