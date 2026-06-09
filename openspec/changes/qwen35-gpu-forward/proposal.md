# Qwen3.6 GPU forward (Phase E)

## Why

Current state (post Phase 2d, PR #91):

| Config | Decode (t/s) | RSS / VRAM |
|---|---:|---:|
| larql CPU (lazy + AVX2 + rayon) | 0.23 | 20 GiB RAM |
| llama.cpp CPU (-ngl 0) | 2.60 | ~16 GiB RAM |
| llama.cpp CUDA GPU (RTX 4090) | **50.60** | 14.76 GiB VRAM |

We're 220× slower than llama.cpp GPU and 11× slower than llama.cpp CPU.
**More CPU work (Phase 3b AVX2 batched matvec) can at best buy us ~2-3×**
— still 10× off the GPU sitting idle. The user has a 4090; correctness
is done; the right axis is GPU dispatch.

larql already has substantial CUDA infrastructure unused by the Qwen3.6
path:

```
crates/larql-compute/src/cuda/
  q4k_mmvq.rs           — Q4_K matvec (893 LoC, mmvq + cooperative variants)
  q6k_mmvq.rs           — Q6_K matvec (440 LoC)
  q4k_direct.rs         — Direct Q4_K kernel (218 LoC)
  matmul.rs             — Higher-level matmul dispatch
  attn.rs / attn_tree.rs— Attention compute
  cache.rs              — KV cache primitives
  dequant.rs            — Format conversions
  backend.rs            — CudaBackend (`new()`, `new_with_index(ordinal)`)
```

Public surface: `larql_compute::backend::QuantMatVec` trait — already
exposes `q4k_matvec`, `q6k_matvec`, `quant_matvec(format, w, x, rows,
hidden) → Option<Vec<f32>>`. Slot into Qwen3.6 forward; no new
kernels needed for FFN / lm_head / attn projections.

What's new for Qwen3.6 specifically: the **DeltaNet recurrence** and
**Conv1D-with-state** kernels. Neither has a CUDA implementation yet.

## What

Pivot the qwen3.6 forward off pure-CPU onto GPU dispatch, reusing the
existing CUDA Q4_K/Q6_K kernels. Multi-PR sequence:

### E.1 — single-matvec PoC (this PR)
Dispatch the lm_head matvec through `CudaBackend::q6k_matvec`. lm_head
is one matvec/token (smallest blast radius), already lazy via
`QuantTensor`, gives a clean tok/s signal. Validates: backend
construction inside larql-inference, weight upload, host↔device
transfer, parity vs the dequant baseline.

### E.2 — FFN matvecs on GPU
Dispatch all 192 FFN matvecs/token through `CudaBackend::q4k_matvec`.
Biggest single perf win — FFN dominates decode time. Reuses E.1
plumbing, just more dispatch sites.

### E.3 — DeltaNet projections + attn q/k/v/o on GPU
The Phase 2b/c/d weights (`attn_qkv`, `attn_gate`, `ssm_out`,
`attn_q`, `attn_k`, `attn_v`, `attn_output`) all become GPU
matvecs. Embed lookup stays CPU (one row dequant per token; not
worth the upload).

### E.4 — DeltaNet recurrence + Conv1D CUDA kernels
The Rust scalar `delta_net_step` and `causal_conv1d_step` get CUDA
implementations. Per-head state matrices are small (128×128 f32
each, fits in shared memory). Reference: llama.cpp's
`ggml_compute_forward_gated_delta_net_one_chunk` which we already
diffed bit-exact in Phase C.

### E.5 — Full-attn forward on GPU
Replace the Rust softmax-attention path with the existing
`cuda/attn.rs` kernel.

### E.6 — Device-resident weights + KV cache
Stop round-tripping host↔device per matvec. Keep weights in VRAM
across forward calls; KV cache lives in VRAM.

## Non-goals (this proposal)

- **Multi-GPU.** Single device only.
- **fp16 / bf16 activations.** Stay f32 to match the existing Rust
  forward's precision and the parity oracle.
- **CUDA Graphs.** Performance optimisation for later — E.6 first.
- **Metal.** Existing Metal kernels are Apple-Silicon only; out of
  scope for this RTX 4090-targeted change.

## Trade-offs

- **VRAM budget.** A 27 B Q4_K_S model is 14.76 GiB on disk; full
  GPU-resident keeps it in VRAM. Fits on 24 GiB cards; will need
  the `--ffn` remote path (or sharding) for 35 B-A3B on consumer
  GPUs.
- **Numerical parity.** GPU f32 reductions can re-order vs CPU
  Kahan-style scalar reductions. The Phase C parity oracle is
  pearson 0.9999 / max|d| 0.006 at layer 0 even with cudaBLAS f32
  dequant (matches what `q6k_matvec`'s default impl does on CUDA).
  Argmax should still match every step.

## Success criteria

- E.1: `real_gguf_qwen35_token_diff_vs_llama_cpp` under
  `LARQL_QWEN35_GPU=1 LARQL_QWEN35_LAZY_LM_HEAD=1` emits the same
  `[<think>, \n\n, </think>, \n\n, Hello]` sequence, GT rank 0
  every step. Wall-time decode improves (the one matvec moves to
  GPU; the rest still scalar) — expect modest gain.
- E.2: decode tok/s ≥ 1.0 (4× over CPU AVX2+rayon).
- E.4: decode tok/s ≥ 10 (≈ 20 % of llama.cpp GPU; remaining gap
  is fused kernels + CUDA Graphs).
- E.6: decode tok/s ≥ 30 (within 2× of llama.cpp's 50.6 t/s, the
  remaining gap being whatever flash-attention / fused softmax /
  in-flight batching wins llama.cpp has that we haven't ported).
