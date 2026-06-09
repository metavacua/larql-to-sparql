# Lazy quantized matmul for Qwen3.6 forward

## Why

The bench-baseline doc captured 2026-05-11 (PR #85) shows that the
Qwen3.6-27B Q4_K_S forward path uses **~100 GiB of system RAM** —
because every Q4_K_S / Q6_K tensor is dequantized to a contiguous
`Array2<f32>` at GGUF load time (26.9 B params × 4 bytes ≈ 107 GiB).
That's 5–7× more memory than llama.cpp uses for the same model (it
keeps weights compressed and dequantizes per-tile during matmul).

Two consequences:
- Decode is 0.49 tok/s vs llama.cpp's 2.6 tok/s on the same CPU.
- A user with 64 GiB RAM (typical workstation) cannot run the 27 B
  model at all; they'd need ≥ 128 GiB. The 35 B-A3B MoE variant is
  worse.

## What

Add a **lazy-quantized weight path** for Qwen3.6 forward:

1. A new `QuantTensor` type in `larql-models` that holds the raw
   GGUF bytes + `tensor_type` + `[rows, cols]`, but **does not
   dequantize at load time**.
2. A `.matvec(&Array1<f32>) -> Array1<f32>` method on `QuantTensor`
   that dispatches to the existing `q4k_row_dot`, `q6k_row_dot`, or
   f32-fallback kernel per row.
3. An opt-in path in `qwen35_forward_step` (env-gated initially,
   default-on once measured stable) that uses `QuantTensor` for the
   heaviest tensors (FFN gate/up/down, lm_head, attn_qkv).
4. Bench numbers in the same format as `bench-baseline.md`.

## Phase 1 scope (this proposal)

Only the **lm_head** matmul. Smallest blast radius (one matmul per
forward), proves the infrastructure, gives an immediate ~4 GiB RAM
reduction (5.1 GiB f32 → 1.0 GiB Q6_K), and validates the dispatch
layer. If Phase 1 confirms the RAM win, Phase 2 expands to FFN.

## Non-goals

- **Speed parity with llama.cpp.** On x86 the existing
  `q4k_row_dot` / `q6k_row_dot` kernels are scalar; ndarray's f32
  matmul uses BLAS. So Phase 1 may be neutral or slightly slower on
  speed — the win here is RAM. Speed needs AVX2/AVX-512 kernels (a
  separate change).
- **Touching the safetensors loader.** The capability spec mandates
  f32 there; we don't change that contract.
- **GPU paths.** Phase E (CUDA for DeltaNet) is unrelated; this is
  CPU-only memory optimization.

## Trade-offs

- **Compatibility:** existing tests that build synthetic `Array2<f32>`
  weights continue to work because the lazy path is gated by GGUF
  load. The bridge constructs `QuantTensor::from_f32_array(...)` as
  a fallback so synthetic tests pass through.
- **Numerical parity:** the lazy path reuses the already-validated
  `q6k_row_dot` (post C.5j). Per-row matvec results should match
  `dequantize_q6_k(...).dot(&x)` to BLAS-level precision (≤ 1e-4
  relative).

## Success criteria

- Qwen3.6-27B Q4_K_S `real_gguf_qwen35_bench` peak RSS drops by
  ≥ 4 GiB.
- `real_gguf_qwen35_token_diff_vs_llama_cpp` still passes with GT
  rank 0 at every step.
- Decode tok/s is within 20% of the dequant-then-f32 baseline (i.e.
  ≥ 0.39 tok/s). If it regresses below that, Phase 1 is reverted
  and the bottleneck is documented for the AVX kernel follow-up.
