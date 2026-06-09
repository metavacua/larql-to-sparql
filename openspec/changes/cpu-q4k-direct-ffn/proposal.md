## Why

CPU decode-step FFN is the dominant cost in `generate_via_cpu_q4k` after
the Arc B correctness fixes (PRs #136/#137) landed. With the dequant
cache from PR #138 the per-step dequant cost is amortised, but the FFN
still feeds f32 weights through BLAS GEMV — reading the full 10 GB of
materialised Gemma 3 4B FFN weights through memory every decode step.

llama.cpp's CPU path avoids the f32 stop entirely: it does
Q4_K × Q8_K matvec directly on the quantised bytes (~4× less bandwidth)
which is what closes its remaining gap to our 0.117 tok/s baseline.

The kernels already exist — `cpu-kquant-matvec-correctness-avx2`
(PRs #102–#119) landed `q4k_q8k_gate_up_into`, `q4k_q8k_matvec_into`,
and `q6k_q8k_matvec_into` (measured at ~17 Gelem/s on this host).
`q4k_ffn_forward_layer_q8k` already wires them together for one layer
(currently only used by the `/v1/walk-ffn-q8k` server route).

This proposal adds a thin `FfnBackend` adapter that plumbs the direct
Q4_K × Q8_K path through `predict_q4k_hidden_with_cache` (the CPU
generate hot path) for the decode step only.

## What This Change Ships

**Capability deltas** (under `inference-residual-engine/`):
- A new `Q4kDirectFfn` backend in `crates/larql-inference/src/ffn/q4k_direct.rs`
  that implements `FfnBackend::forward` by Q8_K-quantising the
  post-FFN-norm activation and calling `q4k_ffn_forward_layer_q8k` —
  skipping f32 weight materialisation entirely.
- `predict_q4k_hidden_with_cache` dispatches `Q4kDirectFfn` when
  `h.shape()[0] == 1` (decode step) and the layer is not a hybrid-MoE
  layer; otherwise falls back to `WeightFfn` (preserves BLAS GEMM-friendly
  prefill via the dequant cache).

The dequant cache from PR #138 is still allocated and resident — it
serves attention Q/K/V/O and prefill FFN. Skipping a redundant per-step
read of FFN weights from RAM is exactly the bandwidth reduction we want.

## How It Threads Through the Code

1. `Q4kDirectFfn { arch, index }` holds only borrowed references to
   `ModelArchitecture` and `VectorIndex`, deliberately *not* `&ModelWeights`.
   That's so it can coexist with the per-layer `&mut ModelWeights` borrow
   needed by `insert_q4k_layer_tensors` in the same loop iteration.
2. `forward(layer, x)`:
   - If `x.shape()[0] == 1`: Q8_K-quantise the single row and call
     `q4k_ffn_forward_layer_q8k` — gate+up via `q4k_q8k_gate_up_into`,
     activation gate, down via format-dispatched matvec (Q4_K or Q6_K).
   - If `x.shape()[0] > 1` (multi-row, unexpected on decode path): loop
     per row. Reserved for diagnostic / capture callers; the production
     dispatch in `hidden.rs` routes multi-row through `WeightFfn` so this
     branch should be cold in practice.
3. `forward_with_activation` is a placeholder — capture callers should
   continue to use `WeightFfn` until a follow-up wires intermediate
   activation extraction through the direct path.
4. `forward_moe_full_layer` falls back to the default (returns `None`);
   MoE layers continue to route through the f32 path entirely.

## Out of Scope (Follow-Ups)

- Direct Q4_K × Q8_K for the attention Q/K/V/O projections. Decode-step
  attention compute is much smaller than FFN on Gemma 3 4B (intermediate
  10240 vs head_dim_total 4×256 = 1024) so it's secondary.
- Multi-row direct path. The per-row loop in `Q4kDirectFfn` works for
  small batch sizes but isn't the production prefill path.
- Removing `WeightFfn` or the dequant cache from PR #138. Both remain
  needed for prefill and intervention callers.

## Bench Plan

Compare against the 0.117 tok/s baseline on Gemma 3 4B Q4_K_M (no CUDA,
48-thread host). The kernel-level measurement (17 Gelem/s for Q4_K × Q8_K
matvec) suggests order-of-magnitude headroom, but the full end-to-end
will be limited by per-token quantisation cost, attention compute, and
embedding/lm_head — so the actual decode speedup is the number that
matters and will be recorded after the wiring lands.
