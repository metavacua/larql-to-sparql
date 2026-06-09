## Why

PR #139 (Q4kDirectFfn) closed the largest CPU decode-step bottleneck —
materialising f32 FFN weights through memory on every step. Gemma 3 4B
moved from 0.117 → 1.36 tok/s (11.6×). The remaining ~10× gap to
llama.cpp's 14.1 tok/s is now off the FFN path.

The next-largest read on every decode step is the attention Q/K/V/O
projections. The `WeightFfn`-equivalent decode path
(`run_attention_block_decode_step_backend`) still reads f32 W_Q/K/V/O
from `weights.tensors` (the dequant cache from PR #138) and routes
them through BLAS GEMV. For Gemma 3 4B this is ~2 GB of f32 read per
decode step across the 34 layers.

The Q4_K × Q8_K matvec kernels in `larql_compute::cpu::ops::q4k_q8k_dot`
(built in `cpu-kquant-matvec-correctness-avx2`) already provide the
direct path. The vindex stores attention weights as Q4_K (Q/K) and
Q4_K or Q6_K (V/O). This proposal adds a thin direct-matvec adapter
that plumbs them through the decode-step attention forward.

## What This Change Ships

**Capability deltas** (under `inference-residual-engine/`):

- A new `run_attention_block_decode_step_q4k_direct` function in
  `crates/larql-inference/src/attention/q4k_direct.rs` that mirrors
  `run_attention_block_decode_step_backend` but uses `q4k_q8k_matvec_into`
  / `q6k_q8k_matvec_into` directly on the vindex bytes for Q/K/V/O,
  skipping `weights.tensors` lookups for those projections.
- `run_attention_block_with_kv_out_with_cache` takes a new
  `vindex: Option<&VectorIndex>` parameter. When `Some`, the residual
  is single-row (decode step), there's no shared K/V donor, and the
  layer satisfies both alignment requirements
  (`hidden % 256 == 0` AND `(num_q * head_dim) % 256 == 0`), the
  direct path engages. All other branches keep the current
  `weights.tensors` + BLAS path.
- `run_layer_with_ffn_with_cache` and `predict_q4k_hidden_with_cache`
  thread `Option<&VectorIndex>` through unchanged.

## Why a parameter and not a trait

`Q4kDirectFfn` used an `FfnBackend` trait. Attention has a single
production decode-step function rather than a layer-by-layer trait
dispatch; threading `Option<&VectorIndex>` through two function
signatures is smaller than introducing an `AttentionBackend` trait and
plumbing it through. The trait pattern is appropriate if the attention
backend grows to dispatch per-layer (e.g., per-layer sliding-window or
per-layer mixed-quant strategies).

## Alignment requirements

Two Q8_K-block-aligned activation streams enter the kernels:

1. **`h_norm`** (input to Q/K/V matvecs): shape `[1, hidden]`. Requires
   `hidden % 256 == 0`. Gemma 3 4B (`hidden=2560`) is aligned; 1B
   (`hidden=1152`) is NOT — it falls back to the `weights.tensors` path.
2. **`attn_out`** (input to O matvec): shape `[1, num_q * head_dim]`.
   Requires `(num_q * head_dim) % 256 == 0`. Gemma 3 4B
   (`num_q * head_dim = 8 * 256 = 2048`) is aligned.

The dispatch gate checks both alignments per-layer; mixed-alignment
arches stay correct (each layer independently chooses its path).

## Out of Scope (Follow-Ups)

- Dropping attention Q/K/V/O from `insert_q4k_layer_tensors` when the
  direct path engages. They'd no longer be read, saving ~2 GB resident
  for 4B. Conservative for now — prefill still uses the dequant cache.
- Multi-row direct path (prefill). Prefill continues to use BLAS GEMM
  via `run_attention_block_with_kv_out`.
- Layers with `shared_kv` (cross-layer K/V borrowing — Gemma 4 family).
  These already short-circuit to `run_attention_block_with_kv_out`
  before the direct path is considered.

## Bench Plan

Compare against the 1.36 tok/s baseline established by #139 on Gemma 3
4B Q4_K_M (CPU-only, 48-thread host). Projection BW is ~1/5 of FFN BW,
so the expected speedup is modest — roughly +20-30% to bring decode
into the ~1.7-1.8 tok/s range. End-to-end numbers will be recorded
post-merge.
