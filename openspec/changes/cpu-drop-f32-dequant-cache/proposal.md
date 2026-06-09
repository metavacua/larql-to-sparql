## Why

After PRs #142 / #144 / #145, every hot-path matvec on the
`/v1/chat/completions` CPU pipeline (FFN gate/up/down, attention Q/K/V/O
decode-step, lm_head) flows through rayon-parallel AVX2 Q4_K × Q8_K
matvec on the vindex bytes. The f32 dequant cache that `insert_q4k_layer_tensors`
populated for the old WeightFfn / `dot_proj_gpu` BLAS GEMV paths is still
allocated — ~10 GB of FFN+attn dequant + 2.6 GB of f32 lm_head — but no
production read-path actually consumes it on a Gemma 3 4B chat decode.

Bench from the prior llama.cpp head-to-head: larql RSS = **24.6 GB**
vs llama.cpp CPU = **3.85 GB** (6.4× more). Most of the gap is dead-
weight cache.

Two paths still touched the cache:
1. **lm_head f32 fallback** (`weights.lm_head`). When `lm_head_quant`
   is populated (PR #144 path; Gemma 3 4B / aligned models), the f32
   form is never read.
2. **Prefill attention** (`run_attention_block_with_kv_out` →
   `run_attention_block_core`). Multi-row matmul against Q/K/V/O via
   `weights.tensors.get(attn_q_key)`. This is the seq>1 case — every
   chat completion's first call.

This PR drops both, plus the FFN dequant insertion when the per-layer
direct dispatch can handle it.

## What This Change Ships

### Code

- **`crates/larql-vindex/src/format/weights/load.rs`** — vindex loader
  skips the f32 dequant of `lm_head_q4.bin` when `hidden_size` is
  256-aligned (i.e., `lm_head_quant` will be populated and PR #144's
  direct path takes over). Saves ~2.6 GB. Non-aligned models keep the
  f32 form.

- **`crates/larql-inference/src/attention/q4k_prefill.rs`** (NEW) —
  `run_attention_block_prefill_q4k_direct`. Multi-row prefill direct
  Q4_K × Q8_K matvec for Q/K/V/O. Mirrors the structure of
  `run_attention_block_with_kv_out`'s body but pulls weights from the
  vindex bytes via `index.attn_q4k_layer_data(layer)` and projects via
  per-row matvec. Per-row matvec uses the existing rayon-parallel AVX2
  kernel.

- **`crates/larql-inference/src/attention/block.rs`** —
  `run_attention_block_with_kv_out_with_cache`'s `cached_len == 0`
  branch (prefill on an empty cache) now dispatches to the prefill
  direct path when `vindex.is_some()` + alignment guards + non-MoE +
  no shared-K/V donor + no attention capture. Other prefill flavours
  (capture, shared-KV, MoE) keep the existing path.

- **`crates/larql-inference/src/vindex/q4k_forward/hidden.rs`** —
  `predict_q4k_hidden_with_cache` skips `insert_q4k_layer_tensors`
  when the architecture guarantees every per-layer dispatch will use
  a direct path (non-MoE + hidden aligned + q_dim aligned + no
  cross-layer K/V donor). The FFN backend selection drops the
  `seq == 1` gate too — Q4kDirectFfn's existing per-row loop handles
  multi-row prefill correctly.

- **`crates/larql-models/src/quant/lazy.rs`** — minor cleanup: rename
  the cache-key destructuring binding `fL` → `f_n` to satisfy
  `non_snake_case` (introduced inadvertently in PR #145; was a warning,
  not an error).

### Capability deltas

Under `inference-residual-engine/` — production CPU decode SHALL not
require `weights.tensors` to be populated for FFN/attention/lm_head on
the direct paths.

## Bench (Gemma 3 4B Q4_K_M, 48-thread EPYC, BLAS=1)

| Metric (chat completion) | Before this PR | After this PR | Δ |
|---|---:|---:|---:|
| RSS RAM                 | 24.6 GB | **10.3 GB** | **−14.3 GB (−58%)** |
| Prefill 14 tokens       |   2.7 s |   **1.0 s** | **−63%** |
| Decode tok/s @ 150 tok  |  9.8    |   **9.7**   | flat |
| Output coherence        | ✓       | ✓           | preserved |

Prefill got faster (not slower as feared) because the `cached_len == 0`
branch no longer dequants Q4_K bytes into a ~10 GB f32 working set per
request — that dequant was the dominant cost. The per-row direct matvec
loop is slower per-MAC than BLAS GEMM, but the savings on the dequant
step (and the avoided allocator pressure) more than make up for it on
the 14-token prefill that's typical for chat.

## Out of Scope (Follow-Ups)

- Multi-row Q4_K × Q8_K matvec kernel (single GEMM-style call instead
  of N sequential matvecs). Per-row dispatch is fine at typical chat
  prefill sizes (10-200 tokens); a long-context prefill (4096+ tokens)
  would benefit from a batched kernel. Current per-row loop is the
  expected slow-path until the GEMM kernel lands.

- `prefill_q4k_from_embeddings` (used by `/v1/attention/*` routes) is
  unchanged. That code path explicitly uses `WeightFfn` and the f32
  cache; its callers don't run chat completions. Independent arc.

- Hybrid MoE (Gemma 4 26B A4B) and cross-layer K/V share (Gemma 4)
  paths still need the cache. No regression — just no win for those
  arches yet.
