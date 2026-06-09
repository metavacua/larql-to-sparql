## Why

After PRs #138 / #139 / #140 / #142, FFN and attention Q/K/V/O for Gemma 3 4B
CPU decode are off the f32 BLAS path. Pure-decode profile (no lm_head /
sampling) showed 193 ms/step on a 48-core EPYC host — ~5.7 ms/layer ×
34 layers, dominated by the AVX2 `q4k_q8k_matvec_*` / `q6k_q8k_matvec_*`
row loop. The kernels iterate rows serially. On a 48-core host this is
the largest remaining single-thread bottleneck.

## What This Change Ships

**Code:**
- `crates/larql-compute/src/cpu/ops/q4k_q8k_dot.rs`: extract per-row
  AVX2 dot products (`compute_row_q4k_avx2`, `compute_row_q6k_avx2`)
  and dispatch row chunks via `rayon::par_chunks_mut`. Small matvecs
  (`rows < MIN_PAR_ROWS = 16`) skip rayon to avoid per-task overhead —
  the existing unit tests use rows ∈ 5..7 and stay on the sequential
  path.

Row-level reduction order is preserved (each row's accumulator stays
thread-local), so the existing `q*k_q8k_matvec_avx2_matches_scalar`
bit-exact tests still pass. The canonical-dequant + Q8_K-noise
correctness oracles also pass (verified on real Gemma 3 4B V / FFN_DOWN
bytes via `tests/q4k_attn_diff.rs` in the prior arc).

**Capability deltas** (under `compute-backend-traits/`):
- Q*K × Q8K matvec MUST scale across CPU threads for the production
  Gemma 3 4B decode shapes (rows ∈ 1024..10240).
- Bit-exact vs scalar invariant SHALL hold for any `rows >= MIN_PAR_ROWS`
  (parallel path) and any `rows < MIN_PAR_ROWS` (sequential path).

## Bench (Gemma 3 4B Q4_K_M, 48-thread EPYC host)

**Pure decode** (single decode step, no lm_head / sampling, KV cache primed):

| Path | Before | After | Speedup |
|---|---:|---:|---:|
| `predict_q4k_hidden_with_cache` | 193 ms/step | **89 ms/step** | **2.16×** |
| Equivalent decode tok/s | 5.18 | **11.24** | **2.17×** |

**End-to-end** via `/v1/chat/completions` (includes lm_head + sampling
+ HTTP, 150 tok completion):

| Tokens | After #142 | After this PR | Speedup |
|------:|-----------:|--------------:|--------:|
| 150   | 1.63 tok/s | **1.79 tok/s** | **+10%** |

The end-to-end gain is smaller than the pure-decode gain because
lm_head (f32 BLAS GEMV against `weights.lm_head`, ~671 M MACs/step)
and sampling are now a larger fraction of total wall-clock; matvec is
no longer dominant. Next-lever candidates are direct-Q4_K lm_head
matvec and overlap of sampling with the next decode step.

## Out of Scope (Follow-Ups)

- Direct Q4_K × Q8_K lm_head (671M MACs single-thread BLAS step).
- Adaptive `MIN_PAR_ROWS` based on `rayon::current_num_threads()` and
  cols (currently a static 16; might warrant tuning for small-thread
  hosts).
- aarch64 NEON parallelisation (kept sequential here — the existing
  `q4k_q8k_matvec_neon` is left untouched).
