# Roadmap — larql-kv

## Current state (as of 2026-05-09)

- Crate extracted from `larql-inference::engines` on 2026-05-09 — see
  [`CHANGELOG.md`](CHANGELOG.md).
- Four engines shipped: `markov_residual`, `unlimited_context`,
  `turbo_quant`, `apollo`. Each implements `KvEngine` plus the Q4K Metal
  fast paths (`prefill_q4k`, `decode_step_q4k`).
- Consumers wired:
  - `larql-cli bench --engine <spec>` (selector dispatch)
  - `kv-cache-benchmark` (criterion comparison vs Standard KV / Graph Walk)
- Coverage policy: 90 % line coverage per source file (see
  `coverage-policy.json`); CI gate at `make larql-kv-coverage-policy`.
  Total line coverage is **85.13 %** as of 2026-05-10 (up from 67.44 % at
  extraction-time on 2026-05-09 after a coverage push that lifted
  `lib.rs`, `profiler.rs`, `apollo/npy.rs`, `apollo/engine.rs`, and
  `markov_residual/engine.rs` past the 90 % floor). 6 files retain
  debt baselines that must ratchet upward, never down.

## Coverage debt (extraction-time, ratchet upward)

| File | Current | Target | Why it's hard |
|---|---:|---:|---|
| `engines/markov_residual/q4k.rs` | 57.1 % | 90 % | Metal-only path; CPU fallback exercises some lines but the live-Metal early-return branches need a `cfg(feature = "metal")` test harness. |
| `engines/unlimited_context/engine.rs` | 50.5 % | 90 % | Q4K Metal entry points + checkpoint reload still uncovered; needs the same Metal fixture as q4k.rs. |
| `engines/unlimited_context/extend.rs` | 74.6 % | 90 % | Checkpoint extend with non-empty prior. Test fixture would need a multi-window synthetic prefill. |
| `engines/turbo_quant/engine.rs` | 79.7 % | 90 % | Q4K-quantised prefill / decode_step paths; integration tests cover the f32 happy path. |
| `engines/markov_residual/compute.rs` | 80.6 % | 90 % | Sliding-window eviction edge cases at small window sizes. |
| `engines/apollo/store.rs` | 89.7 % | 90 % | Just below the floor: remaining miss is all in `StoreLoadError::Io` constructor closures fired on mid-load disk failures. Filesystem-mock test fixture would close it. |

## Open work

### P0 — correctness / performance

- **D-METAL-PLE** *(carries from larql-compute roadmap)*: Per-Layer
  Embeddings not implemented in Metal. Engines on Gemma 4 E2B fall through
  the deliberate CPU fallback in `gpu.rs:372-374`, costing ~30× decode.
  Fix is a 1-2 day Metal port of `forward/ple.rs`. Engines themselves are
  PLE-agnostic; the gain accrues through the shared `decode_token` Metal
  path.
- **Engine-level profiler coverage.** `markov_residual` records
  per-stage `embed/recompute_cold/recompute_hot/attention/ffn/total`. The
  other three engines do not yet populate `EngineProfiler`; they return
  `None` from `stage_summary()`. Wire them so `larql bench --profile`
  produces comparable breakdowns across all four.

### P1 — capability extensions

- **Page-aligned KV slabs for `unlimited_context`.** The current
  `CheckpointStore` uses owned `Vec<f32>` per layer per checkpoint; a
  hugepage-backed slab would cut allocation churn and improve thermal
  steadiness during 370K-token replays.
- **Apollo store on disk.** `apollo` currently expects an in-memory
  `ApolloStore`. Add an mmap loader that reads the constellation map +
  boundary residuals from the same vindex-style on-disk layout as
  `down_meta.bin`, so apollo can serve ~10⁵-entry stores without RAM cost.
- **TurboQuant SIMD packing.** The Lloyd-Max codec works at scalar f32
  today; the rotation step is amenable to NEON / AVX2 vectorisation. The
  encoder is bandwidth-bound at ~95 tok/s decode but the corpus-prep step
  pays N×L×kv_dim of full-precision encode; that's the right place to
  spend the SIMD budget if/when the corpus grows.

### P2 — research / sequencing

- **Cross-engine comparator.** `kv-cache-benchmark` compares each engine
  against Standard KV individually. The synthesis question is: which
  engine wins for which prompt regime (long-context QA vs short-prompt
  multi-turn vs streaming generation)? A criterion harness sweeping
  prompt length × decode length × batch size would surface this.
- **Compositional engines.** `apollo + turbo_quant` would put quantised
  K/V inside the boundary windows; `markov_residual + apollo` would let
  the residual recompute path read pre-projected boundary residuals.
  Neither is wired today; the trait already supports composition because
  engines hold the persistent state, not the dispatch.

## Closed (recent)

- **2026-05-09 — Initial extraction.** `engines/` carved out of
  `larql-inference` into the new `larql-kv` crate. ~5,540 LOC moved with
  no semantic changes. All four engines + `KvEngine` + accuracy /
  profiler helpers now ship from this crate.

## Non-goals

- **Sampling.** Engines return hidden states; sampling lives in
  `larql_inference::layer_graph::generate::Sampler`. Don't add sampling
  helpers here.
- **Tokenisation / chat templates.** Out of scope; the engines operate on
  `&[u32]` token IDs already produced by `larql_inference::tokenizer` /
  `chat`.
- **Generic K/V backends for non-transformer architectures.** The
  `KvEngine` trait references `ModelWeights` directly. Generalising to
  state-space models or RNNs is not on this roadmap; rebuilds are cheap
  and that effort would belong in larql-inference's layer-graph surface.
