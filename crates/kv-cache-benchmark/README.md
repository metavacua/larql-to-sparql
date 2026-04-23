# kv-cache-benchmark

An inference-memory ladder for the LARQL project. The framing here is not
"compress KV better" — it is that **correctness can be stratified**, and each
stratum admits a radically different state representation. The table below is
the ladder: each rung pairs a storage regime with the specific correctness
target it is obligated to hit.

| # | Strategy | Correctness target | Memory @ 370K | Compression |
|---|----------|--------------------|---------------|-------------|
| 1 | **Standard KV** | bit-exact baseline | 25.8 GB | 1× |
| 2 | **TurboQuant** | approximate bit-exact | 6.6 GB | ~4× |
| 3 | **Markov RS (W=512)** | bit-exact via residual-state reconstruction | ~193 MB | ~134× |
| 4 | **Tier 2 — `UnlimitedContextEngine`** | bit-exact within-window (per-window replay) | ~30 MB | ~2,000× |
| 5 | **Tier 3 — `ApolloEngine`** | first-token factual correctness; lossy continuation | ~2.8 MB | ~20,000× |
| 6 | **RS Graph Walk** _(target — requires cracked attention)_ | graph-level semantic recall | 1.5 MB | ~17,200× |

### The correctness ladder

The rungs are not interchangeable — they answer different questions:

1. **Bit-exact continuation** (Standard KV) — identical logits, identical decode.
2. **Approximate bit-exact** (TurboQuant) — KL → 0 under quantization noise.
3. **Bit-exact via residual reconstruction** (Markov RS) — same next-token distribution under the implemented cold-replay path on the benchmarked setup.
4. **Bit-exact within a bounded replay window** (Tier 2) — K/V checkpoint + token archive reproduces the window exactly; behaviour outside the window is not claimed.
5. **First-token factual correctness** (Tier 3) — the right fact lands; continuation is lossy because a single boundary vector cannot uniquely ground arbitrary suffix text.
6. **Graph-level semantic recall** (Graph Walk, target) — answers recoverable from the extracted graphs; not a literal replay of the original forward pass.

Every claim further down this README should be read as attached to exactly
one of these rungs.

## Implementation status

| Strategy | End-to-end real | Synthetic encode/decode | Scaffold only |
|---|---|---|---|
| Standard KV | ✓ `real_model::kv_capture` + `standard_kv` | ✓ | — |
| TurboQuant | ✓ `real_model::turboquant_layer` + `turboquant` | ✓ | — |
| Markov RS (W=512, Tier 1 / variant iv-dense) | ✓ `real_model::markov_layer` (`rs_prefill`, `rs_decode_step`) — proven bit-perfect end-to-end | ✓ | — |
| `UnlimitedContextEngine` (Tier 2) | ✓ `unlimited_context::` — Rust port of `chuk-mlx/.../unlimited_engine.py`; integration tests `tests/test_unlimited_context.rs` | — | — |
| `ApolloEngine` (Tier 3) | ✓ full end-to-end pipeline on real apollo11_store + Gemma 3 4B. **Two paths**: `query_greedy` (forwards window tokens + query, ~519 context tokens) and `query_greedy_compressed` (forwards 10 KB boundary + query, ~9 context tokens — exercises the actual compression claim). Positional-proximity retrieval + answer-only injection produces `" John"` as top-1 for "Who won the porridge eating contest?" on both paths. | — | — |
| Boundary RS (synthetic) | — | ✓ accounting honest, `decode()` is a stub (`boundary_residual::mod.rs:142,156`) | — |
| Graph Walk | partial (`real_model::graph_walk_layer`) | ✓ | — |

**Compression numbers in the headline table above**:

- Rows 1–3: measured end-to-end on Gemma 3 4B via the `real-model` feature.
- Row 4 (Tier 2): measured via `tests/test_unlimited_context::test_compression_ratio`. Within-window K,V is bit-exact via model-forward replay from the prior window's per-layer K,V checkpoint.
- Row 5 (Tier 3): the Rust `ApolloEngine` loads `apollo-demo/apollo11_store/` end-to-end (2.13 MB in RAM) and runs the full pipeline: tf-idf-lite routing → positional-proximity entry retrieval → forward-with-injection (answer-tokens-only, step-0 only) at L30 coefficient 10× → greedy decode.

  Four entry points, all measured end-to-end on Gemma 3 4B:

  - `query_greedy`: single-token top-1. Full window + query context (~519 tokens). `" John"` @ logit 24.0.
  - `query_greedy_compressed`: single-token top-1. 10 KB boundary + query (~9 tokens, 58× smaller). `" John"` @ logit 31.1.
  - `query_generate_uncompressed`: **iterative decode, 12 tokens. Produces `" John Coyle.\n\n02 05 5"`** — correct answer, then drifts into the transcript's time-stamp structure. Grounded on the window's actual token content.
  - `query_generate_compressed`: iterative decode over the 10 KB boundary. Produces `" John and Mary.\n\nJohn and Mary won the porridge eating"` — starts correctly (" John" from injection), hallucinates "Mary" because the single-vector boundary is lossy and can't uniquely identify the "Coyle" continuation.

  **The gap between compressed and uncompressed outputs is exactly the fidelity/compression trade-off the four-rung ladder predicts**: uncompressed forwards have the raw window text to ground on, compressed forwards rely on the ~10 KB boundary (variant-ii-class) + injection — which lands the first-token fact via amplification but can't carry detailed continuation info. A Tier 2-style per-layer K/V checkpoint (~139 KB per window) would reproduce "Coyle" exactly at the cost of ~14× more storage per boundary.

  Python reference: `chuk-mlx/src/chuk_lazarus/inference/context/research/unlimited_engine.py` + `vec_inject/`.
- Row 6: FFN graph walk proven; attention elimination requires cracked attention (see later in this README).

## Quick start

```bash
# Phase 1: Synthetic benchmark (no model needed)
cargo test -p kv-cache-benchmark

# Run the shader/CPU benchmark
cargo run -p kv-cache-benchmark --example shader_bench

# Run the multi-turn simulation
cargo run -p kv-cache-benchmark --example multi_turn_demo

# Criterion benchmarks
cargo bench -p kv-cache-benchmark

# Phase 2: Real model (requires Gemma 3-4B weights + vindex)
cargo run -p kv-cache-benchmark --example real_model_bench --features real-model
```

## Architecture

```
kv-cache-benchmark/
  src/
    lib.rs              KvStrategy trait, run_strategy_benchmark()
    standard_kv.rs      Strategy 1: raw FP16 encode/decode
    turboquant/         Strategy 2: WHT + Lloyd-Max + bit packing
    markov_residual/    Strategy 3: bounded window + cold tier
    boundary_residual/  Strategy 4: tiny hot window + boundary vec + cold IDs
    hybrid_cracked/     Strategy 5: cached static heads + tiny dynamic KV
    graph_walk/         Strategy 6: routing table + vindex lookup
    benchmark.rs        Sweep runner, multi-turn sim, table formatter
    shader_bench.rs     CPU/Metal operation benchmarks
    metrics.rs          MSE, cosine, inner product error
    model_config.rs     Gemma 4B / Llama 8B / 70B dimensions
    real_model/         Phase 2: wired into larql-inference (feature-gated)
    unlimited_context/  Tier 2: per-window K,V checkpoint + model-forward replay
    apollo/             Tier 3: single-vector boundary + vec_inject (scaffold)
  tests/                66+ unit + integration tests
  benches/              Criterion benchmarks
  examples/             Demo runners
  docs/                 Benchmark spec v3
```

## Strategies in detail

### Standard KV (baseline)
What llama.cpp, vLLM, and MLX use. FP16 keys and values stored per-token,
per-layer, per-head. Memory grows linearly with context length.

### TurboQuant (Google, ICLR 2026)
Compresses KV cache to 3-4 bits per coordinate using Walsh-Hadamard rotation
followed by Lloyd-Max scalar quantization. 4-6× compression at the Shannon
limit. Still grows O(context_length).

### Markov Residual Stream (W=512)
Eliminates the KV cache entirely and replaces it with residual state as the
primary persistent representation. Stores a bounded hot window of 512 residuals
per layer (f32) plus cold-tier token IDs (4 bytes each). Hot window dominates:
512 × 34 layers × 2560 dim × 4 bytes ≈ 178 MB fixed. Cold tier adds only
4 bytes/token. Does NOT grow with context.

**Correctness claim (precise form):** under the implemented cold-replay
reconstruction path — `[cold_token_ids ‖ hot_residuals]` recomposed before
`recompute_kv` at each decode step — the stored state is sufficient to
reproduce the next-token distribution bit-for-bit on the benchmarked setup
(Gemma 3-4B, KL = 0.0 vs. Standard KV). This is a statement about the
reconstruction path under the benchmarked conditions, not a general claim
that residuals are context-free Markov states across all architectures.

### Boundary Residual Stream (W=32) — synthetic memory accounting only

The `boundary_residual` strategy accounts memory for the architecture
described above (32-token hot window + boundary vector + cold token IDs),
but its `decode()` method is a **placeholder**: cold positions are
reconstructed as `boundary.clone()` for every cold slot (see
`src/boundary_residual/mod.rs:142, 156`). Useful for synthetic
compression-ratio comparisons; **not** a bit-exact or task-accurate
reproduction of the Python reference.

The real architecture that backs the ~2,000× and ~20,000× compression
claims lives in two places in this crate:

- **`unlimited_context::UnlimitedContextEngine`** (Tier 2) — per-window
  K,V checkpoint (174 KB on Gemma 3 4B) + token archive + model-forward
  replay. Bit-exact within-window. Reference: `chuk-mlx/.../unlimited_engine.py`.
- **`apollo::ApolloEngine`** (Tier 3, scaffold) — single-vector boundary
  at crystal_layer (10 KB per window) + token archive + `vec_inject`
  retrieval index + injection-at-L30 amplification. Task-level correctness
  on queries routable via the injection index. Reference:
  `chuk-mlx/.../vec_inject/` + `apollo-demo/apollo11_store/`.

### Hybrid RS + Cracked Attention (W=512)
The near-term practical win. 97.1% of attention heads produce the same output
regardless of entity (cosine 0.942+). Cache those outputs per template. Only
the ~2.9% dynamic heads (4 layers: L1, L13, L26, L32) need real KV cache.
FFN handled by vindex walk (zero matmul). Memory is bounded by the RS hot
window (~192 MB) plus small dynamic K/V for 4 layers.

### RS Graph Walk _(target architecture — not yet fully operational)_
The endgame once attention is cracked. The forward pass would be a walk over
three composed graphs (FFN, attention, residual). Extract the graphs, walk
them directly.

**Current status:**

- FFN graph walk is proven (348K features in vindex, 34 layers, zero accuracy
  loss on factual queries).
- Attention elimination requires cracked attention — not yet implemented.
- Until then, queries outside the factual graph fall back to Markov RS for the
  full forward pass.

Treat this rung as a target architecture, not a delivered system. The 1.5 MB
figure is a projected steady-state footprint under the assumption that the
cracked-attention path lands; it is not a current end-to-end measurement.

## Memory scaling

| Metric | Standard KV | TurboQuant 4b | Markov RS W=512 | Boundary RS W=32 | Hybrid RS+CA | Graph Walk |
|--------|------------|---------------|-----------------|------------------|--------------|------------|
| Memory @ 4K | 285 MB | 74 MB | 193 MB | 11.5 MB | ~193 MB | 16 KB |
| Memory @ 32K | 2.24 GB | 580 MB | 193 MB | 11.8 MB | ~194 MB | 130 KB |
| Memory @ 370K | 25.8 GB | 6.6 GB | 193 MB | 13.0 MB | 270 MB | 1.5 MB |
| Grows O(N)? | yes | yes | cold only (+4B/tok) | cold only (+4B/tok) | cold only | cold only |
| Hot window fixed? | no | no | ~178 MB | ~11.2 MB | ~178 MB | — |

## Compute per token

| Operation | Standard KV | TurboQuant | Markov RS | Boundary RS | Hybrid RS+CA | Graph Walk |
|-----------|------------|------------|-----------|-------------|--------------|------------|
| Attention matmul | 34 layers | 34 layers | window only | window only | ~1–2L dynamic | **ELIMINATED** |
| FFN matmul | 34 layers | 34 layers | 34 layers | 34 layers | **ZERO (vindex)** | **ELIMINATED** |
| Logits matmul | 1× | 1× | 1× | 1× | **ZERO (KNN)** | **ELIMINATED** |
| KV cache write | 34L | 34L + quant | none | none | ~1–2L dynamic | none |
| Cold K/V replay | none | none | none | bdy+ids | bdy+ids | none |
| Cached attention | none | none | none | none | ~32–33L | none |
| Graph lookup | none | none | none | none | 34L FFN | 3 per hop |

**Key insight:** the rungs trade compute for memory along different axes.
Markov RS and Boundary RS still run the full 34-layer FFN but replace K/V
matmuls with residual recompute. Hybrid RS+CA eliminates FFN matmuls entirely
(vindex) and caches 97.1% of attention. Graph Walk, in the target
configuration — FFN-as-vindex + cracked attention + residual-graph routing —
reduces per-decode-step work to a small, bounded number of keyed lookups
(one per graph traversed). The precise lookup count depends on the cracked-
attention design, and is stated here conditional on that landing; it is not
a measured figure.

## Feature flags

- Default: synthetic benchmark only (zero LARQL dependencies)
- `real-model`: enables Phase 2 integration with larql-inference, larql-vindex, etc.

## Spec

Full benchmark specification: [docs/kv-cache-benchmark-spec-v3.md](docs/kv-cache-benchmark-spec-v3.md)
