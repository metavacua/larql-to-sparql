# kv-cache-benchmark

Six-way KV cache strategy comparison for the LARQL project:

| # | Strategy | What it does | Memory @ 370K | Compression |
|---|----------|-------------|---------------|-------------|
| 1 | **Standard KV** | FP16 keys + values, per-token, per-layer | 25.8 GB | 1× |
| 2 | **TurboQuant** | WHT rotation + Lloyd-Max 3/4-bit quantization | 6.6 GB | ~4× |
| 3 | **Markov RS** | Bounded hot window (W=512) + cold token IDs | ~193 MB | ~134× |
| 4 | **Boundary RS** | Tiny hot window (W=32) + boundary vec + cold IDs | ~13 MB | ~1,985× |
| 5 | **Hybrid RS+CA** | Cached static attention (97.1%) + tiny dynamic KV + vindex FFN | ~270 MB | ~95× |
| 6 | **RS Graph Walk** _(target — requires cracked attention)_ | Graph lookup for factual queries; Markov RS fallback | 1.5 MB | ~17,200× |

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
Eliminates the KV cache entirely. The residual stream has the Markov property:
the current residual IS the complete state. Stores a bounded hot window of 512
residuals per layer (f32) plus cold-tier token IDs (4 bytes each). Hot window
dominates: 512 × 34 layers × 2560 dim × 4 bytes ≈ 178 MB fixed. Cold tier adds
only 4 bytes/token. Does NOT grow with context. Proven bit-perfect (KL = 0.0)
on Gemma 3-4B via cold-tier replay ([cold||hot] concatenation before recompute_kv).

### Boundary Residual Stream (W=32) — production form
The production form of the Python `unlimited_engine.py` approach. Stores:
- Hot window: 32 residuals per layer ≈ 11.2 MB fixed
- Boundary vector: 1 residual per layer ≈ 340 KB fixed (context boundary marker)
- Cold tier: token IDs only, 4 bytes per token

Total stays flat at ~11–13 MB regardless of context length. At 370K tokens this
is ~1,985× smaller than standard KV while achieving the same attention quality
via cold-tier replay from token IDs.

### Hybrid RS + Cracked Attention (W=512)
The near-term practical win. 97.1% of attention heads produce the same output
regardless of entity (cosine 0.942+). Cache those outputs per template. Only
the ~2.9% dynamic heads (4 layers: L1, L13, L26, L32) need real KV cache.
FFN handled by vindex walk (zero matmul). Memory is bounded by the RS hot
window (~192 MB) plus small dynamic K/V for 4 layers.

### RS Graph Walk _(target architecture — not yet fully operational)_
The endgame once attention is cracked. The forward pass IS a graph walk over
three composed graphs (FFN, attention, residual). Extract the graphs, walk them
directly. No matrices, no multiplication.

**Current status:** FFN graph walk is proven (348K features in vindex, 34 layers,
zero accuracy loss on factual queries). Attention elimination requires cracked
attention — not yet implemented. Until then, queries outside the factual graph
fall back to Markov RS for the full forward pass.

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

**Key insight:** Markov RS and Boundary RS trade compute for memory — they still run
the full 34-layer FFN, but replace K/V matmuls with residual recompute. Hybrid RS+CA
eliminates FFN matmuls entirely (vindex) and caches 97.1% of attention. Graph Walk
eliminates everything — it's three hash-table lookups per decode step.

## Feature flags

- Default: synthetic benchmark only (zero LARQL dependencies)
- `real-model`: enables Phase 2 integration with larql-inference, larql-vindex, etc.

## Spec

Full benchmark specification: [docs/kv-cache-benchmark-spec-v3.md](docs/kv-cache-benchmark-spec-v3.md)
