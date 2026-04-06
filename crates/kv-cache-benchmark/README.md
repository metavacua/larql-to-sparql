# kv-cache-benchmark

Five-way KV cache strategy comparison for the LARQL project:

| # | Strategy | What it does | Memory @ 370K |
|---|----------|-------------|---------------|
| 1 | **Standard KV** | FP16 keys + values, per-token, per-layer | 25.8 GB |
| 2 | **TurboQuant** | WHT rotation + Lloyd-Max 3/4-bit quantization | 6.6 GB (3.9x) |
| 3 | **Markov RS** | Bounded window of residuals + cold-tier token IDs | ~20 MB (1,012x) |
| 4 | **Hybrid RS+CA** | Cached static attention (95.5%) + tiny dynamic KV (4.5%) + vindex FFN | ~150-300 MB |
| 5 | **RS Graph Walk** | Graph lookup only — no matmul, no attention, no FFN | 1.5 MB per-conv |

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
    hybrid_cracked/     Strategy 4: cached static heads + tiny dynamic KV
    graph_walk/         Strategy 5: routing table + vindex lookup
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
followed by Lloyd-Max scalar quantization. 4-6x compression at the Shannon
limit. Still grows O(context_length). Reference: Algorithm 1 from the paper,
MSE-only (no QJL — community confirmed it hurts after softmax).

### Markov Residual Stream
Eliminates the KV cache entirely. The residual stream has the Markov property:
the current residual IS the complete state. Stores a bounded window of recent
residuals plus cold-tier token IDs (4 bytes each). Does NOT grow with context.
Proven bit-perfect (KL = 0.0) on Gemma 3-4B.

### Hybrid RS + Cracked Attention
The near-term practical win. 95.5% of attention heads produce the same output
regardless of entity (cosine 0.942+). Cache those outputs per template. Only
the ~4.5% dynamic heads need real KV cache. FFN is handled by vindex walk
(zero matmul). Result: 15-27x memory reduction at 4K tokens without solving
attention fully.

### RS Graph Walk
The endgame. The forward pass IS a graph walk over three composed graphs
(FFN, attention, residual). Extract the graphs, walk them directly. No matrices,
no multiplication. 348K FFN features in vindex, 34 layers validated with zero
accuracy loss. Currently proven for factual queries; free-form falls back to
Hybrid RS+CA or Markov RS.

## Key numbers

| Metric | Standard KV | TurboQuant 4b | Markov RS | Hybrid RS+CA | Graph Walk |
|--------|------------|---------------|-----------|--------------|------------|
| Memory @ 4K | 285 MB | 74 MB | 18 MB | ~20-37 MB | 16 KB |
| Memory @ 370K | 25.8 GB | 6.6 GB | 20 MB | ~150-300 MB | 1.5 MB |
| Cold storage | 978 MB | ~200 MB | 10 KB | 10 KB | 10 KB |
| Grows O(N)? | yes | yes | no | ~4.5% heads | no |
| Forward pass? | 34L | 34L | window | ~1-2L attn | NO |
| FFN matmuls? | 34L | 34L | 34L | 0 (vindex) | 0 (vindex) |

## Feature flags

- Default: synthetic benchmark only (zero LARQL dependencies)
- `real-model`: enables Phase 2 integration with larql-inference, larql-vindex, etc.

## Spec

Full benchmark specification: [docs/kv-cache-benchmark-spec-v3.md](docs/kv-cache-benchmark-spec-v3.md)
