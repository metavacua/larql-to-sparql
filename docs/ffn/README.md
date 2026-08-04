# FFN Backends

Feed-forward network backends for LARQL inference. Each backend implements the `FfnBackend` trait
and can be swapped into the forward pass.

## Production Backends

| Backend | File | Description | Use case |
|---------|------|-------------|----------|
| [WeightFfn](weight.md) | `ffn/weight.rs` | Dense matmul, architecture-correct | Ground truth inference |
| [SparseFfn](sparse.md) | `ffn/sparse.rs` | Gate matmul + top-K sparse up/down | Sparse inference research |
| LayerFfnRouter | `ffn/mod.rs` | Per-layer backend selection | Hybrid strategies |
| HighwayFfn | `ffn/mod.rs` | Returns zeros (skip FFN) | Layer skipping experiments |
| [WalkFfn](walk.md) | `vindex/walk_ffn/` | Gate KNN + sparse FFN + runtime trace | **INFER with interpretability** |

## Experimental Backends

Research backends from FFN optimization work. All in `ffn/experimental/`.

| Backend | File | Speed | Accuracy | Why it fails |
|---------|------|-------|----------|-------------|
| [CachedFfn](cached.md) | `cached.rs` | 4160x (1us/layer) | 100% bit-identical | Not scalable: one cache per prompt |
| [GraphFfn](graph.md) | `graph.rs` | 2.5x | 0% | Embedding != residual (1.5% feature overlap) |
| [EntityRoutedFfn](entity_routed.md) | `entity_routed.rs` | 4.2x | 0% | Same root cause as GraphFfn |
| [ClusteredFfn](clustered.md) | `clustered.rs` | 2.3x (c1) | 0% | Gate activations are distributed, not clustered |
| [DownClusteredFfn](down_clustered.md) | `down_clustered.rs` | ~1x | 0% | Residual direction != answer direction |
| [FeatureListFfn](feature_list.md) | `feature_list.rs` | ~1x | 0-30% | Cascade drift from early sparse layers |

## Key Finding

The gate matmul (`residual @ gate.T`) is **irreducible** for novel residuals. No precomputed
index, clustering, or proxy can predict which features activate without seeing the actual
post-attention residual. Every approach that skips the gate matmul selects the wrong features.

**WalkFfn** uses vindex gate KNN for feature selection (exact `gate_walk` gemv on the hot
path), then runs sparse FFN on only the selected features. Accepts any `GateIndex` implementor
(`VectorIndex` or `PatchedVindex`), so INSERT/DELETE/UPDATE to the vindex immediately affect
inference output. It is the production path for *instrumented and edited* execution (INFER,
traces, patches) and the CPU sparse path — raw decode throughput is served by the Q4K GPU
decode path (~88 tok/s Gemma 3 4B on M3 Max vs ~1.9 tok/s CPU INFER walk; repo README
"Benchmarks").

## Bottleneck Analysis (2026-04 measurement, CPU BLAS)

FFN layer 20, Gemma-3-4b (seq_len=6, hidden=2560, intermediate=10240):

```
gate matmul (x @ gate.T)    1933us   31.3%
up matmul (x @ up.T)        1948us   31.5%
SiLU + element mul            124us    2.0%
down matmul (act @ down.T)  2179us   35.2%
────────────────────────────────────────────
Total dense FFN              6184us  100.0%
```

Three equal matmuls. No single bottleneck. Sparse can't beat dense because the gate matmul
(needed for feature selection) costs as much as the other two matmuls combined.
