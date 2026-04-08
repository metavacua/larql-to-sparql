# ADR-005: PerLayerGraph — Adaptive Per-Layer Strategy

**Status**: Accepted  
**Date**: 2026-04  
**Context**: Different layers have different computational profiles. L0-12 are template-fixed (cache them). L13-23 use stable gate patterns (walk FFN). L24-33 are entity-dependent (need full compute or GPU).

## Decision

`PerLayerGraph` wraps a vector of `LayerGraph` trait objects, one per layer. Each layer can use a different strategy:

- `CachedLayerGraph` — template-fixed layers (returns pre-computed residual)
- `DenseLayerGraph` — full matmul (baseline)
- `WalkLayerGraph` — dense attention + sparse WalkFfn
- `PipelinedLayerGraph` — CPU attention + Metal Q4 FFN

`build_adaptive_graph()` automatically selects the best strategy per layer based on template cosine similarity and available hardware.

## Three regimes (validated)

| Regime | Layers | Strategy | Latency |
|--------|--------|----------|---------|
| Template-fixed | L0-12 | Cache | 0ms |
| Stable knowledge | L13-23 | Walk + cache | 424ms (1.3x dense) |
| Entity-dependent | L24-33 | Dense/GPU | Variable |

## Consequences

- **Good**: Smooth gradient from no GPU to full GPU — each layer independently routed
- **Good**: Composable — strategies can be mixed per layer
- **Trade-off**: More complex than a single uniform strategy. Accepted because the 3-regime structure is empirically validated.
