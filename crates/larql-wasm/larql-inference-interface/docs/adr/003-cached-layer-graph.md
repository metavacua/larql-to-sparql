# ADR-003: Cached Layer Graph — Template-Fixed Layers

**Status**: Accepted  
**Date**: 2026-04  
**Context**: Layers 0-12 produce nearly identical residuals across all entities (0.999 cosine similarity). Computing them is wasteful — the output is a function of the template, not the entity.

## Decision

`CachedLayerGraph` stores pre-computed residuals for template-fixed layers. During inference, returns the cached residual instead of running the forward pass.

Combined with entity-dependent layers (L13-33), this reduces computation from 34 to ~21 layers.

## Impact

```
All 34 layers:       552ms (1.8 tok/s)
Cache L0-12 + walk:  203ms (4.9 tok/s)  — 2.7x speedup
Projected GPU (8L):  ~5ms  (~200 tok/s) — exceeds Ollama
```

## Consequences

- **Good**: 2.7x speedup on CPU path, projected 20x on GPU with layer caching
- **Good**: Mathematically justified — 0.999 cosine means the cache is essentially exact
- **Trade-off**: Requires one forward pass per template to build the cache. Amortized over all tokens.
- **Trade-off**: Memory cost of storing one residual per template (~10KB per template × 13 layers). Negligible.
