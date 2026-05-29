# ADR-001: Transformer Weights as a Queryable Database

**Status**: Accepted  
**Date**: 2026-03  
**Context**: Traditional inference loads all weights into RAM. LARQL needs selective access to individual features.

## Decision

Store transformer weights in a "vindex" — a directory of specialized binary files that can be queried like a database. Each FFN feature (gate row + down column) is independently addressable.

## Key Design

```
model.vindex/
  gate_vectors.bin     — W_gate per layer (KNN index)
  down_features.bin    — Feature-major down vectors (zero-copy slice)
  up_features.bin      — Feature-major up vectors
  embeddings.bin       — W_embed matrix
  index.json           — Config + metadata
```

Gate KNN: `scores = W_gate[layer] @ residual` → top-K features.
Walk: accumulate selected features' up/down contributions.

## Origin

Original LARQL design. No known prior art for treating transformer FFN weights as a queryable database format.

## Consequences

- **Good**: 88-171x RAM reduction (70B model in 4.9GB vs 130GB)
- **Good**: Individual features are inspectable and editable
- **Good**: Selective computation (only compute top-K features per layer)
- **Trade-off**: Requires extraction step (build vindex from safetensors)
- **Trade-off**: Walk accuracy depends on top-K selection quality
