# ADR-002: WalkFfn — Zero-Copy Mmap'd Down Projection

**Status**: Accepted  
**Date**: 2026-04  
**Context**: The dense down projection (hidden × intermediate matmul) is the single largest operation per layer. Vindex stores down vectors in feature-major layout (contiguous per-feature). Can we read them directly instead of computing the matmul?

## Decision

WalkFfn replaces the dense down projection with a zero-copy mmap read from `down_features.bin`:

1. Gate + up projections from model weights (exact, same as dense)
2. GEGLU activation (exact, same as dense)
3. Down projection: for each active feature (above threshold), read the mmap'd down vector and accumulate weighted by activation

The mmap'd feature-major layout has better page cache behavior than the safetensors weight layout.

## Measured Impact

```
Dense FFN:  535ms/token (34 layers)
Walk FFN:   517ms/token (34 layers)
Speedup:    1.03x (walk is faster, not just equivalent)
```

## Consequences

- **Good**: Faster than dense (better cache locality from feature-major layout)
- **Good**: Enables walk-only mode (drop 13GB of FFN weights from memory)
- **Good**: Zero-copy — reads directly from mmap, no heap allocation
- **Trade-off**: Requires pre-built `down_features.bin` (transpose of weight matrix). Build cost amortized over all inference.
