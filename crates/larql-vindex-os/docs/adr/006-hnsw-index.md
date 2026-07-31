# ADR-006: HNSW Graph Index for Sub-Linear KNN

**Status**: Accepted  
**Date**: 2026-04  
**Context**: Brute-force gate KNN is O(N×D) per query. At 28K features × 8192 hidden (Llama 70B), this takes 107ms/layer — too slow for interactive use.

## Decision

Optional HNSW (Hierarchical Navigable Small World) graph index layered on top of brute-force KNN. Built lazily on first query, cached for subsequent queries.

## Benchmark (dim=2560, M3 Max)

| Features | Brute | HNSW | Speedup | Build Cost |
|----------|-------|------|---------|------------|
| 1,024 | 0.19ms | 0.15ms | 1.3x | 4ms |
| 10,240 | 2.95ms | 1.70ms | 1.7x | 47ms |
| 28,672 | 19.7ms | 15.5ms | 1.3x | 158ms |

## Implementation

- Random projection: dim → 64 via `matmul` (larql-compute BLAS)
- Graph traversal in projected space, exact rescoring in full space
- M=16 connections per node, ef_construction=200
- Level assignment: exponential distribution

## Origin

HNSW algorithm by Malkov & Yashunin (2016). Implementation is original LARQL — built on top of larql-compute's BLAS matmul for the projection step.

## Consequences

- **Good**: Sub-linear query time for large feature counts
- **Good**: Lazy build — no cost until first HNSW query
- **Good**: Exact rescoring on candidates ensures accuracy
- **Trade-off**: Build cost (47-158ms) paid once, amortized over queries
- **Trade-off**: Not useful for small models (brute-force is fast enough under 10K features)
