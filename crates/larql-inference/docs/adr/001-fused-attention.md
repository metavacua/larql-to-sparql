# ADR-001: BLAS-Fused Online Softmax Attention

**Status**: Accepted  
**Date**: 2026-03  
**Context**: Need attention that handles GQA, softcap, and arbitrary sequence lengths without allocating the [seq, seq] attention matrix.

## Decision

Use BLAS `gemv` inside an online-softmax loop. For each query position:
1. `scores = K[0..=qi] @ Q[qi]` (BLAS gemv, AMX-accelerated)
2. Scale + optional softcap + two-pass softmax (f64 accumulation)
3. `output = V[0..=qi]^T @ softmax_scores` (BLAS gemv)

Never allocates the attention matrix. GQA handled by mapping Q heads to KV heads.

## Consequences

- **Good**: 1.6x faster than materialized attention at head_dim=256
- **Good**: Memory-constant in sequence length (no [seq, seq] allocation)
- **Good**: Supports softcap (Gemma 2), GQA (all modern models), attention weight capture
- **Trade-off**: Per-position BLAS calls have overhead at very short sequences (seq<4). Acceptable because decode is always seq=1.
