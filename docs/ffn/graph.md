# GraphFfn — Embedding-Based Feature Selection

**File:** `crates/larql-inference/src/ffn/experimental/graph.rs`
**Status:** Experimental (failed — 0% accuracy)
**Speed:** 2.5x FFN speedup
**Accuracy:** 0%

## Description

Uses the GateIndex (precomputed token→feature map) for feature selection. Projects the
residual against the embedding matrix to find nearest tokens, looks up their precomputed
features, then computes actual gate/up/down for those features only.

## Why it fails

The GateIndex is built from `embedding[token] @ gate.T`. At runtime, the FFN input is the
post-attention residual, not an embedding. Measured feature overlap between embedding-selected
and residual-selected features: **1.5%**. The two spaces are nearly orthogonal after 20+
layers of attention transformation.

## Lesson

No embedding-space proxy can predict which features the residual will activate. The gate
matmul on the actual residual is irreducible.
