# EntityRoutedFfn — Preselected Entity Features

**File:** `crates/larql-inference/src/ffn/experimental/entity_routed.rs`
**Status:** Experimental (failed — 0% accuracy)
**Speed:** 4.2x FFN speedup at K=64
**Accuracy:** 0%

## Description

Resolves entity tokens once at construction (via embedding projection or raw token IDs),
then reuses those token IDs to look up features from the GateIndex at every layer. No gate
matmul and no per-layer embedding projection.

## Why it fails

Same root cause as GraphFfn: the GateIndex maps embeddings to features, but the FFN operates
on post-attention residuals. The token→feature mapping doesn't transfer to the residual space.

## FFN Bench Results

| K | FFN time | vs Dense |
|---|----------|----------|
| 64 | 1637us | **4.21x** |
| 256 | 2884us | 2.39x |
| 1024 | 2782us | 2.48x |

Fast because it eliminates both the gate matmul and the embedding projection. The speed is
real; the feature selection is wrong.
