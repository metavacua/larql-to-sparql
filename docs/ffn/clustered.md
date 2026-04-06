# ClusteredFfn — K-Means Gate Vector Clusters

**File:** `crates/larql-inference/src/ffn/experimental/clustered.rs`
**Status:** Experimental (failed — 0% accuracy)
**Speed:** 2.3x at c1 (1 cluster probe)
**Accuracy:** 0% at all configurations

## Description

K-means clustering of gate vectors per layer. At runtime, projects residual against cluster
centroids (128 comparisons), selects top-C clusters, computes gate/up/down only for features
in those clusters.

## Why it fails

The gate activation pattern is **distributed**, not clustered. The top-64 features for any
given input are spread across many clusters. Probing 1 cluster (~80 features) misses most
active features. Probing enough clusters to cover them (32+) eliminates the speed advantage.

## Configurations tested

| Clusters | top_c | Features | Match | Speed |
|----------|-------|----------|-------|-------|
| 128 | 1 | ~80 | 0% | 2.3x |
| 128 | 2 | ~160 | 0% | 0.9x |
| 128 | 8 | ~640 | 0% | 0.6x |
| 128 | 32 | ~2088 | 20% | 0.2x |
