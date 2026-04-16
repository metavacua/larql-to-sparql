# DownClusteredFfn — Output-Directed Feature Selection

**File:** `crates/larql-inference/src/ffn/experimental/down_clustered.rs`
**Status:** Experimental (failed — 0% accuracy)
**Speed:** ~1x (no speedup)
**Accuracy:** 0% at all configurations

## Description

Clusters features by their **down projection** (what they output) instead of their gate vector
(what they respond to). The hypothesis: the residual points toward the answer region, so
matching against down-vector centroids should find features that produce the right output.

## Why it fails

The residual direction and the down-vector direction are in different bases. The FFN
transformation (gate x up x down) is what connects them. The residual at the FFN input
doesn't align with the answer direction in down-vector space.

## Configurations tested

All 0% accuracy across 64/128/256 clusters with 1-8 probes each.
