# FeatureListFfn — Precomputed Feature Lists

**File:** `crates/larql-inference/src/ffn/experimental/feature_list.rs`
**Status:** Experimental (failed — 0-30% accuracy)
**Speed:** ~1x
**Accuracy:** 0-30%

## Description

Runs one calibration forward pass to record which features the gate matmul selects at each
layer. Stores just the feature IDs (not outputs). At query time, attention runs live, and
FFN computes gate/up/down only for the preselected features. Storage: 6.8KB per entity
(50 features x 34 layers x 4 bytes).

## Why it fails

**Cascade drift.** The preselected features produce slightly different FFN outputs than the
dense path (because they're a subset). This changes the residual at the next layer, which
changes the attention output, which changes the FFN input, which selects different features
from what was precomputed. By layer 34 the accumulated drift destroys accuracy.

At K=8092 (nearly all features): 30% match. Even near-complete feature coverage can't prevent
the cascade because the features were selected from a different (dense) residual trajectory.

## The CachedFfn contrast

CachedFfn works (100% match) because it caches the exact FFN **outputs**, not just the feature
selection. The outputs are replayed exactly, keeping the residual stream identical. FeatureListFfn
re-computes outputs from a drifted residual, producing wrong values.

## Lesson

For exact FFN replacement, you must cache the outputs, not the feature selection. Feature lists
are only valid for the exact residual trajectory they were calibrated on.
