//! `gate_override_tests` for [`super`].
//!
//! Split out of `overlay.rs` to keep the implementation file within
//! the repo's per-file size budget.

//! Direct unit tests for the gate-override accessors and mutator
//! used by `COMPILE INTO VINDEX WITH REFINE`. The integration tests
//! in `larql-lql` exercise these via the executor; these tests
//! cover them at the API surface so a regression in the layering
//! contract gets caught here without needing the full executor.
use super::*;
use crate::index::core::VectorIndex;
use larql_models::TopKEntry;
use ndarray::Array2;

fn make_meta(token: &str) -> FeatureMeta {
    FeatureMeta {
        top_token: token.into(),
        top_token_id: 0,
        c_score: 0.9,
        top_k: vec![TopKEntry {
            token: token.into(),
            token_id: 0,
            logit: 0.9,
        }],
    }
}

/// A 2-layer × 3-feature × 4-hidden empty base index for these
/// tests. Gate vectors and metas are zero — overrides land on top.
fn make_empty_base() -> PatchedVindex {
    let gate0 = Array2::<f32>::zeros((3, 4));
    let gate1 = Array2::<f32>::zeros((3, 4));
    let down_meta = vec![Some(vec![None, None, None]), Some(vec![None, None, None])];
    let index = VectorIndex::new(vec![Some(gate0), Some(gate1)], down_meta, 2, 4);
    PatchedVindex::new(index)
}

#[test]
fn set_gate_override_replaces_existing_slot() {
    let mut p = make_empty_base();
    p.insert_feature(0, 1, vec![1.0, 0.0, 0.0, 0.0], make_meta("a"));
    p.set_gate_override(0, 1, vec![0.0, 1.0, 0.0, 0.0]);
    let read = p.overrides_gate_at(0, 1).unwrap();
    assert_eq!(read, &[0.0, 1.0, 0.0, 0.0]);
}

#[test]
fn set_gate_override_is_no_op_when_slot_absent() {
    // The contract is "only refine slots that were already touched
    // by a patch" — set_gate_override should NOT create a new entry
    // out of nothing. Verifying this stops a future caller from
    // accidentally inserting half-state (gate without meta).
    let mut p = make_empty_base();
    p.set_gate_override(0, 1, vec![1.0, 1.0, 1.0, 1.0]);
    assert!(p.overrides_gate_at(0, 1).is_none());
}

#[test]
fn overrides_gate_iter_yields_every_inserted_slot() {
    let mut p = make_empty_base();
    p.insert_feature(0, 0, vec![1.0, 0.0, 0.0, 0.0], make_meta("a"));
    p.insert_feature(0, 2, vec![0.0, 1.0, 0.0, 0.0], make_meta("b"));
    p.insert_feature(1, 1, vec![0.0, 0.0, 1.0, 0.0], make_meta("c"));
    let mut entries: Vec<(usize, usize)> =
        p.overrides_gate_iter().map(|(l, f, _)| (l, f)).collect();
    entries.sort();
    assert_eq!(entries, vec![(0, 0), (0, 2), (1, 1)]);
}

#[test]
fn overrides_gate_iter_returns_actual_vectors() {
    let mut p = make_empty_base();
    let g = vec![0.5_f32, -0.5, 0.25, -0.25];
    p.insert_feature(0, 0, g.clone(), make_meta("x"));
    let mut found = false;
    for (l, f, vec) in p.overrides_gate_iter() {
        if (l, f) == (0, 0) {
            assert_eq!(vec, g.as_slice());
            found = true;
        }
    }
    assert!(found, "iter should yield the inserted slot");
}

#[test]
fn set_up_vector_round_trip() {
    // Up overrides parallel down overrides — set, read back, verify.
    // Used by INSERT to write the slot's up component when installing
    // a constellation fact (mutation.rs install_compiled_slot port).
    let mut p = make_empty_base();
    let up = vec![0.3_f32, -0.4, 0.5, -0.6];
    p.set_up_vector(0, 1, up.clone());
    assert_eq!(p.up_override_at(0, 1), Some(up.as_slice()));
    // Different slot is unaffected.
    assert!(p.up_override_at(0, 2).is_none());
}

#[test]
fn up_and_down_overrides_are_independent() {
    // INSERT writes both per layer; verifying they don't overwrite
    // each other's storage (separate HashMaps on the base index).
    let mut p = make_empty_base();
    let up = vec![1.0_f32, 0.0, 0.0, 0.0];
    let down = vec![0.0_f32, 1.0, 0.0, 0.0];
    p.set_up_vector(0, 0, up.clone());
    p.set_down_vector(0, 0, down.clone());
    assert_eq!(p.up_override_at(0, 0), Some(up.as_slice()));
    assert_eq!(p.down_override_at(0, 0), Some(down.as_slice()));
}

#[test]
fn up_overrides_iterator_yields_every_slot() {
    let mut p = make_empty_base();
    p.set_up_vector(0, 0, vec![1.0_f32, 0.0, 0.0, 0.0]);
    p.set_up_vector(0, 2, vec![0.0_f32, 1.0, 0.0, 0.0]);
    p.set_up_vector(1, 1, vec![0.0_f32, 0.0, 1.0, 0.0]);
    let mut keys: Vec<(usize, usize)> = p.up_overrides().keys().copied().collect();
    keys.sort();
    assert_eq!(keys, vec![(0, 0), (0, 2), (1, 1)]);
}

#[test]
fn iter_then_set_round_trip_preserves_other_slots() {
    // Simulate what run_refine_pass does: snapshot via iter,
    // mutate one slot via set_gate_override, verify the other
    // slot's gate is unchanged.
    let mut p = make_empty_base();
    let original_a = vec![1.0_f32, 0.0, 0.0, 0.0];
    let original_b = vec![0.0_f32, 1.0, 0.0, 0.0];
    p.insert_feature(0, 0, original_a.clone(), make_meta("a"));
    p.insert_feature(0, 1, original_b.clone(), make_meta("b"));

    // Snapshot.
    let snapshot: Vec<(usize, usize, Vec<f32>)> = p
        .overrides_gate_iter()
        .map(|(l, f, v)| (l, f, v.to_vec()))
        .collect();
    assert_eq!(snapshot.len(), 2);

    // Mutate slot a only.
    p.set_gate_override(0, 0, vec![0.5, 0.5, 0.0, 0.0]);

    assert_eq!(p.overrides_gate_at(0, 0).unwrap(), &[0.5, 0.5, 0.0, 0.0]);
    assert_eq!(p.overrides_gate_at(0, 1).unwrap(), original_b.as_slice());
}

// ── Coverage for the 2026-05-16 cache + accessor paths ─────────────

/// Second `gate_knn` query at the same layer should hit the
/// `layer_gate_cache` fast path (read-lock branch).
#[test]
fn gate_knn_second_query_uses_cache_fast_path() {
    let mut p = make_empty_base();
    p.insert_feature(0, 1, vec![1.0, 0.0, 0.0, 0.0], make_meta("a"));
    let q = Array1::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0]);
    // First call: builds the cache under a write lock.
    let _ = p.gate_knn(0, &q, 1);
    // Second call: should reach the `g.get(&layer)` branch and
    // skip the rebuild. Result equivalence is the load-bearing
    // assertion; the perf benefit is measured in
    // `larql-server/benches/shard_query.rs`.
    let second = p.gate_knn(0, &q, 1);
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].0, 1);
}

/// After mutating layer 1, layer 0's cached entry must survive
/// (per-layer invalidation, 2026-05-16).
#[test]
fn cross_layer_mutation_preserves_other_layer_cache() {
    let mut p = make_empty_base();
    p.insert_feature(0, 0, vec![1.0, 0.0, 0.0, 0.0], make_meta("l0"));
    let q = Array1::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0]);
    // Warm layer 0's cache.
    let _ = p.gate_knn(0, &q, 1);
    // Mutate layer 1 — should NOT invalidate layer 0's cache.
    p.insert_feature(1, 0, vec![0.0, 1.0, 0.0, 0.0], make_meta("l1"));
    // Re-query layer 0 — still hits the cached path with the
    // same result.
    let hits = p.gate_knn(0, &q, 1);
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, 0);
}

/// `update_feature_meta` overwrites only meta, leaves gate alone.
#[test]
fn update_feature_meta_replaces_meta_only() {
    let mut p = make_empty_base();
    p.insert_feature(0, 0, vec![1.0, 0.0, 0.0, 0.0], make_meta("a"));
    p.update_feature_meta(0, 0, make_meta("b"));
    assert_eq!(p.feature_meta(0, 0).unwrap().top_token, "b");
    // Gate vector untouched.
    assert_eq!(p.overrides_gate_at(0, 0), Some(&[1.0, 0.0, 0.0, 0.0][..]));
}

/// `is_overridden` reports `true` for inserted slots, `false`
/// otherwise. Trivial accessor — pin behavior so a regression in
/// the storage map shape gets caught.
#[test]
fn is_overridden_tracks_inserted_slots() {
    let mut p = make_empty_base();
    assert!(!p.is_overridden(0, 0));
    p.insert_feature(0, 0, vec![1.0, 0.0, 0.0, 0.0], make_meta("a"));
    assert!(p.is_overridden(0, 0));
    assert!(!p.is_overridden(0, 1));
    assert!(!p.is_overridden(1, 0));
}

/// `base()` / `base_mut()` round-trip the underlying VectorIndex.
#[test]
fn base_and_base_mut_expose_the_inner_index() {
    let mut p = make_empty_base();
    assert_eq!(p.base().num_layers, 2);
    assert_eq!(p.base().hidden_size, 4);
    // `base_mut` is used by callers that need to set down/up
    // vectors directly — verify it round-trips.
    let _: &mut VectorIndex = p.base_mut();
}

/// `find_free_feature` picks the first overlay-and-base-free
/// slot.
#[test]
fn find_free_feature_picks_first_overlay_free_slot() {
    // Empty base + empty overlay → slot 0 is free.
    let p = make_empty_base();
    assert_eq!(p.find_free_feature(0), Some(0));

    // Overlay claims slot 0 → next free is slot 1.
    let mut p = make_empty_base();
    p.insert_feature(0, 0, vec![1.0, 0.0, 0.0, 0.0], make_meta("a"));
    assert_eq!(p.find_free_feature(0), Some(1));

    // Overlay claims 0 and 1, base claims 2 (via metadata) →
    // first preference fails (no slot is *both* base-free AND
    // overlay-free); fallback returns the weakest base-claimed
    // slot that the overlay hasn't taken, but there are no
    // overlay-free base-claimed slots here, so the result is
    // `None`.
    let mut p2 = make_empty_base();
    // Inject base metadata at slot 2 so `feature_meta` returns
    // Some — simulates a populated base slot.
    p2.base_mut().metadata.down_meta[0] = Some(vec![None, None, Some(make_meta("base"))]);
    p2.insert_feature(0, 0, vec![1.0, 0.0, 0.0, 0.0], make_meta("a"));
    p2.insert_feature(0, 1, vec![0.0, 1.0, 0.0, 0.0], make_meta("b"));
    // Slot 2 has base metadata but no overlay claim → returned
    // by the fallback (weakest-c_score) loop.
    assert_eq!(p2.find_free_feature(0), Some(2));
}

/// Regression (review 2026-07-30 H3): a zero-width gate vector
/// coexisting with real ones at the same layer used to poison
/// `layer_gate_cache` — `feature_ids` included the empty entry
/// while the flattened matrix skipped its 0 floats, so row slicing
/// read the wrong feature's data or panicked, depending on HashMap
/// iteration order. Empty vectors now force the safe slow path.
/// Rebuilt across iterations to shake out iteration orders.
#[test]
fn gate_knn_survives_zero_width_gate_vector_in_overlay() {
    // Enough repeats that pre-fix the "empty iterated before real"
    // panic ordering is hit with overwhelming probability.
    const ORDER_SHAKE_ITERATIONS: usize = 32;
    for _ in 0..ORDER_SHAKE_ITERATIONS {
        let mut p = make_empty_base();
        p.insert_feature(0, 1, vec![1.0, 0.0, 0.0, 0.0], make_meta("real"));
        // Simulate a legacy/corrupt overlay carrying zero-width
        // rows (insert_feature no longer produces them).
        p.overrides_gate.insert(0, 0, vec![]);
        p.overrides_gate.insert(0, 2, vec![]);
        let q = Array1::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0]);
        // Twice: first call builds (or refuses) the cache, second
        // exercises whatever got cached.
        for _ in 0..2 {
            let hits = p.gate_knn(0, &q, 3);
            assert!(!hits.is_empty());
            assert_eq!(hits[0].0, 1, "real override must rank first");
            assert!(
                (hits[0].1 - 1.0).abs() < 1e-6,
                "real override must score by its own row, got {}",
                hits[0].1
            );
        }
    }
}

/// An empty gate vec means "no gate override": metadata lands,
/// `overrides_gate` stays clean, and the slot still counts as
/// claimed so successive metadata-only INSERTs (Vindexfile) get
/// distinct slots instead of overwriting each other.
#[test]
fn insert_feature_with_empty_gate_is_metadata_only() {
    let mut p = make_empty_base();
    p.insert_feature(0, 0, vec![], make_meta("a"));
    assert_eq!(p.feature_meta(0, 0).unwrap().top_token, "a");
    assert!(p.overrides_gate_at(0, 0).is_none());
    // Slot 0 is claimed via meta → next free slot is 1.
    assert_eq!(p.find_free_feature(0), Some(1));
    // Meta-only inserts must not poison KNN at the layer.
    let q = Array1::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0]);
    let _ = p.gate_knn(0, &q, 3);
}

/// `find_free_feature` returns `None` on a layer with zero features.
#[test]
fn find_free_feature_returns_none_when_layer_empty() {
    // Use an index where layer 0 has zero features to hit the
    // `n == 0` early return.
    let index = VectorIndex::empty(2, 4);
    let p = PatchedVindex::new(index);
    assert!(p.find_free_feature(0).is_none());
}

// ── Tombstone resurrection contract (review 2026-07-30 M6) ─────────

/// Regression (M6): Delete→Update must resurrect the slot for BOTH
/// query paths. Before the fix, `update_feature_meta` left the
/// tombstone in place, so `feature_meta()` said the feature existed
/// while `gate_knn()` permanently filtered it out.
#[test]
fn update_after_delete_resurrects_feature_for_both_paths() {
    let mut p = make_empty_base();
    p.insert_feature(0, 1, vec![1.0, 0.0, 0.0, 0.0], make_meta("a"));
    p.delete_feature(0, 1);
    p.update_feature_meta(0, 1, make_meta("b"));

    // Path 1: metadata lookup sees the feature again.
    assert_eq!(p.feature_meta(0, 1).unwrap().top_token, "b");
    // Path 2: KNN may return it again. The base gate rows are all
    // zeros, so any result set that includes feature 1 proves the
    // tombstone filter no longer applies.
    let q = Array1::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0]);
    let hits = p.gate_knn(0, &q, 3);
    assert!(
        hits.iter().any(|&(f, _)| f == 1),
        "gate_knn must agree the feature exists again, got {hits:?}"
    );
}

/// Pin the other half of the M6 contract: Delete with NO
/// subsequent Update stays tombstoned for both paths.
#[test]
fn delete_without_update_stays_tombstoned_for_both_paths() {
    let mut p = make_empty_base();
    p.insert_feature(0, 1, vec![1.0, 0.0, 0.0, 0.0], make_meta("a"));
    p.delete_feature(0, 1);

    assert!(
        p.feature_meta(0, 1).is_none(),
        "meta path must report deleted"
    );
    let q = Array1::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0]);
    let hits = p.gate_knn(0, &q, 3);
    assert!(
        hits.iter().all(|&(f, _)| f != 1),
        "KNN path must keep filtering the tombstoned slot, got {hits:?}"
    );
}

// ── Deletion oversampling escalation (review 2026-07-30 M11) ───────

/// A 1-layer base whose feature `i` has gate `[n - i, 0, 0, 0]`, so
/// a query along e0 ranks feature 0 highest, descending from there.
/// Metadata is all-`None`; only the gate scores matter here.
fn make_scored_base(n: usize) -> PatchedVindex {
    let mut gate = Array2::<f32>::zeros((n, 4));
    for i in 0..n {
        gate[[i, 0]] = (n - i) as f32;
    }
    let down_meta = vec![Some(vec![None; n])];
    let index = VectorIndex::new(vec![Some(gate)], down_meta, 1, 4);
    PatchedVindex::new(index)
}

/// Regression (M11): with `top_k + 1` tombstones covering the top
/// base hits, the fixed 2× oversample window used to be hollowed
/// out and the caller silently got fewer than `top_k` hits even
/// though live features remained. The escalation must fill `top_k`.
#[test]
fn gate_knn_fills_top_k_despite_deletions_covering_oversample_window() {
    const N: usize = 8;
    const TOP_K: usize = 3;
    let mut p = make_scored_base(N);
    // Delete the TOP_K + 1 highest-scoring features (0..=3): they
    // occupy the top of the 2× (= 6-wide) oversample window.
    for f in 0..=TOP_K {
        p.delete_feature(0, f);
    }
    let q = Array1::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0]);
    let hits = p.gate_knn(0, &q, TOP_K);
    assert_eq!(
        hits.len(),
        TOP_K,
        "must fill top_k from live features, got {hits:?}"
    );
    let feats: Vec<usize> = hits.iter().map(|&(f, _)| f).collect();
    assert_eq!(
        feats,
        vec![4, 5, 6],
        "next-best live features in rank order"
    );
}

/// The escalation ladder's last rung: enough tombstones that even
/// the 4× retry window is fully deleted — the all-features query
/// must still surface the best surviving feature.
#[test]
fn gate_knn_escalates_to_all_features_when_retry_window_is_deleted() {
    const N: usize = 12;
    let mut p = make_scored_base(N);
    // top_k = 1 → 2× window = 2 hits, 4× window = 4 hits. Delete
    // the top 8 so both fixed windows are hollow.
    for f in 0..8 {
        p.delete_feature(0, f);
    }
    let q = Array1::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0]);
    let hits = p.gate_knn(0, &q, 1);
    assert_eq!(hits.len(), 1, "got {hits:?}");
    assert_eq!(hits[0].0, 8, "best surviving feature");
}

/// When every feature is tombstoned the escalation terminates and
/// returns empty rather than looping or panicking.
#[test]
fn gate_knn_returns_empty_when_all_features_deleted() {
    const N: usize = 4;
    let mut p = make_scored_base(N);
    for f in 0..N {
        p.delete_feature(0, f);
    }
    let q = Array1::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0]);
    assert!(p.gate_knn(0, &q, 2).is_empty());
}
