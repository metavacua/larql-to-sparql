//! PatchedVindex — runtime overlay on an immutable base index.
//!
//! Holds the resolved override maps (`overrides_meta`, `overrides_gate`,
//! `deleted`) plus the L0 `KnnStore`. Knows how to apply a `VindexPatch`
//! (from `super::format`) to its overlay state, query the result via
//! `gate_knn` / `walk` / `feature_meta`, and bake everything back into
//! a clean `VectorIndex` via `bake_down`.
//!
//! The on-the-wire patch format (`VindexPatch`, `PatchOp`,
//! `PatchDownMeta`, base64 helpers) lives in `super::format`.

use std::collections::HashMap;

use ndarray::Array1;

use crate::index::{FeatureMeta, VectorIndex, WalkHit, WalkTrace};

use super::format::VindexPatch;

// ═══════════════════════════════════════════════════════════════
// PatchedVindex — overlay on immutable base
// ═══════════════════════════════════════════════════════════════

/// A vindex with patches applied as an overlay.
/// The base **files on disk** are never modified.
///
/// ## Layering: gate overrides vs down vector overrides
///
/// `PatchedVindex` deliberately stores its overrides in **two different
/// places** depending on what they are:
///
/// - **Gate vectors** (`insert_feature`, `update_feature_meta`) live in
///   `self.overrides_gate` and `self.overrides_meta` — true overlays
///   that don't touch the base. `gate_knn` consults these on top of the
///   base scores.
///
/// - **Down vectors** (`set_down_vector`) are forwarded to
///   `self.base.set_down_vector`, which mutates the base's
///   `down_overrides` HashMap in place. The base files on disk remain
///   unchanged, but the in-memory base picks up the override directly.
///   `walk_ffn`'s `down_override(layer, feat)` lookup then finds the
///   override on the base.
///
/// This asymmetry is **intentional** and load-bearing for
/// `COMPILE INTO VINDEX`. The dense FFN inference path
/// (`walk_ffn_full_mmap`) reads gate scores from `gate_vectors.bin` via
/// `gate_scores_batch`. If the inserted (norm-matched) gate vector were
/// baked into that file, the dense activation at the inserted slot
/// would become moderate-to-large; combined with the override down
/// vector (multi-layer constellation install at α=0.25 per layer) the
/// residual stream blows up. Keeping the source's weak free-slot gate
/// at the inserted index leaves the dense activation small, so
/// `small_activation × poseidon_vector` per layer accumulates into the
/// validated constellation effect.
///
/// `COMPILE INTO VINDEX` therefore:
///   - Hard-links `gate_vectors.bin` from source (unchanged), and
///   - Bakes the down vectors into `down_weights.bin` via column-rewrite
///     at the inserted slots.
///
/// This is why `down_overrides()` reaches through to the base while
/// `overrides_gate_at()` reads the patch overlay — the two types of
/// override live in different places by design. Don't "fix" this by
/// moving down vectors into a separate overlay map, or you'll have to
/// re-solve the activation-blowup problem.
pub struct PatchedVindex {
    /// Immutable base index. Note: `set_down_vector` mutates
    /// `base.down_overrides` in place — see the layering doc above.
    pub base: VectorIndex,
    /// Applied patches (in order).
    pub patches: Vec<VindexPatch>,
    /// Resolved meta overrides: (layer, feature) → effective metadata.
    /// Later patches override earlier ones for the same feature.
    pub(crate) overrides_meta: HashMap<(usize, usize), Option<FeatureMeta>>,
    /// Resolved gate vector overrides: (layer, feature) → gate vector.
    /// Lives in the overlay (not on `base`) so that the source
    /// `gate_vectors.bin` stays clean — see layering doc above.
    pub(crate) overrides_gate: HashMap<(usize, usize), Vec<f32>>,
    /// Tombstones for deleted features.
    pub(crate) deleted: std::collections::HashSet<(usize, usize)>,
    /// Architecture B: per-layer retrieval-override KNN store.
    pub knn_store: super::knn_store::KnnStore,
}

impl PatchedVindex {
    /// Create a patched vindex from a base index.
    pub fn new(base: VectorIndex) -> Self {
        Self {
            base,
            patches: Vec::new(),
            overrides_meta: HashMap::new(),
            overrides_gate: HashMap::new(),
            deleted: std::collections::HashSet::new(),
            knn_store: super::knn_store::KnnStore::default(),
        }
    }

    /// Insert a feature directly into the overlay (auto-patch mode).
    pub fn insert_feature(
        &mut self,
        layer: usize,
        feature: usize,
        gate_vec: Vec<f32>,
        meta: FeatureMeta,
    ) {
        let key = (layer, feature);
        self.overrides_meta.insert(key, Some(meta));
        self.overrides_gate.insert(key, gate_vec);
        self.deleted.remove(&key);
    }

    /// Delete a feature via the overlay.
    pub fn delete_feature(&mut self, layer: usize, feature: usize) {
        let key = (layer, feature);
        self.overrides_meta.insert(key, None);
        self.deleted.insert(key);
        self.overrides_gate.remove(&key);
    }

    /// Update feature metadata via the overlay.
    pub fn update_feature_meta(&mut self, layer: usize, feature: usize, meta: FeatureMeta) {
        let key = (layer, feature);
        self.overrides_meta.insert(key, Some(meta));
    }

    /// Check if a (layer, feature) has been overridden.
    pub fn is_overridden(&self, layer: usize, feature: usize) -> bool {
        self.overrides_meta.contains_key(&(layer, feature))
    }

    /// Access the underlying base index (readonly).
    pub fn base(&self) -> &VectorIndex {
        &self.base
    }

    /// Access the underlying base index (mutable, for down vector overrides).
    pub fn base_mut(&mut self) -> &mut VectorIndex {
        &mut self.base
    }

    /// Set a down vector override for a feature.
    pub fn set_down_vector(&mut self, layer: usize, feature: usize, vector: Vec<f32>) {
        self.base.set_down_vector(layer, feature, vector);
    }

    /// Set an up vector override for a feature. Mirrors
    /// `set_down_vector`; both forward to the base index. INSERT calls
    /// this so the slot's activation `silu(gate · x) * (up · x)`
    /// reflects the constellation install.
    pub fn set_up_vector(&mut self, layer: usize, feature: usize, vector: Vec<f32>) {
        self.base.set_up_vector(layer, feature, vector);
    }

    /// All in-memory up vector overrides on the underlying base vindex.
    /// Parallel to `down_overrides()`. Used by `COMPILE INTO VINDEX` to
    /// bake them into a fresh copy of `up_features.bin`.
    pub fn up_overrides(&self) -> &std::collections::HashMap<(usize, usize), Vec<f32>> {
        self.base.up_overrides()
    }

    /// Up vector override for `(layer, feature)`. Forwards to the base
    /// vindex (up vectors live on `VectorIndex.up_overrides`, not on the
    /// patch overlay — same layering as `down_override_at`).
    pub fn up_override_at(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.base.up_override_at(layer, feature)
    }

    /// All in-memory down vector overrides on the underlying base vindex.
    /// Used by `COMPILE INTO VINDEX` to bake them into a fresh copy of
    /// `down_weights.bin`.
    ///
    /// For a single (layer, feature) lookup, use `down_override_at`.
    pub fn down_overrides(&self) -> &std::collections::HashMap<(usize, usize), Vec<f32>> {
        self.base.down_overrides()
    }

    /// Down vector override for `(layer, feature)`, if any. Forwards to
    /// the base vindex (down vectors live on `VectorIndex.down_overrides`,
    /// not on the patch overlay — see the layering doc on `PatchedVindex`).
    pub fn down_override_at(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.base.down_override_at(layer, feature)
    }

    /// Override gate vector for `(layer, feature)`, if present in the
    /// patch overlay. Used by `COMPILE INTO VINDEX` to read each
    /// inserted gate vector for sidecar serialisation.
    pub fn overrides_gate_at(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.overrides_gate.get(&(layer, feature)).map(|v| v.as_slice())
    }

    /// Read-only iterator over every gate override slot in the overlay.
    /// Used by `COMPILE INTO VINDEX WITH REFINE` to enumerate the
    /// constellation before refining.
    pub fn overrides_gate_iter(
        &self,
    ) -> impl Iterator<Item = (usize, usize, &[f32])> + '_ {
        self.overrides_gate
            .iter()
            .map(|(&(l, f), v)| (l, f, v.as_slice()))
    }

    /// Replace the gate override for `(layer, feature)` with a new
    /// vector. Used by `COMPILE INTO VINDEX WITH REFINE` to write the
    /// refined gate back into the overlay before the bake step. Has no
    /// effect if the slot does not already have a gate override (we
    /// only refine slots that were already touched by a patch).
    pub fn set_gate_override(&mut self, layer: usize, feature: usize, vector: Vec<f32>) {
        let key = (layer, feature);
        if self.overrides_gate.contains_key(&key) {
            self.overrides_gate.insert(key, vector);
        }
    }

    /// Find a free feature slot at this layer that is NOT already
    /// claimed by the patch overlay. The base index only knows about
    /// its own gate matrix and `down_meta`, so its
    /// `find_free_feature` keeps returning the same "weakest" slot
    /// across calls — which is catastrophic for multi-fact INSERT:
    /// every new INSERT picks the same slot and overwrites the
    /// previous install (validated by the `refine_demo` "last fact
    /// always wins" diagnostic). This wrapper asks the base for
    /// candidate slots and skips any that the overlay has already
    /// taken, scanning linearly until it finds one that's free both
    /// in the base AND in the overlay.
    pub fn find_free_feature(&self, layer: usize) -> Option<usize> {
        let n = self.base.num_features(layer);
        if n == 0 {
            return None;
        }

        // First preference: a slot with no base metadata AND no
        // overlay entry. This matches the base's "no metadata = free"
        // semantics but also respects the overlay.
        for i in 0..n {
            let taken_by_base = self.base.feature_meta(layer, i).is_some();
            let taken_by_overlay = self.overrides_gate.contains_key(&(layer, i));
            if !taken_by_base && !taken_by_overlay {
                return Some(i);
            }
        }

        // Second preference: a slot with base metadata (some c_score)
        // that the overlay has NOT claimed, picking the weakest c_score.
        // This mirrors the base's fallback path but filters out
        // overlay-claimed slots.
        let mut weakest_idx: Option<usize> = None;
        let mut weakest_score = f32::MAX;
        for i in 0..n {
            if self.overrides_gate.contains_key(&(layer, i)) {
                continue;
            }
            if let Some(meta) = self.base.feature_meta(layer, i) {
                if meta.c_score < weakest_score {
                    weakest_score = meta.c_score;
                    weakest_idx = Some(i);
                }
            }
        }
        weakest_idx
    }


    /// Look up feature metadata, checking overrides first.
    pub fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        let key = (layer, feature);
        if let Some(override_meta) = self.overrides_meta.get(&key) {
            return override_meta.clone();
        }
        if self.deleted.contains(&key) {
            return None;
        }
        self.base.feature_meta(layer, feature)
    }

    /// Gate KNN with patched vectors.
    /// For features with overridden gate vectors, uses the patch vector.
    /// For deleted features, excludes them from results.
    pub fn gate_knn(&self, layer: usize, residual: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        let mut hits = self.base.gate_knn(layer, residual, top_k * 2); // oversample

        // Apply gate vector overrides
        for (&(l, f), gate_vec) in &self.overrides_gate {
            if l != layer { continue; }
            let score: f32 = gate_vec.iter()
                .zip(residual.iter())
                .map(|(a, b)| a * b)
                .sum();
            // Update or insert
            if let Some(hit) = hits.iter_mut().find(|(feat, _)| *feat == f) {
                hit.1 = score;
            } else {
                hits.push((f, score));
            }
        }

        // Remove deleted features
        hits.retain(|(f, _)| !self.deleted.contains(&(layer, *f)));

        // Re-sort and truncate
        hits.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        hits.truncate(top_k);
        hits
    }

    /// Walk with patch overrides.
    pub fn walk(&self, residual: &Array1<f32>, layers: &[usize], top_k: usize) -> WalkTrace {
        let mut trace_layers = Vec::with_capacity(layers.len());
        for &layer in layers {
            let hits = self.gate_knn(layer, residual, top_k);
            let walk_hits: Vec<WalkHit> = hits
                .into_iter()
                .filter_map(|(feature, gate_score)| {
                    let meta = self.feature_meta(layer, feature)?.clone();
                    Some(WalkHit { layer, feature, gate_score, meta })
                })
                .collect();
            trace_layers.push((layer, walk_hits));
        }
        WalkTrace { layers: trace_layers }
    }

    /// Flatten all patches into the base, producing a new clean VectorIndex (heap mode).
    pub fn bake_down(&self) -> VectorIndex {
        let mut new_gate = Vec::new();
        let mut new_meta = Vec::new();

        for layer in 0..self.base.num_layers {
            // Get base gate vectors (from heap or mmap)
            let base_gate = if let Some(g) = self.base.gate_vectors_at(layer) {
                Some(g.clone())
            } else if let Some(ref mmap) = self.base.gate_mmap_bytes {
                // Mmap mode — decode this layer's slice to an Array2
                self.base.gate_mmap_slices.get(layer).and_then(|slice| {
                    if slice.num_features == 0 { return None; }
                    let bpf = crate::config::dtype::bytes_per_float(self.base.gate_mmap_dtype);
                    let byte_offset = slice.float_offset * bpf;
                    let byte_count = slice.num_features * self.base.hidden_size * bpf;
                    let byte_end = byte_offset + byte_count;
                    if byte_end > mmap.len() { return None; }
                    let floats = crate::config::dtype::decode_floats(
                        &mmap[byte_offset..byte_end], self.base.gate_mmap_dtype
                    );
                    ndarray::Array2::from_shape_vec(
                        (slice.num_features, self.base.hidden_size), floats
                    ).ok()
                })
            } else {
                None
            };

            let gate = base_gate.map(|mut g| {
                // Apply gate vector overrides
                for (&(l, f), vec) in &self.overrides_gate {
                    if l != layer { continue; }
                    if f < g.shape()[0] && vec.len() == g.shape()[1] {
                        for (j, val) in vec.iter().enumerate() {
                            g[[f, j]] = *val;
                        }
                    }
                }
                g
            });
            new_gate.push(gate);

            // Build metadata from heap or mmap
            let num_features = self.base.num_features(layer);
            let mut new_metas: Vec<Option<FeatureMeta>> = if let Some(heap) = self.base.down_meta_at(layer) {
                heap.to_vec()
            } else if num_features > 0 {
                // Mmap: read each feature on demand
                (0..num_features).map(|f| self.base.feature_meta(layer, f)).collect()
            } else {
                Vec::new()
            };

            // Apply meta overrides
            for (&(l, f), override_meta) in &self.overrides_meta {
                if l != layer { continue; }
                while new_metas.len() <= f { new_metas.push(None); }
                new_metas[f] = override_meta.clone();
            }
            // Apply deletes
            for &(l, f) in &self.deleted {
                if l == layer && f < new_metas.len() { new_metas[f] = None; }
            }

            new_meta.push(if new_metas.is_empty() { None } else { Some(new_metas) });
        }

        VectorIndex::new(new_gate, new_meta, self.base.num_layers, self.base.hidden_size)
    }

    /// Number of active patches.
    pub fn num_patches(&self) -> usize {
        self.patches.len()
    }

    /// Total override count.
    pub fn num_overrides(&self) -> usize {
        self.overrides_meta.len()
    }

    // ── Forwarding methods to base (for compatibility) ──

    /// Layers that have gate vectors loaded (delegates to base).
    pub fn loaded_layers(&self) -> Vec<usize> {
        self.base.loaded_layers()
    }

    /// Number of features at a layer (delegates to base).
    pub fn num_features(&self, layer: usize) -> usize {
        self.base.num_features(layer)
    }

    /// Access down metadata for a layer (base only — does not include overrides).
    /// For override-aware lookups, use `feature_meta()`.
    pub fn down_meta_at(&self, layer: usize) -> Option<&[Option<FeatureMeta>]> {
        self.base.down_meta_at(layer)
    }

    /// Access gate vectors matrix for a layer (base only).
    pub fn gate_vectors_at(&self, layer: usize) -> Option<&ndarray::Array2<f32>> {
        self.base.gate_vectors_at(layer)
    }

    /// Number of layers (delegates to base).
    pub fn num_layers(&self) -> usize {
        self.base.num_layers
    }

    /// Hidden size (delegates to base).
    pub fn hidden_size(&self) -> usize {
        self.base.hidden_size
    }
}


#[cfg(test)]
mod gate_override_tests {
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
            top_k: vec![TopKEntry { token: token.into(), token_id: 0, logit: 0.9 }],
        }
    }

    /// A 2-layer × 3-feature × 4-hidden empty base index for these
    /// tests. Gate vectors and metas are zero — overrides land on top.
    fn make_empty_base() -> PatchedVindex {
        let gate0 = Array2::<f32>::zeros((3, 4));
        let gate1 = Array2::<f32>::zeros((3, 4));
        let down_meta = vec![
            Some(vec![None, None, None]),
            Some(vec![None, None, None]),
        ];
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
}
