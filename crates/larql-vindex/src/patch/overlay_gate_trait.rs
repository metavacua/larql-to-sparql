//! Capability-trait conformance for `PatchedVindex`, letting the patch
//! overlay slot in wherever `GateLookup`, `PatchOverrides`, FFN access
//! traits, or compatibility `GateIndex` are expected. Pulled out of
//! `overlay.rs` so the file holding `PatchedVindex`'s own API stays
//! focused.
//!
//! Most methods forward to the inherent `PatchedVindex` impl;
//! `gate_override` reads from the patch overlay (not the base) and
//! `gate_knn_batch` re-ranks per-row to surface inserted slots that
//! the base path would miss.

use ndarray::Array1;

use crate::index::{
    FeatureMeta, Fp4FfnAccess, GateLookup, NativeFfnAccess, PatchOverrides, QuantizedFfnAccess,
};

use super::overlay::PatchedVindex;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

impl GateLookup for PatchedVindex {
    fn gate_knn(&self, layer: usize, residual: &Array1<f32>, top_k: usize) -> Vec<(usize, f32)> {
        self.gate_knn(layer, residual, top_k)
    }

    fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        self.feature_meta(layer, feature)
    }

    fn num_features(&self, layer: usize) -> usize {
        self.num_features(layer)
    }

    fn gate_scores_batch(
        &self,
        layer: usize,
        x: &ndarray::Array2<f32>,
    ) -> Option<ndarray::Array2<f32>> {
        self.base.gate_scores_batch(layer, x)
    }

    fn gate_scores_batch_backend(
        &self,
        layer: usize,
        x: &ndarray::Array2<f32>,
        backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<ndarray::Array2<f32>> {
        self.base.gate_scores_batch_backend(layer, x, backend)
    }

    fn gate_walk(
        &self,
        layer: usize,
        residual: &Array1<f32>,
        top_k: usize,
    ) -> Option<Vec<(usize, f32)>> {
        // Exact batched-gemv selection reads only the base gate matrix,
        // so it is valid exactly when neither a gate override nor a
        // tombstone could change the ranking at this layer — the same
        // short-circuit guard `PatchedVindex::gate_knn` uses. On a
        // patched layer, decline so the caller falls through to
        // `gate_knn`, which this type overrides with the overlay-aware
        // merge (2026-07-30 review, item #13).
        let layer_is_patched =
            self.overrides_gate.has_layer(layer) || self.deleted.iter().any(|&(l, _)| l == layer);
        if layer_is_patched {
            return None;
        }
        self.base.gate_walk(layer, residual, top_k)
    }

    fn gate_knn_batch(&self, layer: usize, x: &ndarray::Array2<f32>, top_k: usize) -> Vec<usize> {
        // The base impl runs a BLAS gemm against the disk-side gate
        // matrix and ignores the patch overlay — so any feature with
        // an overridden gate (e.g. an INSERT slot) wouldn't be in the
        // candidate set. Re-rank per row using the per-row `gate_knn`
        // path, which `PatchedVindex::gate_knn` overrides correctly.
        // Returns the union of selected feature indices across all
        // rows, deduplicated.
        if !self.overrides_gate.has_layer(layer) {
            // No overrides at this layer — base path is correct.
            return self.base.gate_knn_batch(layer, x, top_k);
        }
        let mut selected = std::collections::BTreeSet::<usize>::new();
        for s in 0..x.shape()[0] {
            let row = x.row(s).to_owned();
            let hits = self.gate_knn(layer, &row, top_k);
            for (feat, _) in hits {
                selected.insert(feat);
            }
        }
        selected.into_iter().collect()
    }
}

impl PatchOverrides for PatchedVindex {
    fn down_override(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.base.down_override(layer, feature)
    }

    fn up_override(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.base.up_override(layer, feature)
    }

    fn gate_override(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        // Gate overrides live on the patch overlay (not the base
        // index). Surface them through the trait so the sparse
        // inference fallback can read the strong installed gate.
        self.overrides_gate.get(layer, feature)
    }

    fn has_overrides_at(&self, layer: usize) -> bool {
        self.overrides_gate.has_layer(layer) || self.base.has_overrides_at(layer)
    }

    /// Union of the base's up/down override slots, the overlay's gate
    /// override slots, and the overlay's tombstones at `layer`. A
    /// tombstone wins over any lingering vector override on the same
    /// slot (matching `gate_knn`, which filters deleted features out of
    /// selection regardless of overrides).
    fn override_slots_at(&self, layer: usize) -> Option<Vec<crate::index::OverrideSlot>> {
        let mut slots: std::collections::BTreeMap<usize, bool> = self
            .base
            .override_slots_at(layer)?
            .into_iter()
            .map(|s| (s.feature, s.tombstoned))
            .collect();
        for (l, f, _) in self.overrides_gate.iter() {
            if l == layer {
                slots.entry(f).or_insert(false);
            }
        }
        for &(l, f) in &self.deleted {
            if l == layer {
                slots.insert(f, true);
            }
        }
        Some(
            slots
                .into_iter()
                .map(|(feature, tombstoned)| crate::index::OverrideSlot {
                    feature,
                    tombstoned,
                })
                .collect(),
        )
    }
}

impl NativeFfnAccess for PatchedVindex {
    fn down_feature_vector(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.base.down_feature_vector(layer, feature)
    }

    fn has_down_features(&self) -> bool {
        self.base.has_down_features()
    }

    fn down_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.base.down_layer_matrix(layer)
    }

    fn up_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.base.up_layer_matrix(layer)
    }

    fn has_full_mmap_ffn(&self) -> bool {
        self.base.has_full_mmap_ffn()
    }

    fn has_interleaved(&self) -> bool {
        self.base.has_interleaved()
    }

    fn interleaved_gate(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.base.interleaved_gate(layer)
    }

    fn interleaved_up(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.base.interleaved_up(layer)
    }

    fn interleaved_down(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        self.base.interleaved_down(layer)
    }
}

impl QuantizedFfnAccess for PatchedVindex {
    fn has_interleaved_q4(&self) -> bool {
        self.base.has_interleaved_q4()
    }

    fn interleaved_q4_mmap_ref(&self) -> Option<&[u8]> {
        self.base.interleaved_q4_mmap_ref()
    }

    fn has_interleaved_kquant(&self) -> bool {
        self.base.has_interleaved_kquant()
    }

    fn interleaved_kquant_mmap_ref(&self) -> Option<&[u8]> {
        self.base.interleaved_kquant_mmap_ref()
    }

    fn interleaved_kquant_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 3]> {
        self.base.interleaved_kquant_layer_data(layer)
    }

    fn kquant_ffn_layer(&self, layer: usize, component: usize) -> Option<std::sync::Arc<Vec<f32>>> {
        self.base.kquant_ffn_layer(layer, component)
    }

    fn kquant_ffn_row_into(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        out: &mut [f32],
    ) -> bool {
        self.base.kquant_ffn_row_into(layer, component, feat, out)
    }

    fn kquant_ffn_row_dot(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        x: &[f32],
    ) -> Option<f32> {
        self.base.kquant_ffn_row_dot(layer, component, feat, x)
    }

    fn kquant_ffn_row_scaled_add_via_cache(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        self.base
            .kquant_ffn_row_scaled_add_via_cache(layer, component, feat, alpha, out)
    }

    fn has_down_features_kquant(&self) -> bool {
        self.base.has_down_features_kquant()
    }

    fn kquant_down_feature_scaled_add(
        &self,
        layer: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        self.base
            .kquant_down_feature_scaled_add(layer, feat, alpha, out)
    }

    fn kquant_ffn_row_scaled_add(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        self.base
            .kquant_ffn_row_scaled_add(layer, component, feat, alpha, out)
    }

    fn kquant_matmul_transb(
        &self,
        layer: usize,
        component: usize,
        x: &[f32],
        x_rows: usize,
        backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<Vec<f32>> {
        self.base
            .kquant_matmul_transb(layer, component, x, x_rows, backend)
    }
}

impl Fp4FfnAccess for PatchedVindex {
    fn has_fp4_storage(&self) -> bool {
        self.base.has_fp4_storage()
    }

    fn fp4_ffn_row_dot(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        x: &[f32],
    ) -> Option<f32> {
        self.base.fp4_ffn_row_dot(layer, component, feat, x)
    }

    fn fp4_ffn_row_scaled_add(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        self.base
            .fp4_ffn_row_scaled_add(layer, component, feat, alpha, out)
    }

    fn fp4_ffn_row_into(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        out: &mut [f32],
    ) -> bool {
        self.base.fp4_ffn_row_into(layer, component, feat, out)
    }
}

#[cfg(test)]
mod tests {
    //! Pins for the `gate_walk` trait override (2026-07-30 review,
    //! item #13): exact selection may only serve layers the patch
    //! overlay cannot have changed; patched layers must decline so
    //! callers fall through to the overlay-aware `gate_knn`.
    use super::*;
    use crate::index::core::VectorIndex;
    use larql_models::TopKEntry;
    use ndarray::Array2;

    /// 2-layer × 3-feature × 4-hidden base where feature `f` on each
    /// layer has gate `[3 - f, 0, 0, 0]` — a query along e0 ranks
    /// feature 0 highest, so `gate_walk` has real hits to return.
    fn make_scored_patched() -> PatchedVindex {
        let mut gate = Array2::<f32>::zeros((3, 4));
        for f in 0..3 {
            gate[[f, 0]] = (3 - f) as f32;
        }
        let down_meta = vec![Some(vec![None, None, None]); 2];
        let index = VectorIndex::new(vec![Some(gate.clone()), Some(gate)], down_meta, 2, 4);
        PatchedVindex::new(index)
    }

    fn make_meta(token: &str) -> crate::index::FeatureMeta {
        crate::index::FeatureMeta {
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

    #[test]
    fn gate_walk_delegates_to_base_on_unpatched_layer() {
        let p = make_scored_patched();
        let q = Array1::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0]);
        let via_trait = <PatchedVindex as GateLookup>::gate_walk(&p, 0, &q, 2)
            .expect("unpatched layer must reach the base exact path");
        let via_base = p.base.gate_walk(0, &q, 2).expect("base has gate data");
        assert_eq!(via_trait, via_base);
        assert_eq!(via_trait[0].0, 0, "feature 0 has the largest gate dot");
    }

    #[test]
    fn gate_walk_declines_on_gate_overridden_layer() {
        let mut p = make_scored_patched();
        p.insert_feature(0, 1, vec![0.0, 9.0, 0.0, 0.0], make_meta("ins"));
        let q = Array1::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0]);
        assert!(
            <PatchedVindex as GateLookup>::gate_walk(&p, 0, &q, 2).is_none(),
            "layer with a gate override must decline exact selection"
        );
        // The untouched layer still serves the exact path.
        assert!(<PatchedVindex as GateLookup>::gate_walk(&p, 1, &q, 2).is_some());
    }

    #[test]
    fn gate_walk_declines_on_tombstoned_layer() {
        let mut p = make_scored_patched();
        p.insert_feature(0, 1, vec![0.0, 9.0, 0.0, 0.0], make_meta("tmp"));
        p.delete_feature(0, 1);
        let q = Array1::from_vec(vec![1.0_f32, 0.0, 0.0, 0.0]);
        assert!(
            <PatchedVindex as GateLookup>::gate_walk(&p, 0, &q, 2).is_none(),
            "layer with a tombstone must decline exact selection"
        );
    }

    /// `override_slots_at` merges three sources — base up/down override
    /// maps, overlay gate overrides, overlay tombstones — sorted and
    /// deduplicated, with a tombstone winning over a lingering vector
    /// override on the same slot (matching `gate_knn`'s filtering).
    #[test]
    fn override_slots_at_merges_base_overlay_and_tombstones() {
        use crate::index::{OverrideSlot, PatchOverrides as _};
        let mut p = make_scored_patched();
        // Base up/down overrides land on the inner VectorIndex.
        p.base.set_up_vector(0, 2, vec![1.0; 4]);
        p.base.set_down_vector(0, 2, vec![1.0; 4]);
        // Overlay gate override (INSERT) on slot 1.
        p.insert_feature(0, 1, vec![0.0, 9.0, 0.0, 0.0], make_meta("ins"));
        // Tombstone slot 2 — must override the base's vector overrides.
        p.delete_feature(0, 2);

        let slots = p.override_slots_at(0).expect("overlay enumerates");
        assert_eq!(
            slots,
            vec![
                OverrideSlot {
                    feature: 1,
                    tombstoned: false
                },
                OverrideSlot {
                    feature: 2,
                    tombstoned: true
                },
            ]
        );
        // Unpatched layer → empty enumeration, not None.
        assert_eq!(p.override_slots_at(1), Some(Vec::new()));
    }
}
