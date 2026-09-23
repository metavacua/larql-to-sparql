//! `GateOverlay` — the single override-row retrieval structure shared by
//! `PatchedVindex::gate_knn` and the arch-B `KnnStore`.
//!
//! Holds `(layer, slot) → row` vectors plus a lazily-built per-layer
//! contiguous matrix cache for cache-friendly scoring. Before the
//! 2026-07-31 unification (review item 21) `KnnStore` carried its own
//! parallel copy of this machinery (`key_matrices` + `dirty` tracking);
//! now both retrieval paths score override rows through this one kernel,
//! so the hardening added during the 2026-07-30 campaign — the
//! zero-width-row guard (H3), mixed-width fallback, per-layer cache
//! invalidation — applies to every consumer by construction.
//!
//! Scoring policy is deliberately NOT part of this type: `gate_knn`
//! ranks by `|score|` merged with base hits, `KnnStore` ranks by raw
//! cosine. Match the statistic to the operation — the shared piece is
//! the storage + dot-product kernel, not the ranking.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use ndarray::Array1;

/// Per-layer contiguous row snapshot. Built on first `score_layer`
/// call at a layer, dropped by any mutation touching that layer.
/// Memory cost: one f32 per element per cached layer (doubles the
/// override storage); pays off after ≥1 reuse per build.
#[derive(Debug)]
struct LayerRowCache {
    /// Slot IDs in row-order of `matrix`.
    slot_ids: Vec<usize>,
    /// `n × d` row-major; n = `slot_ids.len()`.
    matrix: Vec<f32>,
    /// Row width for slicing.
    d: usize,
}

/// Override-row store with a unified scoring kernel. See module doc.
#[derive(Debug, Default)]
pub(crate) struct GateOverlay {
    /// Canonical storage: `(layer, slot) → row`. Zero-width rows are
    /// storable (legacy/corrupt overlays) — scoring falls back to the
    /// slow path for the whole layer rather than mis-slicing.
    rows: HashMap<(usize, usize), Vec<f32>>,
    /// Lazy per-layer flattened snapshots. `Arc` lets scoring read a
    /// snapshot without holding the lock across the matvec.
    cache: RwLock<HashMap<usize, Arc<LayerRowCache>>>,
}

impl Clone for GateOverlay {
    fn clone(&self) -> Self {
        Self {
            rows: self.rows.clone(),
            cache: RwLock::new(HashMap::new()),
        }
    }
}

impl GateOverlay {
    /// Insert or replace a row. Invalidates the layer's snapshot.
    pub(crate) fn insert(&mut self, layer: usize, slot: usize, row: Vec<f32>) {
        self.rows.insert((layer, slot), row);
        self.invalidate_layer(layer);
    }

    /// Remove a row. Invalidates the layer's snapshot even when the
    /// row was absent — callers use removal to mean "this slot must
    /// not score", and a stale snapshot is never worth the risk.
    pub(crate) fn remove(&mut self, layer: usize, slot: usize) -> Option<Vec<f32>> {
        self.invalidate_layer(layer);
        self.rows.remove(&(layer, slot))
    }

    /// Drop every row at `layer`. Used by `KnnStore` entity removal,
    /// which renumbers the surviving entries.
    pub(crate) fn remove_layer(&mut self, layer: usize) {
        self.rows.retain(|&(l, _), _| l != layer);
        self.invalidate_layer(layer);
    }

    /// Replace the row for an EXISTING slot; no-op (returning `false`)
    /// when the slot has no row. `set_gate_override` semantics: refine
    /// passes may only touch slots already claimed by a patch.
    pub(crate) fn replace_existing(&mut self, layer: usize, slot: usize, row: Vec<f32>) -> bool {
        match self.rows.get_mut(&(layer, slot)) {
            Some(existing) => {
                *existing = row;
                self.invalidate_layer(layer);
                true
            }
            None => false,
        }
    }

    /// Remove every row and snapshot.
    pub(crate) fn clear(&mut self) {
        self.rows.clear();
        if let Ok(mut c) = self.cache.write() {
            c.clear();
        }
    }

    pub(crate) fn get(&self, layer: usize, slot: usize) -> Option<&[f32]> {
        self.rows.get(&(layer, slot)).map(|v| v.as_slice())
    }

    pub(crate) fn contains(&self, layer: usize, slot: usize) -> bool {
        self.rows.contains_key(&(layer, slot))
    }

    pub(crate) fn has_layer(&self, layer: usize) -> bool {
        self.rows.keys().any(|&(l, _)| l == layer)
    }

    /// Test-only assertion helper; production paths query per-layer
    /// (`has_layer`) or per-slot (`contains`/`get`).
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (usize, usize, &[f32])> + '_ {
        self.rows.iter().map(|(&(l, s), v)| (l, s, v.as_slice()))
    }

    /// Score every row at `layer` against `residual` (plain dot
    /// product; rows stored L2-normalized by the caller make this a
    /// cosine). Returns `(slot, score)` in unspecified order — ranking
    /// policy belongs to the caller.
    ///
    /// Fast path: the flattened per-layer snapshot. Zero-width or
    /// mixed-width rows at the layer refuse the snapshot (a 0-float
    /// row in a flattened matrix misaligns every later slice — review
    /// 2026-07-30 H3) and score via the per-entry slow path instead,
    /// where a zero-width row degrades to score 0.0.
    pub(crate) fn score_layer(&self, layer: usize, residual: &Array1<f32>) -> Vec<(usize, f32)> {
        if let Some(cache) = self.layer_cache(layer) {
            let mut out = Vec::with_capacity(cache.slot_ids.len());
            for (i, &slot) in cache.slot_ids.iter().enumerate() {
                let row = &cache.matrix[i * cache.d..(i + 1) * cache.d];
                let score: f32 = row.iter().zip(residual.iter()).map(|(a, b)| a * b).sum();
                out.push((slot, score));
            }
            out
        } else {
            self.rows
                .iter()
                .filter(|(&(l, _), _)| l == layer)
                .map(|(&(_, slot), row)| {
                    let score: f32 = row.iter().zip(residual.iter()).map(|(a, b)| a * b).sum();
                    (slot, score)
                })
                .collect()
        }
    }

    /// Build (or reuse) the layer snapshot. `None` when the layer has
    /// no rows or the rows are unsliceable (zero-width / mixed-width).
    fn layer_cache(&self, layer: usize) -> Option<Arc<LayerRowCache>> {
        if let Ok(c) = self.cache.read() {
            if let Some(cache) = c.get(&layer) {
                return Some(Arc::clone(cache));
            }
        }
        let mut slot_ids: Vec<usize> = Vec::new();
        let mut d = 0usize;
        for (&(l, s), row) in &self.rows {
            if l != layer {
                continue;
            }
            if row.is_empty() {
                return None; // zero-width row → slow path (H3 guard)
            }
            if d == 0 {
                d = row.len();
            } else if row.len() != d {
                return None; // mixed widths → slow path
            }
            slot_ids.push(s);
        }
        if slot_ids.is_empty() {
            return None;
        }
        let mut matrix = Vec::with_capacity(slot_ids.len() * d);
        for &slot in &slot_ids {
            matrix.extend_from_slice(&self.rows[&(layer, slot)]);
        }
        let cache = Arc::new(LayerRowCache {
            slot_ids,
            matrix,
            d,
        });
        if let Ok(mut c) = self.cache.write() {
            c.insert(layer, Arc::clone(&cache));
        }
        Some(cache)
    }

    fn invalidate_layer(&mut self, layer: usize) {
        if let Ok(mut c) = self.cache.write() {
            c.remove(&layer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(v: Vec<f32>) -> Array1<f32> {
        Array1::from_vec(v)
    }

    #[test]
    fn score_layer_uses_snapshot_and_survives_reuse() {
        let mut o = GateOverlay::default();
        o.insert(0, 3, vec![1.0, 0.0]);
        o.insert(0, 5, vec![0.0, 2.0]);
        for _ in 0..2 {
            let mut s = o.score_layer(0, &q(vec![1.0, 1.0]));
            s.sort_by_key(|&(slot, _)| slot);
            assert_eq!(s, vec![(3, 1.0), (5, 2.0)]);
        }
    }

    #[test]
    fn mutation_invalidates_only_the_touched_layer() {
        let mut o = GateOverlay::default();
        o.insert(0, 0, vec![1.0, 0.0]);
        o.insert(1, 0, vec![0.0, 1.0]);
        let _ = o.score_layer(0, &q(vec![1.0, 0.0])); // warm layer 0
        let _ = o.score_layer(1, &q(vec![1.0, 0.0])); // warm layer 1
        o.insert(1, 1, vec![1.0, 1.0]); // must invalidate layer 1 only
        assert_eq!(o.score_layer(0, &q(vec![1.0, 0.0])), vec![(0, 1.0)]);
        let mut s1 = o.score_layer(1, &q(vec![0.0, 1.0]));
        s1.sort_by_key(|&(slot, _)| slot);
        assert_eq!(s1, vec![(0, 1.0), (1, 1.0)]);
    }

    #[test]
    fn zero_width_row_forces_slow_path_with_correct_scores() {
        let mut o = GateOverlay::default();
        o.insert(0, 1, vec![1.0, 0.0]);
        o.insert(0, 2, vec![]); // legacy/corrupt zero-width row
        for _ in 0..2 {
            let mut s = o.score_layer(0, &q(vec![1.0, 0.0]));
            s.sort_by_key(|&(slot, _)| slot);
            assert_eq!(s, vec![(1, 1.0), (2, 0.0)]);
        }
    }

    #[test]
    fn mixed_width_rows_force_slow_path() {
        let mut o = GateOverlay::default();
        o.insert(0, 1, vec![1.0, 0.0]);
        o.insert(0, 2, vec![2.0]);
        let mut s = o.score_layer(0, &q(vec![1.0, 0.0]));
        s.sort_by_key(|&(slot, _)| slot);
        assert_eq!(s, vec![(1, 1.0), (2, 2.0)]);
    }

    #[test]
    fn remove_layer_and_replace_existing_contracts() {
        let mut o = GateOverlay::default();
        o.insert(0, 0, vec![1.0]);
        o.insert(1, 0, vec![2.0]);
        o.remove_layer(0);
        assert!(!o.has_layer(0));
        assert!(o.has_layer(1));
        // replace_existing only touches claimed slots.
        assert!(!o.replace_existing(0, 0, vec![9.0]));
        assert!(o.replace_existing(1, 0, vec![9.0]));
        assert_eq!(o.get(1, 0), Some(&[9.0][..]));
        assert_eq!(o.score_layer(1, &q(vec![1.0])), vec![(0, 9.0)]);
    }

    #[test]
    fn clone_resets_cache_but_keeps_rows() {
        let mut o = GateOverlay::default();
        o.insert(0, 0, vec![1.0, 2.0]);
        let _ = o.score_layer(0, &q(vec![1.0, 0.0])); // warm cache
        let c = o.clone();
        assert_eq!(c.get(0, 0), Some(&[1.0, 2.0][..]));
        assert_eq!(c.score_layer(0, &q(vec![0.0, 1.0])), vec![(0, 2.0)]);
        assert!(!c.is_empty());
    }

    #[test]
    fn remove_and_clear_empty_the_overlay() {
        let mut o = GateOverlay::default();
        o.insert(0, 0, vec![1.0]);
        let _ = o.score_layer(0, &q(vec![1.0]));
        assert_eq!(o.remove(0, 0), Some(vec![1.0]));
        assert_eq!(o.remove(0, 0), None);
        assert!(o.score_layer(0, &q(vec![1.0])).is_empty());
        o.insert(2, 7, vec![1.0]);
        o.clear();
        assert!(o.is_empty());
        assert!(!o.contains(2, 7));
        assert_eq!(o.iter().count(), 0);
    }
}
