//! Per-window boundary K,V checkpoint store (WARM tier).
//!
//! Each checkpoint is the K,V at the *last* position of a closed window, one
//! (K, V) pair per layer. K,V carry their baked-in RoPE offsets — so replay
//! from this checkpoint aligns positions correctly.
//!
//! Bytes per checkpoint (Gemma 3 4B, bf16):
//!   34 layers × 2 (K,V) × 4 kv_heads × 256 head_dim × 2 bytes ≈ 139 KB
//! (stored here as f32; multiply by 2 for the in-memory figure).

use std::collections::HashMap;

use larql_inference::attention::SharedKV;

#[derive(Default)]
pub struct CheckpointStore {
    kv: HashMap<usize, Vec<SharedKV>>,
    abs_pos: HashMap<usize, usize>,
}

impl CheckpointStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Save the last-position K,V for a closed window.
    /// `kv_last[layer]` has shape (1, num_kv * head_dim) for both K and V.
    pub fn save(&mut self, window_id: usize, kv_last: Vec<SharedKV>, abs_pos: usize) {
        debug_assert!(
            kv_last.iter().all(|(k, v)| k.shape()[0] == 1 && v.shape()[0] == 1),
            "checkpoint must be single-row K/V per layer"
        );
        self.kv.insert(window_id, kv_last);
        self.abs_pos.insert(window_id, abs_pos);
    }

    /// Return `(kv_last, abs_pos)` for a saved window.
    pub fn load(&self, window_id: usize) -> Option<(Vec<SharedKV>, usize)> {
        let kv = self.kv.get(&window_id)?.clone();
        let pos = *self.abs_pos.get(&window_id)?;
        Some((kv, pos))
    }

    pub fn contains(&self, window_id: usize) -> bool {
        self.kv.contains_key(&window_id)
    }

    pub fn len(&self) -> usize {
        self.kv.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kv.is_empty()
    }

    /// Discard checkpoints (e.g. after persisting to disk).
    pub fn evict(&mut self, window_ids: &[usize]) {
        for id in window_ids {
            self.kv.remove(id);
            self.abs_pos.remove(id);
        }
    }

    /// Total bytes held across all checkpoints (f32 accounting).
    pub fn total_bytes(&self) -> usize {
        self.kv
            .values()
            .flat_map(|layers| layers.iter())
            .map(|(k, v)| (k.len() + v.len()) * 4)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn mk_kv(layers: usize, kv_dim: usize) -> Vec<SharedKV> {
        (0..layers)
            .map(|l| {
                let mut k = Array2::<f32>::zeros((1, kv_dim));
                let mut v = Array2::<f32>::zeros((1, kv_dim));
                for j in 0..kv_dim {
                    k[[0, j]] = l as f32 + j as f32 * 0.01;
                    v[[0, j]] = l as f32 * 2.0 + j as f32 * 0.01;
                }
                (k, v)
            })
            .collect()
    }

    #[test]
    fn save_and_load_roundtrip() {
        let mut store = CheckpointStore::new();
        let kv = mk_kv(4, 8);
        store.save(0, kv, 511);
        assert!(store.contains(0));
        assert_eq!(store.len(), 1);

        let (loaded, pos) = store.load(0).expect("should load");
        assert_eq!(pos, 511);
        assert_eq!(loaded.len(), 4);
        assert_eq!(loaded[0].0.shape(), &[1, 8]);
    }

    #[test]
    fn evict_removes_window() {
        let mut store = CheckpointStore::new();
        store.save(0, mk_kv(2, 4), 0);
        store.save(1, mk_kv(2, 4), 511);
        assert_eq!(store.len(), 2);

        store.evict(&[0]);
        assert_eq!(store.len(), 1);
        assert!(!store.contains(0));
        assert!(store.contains(1));
    }

    #[test]
    fn total_bytes_scales_with_layers_and_dim() {
        let mut store = CheckpointStore::new();
        // 4 layers × (K + V, each 1×8 f32) = 4 × 2 × 8 × 4 = 256 bytes per window
        store.save(0, mk_kv(4, 8), 0);
        assert_eq!(store.total_bytes(), 4 * 2 * 8 * 4);
    }

    #[test]
    #[should_panic]
    fn save_rejects_multi_row_kv_in_debug() {
        let mut store = CheckpointStore::new();
        let multi_row: Vec<SharedKV> = (0..2)
            .map(|_| (Array2::<f32>::zeros((3, 8)), Array2::<f32>::zeros((3, 8))))
            .collect();
        store.save(0, multi_row, 0); // debug_assert fires
    }
}
