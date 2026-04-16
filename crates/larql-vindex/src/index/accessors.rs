//! `VectorIndex` metadata + gate-vector accessors and one-time setup.
//!
//! Pulls the read-only getters and `warmup` out of `gate.rs` so the
//! KNN-dispatch file stays focused on hot-path search code.
//!
//! - `feature_meta`, `down_meta_at`, `loaded_layers`, `num_features`,
//!   `num_features_at`, `total_gate_vectors`, `total_down_meta`:
//!   metadata readers (heap + mmap aware).
//! - `gate_vector`, `gate_vectors_at`, `gate_vectors_flat`:
//!   raw gate-matrix accessors (heap + mmap, single-row + bulk).
//! - `warmup`: pre-decode f16 mmap to f32 once so per-query KNN avoids
//!   re-decoding on every dispatch.

use ndarray::Array2;

use super::core::VectorIndex;
use super::types::*;

impl VectorIndex {
    /// Look up metadata for a specific feature.
    /// Checks heap first (mutation overrides), then mmap (production read path).
    pub fn feature_meta(&self, layer: usize, feature: usize) -> Option<FeatureMeta> {
        // Heap path first — catches mutation overrides (INSERT/UPDATE)
        if let Some(meta) = self
            .down_meta
            .get(layer)
            .and_then(|v| v.as_ref())
            .and_then(|metas| metas.get(feature))
            .and_then(|m| m.clone())
        {
            return Some(meta);
        }
        // Mmap path (production — zero heap, no mutations)
        if let Some(ref dm) = self.down_meta_mmap {
            return dm.feature_meta(layer, feature);
        }
        None
    }

    /// Number of features indexed at a layer.
    pub fn num_features(&self, layer: usize) -> usize {
        // Check mmap first
        if self.gate_mmap_bytes.is_some() {
            return self
                .gate_mmap_slices
                .get(layer)
                .map(|s| s.num_features)
                .unwrap_or(0);
        }
        self.gate_vectors
            .get(layer)
            .and_then(|v| v.as_ref())
            .map(|m| m.shape()[0])
            .unwrap_or(0)
    }

    /// Total gate vectors loaded across all layers.
    pub fn total_gate_vectors(&self) -> usize {
        if self.gate_mmap_bytes.is_some() {
            return self.gate_mmap_slices.iter().map(|s| s.num_features).sum();
        }
        self.gate_vectors
            .iter()
            .filter_map(|v| v.as_ref())
            .map(|m| m.shape()[0])
            .sum()
    }

    /// Total down metadata entries loaded across all layers.
    pub fn total_down_meta(&self) -> usize {
        if let Some(ref dm) = self.down_meta_mmap {
            return dm.total_features();
        }
        self.down_meta
            .iter()
            .filter_map(|v| v.as_ref())
            .map(|metas| metas.iter().filter(|m| m.is_some()).count())
            .sum()
    }

    /// Layers that have gate vectors loaded.
    pub fn loaded_layers(&self) -> Vec<usize> {
        if self.gate_mmap_bytes.is_some() {
            return self
                .gate_mmap_slices
                .iter()
                .enumerate()
                .filter(|(_, s)| s.num_features > 0)
                .map(|(i, _)| i)
                .collect();
        }
        self.gate_vectors
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.as_ref().map(|_| i))
            .collect()
    }

    /// Access down metadata for a specific layer.
    pub fn down_meta_at(&self, layer: usize) -> Option<&[Option<FeatureMeta>]> {
        self.down_meta
            .get(layer)
            .and_then(|v| v.as_ref())
            .map(|v| v.as_slice())
    }

    /// Access gate vectors matrix for a specific layer (heap mode only).
    /// Returns None in mmap mode — use gate_knn() directly instead.
    pub fn gate_vectors_at(&self, layer: usize) -> Option<&Array2<f32>> {
        self.gate_vectors.get(layer).and_then(|v| v.as_ref())
    }

    /// Extract a single gate vector for a feature. Works in both heap and mmap mode.
    /// Returns the raw f32 vector (hidden_size elements).
    pub fn gate_vector(&self, layer: usize, feature: usize) -> Option<Vec<f32>> {
        // Heap path
        if let Some(Some(matrix)) = self.gate_vectors.get(layer) {
            if feature < matrix.shape()[0] {
                return Some(matrix.row(feature).to_vec());
            }
            return None;
        }
        // Mmap path
        if let Some(ref mmap) = self.gate_mmap_bytes {
            if let Some(slice) = self.gate_mmap_slices.get(layer) {
                if feature >= slice.num_features {
                    return None;
                }
                let bpf = crate::config::dtype::bytes_per_float(self.gate_mmap_dtype);
                let byte_offset = (slice.float_offset + feature * self.hidden_size) * bpf;
                let byte_count = self.hidden_size * bpf;
                if byte_offset + byte_count > mmap.len() {
                    return None;
                }
                let raw = &mmap[byte_offset..byte_offset + byte_count];
                return Some(crate::config::dtype::decode_floats(raw, self.gate_mmap_dtype));
            }
        }
        None
    }

    /// Extract all gate vectors at a layer as flat f32 data.
    /// Returns (flat_data, num_features, hidden_size). Works in both heap and mmap mode.
    /// Use for bulk operations (SVD, PCA, numpy export).
    pub fn gate_vectors_flat(&self, layer: usize) -> Option<(Vec<f32>, usize, usize)> {
        // Heap path
        if let Some(Some(matrix)) = self.gate_vectors.get(layer) {
            let (rows, cols) = (matrix.shape()[0], matrix.shape()[1]);
            if let Some(data) = matrix.as_slice() {
                return Some((data.to_vec(), rows, cols));
            }
            // Non-contiguous — copy row by row
            let mut data = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                data.extend(matrix.row(r).iter());
            }
            return Some((data, rows, cols));
        }
        // Mmap path
        if let Some(ref mmap) = self.gate_mmap_bytes {
            if let Some(slice) = self.gate_mmap_slices.get(layer) {
                if slice.num_features == 0 {
                    return None;
                }
                let bpf = crate::config::dtype::bytes_per_float(self.gate_mmap_dtype);
                let byte_offset = slice.float_offset * bpf;
                let byte_count = slice.num_features * self.hidden_size * bpf;
                if byte_offset + byte_count > mmap.len() {
                    return None;
                }
                let raw = &mmap[byte_offset..byte_offset + byte_count];
                let data = crate::config::dtype::decode_floats(raw, self.gate_mmap_dtype);
                return Some((data, slice.num_features, self.hidden_size));
            }
        }
        None
    }

    /// Number of features at a layer (works in both heap and mmap mode).
    pub fn num_features_at(&self, layer: usize) -> usize {
        if self.gate_mmap_bytes.is_some() {
            self.gate_mmap_slices
                .get(layer)
                .map(|s| s.num_features)
                .unwrap_or(0)
        } else {
            self.num_features(layer)
        }
    }

    /// Pre-decode f16 gate vectors to f32 for lock-free access.
    /// For f32 vindexes this is a no-op — the mmap path is already zero-copy.
    pub fn warmup(&self) {
        if self.gate_mmap_dtype == crate::config::dtype::StorageDtype::F32 {
            return;
        }

        let Some(ref mmap) = self.gate_mmap_bytes else {
            return;
        };
        let mut warmed = self.warmed_gates.write().unwrap();
        if warmed.len() < self.num_layers {
            warmed.resize_with(self.num_layers, || None);
        }
        for layer in 0..self.num_layers {
            if warmed[layer].is_some() {
                continue;
            }
            if let Some(slice) = self.gate_mmap_slices.get(layer) {
                if slice.num_features == 0 {
                    continue;
                }
                let bpf = crate::config::dtype::bytes_per_float(self.gate_mmap_dtype);
                let byte_offset = slice.float_offset * bpf;
                let byte_count = slice.num_features * self.hidden_size * bpf;
                let byte_end = byte_offset + byte_count;
                if byte_end > mmap.len() {
                    continue;
                }
                let raw = &mmap[byte_offset..byte_end];
                warmed[layer] = Some(larql_models::quant::half::decode_f16(raw));
            }
        }
    }
}
