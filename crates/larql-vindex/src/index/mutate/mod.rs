//! VectorIndex mutation and persistence methods.
//!
//! Adds INSERT/DELETE/UPDATE support and the ability to save a
//! modified vindex back to disk. NDJSON heap loaders live in the
//! sibling `loaders` module.

pub mod loaders;

use std::io::{BufWriter, Write};
use std::path::Path;

use ndarray::Array1;

use crate::config::VindexConfig;
use crate::error::VindexError;
use crate::format::filenames::*;
use crate::index::storage::vindex_storage::VindexStorage;
use crate::index::{FeatureMeta, VectorIndex};

impl VectorIndex {
    /// Set metadata for a feature. Used by INSERT and UPDATE.
    pub fn set_feature_meta(&mut self, layer: usize, feature: usize, meta: FeatureMeta) {
        // Ensure layer slot exists
        while self.metadata.down_meta.len() <= layer {
            self.metadata.down_meta.push(None);
        }
        if self.metadata.down_meta[layer].is_none() {
            self.metadata.down_meta[layer] = Some(Vec::new());
        }
        if let Some(ref mut metas) = self.metadata.down_meta[layer] {
            while metas.len() <= feature {
                metas.push(None);
            }
            metas[feature] = Some(meta);
        }
    }

    /// Set the gate vector for a specific feature at a layer.
    /// The vector length must match hidden_size.
    /// If the index is in mmap mode, promotes this layer to heap first.
    pub fn set_gate_vector(&mut self, layer: usize, feature: usize, vector: &Array1<f32>) {
        // Promote from mmap to heap if needed
        if self.storage.has_gate_vectors()
            && self
                .gate
                .gate_vectors
                .get(layer)
                .map(|v| v.is_none())
                .unwrap_or(true)
        {
            self.promote_layer_to_heap(layer);
        }

        if let Some(Some(ref mut matrix)) = self.gate.gate_vectors.get_mut(layer) {
            if feature < matrix.shape()[0] && vector.len() == matrix.shape()[1] {
                for (j, val) in vector.iter().enumerate() {
                    matrix[[feature, j]] = *val;
                }
            }
        }
    }

    /// Set a custom down vector override for a feature.
    /// During sparse FFN, this vector is used instead of the model's down weight row.
    pub fn set_down_vector(&mut self, layer: usize, feature: usize, vector: Vec<f32>) {
        self.metadata
            .down_overrides
            .insert((layer, feature), vector);
    }

    /// All in-memory down vector overrides keyed by `(layer, feature)`.
    /// Used by `COMPILE INTO VINDEX` to bake the overrides into a fresh
    /// copy of `down_weights.bin`.
    ///
    /// For a single (layer, feature) lookup, use `down_override_at` —
    /// it has the same shape as `PatchedVindex::overrides_gate_at`.
    pub fn down_overrides(&self) -> &std::collections::HashMap<(usize, usize), Vec<f32>> {
        &self.metadata.down_overrides
    }

    /// Down vector override for `(layer, feature)`, if any has been set
    /// via `set_down_vector`. Returns the same data as the
    /// `GateIndex::down_override` trait method.
    pub fn down_override_at(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.metadata
            .down_overrides
            .get(&(layer, feature))
            .map(|v| v.as_slice())
    }

    /// Set a custom up vector override for a feature. Mirrors
    /// `set_down_vector`. INSERT writes here so the slot's activation
    /// `silu(gate · x) * (up · x)` reflects the constellation install
    /// instead of the original weak free-slot up vector.
    pub fn set_up_vector(&mut self, layer: usize, feature: usize, vector: Vec<f32>) {
        self.metadata.up_overrides.insert((layer, feature), vector);
    }

    /// All in-memory up vector overrides keyed by `(layer, feature)`.
    /// Parallel to `down_overrides()`. Used by `COMPILE INTO VINDEX` to
    /// bake the overrides into a fresh copy of `up_features.bin`.
    pub fn up_overrides(&self) -> &std::collections::HashMap<(usize, usize), Vec<f32>> {
        &self.metadata.up_overrides
    }

    /// Up vector override for `(layer, feature)`, if any has been set
    /// via `set_up_vector`. Same shape as `down_override_at`.
    pub fn up_override_at(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.metadata
            .up_overrides
            .get(&(layer, feature))
            .map(|v| v.as_slice())
    }

    /// Copy a layer's gate vectors from mmap to heap (for mutation).
    fn promote_layer_to_heap(&mut self, layer: usize) {
        if let Some(view) = self.storage.gate_layer_view(layer) {
            if view.slice.num_features > 0 {
                let bpf = crate::config::dtype::bytes_per_float(view.dtype);
                let byte_offset = view.slice.float_offset * bpf;
                let byte_count = view.slice.num_features * self.hidden_size * bpf;
                let byte_end = byte_offset + byte_count;
                let mmap: &[u8] = view.bytes.as_ref();
                if byte_end <= mmap.len() {
                    let raw = &mmap[byte_offset..byte_end];
                    let floats = crate::config::dtype::decode_floats(raw, view.dtype);
                    let matrix = ndarray::Array2::from_shape_vec(
                        (view.slice.num_features, self.hidden_size),
                        floats,
                    )
                    .unwrap();
                    while self.gate.gate_vectors.len() <= layer {
                        self.gate.gate_vectors.push(None);
                    }
                    self.gate.gate_vectors[layer] = Some(matrix);
                }
            }
        }
    }

    /// Clear metadata for a feature. Used by DELETE.
    pub fn delete_feature_meta(&mut self, layer: usize, feature: usize) {
        if let Some(Some(ref mut metas)) = self.metadata.down_meta.get_mut(layer) {
            if feature < metas.len() {
                metas[feature] = None;
            }
        }
    }

    /// Find a free (unused) feature slot at a layer — one with no metadata.
    /// If all slots have metadata, returns the weakest feature (lowest c_score).
    pub fn find_free_feature(&self, layer: usize) -> Option<usize> {
        // Mmap path: scan on demand
        if let Some(ref dm) = self.metadata.down_meta_mmap {
            let nf = dm.num_features(layer);
            if nf == 0 {
                return None;
            }
            // Look for empty slot
            for i in 0..nf {
                if dm.feature_meta(layer, i).is_none() {
                    return Some(i);
                }
            }
            // No empty — find weakest
            let mut weakest_idx = 0;
            let mut weakest_score = f32::MAX;
            for i in 0..nf {
                if let Some(meta) = dm.feature_meta(layer, i) {
                    if meta.c_score < weakest_score {
                        weakest_score = meta.c_score;
                        weakest_idx = i;
                    }
                }
            }
            return Some(weakest_idx);
        }

        // Heap path
        if let Some(Some(ref metas)) = self.metadata.down_meta.get(layer) {
            for (i, m) in metas.iter().enumerate() {
                if m.is_none() {
                    return Some(i);
                }
            }
            let mut weakest_idx = 0;
            let mut weakest_score = f32::MAX;
            for (i, m) in metas.iter().enumerate() {
                if let Some(meta) = m {
                    if meta.c_score < weakest_score {
                        weakest_score = meta.c_score;
                        weakest_idx = i;
                    }
                }
            }
            Some(weakest_idx)
        } else {
            None
        }
    }

    /// Find features matching entity and/or relation filters at a given layer (or all layers).
    /// Returns (layer, feature) pairs.
    ///
    /// Works in both heap and mmap mode — uses `feature_meta(layer, f)`
    /// (which checks heap first, then mmap) instead of reading the
    /// heap-side `down_meta` directly. This is important for the LQL
    /// executor's DELETE / UPDATE paths, which run against vindexes
    /// loaded fresh from disk (mmap mode).
    pub fn find_features(
        &self,
        entity: Option<&str>,
        relation_label: Option<&str>,
        layer_filter: Option<usize>,
    ) -> Vec<(usize, usize)> {
        let mut results = Vec::new();
        let layers = self.loaded_layers();
        // Relation matching is not yet implemented at this layer — reject
        // anything that asks for it.
        let relation_match = relation_label.is_none();

        for layer in layers {
            if let Some(l) = layer_filter {
                if layer != l {
                    continue;
                }
            }
            let n = self.num_features(layer);
            for feat in 0..n {
                let Some(meta) = self.feature_meta(layer, feat) else {
                    continue;
                };
                let entity_match = entity
                    .map(|e| {
                        meta.top_token.to_lowercase().contains(&e.to_lowercase())
                            || meta
                                .top_k
                                .iter()
                                .any(|t| t.token.to_lowercase().contains(&e.to_lowercase()))
                    })
                    .unwrap_or(true);
                if entity_match && relation_match {
                    results.push((layer, feat));
                }
            }
        }
        results
    }

    /// Write down_meta to disk as binary format (down_meta.bin).
    /// JSONL is no longer written — use `larql dump-meta` for human-readable output.
    /// Loading still falls back to JSONL for v1 compat if binary is absent.
    pub fn save_down_meta(&self, dir: &Path) -> Result<usize, VindexError> {
        let down_meta = self.materialize_down_meta();
        let max_top_k = down_meta
            .iter()
            .filter_map(|l| l.as_ref())
            .flat_map(|metas| metas.iter().filter_map(|m| m.as_ref()))
            .map(|m| m.top_k.len())
            .max()
            .unwrap_or(10);

        crate::format::down_meta::write_binary(dir, &down_meta, max_top_k)
    }

    fn materialize_down_meta(&self) -> Vec<Option<Vec<Option<FeatureMeta>>>> {
        let mut out = Vec::with_capacity(self.num_layers);
        for layer in 0..self.num_layers {
            let heap_len = self
                .metadata
                .down_meta
                .get(layer)
                .and_then(|m| m.as_ref())
                .map(|m| m.len())
                .unwrap_or(0);
            let mmap_len = self
                .metadata
                .down_meta_mmap
                .as_ref()
                .map(|dm| dm.num_features(layer))
                .unwrap_or(0);
            let gate_len = self.num_features(layer);
            let num_features = heap_len.max(mmap_len).max(gate_len);
            if num_features == 0 {
                out.push(None);
                continue;
            }

            let mut layer_meta = Vec::with_capacity(num_features);
            for feature in 0..num_features {
                layer_meta.push(self.feature_meta(layer, feature));
            }
            out.push(Some(layer_meta));
        }
        out
    }

    /// Write gate_vectors.bin back to disk and return updated layer info.
    /// Handles both heap and mmap modes.
    /// Writes to a temp file and renames to avoid invalidating active mmaps.
    pub fn save_gate_vectors(
        &self,
        dir: &Path,
    ) -> Result<Vec<crate::config::VindexLayerInfo>, VindexError> {
        self.save_gate_vectors_with_dtype(dir, crate::config::dtype::StorageDtype::F32)
    }

    /// Write gate vectors using the dtype and per-layer metadata from config.
    pub fn save_gate_vectors_with_config(
        &self,
        dir: &Path,
        config: &crate::config::VindexConfig,
    ) -> Result<Vec<crate::config::VindexLayerInfo>, VindexError> {
        let mut layer_infos = self.save_gate_vectors_with_dtype(dir, config.dtype)?;
        for info in &mut layer_infos {
            if let Some(existing) = config.layers.iter().find(|layer| layer.layer == info.layer) {
                info.num_experts = existing.num_experts;
                info.num_features_per_expert = existing.num_features_per_expert;
            }
        }
        Ok(layer_infos)
    }

    fn save_gate_vectors_with_dtype(
        &self,
        dir: &Path,
        dtype: crate::config::dtype::StorageDtype,
    ) -> Result<Vec<crate::config::VindexLayerInfo>, VindexError> {
        let path = dir.join(GATE_VECTORS_BIN);
        let tmp_path = dir.join("gate_vectors.bin.tmp");
        let file = std::fs::File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);
        let mut layer_infos = Vec::new();
        let mut offset: u64 = 0;

        for layer in 0..self.num_layers {
            // Try heap first (may have promoted layers), then mmap
            let data: Option<Vec<f32>> = if let Some(Some(ref matrix)) =
                self.gate.gate_vectors.get(layer)
            {
                Some(
                    matrix
                        .as_slice()
                        .ok_or_else(|| VindexError::Parse("gate vectors not contiguous".into()))?
                        .to_vec(),
                )
            } else if let Some(view) = self.storage.gate_layer_view(layer) {
                if view.slice.num_features > 0 {
                    let bpf = crate::config::dtype::bytes_per_float(view.dtype);
                    let byte_offset = view.slice.float_offset * bpf;
                    let byte_count = view.slice.num_features * self.hidden_size * bpf;
                    let byte_end = byte_offset + byte_count;
                    let mmap: &[u8] = view.bytes.as_ref();
                    if byte_end <= mmap.len() {
                        Some(crate::config::dtype::decode_floats(
                            &mmap[byte_offset..byte_end],
                            view.dtype,
                        ))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(ref data) = data {
                let num_features = data.len() / self.hidden_size;
                let length = crate::config::dtype::write_floats(&mut writer, data, dtype)?;

                layer_infos.push(crate::config::VindexLayerInfo {
                    layer,
                    num_features,
                    offset,
                    length,
                    num_experts: None,
                    num_features_per_expert: None,
                });
                offset += length;
            }
        }

        writer.flush()?;
        drop(writer); // close file before rename
        std::fs::rename(&tmp_path, &path)?;
        Ok(layer_infos)
    }

    /// Save config (index.json) to disk.
    pub fn save_config(config: &VindexConfig, dir: &Path) -> Result<(), VindexError> {
        let path = dir.join(INDEX_JSON);
        let json =
            serde_json::to_string_pretty(config).map_err(|e| VindexError::Parse(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Save the full vindex (gate_vectors.bin + down_meta.jsonl + index.json).
    /// Updates the config's layer info to match current state.
    pub fn save_vindex(&self, dir: &Path, config: &mut VindexConfig) -> Result<(), VindexError> {
        let layer_infos = self.save_gate_vectors_with_config(dir, config)?;
        config.layers = layer_infos;
        self.save_down_meta(dir)?;
        Self::save_config(config, dir)?;
        Ok(())
    }
}
