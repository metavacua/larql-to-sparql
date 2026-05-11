//! Q4_0 gate vectors (`gate_vectors_q4.bin`) — KNN-side quantised
//! gates consumed by `gate_knn_q4` / `gate_knn_adaptive`.
//!
//! Lives in the FFN-store directory because it shares the substore
//! footprint, even though the data targets gate-side KNN rather than
//! FFN forward — the Q4 file is a compressed companion to
//! `gate_vectors.bin`.

use std::sync::Arc;

use crate::error::VindexError;
use crate::format::filenames::GATE_VECTORS_Q4_BIN;
use crate::index::core::VectorIndex;
use crate::index::storage::vindex_storage::VindexStorage;
use crate::mmap_util::mmap_optimized;

impl VectorIndex {
    /// Load Q4_0 gate vectors from gate_vectors_q4.bin.
    ///
    /// File layout: layers packed contiguously, each layer is
    /// [num_features × hidden] in Q4_0 format (18 bytes per 32 elements).
    /// The per-layer feature count comes from gate_mmap_slices (must load
    /// f32/f16 gates first for the slice metadata, or pass feature counts).
    pub fn load_gate_vectors_q4(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join(GATE_VECTORS_Q4_BIN);
        if !path.exists() {
            return Err(VindexError::Parse("gate_vectors_q4.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };

        // Compute per-layer byte offsets from feature counts
        let mut slices = Vec::with_capacity(self.num_layers);
        let mut offset = 0usize;
        for layer in 0..self.num_layers {
            let num_features = self.num_features(layer);
            let floats = num_features * self.hidden_size;
            let q4_bytes = floats / larql_models::quant::ggml::Q4_0_BLOCK_ELEMS
                * larql_models::quant::ggml::Q4_0_BLOCK_BYTES;
            slices.push(crate::index::types::GateQ4Slice {
                byte_offset: offset,
                byte_len: q4_bytes,
                num_features,
            });
            offset += q4_bytes;
        }

        Arc::make_mut(&mut self.storage).set_gate_q4(Arc::new(mmap), slices);
        Ok(())
    }

    /// Whether Q4 gate vectors are loaded.
    pub fn has_gate_q4(&self) -> bool {
        self.storage.has_gate_q4()
    }

    /// Get Q4 data slice for a layer's gate vectors. Returns the raw Q4_0 bytes.
    ///
    /// Forwarded through [`VectorIndex::storage`] (step 4 of the
    /// `VindexStorage` migration).
    pub fn gate_q4_data(&self, layer: usize) -> Option<&[u8]> {
        Some(self.storage.gate_q4_layer_data(layer)?.as_slice())
    }
}
