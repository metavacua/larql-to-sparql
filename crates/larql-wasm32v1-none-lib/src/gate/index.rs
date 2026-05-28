//! In-memory gate index — flat f32 arrays per layer.

use alloc::vec::Vec;
use super::decode::{StorageDtype, decode_floats};

/// In-memory gate index. One flat f32 array per layer.
pub struct GateIndex {
    pub(crate) hidden_size: usize,
    /// `data[layer]` = flat `num_features × hidden_size` f32 row-major.
    pub(crate) data: Vec<Option<Vec<f32>>>,
    pub(crate) num_features: Vec<usize>,
}

impl GateIndex {
    pub fn new(num_layers: usize, hidden_size: usize) -> Self {
        Self {
            hidden_size,
            data: alloc::vec![None; num_layers],
            num_features: alloc::vec![0; num_layers],
        }
    }

    pub fn empty() -> Self {
        Self::new(0, 0)
    }

    pub fn load_layer(&mut self, layer: usize, raw: &[u8], dtype: StorageDtype) {
        let floats = decode_floats(raw, dtype);
        if self.hidden_size == 0 || floats.len() < self.hidden_size {
            return;
        }
        if layer >= self.data.len() {
            return;
        }
        let nf = floats.len() / self.hidden_size;
        self.num_features[layer] = nf;
        self.data[layer] = Some(floats);
    }

    pub fn layer_num_features(&self, layer: usize) -> usize {
        self.num_features.get(layer).copied().unwrap_or(0)
    }

    pub fn total_gate_vectors(&self) -> usize {
        self.num_features.iter().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_index_has_no_vectors() {
        let idx = GateIndex::empty();
        assert_eq!(idx.total_gate_vectors(), 0);
        assert_eq!(idx.layer_num_features(0), 0);
    }

    #[test]
    fn load_layer_stores_floats() {
        let mut idx = GateIndex::new(1, 2);
        let raw: Vec<u8> = [1.0f32, 0.0, 0.0, 1.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        idx.load_layer(0, &raw, StorageDtype::F32);
        assert_eq!(idx.layer_num_features(0), 2);
        assert_eq!(idx.total_gate_vectors(), 2);
    }
}
