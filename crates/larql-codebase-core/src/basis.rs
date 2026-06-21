use alloc::{format, string::String, vec, vec::Vec};
use serde::{Deserialize, Serialize};

use crate::{
    arch::ArchConfig,
    norm::NormAdj,
    trit::{pack_i2s_block, quantise_to_trits},
};

/// GGML type identifier for I2_S (2-bit signed ternary).
const GGML_TYPE_I2_S: u32 = 36;

/// A single named weight tensor in GGUF-compatible format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub data: Vec<u8>,
}

/// Collection of tensors making up a model's initial weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightRepr {
    pub tensors: Vec<NamedTensor>,
    pub arch: ArchConfig,
}

/// Transform a normalised adjacency matrix into a set of model weight tensors.
pub trait BasisTransform {
    fn name(&self) -> &'static str;
    fn transform(&self, adj: &NormAdj, arch: &ArchConfig) -> WeightRepr;
}

/// Convenience end-to-end entrypoint: raw edges → WeightRepr.
/// This is the Tier-0 equivalent of Plan 3's `graph_to_weight_repr` (which lives in Tier 1).
pub fn edges_to_weight_repr(edges: &[(&str, &str, f64)], basis: &dyn BasisTransform) -> WeightRepr {
    use crate::{
        adj::{build_adjacency, build_node_index},
        arch::size_architecture,
        norm::symmetric_normalise,
    };
    let idx = build_node_index(edges);
    let adj = build_adjacency(edges, &idx);
    let norm = symmetric_normalise(&adj);
    let arch = size_architecture(idx.names.len(), edges.len());
    basis.transform(&norm, &arch)
}

/// BitNet I2_S basis: encodes each transformer layer as a hidden_size × hidden_size
/// trit matrix packed in Microsoft I2_S strided layout.
pub struct BitNetBasis;

impl BasisTransform for BitNetBasis {
    fn name(&self) -> &'static str {
        "bitnet_i2s"
    }

    fn transform(&self, adj: &NormAdj, arch: &ArchConfig) -> WeightRepr {
        let h = arch.hidden_size;
        let mut tensors = Vec::new();

        for layer in 0..arch.n_layers {
            // Fill a flat h*h buffer (row-major, zero-padded) with adjacency values.
            let mut dense = vec![0.0_f64; h * h];
            for &(i, j, w) in adj.entries.iter() {
                let row = i % h;
                let col = j % h;
                // Assign entries to layers based on which "band" they fall in.
                if (i / h) % arch.n_layers == layer {
                    dense[row * h + col] += w;
                }
            }

            // Find scale as max absolute value (avoid div by zero).
            let mut scale = 1e-6_f64;
            for &v in &dense {
                let av = if v < 0.0 { -v } else { v };
                if av > scale {
                    scale = av;
                }
            }

            // Quantise entire h*h matrix to trits.
            let trits = quantise_to_trits(&dense, scale);

            // Pack in 128-element I2_S blocks (zero-pad last block if needed).
            let total = trits.len();
            let n_blocks = total.div_ceil(128);
            let mut data = Vec::with_capacity(n_blocks * 32);
            for b in 0..n_blocks {
                let mut block = [0i8; 128];
                for (k, slot) in block.iter_mut().enumerate() {
                    let idx = b * 128 + k;
                    *slot = if idx < total { trits[idx] } else { 0 };
                }
                data.extend_from_slice(&pack_i2s_block(&block));
            }

            tensors.push(NamedTensor {
                name: format!("blk.{layer}.attn_qkv.weight"),
                dims: vec![h as u64, h as u64],
                ggml_type: GGML_TYPE_I2_S,
                data,
            });
        }

        WeightRepr {
            tensors,
            arch: arch.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        adj::{build_adjacency, build_node_index},
        arch::size_architecture,
        norm::symmetric_normalise,
    };

    fn small_edges() -> Vec<(&'static str, &'static str, f64)> {
        let nodes = [
            "node_0", "node_1", "node_2", "node_3", "node_4", "node_5", "node_6", "node_7",
            "node_8", "node_9",
        ];
        let mut v = Vec::new();
        for i in 0..10usize {
            v.push((nodes[i], nodes[(i + 1) % 10], 1.0));
        }
        v
    }

    #[test]
    fn bitnet_basis_produces_named_tensors() {
        let edges = small_edges();
        let idx = build_node_index(&edges);
        let adj = build_adjacency(&edges, &idx);
        let norm = symmetric_normalise(&adj);
        let arch = size_architecture(idx.names.len(), edges.len());
        let basis = BitNetBasis;
        let repr = basis.transform(&norm, &arch);
        assert!(!repr.tensors.is_empty());
        assert!(repr.tensors[0].name.starts_with("blk."));
    }

    #[test]
    fn tensor_ggml_type_is_i2s() {
        let edges = small_edges();
        let idx = build_node_index(&edges);
        let adj = build_adjacency(&edges, &idx);
        let norm = symmetric_normalise(&adj);
        let arch = size_architecture(idx.names.len(), edges.len());
        let repr = BitNetBasis.transform(&norm, &arch);
        for t in &repr.tensors {
            assert_eq!(t.ggml_type, 36u32, "I2_S = 36, got {}", t.ggml_type);
        }
    }

    #[test]
    fn tensor_data_is_32byte_aligned() {
        let edges = small_edges();
        let idx = build_node_index(&edges);
        let adj = build_adjacency(&edges, &idx);
        let norm = symmetric_normalise(&adj);
        let arch = size_architecture(idx.names.len(), edges.len());
        let repr = BitNetBasis.transform(&norm, &arch);
        for t in &repr.tensors {
            assert_eq!(
                t.data.len() % 32,
                0,
                "I2_S data must be 32-byte aligned, got {} bytes",
                t.data.len()
            );
        }
    }
}
