#![cfg_attr(not(test), no_std)]

// Tier 0: no filesystem, no network. Must compile for wasm32v1-none.
// alloc is always available (no_std + alloc).
extern crate alloc;

pub mod adj;
pub mod arch;
pub mod basis;
pub mod god_node;
pub mod norm;
pub mod trit;

pub use adj::{build_adjacency, build_node_index, NodeIndex, SparseAdj};
pub use arch::{size_architecture, ArchConfig};
pub use basis::{edges_to_weight_repr, BasisTransform, BitNetBasis, NamedTensor, WeightRepr};
pub use god_node::{degree_stats, god_nodes, DegreeStats};
pub use norm::{symmetric_normalise, NormAdj};
pub use trit::{pack_i2s_block, quantise_to_trits, unpack_i2s_block};

#[cfg(test)]
mod integration {
    use crate::{basis::BitNetBasis, edges_to_weight_repr};

    // Static node/fn names to avoid heap allocation in const context.
    const MODS: &[&str] = &[
        "mod_0", "mod_1", "mod_2", "mod_3", "mod_4", "mod_5", "mod_6", "mod_7", "mod_8", "mod_9",
    ];
    const FNS: &[&str] = &[
        "fn_0", "fn_1", "fn_2", "fn_3", "fn_4", "fn_5", "fn_6", "fn_7", "fn_8", "fn_9", "fn_10",
        "fn_11", "fn_12", "fn_13", "fn_14", "fn_15", "fn_16", "fn_17", "fn_18", "fn_19",
    ];

    #[test]
    fn end_to_end_edges_to_weight_repr() {
        let edges: Vec<(&str, &str, f64)> = (0..50usize)
            .map(|i| (MODS[i % 10], FNS[i % 20], 1.0))
            .collect();
        let repr = edges_to_weight_repr(&edges, &BitNetBasis);
        assert!(!repr.tensors.is_empty());
        // Every tensor must have I2_S data (32 bytes per 128-trit block)
        for t in &repr.tensors {
            assert!(t.data.len() % 32 == 0, "I2_S data must be 32-byte aligned");
        }
    }

    #[test]
    fn tensor_names_are_blk_prefixed() {
        let edges: Vec<(&str, &str, f64)> = (0..10usize)
            .map(|i| (MODS[i % 10], FNS[i % 10], 1.0))
            .collect();
        let repr = edges_to_weight_repr(&edges, &BitNetBasis);
        for t in &repr.tensors {
            assert!(
                t.name.starts_with("blk."),
                "tensor name must start with 'blk.', got: {}",
                t.name
            );
        }
    }
}
