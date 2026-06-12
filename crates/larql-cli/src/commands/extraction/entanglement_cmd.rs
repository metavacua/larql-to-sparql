use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use larql_hilbert::entanglement_entropy;
use larql_vindex::canonical::{
    commutator_residual, complex_structure_split_half, head_block, head_coupling, kv_head_for_query,
};
use larql_vindex::format::{attn_load::load_attention_qk, load::load_vindex_config};

/// Per-head compressibility metrics on the QK coupling `C = W_Q W_Kᵀ`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadEntanglementInfo {
    pub layer: usize,
    pub query_head: usize,
    pub kv_head: usize,
    /// Hilbertian residual ‖[C,J]‖/‖C‖ ∈ [0,2] — complex-linearity.
    pub residual: f64,
    /// Entanglement entropy of C, in ebits ∈ [0, log2(head_dim)].
    pub entropy: f64,
}

/// Root metadata written to `entanglement_meta.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntanglementMeta {
    pub version: u32,
    pub model: String,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub heads: Vec<HeadEntanglementInfo>,
}

#[derive(Args)]
pub struct EntanglementArgs {
    /// Path to the .vindex directory to analyse.
    vindex: PathBuf,
}

/// The two compressibility metrics of a coupling matrix `C` against the
/// split-half complex structure `J`: (Hilbertian residual, entanglement entropy).
fn coupling_metrics(coupling: &Array2<f64>, j: &Array2<f64>) -> (f64, f64) {
    let residual = commutator_residual(coupling, j);
    let entropy = entanglement_entropy(coupling);
    (residual, entropy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn identity_coupling_is_complex_linear_and_flat() {
        // I₄ commutes with J → residual 0; flat spectrum → log2(4) = 2 ebits.
        let c = Array2::<f64>::eye(4);
        let j = complex_structure_split_half(4);
        let (r, s) = coupling_metrics(&c, &j);
        assert!(r.abs() < 1e-9, "identity should commute with J → residual 0, got {r}");
        assert!((s - 2.0).abs() < 1e-9, "I₄ flat spectrum → 2 ebits, got {s}");
    }

    #[test]
    fn rank_one_coupling_has_zero_entanglement() {
        // Rank-1 4×4 (a 2×2 rank-1 block, rest zero) → 0 ebits.
        let c = array![
            [1.0, 2.0, 0.0, 0.0],
            [2.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let j = complex_structure_split_half(4);
        let (_r, s) = coupling_metrics(&c, &j);
        assert!(s.abs() < 1e-9, "rank-1 coupling → 0 ebits, got {s}");
    }
}
