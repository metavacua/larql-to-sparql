use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use larql_hilbert::{classical_bits, entanglement_entropy, NQubit};
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
    /// Classical storage cost: Shannon entropy of the flattened |C|², in bits.
    pub classical_bits: f64,
    /// Compressibility gap `classical_bits − entropy` (≥ 0): how much more the
    /// classical description costs than the quantum entanglement across the cut.
    pub gap: f64,
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

/// Classical storage cost `H` of a coupling matrix `C` (Shannon entropy of the
/// flattened, normalized |C|², in bits) via the n-qubit reading. Pairs with the
/// existing `entanglement_entropy(C)` (the quantum ebits `S`); the
/// compressibility gap is `H − S ≥ 0`. Cheap — no eigensolver. Zero-safe (an
/// all-zero / pruned head returns 0) and zero-pads non-power-of-two dims.
fn classical_cost(coupling: &Array2<f64>) -> f64 {
    let fro2: f64 = coupling.iter().map(|&v| v * v).sum();
    if fro2 < 1e-300 {
        return 0.0; // degenerate head: no measurement entropy, no panic.
    }
    let (rows, cols) = (coupling.shape()[0], coupling.shape()[1]);
    let (pr, pc) = (rows.next_power_of_two(), cols.next_power_of_two());
    let mut flat = Vec::with_capacity(pr * pc);
    for r in 0..pr {
        for c in 0..pc {
            flat.push(if r < rows && c < cols { coupling[[r, c]] } else { 0.0 });
        }
    }
    classical_bits(&NQubit::from_real_amplitudes(&flat))
}

pub fn run(args: EntanglementArgs) -> Result<(), Box<dyn std::error::Error>> {
    let dir = &args.vindex;
    let t0 = Instant::now();
    println!("Entanglement / compressibility analysis for vindex at {}", dir.display());

    let config = load_vindex_config(dir)?;
    let mc = config
        .model_config
        .as_ref()
        .ok_or("entanglement: index.json has no model_config (need head_dim / head counts)")?;
    let (num_q, num_kv, head_dim) = (mc.num_q_heads, mc.num_kv_heads, mc.head_dim);
    println!(
        "  model: {} ({}), {} layers, q_heads={num_q}, kv_heads={num_kv}, head_dim={head_dim}",
        config.model, config.family, config.num_layers
    );

    let j = complex_structure_split_half(head_dim);

    print!("  loading attention Q/K ... ");
    let qk = load_attention_qk(dir, config.num_layers)?;
    println!("{} layers ({:.1}ms)", qk.len(), t0.elapsed().as_secs_f64() * 1000.0);

    let mut heads: Vec<HeadEntanglementInfo> = Vec::with_capacity(config.num_layers * num_q);
    for (layer, (wq, wk)) in qk.iter().enumerate() {
        if wq.nrows() < num_q * head_dim || wk.nrows() < num_kv * head_dim {
            return Err(format!(
                "layer {layer}: attention weights too small (wq {} rows, wk {} rows; need {} and {})",
                wq.nrows(),
                wk.nrows(),
                num_q * head_dim,
                num_kv * head_dim
            )
            .into());
        }
        let wq64 = wq.mapv(|v| v as f64);
        let wk64 = wk.mapv(|v| v as f64);
        for h in 0..num_q {
            let g = kv_head_for_query(h, num_q, num_kv);
            let wq_h = head_block(&wq64, h, head_dim);
            let wk_g = head_block(&wk64, g, head_dim);
            let coupling = head_coupling(&wq_h, &wk_g);
            let (residual, entropy) = coupling_metrics(&coupling, &j);
            let classical = classical_cost(&coupling);
            heads.push(HeadEntanglementInfo {
                layer,
                query_head: h,
                kv_head: g,
                residual,
                entropy,
                classical_bits: classical,
                gap: (classical - entropy).max(0.0),
            });
        }
    }

    let meta = EntanglementMeta {
        version: 2,
        model: config.model.clone(),
        head_dim,
        num_q_heads: num_q,
        num_kv_heads: num_kv,
        heads,
    };

    let out = dir.join("entanglement_meta.json");
    std::fs::write(&out, serde_json::to_string_pretty(&meta)?)?;

    let entropies: Vec<f64> = meta.heads.iter().map(|h| h.entropy).collect();
    let residuals: Vec<f64> = meta.heads.iter().map(|h| h.residual).collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let min = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = |v: &[f64]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "  entropy (ebits)  over {} heads: mean {:.3}, min {:.3}, max {:.3}  (max possible {:.1})",
        entropies.len(),
        mean(&entropies),
        min(&entropies),
        max(&entropies),
        (head_dim as f64).log2()
    );
    println!(
        "  residual         over {} heads: mean {:.3}, min {:.3}, max {:.3}",
        residuals.len(),
        mean(&residuals),
        min(&residuals),
        max(&residuals)
    );
    let gaps: Vec<f64> = meta.heads.iter().map(|h| h.gap).collect();
    println!(
        "  classical−quantum gap (bits) over {} heads: mean {:.3}, min {:.3}, max {:.3}",
        gaps.len(),
        mean(&gaps),
        min(&gaps),
        max(&gaps)
    );
    println!("  wrote {}", out.display());
    println!("  total: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    Ok(())
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

    #[test]
    fn classical_cost_pairs_with_matrix_entropy_for_a_nonnegative_gap() {
        use larql_hilbert::{entanglement_entropy_bipartition, row_qubits, NQubit};
        // A real-ish 4×4 coupling. Cross-check (in the test only): the n-qubit
        // bipartition equals entanglement_entropy(C). Then H ≥ S, so gap ≥ 0.
        let coupling = array![
            [1.0, 0.3, 0.0, 0.2],
            [0.3, 1.0, 0.1, 0.0],
            [0.0, 0.1, 1.0, 0.4],
            [0.2, 0.0, 0.4, 1.0],
        ];
        let quantum = entanglement_entropy(&coupling);
        let q = NQubit::from_matrix(&coupling);
        let bipart = entanglement_entropy_bipartition(&q, &row_qubits(4));
        assert!((quantum - bipart).abs() < 1e-9, "bipartition {bipart} vs matrix entropy {quantum}");
        let classical = classical_cost(&coupling);
        assert!(classical - quantum >= -1e-9, "gap must be ≥ 0: H={classical} S={quantum}");
    }

    #[test]
    fn product_coupling_has_a_positive_gap() {
        // Rank-1 (product) coupling: quantum S = 0, classical H > 0 → strictly positive gap.
        let coupling = array![
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ];
        let quantum = entanglement_entropy(&coupling);
        let classical = classical_cost(&coupling);
        assert!(quantum.abs() < 1e-9, "rank-1 → 0 ebits, got {quantum}");
        assert!(classical - quantum > 0.5, "uniform coupling has a large classical cost");
    }

    #[test]
    fn classical_cost_is_zero_safe_and_pads_non_power_of_two() {
        // Degenerate all-zero coupling → 0 (no panic from normalizing a zero state).
        let zero = Array2::<f64>::zeros((4, 4));
        assert_eq!(classical_cost(&zero), 0.0);
        // Non-power-of-two dims (e.g. a 3×3 head block) must not panic — zero-padded.
        let odd = array![[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [3.0, 0.0, 1.0]];
        let h = classical_cost(&odd);
        assert!(h > 0.0 && h.is_finite());
    }
}
