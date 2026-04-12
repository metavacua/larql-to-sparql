//! MEMIT closed-form weight editing — compile vindex patches into W_down.
//!
//! Implements the core MEMIT algorithm from Meng et al. (2022–2023),
//! adapted for GatedFFN architectures (Gemma, Llama, etc.):
//!
//!   ΔW = (V* - W·K*) · (K*ᵀ C⁻¹ K* + λI)⁻¹ · K*ᵀ · C⁻¹
//!
//! where:
//!   K* = stacked FFN activation vectors at fact prompts [N × ffn_dim]
//!   V* = target output vectors [N × hidden_dim]
//!   C  = FFN activation covariance over random text
//!   W  = current W_down [hidden_dim × ffn_dim]
//!   λ  = ridge regularisation
//!
//! The solve pushes W_down updates into the null-space of typical
//! activations (high-variance directions in C get suppressed by C⁻¹),
//! so installed facts route through rarely-used directions — invisible
//! to normal text.
//!
//! Validated in Python: 200/200 (100%) at N=200 with multi-layer
//! distribution across L8-L12 on v11 TinyStories 115M. See
//! `experiments/15_v11_model/RESULTS.md §20`.

use ndarray::{Array1, Array2};
use crate::model::ModelWeights;
use super::trace::{capture_ffn_activation_matrix, estimate_ffn_covariance};

/// A single fact to be compiled via MEMIT.
#[derive(Debug, Clone)]
pub struct MemitFact {
    /// Canonical prompt token IDs (with BOS if the model uses it).
    pub prompt_tokens: Vec<u32>,
    /// Target token ID — the token MEMIT should make W_down produce.
    pub target_token_id: u32,
    /// Install layer.
    pub layer: usize,
    /// Human-readable label for diagnostics.
    pub label: String,
}

/// Result of a MEMIT solve at one layer.
#[derive(Debug)]
pub struct MemitResult {
    pub layer: usize,
    /// The weight delta: add this to W_down at the target layer.
    /// Shape: [hidden_dim, ffn_dim] (same as W_down).
    pub delta_w: Array2<f32>,
    /// Per-fact diagnostics.
    pub fact_results: Vec<MemitFactResult>,
}

/// Per-fact diagnostic from the MEMIT solve.
#[derive(Debug)]
pub struct MemitFactResult {
    pub label: String,
    pub k_star_norm: f32,
    pub target_norm: f32,
}

/// Covariance prompts — diverse short texts for estimating the FFN
/// activation covariance C = E[k(x) k(x)^T]. Sampling across varied
/// domains gives a well-conditioned C. Python reference used ~2000
/// prompts with ~14K total positions.
const COVARIANCE_PROMPTS: &[&str] = &[
    "Once upon a time, there was a",
    "The quick brown fox jumps over the",
    "In a distant land, far beyond the",
    "Scientists recently discovered that the",
    "The president announced that new",
    "Water boils at one hundred degrees",
    "The largest city in Europe is",
    "She walked through the old wooden door",
    "Mathematical proofs require careful",
    "The recipe calls for two cups of",
    "During the summer months, many people",
    "The history of ancient Rome begins",
    "A neural network consists of layers",
    "The stock market opened higher as",
    "Children learn best when they are",
    "The sun rises in the east and",
    "Programming languages differ in their",
    "The weather forecast predicts heavy",
    "Music has been a part of human",
    "The periodic table organizes chemical",
    "Birds migrate thousands of miles each",
    "The constitution guarantees certain",
    "Artificial intelligence continues to",
    "The ocean covers more than seventy",
    "A healthy diet includes plenty of",
    "The industrial revolution transformed",
    "Quantum mechanics describes the behavior",
    "The library contains thousands of",
    "Climate change affects ecosystems",
    "The painting was created during the",
];

/// Run the full MEMIT pipeline: estimate covariance, compute per-fact
/// activations and targets, solve the closed-form weight edit.
///
/// Returns one `MemitResult` per unique layer in the fact set.
/// The caller applies each `delta_w` to the corresponding layer's
/// W_down tensor.
pub fn run_memit(
    weights: &ModelWeights,
    facts: &[MemitFact],
    ridge: f64,
    target_alpha: f32,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<Vec<MemitResult>, String> {
    if facts.is_empty() {
        return Ok(Vec::new());
    }

    // Group facts by layer.
    let mut by_layer: std::collections::HashMap<usize, Vec<&MemitFact>> =
        std::collections::HashMap::new();
    for fact in facts {
        by_layer.entry(fact.layer).or_default().push(fact);
    }

    // Tokenise covariance prompts once.
    let cov_tokens: Vec<Vec<u32>> = COVARIANCE_PROMPTS
        .iter()
        .filter_map(|p| {
            tokenizer
                .encode(*p, true)
                .ok()
                .map(|e| e.get_ids().to_vec())
        })
        .collect();

    let mut results = Vec::new();

    for (layer, layer_facts) in &by_layer {
        let result = memit_solve_layer(
            weights,
            layer_facts,
            *layer,
            &cov_tokens,
            ridge,
            target_alpha,
        )?;
        results.push(result);
    }

    Ok(results)
}

/// MEMIT solve for a single layer — the core algorithm.
fn memit_solve_layer(
    weights: &ModelWeights,
    facts: &[&MemitFact],
    layer: usize,
    cov_tokens: &[Vec<u32>],
    ridge: f64,
    target_alpha: f32,
) -> Result<MemitResult, String> {
    let n = facts.len();
    let hidden = weights.hidden_size;
    let ffn_dim = weights.intermediate_size;

    // ── Step 1: Estimate covariance C at this layer ──
    let (cov_f32, sample_count) = estimate_ffn_covariance(weights, cov_tokens, layer)
        .ok_or_else(|| format!("MEMIT: failed to estimate covariance at layer {layer}"))?;

    if sample_count < 100 {
        return Err(format!(
            "MEMIT: only {sample_count} covariance samples at layer {layer} — need ≥100"
        ));
    }

    // ── Step 2: Compute K* — per-fact FFN activation at last position ──
    let mut k_stars: Vec<Array1<f64>> = Vec::with_capacity(n);
    let mut fact_results: Vec<MemitFactResult> = Vec::with_capacity(n);

    for fact in facts {
        let act_matrix = capture_ffn_activation_matrix(weights, &fact.prompt_tokens, layer)
            .ok_or_else(|| format!("MEMIT: activation capture failed for '{}'", fact.label))?;

        // Last token's activation row.
        let seq_len = act_matrix.shape()[0];
        let k_row = act_matrix.row(seq_len - 1);
        let k_f64: Array1<f64> = k_row.mapv(|v| v as f64);
        let k_norm = k_row.iter().map(|v| v * v).sum::<f32>().sqrt();

        k_stars.push(k_f64);
        fact_results.push(MemitFactResult {
            label: fact.label.clone(),
            k_star_norm: k_norm,
            target_norm: 0.0, // filled below
        });
    }

    // ── Step 3: Compute V* — target outputs ──
    //
    // v_star_i = W_down @ k_star_i + delta_i
    //
    // where delta_i = target_alpha * unit(embed[target_token]) — a
    // nudge in the direction of the target token's embedding. This
    // is the v1 approach matching the existing INSERT pipeline. The
    // Python reference uses 80-step SGD to find delta; this is the
    // closed-form approximation.
    // Verify W_down exists at this layer (the delta will be added to it).
    let w_down_key = weights.arch.ffn_down_key(layer);
    if !weights.tensors.contains_key(&w_down_key) {
        return Err(format!("MEMIT: W_down not found at layer {layer} (key: {w_down_key})"));
    }

    // ── Step 3+4: Compute R (deltas) and K matrices ──
    //
    // v_star = W @ k + delta, so R = V* - K W^T = delta (the embedding nudge).
    // We skip the explicit v_star computation and build R directly.
    //
    // MEMIT solve:  ΔW = R^T S⁻¹ Q   [hidden × ffn_dim]
    //   where Q = K C⁻¹, S = Q K^T + λI  (N×N, small)

    // Build K_star matrix [N × ffn_dim]
    let mut k_mat = Array2::<f64>::zeros((n, ffn_dim));
    for (i, k) in k_stars.iter().enumerate() {
        k_mat.row_mut(i).assign(k);
    }

    // Build R matrix [N × hidden] — the target embedding deltas.
    let mut r_mat = Array2::<f64>::zeros((n, hidden));
    for (i, fact) in facts.iter().enumerate() {
        let embed_row = weights.embed.row(fact.target_token_id as usize);
        let embed_norm: f32 = embed_row.iter().map(|v| v * v).sum::<f32>().sqrt();
        let scale = if embed_norm > 1e-8 { target_alpha / embed_norm } else { 0.0 };
        for j in 0..hidden {
            r_mat[[i, j]] = (embed_row[j] * scale) as f64;
        }
        fact_results[i].target_norm = embed_norm;
    }

    // C⁻¹ via Cholesky [ffn_dim × ffn_dim]
    let mut cov_f64 = Array2::<f64>::zeros((ffn_dim, ffn_dim));
    for i in 0..ffn_dim {
        for j in 0..ffn_dim {
            cov_f64[[i, j]] = cov_f32[[i, j]] as f64;
        }
    }

    let l = larql_compute::cpu::ops::linalg::cholesky(&cov_f64, ridge)
        .map_err(|e| format!("MEMIT: Cholesky failed — {e}"))?;

    // Q = K @ C⁻¹  [N × ffn_dim]
    // We compute this as: for each fact i, q_i = C⁻¹ @ k_i (column),
    // then Q[i, :] = q_i^T.
    // cholesky_solve(L, B) solves L L^T X = B, so X = C⁻¹ B.
    // We need C⁻¹ K^T [ffn_dim × N], then Q = (C⁻¹ K^T)^T = K C⁻¹.
    let k_t = k_mat.t().to_owned(); // [ffn_dim × N]
    let c_inv_kt = larql_compute::cpu::ops::linalg::cholesky_solve(&l, &k_t); // [ffn_dim × N]
    let q = c_inv_kt.t().to_owned(); // [N × ffn_dim] = K C⁻¹

    // S = Q K^T + λI  [N × N]
    let mut s = q.dot(&k_t); // [N × N]
    for i in 0..n {
        s[[i, i]] += ridge;
    }

    // S⁻¹ via Cholesky (S is N×N, small)
    let l_s = larql_compute::cpu::ops::linalg::cholesky(&s, 0.0)
        .map_err(|e| format!("MEMIT: S matrix Cholesky failed — {e}"))?;

    // ΔW = R^T @ S⁻¹ @ Q  [hidden × ffn_dim]
    //     = R^T @ (S⁻¹ Q)
    let s_inv_q = larql_compute::cpu::ops::linalg::cholesky_solve(&l_s, &q); // [N × ffn_dim]
    let r_t = r_mat.t().to_owned(); // [hidden × N]
    let delta_w_f64 = r_t.dot(&s_inv_q); // [hidden × ffn_dim]

    // Convert back to f32.
    let delta_w = delta_w_f64.mapv(|v| v as f32);

    Ok(MemitResult {
        layer,
        delta_w,
        fact_results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memit_fact_creation() {
        let fact = MemitFact {
            prompt_tokens: vec![1, 2, 3],
            target_token_id: 42,
            layer: 10,
            label: "test fact".into(),
        };
        assert_eq!(fact.layer, 10);
        assert_eq!(fact.target_token_id, 42);
    }
}
