//! Layer normalization primitives.
//!
//! Leaf math has no model-architecture or env-var coupling. The
//! `*_for_arch` convenience wrappers compose `arch.norm_eps()` with
//! `LARQL_NORM_EPS_OVERRIDE` (registered in
//! [`crate::forward_overrides`]) — Step 2e moved them down from
//! `larql-inference` once `forward_overrides` followed.

use larql_models::QkNormScope;
use ndarray::Array2;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Default norm epsilon. Most models use 1e-5 or 1e-6.
///
/// Callers with an architecture handle should prefer
/// `arch.norm_eps()`; this constant is for tests and for code paths
/// that genuinely have no model context.
pub const DEFAULT_EPS: f64 = 1e-6;

/// RMS norm with the legacy default epsilon ([`DEFAULT_EPS`] = 1e-6).
pub fn rms_norm(x: &Array2<f32>, weight: Option<&Vec<f32>>, offset: f32) -> Array2<f32> {
    rms_norm_eps(x, weight, offset, DEFAULT_EPS)
}

/// RMS norm with eps sourced from `arch.norm_eps()` (parsed from `config.json`)
/// or overridden by `LARQL_NORM_EPS_OVERRIDE`. The arch-driven path is the
/// permanent fix for bug 2 in
/// `docs/diagnoses/shannon-cross-engine-divergence.md`; the env var stays
/// as a diagnostic instrument.
pub fn rms_norm_for_arch(
    x: &Array2<f32>,
    weight: Option<&Vec<f32>>,
    offset: f32,
    arch: &dyn larql_models::ModelArchitecture,
) -> Array2<f32> {
    rms_norm_eps(x, weight, offset, effective_eps(arch))
}

/// LayerNorm with eps sourced from `arch.norm_eps()`, overridden by
/// `LARQL_NORM_EPS_OVERRIDE`. Companion to [`rms_norm_for_arch`].
pub fn layer_norm_for_arch(
    x: &Array2<f32>,
    weight: Option<&Vec<f32>>,
    bias: Option<&Vec<f32>>,
    arch: &dyn larql_models::ModelArchitecture,
) -> Array2<f32> {
    layer_norm_eps(x, weight, bias, effective_eps(arch))
}

fn effective_eps(arch: &dyn larql_models::ModelArchitecture) -> f64 {
    crate::forward_overrides::norm_eps_override()
        .map(|v| v as f64)
        .unwrap_or_else(|| arch.norm_eps() as f64)
}

/// RMS norm with explicit epsilon.
pub fn rms_norm_eps(
    x: &Array2<f32>,
    weight: Option<&Vec<f32>>,
    offset: f32,
    eps: f64,
) -> Array2<f32> {
    let (rows, cols) = (x.shape()[0], x.shape()[1]);
    let mut out = Array2::zeros((rows, cols));

    for i in 0..rows {
        let row = x.row(i);
        let sq_sum: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let rms = (sq_sum / cols as f64 + eps).sqrt() as f32;
        for j in 0..cols {
            let w = match weight {
                Some(wt) => offset + wt[j],
                None => 1.0,
            };
            out[[i, j]] = row[j] / rms * w;
        }
    }
    out
}

/// LayerNorm: (x - mean) / std * weight + bias.
/// Uses f64 accumulation for mean/variance.
pub fn layer_norm(
    x: &Array2<f32>,
    weight: Option<&Vec<f32>>,
    bias: Option<&Vec<f32>>,
) -> Array2<f32> {
    layer_norm_eps(x, weight, bias, DEFAULT_EPS)
}

/// LayerNorm with explicit epsilon.
pub fn layer_norm_eps(
    x: &Array2<f32>,
    weight: Option<&Vec<f32>>,
    bias: Option<&Vec<f32>>,
    eps: f64,
) -> Array2<f32> {
    let (rows, cols) = (x.shape()[0], x.shape()[1]);
    let mut out = Array2::zeros((rows, cols));

    for i in 0..rows {
        let row = x.row(i);
        let mean: f64 = row.iter().map(|&v| v as f64).sum::<f64>() / cols as f64;
        let var: f64 = row
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / cols as f64;
        let std = (var + eps).sqrt() as f32;
        let mean_f = mean as f32;
        for j in 0..cols {
            let normed = (row[j] - mean_f) / std;
            let w = weight.map_or(1.0, |wt| wt[j]);
            let b = bias.map_or(0.0, |bt| bt[j]);
            out[[i, j]] = normed * w + b;
        }
    }
    out
}

/// Per-head RMS norm without learned weights (parameter-free normalization).
/// Used for V-norm in Gemma 4: just normalizes, no scaling.
pub fn rms_norm_heads_no_weight(x: &Array2<f32>, num_heads: usize, head_dim: usize) -> Array2<f32> {
    rms_norm_heads_no_weight_eps(x, num_heads, head_dim, DEFAULT_EPS)
}

/// Per-head parameter-free RMS norm with explicit epsilon.
pub fn rms_norm_heads_no_weight_eps(
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    eps: f64,
) -> Array2<f32> {
    let seq_len = x.shape()[0];
    let mut out = x.clone();

    for s in 0..seq_len {
        for h in 0..num_heads {
            let off = h * head_dim;
            let mut sq_sum = 0.0f64;
            for d in 0..head_dim {
                let v = x[[s, off + d]] as f64;
                sq_sum += v * v;
            }
            let rms = (sq_sum / head_dim as f64 + eps).sqrt() as f32;
            for d in 0..head_dim {
                out[[s, off + d]] = x[[s, off + d]] / rms;
            }
        }
    }
    out
}

/// Per-head RMS norm for Q/K projections with configurable weight offset.
/// Uses f64 accumulation for the sum-of-squares.
pub fn rms_norm_heads(
    x: &Array2<f32>,
    weight: &[f32],
    num_heads: usize,
    head_dim: usize,
    offset: f32,
) -> Array2<f32> {
    rms_norm_heads_eps(x, weight, num_heads, head_dim, offset, DEFAULT_EPS)
}

/// Per-head RMS norm with explicit epsilon.
pub fn rms_norm_heads_eps(
    x: &Array2<f32>,
    weight: &[f32],
    num_heads: usize,
    head_dim: usize,
    offset: f32,
    eps: f64,
) -> Array2<f32> {
    let seq_len = x.shape()[0];
    let mut out = x.clone();

    for s in 0..seq_len {
        for h in 0..num_heads {
            let off = h * head_dim;
            let mut sq_sum = 0.0f64;
            for d in 0..head_dim {
                let v = x[[s, off + d]] as f64;
                sq_sum += v * v;
            }
            let rms = (sq_sum / head_dim as f64 + eps).sqrt() as f32;
            for d in 0..head_dim {
                out[[s, off + d]] = x[[s, off + d]] / rms * (offset + weight[d]);
            }
        }
    }
    out
}

/// Whole-projection RMS norm for Q/K, with explicit epsilon.
///
/// One statistic per row over all `num_heads * head_dim` elements, matching
/// `OlmoeRMSNorm(hidden_size)` applied before the reshape into heads. The
/// weight is still indexed across the full projection, so `weight` must be
/// `num_heads * head_dim` long — unlike the per-head form, where it is
/// `head_dim` long and reused by every head.
pub fn rms_norm_full_projection_eps(
    x: &Array2<f32>,
    weight: &[f32],
    num_heads: usize,
    head_dim: usize,
    offset: f32,
    eps: f64,
) -> Array2<f32> {
    let seq_len = x.shape()[0];
    let width = num_heads * head_dim;
    let mut out = x.clone();

    for s in 0..seq_len {
        let mut sq_sum = 0.0f64;
        for d in 0..width {
            let v = x[[s, d]] as f64;
            sq_sum += v * v;
        }
        let rms = (sq_sum / width as f64 + eps).sqrt() as f32;
        for d in 0..width {
            out[[s, d]] = x[[s, d]] / rms * (offset + weight[d]);
        }
    }
    out
}

/// QK norm under the architecture's declared scope — **the single entry point
/// every forward path must use.**
///
/// The two scopes apply the same weight and differ only in the denominator,
/// they share tensor names across architectures, and for MHA models they share
/// tensor shapes too. That leaves nothing at a call site to reveal a wrong
/// choice, which is why picking the scope is not a call-site decision: it
/// belongs to the architecture, and this function is where it is read.
///
/// `weight.len()` is cross-checked against the declared scope. The check is
/// cheap and it is the only signal that fires when an architecture declares
/// one convention while shipping weights for the other; it is skipped when
/// `num_heads == 1`, where the two conventions genuinely coincide.
pub fn rms_norm_qk_eps(
    x: &Array2<f32>,
    weight: &[f32],
    num_heads: usize,
    head_dim: usize,
    offset: f32,
    scope: QkNormScope,
    eps: f64,
) -> Array2<f32> {
    debug_assert!(
        num_heads == 1
            || weight.len()
                == match scope {
                    QkNormScope::PerHead => head_dim,
                    QkNormScope::FullProjection => num_heads * head_dim,
                },
        "QK-norm weight is {} long, which does not match the declared {:?} \
         scope for {num_heads} heads of {head_dim} — the architecture and the \
         checkpoint disagree about the reduction",
        weight.len(),
        scope,
    );
    match scope {
        QkNormScope::PerHead => rms_norm_heads_eps(x, weight, num_heads, head_dim, offset, eps),
        QkNormScope::FullProjection => {
            rms_norm_full_projection_eps(x, weight, num_heads, head_dim, offset, eps)
        }
    }
}

/// [`rms_norm_qk_eps`] with the legacy default epsilon ([`DEFAULT_EPS`]).
///
/// **Leaf form — tests and code with no architecture handle only.** Production
/// forwards want [`rms_norm_qk_for_arch`]: the scope *and* the epsilon are both
/// architecture facts, and this signature only asks for one of them.
pub fn rms_norm_qk(
    x: &Array2<f32>,
    weight: &[f32],
    num_heads: usize,
    head_dim: usize,
    offset: f32,
    scope: QkNormScope,
) -> Array2<f32> {
    rms_norm_qk_eps(x, weight, num_heads, head_dim, offset, scope, DEFAULT_EPS)
}

/// QK norm with **both** the scope and the epsilon sourced from the
/// architecture — the form every forward path should call.
///
/// The reference modules build their QK norm from the same config field as
/// every other norm in the model (`Qwen3RMSNorm(head_dim, eps=config.
/// rms_norm_eps)`, `OlmoeRMSNorm(hidden_size, eps=config.rms_norm_eps)`), so a
/// QK norm running at a *different* epsilon from the layer norms beside it is
/// wrong by construction. That is what the `DEFAULT_EPS` form did at every
/// call site: [`rms_norm_for_arch`] read the config while the QK norm three
/// lines above it did not, and on OLMoE and GPT-OSS those are an order of
/// magnitude apart. Companion to [`rms_norm_for_arch`], and it honours
/// `LARQL_NORM_EPS_OVERRIDE` the same way.
pub fn rms_norm_qk_for_arch(
    x: &Array2<f32>,
    weight: &[f32],
    num_heads: usize,
    head_dim: usize,
    offset: f32,
    arch: &dyn larql_models::ModelArchitecture,
) -> Array2<f32> {
    rms_norm_qk_eps(
        x,
        weight,
        num_heads,
        head_dim,
        offset,
        arch.qk_norm_scope(),
        effective_eps(arch),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── rms_norm ──────────────────────────────────────────────────────────────

    #[test]
    fn rms_norm_shape_preserved() {
        let x = Array2::from_shape_vec((3, 4), vec![1.0f32; 12]).unwrap();
        let out = rms_norm(&x, None, 0.0);
        assert_eq!(out.shape(), x.shape());
    }

    #[test]
    fn rms_norm_output_is_finite() {
        let x = Array2::from_shape_vec((2, 8), (0..16).map(|i| i as f32 * 0.1).collect()).unwrap();
        let out = rms_norm(&x, None, 0.0);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "rms_norm produced non-finite values"
        );
    }

    #[test]
    fn rms_norm_with_ones_weight_and_offset_one() {
        // weight=ones, offset=1.0 → Gemma-style: weight = 1.0 + learned (learned=0 here)
        let x = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let w = vec![0.0f32; 4]; // learned weight = zeros
        let out = rms_norm(&x, Some(&w), 1.0); // effective weight = 1.0 + 0.0 = 1.0
        let out_no_w = rms_norm(&x, None, 0.0);
        // Both paths should give the same result since effective weight=1 for both
        for (a, b) in out.iter().zip(out_no_w.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "offset=1 with zero weight should match no-weight norm"
            );
        }
    }

    #[test]
    fn rms_norm_zero_row_is_finite() {
        // Zero input → norm = 0 → eps prevents div-by-zero
        let x = Array2::zeros((1, 4));
        let out = rms_norm(&x, None, 0.0);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rms_norm_eps_changes_output_at_small_magnitudes() {
        // Pin the contract that callers depend on: different eps values
        // produce visibly different outputs when the squared mean is
        // small enough to be comparable to eps. This is the unit-level
        // gate that lets arch-aware callers trust they can swap eps
        // sources without silently no-op'ing.
        let x = Array2::from_shape_vec((1, 4), vec![0.001_f32, 0.001, 0.001, 0.001]).unwrap();
        let strict = rms_norm_eps(&x, None, 0.0, 1e-6);
        let loose = rms_norm_eps(&x, None, 0.0, 1e-5);
        let max_diff = strict
            .iter()
            .zip(loose.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 0.01,
            "rms_norm_eps did not honour explicit eps (max diff {max_diff})"
        );
    }

    // ── layer_norm ────────────────────────────────────────────────────────────

    #[test]
    fn layer_norm_shape_and_finite() {
        let x = Array2::from_shape_vec((2, 4), (0..8).map(|i| i as f32).collect()).unwrap();
        let w = vec![1.0f32; 4];
        let b = vec![0.0f32; 4];
        let out = layer_norm(&x, Some(&w), Some(&b));
        assert_eq!(out.shape(), x.shape());
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn layer_norm_zero_mean_unit_var() {
        let x = Array2::from_shape_vec((1, 8), (0..8).map(|i| i as f32).collect()).unwrap();
        let w = vec![1.0f32; 8];
        let b = vec![0.0f32; 8];
        let out = layer_norm(&x, Some(&w), Some(&b));
        let mean: f32 = out.row(0).iter().sum::<f32>() / 8.0;
        let var: f32 = out.row(0).iter().map(|v| (v - mean).powi(2)).sum::<f32>() / 8.0;
        assert!(mean.abs() < 1e-5, "mean should be ~0, got {mean}");
        assert!((var - 1.0).abs() < 0.1, "var should be ~1, got {var}");
    }

    #[test]
    fn layer_norm_eps_changes_output_on_nonuniform_input() {
        // Companion to `rms_norm_eps_changes_output_at_small_magnitudes`:
        // ensure explicit eps is honoured for LayerNorm too. Uniform
        // inputs collapse to 0 regardless of eps (mean-subtracted), so
        // use a non-uniform vector with small magnitudes.
        let x = Array2::from_shape_vec((1, 4), vec![0.001_f32, 0.002, 0.003, 0.004]).unwrap();
        let strict = layer_norm_eps(&x, None, None, 1e-6);
        let loose = layer_norm_eps(&x, None, None, 1e-5);
        let max_diff = strict
            .iter()
            .zip(loose.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 1e-5,
            "layer_norm_eps did not honour explicit eps (max diff {max_diff})"
        );
    }

    #[test]
    fn layer_norm_default_weight_and_bias_path() {
        // The `map_or(1.0, ...)` / `map_or(0.0, ...)` branches for the
        // None case are easy to regress when refactoring; pin them.
        let x = Array2::from_shape_vec((1, 4), vec![0.0_f32, 1.0, 2.0, 3.0]).unwrap();
        let out_none = layer_norm(&x, None, None);
        let w = vec![1.0_f32; 4];
        let b = vec![0.0_f32; 4];
        let out_explicit = layer_norm(&x, Some(&w), Some(&b));
        for (a, b) in out_none.iter().zip(out_explicit.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "None weight/bias should match explicit 1s/0s"
            );
        }
    }

    // ── rms_norm_heads ────────────────────────────────────────────────────────

    #[test]
    fn rms_norm_heads_no_weight_shape() {
        // [seq, num_heads * head_dim]
        let x = Array2::from_shape_vec((3, 8), (0..24).map(|i| i as f32 * 0.1).collect()).unwrap();
        let out = rms_norm_heads_no_weight(&x, 2, 4);
        assert_eq!(out.shape(), &[3, 8]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rms_norm_heads_normalises_each_head_independently() {
        // Two heads with very different magnitudes → both normalised
        let mut data = vec![0.0f32; 8];
        for (i, slot) in data.iter_mut().enumerate().take(4) {
            *slot = (i + 1) as f32;
        } // head 0: [1,2,3,4]
        for (i, slot) in data.iter_mut().enumerate().skip(4).take(4) {
            *slot = 100.0 * (i - 4 + 1) as f32;
        } // head 1: [100,200,300,400]
        let x = Array2::from_shape_vec((1, 8), data).unwrap();
        let out = rms_norm_heads_no_weight(&x, 2, 4);
        // Both heads should have similar L2 norm after per-head normalisation
        let h0_norm: f32 = out.row(0).iter().take(4).map(|v| v * v).sum::<f32>().sqrt();
        let h1_norm: f32 = out.row(0).iter().skip(4).map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (h0_norm - h1_norm).abs() < 0.1,
            "both heads should have similar L2 norm"
        );
    }

    #[test]
    fn rms_norm_heads_with_weight_scales() {
        let x = Array2::from_shape_vec((1, 4), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let w = vec![2.0f32, 2.0, 2.0, 2.0]; // scale by 2
        let out_scaled = rms_norm_heads(&x, &w, 1, 4, 0.0);
        let out_unscaled = rms_norm_heads_no_weight(&x, 1, 4);
        // Scaled output should be ~2× the unscaled
        for (s, u) in out_scaled.iter().zip(out_unscaled.iter()) {
            assert!(
                (s - 2.0 * u).abs() < 1e-5,
                "weight=2 should double the output"
            );
        }
    }

    #[test]
    fn rms_norm_heads_eps_changes_output_at_small_magnitudes() {
        // Pin explicit-eps contract for the per-head variant too.
        let x = Array2::from_shape_vec((1, 8), (0..8).map(|i| (i + 1) as f32 * 1e-4).collect())
            .unwrap();
        let strict = rms_norm_heads_no_weight_eps(&x, 2, 4, 1e-6);
        let loose = rms_norm_heads_no_weight_eps(&x, 2, 4, 1e-5);
        let max_diff = strict
            .iter()
            .zip(loose.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 0.01,
            "rms_norm_heads_no_weight_eps did not honour explicit eps (max diff {max_diff})"
        );
    }

    // ── QK-norm scope ─────────────────────────────────────────────────────────
    //
    // The two scopes coincide whenever every head carries the same RMS, so a
    // fixture built from uniform or symmetric values passes under either
    // reading and proves nothing. Every fixture below is deliberately
    // *unequal* across heads. This is the same trap `kda_a_log` ships a
    // rectangular fixture for: when two candidate readings agree on the easy
    // input, only the asymmetric one is evidence.

    /// `x = [1,2 | 3,4]`: head 0 has RMS √2.5, head 1 has RMS √12.5, and the
    /// whole projection has RMS √7.5. No two of those agree, so the scopes
    /// cannot produce the same answer.
    fn unequal_head_norms() -> Array2<f32> {
        Array2::from_shape_vec((1, 4), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap()
    }

    #[test]
    fn full_projection_matches_hand_computed_olmoe_reduction() {
        let x = unequal_head_norms();
        let w = vec![1.0f32; 4];
        let out = rms_norm_full_projection_eps(&x, &w, 2, 2, 0.0, 0.0);
        // mean(x²) = (1+4+9+16)/4 = 7.5 → rms = √7.5 = 2.7386128
        let rms = 7.5f32.sqrt();
        for (i, expected) in [1.0f32, 2.0, 3.0, 4.0].iter().enumerate() {
            assert!(
                (out[[0, i]] - expected / rms).abs() < 1e-6,
                "element {i}: {} != {}",
                out[[0, i]],
                expected / rms
            );
        }
    }

    #[test]
    fn per_head_and_full_projection_disagree_on_unequal_head_norms() {
        let x = unequal_head_norms();
        let per_head_w = vec![1.0f32; 2];
        let full_w = vec![1.0f32; 4];
        let per_head = rms_norm_qk(&x, &per_head_w, 2, 2, 0.0, QkNormScope::PerHead);
        let full = rms_norm_qk(&x, &full_w, 2, 2, 0.0, QkNormScope::FullProjection);
        let max_diff = per_head
            .iter()
            .zip(full.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff > 0.1,
            "the two QK-norm scopes must be distinguishable on unequal head \
             norms, else no test can catch choosing the wrong one (max diff \
             {max_diff})"
        );
    }

    #[test]
    fn scopes_coincide_when_every_head_has_equal_norm() {
        // Documents *why* the fixtures above are asymmetric: a per-head-equal
        // input is normalised identically under both readings, so it would
        // pass whichever scope the code picked. Same shape as the
        // `split_gate_up_experts` out_features=2 fixture that hid the
        // interleave bug (§4.7.6) — too small to distinguish the candidates.
        let x = Array2::from_shape_vec((1, 4), vec![1.0f32, 2.0, 2.0, 1.0]).unwrap();
        let per_head = rms_norm_qk(&x, &[1.0f32; 2], 2, 2, 0.0, QkNormScope::PerHead);
        let full = rms_norm_qk(&x, &[1.0f32; 4], 2, 2, 0.0, QkNormScope::FullProjection);
        let max_diff = per_head
            .iter()
            .zip(full.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-6,
            "equal head norms must coincide, got {max_diff}"
        );
    }

    #[test]
    fn full_projection_applies_weight_across_the_whole_projection() {
        // The per-head form reuses a head_dim-long weight for every head; the
        // full-projection form indexes a distinct scalar per output element.
        let x = Array2::from_shape_vec((1, 4), vec![1.0f32, 1.0, 1.0, 1.0]).unwrap();
        let w = vec![1.0f32, 2.0, 3.0, 4.0];
        let out = rms_norm_full_projection_eps(&x, &w, 2, 2, 0.0, 0.0);
        // Every x is 1 so rms = 1: the output is exactly the weight.
        for (i, expected) in w.iter().enumerate() {
            assert!(
                (out[[0, i]] - expected).abs() < 1e-6,
                "element {i} did not receive its own weight"
            );
        }
    }

    #[test]
    fn qk_scope_dispatch_is_row_independent() {
        // Two rows with different magnitudes: each must get its own statistic,
        // not a batch-wide one.
        let x = Array2::from_shape_vec((2, 4), vec![1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
            .unwrap();
        let out = rms_norm_qk(&x, &[1.0f32; 4], 2, 2, 0.0, QkNormScope::FullProjection);
        // Row 1 is exactly 10x row 0, so after per-row normalisation they match.
        for i in 0..4 {
            assert!(
                (out[[0, i]] - out[[1, i]]).abs() < 1e-5,
                "row {i} normalisation leaked across rows"
            );
        }
    }

    // ── QK-norm epsilon comes from the architecture ───────────────────────

    /// OLMoE ships no `rms_norm_eps`, so its family fallback (1e-5) is what
    /// runs — an order of magnitude off [`DEFAULT_EPS`]. The fixture is a row
    /// of *tiny* values so the epsilon dominates the mean square and the two
    /// choices are far apart; on an O(1) row they would agree to noise and the
    /// test would pass under either reading.
    #[test]
    fn qk_norm_for_arch_takes_its_epsilon_from_the_architecture() {
        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "olmoe", "hidden_size": 4, "num_hidden_layers": 1,
            "intermediate_size": 8, "num_attention_heads": 2, "num_key_value_heads": 2,
            "num_experts": 8, "num_experts_per_tok": 2, "norm_topk_prob": false,
        }));
        assert_eq!(arch.norm_eps(), 1e-5, "fixture premise: OLMoE falls back");

        let x = Array2::from_shape_vec((1, 4), vec![1e-3f32, 2e-3, 3e-3, 4e-3]).unwrap();
        let w = vec![1.0f32; 4];
        let from_arch = rms_norm_qk_for_arch(&x, &w, 2, 2, 0.0, &*arch);
        let hardcoded = rms_norm_qk(&x, &w, 2, 2, 0.0, QkNormScope::FullProjection);

        let max_diff = from_arch
            .iter()
            .zip(hardcoded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-3,
            "arch-driven and DEFAULT_EPS forms agreed (max diff {max_diff}) — the \
             epsilon is not being read from the architecture"
        );

        // And it is the *architecture's* value, not merely "some other" value.
        let expected = rms_norm_qk_eps(
            &x,
            &w,
            2,
            2,
            0.0,
            QkNormScope::FullProjection,
            arch.norm_eps() as f64,
        );
        assert_eq!(from_arch, expected);
    }

    /// The scope travels with the epsilon: an architecture declaring
    /// `FullProjection` must not be normalised per head just because the
    /// call site forgot to pass a scope.
    #[test]
    fn qk_norm_for_arch_also_carries_the_declared_scope() {
        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "olmoe", "hidden_size": 4, "num_hidden_layers": 1,
            "intermediate_size": 8, "num_attention_heads": 2, "num_key_value_heads": 2,
            "num_experts": 8, "num_experts_per_tok": 2, "norm_topk_prob": false,
        }));
        let x = Array2::from_shape_vec((1, 4), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let w = vec![1.0f32; 4];
        let expected = rms_norm_qk_eps(
            &x,
            &w,
            2,
            2,
            0.0,
            QkNormScope::FullProjection,
            arch.norm_eps() as f64,
        );
        assert_eq!(rms_norm_qk_for_arch(&x, &w, 2, 2, 0.0, &*arch), expected);
    }
}
