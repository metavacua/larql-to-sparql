//! Layer normalization and residual stream operations.

use ndarray::Array2;

/// Default norm epsilon. Most models use 1e-5 or 1e-6.
/// Callers should prefer passing `arch.norm_eps()` explicitly.
pub const DEFAULT_EPS: f64 = 1e-6;

/// RMS norm with configurable weight offset and epsilon.
/// offset=1.0 for Gemma 2/3 (weight = 1 + learned), offset=0.0 for most layers.
/// Uses f64 accumulation for the sum-of-squares to avoid order-dependent rounding.
pub fn rms_norm(x: &Array2<f32>, weight: Option<&Vec<f32>>, offset: f32) -> Array2<f32> {
    rms_norm_eps(x, weight, offset, DEFAULT_EPS)
}

/// RMS norm with explicit epsilon.
pub fn rms_norm_eps(x: &Array2<f32>, weight: Option<&Vec<f32>>, offset: f32, eps: f64) -> Array2<f32> {
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
        let var: f64 = row.iter().map(|&v| {
            let d = v as f64 - mean;
            d * d
        }).sum::<f64>() / cols as f64;
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
pub fn rms_norm_heads_no_weight(
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
) -> Array2<f32> {
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
