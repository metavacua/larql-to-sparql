//! Shared sparse FFN computation — architecture-correct.
//!
//! Given a set of pre-selected feature indices, computes the FFN output using
//! the model architecture trait for activation function, gating, and bias.
//! Backends only need to provide the feature selection; this module handles
//! the gather, activation, and down projection.
//!
//! Supports gated (Gemma/Llama/Mistral) and non-gated (StarCoder2) models,
//! SiLU and GELU activations, optional up/down bias, and down vector overrides.

use ndarray::Array2;

use crate::forward::add_bias;
use crate::model::ModelWeights;
use super::{sigmoid, gelu_tanh};
use super::weight::dense_ffn_forward;

/// Compute FFN output for a pre-selected set of features.
///
/// Architecture-correct: reads ffn_type, activation, and bias from the model.
/// Falls back to dense (via `weight::dense_ffn_forward`) when K >= 80%.
///
/// `overrides`: optional down vector replacements for patched features.
/// When a feature has an override, its custom down vector is used instead of
/// the model's W_down column. This enables training-free INSERT.
pub fn sparse_ffn_forward(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
    features: &[usize],
) -> (Array2<f32>, Array2<f32>) {
    sparse_ffn_forward_impl(weights, layer, x, features, &[])
}

/// Sparse FFN with down vector overrides (for patched features).
pub fn sparse_ffn_forward_with_overrides(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
    features: &[usize],
    overrides: &[(usize, &[f32])],
) -> (Array2<f32>, Array2<f32>) {
    sparse_ffn_forward_impl(weights, layer, x, features, overrides)
}

fn sparse_ffn_forward_impl(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
    features: &[usize],
    overrides: &[(usize, &[f32])],
) -> (Array2<f32>, Array2<f32>) {
    let arch = &*weights.arch;
    let w_up = weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
    let w_down = weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();
    let hidden = x.shape()[1];
    let intermediate = w_up.shape()[0];
    let seq_len = x.shape()[0];
    let k = features.len();

    if k == 0 {
        return (
            Array2::<f32>::zeros((seq_len, hidden)),
            Array2::<f32>::zeros((seq_len, intermediate)),
        );
    }

    // Fall back to dense when most features are selected
    if k * 5 >= intermediate * 4 && overrides.is_empty() {
        return dense_ffn_forward(weights, layer, x);
    }

    let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
    let use_gelu = matches!(
        arch.activation(),
        larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
    );

    // Gather weight rows for selected features
    let up_buf = gather_rows(w_up, features, hidden);
    let up_sub = ndarray::ArrayView2::from_shape((k, hidden), &up_buf).unwrap();

    let _gate_buf;
    let gate_sub = if is_gated {
        let w_gate = weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
        _gate_buf = gather_rows(w_gate, features, hidden);
        Some(ndarray::ArrayView2::from_shape((k, hidden), &_gate_buf).unwrap())
    } else {
        _gate_buf = Vec::new();
        None
    };

    // Gather down-projection columns: w_down[:, features] → [hidden, K]
    let down_sub = gather_columns(w_down, features, hidden);
    let down_view = ndarray::ArrayView2::from_shape((hidden, k), &down_sub).unwrap();

    // Override lookup (only built when overrides are present)
    let override_map: std::collections::HashMap<usize, &[f32]> = if overrides.is_empty() {
        std::collections::HashMap::new()
    } else {
        overrides.iter().copied().collect()
    };

    let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));
    let mut sparse_act = vec![0.0f32; k];
    let mut out = Array2::<f32>::zeros((seq_len, hidden));

    for s in 0..seq_len {
        let x_row = x.row(s);

        // Compute sparse activations
        if let Some(ref gate_sub) = gate_sub {
            let gate_proj = gate_sub.dot(&x_row);
            let up_proj = up_sub.dot(&x_row);
            for (i, &feat) in features.iter().enumerate() {
                let g = gate_proj[i];
                let activated = if use_gelu { gelu_tanh(g) } else { g * sigmoid(g) };
                let val = activated * up_proj[i];
                sparse_act[i] = val;
                full_activation[[s, feat]] = val;
            }
        } else {
            let up_proj = up_sub.dot(&x_row);
            let mut vals = up_proj.to_vec();
            if let Some(bias) = arch.ffn_up_bias_key(layer).and_then(|bk| weights.vectors.get(&bk)) {
                for (i, &feat) in features.iter().enumerate() {
                    if feat < bias.len() { vals[i] += bias[feat]; }
                }
            }
            for (i, &feat) in features.iter().enumerate() {
                let v = vals[i];
                let val = if use_gelu { gelu_tanh(v) } else { v * sigmoid(v) };
                sparse_act[i] = val;
                full_activation[[s, feat]] = val;
            }
        }

        // Sparse down projection: w_down[:, features] @ sparse_act
        let act_view = ndarray::ArrayView1::from(&sparse_act[..k]);
        let out_vec = down_view.dot(&act_view);
        let mut out_row = out.row_mut(s);
        ndarray::Zip::from(&mut out_row).and(&out_vec).for_each(|o, &v| *o = v);

        // Apply overrides: swap standard down contribution with custom vector
        if !override_map.is_empty() {
            for (local_i, &feat) in features.iter().enumerate() {
                if let Some(override_down) = override_map.get(&feat) {
                    let activation = sparse_act[local_i];
                    if activation.abs() > 1e-8 && override_down.len() == hidden {
                        for j in 0..hidden {
                            out_row[j] -= down_view[[j, local_i]] * activation;
                            out_row[j] += override_down[j] * activation;
                        }
                    }
                }
            }
        }
    }

    if let Some(bias) = arch.ffn_down_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut out, bias);
    }

    (out, full_activation)
}

/// Gather rows from a weight matrix for selected features.
/// Input: w is [num_features, hidden], output: [K, hidden] contiguous.
fn gather_rows(
    w: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    features: &[usize],
    hidden: usize,
) -> Vec<f32> {
    let k = features.len();
    let raw = w.as_slice().unwrap();
    let mut buf = vec![0.0f32; k * hidden];
    for (i, &feat) in features.iter().enumerate() {
        let src = feat * hidden;
        buf[i * hidden..(i + 1) * hidden].copy_from_slice(&raw[src..src + hidden]);
    }
    buf
}

/// Gather columns from a weight matrix for selected features.
/// Input: w is [hidden, intermediate] (row-major), output: [hidden, K] contiguous.
fn gather_columns(
    w: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    features: &[usize],
    hidden: usize,
) -> Vec<f32> {
    let k = features.len();
    let cols = w.shape()[1];
    let raw = w.as_slice().unwrap();
    let mut buf = vec![0.0f32; hidden * k];
    for row in 0..hidden {
        let row_start = row * cols;
        for (j, &feat) in features.iter().enumerate() {
            buf[row * k + j] = raw[row_start + feat];
        }
    }
    buf
}

/// Select top-K features by gate activation magnitude (architecture-correct).
pub fn select_top_k_features(
    weights: &ModelWeights,
    layer: usize,
    x_row: &ndarray::ArrayView1<f32>,
    top_k: usize,
) -> Vec<usize> {
    let arch = &*weights.arch;
    let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
    let use_gelu = matches!(
        arch.activation(),
        larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
    );

    let proj = if is_gated {
        let w_gate = weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
        w_gate.dot(x_row)
    } else {
        let w_up = weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
        let mut p = w_up.dot(x_row);
        if let Some(bias) = arch.ffn_up_bias_key(layer).and_then(|bk| weights.vectors.get(&bk)) {
            for i in 0..p.len().min(bias.len()) {
                p[i] += bias[i];
            }
        }
        p
    };

    let intermediate = proj.len();
    let k = top_k.min(intermediate);

    let mut indexed: Vec<(usize, f32)> = proj
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| {
            let act = if use_gelu { gelu_tanh(v) } else { v * sigmoid(v) };
            (i, act)
        })
        .collect();

    if k > 0 && k < indexed.len() {
        indexed.select_nth_unstable_by(k, |a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        indexed.truncate(k);
    }
    indexed.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    indexed.into_iter().map(|(id, _)| id).collect()
}
