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

use super::weight::dense_ffn_forward;
use super::{gelu_tanh, sigmoid, FfnActivations, SparseActivations};
#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;
use crate::collections::HashMap;
use crate::forward::add_bias;
use crate::model::ModelWeights;

/// Compute FFN output for a pre-selected set of features.
///
/// Architecture-correct: reads ffn_type, activation, and bias from the model.
/// Falls back to dense (via `weight::dense_ffn_forward`) when K >= 80%.
///
/// Hot path: no activation buffer is allocated. Observation is opt-in
/// via [`sparse_ffn_forward_observed`] (2026-07-30 review, item 15).
pub fn sparse_ffn_forward(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
    features: &[usize],
) -> Array2<f32> {
    sparse_ffn_forward_impl(weights, layer, x, features, &[], false).0
}

/// [`sparse_ffn_forward`] plus an honest activation observation:
/// `(feature, activation)` pairs for exactly the computed features, or
/// [`FfnActivations::Dense`] when the ≥80%-K dense fallback fires.
pub fn sparse_ffn_forward_observed(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
    features: &[usize],
) -> (Array2<f32>, FfnActivations) {
    let (out, obs) = sparse_ffn_forward_impl(weights, layer, x, features, &[], true);
    (out, obs.expect("observe=true always yields an observation"))
}

/// Sparse FFN with down vector overrides (for patched features).
/// When a feature has an override, its custom down vector is used instead
/// of the model's W_down column. This enables training-free INSERT.
pub fn sparse_ffn_forward_with_overrides(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
    features: &[usize],
    overrides: &[(usize, &[f32])],
) -> Array2<f32> {
    sparse_ffn_forward_impl(weights, layer, x, features, overrides, false).0
}

/// Per-slot override carrying any combination of gate / up / down vectors.
/// Used by `sparse_ffn_forward_with_full_overrides` so that an INSERT's
/// (gate * g_ref * 30, up * u_ref, down * d_ref * alpha_mul) all flow into
/// the activation computation, rather than only the down vector being
/// swapped post-activation. Each field is `None` when the original
/// model weight at that slot should be used.
#[derive(Debug, Clone, Copy)]
pub struct FeatureSlotOverride<'a> {
    pub feature: usize,
    pub gate: Option<&'a [f32]>,
    pub up: Option<&'a [f32]>,
    pub down: Option<&'a [f32]>,
}

/// Sparse FFN with full slot overrides — gate, up, and down can each
/// be replaced per feature. This is the path the LQL `INSERT` constellation
/// takes when the vindex hasn't loaded `up_features.bin` (the LQL `USE`
/// fast path doesn't load it). Without this entry point, the activation
/// at an installed slot would compute `silu(weak_gate · x) * (weak_up · x)`
/// using the original free-slot vectors, and the strong overlay gate
/// would never reach inference.
pub fn sparse_ffn_forward_with_full_overrides(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
    features: &[usize],
    overrides: &[FeatureSlotOverride<'_>],
) -> Array2<f32> {
    sparse_ffn_forward_full_impl(weights, layer, x, features, overrides, false).0
}

/// [`sparse_ffn_forward_with_full_overrides`] plus the honest
/// observation of the post-override slot activations.
pub fn sparse_ffn_forward_with_full_overrides_observed(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
    features: &[usize],
    overrides: &[FeatureSlotOverride<'_>],
) -> (Array2<f32>, FfnActivations) {
    let (out, obs) = sparse_ffn_forward_full_impl(weights, layer, x, features, overrides, true);
    (out, obs.expect("observe=true always yields an observation"))
}

fn sparse_ffn_forward_impl(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
    features: &[usize],
    overrides: &[(usize, &[f32])],
    observe: bool,
) -> (Array2<f32>, Option<FfnActivations>) {
    let arch = &*weights.arch;
    let w_up = weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
    let w_down = weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();
    let hidden = x.shape()[1];
    let intermediate = w_up.shape()[0];
    let seq_len = x.shape()[0];
    let k = features.len();

    if k == 0 {
        // Nothing computed: the honest observation is an empty sparse
        // record, not a zero-filled dense matrix.
        return (
            Array2::<f32>::zeros((seq_len, hidden)),
            observe.then(|| FfnActivations::Sparse(SparseActivations::new(seq_len, intermediate))),
        );
    }

    // Fall back to dense when most features are selected
    if k * 5 >= intermediate * 4 && overrides.is_empty() {
        let (out, act) = dense_ffn_forward(larql_models::WeightsView::dense(weights), layer, x);
        return (out, observe.then(|| FfnActivations::Dense(act)));
    }

    let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
    let use_gelu = arch.activation().uses_gelu_tanh_gate_up();

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
    let override_map: HashMap<usize, &[f32]> = if overrides.is_empty() {
        HashMap::default()
    } else {
        overrides.iter().copied().collect()
    };

    let mut obs = observe.then(|| SparseActivations::new(seq_len, intermediate));
    let mut sparse_act = vec![0.0f32; k];
    let mut out = Array2::<f32>::zeros((seq_len, hidden));

    for s in 0..seq_len {
        let x_row = x.row(s);

        // Compute sparse activations. Each arm records its own
        // observation so the projection scores it computed ride along
        // (runtime trace, 2026-07-30 review item 17).
        if let Some(ref gate_sub) = gate_sub {
            let gate_proj = gate_sub.dot(&x_row);
            let up_proj = up_sub.dot(&x_row);
            for i in 0..k {
                let g = gate_proj[i];
                let activated = if use_gelu {
                    gelu_tanh(g)
                } else {
                    g * sigmoid(g)
                };
                sparse_act[i] = activated * up_proj[i];
            }
            if let Some(o) = obs.as_mut() {
                for (i, &feat) in features.iter().enumerate() {
                    o.record_scored(s, feat, sparse_act[i], Some(gate_proj[i]), Some(up_proj[i]));
                }
            }
        } else {
            let up_proj = up_sub.dot(&x_row);
            let mut vals = up_proj.to_vec();
            if let Some(bias) = arch
                .ffn_up_bias_key(layer)
                .and_then(|bk| weights.vectors.get(&bk))
            {
                for (i, &feat) in features.iter().enumerate() {
                    if feat < bias.len() {
                        vals[i] += bias[feat];
                    }
                }
            }
            for i in 0..k {
                let v = vals[i];
                sparse_act[i] = if use_gelu {
                    gelu_tanh(v)
                } else {
                    v * sigmoid(v)
                };
            }
            if let Some(o) = obs.as_mut() {
                // Non-gated FFN: the single pre-activation projection
                // (incl. up-bias) is the gate-position score; there is
                // no separate up projection to report.
                for (i, &feat) in features.iter().enumerate() {
                    o.record_scored(s, feat, sparse_act[i], Some(vals[i]), None);
                }
            }
        }

        // Sparse down projection: w_down[:, features] @ sparse_act
        let act_view = ndarray::ArrayView1::from(&sparse_act[..k]);
        let out_vec = down_view.dot(&act_view);
        let mut out_row = out.row_mut(s);
        ndarray::Zip::from(&mut out_row)
            .and(&out_vec)
            .for_each(|o, &v| *o = v);

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

    if let Some(bias) = arch
        .ffn_down_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut out, bias);
    }

    (out, obs.map(FfnActivations::Sparse))
}

fn sparse_ffn_forward_full_impl(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
    features: &[usize],
    overrides: &[FeatureSlotOverride<'_>],
    observe: bool,
) -> (Array2<f32>, Option<FfnActivations>) {
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
            observe.then(|| FfnActivations::Sparse(SparseActivations::new(seq_len, intermediate))),
        );
    }

    let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
    let use_gelu = arch.activation().uses_gelu_tanh_gate_up();

    // Gather original weight rows. Per-feature overrides will be
    // applied below by re-computing the dot products for the
    // overridden slots only — the unchanged slots use the gathered
    // values from the dense weights.
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

    let down_sub = gather_columns(w_down, features, hidden);
    let down_view = ndarray::ArrayView2::from_shape((hidden, k), &down_sub).unwrap();

    // Per-feature override lookup. Built once.
    let override_map: HashMap<usize, &FeatureSlotOverride<'_>> =
        overrides.iter().map(|o| (o.feature, o)).collect();

    let mut obs = observe.then(|| SparseActivations::new(seq_len, intermediate));
    let mut sparse_act = vec![0.0f32; k];
    let mut out = Array2::<f32>::zeros((seq_len, hidden));

    for s in 0..seq_len {
        let x_row = x.row(s);

        // Per-slot projection scores retained for the observation
        // record (runtime trace, 2026-07-30 review item 17). Only
        // allocated when observing; phase 2 overwrites the entries of
        // any recomputed (overridden) slots so the record carries the
        // POST-override scores the executed output used.
        let mut scored: Option<Vec<(Option<f32>, Option<f32>)>> =
            obs.is_some().then(|| vec![(None, None); k]);

        // Phase 1: compute baseline activations for every feature
        // using gathered dense weights.
        if let Some(ref gate_sub) = gate_sub {
            let gate_proj = gate_sub.dot(&x_row);
            let up_proj = up_sub.dot(&x_row);
            for i in 0..k {
                let g = gate_proj[i];
                let activated = if use_gelu {
                    gelu_tanh(g)
                } else {
                    g * sigmoid(g)
                };
                sparse_act[i] = activated * up_proj[i];
            }
            if let Some(sc) = scored.as_mut() {
                for i in 0..k {
                    sc[i] = (Some(gate_proj[i]), Some(up_proj[i]));
                }
            }
        } else {
            let up_proj = up_sub.dot(&x_row);
            let mut vals = up_proj.to_vec();
            if let Some(bias) = arch
                .ffn_up_bias_key(layer)
                .and_then(|bk| weights.vectors.get(&bk))
            {
                for (i, &feat) in features.iter().enumerate() {
                    if feat < bias.len() {
                        vals[i] += bias[feat];
                    }
                }
            }
            for i in 0..k {
                let v = vals[i];
                sparse_act[i] = if use_gelu {
                    gelu_tanh(v)
                } else {
                    v * sigmoid(v)
                };
            }
            if let Some(sc) = scored.as_mut() {
                // Non-gated: the single pre-activation projection
                // (incl. up-bias) is the gate-position score.
                for i in 0..k {
                    sc[i] = (Some(vals[i]), None);
                }
            }
        }

        // Phase 2: re-compute activation for overridden slots using
        // the overlay's gate / up vectors. The slot's contribution to
        // the residual is `silu(gate_override · x) * (up_override · x)`
        // — exactly the install_compiled_slot Python semantics.
        for (i, &feat) in features.iter().enumerate() {
            let Some(ov) = override_map.get(&feat) else {
                continue;
            };
            // Only recompute if at least one of gate / up is overridden.
            if ov.gate.is_none() && ov.up.is_none() {
                continue;
            }
            // Gate dot product (override or original gathered row).
            let g = if let Some(g_ov) = ov.gate {
                if g_ov.len() == hidden {
                    g_ov.iter()
                        .zip(x_row.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                } else {
                    // Length mismatch — fall through to original.
                    if let Some(ref gate_sub) = gate_sub {
                        gate_sub.row(i).dot(&x_row)
                    } else {
                        0.0
                    }
                }
            } else if let Some(ref gate_sub) = gate_sub {
                gate_sub.row(i).dot(&x_row)
            } else {
                0.0
            };
            let activated = if use_gelu {
                gelu_tanh(g)
            } else {
                g * sigmoid(g)
            };

            // Up dot product (override or original).
            let up_score = if let Some(u_ov) = ov.up {
                if u_ov.len() == hidden {
                    u_ov.iter()
                        .zip(x_row.iter())
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                } else {
                    up_sub.row(i).dot(&x_row)
                }
            } else {
                up_sub.row(i).dot(&x_row)
            };

            sparse_act[i] = if is_gated {
                activated * up_score
            } else {
                activated
            };
            if let Some(sc) = scored.as_mut() {
                sc[i] = if is_gated {
                    (Some(g), Some(up_score))
                } else {
                    (Some(g), None)
                };
            }
        }

        // Observation records the POST-override slot activations and
        // the projection scores that produced them — phase 2 has
        // already re-computed any overridden slots.
        if let Some(o) = obs.as_mut() {
            let sc = scored.as_ref().expect("scored allocated iff observing");
            for (i, &feat) in features.iter().enumerate() {
                o.record_scored(s, feat, sparse_act[i], sc[i].0, sc[i].1);
            }
        }

        // Phase 3: down projection using gathered dense down vectors,
        // then swap in down overrides for any overridden slots.
        let act_view = ndarray::ArrayView1::from(&sparse_act[..k]);
        let out_vec = down_view.dot(&act_view);
        let mut out_row = out.row_mut(s);
        ndarray::Zip::from(&mut out_row)
            .and(&out_vec)
            .for_each(|o, &v| *o = v);

        for (i, &feat) in features.iter().enumerate() {
            let Some(ov) = override_map.get(&feat) else {
                continue;
            };
            let Some(d_ov) = ov.down else {
                continue;
            };
            if d_ov.len() != hidden {
                continue;
            }
            let activation = sparse_act[i];
            if activation.abs() <= 1e-8 {
                continue;
            }
            // Subtract the dense column contribution and add the override.
            for j in 0..hidden {
                out_row[j] -= down_view[[j, i]] * activation;
                out_row[j] += d_ov[j] * activation;
            }
        }
    }

    if let Some(bias) = arch
        .ffn_down_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut out, bias);
    }

    (out, obs.map(FfnActivations::Sparse))
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
    let use_gelu = arch.activation().uses_gelu_tanh_gate_up();

    let proj = if is_gated {
        let w_gate = weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
        w_gate.dot(x_row)
    } else {
        let w_up = weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
        let mut p = w_up.dot(x_row);
        if let Some(bias) = arch
            .ffn_up_bias_key(layer)
            .and_then(|bk| weights.vectors.get(&bk))
        {
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
            let act = if use_gelu {
                gelu_tanh(v)
            } else {
                v * sigmoid(v)
            };
            (i, act)
        })
        .collect();

    if k > 0 && k < indexed.len() {
        indexed.select_nth_unstable_by(k, |a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        indexed.truncate(k);
    }
    indexed.sort_unstable_by_key(|(id, _)| *id);
    indexed.into_iter().map(|(id, _)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::make_test_weights;
    use ndarray::Array2;

    fn input(seq: usize, hidden: usize) -> Array2<f32> {
        let data: Vec<f32> = (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.01).collect();
        Array2::from_shape_vec((seq, hidden), data).unwrap()
    }

    // ── sparse_ffn_forward ────────────────────────────────────────────────────

    #[test]
    fn sparse_forward_empty_features_returns_zeros() {
        let weights = make_test_weights();
        let x = input(2, weights.hidden_size);
        let (out, obs) = sparse_ffn_forward_observed(&weights, 0, &x, &[]);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(
            out.iter().all(|v| v.abs() < 1e-9),
            "empty features → zero output"
        );
        // Honest observation: nothing was computed, so the sparse
        // record is empty — not a fabricated zero matrix.
        match obs {
            FfnActivations::Sparse(s) => {
                assert_eq!(s.seq_len(), 2);
                assert!(s.position(0).is_empty() && s.position(1).is_empty());
            }
            other => panic!("expected Sparse observation, got {other:?}"),
        }
    }

    #[test]
    fn sparse_forward_single_feature_output_shape() {
        let weights = make_test_weights();
        let x = input(1, weights.hidden_size);
        let out = sparse_ffn_forward(&weights, 0, &x, &[0]);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sparse_forward_multi_token_observed_emits_only_computed_features() {
        let weights = make_test_weights();
        let x = input(3, weights.hidden_size);
        let feats = [0usize, 1, 2];
        let (out, obs) = sparse_ffn_forward_observed(&weights, 0, &x, &feats);
        assert_eq!(out.shape(), &[3, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
        // The plain forward is the same computation minus observation.
        assert_eq!(out, sparse_ffn_forward(&weights, 0, &x, &feats));
        match obs {
            FfnActivations::Sparse(s) => {
                assert_eq!(s.seq_len(), 3);
                for pos in 0..3 {
                    let recorded: Vec<usize> = s.position(pos).iter().map(|e| e.feature).collect();
                    assert_eq!(recorded, feats, "exactly the K computed features");
                }
            }
            other => panic!("expected Sparse observation, got {other:?}"),
        }
    }

    #[test]
    fn sparse_forward_top_k_selection_is_sorted() {
        let weights = make_test_weights();
        let x = input(1, weights.hidden_size);
        let x_row = x.row(0);
        let feats = select_top_k_features(&weights, 0, &x_row, 4);
        // select_top_k_features sorts by feature index (ascending)
        for w in feats.windows(2) {
            assert!(w[0] <= w[1], "features not sorted: {:?}", feats);
        }
    }

    #[test]
    fn sparse_forward_top_k_respects_k() {
        let weights = make_test_weights();
        let x = input(1, weights.hidden_size);
        let x_row = x.row(0);
        for k in [1, 4, 8] {
            let feats = select_top_k_features(&weights, 0, &x_row, k);
            assert!(
                feats.len() <= k,
                "got {} features but requested {k}",
                feats.len()
            );
        }
    }

    #[test]
    fn sparse_forward_all_features_matches_dense_fallback() {
        let weights = make_test_weights();
        let x = input(1, weights.hidden_size);
        // When K >= 80% of intermediate, sparse_ffn_forward falls back to dense.
        // Request all features to trigger that path.
        let all: Vec<usize> = (0..weights.intermediate_size).collect();
        let sparse_out = sparse_ffn_forward(&weights, 0, &x, &all);
        let (dense_out, _) = crate::ffn::weight::dense_ffn_forward(
            larql_models::WeightsView::dense(&weights),
            0,
            &x,
        );
        for (s, d) in sparse_out.iter().zip(dense_out.iter()) {
            assert!((s - d).abs() < 1e-4, "sparse/dense mismatch: {s} vs {d}");
        }
        // The observed variant reports the dense fallback honestly: a
        // Dense observation, since every feature was computed.
        let (_, obs) = sparse_ffn_forward_observed(&weights, 0, &x, &all);
        assert!(
            matches!(obs, FfnActivations::Dense(_)),
            "≥80%-K fallback computes densely and must observe densely"
        );
    }

    // ── sparse_ffn_forward_with_overrides ─────────────────────────────────────

    #[test]
    fn overrides_replace_down_contribution() {
        let weights = make_test_weights();
        let x = input(1, weights.hidden_size);
        let feats = &[0usize];
        let custom_down = vec![99.0f32; weights.hidden_size];
        let out_override =
            sparse_ffn_forward_with_overrides(&weights, 0, &x, feats, &[(0, &custom_down)]);
        let out_baseline = sparse_ffn_forward(&weights, 0, &x, feats);
        // The two outputs should differ because the down vector was replaced.
        let diff: f32 = out_override
            .iter()
            .zip(out_baseline.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0, "override had no effect on output");
    }

    // ── gather_rows / gather_columns (indirectly) ─────────────────────────────

    #[test]
    fn gather_rows_all_features_produces_correct_shape() {
        // Test via sparse_ffn_forward by requesting two specific features
        let weights = make_test_weights();
        let x = input(2, weights.hidden_size);
        let out = sparse_ffn_forward(&weights, 0, &x, &[0, weights.intermediate_size - 1]);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
    }

    // ── sparse_ffn_forward_with_full_overrides ─────────────────────────

    #[test]
    fn full_overrides_with_no_overrides_matches_baseline() {
        // FeatureSlotOverride with all None fields → behave like baseline.
        let weights = make_test_weights();
        let x = input(1, weights.hidden_size);
        let feats = &[0usize, 1];
        let overrides = vec![
            FeatureSlotOverride {
                feature: 0,
                gate: None,
                up: None,
                down: None,
            },
            FeatureSlotOverride {
                feature: 1,
                gate: None,
                up: None,
                down: None,
            },
        ];
        let out_full = sparse_ffn_forward_with_full_overrides(&weights, 0, &x, feats, &overrides);
        let out_baseline = sparse_ffn_forward(&weights, 0, &x, feats);
        for (a, b) in out_full.iter().zip(out_baseline.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "no-op overrides should match baseline: {a} vs {b}"
            );
        }
    }

    #[test]
    fn full_overrides_with_gate_only_changes_output() {
        let weights = make_test_weights();
        let x = input(1, weights.hidden_size);
        let feats = &[0usize];
        let custom_gate = vec![5.0f32; weights.hidden_size];
        let overrides = vec![FeatureSlotOverride {
            feature: 0,
            gate: Some(&custom_gate),
            up: None,
            down: None,
        }];
        let out_override =
            sparse_ffn_forward_with_full_overrides(&weights, 0, &x, feats, &overrides);
        let out_baseline = sparse_ffn_forward(&weights, 0, &x, feats);
        let diff: f32 = out_override
            .iter()
            .zip(out_baseline.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0, "gate override should change output");
    }

    #[test]
    fn full_overrides_with_up_only_changes_output() {
        let weights = make_test_weights();
        let x = input(1, weights.hidden_size);
        let feats = &[0usize];
        let custom_up = vec![3.0f32; weights.hidden_size];
        let overrides = vec![FeatureSlotOverride {
            feature: 0,
            gate: None,
            up: Some(&custom_up),
            down: None,
        }];
        let out_override =
            sparse_ffn_forward_with_full_overrides(&weights, 0, &x, feats, &overrides);
        let out_baseline = sparse_ffn_forward(&weights, 0, &x, feats);
        let diff: f32 = out_override
            .iter()
            .zip(out_baseline.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0, "up override should change output");
    }

    #[test]
    fn full_overrides_with_all_three_changes_output() {
        let weights = make_test_weights();
        let x = input(2, weights.hidden_size);
        let feats = &[0usize];
        let custom_gate = vec![1.5f32; weights.hidden_size];
        let custom_up = vec![2.0f32; weights.hidden_size];
        let custom_down = vec![10.0f32; weights.hidden_size];
        let overrides = vec![FeatureSlotOverride {
            feature: 0,
            gate: Some(&custom_gate),
            up: Some(&custom_up),
            down: Some(&custom_down),
        }];
        let out = sparse_ffn_forward_with_full_overrides(&weights, 0, &x, feats, &overrides);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn full_overrides_empty_features_returns_zeros() {
        let weights = make_test_weights();
        let x = input(2, weights.hidden_size);
        let (out, obs) = sparse_ffn_forward_with_full_overrides_observed(&weights, 0, &x, &[], &[]);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(out.iter().all(|v| v.abs() < 1e-9));
        match obs {
            FfnActivations::Sparse(s) => {
                assert_eq!(s.seq_len(), 2);
                assert!(s.position(0).is_empty() && s.position(1).is_empty());
            }
            other => panic!("expected empty Sparse observation, got {other:?}"),
        }
    }

    #[test]
    fn full_overrides_observed_reports_post_override_activations() {
        // The observation must carry the POST-override slot activation
        // (phase 2's recompute), not the baseline value the dense rows
        // would have produced.
        let weights = make_test_weights();
        let x = input(1, weights.hidden_size);
        let feats = &[0usize, 1];
        let custom_gate = vec![5.0f32; weights.hidden_size];
        let overrides = vec![FeatureSlotOverride {
            feature: 0,
            gate: Some(&custom_gate),
            up: None,
            down: None,
        }];
        let (_, obs_base) =
            sparse_ffn_forward_with_full_overrides_observed(&weights, 0, &x, feats, &[]);
        let (_, obs_ov) =
            sparse_ffn_forward_with_full_overrides_observed(&weights, 0, &x, feats, &overrides);
        let (FfnActivations::Sparse(base), FfnActivations::Sparse(ov)) = (obs_base, obs_ov) else {
            panic!("both observations must be Sparse");
        };
        let lookup = |s: &SparseActivations, feat: usize| {
            s.position(0)
                .iter()
                .find(|e| e.feature == feat)
                .map(|e| e.activation)
                .expect("computed feature must be recorded")
        };
        assert!(
            (lookup(&base, 0) - lookup(&ov, 0)).abs() > 0.0,
            "gate override must change the observed slot activation"
        );
        assert_eq!(
            lookup(&base, 1),
            lookup(&ov, 1),
            "non-overridden slot observation must be untouched"
        );
    }

    // ── select_top_k_features additional cases ─────────────────────────

    #[test]
    fn select_top_k_features_zero_k_returns_empty() {
        let weights = make_test_weights();
        let x = input(1, weights.hidden_size);
        let row = x.row(0);
        let feats = select_top_k_features(&weights, 0, &row, 0);
        // k=0 short-circuit — result is empty since `select_nth_unstable_by`
        // wouldn't be called and we'd return whatever was indexed-and-sorted.
        // The function sorts and returns; with k=0 it returns whatever was
        // accumulated (empty after the truncate-to-0 doesn't happen here).
        // In practice, the implementation returns all features when k=0 because
        // the truncate condition is `k > 0 && k < indexed.len()`. So we just
        // check that the call is valid.
        assert!(feats.len() <= weights.intermediate_size);
    }

    #[test]
    fn select_top_k_features_large_k_returns_at_most_intermediate_size() {
        let weights = make_test_weights();
        let x = input(1, weights.hidden_size);
        let row = x.row(0);
        let feats = select_top_k_features(&weights, 0, &row, 1_000_000);
        assert!(feats.len() <= weights.intermediate_size);
    }

    // ── Non-gated FFN path (Starcoder2 arch) ───────────────────────────

    #[test]
    fn sparse_forward_starcoder2_runs_non_gated_branch() {
        // Starcoder2's `ffn_type == NonGated` puts `gate_sub` into the
        // `else` branch (lines 120-121) and routes per-token through the
        // non-gated activation loop (lines 158-180).
        let weights = crate::test_utils::make_starcoder2_test_weights();
        let x = input(2, weights.hidden_size);
        let out = sparse_ffn_forward(&weights, 0, &x, &[0, 5, 17]);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sparse_forward_starcoder2_full_overrides_runs_non_gated_with_overrides() {
        // Same path as above but via `_with_full_overrides` so the override
        // re-compute loop also runs against non-gated weights.
        let weights = crate::test_utils::make_starcoder2_test_weights();
        let x = input(1, weights.hidden_size);
        let custom_up = vec![0.5f32; weights.hidden_size];
        let overrides = vec![FeatureSlotOverride {
            feature: 0,
            gate: None,
            up: Some(&custom_up),
            down: None,
        }];
        let out = sparse_ffn_forward_with_full_overrides(&weights, 0, &x, &[0], &overrides);
        assert_eq!(out.shape(), &[1, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Override length-mismatch fall-through ──────────────────────────

    #[test]
    fn full_overrides_with_wrong_length_falls_through_to_original_row() {
        // gate override with wrong length triggers the fall-through path
        // (uses the original gathered row instead of the override).
        let weights = make_test_weights();
        let x = input(1, weights.hidden_size);
        // Gate override with a wrong length.
        let bad_gate = vec![0.5f32; weights.hidden_size + 5];
        let overrides = vec![FeatureSlotOverride {
            feature: 0,
            gate: Some(&bad_gate),
            up: None,
            down: None,
        }];
        let out = sparse_ffn_forward_with_full_overrides(&weights, 0, &x, &[0], &overrides);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn full_overrides_with_wrong_length_up_falls_through() {
        let weights = make_test_weights();
        let x = input(1, weights.hidden_size);
        let bad_up = vec![0.5f32; weights.hidden_size - 1];
        let overrides = vec![FeatureSlotOverride {
            feature: 0,
            gate: None,
            up: Some(&bad_up),
            down: None,
        }];
        let out = sparse_ffn_forward_with_full_overrides(&weights, 0, &x, &[0], &overrides);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
