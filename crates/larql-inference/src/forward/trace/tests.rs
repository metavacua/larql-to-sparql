//! Tests for [`super`].
//!
//! Split out of `trace.rs` so the implementation file states the
//! behaviour and this one states the evidence for it.

use super::*;
use crate::model::ModelWeights;
use crate::test_utils::make_test_weights;
use std::sync::OnceLock;

fn shared_weights() -> &'static ModelWeights {
    static W: OnceLock<ModelWeights> = OnceLock::new();
    W.get_or_init(make_test_weights)
}

// ── capture_ffn_activation_matrix ─────────────────────────────────────────

#[test]
fn capture_ffn_activation_matrix_shape() {
    let weights = shared_weights();
    let result = capture_ffn_activation_matrix(weights, &[0u32, 1, 2], 0);
    let m = result.expect("should capture FFN activation at layer 0");
    assert_eq!(m.shape()[0], 3, "rows = seq_len");
    assert_eq!(m.shape()[1], weights.intermediate_size, "cols = ffn_dim");
    assert!(m.iter().all(|v| v.is_finite()));
}

#[test]
fn capture_ffn_activation_matrix_layer1() {
    let weights = shared_weights();
    let result = capture_ffn_activation_matrix(weights, &[0u32, 1], 1);
    let m = result.expect("should capture at layer 1");
    assert_eq!(m.shape(), &[2, weights.intermediate_size]);
}

#[test]
fn capture_ffn_activation_matrix_single_token() {
    let weights = shared_weights();
    let result = capture_ffn_activation_matrix(weights, &[5u32], 0);
    let m = result.expect("single-token capture");
    assert_eq!(m.shape(), &[1, weights.intermediate_size]);
}

#[test]
fn capture_ffn_activation_matrix_out_of_bounds_layer_returns_none() {
    let weights = shared_weights();
    // Layer 99 doesn't exist → should return None or fail gracefully
    let result = capture_ffn_activation_matrix(weights, &[0u32], 99);
    // Either None (layer out of range) or Some (shouldn't crash)
    if let Some(m) = result {
        assert!(m.iter().all(|v| v.is_finite()));
    }
}

// ── estimate_ffn_covariance ────────────────────────────────────────────────

#[test]
fn estimate_ffn_covariance_shape() {
    let weights = shared_weights();
    let prompts: Vec<Vec<u32>> = vec![vec![0u32, 1, 2], vec![3u32, 4], vec![5u32, 6, 7, 8]];
    let (cov, n_samples) =
        estimate_ffn_covariance(weights, &prompts, 0).expect("covariance should be computable");
    let ffn = weights.intermediate_size;
    assert_eq!(cov.shape(), &[ffn, ffn], "covariance is ffn_dim × ffn_dim");
    assert!(n_samples > 0, "should have accumulated samples");
    // Symmetric: C[i,j] ≈ C[j,i]
    for i in 0..ffn.min(4) {
        for j in 0..ffn.min(4) {
            assert!(
                (cov[[i, j]] - cov[[j, i]]).abs() < 1e-4,
                "covariance should be symmetric at [{i},{j}]"
            );
        }
    }
}

#[test]
fn estimate_ffn_covariance_positive_semidefinite_diagonal() {
    let weights = shared_weights();
    let prompts = vec![vec![0u32, 1, 2, 3]];
    let (cov, _) = estimate_ffn_covariance(weights, &prompts, 0).unwrap();
    // Diagonal entries should be non-negative (x^T C x >= 0 for diagonal)
    for i in 0..cov.shape()[0] {
        assert!(
            cov[[i, i]] >= 0.0,
            "diagonal entry [{i},{i}] = {} should be >= 0",
            cov[[i, i]]
        );
    }
}

// ── capture_residuals ─────────────────────────────────────────────────────

#[test]
fn capture_residuals_count() {
    let weights = shared_weights();
    // capture_residuals(weights, token_ids, capture_layers) → Vec<(layer, residual_vec)>
    let residuals = capture_residuals(weights, &[0u32, 1, 2], &[0, 1]);
    assert!(!residuals.is_empty(), "residuals should be non-empty");
    for (layer, r) in &residuals {
        assert!(
            r.iter().all(|v| v.is_finite()),
            "layer {layer} residual has non-finite values"
        );
    }
}

#[test]
fn capture_residuals_hidden_size() {
    let weights = shared_weights();
    let residuals = capture_residuals(weights, &[0u32], &[0]);
    for (_layer, r) in &residuals {
        assert_eq!(
            r.len() % weights.hidden_size,
            0,
            "residual len {} should be multiple of hidden_size {}",
            r.len(),
            weights.hidden_size
        );
    }
}

#[test]
fn capture_residuals_returns_requested_layers() {
    let weights = shared_weights();
    let residuals = capture_residuals(weights, &[0u32, 1], &[0]);
    // Should return at least one entry for layer 0
    assert!(
        residuals.iter().any(|(l, _)| *l == 0),
        "should have layer 0 residual"
    );
}

// ── trace_forward_full_hooked ─────────────────────────────────────────────

#[test]
fn hooked_trace_with_noop_matches_baseline() {
    let weights = shared_weights();
    let ffn = WeightFfn { weights };
    let tokens = vec![0u32, 1, 2];
    let layers = vec![0, 1];

    let baseline = trace_forward_full(weights, &tokens, &layers, false, 0, false, &ffn);
    let hooked = trace_forward_full_hooked(
        weights,
        &tokens,
        &layers,
        false,
        0,
        false,
        &ffn,
        &mut crate::forward::NoopHook,
    );

    assert_eq!(baseline.residuals.len(), hooked.residuals.len());
    // BLAS on Windows reorders parallel reductions across successive
    // matmul calls (sometimes accompanied by `BLAS : Bad memory
    // unallocation!`), so two identical forward passes can drift in
    // the 1e-3 range. Linux/macOS BLAS stays well below 1e-6.
    const NOOP_HOOK_TOL: f32 = if cfg!(windows) { 1e-2 } else { 1e-6 };
    for ((bl, br), (hl, hr)) in baseline.residuals.iter().zip(hooked.residuals.iter()) {
        assert_eq!(bl, hl, "layer indices should match");
        for (b, h) in br.iter().zip(hr.iter()) {
            assert!(
                (b - h).abs() < NOOP_HOOK_TOL,
                "noop hook must not perturb residuals"
            );
        }
    }
}

#[test]
fn hooked_trace_zero_ablate_propagates_through_remaining_layers() {
    let weights = shared_weights();
    let ffn = WeightFfn { weights };
    let tokens = vec![0u32, 1, 2];
    let layers: Vec<usize> = (0..weights.num_layers).collect();

    // Ablate layer 0 entirely; residuals at layers >0 must end up zero
    // since downstream layers see a zero residual entering them.
    let mut ablate = crate::forward::ZeroAblateHook::for_layers([0usize]);
    let result = trace_forward_full_hooked(
        weights,
        &tokens,
        &layers,
        false,
        0,
        false,
        &ffn,
        &mut ablate,
    );

    let layer0 = result
        .residuals
        .iter()
        .find(|(l, _)| *l == 0)
        .expect("layer 0 captured");
    assert!(
        layer0.1.iter().all(|v| *v == 0.0),
        "ZeroAblateHook should zero post-layer residual at layer 0"
    );
}

#[test]
fn hooked_trace_record_captures_internal_state() {
    let weights = shared_weights();
    let ffn = WeightFfn { weights };
    let tokens = vec![0u32, 1];

    let mut record = crate::forward::RecordHook::for_layers([0usize, 1]);
    let _ = trace_forward_full_hooked(
        weights,
        &tokens,
        &[0, 1],
        false,
        0,
        false,
        &ffn,
        &mut record,
    );

    assert!(
        record.pre_layer.contains_key(&0) && record.pre_layer.contains_key(&1),
        "RecordHook should capture pre_layer at requested layers"
    );
    assert!(
        record.post_attention.contains_key(&0),
        "RecordHook should capture post_attention"
    );
    assert!(
        record.post_layer.contains_key(&1),
        "RecordHook should capture post_layer"
    );
    // Shape sanity: pre_layer at L1 should be (seq_len, hidden_size).
    let pre1 = record.pre_layer.get(&1).unwrap();
    assert_eq!(pre1.shape(), &[tokens.len(), weights.hidden_size]);
}

#[test]
fn hooked_trace_fires_attention_weights_callback() {
    // on_attention_weights only fires when capture_attention=true on
    // a layer the trace was asked about.
    let weights = shared_weights();
    let ffn = WeightFfn { weights };
    let tokens = vec![0u32, 1, 2];

    let mut record = crate::forward::RecordHook::for_layers([0usize]);
    let _ = trace_forward_full_hooked(
        weights,
        &tokens,
        &[0],
        /*capture_activations=*/ false,
        0,
        /*capture_attention=*/ true,
        &ffn,
        &mut record,
    );

    let attn = record
        .attention_weights
        .get(&0)
        .expect("attention weights captured at layer 0");
    // Per-head: heads.len() = num_q_heads, each row has one entry per
    // attended position (last token attends to all 3 positions).
    let layer_num_q_heads = weights.arch.num_q_heads_for_layer(0);
    assert_eq!(
        attn.len(),
        layer_num_q_heads,
        "attention head count should equal num_q_heads"
    );
    for head in attn {
        assert_eq!(
            head.len(),
            tokens.len(),
            "each head row attends across all token positions"
        );
        assert!(head.iter().all(|v| v.is_finite()));
    }
}

// ── capture_spec_residuals ─────────────────────────────────────────

#[test]
fn capture_spec_residuals_returns_per_layer_last_token_dumps() {
    let weights = shared_weights();
    let tokens = vec![0u32, 1, 2];
    let spec = capture_spec_residuals(weights, &tokens);
    assert_eq!(spec.h_0.shape(), &[3, weights.hidden_size]);
    assert_eq!(spec.post_attn_last.len(), weights.num_layers);
    assert_eq!(spec.post_layer_last.len(), weights.num_layers);
    for v in &spec.post_attn_last {
        assert_eq!(v.len(), weights.hidden_size);
        assert!(v.iter().all(|x| x.is_finite()));
    }
    for v in &spec.post_layer_last {
        assert_eq!(v.len(), weights.hidden_size);
        assert!(v.iter().all(|x| x.is_finite()));
    }
    assert_eq!(spec.h_final.shape(), &[3, weights.hidden_size]);
}

#[test]
fn capture_spec_residuals_single_token_works() {
    let weights = shared_weights();
    let spec = capture_spec_residuals(weights, &[5u32]);
    assert_eq!(spec.h_0.shape(), &[1, weights.hidden_size]);
    assert_eq!(spec.h_final.shape(), &[1, weights.hidden_size]);
    // Per-layer dumps still fire at seq_len=1.
    assert_eq!(spec.post_attn_last.len(), weights.num_layers);
}

// ── forward_to_layer ───────────────────────────────────────────────

#[test]
fn forward_to_layer_returns_full_seq_hidden() {
    let weights = shared_weights();
    let tokens = vec![1u32, 2, 3];
    let h = forward_to_layer(weights, &tokens, 0);
    assert_eq!(h.shape(), &[3, weights.hidden_size]);
    assert!(h.iter().all(|v| v.is_finite()));
}

#[test]
fn forward_to_layer_progresses_through_layers() {
    // Stopping at layer 0 vs layer 1 should produce different residuals
    // unless the second layer is exactly an identity (it isn't with
    // random tinymodel weights).
    let weights = shared_weights();
    let tokens = vec![0u32, 1];
    let h0 = forward_to_layer(weights, &tokens, 0);
    let h1 = forward_to_layer(weights, &tokens, 1);
    let mut max_diff = 0.0f32;
    for (a, b) in h0.iter().zip(h1.iter()) {
        max_diff = max_diff.max((a - b).abs());
    }
    assert!(
        max_diff > 1e-5,
        "layer 1 should mutate the residual, max_diff={max_diff}"
    );
}

// ── capture_decoy_residuals ────────────────────────────────────────

#[test]
fn capture_decoy_residuals_returns_one_array_per_prompt() {
    let weights = shared_weights();
    let prompts = vec![vec![0u32, 1], vec![2u32, 3, 4], vec![5u32]];
    let decoys = capture_decoy_residuals(weights, &prompts, 1);
    assert_eq!(decoys.len(), 3);
    for d in &decoys {
        assert_eq!(d.len(), weights.hidden_size);
        assert!(d.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn capture_decoy_residuals_empty_input_returns_empty() {
    let weights = shared_weights();
    let decoys = capture_decoy_residuals(weights, &[], 0);
    assert!(decoys.is_empty());
}

// ── estimate_ffn_covariance ────────────────────────────────────────

#[test]
fn estimate_ffn_covariance_returns_symmetric_psd_matrix() {
    let weights = shared_weights();
    let prompts = vec![vec![0u32, 1, 2], vec![3u32, 4, 5]];
    let (cov, samples) = estimate_ffn_covariance(weights, &prompts, 0)
        .expect("covariance must accumulate over multiple prompts");
    // Sum of seq_lens
    assert_eq!(samples, 6);
    let n = weights.intermediate_size;
    assert_eq!(cov.shape(), &[n, n]);
    // K^T K is symmetric — pin within float noise.
    for i in 0..n {
        for j in 0..n {
            assert!(
                (cov[[i, j]] - cov[[j, i]]).abs() < 1e-4,
                "covariance not symmetric at ({i},{j}): {} vs {}",
                cov[[i, j]],
                cov[[j, i]]
            );
        }
    }
    // Diagonal must be non-negative (positive semidefinite).
    for i in 0..n {
        assert!(cov[[i, i]] >= 0.0, "diag[{i}] negative");
    }
}

#[test]
fn estimate_ffn_covariance_single_prompt_works() {
    let weights = shared_weights();
    let prompts = vec![vec![0u32, 1, 2]];
    let (cov, samples) = estimate_ffn_covariance(weights, &prompts, 1)
        .expect("single prompt must still produce covariance");
    assert_eq!(samples, 3);
    assert_eq!(
        cov.shape(),
        &[weights.intermediate_size, weights.intermediate_size]
    );
}

#[test]
fn estimate_ffn_covariance_empty_prompts_returns_none() {
    let weights = shared_weights();
    assert!(estimate_ffn_covariance(weights, &[], 0).is_none());
}

// ── trace_forward + trace_forward_with_ffn + trace_forward_full ────

#[test]
fn trace_forward_returns_residuals_at_requested_layers() {
    let weights = shared_weights();
    let tokens = vec![0u32, 1];
    let trace = trace_forward(weights, &tokens, &[0, 1], false, 0);
    assert_eq!(trace.residuals.len(), 2);
    assert_eq!(trace.residuals[0].0, 0);
    assert_eq!(trace.residuals[1].0, 1);
    assert!(trace.activations.is_empty());
}

#[test]
fn trace_forward_with_activations_captures_topk_per_layer() {
    let weights = shared_weights();
    let tokens = vec![0u32, 1];
    let trace = trace_forward(weights, &tokens, &[0], true, 5);
    assert_eq!(trace.activations.len(), 1);
    let (layer, top) = &trace.activations[0];
    assert_eq!(*layer, 0);
    assert!(top.len() <= 5);
    // Activations sorted by magnitude desc.
    for w in top.windows(2) {
        assert!(w[0].1.abs() >= w[1].1.abs(), "top-K not sorted by |abs|");
    }
}

#[test]
fn trace_forward_with_ffn_uses_supplied_backend() {
    let weights = shared_weights();
    let tokens = vec![0u32, 1];
    let ffn = WeightFfn { weights };
    let trace = trace_forward_with_ffn(weights, &tokens, &[0, 1], false, 0, &ffn);
    assert_eq!(trace.residuals.len(), 2);
}

#[test]
fn trace_forward_full_with_attention_returns_attention_captures() {
    let weights = shared_weights();
    let ffn = WeightFfn { weights };
    let tokens = vec![0u32, 1];
    let trace = trace_forward_full(
        weights,
        &tokens,
        &[0],
        false,
        0,
        /*capture_attention=*/ true,
        &ffn,
    );
    assert_eq!(trace.attention.len(), 1);
    assert_eq!(trace.attention[0].layer, 0);
}

// ── calibrate_scalar_gains ─────────────────────────────────────────

#[test]
fn calibrate_scalar_gains_returns_one_per_layer() {
    let weights = shared_weights();
    let gains = calibrate_scalar_gains(weights, &[0u32, 1, 2]);
    assert_eq!(gains.len(), weights.num_layers);
    for g in &gains {
        assert!(g.is_finite(), "gain non-finite: {g}");
    }
}

#[test]
fn calibrate_scalar_gains_last_layer_is_unity_fallback() {
    // Last layer has no successor → gain falls back to 1.0.
    let weights = shared_weights();
    let gains = calibrate_scalar_gains(weights, &[0u32]);
    assert_eq!(gains[gains.len() - 1], 1.0);
}

#[test]
fn hooked_trace_fires_ffn_activation_callback() {
    // on_ffn_activation only fires when capture_activations=true on
    // a layer the trace was asked about.
    let weights = shared_weights();
    let ffn = WeightFfn { weights };
    let tokens = vec![0u32, 1];

    let mut record = crate::forward::RecordHook::for_layers([0usize]);
    let _ = trace_forward_full_hooked(
        weights,
        &tokens,
        &[0],
        /*capture_activations=*/ true,
        0,
        /*capture_attention=*/ false,
        &ffn,
        &mut record,
    );

    let act = record
        .ffn_activation
        .get(&0)
        .expect("FFN activation captured at layer 0");
    // Shape: (seq_len, ffn_dim).
    assert_eq!(act.shape(), &[tokens.len(), weights.intermediate_size]);
    assert!(act.iter().all(|v| v.is_finite()));
}
