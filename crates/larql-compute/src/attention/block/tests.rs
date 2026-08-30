//! Tests for [`super`].
//!
//! Split out of `block.rs` so the implementation file states the
//! behaviour and this one states the evidence for it.

use super::*;
use larql_models::test_fixtures::make_test_weights;
use ndarray::Array2;

fn hidden(rows: usize, hidden: usize) -> Array2<f32> {
    Array2::from_shape_vec(
        (rows, hidden),
        (0..rows * hidden)
            .map(|i| (i as f32 + 1.0) * 0.01)
            .collect(),
    )
    .unwrap()
}

// run_attention_block returns (h_post_attn, attn_proj, attn_weights)
// — the second element is the projected attention output, not K/V.

#[test]
fn attention_block_output_shape() {
    let weights = make_test_weights();
    let h = hidden(3, weights.hidden_size);
    let (h_out, attn_proj, _) =
        run_attention_block(larql_models::WeightsView::dense(&weights), &h, 0, false)
            .expect("run_attention_block failed");
    assert_eq!(h_out.shape(), &[3, weights.hidden_size]);
    assert_eq!(attn_proj.shape()[0], 3);
}

#[test]
fn attention_block_output_finite() {
    let weights = make_test_weights();
    let h = hidden(2, weights.hidden_size);
    let (h_out, _, _) =
        run_attention_block(larql_models::WeightsView::dense(&weights), &h, 0, false).unwrap();
    assert!(h_out.iter().all(|v| v.is_finite()));
}

#[test]
fn attention_block_single_token() {
    let weights = make_test_weights();
    let h = hidden(1, weights.hidden_size);
    let (h_out, attn_proj, _) =
        run_attention_block(larql_models::WeightsView::dense(&weights), &h, 0, false).unwrap();
    assert_eq!(h_out.shape(), &[1, weights.hidden_size]);
    assert_eq!(attn_proj.shape()[0], 1);
}

#[test]
fn attention_block_all_layers() {
    let weights = make_test_weights();
    let h = hidden(2, weights.hidden_size);
    for layer in 0..weights.num_layers {
        assert!(
            run_attention_block(larql_models::WeightsView::dense(&weights), &h, layer, false)
                .is_some(),
            "layer {layer} failed"
        );
    }
}

#[test]
fn attention_block_with_kv_out_returns_kv() {
    let weights = make_test_weights();
    let h = hidden(3, weights.hidden_size);
    let result = run_attention_block_with_kv_out(
        larql_models::WeightsView::dense(&weights),
        &h,
        0,
        false,
        None,
    );
    // Returns (h_post, attn_proj, attn_w, k_rope, v_final) — 5 elements
    let (h_out, _attn_proj, _attn_w, k_rope, v_final) = result.unwrap();
    assert_eq!(h_out.shape(), &[3, weights.hidden_size]);
    assert_eq!(k_rope.shape()[0], 3);
    assert_eq!(v_final.shape()[0], 3);
}

#[test]
fn attention_block_capture_returns_per_head_weights() {
    let weights = make_test_weights();
    let h = hidden(3, weights.hidden_size);
    let (_, _, attn_w) =
        run_attention_block(larql_models::WeightsView::dense(&weights), &h, 0, true).unwrap();
    let aw = attn_w.expect("capture=true must yield weights");
    assert_eq!(aw.heads.len(), weights.num_q_heads);
}

#[test]
fn attention_block_with_pre_o_returns_per_head_pre_projection() {
    let weights = make_test_weights();
    let h = hidden(2, weights.hidden_size);
    let (h_post, pre_o) =
        run_attention_block_with_pre_o(larql_models::WeightsView::dense(&weights), &h, 0).unwrap();
    assert_eq!(h_post.shape(), &[2, weights.hidden_size]);
    // pre_o is `[seq, num_q * head_dim]`.
    assert_eq!(pre_o.shape(), &[2, weights.num_q_heads * weights.head_dim]);
}

#[test]
fn attention_block_shared_with_pre_o_works_under_kv_share() {
    let weights = make_test_weights();
    let h = hidden(2, weights.hidden_size);
    let (_, shared) = crate::attention::run_attention_block_with_kv_out(
        larql_models::WeightsView::dense(&weights),
        &h,
        0,
        false,
        None,
    )
    .map(|(p, _, _, k, v)| (p, (k, v)))
    .unwrap();
    let (h_post, pre_o) = run_attention_block_shared_with_pre_o(
        larql_models::WeightsView::dense(&weights),
        &h,
        1,
        Some(&shared),
    )
    .unwrap();
    assert_eq!(h_post.shape(), &[2, weights.hidden_size]);
    assert_eq!(pre_o.shape()[1], weights.num_q_heads * weights.head_dim);
}

#[test]
fn attention_block_with_all_attention_weights_returns_per_position_dist() {
    let weights = make_test_weights();
    let h = hidden(3, weights.hidden_size);
    let (_, _, all) = run_attention_block_with_pre_o_and_all_attention_weights(
        larql_models::WeightsView::dense(&weights),
        &h,
        0,
        None,
    )
    .unwrap();
    assert_eq!(all.heads.len(), weights.num_q_heads);
    for head in &all.heads {
        assert_eq!(head.len(), 3); // one distribution per Q position
    }
}

#[test]
fn attention_block_with_reduced_qk_attention_weights_clamps_rank() {
    let weights = make_test_weights();
    let h = hidden(2, weights.hidden_size);
    let (_, _, all) = run_attention_block_with_pre_o_and_reduced_qk_attention_weights(
        larql_models::WeightsView::dense(&weights),
        &h,
        0,
        None,
        /*qk_rank=*/ 4, // half of head_dim=8
    )
    .unwrap();
    assert_eq!(all.heads.len(), weights.num_q_heads);
}

// ── Intervention surfaces ──────────────────────────────────────────

#[test]
fn zero_pre_o_heads_changes_output_when_head_active() {
    let weights = make_test_weights();
    let h = hidden(2, weights.hidden_size);
    let baseline = run_attention_block(larql_models::WeightsView::dense(&weights), &h, 0, false)
        .unwrap()
        .0;
    let (zeroed, kv_out) = run_attention_block_zero_pre_o_heads(
        larql_models::WeightsView::dense(&weights),
        &h,
        0,
        &[0],
        None,
    )
    .unwrap();
    assert_eq!(zeroed.shape(), baseline.shape());
    // KV is computed when shared_kv is None.
    assert!(kv_out.is_some());
    let mut max_diff = 0.0f32;
    for (a, b) in baseline.iter().zip(zeroed.iter()) {
        max_diff = max_diff.max((a - b).abs());
    }
    assert!(max_diff > 1e-5, "zeroing head 0 should change output");
}

#[test]
fn zero_pre_o_heads_under_shared_kv_omits_kv_output() {
    let weights = make_test_weights();
    let h = hidden(2, weights.hidden_size);
    let (_, shared) = crate::attention::run_attention_block_with_kv_out(
        larql_models::WeightsView::dense(&weights),
        &h,
        0,
        false,
        None,
    )
    .map(|(p, _, _, k, v)| (p, (k, v)))
    .unwrap();
    let (_, kv_out) = run_attention_block_zero_pre_o_heads(
        larql_models::WeightsView::dense(&weights),
        &h,
        1,
        &[0],
        Some(&shared),
    )
    .unwrap();
    assert!(kv_out.is_none(), "shared-KV path must not return KV");
}

#[test]
fn replace_pre_o_head_substitutes_one_head() {
    let weights = make_test_weights();
    let h = hidden(2, weights.hidden_size);
    // Replacement is `[seq, head_dim]` of all-zeros — equivalent to
    // zeroing that head, so output must differ from baseline.
    let baseline = run_attention_block(larql_models::WeightsView::dense(&weights), &h, 0, false)
        .unwrap()
        .0;
    let zero_head = Array2::<f32>::zeros((2, weights.head_dim));
    let (replaced, _) = run_attention_block_replace_pre_o_head(
        larql_models::WeightsView::dense(&weights),
        &h,
        0,
        0,
        &zero_head,
        None,
    )
    .unwrap();
    let mut max_diff = 0.0f32;
    for (a, b) in baseline.iter().zip(replaced.iter()) {
        max_diff = max_diff.max((a - b).abs());
    }
    assert!(max_diff > 1e-5, "head replacement should change output");
}

#[test]
fn subtract_pre_o_heads_matches_zero_pre_o_heads_numerically() {
    // Both paths zero the head's W_O contribution — output must match.
    let weights = make_test_weights();
    let h = hidden(2, weights.hidden_size);
    let (zeroed, _) = run_attention_block_zero_pre_o_heads(
        larql_models::WeightsView::dense(&weights),
        &h,
        0,
        &[0],
        None,
    )
    .unwrap();
    let (subtracted, _) = run_attention_block_subtract_pre_o_heads(
        larql_models::WeightsView::dense(&weights),
        &h,
        0,
        &[0],
        None,
    )
    .unwrap();
    for (a, b) in zeroed.iter().zip(subtracted.iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "zero-pre-O and subtract-pre-O must match: {a} vs {b}"
        );
    }
}

#[test]
fn replace_head_residual_delta_zero_replacement_matches_zero_head() {
    // A zero residual-space replacement should match the zero-pre-O path
    // up to W_O numerical noise (both eliminate the head contribution).
    let weights = make_test_weights();
    let h = hidden(2, weights.hidden_size);
    let zero_delta = Array2::<f32>::zeros((2, weights.hidden_size));
    let (with_delta, _) = run_attention_block_replace_head_residual_delta(
        larql_models::WeightsView::dense(&weights),
        &h,
        0,
        0,
        &zero_delta,
        None,
    )
    .unwrap();
    let (zeroed, _) = run_attention_block_zero_pre_o_heads(
        larql_models::WeightsView::dense(&weights),
        &h,
        0,
        &[0],
        None,
    )
    .unwrap();
    for (a, b) in with_delta.iter().zip(zeroed.iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "zero residual delta should match zero-head: {a} vs {b}"
        );
    }
}

#[test]
fn intervention_paths_return_none_for_missing_layer() {
    // Layer index past num_layers — every variant must return None
    // gracefully rather than panic.
    let weights = make_test_weights();
    let h = hidden(2, weights.hidden_size);
    let bogus_layer = weights.num_layers + 5;
    assert!(run_attention_block(
        larql_models::WeightsView::dense(&weights),
        &h,
        bogus_layer,
        false
    )
    .is_none());
    assert!(run_attention_block_with_pre_o(
        larql_models::WeightsView::dense(&weights),
        &h,
        bogus_layer
    )
    .is_none());
    assert!(run_attention_block_zero_pre_o_heads(
        larql_models::WeightsView::dense(&weights),
        &h,
        bogus_layer,
        &[0],
        None
    )
    .is_none());
}

// ── Gemma3-arch fixture (post-norms, QK norm, gelu_tanh) ───────────

#[test]
fn attention_block_with_qk_norm_keys_routes_through_qk_norm_branch() {
    // Gemma3 returns Some from attn_q_norm_key/attn_k_norm_key, hitting
    // the rms_norm_heads branch in run_attention_block_core that
    // tinymodel never exercises.
    let weights = larql_models::test_fixtures::make_gemma3_test_weights();
    let h = hidden(2, weights.hidden_size);
    let (h_post, _, _) =
        run_attention_block(larql_models::WeightsView::dense(&weights), &h, 0, false).unwrap();
    assert_eq!(h_post.shape(), &[2, weights.hidden_size]);
    assert!(h_post.iter().all(|v| v.is_finite()));
}

#[test]
fn attention_block_post_norms_arch_runs_through_post_norm_branch() {
    // Gemma3 has has_post_norms=true, so the post-attention path
    // takes a different branch in run_attention_block_core.
    let weights = larql_models::test_fixtures::make_gemma3_test_weights();
    let h = hidden(2, weights.hidden_size);
    let (h_post, _, _) =
        run_attention_block(larql_models::WeightsView::dense(&weights), &h, 1, false).unwrap();
    assert_eq!(h_post.shape(), &[2, weights.hidden_size]);
    assert!(h_post.iter().all(|v| v.is_finite()));
}

// ── Starcoder2-arch fixture (attention + FFN biases) ───────────────

#[test]
fn attention_block_with_q_k_v_o_biases_runs_add_bias_branches() {
    // Starcoder2 returns Some from every attn_*_bias_key, so every
    // `add_bias` call site fires.
    let weights = larql_models::test_fixtures::make_starcoder2_test_weights();
    let h = hidden(2, weights.hidden_size);
    let (h_post, _, _) =
        run_attention_block(larql_models::WeightsView::dense(&weights), &h, 0, false).unwrap();
    assert_eq!(h_post.shape(), &[2, weights.hidden_size]);
    assert!(h_post.iter().all(|v| v.is_finite()));
}

#[test]
fn attention_block_starcoder_with_kv_out_returns_finite_kv() {
    let weights = larql_models::test_fixtures::make_starcoder2_test_weights();
    let h = hidden(3, weights.hidden_size);
    let (_, _, _, k, v) = run_attention_block_with_kv_out(
        larql_models::WeightsView::dense(&weights),
        &h,
        0,
        false,
        None,
    )
    .unwrap();
    assert_eq!(k.shape()[0], 3);
    assert_eq!(v.shape()[0], 3);
    assert!(k.iter().all(|x| x.is_finite()));
    assert!(v.iter().all(|x| x.is_finite()));
}
