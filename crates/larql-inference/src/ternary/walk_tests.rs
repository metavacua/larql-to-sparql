//! `walk_tests` for [`super`].
//!
//! Split out of `ternary.rs` to keep the implementation file within
//! the repo's per-file size budget.

use super::*;
use larql_compute::cpu::ops::ternary_matvec::BitLinearWeight;

fn tiny_model() -> BitnetModel {
    let hidden = 4;
    let inter = 4;
    let vocab = 8;
    let n_heads = 1;
    let head_dim = hidden / n_heads;
    let mk_w = |rows: usize, cols: usize, scale: f32| {
        let mut bytes = vec![0u8; rows * cols / 4];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = match i % 4 {
                0 => 0b01_10_00_01,
                1 => 0b10_01_01_00,
                2 => 0b00_01_10_01,
                _ => 0b01_00_01_10,
            };
        }
        BitLinearWeight::new(rows, cols, bytes, vec![scale; rows]).unwrap()
    };
    let mk_layer = || BitnetLayer {
        attn_norm: vec![1.0; hidden],
        attn_q: mk_w(hidden, hidden, 0.3),
        attn_k: mk_w(hidden, hidden, 0.4),
        attn_v: mk_w(hidden, hidden, 0.5),
        attn_sub_norm: vec![1.0; hidden],
        attn_o: mk_w(hidden, hidden, 0.6),
        ffn: BitNetFfn {
            gate: mk_w(inter, hidden, 0.2),
            up: mk_w(inter, hidden, 0.3),
            down: mk_w(hidden, inter, 0.7),
            ffn_norm: vec![1.0; hidden],
            ffn_sub_norm: vec![1.0; inter],
            eps: 1e-5,
        },
    };
    BitnetModel {
        layers: vec![mk_layer(), mk_layer()],
        embed: Array2::from_shape_fn((vocab, hidden), |(i, j)| {
            ((i * 7 + j * 3) as f32 % 5.0) - 2.0
        }),
        embed_scale: 1.0,
        output_norm: vec![1.0; hidden],
        lm_head: Array2::from_shape_fn((vocab, hidden), |(i, j)| {
            ((i * 11 + j * 5) as f32 % 4.0) - 1.5
        }),
        eps: 1e-5,
        head_dim,
        n_q_heads: n_heads,
        n_kv_heads: n_heads,
        rope_base: 10000.0,
    }
}

fn tiny_tokenizer() -> larql_vindex::tokenizers::Tokenizer {
    let tok_json = r#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]},"added_tokens":[]}"#;
    larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap()
}

/// One residual per layer, captured at the last-token position.
/// Width must match `hidden`.
#[test]
fn predict_bitnet_with_residuals_emits_one_per_layer() {
    let model = tiny_model();
    let tok = tiny_tokenizer();
    let tokens = vec![0u32, 1, 2];
    let (preds, residuals) = predict_bitnet_with_residuals(&model, &tok, &tokens, 3);
    assert_eq!(preds.len(), 3);
    assert_eq!(residuals.len(), model.layers.len());
    for (i, (layer_idx, r)) in residuals.iter().enumerate() {
        assert_eq!(*layer_idx, i, "layer index sequence");
        assert_eq!(r.len(), model.embed.shape()[1], "hidden width");
    }
}

/// Top-K from `predict_bitnet_with_residuals` matches the
/// legacy `predict_bitnet` (same top-K tokens in the same order).
/// Guards against drift in the shared softmax_topk helper.
#[test]
fn predict_with_residuals_matches_legacy_top_k() {
    let model = tiny_model();
    let tok = tiny_tokenizer();
    let tokens = vec![0u32, 1, 2, 3];
    let legacy = predict_bitnet(&model, &tok, &tokens, 5);
    let (with_res, _) = predict_bitnet_with_residuals(&model, &tok, &tokens, 5);
    assert_eq!(legacy.len(), with_res.len());
    for (a, b) in legacy.iter().zip(with_res.iter()) {
        assert_eq!(a.token, b.token);
        assert!(
            (a.probability - b.probability).abs() < 1e-9,
            "{} vs {}",
            a.probability,
            b.probability,
        );
    }
}

/// `infer_bitnet_walk` with no KNN store: knn_override is None,
/// predictions == raw bitnet predictions, model_top1 = predictions[0].
#[test]
fn walk_without_knn_store_passes_predictions_through() {
    let model = tiny_model();
    let tok = tiny_tokenizer();
    let tokens = vec![0u32, 1, 2];
    let result = infer_bitnet_walk(&model, &tok, None, &tokens, 4);
    assert!(result.knn_override.is_none());
    assert_eq!(result.predictions.len(), 4);
    let raw = predict_bitnet(&model, &tok, &tokens, 4);
    assert_eq!(result.predictions.len(), raw.len());
    assert_eq!(result.model_top1.as_ref().unwrap().0, raw[0].token);
}

/// Empty tokens: walk returns empty everything, doesn't panic.
#[test]
fn walk_empty_tokens_returns_empty() {
    let model = tiny_model();
    let tok = tiny_tokenizer();
    let result = infer_bitnet_walk(&model, &tok, None, &[], 5);
    assert!(result.predictions.is_empty());
    assert!(result.residuals.is_empty());
    assert!(result.model_top1.is_none());
}
