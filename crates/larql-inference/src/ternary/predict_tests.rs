//! `predict_tests` for [`super`].
//!
//! Split out of `ternary.rs` to keep the implementation file within
//! the repo's per-file size budget.

// Test-only helper that stayed with the code it exercises.
use super::predict::scaled_dot_product_attention_gqa;

use super::*;

/// End-to-end smoke test: build a tiny synthetic BitNet model
/// (1 layer, hidden=4, vocab=8, head_dim=4, 1 head) and confirm
/// `predict_bitnet` produces a top-K of the right shape with
/// probabilities summing to ~1.
#[test]
fn predict_bitnet_runs_end_to_end_on_synthetic_model() {
    let hidden = 4;
    let inter = 4;
    let vocab = 8;
    let n_heads = 1;
    let head_dim = hidden / n_heads;

    // Tiny tokeniser stub: HF Tokenizer with byte-level vocab.
    // We don't actually need it to decode meaningfully — the
    // test asserts shape + numerical sanity, not which tokens
    // come out.
    let tok_json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]}}"#;
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();

    // Trivial weights: zero everywhere; predict_bitnet should
    // still produce a uniform-ish distribution and not crash.
    let mk_w = |rows: usize, cols: usize| {
        BitLinearWeight::new(rows, cols, vec![0u8; rows * cols / 4], vec![0.1f32; rows]).unwrap()
    };

    let layer = BitnetLayer {
        attn_norm: vec![1.0; hidden],
        attn_q: mk_w(hidden, hidden),
        attn_k: mk_w(hidden, hidden),
        attn_v: mk_w(hidden, hidden),
        attn_sub_norm: vec![1.0; hidden],
        attn_o: mk_w(hidden, hidden),
        ffn: BitNetFfn {
            gate: mk_w(inter, hidden),
            up: mk_w(inter, hidden),
            down: mk_w(hidden, inter),
            ffn_norm: vec![1.0; hidden],
            ffn_sub_norm: vec![1.0; inter],
            eps: 1e-5,
        },
    };

    let model = BitnetModel {
        layers: vec![layer],
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
    };

    let token_ids = vec![0u32, 1, 2, 3];
    let preds = predict_bitnet(&model, &tokenizer, &token_ids, 4);
    assert_eq!(preds.len(), 4, "top_k=4 should return 4 predictions");

    // Probabilities must form a valid prefix of a softmax
    // distribution: each in [0, 1], sorted descending, summing
    // to <= 1 (we only return top-K).
    let mut prev = 1.0_f64;
    let mut sum = 0.0_f64;
    for p in &preds {
        assert!(p.probability >= 0.0 && p.probability <= 1.0);
        assert!(p.probability <= prev + 1e-9);
        prev = p.probability;
        sum += p.probability;
    }
    assert!(sum <= 1.0 + 1e-6, "top-K sum {sum} > 1");
}

/// Empty token_ids returns no predictions.
#[test]
fn predict_bitnet_empty_tokens_returns_empty() {
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();

    let mk_w = |rows: usize, cols: usize| {
        BitLinearWeight::new(rows, cols, vec![0u8; rows * cols / 4], vec![0.0; rows]).unwrap()
    };
    let layer = BitnetLayer {
        attn_norm: vec![1.0; 4],
        attn_q: mk_w(4, 4),
        attn_k: mk_w(4, 4),
        attn_v: mk_w(4, 4),
        attn_sub_norm: vec![1.0; 4],
        attn_o: mk_w(4, 4),
        ffn: BitNetFfn {
            gate: mk_w(4, 4),
            up: mk_w(4, 4),
            down: mk_w(4, 4),
            ffn_norm: vec![1.0; 4],
            ffn_sub_norm: vec![1.0; 4],
            eps: 1e-5,
        },
    };
    let model = BitnetModel {
        layers: vec![layer],
        embed: Array2::zeros((4, 4)),
        embed_scale: 1.0,
        output_norm: vec![1.0; 4],
        lm_head: Array2::zeros((4, 4)),
        eps: 1e-5,
        head_dim: 4,
        n_q_heads: 1,
        n_kv_heads: 1,
        rope_base: 10000.0,
    };
    let preds = predict_bitnet(&model, &tokenizer, &[], 5);
    assert!(preds.is_empty());
}

/// Causal mask self-test: position 0 can only attend to itself,
/// so its attention output must equal v[0] (after the implicit
/// softmax-of-one-element).
#[test]
fn scaled_dot_product_attention_position_zero_is_self_attended() {
    let n_heads = 1;
    let head_dim = 4;
    let q = Array2::from_shape_vec((1, head_dim), vec![1.0, 0.5, -0.5, 0.25]).unwrap();
    let k = q.clone();
    let v = Array2::from_shape_vec((1, head_dim), vec![3.0, -1.0, 2.5, 0.0]).unwrap();
    let mut out = Array2::<f32>::zeros((1, head_dim));
    scaled_dot_product_attention_gqa(
        q.view(),
        k.view(),
        v.view(),
        n_heads,
        n_heads,
        head_dim,
        out.view_mut(),
    );
    for (a, b) in out.row(0).iter().zip(v.row(0).iter()) {
        assert!((a - b).abs() < 1e-5, "expected v, got {a} vs {b}");
    }
}
