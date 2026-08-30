//! `kv_cache_tests` for [`super`].
//!
//! Split out of `ternary.rs` to keep the implementation file within
//! the repo's per-file size budget.

// Test-only helper that stayed with the code it exercises.
use super::kv_cache::argmax;

use super::*;
use larql_compute::cpu::ops::ternary_matvec::BitLinearWeight;

/// Reusable tiny model factory: hidden=4, vocab=8, 1 head, 1 layer.
fn tiny_model() -> BitnetModel {
    let hidden = 4;
    let inter = 4;
    let vocab = 8;
    let n_heads = 1;
    let head_dim = hidden / n_heads;
    let mk_w = |rows: usize, cols: usize, scale: f32| {
        // Cycle through ternary patterns so the matvec output
        // varies meaningfully across rows.
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
    let layer = BitnetLayer {
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
    }
}

/// `prefill` should produce exactly the same logits as
/// `predict_bitnet` produces internally for the last position.
/// (predict_bitnet returns top-K after softmax; prefill returns
/// raw logits, so we re-derive the softmax here for comparison.)
#[test]
fn prefill_logits_match_predict_bitnet_top1() {
    let model = tiny_model();
    let tok_json = r#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]},"added_tokens":[]}"#;
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();

    let tokens = vec![0u32, 1, 2, 3];
    let preds = predict_bitnet(&model, &tokenizer, &tokens, 1);
    let (_cache, logits) = prefill(&model, &tokens);
    let argmax_logit = argmax(&logits);
    let predicted = tokenizer.id_to_token(argmax_logit).unwrap();
    assert_eq!(preds[0].token, predicted);
}

/// Prefill cache should hold one row per token in K and V.
#[test]
fn prefill_populates_cache_rows() {
    let model = tiny_model();
    let tokens = vec![0u32, 1, 2, 3, 4];
    let (cache, _logits) = prefill(&model, &tokens);
    assert_eq!(cache.seq_len, tokens.len());
    for (k_layer, v_layer) in cache.k.iter().zip(cache.v.iter()) {
        assert_eq!(k_layer.shape()[0], tokens.len());
        assert_eq!(v_layer.shape()[0], tokens.len());
    }
}

/// A decode_step appends one row to each layer's K and V cache.
#[test]
fn decode_step_grows_cache_by_one() {
    let model = tiny_model();
    let tokens = vec![0u32, 1, 2];
    let (mut cache, _) = prefill(&model, &tokens);
    let before = cache.seq_len;
    let logits = decode_step(&model, &mut cache, 5);
    assert_eq!(cache.seq_len, before + 1);
    assert_eq!(cache.k[0].shape()[0], before + 1);
    assert_eq!(cache.v[0].shape()[0], before + 1);
    assert_eq!(logits.len(), model.lm_head.shape()[0]);
}

/// Greedy generate on a tiny model returns the requested number
/// of tokens (or stops at stop_token).
#[test]
fn generate_produces_max_new_tokens() {
    let model = tiny_model();
    let tok_json = r#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]},"added_tokens":[]}"#;
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();
    let prompt = vec![0u32, 1];
    let out = generate(&model, &tokenizer, &prompt, 4, None);
    assert_eq!(out.len(), 4);
    for &id in &out {
        assert!(id < 8, "vocab=8");
    }
}

/// Decode equivalence: prefilling N tokens then decoding one
/// must produce the same logits as prefilling all N+1 tokens
/// for the last position.  This is the load-bearing correctness
/// test for the cache.
#[test]
fn decode_step_matches_full_prefill_at_last_position() {
    let model = tiny_model();
    let tokens = vec![0u32, 1, 2, 3];

    // Path 1: prefill all then read last_logits.
    let (_, logits_full) = prefill(&model, &tokens);

    // Path 2: prefill the prefix, decode the last token, read
    // the resulting logits.
    let (mut cache, _) = prefill(&model, &tokens[..tokens.len() - 1]);
    let logits_decoded = decode_step(&model, &mut cache, *tokens.last().unwrap());

    // Equivalence within fp noise.  Tolerance is generous because
    // the decode path uses apply_rope_partial_at while the
    // prefill path uses apply_rope, which are subtly different
    // kernels but produce identical output at integer positions.
    assert_eq!(logits_full.len(), logits_decoded.len());
    for (i, (a, b)) in logits_full.iter().zip(logits_decoded.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-3,
            "logit {i}: prefill={a} decoded={b} diff={diff}"
        );
    }
}

/// argmax is stable and returns the right index.
#[test]
fn argmax_picks_max() {
    assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
    assert_eq!(argmax(&[5.0, 0.0, -1.0]), 0);
    // Ties resolve to the first occurrence (consistent with
    // strict `>` test).
    assert_eq!(argmax(&[2.0, 2.0, 2.0]), 0);
}

/// Empty prompt for generate: returns no new tokens.
#[test]
fn generate_empty_prompt_returns_empty() {
    let model = tiny_model();
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();
    let out = generate(&model, &tokenizer, &[], 5, None);
    assert!(out.is_empty());
}

/// Stop token short-circuits generation.
#[test]
fn generate_stops_on_stop_token() {
    let model = tiny_model();
    let tok_json = r#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]},"added_tokens":[]}"#;
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap();
    let prompt = vec![0u32, 1];
    // Set stop_token = the next-token argmax for this tiny
    // model.  We compute it via prefill.
    let (_, logits) = prefill(&model, &prompt);
    let first_pred = argmax(&logits);
    let out = generate(&model, &tokenizer, &prompt, 10, Some(first_pred));
    // Generate breaks before pushing the stop token.
    assert!(
        !out.contains(&first_pred),
        "stop_token leaked into output: {out:?}"
    );
}
