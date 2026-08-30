//! `streaming_tests` for [`super`].
//!
//! Split out of `ternary.rs` to keep the implementation file within
//! the repo's per-file size budget.

use super::*;
use crate::layer_graph::generate::{EosConfig, SamplingConfig};
use larql_compute::cpu::ops::ternary_matvec::BitLinearWeight;

/// Same fixture as kv_cache_tests but inlined here so the streaming
/// suite is self-contained.
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

fn tiny_tokenizer() -> larql_vindex::tokenizers::Tokenizer {
    let tok_json = r#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]},"added_tokens":[]}"#;
    larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap()
}

/// Greedy sampling matches the legacy generate() path token-for-token.
/// This is the load-bearing backwards-compat test \u2014 if generate()
/// drifts from generate_sampled(SamplingConfig::greedy()) we want
/// to know.
#[test]
fn greedy_generate_sampled_matches_legacy_generate() {
    let model = tiny_model();
    let tok = tiny_tokenizer();
    let prompt = vec![0u32, 1, 2];
    let legacy = generate(&model, &tok, &prompt, 5, None);
    let sampled = generate_sampled(&model, &prompt, 5, SamplingConfig::greedy(), None);
    assert_eq!(legacy, sampled);
}

/// Seeded temperature sampling is reproducible: same seed +
/// same prompt = same token stream.
#[test]
fn seeded_sampling_is_deterministic() {
    let model = tiny_model();
    let prompt = vec![0u32, 1];
    let cfg = SamplingConfig::temperature(1.0).with_seed(42);
    let a = generate_sampled(&model, &prompt, 5, cfg, None);
    let b = generate_sampled(&model, &prompt, 5, cfg, None);
    assert_eq!(a, b);
    assert_eq!(a.len(), 5);
}

/// Distinct seeds must produce different streams (with overwhelming
/// probability for vocab=8 over 5 tokens).  Lock guards against a
/// regression where seeding silently no-ops.
#[test]
fn distinct_seeds_diverge() {
    let model = tiny_model();
    let prompt = vec![0u32, 1];
    let a = generate_sampled(
        &model,
        &prompt,
        5,
        SamplingConfig::temperature(1.5).with_seed(1),
        None,
    );
    let b = generate_sampled(
        &model,
        &prompt,
        5,
        SamplingConfig::temperature(1.5).with_seed(99999),
        None,
    );
    // 8^5 = 32768 distinct streams: ~1-in-32768 odds of accidental match.
    assert_ne!(a, b, "seeds {{1, 99999}} produced identical streams");
}

/// Sampling filters are applied: top_k=1 with high temperature
/// produces a single deterministic stream (only one candidate
/// survives top_k=1 truncation, so multinomial degenerates).
/// Note: top_k=1 routes through the sampling code path, not the
/// `is_greedy()` short-circuit, so it can diverge from raw argmax
/// when ties exist in the logits — we only assert that successive
/// runs with the same seed match.
#[test]
fn top_k_one_is_deterministic() {
    let model = tiny_model();
    let prompt = vec![0u32, 1];
    let cfg = SamplingConfig::temperature(2.0).with_top_k(1).with_seed(7);
    let a = generate_sampled(&model, &prompt, 4, cfg, None);
    let b = generate_sampled(&model, &prompt, 4, cfg, None);
    assert_eq!(a, b);
    assert_eq!(a.len(), 4);
}

/// Streaming callback fires once per emitted token with the
/// cumulative-decode delta.
#[test]
fn streaming_callback_fires_per_token() {
    let model = tiny_model();
    let tok = tiny_tokenizer();
    let prompt = vec![0u32, 1];
    let mut events: Vec<(u32, String)> = Vec::new();
    let n = generate_streaming_bitnet(
        &model,
        &tok,
        &prompt,
        4,
        SamplingConfig::greedy(),
        &EosConfig::empty(),
        |id, text, _ms| events.push((id, text.to_string())),
    );
    assert_eq!(n, events.len());
    assert_eq!(n, 4, "no early EOS");
    for (id, text) in &events {
        assert!(!text.is_empty(), "empty delta for token {id}");
    }
    let concat: String = events.iter().map(|(_, s)| s.as_str()).collect();
    assert!(
        !concat.is_empty(),
        "concatenated stream surface form was empty"
    );
}

/// EOS token id halts the stream before emitting that token.
#[test]
fn streaming_stops_on_eos_id() {
    let model = tiny_model();
    let tok = tiny_tokenizer();
    let prompt = vec![0u32, 1];
    let baseline = generate_sampled(&model, &prompt, 1, SamplingConfig::greedy(), None);
    let first = baseline[0];

    let mut emitted = 0;
    let _ = generate_streaming_bitnet(
        &model,
        &tok,
        &prompt,
        10,
        SamplingConfig::greedy(),
        &EosConfig::empty().with_eos_id(first),
        |_, _, _| emitted += 1,
    );
    assert_eq!(emitted, 0, "first sampled token = EOS id, no emits");
}

/// Empty prompt: zero tokens emitted, no callback invocations.
#[test]
fn streaming_empty_prompt_emits_nothing() {
    let model = tiny_model();
    let tok = tiny_tokenizer();
    let mut count = 0;
    let n = generate_streaming_bitnet(
        &model,
        &tok,
        &[],
        5,
        SamplingConfig::greedy(),
        &EosConfig::empty(),
        |_, _, _| count += 1,
    );
    assert_eq!(n, 0);
    assert_eq!(count, 0);
}
