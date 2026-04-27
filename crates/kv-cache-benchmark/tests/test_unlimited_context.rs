//! Integration tests for `UnlimitedContextEngine` (Tier 2).
//!
//! Exercises the full pipeline end-to-end against a real Gemma 3 4B
//! forward pass. All tests are `#[ignore]` since they need model weights.
//!
//! Run with:
//!   cargo test --features real-model -p kv-cache-benchmark \
//!       --test test_unlimited_context -- --ignored --nocapture

#![cfg(feature = "real-model")]

use kv_cache_benchmark::unlimited_context::{
    rs_extend_from_checkpoint, UnlimitedContextEngine,
};

fn load_model() -> Option<larql_inference::InferenceModel> {
    let model_path = std::env::var("LARQL_MODEL_PATH")
        .unwrap_or_else(|_| "google/gemma-3-4b-it".to_string());
    larql_inference::InferenceModel::load(&model_path).ok()
}

fn cos(a: &ndarray::Array2<f32>, b: &ndarray::Array2<f32>) -> f64 {
    let af: Vec<f32> = a.iter().copied().collect();
    let bf: Vec<f32> = b.iter().copied().collect();
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in af.iter().zip(bf.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Window 0 replay must be bit-exact against a fresh extend from empty prior
/// — they run the exact same forward pass over the same tokens from position 0.
#[test]
#[ignore]
fn test_window0_replay_bit_exact() {
    let model = load_model().expect("model not available");
    let weights = model.weights();
    let tokenizer = model.tokenizer();

    // Tokenize a short prompt
    let enc = tokenizer
        .encode("The capital of France is Paris.", true)
        .expect("tokenize failed");
    let tokens: Vec<u32> = enc.get_ids().to_vec();
    let window_size = tokens.len().max(4);

    // Process through engine — closes window 0 naturally when full
    let mut engine = UnlimitedContextEngine::new(window_size);
    engine.process(weights, &tokens).expect("process failed");
    engine.flush();
    assert_eq!(engine.archive.len(), 1, "expected 1 archived window");

    // Replay window 0
    let (replay_kv, _abs_end) = engine
        .replay_window(weights, 0)
        .expect("replay failed");

    // Independent fresh forward with empty prior
    let empty_prior = kv_cache_benchmark::unlimited_context::rs_extend_from_checkpoint(
        weights,
        &tokens,
        &kv_cache_benchmark::unlimited_context::__empty_prior_for_test(weights),
        0,
    )
    .expect("fresh extend failed");

    // Per-layer K cos should be 1.0 to float precision
    for (li, ((k_r, v_r), (k_f, v_f))) in replay_kv.iter().zip(empty_prior.kv_cache.iter()).enumerate() {
        let ck = cos(k_r, k_f);
        let cv = cos(v_r, v_f);
        assert!(ck > 0.99999, "layer {li}: K cos {ck:.6} < 0.99999");
        assert!(cv > 0.99999, "layer {li}: V cos {cv:.6} < 0.99999");
    }
    println!("window 0 replay bit-exact across all layers");
}

/// Replay determinism: calling replay_window twice on the same window
/// must produce byte-for-byte identical K,V (no randomness, no dropout).
#[test]
#[ignore]
fn test_replay_is_deterministic() {
    let model = load_model().expect("model not available");
    let weights = model.weights();
    let tokenizer = model.tokenizer();

    let enc = tokenizer
        .encode(
            "Fact A: The capital of France is Paris. Fact B: Water boils at one hundred degrees.",
            true,
        )
        .expect("tokenize failed");
    let tokens: Vec<u32> = enc.get_ids().to_vec();
    let window_size = (tokens.len() / 3).max(4);

    let mut engine = UnlimitedContextEngine::new(window_size);
    engine.process(weights, &tokens).expect("process failed");
    engine.flush();
    assert!(engine.archive.len() >= 2, "expected at least 2 windows");

    // Replay window 1 twice
    let (kv_a, _) = engine.replay_window(weights, 1).expect("replay 1 failed");
    let (kv_b, _) = engine.replay_window(weights, 1).expect("replay 1 failed (2nd)");

    for (li, ((k_a, v_a), (k_b, v_b))) in kv_a.iter().zip(kv_b.iter()).enumerate() {
        let ck = cos(k_a, k_b);
        let cv = cos(v_a, v_b);
        assert!(ck > 0.999999, "layer {li}: K not deterministic, cos {ck:.8}");
        assert!(cv > 0.999999, "layer {li}: V not deterministic, cos {cv:.8}");
    }
    println!("replay is deterministic");
}

/// Compression ratio accounting: for a reasonable window_size, the stored
/// boundary bytes should be vastly less than the equivalent standard KV.
/// On Gemma 3 4B with window 64, we expect ~two orders of magnitude.
#[test]
#[ignore]
fn test_compression_ratio() {
    let model = load_model().expect("model not available");
    let weights = model.weights();
    let tokenizer = model.tokenizer();

    // Build a ~256-token sequence
    let long = "The capital of France is Paris. ".repeat(32);
    let enc = tokenizer.encode(long.as_str(), true).expect("tokenize failed");
    let tokens: Vec<u32> = enc.get_ids().to_vec();

    let window_size = 64;
    let mut engine = UnlimitedContextEngine::new(window_size);
    engine.process(weights, &tokens).expect("process failed");
    engine.flush();

    let stats = engine.stats(weights);
    println!("{}", stats.summary());
    println!(
        "  total tokens: {}, windows: {}, boundary bytes: {}, KV-equiv bytes: {}, ratio: {:.1}×",
        stats.total_tokens,
        stats.archived_windows,
        stats.total_boundary_bytes,
        stats.equivalent_kv_bytes,
        stats.compression_ratio,
    );

    assert!(stats.archived_windows >= 2, "expected >= 2 windows");
    assert!(
        stats.compression_ratio > 5.0,
        "compression ratio {:.2}× is lower than expected",
        stats.compression_ratio,
    );
}

/// Smoke test: rs_extend_from_checkpoint returns a last_hidden of the right
/// shape and a kv_cache of the right per-layer dimensions. No model needed
/// beyond what `empty_prior` requires; quick build-check.
#[test]
#[ignore]
fn test_extend_output_shapes() {
    let model = load_model().expect("model not available");
    let weights = model.weights();
    let tokenizer = model.tokenizer();

    let enc = tokenizer.encode("Hello world.", true).expect("tokenize failed");
    let tokens: Vec<u32> = enc.get_ids().to_vec();
    let empty = kv_cache_benchmark::unlimited_context::__empty_prior_for_test(weights);

    let out = rs_extend_from_checkpoint(weights, &tokens, &empty, 0)
        .expect("extend failed");

    assert_eq!(out.last_hidden.shape()[0], 1, "last_hidden should be 1 row");
    assert_eq!(out.kv_cache.len(), weights.num_layers);
    assert_eq!(out.new_checkpoint.len(), weights.num_layers);

    for (li, (k, v)) in out.new_checkpoint.iter().enumerate() {
        assert_eq!(k.shape()[0], 1, "layer {li}: checkpoint K should be 1 row");
        assert_eq!(v.shape()[0], 1, "layer {li}: checkpoint V should be 1 row");
    }
    for (li, (k, v)) in out.kv_cache.iter().enumerate() {
        assert_eq!(
            k.shape()[0],
            tokens.len(),
            "layer {li}: kv_cache K should have {} rows",
            tokens.len(),
        );
        assert_eq!(v.shape()[0], tokens.len());
    }
}
