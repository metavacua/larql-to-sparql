//! Integration tests for real model strategies.
//!
//! These tests require:
//!   1. The `real-model` feature flag
//!   2. A downloaded Gemma 3-4B model (via HuggingFace)
//!   3. A built vindex for that model
//!
//! Run with:
//!   cargo test --features real-model -p kv-cache-benchmark --test test_real_model -- --ignored
//!
//! All tests are #[ignore] by default since they need model weights.

#![cfg(feature = "real-model")]

use kv_cache_benchmark::real_model::*;
use kv_cache_benchmark::real_model::runner::*;

/// Helper to load model + vindex for tests. Returns None if model not available.
fn load_test_model() -> Option<(
    larql_inference::InferenceModel,
    larql_vindex::VectorIndex,
)> {
    let model = larql_inference::InferenceModel::load("google/gemma-3-4b-it").ok()?;

    let vindex_dir = larql_vindex::resolve_hf_vindex("google/gemma-3-4b-it").ok()?;
    let config = larql_vindex::load_vindex_config(&vindex_dir).ok()?;
    let index = larql_vindex::VectorIndex::load(
        &vindex_dir, &config, &larql_vindex::SilentLoadCallbacks,
    ).ok()?;

    Some((model, index))
}

#[test]
#[ignore]
fn test_all_strategies_produce_paris() {
    let (model, index) = load_test_model().expect("Model not available");
    let backend = larql_inference::default_backend();

    let bench = RealModelBenchmark::new(
        model.weights(), model.tokenizer(), &index, backend.as_ref(),
    );

    let results = run_all_strategies(&bench, "The capital of France is", 5, 512);

    println!("{}", format_results(&results));

    // Standard KV must predict "Paris"
    assert!(
        results[0].top1_token.contains("Paris"),
        "Standard KV didn't predict Paris: got '{}'",
        results[0].top1_token,
    );

    // Markov RS should match (bit-perfect, same forward pass)
    assert!(
        results[2].top1_match,
        "Markov RS top-1 didn't match baseline: got '{}', expected '{}'",
        results[2].top1_token,
        results[0].top1_token,
    );

    // Graph Walk should ideally get Paris too (for factual queries)
    println!(
        "Graph Walk predicted: '{}' (match={})",
        results[3].top1_token, results[3].top1_match,
    );
}

#[test]
#[ignore]
fn test_markov_rs_bit_perfect() {
    let (model, index) = load_test_model().expect("Model not available");
    let backend = larql_inference::default_backend();

    let bench = RealModelBenchmark::new(
        model.weights(), model.tokenizer(), &index, backend.as_ref(),
    );

    let prompts = default_prompts();
    for prompt in &prompts {
        let results = run_all_strategies(&bench, prompt, 5, 512);

        // Markov RS runs the same forward pass — hidden state must match exactly
        let markov = &results[2];
        if let Some(cosine) = markov.hidden_cosine {
            assert!(
                cosine > 0.9999,
                "Markov RS hidden cosine too low for '{}': {cosine:.6}",
                prompt,
            );
        }

        assert!(
            markov.top1_match,
            "Markov RS didn't match baseline for '{}': got '{}', expected '{}'",
            prompt, markov.top1_token, results[0].top1_token,
        );
    }
}

#[test]
#[ignore]
fn test_turboquant_compression_on_real_vectors() {
    let (model, _index) = load_test_model().expect("Model not available");

    let encoding = model.tokenizer().encode("The capital of France is", true).unwrap();
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    let kv = kv_capture::capture_kv(model.weights(), &token_ids);
    let tq = kv_cache_benchmark::turboquant::TurboQuant::new(4);
    let result = turboquant_layer::apply_turboquant(&kv, &tq);

    println!("TurboQuant 4-bit on real K/V:");
    println!("  MSE:         {:.6}", result.mse);
    println!("  Cosine:      {:.4}", result.cosine_sim);
    println!("  Compression: {:.2}x", result.compression_ratio);
    println!("  Original:    {} bytes", result.original_bytes);
    println!("  Compressed:  {} bytes", result.compressed_bytes);

    // Paper targets: MSE ≤ 0.009, cosine ≥ 0.997
    assert!(result.mse < 0.05, "MSE too high: {}", result.mse);
    assert!(result.cosine_sim > 0.95, "Cosine too low: {}", result.cosine_sim);
    assert!(result.compression_ratio > 2.0, "Compression too low: {}", result.compression_ratio);
}

#[test]
#[ignore]
fn test_multi_turn_memory_bounded() {
    let (model, index) = load_test_model().expect("Model not available");
    let backend = larql_inference::default_backend();

    let bench = RealModelBenchmark::new(
        model.weights(), model.tokenizer(), &index, backend.as_ref(),
    );

    // Simulate growing context
    let base_prompt = "The capital of France is Paris. The capital of Germany is Berlin. ";
    let mut growing_prompt = base_prompt.to_string();

    let mut standard_mems = Vec::new();
    let mut markov_mems = Vec::new();

    for turn in 0..5 {
        let results = run_all_strategies(&bench, &growing_prompt, 5, 512);
        standard_mems.push(results[0].memory_bytes);
        markov_mems.push(results[2].memory_bytes);

        growing_prompt.push_str("The capital of Japan is Tokyo. ");
    }

    // Standard KV memory should grow
    assert!(
        standard_mems.last() > standard_mems.first(),
        "Standard KV memory didn't grow with context",
    );

    // Markov RS memory growth should be much less than Standard KV
    let std_growth = *standard_mems.last().unwrap() as f64 / *standard_mems.first().unwrap() as f64;
    let mrk_growth = *markov_mems.last().unwrap() as f64 / *markov_mems.first().unwrap() as f64;
    println!("Standard KV growth: {std_growth:.2}x, Markov RS growth: {mrk_growth:.2}x");
}

#[test]
#[ignore]
fn test_graph_walk_factual_accuracy() {
    let (model, index) = load_test_model().expect("Model not available");
    let backend = larql_inference::default_backend();

    let bench = RealModelBenchmark::new(
        model.weights(), model.tokenizer(), &index, backend.as_ref(),
    );

    let prompts = default_prompts();
    let mut matches = 0;
    let total = prompts.len();

    for prompt in &prompts {
        let results = run_all_strategies(&bench, prompt, 5, 512);
        let gw = &results[3];
        if gw.top1_match {
            matches += 1;
        }
        println!(
            "  '{}' → Graph Walk: '{}' (match={}, tier={:?})",
            prompt, gw.top1_token, gw.top1_match, gw.tier,
        );
    }

    let accuracy = matches as f64 / total as f64;
    println!("\nGraph Walk factual accuracy: {matches}/{total} = {accuracy:.0}%");
}
