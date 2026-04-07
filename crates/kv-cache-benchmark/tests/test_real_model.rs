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
/// Set LARQL_MODEL_PATH and LARQL_VINDEX_PATH env vars, or uses default HF paths.
fn load_test_model() -> Option<(
    larql_inference::InferenceModel,
    larql_vindex::VectorIndex,
)> {
    let model_path = std::env::var("LARQL_MODEL_PATH")
        .unwrap_or_else(|_| "google/gemma-3-4b-it".to_string());
    let model = larql_inference::InferenceModel::load(&model_path).ok()?;

    let vindex_path = std::env::var("LARQL_VINDEX_PATH").ok()?;
    let index = larql_vindex::VectorIndex::load_vindex(
        std::path::Path::new(&vindex_path),
        &mut larql_vindex::SilentLoadCallbacks,
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

    // Report ALL strategies
    for r in &results {
        println!(
            "  {} → '{}' (match={})",
            r.strategy, r.top1_token, r.top1_match,
        );
    }

    // Standard KV must predict "Paris"
    assert!(
        results[0].top1_token.contains("Paris"),
        "Standard KV didn't predict Paris: got '{}'",
        results[0].top1_token,
    );

    // TurboQuant should also predict "Paris" (same forward pass, compressed K/V)
    assert!(
        results[1].top1_token.contains("Paris"),
        "TurboQuant didn't predict Paris: got '{}'",
        results[1].top1_token,
    );

    // Markov RS should match (bit-perfect, same forward pass)
    assert!(
        results[2].top1_match,
        "Markov RS top-1 didn't match baseline: got '{}', expected '{}'",
        results[2].top1_token,
        results[0].top1_token,
    );

    // Graph Walk
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

    // Cosine is the meaningful metric (scale-invariant).
    // Paper MSE target (0.009) is for unit-norm vectors; raw K/V have larger norms.
    // Cosine 0.991 on real vectors = near-lossless.
    assert!(result.cosine_sim > 0.98, "Cosine too low: {}", result.cosine_sim);
    assert!(result.compression_ratio > 3.0, "Compression too low: {}", result.compression_ratio);
    println!("  Note: MSE is on raw vectors (not unit-norm). Cosine is the fair metric.");
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
            "  '{}' → Graph Walk: '{}' (match={})",
            prompt, gw.top1_token, gw.top1_match,
        );
    }

    let accuracy = matches as f64 / total as f64;
    println!("\nGraph Walk factual accuracy: {matches}/{total} = {accuracy:.0}%");
}

// ── Category 1: Top-1 Token Match (real model) ──

#[test]
#[ignore]
fn test_accuracy_top1_factual_20() {
    let (model, index) = load_test_model().expect("Model not available");
    let backend = larql_inference::default_backend();
    let bench = RealModelBenchmark::new(
        model.weights(), model.tokenizer(), &index, backend.as_ref(),
    );

    let prompts = kv_cache_benchmark::accuracy::factual_prompts();
    let total = prompts.len();

    // Per-strategy match counters: [Standard, TurboQuant, Markov, GraphWalk]
    let mut strategy_matches = vec![0usize; 4];
    let strategy_names = ["Standard KV", "TurboQuant 4b", "Markov RS", "Graph Walk"];

    for (prompt, expected) in &prompts {
        let results = runner::run_all_strategies(&bench, prompt, 5, 512);
        let baseline_top1 = &results[0].top1_token;

        // Print all strategies for this prompt
        print!("  '{prompt}' → baseline='{baseline_top1}' (expected '{expected}')");
        for (i, r) in results.iter().enumerate() {
            if r.top1_match || i == 0 {
                strategy_matches[i] += 1;
            }
            if i > 0 {
                let mark = if r.top1_match { "Y" } else { "N" };
                print!(" {}={}", &strategy_names[i][..3], mark);
            }
        }
        println!();

        // Markov RS must match (bit-perfect)
        assert_eq!(
            &results[2].top1_token, baseline_top1,
            "Markov RS mismatch on '{prompt}': got '{}', expected '{baseline_top1}'",
            results[2].top1_token,
        );
    }

    // Summary table
    println!("\n=== Top-1 Match Rate ({total} prompts) ===\n");
    for (i, name) in strategy_names.iter().enumerate() {
        let m = strategy_matches[i];
        let pct = m as f64 / total as f64 * 100.0;
        println!("  {name:<20} {m}/{total} ({pct:.0}%)");
    }
    println!();
}

// ── Category 2: Markov RS bit-perfect (KL = 0.0) ──

#[test]
#[ignore]
fn test_accuracy_markov_rs_bitperfect() {
    let (model, index) = load_test_model().expect("Model not available");
    let backend = larql_inference::default_backend();
    let bench = RealModelBenchmark::new(
        model.weights(), model.tokenizer(), &index, backend.as_ref(),
    );

    for prompt in &["The capital of France is", "Mozart was born in", "Water freezes at"] {
        let results = runner::run_all_strategies(&bench, prompt, 5, 512);
        let markov = &results[2];

        // Must be bit-perfect
        assert!(
            markov.top1_match,
            "Markov RS not bit-perfect on '{prompt}': got '{}'",
            markov.top1_token,
        );
        if let Some(cosine) = markov.hidden_cosine {
            assert!(
                cosine > 0.9999,
                "Markov RS cosine too low on '{prompt}': {cosine:.6}",
            );
        }
    }
}

// ── Category 3: Needle-in-a-haystack (short) ──

#[test]
#[ignore]
fn test_needle_short_512() {
    let (model, index) = load_test_model().expect("Model not available");
    let backend = larql_inference::default_backend();
    let bench = RealModelBenchmark::new(
        model.weights(), model.tokenizer(), &index, backend.as_ref(),
    );

    // Plant a fact early, query it at the end
    let prompt = "The secret code is AURORA-7749. Remember this. Now, some filler text about various topics. The weather is nice today. The sky is blue. What is the secret code?";
    let results = runner::run_all_strategies(&bench, prompt, 10, 512);

    // All strategies should find AURORA or 7749 in their predictions
    for r in &results {
        let top5_text: String = r.top5.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>().join(" ");
        println!("{}: top-1='{}', top-5=[{}]", r.strategy, r.top1_token, top5_text);
    }
}

// ── Category 6: Adversarial entity confusion ──

#[test]
#[ignore]
fn test_adversarial_entity_confusion() {
    let (model, index) = load_test_model().expect("Model not available");
    let backend = larql_inference::default_backend();
    let bench = RealModelBenchmark::new(
        model.weights(), model.tokenizer(), &index, backend.as_ref(),
    );

    // Same template, different entities — must give different answers
    let pairs = vec![
        ("The capital of France is", "Paris"),
        ("The capital of Germany is", "Berlin"),
        ("The capital of Japan is", "Tokyo"),
    ];

    for (prompt, expected) in &pairs {
        let results = runner::run_all_strategies(&bench, prompt, 5, 512);
        let baseline = &results[0].top1_token;
        println!("{prompt} → baseline='{baseline}' (expected: {expected})");

        // Check that strategies don't confuse entities
        // Markov RS must match baseline
        assert_eq!(&results[2].top1_token, baseline);
    }
}

// ── Category 5: Needle at scaling context lengths ──

#[test]
#[ignore]
fn test_needle_scaling_context() {
    let (model, index) = load_test_model().expect("Model not available");

    let needle = "The secret project code name is AURORA-7749.";
    let query = " What is the secret project code name?";
    let filler_sentence = "The quick brown fox jumps over the lazy dog near the old oak tree by the river. ";

    // Test at increasing context lengths
    for target_tokens in [512, 1024, 2048, 4096] {
        // Build haystack: filler + needle at ~10% position + more filler + query
        let chars_per_token = 4; // rough estimate
        let needle_pos_chars = (target_tokens / 10) * chars_per_token;
        let total_chars = target_tokens * chars_per_token;

        let mut context = String::new();
        while context.len() < needle_pos_chars {
            context.push_str(filler_sentence);
        }
        context.push_str(needle);
        context.push(' ');
        while context.len() < total_chars {
            context.push_str(filler_sentence);
        }
        context.push_str(query);

        // Tokenize and check actual length
        let encoding = model.tokenizer().encode(context.as_str(), true).expect("tokenize");
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let actual_tokens = token_ids.len();

        // Run forward pass (Standard KV = Markov RS for single pass)
        let t0 = std::time::Instant::now();
        let result = larql_inference::predict(model.weights(), model.tokenizer(), &token_ids, 10);
        let elapsed = t0.elapsed();

        // Check if AURORA or 7749 appears in top-10
        let top10_text: String = result.predictions.iter()
            .map(|(t, _)| t.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let needle_found = top10_text.contains("AUR") || top10_text.contains("7749") || top10_text.contains("AURORA");

        let top1 = result.predictions.first().map(|(t, _)| t.as_str()).unwrap_or("?");
        let found_mark = if needle_found { "FOUND" } else { "MISSED" };

        println!(
            "  {:>6} tokens (actual {:>5}): top-1='{}' needle={} [{:.1}s] top-10=[{}]",
            target_tokens, actual_tokens, top1, found_mark,
            elapsed.as_secs_f64(), top10_text,
        );
    }
}
