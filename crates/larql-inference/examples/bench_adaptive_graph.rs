//! Benchmark the adaptive graph pipeline:
//!   L0-12:  cached (precomputed from template)
//!   L13-33: dense attention + walk FFN (vindex)
//!
//! Measures: correctness, speedup vs dense, cache build time.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example bench_adaptive_graph -- \
//!     --vindex output/gemma3-4b-v2.vindex

use std::time::Instant;

use larql_inference::{
    predict, predict_with_graph,
    InferenceModel, WeightFfn, WalkLayerGraph, DenseLayerGraph,
    CachedLayerGraph, build_adaptive_graph,
};
use larql_inference::vindex::WalkFfn;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut vindex_path = std::path::PathBuf::from("output/gemma3-4b-v2.vindex");
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--vindex" { i += 1; vindex_path = std::path::PathBuf::from(&args[i]); }
        i += 1;
    }

    let model = InferenceModel::load("google/gemma-3-4b-it")?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let num_layers = weights.num_layers;

    // Load vindex
    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    index.load_down_features(&vindex_path)?;
    index.load_up_features(&vindex_path)?;

    let top_k_features = 8092;
    let walk_ffn = WalkFfn::new(weights, &index, top_k_features);
    let dense_ffn = WeightFfn { weights };

    println!("=== Adaptive Graph Benchmark ===\n");

    let prompts = [
        ("capital", "The capital of France is"),
        ("language", "The language spoken in Japan is"),
        ("currency", "The currency of the United Kingdom is the"),
        ("born", "Albert Einstein was born in"),
    ];

    let n = 3;

    for (tname, prompt) in &prompts {
        let encoding = tokenizer.encode(*prompt, true).map_err(|e| format!("{e}"))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        println!("--- {tname}: \"{prompt}\" ({} tokens) ---", token_ids.len());

        // Dense baseline
        let _ = predict(weights, tokenizer, &token_ids, 5);
        let t0 = Instant::now();
        for _ in 0..n { let _ = predict(weights, tokenizer, &token_ids, 5); }
        let dense_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let dense_result = predict(weights, tokenizer, &token_ids, 5);
        let (dense_tok, dense_prob) = dense_result.predictions.first()
            .map(|(t, p)| (t.clone(), *p)).unwrap_or_default();

        // Walk (full, no cache)
        let walk_graph = WalkLayerGraph { ffn: &walk_ffn, backend: None };
        let _ = predict_with_graph(weights, tokenizer, &token_ids, 5, &walk_graph);
        let t0 = Instant::now();
        for _ in 0..n { let _ = predict_with_graph(weights, tokenizer, &token_ids, 5, &walk_graph); }
        let walk_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let walk_result = predict_with_graph(weights, tokenizer, &token_ids, 5, &walk_graph);
        let (walk_tok, walk_prob) = walk_result.predictions.first()
            .map(|(t, p)| (t.clone(), *p)).unwrap_or_default();

        // Build cache for L0-12 using this template's tokens
        let cached_layers: Vec<usize> = (0..=12).collect();
        let t_cache = Instant::now();
        let cache = CachedLayerGraph::build(weights, &token_ids, &cached_layers, &dense_ffn);
        let cache_ms = t_cache.elapsed().as_secs_f64() * 1000.0;

        // Adaptive: cached L0-12 + walk L13-33
        let cached_range = 0..=12;
        let adaptive = build_adaptive_graph(&cache, &walk_graph, num_layers, &cached_range);

        let _ = predict_with_graph(weights, tokenizer, &token_ids, 5, &adaptive);
        let t0 = Instant::now();
        for _ in 0..n { let _ = predict_with_graph(weights, tokenizer, &token_ids, 5, &adaptive); }
        let adaptive_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let adaptive_result = predict_with_graph(weights, tokenizer, &token_ids, 5, &adaptive);
        let (adaptive_tok, adaptive_prob) = adaptive_result.predictions.first()
            .map(|(t, p)| (t.clone(), *p)).unwrap_or_default();

        // Adaptive with dense fallback (cached L0-12 + dense L13-33)
        let dense_graph = DenseLayerGraph { ffn: &dense_ffn, backend: None, capture_activation: false, capture_attention: false };
        let adaptive_dense = build_adaptive_graph(&cache, &dense_graph, num_layers, &cached_range);
        let _ = predict_with_graph(weights, tokenizer, &token_ids, 5, &adaptive_dense);
        let t0 = Instant::now();
        for _ in 0..n { let _ = predict_with_graph(weights, tokenizer, &token_ids, 5, &adaptive_dense); }
        let ad_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let ad_result = predict_with_graph(weights, tokenizer, &token_ids, 5, &adaptive_dense);
        let (ad_tok, ad_prob) = ad_result.predictions.first()
            .map(|(t, p)| (t.clone(), *p)).unwrap_or_default();

        println!("  Dense:            {dense_tok:>10} ({:.2}%)  {dense_ms:>6.0}ms",
            dense_prob * 100.0);
        println!("  Walk (full):      {walk_tok:>10} ({:.2}%)  {walk_ms:>6.0}ms",
            walk_prob * 100.0);
        println!("  Cache+Walk:       {adaptive_tok:>10} ({:.2}%)  {adaptive_ms:>6.0}ms  (cache build: {cache_ms:.0}ms, {cached} layers cached)",
            adaptive_prob * 100.0, cached = cache.num_cached());
        println!("  Cache+Dense:      {ad_tok:>10} ({:.2}%)  {ad_ms:>6.0}ms",
            ad_prob * 100.0);

        let speedup = dense_ms / adaptive_ms;
        let saved = dense_ms - adaptive_ms;
        println!("  Speedup (cache+walk vs dense): {speedup:.2}x ({saved:.0}ms saved)");
        println!();
    }

    println!("=== Done ===");
    Ok(())
}
