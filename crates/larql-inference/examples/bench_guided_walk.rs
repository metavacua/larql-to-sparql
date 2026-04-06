//! End-to-end benchmark: cache L0-12 + guided walk L13-33.
//!
//! 1. Build template universe (which features fire for this template)
//! 2. Build cache for L0-12 (precomputed residuals)
//! 3. Run adaptive: cached layers + guided walk (universe-restricted FFN)
//! 4. Compare vs dense baseline
//!
//! Usage:
//!   cargo run --release -p larql-inference --example bench_guided_walk -- \
//!     --vindex output/gemma3-4b-v2.vindex

use std::time::Instant;

use larql_inference::{
    predict, predict_with_graph,
    InferenceModel, WeightFfn, DenseLayerGraph,
    CachedLayerGraph, GuidedWalkLayerGraph, TemplateUniverse,
    build_adaptive_graph,
};
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

    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    index.load_down_features(&vindex_path)?;
    index.load_up_features(&vindex_path)?;

    let dense_ffn = WeightFfn { weights };

    println!("=== Guided Walk Benchmark ===\n");

    let templates: Vec<(&str, &str, Vec<&str>, &str)> = vec![
        ("capital", "The capital of {} is", vec![
            "France", "Germany", "Japan", "Brazil", "Egypt",
            "Australia", "Mexico", "India", "Canada", "Italy",
            "Spain", "China", "Russia", "Turkey", "Thailand",
            "Argentina", "Nigeria", "Kenya", "Poland", "Sweden",
        ], "The capital of France is"),
        ("language", "The language spoken in {} is", vec![
            "France", "Germany", "Japan", "Brazil", "Egypt",
            "China", "Russia", "Thailand", "Mexico", "Italy",
            "Spain", "India", "Turkey", "Poland", "Sweden",
            "Greece", "Portugal", "Vietnam", "Indonesia", "Korea",
        ], "The language spoken in Japan is"),
        ("born", "{} was born in", vec![
            "Einstein", "Mozart", "Shakespeare", "Picasso", "Darwin",
            "Beethoven", "Galileo", "Newton", "Tesla", "Curie",
            "Aristotle", "Plato", "Napoleon", "Cleopatra", "Gandhi",
            "Confucius", "Columbus", "Copernicus", "Gutenberg", "Euler",
        ], "Albert Einstein was born in"),
    ];

    let n = 3;

    for (tname, template, entities, test_prompt) in &templates {
        println!("--- {tname}: \"{test_prompt}\" ---\n");

        let encoding = tokenizer.encode(*test_prompt, true).map_err(|e| format!("{e}"))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        // 1. Dense baseline
        let _ = predict(weights, tokenizer, &token_ids, 5);
        let t0 = Instant::now();
        for _ in 0..n { let _ = predict(weights, tokenizer, &token_ids, 5); }
        let dense_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let dense_result = predict(weights, tokenizer, &token_ids, 5);
        let (dense_tok, dense_prob) = dense_result.predictions.first()
            .map(|(t, p)| (t.clone(), *p)).unwrap_or_default();

        // 2. Build template universe
        let t_build = Instant::now();
        let universe = TemplateUniverse::build(
            weights, tokenizer, tname, template, entities,
            &dense_ffn, 1.0,
        );
        let universe_ms = t_build.elapsed().as_secs_f64() * 1000.0;

        // Print universe sizes
        print!("  Universe: ");
        for &layer in &[0, 8, 12, 16, 20, 24, 28, 33] {
            if let Some(feats) = universe.get(layer) {
                print!("L{layer}:{} ", feats.len());
            }
        }
        println!("(built in {universe_ms:.0}ms)");

        // 3. Build cache for L0-12
        let cached_layers: Vec<usize> = (0..=12).collect();
        let cache = CachedLayerGraph::build(weights, &token_ids, &cached_layers, &dense_ffn);

        // 4. Cache+Dense (baseline for cache speedup)
        let dense_graph = DenseLayerGraph { ffn: &dense_ffn, backend: None, capture_activation: false, capture_attention: false };
        let adaptive_dense = build_adaptive_graph(&cache, &dense_graph, num_layers, &(0..=12));
        let _ = predict_with_graph(weights, tokenizer, &token_ids, 5, &adaptive_dense);
        let t0 = Instant::now();
        for _ in 0..n { let _ = predict_with_graph(weights, tokenizer, &token_ids, 5, &adaptive_dense); }
        let cd_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let cd_result = predict_with_graph(weights, tokenizer, &token_ids, 5, &adaptive_dense);
        let (cd_tok, _) = cd_result.predictions.first()
            .map(|(t, p)| (t.clone(), *p)).unwrap_or_default();

        // 5. Cache+GuidedWalk
        let guided = GuidedWalkLayerGraph { weights, universe: &universe, index: &index };
        let adaptive_guided = build_adaptive_graph(&cache, &guided, num_layers, &(0..=12));

        let _ = predict_with_graph(weights, tokenizer, &token_ids, 5, &adaptive_guided);
        let t0 = Instant::now();
        for _ in 0..n { let _ = predict_with_graph(weights, tokenizer, &token_ids, 5, &adaptive_guided); }
        let gw_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let gw_result = predict_with_graph(weights, tokenizer, &token_ids, 5, &adaptive_guided);
        let (gw_tok, gw_prob) = gw_result.predictions.first()
            .map(|(t, p)| (t.clone(), *p)).unwrap_or_default();

        println!("  Dense:          {dense_tok:>12} ({:.2}%)  {dense_ms:>6.0}ms", dense_prob * 100.0);
        println!("  Cache+Dense:    {cd_tok:>12}           {cd_ms:>6.0}ms");
        println!("  Cache+Guided:   {gw_tok:>12} ({:.2}%)  {gw_ms:>6.0}ms", gw_prob * 100.0);

        let speedup = dense_ms / gw_ms;
        let correct = if gw_tok == dense_tok { "MATCH" } else { "DIFF" };
        println!("  → {correct} | {speedup:.2}x vs dense | {:.0}ms saved\n",
            dense_ms - gw_ms);
    }

    println!("=== Done ===");
    Ok(())
}
