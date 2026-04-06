//! Real Model Benchmark: Standard KV vs TurboQuant vs Markov RS vs Graph Walk
//!
//! Usage:
//!   cargo run --example real_model_bench --features real-model -- [model-path] [vindex-path]
//!
//! Defaults to google/gemma-3-4b-it if no path given.

#[cfg(feature = "real-model")]
fn main() {
    use kv_cache_benchmark::real_model::*;

    let args: Vec<String> = std::env::args().collect();

    // Load model
    let model_name = args.get(1).map(|s| s.as_str()).unwrap_or("google/gemma-3-4b-it");
    println!("Loading model: {model_name}");
    let model = larql_inference::InferenceModel::load(model_name)
        .expect("Failed to load model");

    // Load vindex
    let vindex_path = args.get(2).map(|s| s.as_str());
    let index = if let Some(path) = vindex_path {
        println!("Loading vindex from: {path}");
        let config = larql_vindex::load_vindex_config(path).expect("Failed to load vindex config");
        larql_vindex::VectorIndex::load(path, &config, &larql_vindex::SilentLoadCallbacks)
            .expect("Failed to load vindex")
    } else {
        println!("No vindex path provided, resolving from HuggingFace...");
        let vindex_dir = larql_vindex::resolve_hf_vindex(model_name)
            .expect("Failed to resolve vindex");
        let config = larql_vindex::load_vindex_config(&vindex_dir).expect("Failed to load vindex config");
        larql_vindex::VectorIndex::load(&vindex_dir, &config, &larql_vindex::SilentLoadCallbacks)
            .expect("Failed to load vindex")
    };

    // Create compute backend
    let backend = larql_inference::default_backend();

    let bench = RealModelBenchmark::new(
        model.weights(),
        model.tokenizer(),
        &index,
        backend.as_ref(),
    );

    // Run default prompts
    let prompts = runner::default_prompts();
    println!("\nRunning {} prompts through 4 strategies...\n", prompts.len());

    for prompt in &prompts {
        let results = runner::run_all_strategies(&bench, prompt, 5, 512);
        println!("{}", runner::format_results(&results));
    }

    // Memory scaling comparison
    println!("\n=== Memory Scaling (Analytical) ===\n");
    let config = kv_cache_benchmark::model_config::ModelConfig::gemma_4b();
    let standard = kv_cache_benchmark::standard_kv::StandardKv;
    let tq4 = kv_cache_benchmark::turboquant::TurboQuant::new(4);
    let markov = kv_cache_benchmark::markov_residual::MarkovResidual::new(512);
    let graph = kv_cache_benchmark::graph_walk::GraphWalk::gemma_4b();

    use kv_cache_benchmark::KvStrategy;
    let strategies: Vec<&dyn KvStrategy> = vec![&standard, &tq4, &markov, &graph];
    println!("{}", kv_cache_benchmark::benchmark::format_comparative_table(&config, &strategies));

    // Write results JSON
    let all_results: Vec<Vec<RealModelResult>> = prompts
        .iter()
        .map(|p| runner::run_all_strategies(&bench, p, 5, 512))
        .collect();

    let json = serde_json::to_string_pretty(&all_results).unwrap();
    std::fs::write("crates/kv-cache-benchmark/results/real_model.json", &json).ok();
    println!("Results written to results/real_model.json");
}

#[cfg(not(feature = "real-model"))]
fn main() {
    eprintln!("This example requires the 'real-model' feature:");
    eprintln!("  cargo run --example real_model_bench --features real-model");
}
