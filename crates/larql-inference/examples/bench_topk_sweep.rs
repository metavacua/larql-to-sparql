//! Top-K sweep — measures accuracy and speed at different gate KNN K values.
//! Finds the minimum K that maintains correct predictions.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example bench_topk_sweep -- \
//!     --model google/gemma-3-4b-it \
//!     --vindex path/to/gemma3-4b.vindex

use std::time::Instant;

use larql_inference::{
    predict, predict_with_ffn, InferenceModel,
    vindex::WalkFfn,
};
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_name = String::new();
    let mut vindex_path = std::path::PathBuf::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { i += 1; model_name = args[i].clone(); }
            "--vindex" => { i += 1; vindex_path = std::path::PathBuf::from(&args[i]); }
            _ => {}
        }
        i += 1;
    }
    if model_name.is_empty() || !vindex_path.is_dir() {
        eprintln!("Usage: bench_topk_sweep --model MODEL --vindex PATH");
        std::process::exit(1);
    }

    println!("=== Top-K Sweep ===\n");

    let model = InferenceModel::load(&model_name)?;
    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;

    let weights = model.weights();
    let tokenizer = model.tokenizer();

    let prompts: Vec<(&str, &str)> = vec![
        ("The capital of France is", "Paris"),
        ("The capital of Germany is", "Berlin"),
        ("The capital of Japan is", "Tokyo"),
        ("The capital of Italy is", "Rome"),
        ("The largest planet in our solar system is", "Jupiter"),
    ];

    // Ground truth
    println!("Ground truth (dense):");
    let mut ground: Vec<(String, f64)> = Vec::new();
    for (prompt, _) in &prompts {
        let enc = tokenizer.encode(*prompt, true).map_err(|e| format!("{e}"))?;
        let ids: Vec<u32> = enc.get_ids().to_vec();
        let r = predict(weights, tokenizer, &ids, 5);
        let (tok, prob) = r.predictions.first().map(|(t, p)| (t.clone(), *p)).unwrap_or_default();
        println!("  {prompt} -> {tok} ({:.1}%)", prob * 100.0);
        ground.push((tok, prob));
    }
    println!();

    // K values to test
    let k_values = vec![50, 100, 200, 500, 1000, 2000, 4000, 8092];

    println!("{:>6}  {:>7}  {:>8}  {:>10}  {}", "K", "correct", "avg_prob", "time/tok", "divergences");
    println!("{:-<70}", "");

    for &top_k in &k_values {
        let walk_ffn = WalkFfn::new(weights, &index, top_k);

        // Warmup
        let enc = tokenizer.encode(prompts[0].0, true).map_err(|e| format!("{e}"))?;
        let ids: Vec<u32> = enc.get_ids().to_vec();
        let _ = predict_with_ffn(weights, tokenizer, &ids, 5, &walk_ffn);

        let mut correct = 0;
        let mut total_prob = 0.0;
        let mut divergences = Vec::new();
        let t0 = Instant::now();

        for (i, (prompt, expected)) in prompts.iter().enumerate() {
            let enc = tokenizer.encode(*prompt, true).map_err(|e| format!("{e}"))?;
            let ids: Vec<u32> = enc.get_ids().to_vec();
            let r = predict_with_ffn(weights, tokenizer, &ids, 5, &walk_ffn);
            let (tok, prob) = r.predictions.first().map(|(t, p)| (t.clone(), *p)).unwrap_or_default();

            if tok.to_lowercase().contains(&expected.to_lowercase()) {
                correct += 1;
            }
            total_prob += prob;

            if tok != ground[i].0 {
                divergences.push(format!("{}->{}({:.0}%)", ground[i].0, tok, prob * 100.0));
            }
        }

        let elapsed = t0.elapsed();
        let per_tok = elapsed.as_secs_f64() * 1000.0 / prompts.len() as f64;
        let avg_prob = total_prob / prompts.len() as f64 * 100.0;
        let div_str = if divergences.is_empty() {
            "none".to_string()
        } else {
            divergences.join(", ")
        };

        println!(
            "{top_k:>6}  {correct:>3}/{:<3}  {avg_prob:>7.2}%  {per_tok:>8.0}ms  {div_str}",
            prompts.len()
        );
    }

    println!("\n=== Done ===");
    Ok(())
}
