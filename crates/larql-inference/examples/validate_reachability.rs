//! Validate OV-gate reachability: do the actually-firing features
//! fall within the OV-gate reachable set?
//!
//! Runs a dense forward pass, captures per-layer feature activations,
//! then checks overlap with the precomputed OV-gate edge data.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example validate_reachability -- \
//!     --edges output/circuits/ov_gate_edges.jsonl

use std::collections::{HashMap, HashSet};
use std::io::BufRead;

use larql_inference::{InferenceModel, WeightFfn};
use larql_inference::forward::trace_forward_full;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut edges_path = String::from("output/circuits/ov_gate_edges.jsonl");
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--edges" { i += 1; edges_path = args[i].clone(); }
        i += 1;
    }

    // Load model
    let model = InferenceModel::load("google/gemma-3-4b-it")?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let num_layers = weights.num_layers;

    // Load OV-gate edges
    let mut reachable: HashMap<usize, HashSet<usize>> = HashMap::new();
    let file = std::io::BufReader::new(std::fs::File::open(&edges_path)?);
    for line in file.lines() {
        let line = line?;
        let v: serde_json::Value = serde_json::from_str(&line)?;
        if v.get("_header").is_some() { continue; }
        let layer = v["layer"].as_u64().unwrap() as usize;
        let feature = v["feature"].as_u64().unwrap() as usize;
        reachable.entry(layer).or_default().insert(feature);
    }

    println!("=== OV-Gate Reachability Validation ===\n");
    println!("Edges: {edges_path}");
    println!("Reachable features per layer: {:.0} avg\n",
        reachable.values().map(|s| s.len()).sum::<usize>() as f64 / num_layers as f64);

    // Test prompts
    let prompts = [
        "The capital of France is",
        "The language spoken in Japan is",
        "The currency of the United Kingdom is",
        "Albert Einstein was born in",
        "Python is a programming",
    ];

    let all_layers: Vec<usize> = (0..num_layers).collect();
    let dense_ffn = WeightFfn { weights };
    let activation_top_k = 500; // capture top-500 features per layer

    for prompt in &prompts {
        let encoding = tokenizer.encode(*prompt, true).map_err(|e| format!("{e}"))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        let trace = trace_forward_full(
            weights, &token_ids, &all_layers,
            true, activation_top_k, false, &dense_ffn,
        );

        println!("Prompt: \"{prompt}\"");

        let mut total_firing = 0usize;
        let mut total_covered = 0usize;
        let mut total_reachable = 0usize;

        for &(layer, ref top_feats) in &trace.activations {
            let firing: HashSet<usize> = top_feats.iter()
                .filter(|(_, act)| act.abs() > 1.0) // significant activation
                .map(|(f, _)| *f)
                .collect();

            let reach = reachable.get(&layer).cloned().unwrap_or_default();
            let covered: HashSet<usize> = firing.intersection(&reach).copied().collect();

            total_firing += firing.len();
            total_covered += covered.len();
            total_reachable += reach.len();

            if layer % 8 == 0 || layer == num_layers - 1 {
                let pct = if firing.is_empty() { 0.0 } else { covered.len() as f64 / firing.len() as f64 * 100.0 };
                println!("  L{layer:2}: {}/{} firing covered ({pct:.0}%), reach={}, firing={}",
                    covered.len(), firing.len(), reach.len(), firing.len());
            }
        }

        let overall_pct = if total_firing == 0 { 0.0 }
            else { total_covered as f64 / total_firing as f64 * 100.0 };
        println!("  TOTAL: {total_covered}/{total_firing} covered ({overall_pct:.1}%), reachable={total_reachable}\n");
    }

    Ok(())
}
