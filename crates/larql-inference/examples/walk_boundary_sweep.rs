//! Walk boundary sweep — tests vindex FFN walk at every layer boundary.
//!
//! For each boundary B:
//!   Layers 0..B:   dense attention + dense FFN (WeightFfn)
//!   Layers B..33:  dense attention + vindex FFN (WalkFfn)
//!
//! Reports top-1 prediction and probability for each boundary, comparing
//! against the ground truth (all-dense forward pass).
//!
//! The vindex has all 34 layers (1,307,232 vectors). This sweep finds
//! how far down the walk can go while maintaining accuracy.
//!
//! Usage:
//!   cargo run --release -p larql-inference --example walk_boundary_sweep -- \
//!     --model google/gemma-3-4b-it \
//!     --vindex path/to/gemma3-4b.vindex
//!
//! Optional:
//!   --top-k 8092     Gate KNN top-K (default: 8092)
//!   --prompts        Comma-separated prompts (default: built-in entity set)

use std::path::PathBuf;
use std::time::Instant;

use larql_inference::{
    predict, predict_with_ffn, predict_with_router,
    InferenceModel, LayerFfnRouter, WeightFfn, PredictResult,
    vindex::WalkFfn,
};
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

/// Default test prompts — entities with known ground truth answers.
/// Keep small for fast sweep; add --prompts for larger sets.
const DEFAULT_PROMPTS: &[(&str, &str)] = &[
    ("The capital of France is", "Paris"),
    ("The capital of Germany is", "Berlin"),
    ("The capital of Japan is", "Tokyo"),
    ("The capital of Italy is", "Rome"),
    ("The largest planet in our solar system is", "Jupiter"),
];

#[allow(clippy::type_complexity)]
fn parse_args() -> (String, PathBuf, usize, Option<Vec<(String, String)>>) {
    let args: Vec<String> = std::env::args().collect();
    let mut model = String::new();
    let mut vindex = PathBuf::new();
    let mut top_k = 8092;
    let mut prompts: Option<Vec<(String, String)>> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { i += 1; model = args[i].clone(); }
            "--vindex" => { i += 1; vindex = PathBuf::from(&args[i]); }
            "--top-k" => {
                i += 1;
                top_k = if args[i] == "full" || args[i] == "unlimited" {
                    usize::MAX
                } else {
                    args[i].parse().unwrap()
                };
            }
            "--prompts" => {
                i += 1;
                prompts = Some(
                    args[i].split(';')
                        .map(|p| {
                            let parts: Vec<&str> = p.splitn(2, '=').collect();
                            if parts.len() == 2 {
                                (parts[0].trim().to_string(), parts[1].trim().to_string())
                            } else {
                                (p.trim().to_string(), String::new())
                            }
                        })
                        .collect(),
                );
            }
            _ => {}
        }
        i += 1;
    }

    if model.is_empty() || !vindex.is_dir() {
        eprintln!("Usage: walk_boundary_sweep --model MODEL --vindex PATH [--top-k N]");
        eprintln!("  --model   HuggingFace model ID or local path");
        eprintln!("  --vindex  Path to .vindex directory");
        eprintln!("  --top-k   Gate KNN top-K (default: 8092)");
        std::process::exit(1);
    }

    (model, vindex, top_k, prompts)
}

/// Check if the ground truth is in the top-1 prediction.
fn is_correct(result: &PredictResult, expected: &str) -> bool {
    if expected.is_empty() { return true; }
    result.predictions.first()
        .map(|(tok, _)| tok.to_lowercase().contains(&expected.to_lowercase()))
        .unwrap_or(false)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (model_name, vindex_path, top_k, custom_prompts) = parse_args();

    println!("=== Walk Boundary Sweep ===\n");

    // ── Load model ──
    println!("Loading model: {model_name}");
    let t0 = Instant::now();
    let model = InferenceModel::load(&model_name)?;
    println!("  Model loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let num_layers = weights.num_layers;
    println!("  {} layers, hidden={}", num_layers, weights.hidden_size);

    // ── Load vindex ──
    println!("Loading vindex: {}", vindex_path.display());
    let t0 = Instant::now();
    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    println!(
        "  {} layers, {} vectors loaded in {:.1}s",
        index.num_layers,
        index.total_gate_vectors(),
        t0.elapsed().as_secs_f64()
    );
    println!();

    // ── Test prompts ──
    let prompts: Vec<(String, String)> = match custom_prompts {
        Some(p) => p,
        None => DEFAULT_PROMPTS
            .iter()
            .map(|(p, e)| (p.to_string(), e.to_string()))
            .collect(),
    };

    // ── Ground truth: all-dense forward pass ──
    println!("--- Ground Truth (all-dense) ---\n");
    let mut ground_truth: Vec<(String, f64)> = Vec::new();
    for (prompt, expected) in &prompts {
        let encoding = tokenizer.encode(prompt.as_str(), true)
            .map_err(|e| format!("tokenize: {e}"))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let result = predict(weights, tokenizer, &token_ids, 5);
        let (top1, prob) = result.predictions.first()
            .map(|(t, p)| (t.clone(), *p))
            .unwrap_or_default();
        let correct = is_correct(&result, expected);
        let mark = if correct { "+" } else { "-" };
        println!("  [{mark}] \"{prompt}\" -> {top1} ({:.2}%)", prob * 100.0);
        ground_truth.push((top1, prob));
    }
    println!();

    // ── Sweep boundaries ──
    let boundaries: Vec<usize> = {
        let mut b = vec![0, 4, 8, 12, 16, 20, 24, 28];
        b.push(num_layers); // all-dense baseline
        b.retain(|&v| v <= num_layers);
        b.sort_unstable();
        b.dedup();
        b
    };

    println!("--- Boundary Sweep (dense 0..B, walk B..{num_layers}) ---");
    println!("  {} boundaries x {} prompts = {} forward passes\n",
        boundaries.len(), prompts.len(), boundaries.len() * prompts.len());
    println!(
        "  {:>4}  {:>6}  {:>8}  {:>8}  {:>6}  details",
        "B", "walk%", "correct", "top1_avg", "time"
    );
    println!("  {:-<74}", "");

    for &boundary in &boundaries {
        let walk_pct = (num_layers - boundary) as f64 / num_layers as f64 * 100.0;

        let weight_ffn = WeightFfn { weights };
        let walk_ffn = WalkFfn::new(weights, &index, top_k);

        // Build per-layer backend routing
        let mut backends: Vec<&dyn larql_inference::FfnBackend> = vec![&weight_ffn; num_layers];
        for backend in backends.iter_mut().take(num_layers).skip(boundary) {
            *backend = &walk_ffn;
        }
        let router = LayerFfnRouter::per_layer(backends);

        let mut correct_count = 0;
        let mut total_prob = 0.0;
        let mut details = Vec::new();
        let sweep_start = Instant::now();

        for (i, (prompt, expected)) in prompts.iter().enumerate() {
            let encoding = tokenizer.encode(prompt.as_str(), true)
                .map_err(|e| format!("tokenize: {e}"))?;
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();

            let result = if boundary == num_layers {
                // All dense — skip router overhead
                predict(weights, tokenizer, &token_ids, 5)
            } else if boundary == 0 {
                // All walk
                predict_with_ffn(weights, tokenizer, &token_ids, 5, &walk_ffn)
            } else {
                predict_with_router(weights, tokenizer, &token_ids, 5, &router)
            };

            let (top1, prob) = result.predictions.first()
                .map(|(t, p)| (t.clone(), *p))
                .unwrap_or_default();

            let matches_ground = top1 == ground_truth[i].0;
            let correct = is_correct(&result, expected);
            if correct { correct_count += 1; }
            total_prob += prob;

            // Track divergence from ground truth
            if !matches_ground {
                details.push(format!("{}->{}({:.0}%)",
                    ground_truth[i].0, top1, prob * 100.0));
            }
        }

        let elapsed = sweep_start.elapsed();
        let avg_prob = total_prob / prompts.len() as f64 * 100.0;
        let detail_str = if details.is_empty() {
            "all match ground truth".to_string()
        } else {
            details.join(", ")
        };

        println!(
            "  L{boundary:<3} {walk_pct:>5.0}%  {correct_count:>3}/{:<3}  {avg_prob:>7.2}%  {:.1}s  {detail_str}",
            prompts.len(),
            elapsed.as_secs_f64()
        );
    }

    println!();
    println!("  Legend:");
    println!("  B       = boundary layer (dense 0..B, walk B..{num_layers})");
    println!("  walk%   = percentage of layers using vindex FFN");
    println!("  correct = prompts where top-1 matches expected answer");
    println!("  top1_avg = average top-1 probability across all prompts");
    println!("  details = divergences from ground truth");
    println!();

    // ── Summary ──
    println!("--- Summary ---\n");
    println!("  Ground truth: all-dense f32 forward pass");
    println!("  Walk: vindex gate KNN top-{top_k} -> sparse FFN");
    println!("  Attention: BLAS-fused (dense) at all layers for all boundaries");
    println!("  {} test prompts, {} layers", prompts.len(), num_layers);
    println!();
    println!("  If walk holds to L0: FFN quantization is unnecessary.");
    println!("  Only attention weights (Q/K/V/O) and embed/logits need quantization.");

    println!("\n=== Done ===");
    Ok(())
}
