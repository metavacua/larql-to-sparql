//! Speculation error experiment: can we walk FFN layers in parallel?
//!
//! For each layer N, measures the error between:
//!   true path:  run_ffn(post_attn_residual_N,  layer=N)  — actual residual
//!   spec path:  run_ffn(initial_embedding,      layer=N)  — speculative residual
//!
//! Metrics:
//!   cosine_distance   between the two FFN deltas
//!   feature_overlap   Jaccard of top-K active FFN features (K=200)
//!   top1_match        logit-lens argmax match at each layer
//!
//! Usage:
//!   cargo run --release -p larql-inference --example speculation_error -- \
//!       --model google/gemma-3-4b-it \
//!       [--threshold 0.05] [--prompt-sets factual,arithmetic,code]

use ndarray::Array2;
use larql_inference::{
    forward::{run_ffn, apply_norm, dot_proj, capture_spec_residuals},
    ffn::WeightFfn,
    InferenceModel,
};

// ── Prompts ─────────────────────────────────────────────────────────────

const PROMPTS_FACTUAL: &[&str] = &[
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Japan is",
    "The capital of Australia is",
    "The capital of Brazil is",
    "Albert Einstein was born in",
    "Marie Curie was born in",
    "Python was created by",
    "The Eiffel Tower is located in",
    "The Great Wall is located in",
];

const PROMPTS_ARITHMETIC: &[&str] = &[
    "2 + 2 =",
    "7 × 8 =",
    "15 - 6 =",
    "100 / 4 =",
];

const PROMPTS_CODE: &[&str] = &[
    "def fibonacci(n):",
    "import numpy as",
    "for i in range(",
];

const TOP_K_FEATURES: usize = 200;

// ── Args ─────────────────────────────────────────────────────────────────

struct Args {
    model: String,
    threshold: f32,
    prompt_sets: Vec<String>,
}

fn parse_args() -> Args {
    let raw: Vec<String> = std::env::args().collect();
    let mut model = String::new();
    let mut threshold = 0.05_f32;
    let mut prompt_sets = vec!["factual".to_string(), "arithmetic".to_string(), "code".to_string()];

    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--model"       => { i += 1; model = raw[i].clone(); }
            "--threshold"   => { i += 1; threshold = raw[i].parse().unwrap_or(0.05); }
            "--prompt-sets" => { i += 1; prompt_sets = raw[i].split(',').map(|s| s.to_string()).collect(); }
            _ => {}
        }
        i += 1;
    }

    if model.is_empty() {
        eprintln!("Usage: speculation_error --model MODEL [--threshold 0.05] [--prompt-sets factual,arithmetic,code]");
        std::process::exit(1);
    }

    Args { model, threshold, prompt_sets }
}

// ── Math helpers ─────────────────────────────────────────────────────────

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 { 1.0 } else { 1.0 - dot / denom }
}

fn top_k_indices(vals: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = vals.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    indexed.into_iter().map(|(i, _)| i).collect()
}

fn jaccard(a: &[usize], b: &[usize]) -> f32 {
    use std::collections::HashSet;
    let sa: HashSet<usize> = a.iter().copied().collect();
    let sb: HashSet<usize> = b.iter().copied().collect();
    let intersect = sa.intersection(&sb).count();
    let union_ = sa.union(&sb).count();
    if union_ == 0 { 1.0 } else { intersect as f32 / union_ as f32 }
}

fn lm_head_top1(weights: &larql_inference::ModelWeights, h_last: &[f32]) -> usize {
    let hidden = h_last.len();
    let norm_offset = weights.arch.norm_weight_offset();
    let h_2d = Array2::from_shape_vec((1, hidden), h_last.to_vec()).unwrap();
    let h_normed = apply_norm(weights, &h_2d, weights.arch.final_norm_key(), norm_offset);
    let logits = dot_proj(&h_normed, &weights.lm_head);
    let row = logits.row(0);
    row.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ── Per-layer stats accumulator ───────────────────────────────────────────

#[derive(Default)]
struct LayerStats {
    cosine_errs: Vec<f32>,
    feature_overlaps: Vec<f32>,
    top1_matches: Vec<f32>,
}

// ── Main ─────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();

    // Build prompt list
    let mut prompts: Vec<String> = Vec::new();
    for set in &args.prompt_sets {
        match set.as_str() {
            "factual"    => prompts.extend(PROMPTS_FACTUAL.iter().map(|s| s.to_string())),
            "arithmetic" => prompts.extend(PROMPTS_ARITHMETIC.iter().map(|s| s.to_string())),
            "code"       => prompts.extend(PROMPTS_CODE.iter().map(|s| s.to_string())),
            other => eprintln!("unknown prompt set: {other}"),
        }
    }

    println!("=== Speculation Error Experiment ===\n");
    println!("  Model:      {}", args.model);
    println!("  Prompts:    {}", prompts.len());
    println!("  Threshold:  cosine_distance < {}", args.threshold);
    println!("  Top-K feat: {TOP_K_FEATURES}\n");

    eprintln!("Loading model...");
    let t0 = std::time::Instant::now();
    let inference_model = InferenceModel::load(&args.model)?;
    let weights = inference_model.weights();
    let tokenizer = inference_model.tokenizer();
    let num_layers = weights.num_layers;
    eprintln!("  loaded in {:.1}s ({num_layers} layers, hidden={})\n", t0.elapsed().as_secs_f64(), weights.hidden_size);

    let ffn = WeightFfn { weights };

    // Per-layer accumulators
    let mut stats: Vec<LayerStats> = (0..num_layers).map(|_| LayerStats::default()).collect();

    for (pi, prompt) in prompts.iter().enumerate() {
        eprint!("  [{}/{}] {:?}... ", pi + 1, prompts.len(), &prompt[..prompt.len().min(40)]);
        let t = std::time::Instant::now();

        let enc = tokenizer.encode(prompt.as_str(), true).map_err(|e| format!("tokenize: {e}"))?;
        let token_ids: Vec<u32> = enc.get_ids().to_vec();
        let seq_len = token_ids.len();

        // Single-pass: capture post-attn and post-layer residuals at every layer
        let capture = capture_spec_residuals(weights, &token_ids);

        // Speculative residual: last token of initial embedding
        let spec_h0: Vec<f32> = capture.h_0.row(seq_len - 1).to_vec();
        let spec_2d = Array2::from_shape_vec((1, weights.hidden_size), spec_h0.clone())?;

        // Precompute spec FFN (delta + activation) for all layers in one pass
        let mut spec_deltas: Vec<Vec<f32>> = Vec::with_capacity(num_layers);
        let mut spec_acts: Vec<Option<Array2<f32>>> = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let (spec_out, spec_act) = run_ffn(weights, &spec_2d, layer, &ffn, true);
            let delta: Vec<f32> = spec_out.row(0).iter().zip(spec_h0.iter()).map(|(o, i)| o - i).collect();
            spec_deltas.push(delta);
            spec_acts.push(spec_act);
        }

        // Per-layer metrics
        let mut spec_accum: Vec<f32> = spec_h0.clone();

        for layer in 0..num_layers {
            // True FFN delta using actual post-attn residual
            let true_h: &[f32] = &capture.post_attn_last[layer];
            let true_2d = Array2::from_shape_vec((1, weights.hidden_size), true_h.to_vec())?;
            let (true_out, true_act_opt) = run_ffn(weights, &true_2d, layer, &ffn, true);
            let true_delta: Vec<f32> = true_out.row(0).iter().zip(true_h.iter()).map(|(o, i)| o - i).collect();

            let spec_delta = &spec_deltas[layer];
            let spec_act_opt = spec_acts[layer].as_ref();

            // Cosine distance between FFN deltas
            let cos_err = cosine_distance(&true_delta, spec_delta);

            // Feature overlap: Jaccard of top-K active FFN features by activation magnitude
            let overlap = match (true_act_opt, spec_act_opt) {
                (Some(ta), Some(sa)) => {
                    let true_features = top_k_indices(&ta.row(0).to_vec(), TOP_K_FEATURES);
                    let spec_features = top_k_indices(&sa.row(0).to_vec(), TOP_K_FEATURES);
                    jaccard(&true_features, &spec_features)
                }
                _ => 0.0,
            };

            // Top-1 match via logit lens
            // Accumulate spec residual through layer N
            for (acc, d) in spec_accum.iter_mut().zip(spec_delta.iter()) {
                *acc += d;
            }
            let true_top1 = lm_head_top1(weights, &capture.post_layer_last[layer]);
            let spec_top1 = lm_head_top1(weights, &spec_accum);
            let top1_match = if true_top1 == spec_top1 { 1.0_f32 } else { 0.0 };

            stats[layer].cosine_errs.push(cos_err);
            stats[layer].feature_overlaps.push(overlap);
            stats[layer].top1_matches.push(top1_match);
        }

        eprintln!("{:.1}s", t.elapsed().as_secs_f64());
    }

    // ── Classification ─────────────────────────────────────────────────

    let threshold = args.threshold;
    let mut parallelisable: Vec<usize> = Vec::new();
    let mut serial: Vec<usize> = Vec::new();

    // Print header
    println!();
    println!("Per-layer cosine distance (true vs speculative delta):");
    println!("  {:>5}  {:>9}  {:>6}  {:>6}  {:>16}  {:>11}  {:>10}",
             "Layer", "Mean err", "Min", "Max", "Feature overlap", "Top-1 match", "Verdict");
    println!("  {}", "─".repeat(75));

    for layer in 0..num_layers {
        let s = &stats[layer];
        if s.cosine_errs.is_empty() { continue; }

        let mean_err  = s.cosine_errs.iter().sum::<f32>() / s.cosine_errs.len() as f32;
        let min_err   = s.cosine_errs.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_err   = s.cosine_errs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean_ov   = s.feature_overlaps.iter().sum::<f32>() / s.feature_overlaps.len() as f32;
        let mean_top1 = s.top1_matches.iter().sum::<f32>() / s.top1_matches.len() as f32;

        let verdict = if mean_err < threshold {
            parallelisable.push(layer);
            "PARALLEL"
        } else {
            serial.push(layer);
            "serial"
        };

        println!("  {:>5}  {:>9.4}  {:>6.4}  {:>6.4}  {:>16.3}  {:>11.3}  {:>10}",
                 layer, mean_err, min_err, max_err, mean_ov, mean_top1, verdict);
    }

    // ── Band structure ─────────────────────────────────────────────────

    println!();
    println!("Band structure (threshold = {threshold}):");

    struct Band { kind: &'static str, start: usize, end: usize }
    let mut bands: Vec<Band> = Vec::new();

    for layer in 0..num_layers {
        let kind = if parallelisable.contains(&layer) { "PARALLEL" } else { "serial" };
        match bands.last_mut() {
            Some(b) if b.kind == kind => { b.end = layer; }
            _ => bands.push(Band { kind, start: layer, end: layer }),
        }
    }

    let parallel_ms_per_band = 55.0_f32;
    let serial_ms_per_layer  = 8.0_f32;
    let mut estimated_ms = 0.0_f32;

    for b in &bands {
        let n = b.end - b.start + 1;
        let ms = if b.kind == "PARALLEL" {
            estimated_ms += parallel_ms_per_band;
            parallel_ms_per_band
        } else {
            let m = n as f32 * serial_ms_per_layer;
            estimated_ms += m;
            m
        };
        println!("  L{:02}–L{:02}  ({:2} layers)  {}  ~{:.0}ms",
                 b.start, b.end, n, b.kind, ms);
    }

    let serial_baseline = num_layers as f32 * serial_ms_per_layer;
    let speedup = serial_baseline / estimated_ms.max(1.0);

    println!();
    println!("  Round trips:      {}", bands.len());
    println!("  Estimated wall:   {estimated_ms:.0}ms");
    println!("  Serial baseline:  {serial_baseline:.0}ms");
    println!("  Speedup:          {speedup:.1}×");
    println!();

    // ── Aggressive threshold ───────────────────────────────────────────

    let aggressive = 0.15_f32;
    let agg_parallel = stats.iter().enumerate()
        .filter(|(_, s)| !s.cosine_errs.is_empty() && {
            let mean = s.cosine_errs.iter().sum::<f32>() / s.cosine_errs.len() as f32;
            mean < aggressive
        })
        .count();
    let agg_serial = num_layers - agg_parallel;
    println!("  Aggressive threshold ({aggressive}): {agg_parallel}/{num_layers} layers PARALLEL, {agg_serial} serial");

    Ok(())
}
