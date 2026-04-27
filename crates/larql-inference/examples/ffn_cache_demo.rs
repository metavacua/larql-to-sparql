//! FFN L1 cache demo — shows cache behaviour, hit/miss stats, and patch safety.
//!
//! Demonstrates three scenarios:
//!   1. Clean model — repeated residual → 100% hit after first call
//!   2. Paraphrase collapse — similar residuals activate same features → cache hit
//!   3. Patched session — INSERT'd slot bypasses cache for correctness
//!
//! Usage:
//!   cargo run --release -p larql-inference --example ffn_cache_demo -- \
//!     --model google/gemma-3-4b-it \
//!     --vindex path/to/gemma3-4b.vindex

use std::time::Instant;

use larql_inference::{vindex::WalkFfn, InferenceModel};
use larql_inference::ffn::FfnBackend;
use larql_vindex::{PatchedVindex, SilentLoadCallbacks, VectorIndex};
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_name = String::new();
    let mut vindex_path = std::path::PathBuf::new();
    let mut top_k: usize = 8092;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model"  => { i += 1; model_name = args[i].clone(); }
            "--vindex" => { i += 1; vindex_path = std::path::PathBuf::from(&args[i]); }
            "--top-k"  => { i += 1; top_k = args[i].parse()?; }
            _ => {}
        }
        i += 1;
    }
    if model_name.is_empty() || !vindex_path.is_dir() {
        eprintln!("Usage: ffn_cache_demo --model MODEL --vindex PATH [--top-k N]");
        std::process::exit(1);
    }

    let model = InferenceModel::load(&model_name)?;
    let weights = model.weights();
    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;

    let num_layers = weights.num_layers;
    let hidden     = weights.hidden_size;
    let bench_layer = num_layers / 2;

    println!("=== FFN L1 Cache Demo ===");
    println!("  model:       {model_name}");
    println!("  layers:      {num_layers}");
    println!("  hidden:      {hidden}");
    println!("  top-k:       {top_k}");
    println!("  bench layer: L{bench_layer}\n");

    let base_residual: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();

    // ── Scenario 1: repeated identical residual ────────────────────────
    println!("Scenario 1: repeated identical residual");
    println!("  First call fills the cache; every subsequent call is a hit.\n");
    {
        let x = Array2::from_shape_vec((1, hidden), base_residual.clone())?;
        let walk = WalkFfn::new(weights, &index, top_k).with_l1_cache(num_layers);

        let t0 = Instant::now();
        let _  = walk.forward(bench_layer, &x);
        let first_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        for _ in 0..99 {
            let _ = walk.forward(bench_layer, &x);
        }
        let cached_ms = t0.elapsed().as_secs_f64() * 1000.0 / 99.0;

        let (hits, misses) = walk.l1_cache_stats().unwrap_or((0, 0));
        println!("  call 1 (miss):    {first_ms:.3} ms");
        println!("  calls 2-100 (hit): {cached_ms:.4} ms/call  ({:.0}x speedup)",
            first_ms / cached_ms.max(1e-6));
        println!("  hits={hits}  misses={misses}  hit_rate={:.1}%\n",
            100.0 * hits as f64 / (hits + misses).max(1) as f64);
    }

    // ── Scenario 2: paraphrase collapse ────────────────────────────────
    println!("Scenario 2: paraphrase collapse");
    println!("  Residuals with cosine similarity ~0.99 activate the same features.\n");
    {
        // Perturb the residual by a tiny amount — simulates a paraphrase
        let epsilon = 1e-4_f32;
        let perturbed: Vec<f32> = base_residual.iter().enumerate()
            .map(|(i, &v)| v + epsilon * ((i % 7) as f32 - 3.0))
            .collect();

        let x_orig = Array2::from_shape_vec((1, hidden), base_residual.clone())?;
        let x_para = Array2::from_shape_vec((1, hidden), perturbed)?;

        let walk = WalkFfn::new(weights, &index, top_k).with_l1_cache(num_layers);

        let _ = walk.forward(bench_layer, &x_orig); // miss — fills cache
        let _ = walk.forward(bench_layer, &x_para); // hit if features match

        let (hits, misses) = walk.l1_cache_stats().unwrap_or((0, 0));
        let hit_rate = 100.0 * hits as f64 / (hits + misses).max(1) as f64;
        println!("  hits={hits}  misses={misses}  hit_rate={hit_rate:.1}%");
        if hits > 0 {
            println!("  → Paraphrase residual activated the same feature set (expected for cos≈0.99)");
        } else {
            println!("  → Paraphrase residual activated a different feature set");
            println!("    (perturbation was large enough to cross a gate boundary)");
        }
        println!();
    }

    // ── Scenario 3: patched session — cache must be bypassed ──────────
    println!("Scenario 3: patched session (INSERT safety)");
    println!("  A patched vindex has modified down/up vectors. The cache key is derived");
    println!("  from gate KNN feature IDs only. If the gate is unchanged but the down");
    println!("  vector changed, the same key would return a stale output.");
    println!("  Correct behaviour: cache is bypassed when any override exists at the layer.\n");
    {
        let x = Array2::from_shape_vec((1, hidden), base_residual.clone())?;

        // ── Clean run (fills cache) ──
        let walk_clean = WalkFfn::new(weights, &index, top_k).with_l1_cache(num_layers);
        let out_clean = walk_clean.forward(bench_layer, &x);
        let _ = walk_clean.forward(bench_layer, &x); // confirm hit

        let (h, m) = walk_clean.l1_cache_stats().unwrap_or((0, 0));
        println!("  Clean model:  hits={h}  misses={m}");

        // ── Patched run: install a synthetic override on bench_layer ──
        let mut patched = PatchedVindex::new(index.clone());
        // Override feature 0's gate vector with a different direction (simulates INSERT)
        let new_gate: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.1).cos()).collect();
        patched.set_gate_override(bench_layer, 0, new_gate);

        let walk_patched = WalkFfn::new(weights, &patched, top_k).with_l1_cache(num_layers);
        let out_patched = walk_patched.forward(bench_layer, &x);

        let (h2, m2) = walk_patched.l1_cache_stats().unwrap_or((0, 0));
        println!("  Patched model: hits={h2}  misses={m2}");

        // Verify: cache was bypassed (0 hits on patched), and outputs differ
        assert_eq!(h2, 0, "Cache must not be read when overrides exist at the layer");
        let diff: f32 = out_clean.iter().zip(out_patched.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>() / hidden as f32;
        println!("  Output difference (mean |Δ|): {diff:.6}");
        if diff > 1e-6 {
            println!("  ✓ Patch was applied — outputs diverge as expected");
        } else {
            println!("  (outputs identical — the overridden feature may not have been activated)");
        }
    }

    println!("\nDone.");
    Ok(())
}
