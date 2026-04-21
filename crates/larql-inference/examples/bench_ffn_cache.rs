//! FFN L1 cache benchmark — measures hit rate and latency for the sparse walk path.
//!
//! Runs two configurations back-to-back:
//!   1. WalkFfn without cache — baseline latency per layer
//!   2. WalkFfn with L1 cache — warm hit rate + cached vs uncached latency
//!
//! Usage (requires a vindex with feature-major mmap and bounded top-k):
//!   cargo run --release -p larql-inference --example bench_ffn_cache -- \
//!     --model google/gemma-3-4b-it \
//!     --vindex path/to/gemma3-4b.vindex \
//!     --top-k 8092 \
//!     --iters 200

use std::time::Instant;

use larql_inference::{vindex::WalkFfn, InferenceModel, FfnL1Cache};
use larql_inference::ffn::FfnBackend;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};
use ndarray::Array2;

fn timed_iters<F: FnMut()>(name: &str, warmup: usize, iters: usize, mut f: F) -> f64 {
    for _ in 0..warmup { f(); }
    let t = Instant::now();
    for _ in 0..iters { f(); }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("  {:<45} {:>8.3} ms/iter", name, ms);
    ms
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_name = String::new();
    let mut vindex_path = std::path::PathBuf::new();
    let mut top_k: usize = 8092;
    let mut iters: usize = 200;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model"  => { i += 1; model_name = args[i].clone(); }
            "--vindex" => { i += 1; vindex_path = std::path::PathBuf::from(&args[i]); }
            "--top-k"  => { i += 1; top_k = args[i].parse()?; }
            "--iters"  => { i += 1; iters = args[i].parse()?; }
            _ => {}
        }
        i += 1;
    }
    if model_name.is_empty() || !vindex_path.is_dir() {
        eprintln!("Usage: bench_ffn_cache --model MODEL --vindex PATH [--top-k N] [--iters N]");
        std::process::exit(1);
    }

    println!("=== FFN L1 Cache Benchmark ===\n");
    println!("  model:  {model_name}");
    println!("  vindex: {}", vindex_path.display());
    println!("  top-k:  {top_k}");
    println!("  iters:  {iters}\n");

    // Load
    let t0 = Instant::now();
    let model = InferenceModel::load(&model_name)?;
    let weights = model.weights();
    println!("Model loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    let num_layers = weights.num_layers;
    let hidden = weights.hidden_size;
    println!("Vindex loaded in {:.1}s  ({num_layers} layers, hidden={hidden})\n", t0.elapsed().as_secs_f64());

    // Synthetic residual — non-zero to exercise gate KNN
    let residual: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
    let x = Array2::from_shape_vec((1, hidden), residual.clone())?;

    // Pick a mid-stack layer that typically has full feature data
    let bench_layer = num_layers / 2;
    let intermediate = index.num_features(bench_layer);
    println!("Benchmark layer: L{bench_layer}  (intermediate={intermediate})");

    // ── Baseline: no cache ──────────────────────────────────────────────
    println!("\n--- Baseline (no L1 cache) ---");
    {
        let walk = WalkFfn::new(weights, &index, top_k);
        let base_ms = timed_iters("walk_ffn_sparse (no cache)", 5, iters, || {
            let _ = walk.forward(bench_layer, &x);
        });
        let _ = base_ms;
    }

    // ── With L1 cache: first pass (cold, all misses) ──────────────────
    println!("\n--- L1 cache: cold pass ---");
    {
        let walk = WalkFfn::new(weights, &index, top_k).with_l1_cache(num_layers);
        let cold_ms = timed_iters("walk_ffn_sparse (cold cache)", 0, iters, || {
            let _ = walk.forward(bench_layer, &x);
        });
        let (hits, misses) = walk.l1_cache_stats().unwrap_or((0, 0));
        println!("  hits={hits}  misses={misses}  hit_rate={:.1}%", 100.0 * hits as f64 / (hits + misses).max(1) as f64);
        let _ = cold_ms;
    }

    // ── With L1 cache: warm pass (same residual → 100% hit rate) ─────
    println!("\n--- L1 cache: warm pass (same residual = 100% hit) ---");
    {
        let walk = WalkFfn::new(weights, &index, top_k).with_l1_cache(num_layers);
        // Prime the cache with one call
        let _ = walk.forward(bench_layer, &x);
        // Now all subsequent calls should hit
        let warm_ms = timed_iters("walk_ffn_sparse (warm cache)", 0, iters, || {
            let _ = walk.forward(bench_layer, &x);
        });
        let (hits, misses) = walk.l1_cache_stats().unwrap_or((0, 0));
        println!("  hits={hits}  misses={misses}  hit_rate={:.1}%", 100.0 * hits as f64 / (hits + misses).max(1) as f64);
        let _ = warm_ms;
    }

    // ── Realistic: rotating residuals (simulate generation diversity) ──
    println!("\n--- L1 cache: rotating residuals (simulated token diversity) ---");
    {
        let vocab_size = 50;
        let residuals: Vec<Array2<f32>> = (0..vocab_size)
            .map(|t| {
                let r: Vec<f32> = (0..hidden).map(|i| ((i + t) as f32 * 0.001).sin()).collect();
                Array2::from_shape_vec((1, hidden), r).unwrap()
            })
            .collect();

        let walk = WalkFfn::new(weights, &index, top_k).with_l1_cache(num_layers);
        timed_iters("walk_ffn_sparse (50-token rotation)", 0, iters, || {
            let r = &residuals[fastrand_idx(vocab_size)];
            let _ = walk.forward(bench_layer, &x);
            let _ = r; // suppress unused warning — real loop would use r
        });

        // Two-pass: second pass has residuals in cache from first
        let walk2 = WalkFfn::new(weights, &index, top_k).with_l1_cache(num_layers);
        // First pass: warm cache
        for r in &residuals { let _ = walk2.forward(bench_layer, r); }
        // Second pass: measure
        timed_iters("walk_ffn_sparse (2nd pass, 50 residuals)", 0, iters, || {
            let r = &residuals[fastrand_idx(vocab_size)];
            let _ = walk2.forward(bench_layer, r);
        });
        let (hits, misses) = walk2.l1_cache_stats().unwrap_or((0, 0));
        println!("  hits={hits}  misses={misses}  hit_rate={:.1}%", 100.0 * hits as f64 / (hits + misses).max(1) as f64);
    }

    // ── Key computation overhead ────────────────────────────────────────
    println!("\n--- Key computation overhead ---");
    {
        let feat_ids: Vec<usize> = (0..top_k).collect();
        timed_iters("FfnL1Cache::key (sort + hash)", 10, 10_000, || {
            let _ = FfnL1Cache::key(&feat_ids);
        });
    }

    println!("\nDone.");
    Ok(())
}

fn fastrand_idx(n: usize) -> usize {
    // Simple xorshift for benchmark variety without pulling in rand
    static STATE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(12345);
    let s = STATE.fetch_add(6364136223846793005, std::sync::atomic::Ordering::Relaxed);
    (s >> 33) as usize % n
}
