use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_inference::{
    trace_forward, CachedFfn, ClusteredFfn, ClusteredGateIndex, EntityRoutedFfn, GateIndex,
    InferenceModel, SparseFfn, WeightFfn, FfnBackend,
};

#[derive(Args)]
pub struct FfnBenchArgs {
    /// Model path or HuggingFace model ID.
    #[arg(short, long)]
    model: String,

    /// Prompt to get a realistic residual from.
    #[arg(short, long, default_value = "The capital of France is")]
    prompt: String,

    /// Layer to benchmark (default: 20).
    #[arg(short, long, default_value = "20")]
    layer: usize,

    /// Comma-separated K values to test.
    #[arg(short = 'k', long, default_value = "64,128,256,512,1024,2048,4096,8192,10240")]
    top_k_values: String,

    /// Number of iterations per K value.
    #[arg(short, long, default_value = "20")]
    iterations: usize,

    /// Path to gate index file for entity-routed benchmark.
    #[arg(long)]
    gate_index: Option<PathBuf>,

    /// Number of K-means clusters for hierarchical index.
    #[arg(long, default_value = "128")]
    clusters: usize,

    /// Number of top clusters to probe at runtime.
    #[arg(long, default_value = "1,2,4,8,16")]
    top_c_values: String,
}

pub fn run(args: FfnBenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let model = InferenceModel::load(&args.model)?;
    let weights = model.weights();

    let encoding = model
        .tokenizer()
        .encode(args.prompt.as_str(), true)
        .map_err(|e| format!("tokenize error: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    // Load gate index if provided
    let gate_index = if let Some(ref path) = args.gate_index {
        eprintln!("Loading gate index: {}", path.display());
        let start = Instant::now();
        let gi = GateIndex::load(path, 10)?;
        eprintln!("  {} layers ({:.1}s)", gi.num_layers(), start.elapsed().as_secs_f64());
        Some(gi)
    } else {
        None
    };

    // Get the real pre-FFN residual at the target layer
    eprintln!("Capturing residual at layer {}...", args.layer);
    let trace = trace_forward(weights, &token_ids, &[args.layer], false, 0);
    let residual_vec = &trace.residuals[0].1;
    let hidden = weights.hidden_size;
    let seq_len = token_ids.len();

    // Build (seq_len, hidden) input from captured residual
    let mut x_data = vec![0.0f32; seq_len * hidden];
    for s in 0..seq_len {
        x_data[s * hidden..(s + 1) * hidden].copy_from_slice(residual_vec);
    }
    let x = larql_inference::ndarray::Array2::from_shape_vec((seq_len, hidden), x_data)?;

    let layer = args.layer;
    let intermediate = weights
        .tensors
        .get(&weights.arch.ffn_gate_key(layer))
        .unwrap()
        .shape()[0];

    eprintln!(
        "Benchmarking FFN layer {} — hidden={}, intermediate={}, seq_len={}, iters={}",
        layer, hidden, intermediate, seq_len, args.iterations
    );

    let k_values: Vec<usize> = args
        .top_k_values
        .split(',')
        .map(|s| s.trim().parse().unwrap())
        .collect();

    // Dense baseline
    let dense_ffn = WeightFfn { weights };
    let _ = dense_ffn.forward(layer, &x);
    let start = Instant::now();
    for _ in 0..args.iterations {
        let _ = dense_ffn.forward(layer, &x);
    }
    let dense_us = start.elapsed().as_micros() as f64 / args.iterations as f64;

    println!(
        "{:>12} {:>10} {:>10} {:>8}",
        "Backend", "FFN (us)", "vs Dense", "Features"
    );
    println!("{}", "-".repeat(46));
    println!(
        "{:>12} {:>8.0}us {:>10} {:>7.0}%",
        "dense", dense_us, "baseline", 100.0
    );

    // Cached FFN: zero matmuls
    let cached_ffn = CachedFfn::calibrate(weights, &token_ids);
    let _ = cached_ffn.forward(layer, &x);
    let start = Instant::now();
    for _ in 0..args.iterations {
        let _ = cached_ffn.forward(layer, &x);
    }
    let cached_us = start.elapsed().as_micros() as f64 / args.iterations as f64;
    println!(
        "{:>12} {:>8.0}us {:>9.1}x {:>8}",
        "cached", cached_us, dense_us / cached_us, "lookup"
    );

    // Sparse at each K
    for &k in &k_values {
        let k = k.min(intermediate);
        let sparse_ffn = SparseFfn { weights, top_k: k };
        let _ = sparse_ffn.forward(layer, &x);

        let start = Instant::now();
        for _ in 0..args.iterations {
            let _ = sparse_ffn.forward(layer, &x);
        }
        let sparse_us = start.elapsed().as_micros() as f64 / args.iterations as f64;

        println!(
            "{:>12} {:>8.0}us {:>9.2}x {:>7.1}%",
            format!("sparse:{k}"), sparse_us, dense_us / sparse_us,
            k as f64 / intermediate as f64 * 100.0,
        );
    }

    // Entity-routed at each K (if gate index provided)
    if let Some(ref gi) = gate_index {
        println!("{}", "-".repeat(46));
        for &k in &k_values {
            let k = k.min(intermediate);
            let entity_ffn = EntityRoutedFfn::from_token_ids(weights, gi, &token_ids, k);
            let _ = entity_ffn.forward(layer, &x);

            let start = Instant::now();
            for _ in 0..args.iterations {
                let _ = entity_ffn.forward(layer, &x);
            }
            let entity_us = start.elapsed().as_micros() as f64 / args.iterations as f64;

            println!(
                "{:>12} {:>8.0}us {:>9.2}x {:>7.1}%",
                format!("entity:{k}"), entity_us, dense_us / entity_us,
                k as f64 / intermediate as f64 * 100.0,
            );
        }
    }

    // Clustered hierarchical index
    let top_c_values: Vec<usize> = args.top_c_values.split(',')
        .map(|s| s.trim().parse().unwrap()).collect();

    eprintln!("\nBuilding clustered index: {} clusters, {} iters...",
        args.clusters, 10);
    let cluster_start = Instant::now();
    let cluster_index = ClusteredGateIndex::build(
        weights, &[layer], args.clusters, 1, 10,
        |idx, total| { eprint!("\r  K-means layer {}/{}...", idx + 1, total); },
    );
    eprintln!("\r  Built in {:.1}s, avg cluster size: {:.0}",
        cluster_start.elapsed().as_secs_f64(), cluster_index.avg_cluster_size());

    println!("{}", "-".repeat(46));
    for &tc in &top_c_values {
        // Rebuild with this top_c (cheap — just changes the probe count)
        let mut ci = ClusteredGateIndex::build(
            weights, &[layer], args.clusters, tc, 10,
            |_, _| {},
        );
        ci.top_c = tc;

        let clustered_ffn = ClusteredFfn { weights, cluster_index: &ci, top_k: 10240 };
        let _ = clustered_ffn.forward(layer, &x);

        let start = Instant::now();
        for _ in 0..args.iterations {
            let _ = clustered_ffn.forward(layer, &x);
        }
        let clust_us = start.elapsed().as_micros() as f64 / args.iterations as f64;

        // How many features does this probe count yield?
        let sample_feats = ci.lookup(layer, &x.row(0), 10240).len();

        println!(
            "{:>12} {:>8.0}us {:>9.2}x {:>5} feats",
            format!("clust:c{tc}"), clust_us, dense_us / clust_us, sample_feats,
        );
    }

    Ok(())
}
