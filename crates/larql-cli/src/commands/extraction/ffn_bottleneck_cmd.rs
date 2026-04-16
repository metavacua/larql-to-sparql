use std::time::Instant;

use clap::Args;
use larql_inference::{
    trace_forward, InferenceModel,
};

#[derive(Args)]
pub struct FfnBottleneckArgs {
    /// Model path or HuggingFace model ID.
    #[arg(short, long)]
    model: String,

    /// Prompt to profile.
    #[arg(short, long, default_value = "The capital of France is")]
    prompt: String,

    /// Number of iterations for timing.
    #[arg(short, long, default_value = "10")]
    iterations: usize,

    /// Layer to profile (default: 20).
    #[arg(short, long, default_value = "20")]
    layer: usize,
}

pub fn run(args: FfnBottleneckArgs) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("Loading model: {}", args.model);
    let model = InferenceModel::load(&args.model)?;
    let weights = model.weights();

    let encoding = model.tokenizer().encode(args.prompt.as_str(), true)
        .map_err(|e| format!("tokenize error: {e}"))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let seq_len = token_ids.len();

    eprintln!("Capturing residual at layer {}...", args.layer);
    let trace = trace_forward(weights, &token_ids, &[args.layer], false, 0);
    let residual_vec = &trace.residuals[0].1;
    let hidden = weights.hidden_size;

    // Build input
    let mut x_data = vec![0.0f32; seq_len * hidden];
    for s in 0..seq_len {
        x_data[s * hidden..(s + 1) * hidden].copy_from_slice(residual_vec);
    }
    let x = larql_inference::ndarray::Array2::from_shape_vec((seq_len, hidden), x_data)?;

    let layer = args.layer;
    let arch = &*weights.arch;
    let w_gate = weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
    let w_up = weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
    let w_down = weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();
    let intermediate = w_gate.shape()[0];

    eprintln!(
        "Profiling FFN layer {} — seq_len={}, hidden={}, intermediate={}, iters={}",
        layer, seq_len, hidden, intermediate, args.iterations
    );

    let iters = args.iterations;

    // 1. Gate matmul: x @ gate.T → (seq, intermediate)
    let _ = x.dot(&w_gate.t());
    let start = Instant::now();
    for _ in 0..iters { let _ = x.dot(&w_gate.t()); }
    let gate_us = start.elapsed().as_micros() as f64 / iters as f64;

    // 2. Up matmul: x @ up.T → (seq, intermediate)
    let _ = x.dot(&w_up.t());
    let start = Instant::now();
    for _ in 0..iters { let _ = x.dot(&w_up.t()); }
    let up_us = start.elapsed().as_micros() as f64 / iters as f64;

    // 3. SiLU activation: element-wise on (seq, intermediate)
    let gate_proj = x.dot(&w_gate.t());
    let up_proj = x.dot(&w_up.t());
    let start = Instant::now();
    for _ in 0..iters {
        let activated = gate_proj.mapv(|v| v * larql_inference::ffn::sigmoid(v));
        let _ = &activated * &up_proj;
    }
    let silu_us = start.elapsed().as_micros() as f64 / iters as f64;

    // 4. Down matmul: activation @ down.T → (seq, hidden)
    let activated = gate_proj.mapv(|v| v * larql_inference::ffn::sigmoid(v));
    let activation = &activated * &up_proj;
    let _ = activation.dot(&w_down.t());
    let start = Instant::now();
    for _ in 0..iters { let _ = activation.dot(&w_down.t()); }
    let down_us = start.elapsed().as_micros() as f64 / iters as f64;

    // 5. Top-K selection from gate activations (for sparse path)
    let gate_act = gate_proj.mapv(|v| v * larql_inference::ffn::sigmoid(v));
    let start = Instant::now();
    for _ in 0..iters {
        for s in 0..seq_len {
            let mut indexed: Vec<(usize, f32)> = gate_act.row(s).iter().copied().enumerate().collect();
            indexed.select_nth_unstable_by(64, |a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        }
    }
    let topk_us = start.elapsed().as_micros() as f64 / iters as f64;

    // 6. Gather 64 rows from up matrix
    let up_raw = w_up.as_slice().unwrap();
    let start = Instant::now();
    for _ in 0..iters {
        let mut buf = vec![0.0f32; 64 * hidden];
        for i in 0..64 {
            let src = i * 160 * hidden; // spread out features
            let src = src % (intermediate * hidden);
            buf[i * hidden..(i + 1) * hidden].copy_from_slice(&up_raw[src..src + hidden]);
        }
    }
    let gather_us = start.elapsed().as_micros() as f64 / iters as f64;

    // 7. Sparse gemv: (64, hidden) @ (hidden,) → (64,)
    let mut buf = vec![0.0f32; 64 * hidden];
    for i in 0..64 {
        let src = (i * 160) % intermediate * hidden;
        let src = src.min((intermediate - 1) * hidden);
        buf[i * hidden..(i + 1) * hidden].copy_from_slice(&up_raw[src..src + hidden]);
    }
    let sub = larql_inference::ndarray::ArrayView2::from_shape((64, hidden), &buf).unwrap();
    let x_row = x.row(0);
    let _ = sub.dot(&x_row);
    let start = Instant::now();
    for _ in 0..iters {
        for _ in 0..seq_len {
            let _ = sub.dot(&x_row);
        }
    }
    let sparse_gemv_us = start.elapsed().as_micros() as f64 / iters as f64;

    // Full dense FFN for reference
    let ffn = larql_inference::WeightFfn { weights };
    let _ = larql_inference::FfnBackend::forward(&ffn, layer, &x);
    let start = Instant::now();
    for _ in 0..iters { let _ = larql_inference::FfnBackend::forward(&ffn, layer, &x); }
    let total_us = start.elapsed().as_micros() as f64 / iters as f64;

    let total_parts = gate_us + up_us + silu_us + down_us;

    println!();
    println!("FFN Layer {} Bottleneck Analysis (seq_len={}, hidden={}, intermediate={})",
        layer, seq_len, hidden, intermediate);
    println!("{}", "=".repeat(65));
    println!("{:>30} {:>10} {:>10} {:>10}", "Component", "Time (us)", "% of FFN", "GFLOPS");
    println!("{}", "-".repeat(65));

    let gate_flops = 2.0 * seq_len as f64 * hidden as f64 * intermediate as f64;
    let up_flops = gate_flops;
    let silu_flops = 2.0 * seq_len as f64 * intermediate as f64;
    let down_flops = 2.0 * seq_len as f64 * intermediate as f64 * hidden as f64;

    println!("{:>30} {:>8.0}us {:>9.1}% {:>9.1}",
        "gate matmul (x @ gate.T)", gate_us, gate_us / total_parts * 100.0,
        gate_flops / gate_us / 1000.0);
    println!("{:>30} {:>8.0}us {:>9.1}% {:>9.1}",
        "up matmul (x @ up.T)", up_us, up_us / total_parts * 100.0,
        up_flops / up_us / 1000.0);
    println!("{:>30} {:>8.0}us {:>9.1}% {:>9.1}",
        "SiLU + element mul", silu_us, silu_us / total_parts * 100.0,
        silu_flops / silu_us / 1000.0);
    println!("{:>30} {:>8.0}us {:>9.1}% {:>9.1}",
        "down matmul (act @ down.T)", down_us, down_us / total_parts * 100.0,
        down_flops / down_us / 1000.0);
    println!("{}", "-".repeat(65));
    println!("{:>30} {:>8.0}us {:>9.1}%",
        "Sum of parts", total_parts, 100.0);
    println!("{:>30} {:>8.0}us",
        "Actual dense FFN", total_us);

    println!();
    println!("Sparse path components:");
    println!("{}", "-".repeat(65));
    println!("{:>30} {:>8.0}us    (gate matmul still required)",
        "gate matmul", gate_us);
    println!("{:>30} {:>8.0}us    (select top-64 from {})",
        "top-K selection", topk_us, intermediate);
    println!("{:>30} {:>8.0}us    (64 rows × {} dims)",
        "gather rows", gather_us, hidden);
    println!("{:>30} {:>8.0}us    (64,{}) @ ({},) × {} pos",
        "sparse gate+up gemv", sparse_gemv_us, hidden, hidden, seq_len);
    println!("{:>30} {:>8.0}us    (minimum sparse overhead)",
        "sparse total (no down)", gate_us + topk_us + gather_us + sparse_gemv_us);
    println!();
    println!("{:>30} {:>8.0}us    ({:.0}% of FFN is gate+up matmul)",
        "gate + up matmuls", gate_us + up_us, (gate_us + up_us) / total_parts * 100.0);

    Ok(())
}
