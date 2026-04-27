use std::time::Instant;

use clap::Args;
use larql_inference::{
    trace_forward, InferenceModel,
};

#[derive(Args)]
pub struct AttnBottleneckArgs {
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

pub fn run(args: AttnBottleneckArgs) -> Result<(), Box<dyn std::error::Error>> {
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
    let head_dim = weights.head_dim;
    let num_q = weights.num_q_heads;
    let num_kv = weights.num_kv_heads;

    // Build input
    let mut x_data = vec![0.0f32; seq_len * hidden];
    for s in 0..seq_len {
        x_data[s * hidden..(s + 1) * hidden].copy_from_slice(residual_vec);
    }
    let x = larql_inference::ndarray::Array2::from_shape_vec((seq_len, hidden), x_data)?;

    let layer = args.layer;
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let iters = args.iterations;

    let w_q = weights.tensors.get(&arch.attn_q_key(layer)).unwrap();
    let w_k = weights.tensors.get(&arch.attn_k_key(layer)).unwrap();
    let w_v = weights.tensors.get(&arch.attn_v_key(layer)).unwrap();
    let w_o = weights.tensors.get(&arch.attn_o_key(layer)).unwrap();

    let q_dim = num_q * head_dim;
    let kv_dim = num_kv * head_dim;

    eprintln!(
        "Profiling attention layer {} — seq_len={}, hidden={}, heads={}q/{}kv, head_dim={}, iters={}",
        layer, seq_len, hidden, num_q, num_kv, head_dim, iters
    );

    // 0. Input layernorm
    let start = Instant::now();
    for _ in 0..iters {
        let _ = larql_inference::residual::rms_norm(
            &x,
            weights.vectors.get(&arch.input_layernorm_key(layer)),
            norm_offset,
        );
    }
    let norm_us = start.elapsed().as_micros() as f64 / iters as f64;

    let h_norm = larql_inference::residual::rms_norm(
        &x,
        weights.vectors.get(&arch.input_layernorm_key(layer)),
        norm_offset,
    );

    // 1. Q projection: (seq, hidden) @ (hidden, q_dim) → (seq, q_dim)
    let _ = h_norm.dot(&w_q.t());
    let start = Instant::now();
    for _ in 0..iters { let _ = h_norm.dot(&w_q.t()); }
    let q_proj_us = start.elapsed().as_micros() as f64 / iters as f64;

    // 2. K projection
    let _ = h_norm.dot(&w_k.t());
    let start = Instant::now();
    for _ in 0..iters { let _ = h_norm.dot(&w_k.t()); }
    let k_proj_us = start.elapsed().as_micros() as f64 / iters as f64;

    // 3. V projection
    let _ = h_norm.dot(&w_v.t());
    let start = Instant::now();
    for _ in 0..iters { let _ = h_norm.dot(&w_v.t()); }
    let v_proj_us = start.elapsed().as_micros() as f64 / iters as f64;

    // 4. RoPE (approximate — just measure the time to apply_rope)
    let q_full = h_norm.dot(&w_q.t());
    let k_full = h_norm.dot(&w_k.t());
    let start = Instant::now();
    for _ in 0..iters {
        let _ = larql_inference::attention::apply_rope(&q_full, num_q, head_dim, weights.rope_base);
        let _ = larql_inference::attention::apply_rope(&k_full, num_kv, head_dim, weights.rope_base);
    }
    let rope_us = start.elapsed().as_micros() as f64 / iters as f64;

    // 5. QK^T attention scores + softmax + V multiply (the full GQA attention)
    let q_rope = larql_inference::attention::apply_rope(&q_full, num_q, head_dim, weights.rope_base);
    let k_rope = larql_inference::attention::apply_rope(&k_full, num_kv, head_dim, weights.rope_base);
    let v_full = h_norm.dot(&w_v.t());
    let reps = num_q / num_kv;
    let scale = (head_dim as f64).powf(-0.5) * arch.attention_multiplier() as f64;

    let start = Instant::now();
    for _ in 0..iters {
        let _ = larql_inference::attention::gqa_attention_with_weights(
            &q_rope, &k_rope, &v_full, num_q, head_dim, reps, scale, seq_len, false, None,
        );
    }
    let attn_core_us = start.elapsed().as_micros() as f64 / iters as f64;

    // 6. Output projection: (seq, q_dim) @ (q_dim, hidden) → (seq, hidden)
    let (attn_out, _) = larql_inference::attention::gqa_attention_with_weights(
        &q_rope, &k_rope, &v_full, num_q, head_dim, reps, scale, seq_len, false, None,
    );
    let start = Instant::now();
    for _ in 0..iters { let _ = attn_out.dot(&w_o.t()); }
    let o_proj_us = start.elapsed().as_micros() as f64 / iters as f64;

    // 7. Full attention (end-to-end via run_attention_public)
    let start = Instant::now();
    for _ in 0..iters {
        let _ = larql_inference::forward::run_attention_public(weights, &x, layer);
    }
    let full_attn_us = start.elapsed().as_micros() as f64 / iters as f64;

    let sum_parts = norm_us + q_proj_us + k_proj_us + v_proj_us + rope_us + attn_core_us + o_proj_us;

    println!();
    println!("Attention Layer {} Bottleneck (seq_len={}, hidden={}, {}q/{}kv, head_dim={})",
        layer, seq_len, hidden, num_q, num_kv, head_dim);
    println!("{}", "=".repeat(65));
    println!("{:>30} {:>10} {:>10}", "Component", "Time (us)", "% of Attn");
    println!("{}", "-".repeat(65));

    println!("{:>30} {:>8.0}us {:>9.1}%", "input layernorm", norm_us, norm_us / sum_parts * 100.0);
    println!("{:>30} {:>8.0}us {:>9.1}%",
        format!("Q proj ({}→{})", hidden, q_dim), q_proj_us, q_proj_us / sum_parts * 100.0);
    println!("{:>30} {:>8.0}us {:>9.1}%",
        format!("K proj ({}→{})", hidden, kv_dim), k_proj_us, k_proj_us / sum_parts * 100.0);
    println!("{:>30} {:>8.0}us {:>9.1}%",
        format!("V proj ({}→{})", hidden, kv_dim), v_proj_us, v_proj_us / sum_parts * 100.0);
    println!("{:>30} {:>8.0}us {:>9.1}%", "RoPE (Q+K)", rope_us, rope_us / sum_parts * 100.0);
    println!("{:>30} {:>8.0}us {:>9.1}%",
        format!("QK^T + softmax + V ({}h)", num_q), attn_core_us, attn_core_us / sum_parts * 100.0);
    println!("{:>30} {:>8.0}us {:>9.1}%",
        format!("O proj ({}→{})", q_dim, hidden), o_proj_us, o_proj_us / sum_parts * 100.0);
    println!("{}", "-".repeat(65));
    println!("{:>30} {:>8.0}us {:>9.1}%", "Sum of parts", sum_parts, 100.0);
    println!("{:>30} {:>8.0}us", "Actual full attention", full_attn_us);

    println!();
    let proj_total = q_proj_us + k_proj_us + v_proj_us + o_proj_us;
    println!("{:>30} {:>8.0}us {:>9.1}%  (4 linear projections)",
        "Total projections", proj_total, proj_total / sum_parts * 100.0);
    println!("{:>30} {:>8.0}us {:>9.1}%  (RoPE + QK^T + softmax + V)",
        "Total attention math", rope_us + attn_core_us, (rope_us + attn_core_us) / sum_parts * 100.0);
    println!("{:>30} {:>8.0}us {:>9.1}%  (input layernorm)",
        "Total norms", norm_us, norm_us / sum_parts * 100.0);

    Ok(())
}
