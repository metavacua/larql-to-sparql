//! Component-level benchmark for all forward pass operations.
//!
//! Profiles every operation in a transformer layer at realistic dimensions
//! WITHOUT requiring model weights. Pure synthetic data.
//!
//! Components: embed lookup, RMSNorm, LayerNorm, RoPE, attention (QKV + scores + output),
//! FFN (gate + up + GEGLU + down), residual add, layer scalar.
//!
//! Run: cargo run --release -p larql-inference --example bench_components

use std::time::Instant;
use ndarray::{Array1, Array2};

fn main() {
    println!("=== Inference Component Benchmark ===\n");
    println!("Gemma 3 4B dims: hidden=2560, inter=10240, heads=8Q/4KV, hd=256\n");

    let hidden = 2560;
    let inter = 10240;
    let head_dim = 256;
    let num_q_heads = 8;
    let num_kv_heads = 4;
    let seq = 6;
    let vocab = 262144;
    let iters = 50;

    let backend = larql_compute::cpu_backend();

    // ── Synthetic data ──
    let h = synth_2d(seq, hidden, 42);
    let norm_weight = synth_1d(hidden, 43);
    let norm_bias = synth_1d(hidden, 44);
    let wq = synth_2d(num_q_heads * head_dim, hidden, 45);
    let wk = synth_2d(num_kv_heads * head_dim, hidden, 46);
    let wv = synth_2d(num_kv_heads * head_dim, hidden, 47);
    let wo = synth_2d(hidden, num_q_heads * head_dim, 48);
    let wgate = synth_2d(inter, hidden, 49);
    let wup = synth_2d(inter, hidden, 50);
    let wdown = synth_2d(hidden, inter, 51);
    let embed_table = synth_2d(vocab, hidden, 52);

    // ── 1. Embedding lookup ──
    let token_ids: Vec<u32> = (0..seq as u32).collect();
    let t = Instant::now();
    for _ in 0..iters {
        let _e: Vec<f32> = token_ids.iter()
            .flat_map(|&tid| embed_table.row(tid as usize).to_vec())
            .collect();
    }
    let embed_us = t.elapsed().as_micros() as f64 / iters as f64;
    println!("  Embed lookup ({seq} tokens):        {:>8.1}µs", embed_us);

    // ── 2. RMSNorm ──
    let t = Instant::now();
    for _ in 0..iters {
        let _normed = rms_norm(&h, &norm_weight, 0.0, 1e-6);
    }
    let rmsnorm_us = t.elapsed().as_micros() as f64 / iters as f64;
    println!("  RMSNorm [{seq},{hidden}]:             {:>8.1}µs", rmsnorm_us);

    // ── 3. LayerNorm (for StarCoder2 comparison) ──
    let t = Instant::now();
    for _ in 0..iters {
        let _normed = layer_norm(&h, &norm_weight, &norm_bias, 1e-5);
    }
    let layernorm_us = t.elapsed().as_micros() as f64 / iters as f64;
    println!("  LayerNorm [{seq},{hidden}]:           {:>8.1}µs  ({:.2}x RMSNorm)",
        layernorm_us, layernorm_us / rmsnorm_us);

    // ── 4. RoPE ──
    let q_proj = synth_2d(seq, num_q_heads * head_dim, 60);
    let t = Instant::now();
    for _ in 0..iters {
        let mut q = q_proj.clone();
        apply_rope_inplace(&mut q, head_dim, num_q_heads, 10000.0, 0);
    }
    let rope_us = t.elapsed().as_micros() as f64 / iters as f64;
    println!("  RoPE (full, {num_q_heads}Q heads):          {:>8.1}µs", rope_us);

    // Partial RoPE (Gemma 4: 25%)
    let t = Instant::now();
    for _ in 0..iters {
        let mut q = q_proj.clone();
        apply_rope_partial_inplace(&mut q, head_dim, num_q_heads, 1000000.0, 0, head_dim / 4);
    }
    let rope_partial_us = t.elapsed().as_micros() as f64 / iters as f64;
    println!("  RoPE (25%, Gemma 4 global):     {:>8.1}µs  ({:.1}x faster)", rope_partial_us, rope_us / rope_partial_us);

    // ── 5. QKV Projection (BLAS) ──
    let t = Instant::now();
    for _ in 0..iters {
        let _q = backend.matmul_transb(h.view(), wq.view());
        let _k = backend.matmul_transb(h.view(), wk.view());
        let _v = backend.matmul_transb(h.view(), wv.view());
    }
    let qkv_us = t.elapsed().as_micros() as f64 / iters as f64;
    println!("  QKV projection (3× BLAS):       {:>8.1}µs", qkv_us);

    // ── 6. Attention scores + softmax + V-weighted sum ──
    let q_mat = synth_2d(seq, num_q_heads * head_dim, 70);
    let k_mat = synth_2d(seq, num_kv_heads * head_dim, 71);
    let v_mat = synth_2d(seq, num_kv_heads * head_dim, 72);
    let t = Instant::now();
    for _ in 0..iters {
        let _attn = attention_reference(&q_mat, &k_mat, &v_mat,
            num_q_heads, num_kv_heads, head_dim, seq);
    }
    let attn_us = t.elapsed().as_micros() as f64 / iters as f64;
    println!("  Attention (scores+softmax+V):    {:>8.1}µs", attn_us);

    // ── 7. O Projection ──
    let attn_out = synth_2d(seq, num_q_heads * head_dim, 73);
    let t = Instant::now();
    for _ in 0..iters {
        let _o = backend.matmul_transb(attn_out.view(), wo.view());
    }
    let o_us = t.elapsed().as_micros() as f64 / iters as f64;
    println!("  O projection (BLAS):             {:>8.1}µs", o_us);

    // ── 8. Residual add ──
    let a = synth_2d(seq, hidden, 80);
    let b = synth_2d(seq, hidden, 81);
    let t = Instant::now();
    for _ in 0..iters {
        let _r = &a + &b;
    }
    let resadd_us = t.elapsed().as_micros() as f64 / iters as f64;
    println!("  Residual add [{seq},{hidden}]:        {:>8.1}µs", resadd_us);

    // ── 9. FFN Gate + Up (BLAS) ──
    let t = Instant::now();
    for _ in 0..iters {
        let _gate = backend.matmul_transb(h.view(), wgate.view());
        let _up = backend.matmul_transb(h.view(), wup.view());
    }
    let ffn_gu_us = t.elapsed().as_micros() as f64 / iters as f64;
    println!("  FFN gate+up (2× BLAS):           {:>8.1}µs", ffn_gu_us);

    // ── 10. GEGLU SiLU activation ──
    let gate_vals = synth_2d(seq, inter, 90);
    let up_vals = synth_2d(seq, inter, 91);
    let t = Instant::now();
    for _ in 0..iters {
        let _act = geglu_silu(&gate_vals, &up_vals);
    }
    let geglu_us = t.elapsed().as_micros() as f64 / iters as f64;
    println!("  GEGLU SiLU [{seq},{inter}]:        {:>8.1}µs", geglu_us);

    // ── 11. FFN Down (BLAS) ──
    let act_buf = synth_2d(seq, inter, 92);
    let t = Instant::now();
    for _ in 0..iters {
        let _down = backend.matmul_transb(act_buf.view(), wdown.view());
    }
    let ffn_down_us = t.elapsed().as_micros() as f64 / iters as f64;
    println!("  FFN down (BLAS):                 {:>8.1}µs", ffn_down_us);

    // ── 12. Logits (vocab projection) ──
    let last_hidden = synth_2d(1, hidden, 100);
    let t = Instant::now();
    for _ in 0..5 {
        let _logits = backend.matmul_transb(last_hidden.view(), embed_table.view());
    }
    let logits_us = t.elapsed().as_micros() as f64 / 5.0;
    println!("  Logits [1,{hidden}]×[{vocab},{hidden}]^T: {:>8.0}µs", logits_us);

    // ── Summary ──
    let layer_total = rmsnorm_us + qkv_us + rope_us + attn_us + o_us + resadd_us
        + rmsnorm_us + ffn_gu_us + geglu_us + ffn_down_us + resadd_us;
    let full_model = layer_total * 34.0 + embed_us + logits_us;

    println!("\n--- Per-Layer Breakdown (CPU BLAS, seq={seq}) ---\n");
    println!("  Component          Time      %");
    println!("  ──────────────── ────────── ─────");
    print_pct("  RMSNorm (×2)", rmsnorm_us * 2.0, layer_total);
    print_pct("  QKV projection", qkv_us, layer_total);
    print_pct("  RoPE", rope_us, layer_total);
    print_pct("  Attention", attn_us, layer_total);
    print_pct("  O projection", o_us, layer_total);
    print_pct("  Residual add (×2)", resadd_us * 2.0, layer_total);
    print_pct("  FFN gate+up", ffn_gu_us, layer_total);
    print_pct("  GEGLU", geglu_us, layer_total);
    print_pct("  FFN down", ffn_down_us, layer_total);
    println!("  ──────────────── ──────────");
    println!("  Layer total:     {:>8.0}µs", layer_total);
    println!("  34-layer model:  {:>8.1}ms", full_model / 1000.0);
    println!("  Projected tok/s: {:.0}", 1_000_000.0 / full_model);

    println!("\n--- Comparison ---\n");
    println!("  LARQL CPU (projected):  {:.1}ms  ({:.0} tok/s)", full_model / 1000.0, 1_000_000.0 / full_model);
    println!("  LARQL GPU Q4_K decode:  17.5ms  (57 tok/s)");
    println!("  Ollama (34L, Metal):    10.3ms  (97 tok/s)");
    println!("  Projected cached (8L):  {:.1}ms  ({:.0} tok/s)",
        layer_total * 8.0 / 1000.0 + logits_us / 1000.0,
        1_000_000.0 / (layer_total * 8.0 + logits_us));
}

fn print_pct(label: &str, us: f64, total: f64) {
    println!("{label:<20} {:>8.0}µs  {:>4.1}%", us, us / total * 100.0);
}

fn synth_2d(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut s = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

fn synth_1d(len: usize, seed: u64) -> Array1<f32> {
    let mut s = seed;
    Array1::from_shape_fn(len, |_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

fn rms_norm(x: &Array2<f32>, weight: &Array1<f32>, offset: f32, eps: f64) -> Array2<f32> {
    let (seq, dim) = (x.shape()[0], x.shape()[1]);
    let mut out = Array2::zeros((seq, dim));
    for s in 0..seq {
        let row = x.row(s);
        let sq_sum: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let rms = (1.0 / (sq_sum / dim as f64 + eps).sqrt()) as f32;
        for d in 0..dim {
            out[[s, d]] = row[d] * (weight[d] + offset) * rms;
        }
    }
    out
}

fn layer_norm(x: &Array2<f32>, weight: &Array1<f32>, bias: &Array1<f32>, eps: f64) -> Array2<f32> {
    let (seq, dim) = (x.shape()[0], x.shape()[1]);
    let mut out = Array2::zeros((seq, dim));
    for s in 0..seq {
        let row = x.row(s);
        let mean: f64 = row.iter().map(|&v| v as f64).sum::<f64>() / dim as f64;
        let var: f64 = row.iter().map(|&v| ((v as f64) - mean).powi(2)).sum::<f64>() / dim as f64;
        let inv_std = (1.0 / (var + eps).sqrt()) as f32;
        let mean_f32 = mean as f32;
        for d in 0..dim {
            out[[s, d]] = (row[d] - mean_f32) * inv_std * weight[d] + bias[d];
        }
    }
    out
}

fn apply_rope_inplace(q: &mut Array2<f32>, head_dim: usize, num_heads: usize, base: f32, start_pos: usize) {
    let seq = q.shape()[0];
    for s in 0..seq {
        let pos = (start_pos + s) as f32;
        for h in 0..num_heads {
            let offset = h * head_dim;
            let half = head_dim / 2;
            for d in 0..half {
                let freq = 1.0 / base.powf(2.0 * d as f32 / head_dim as f32);
                let angle = pos * freq;
                let (sin_a, cos_a) = angle.sin_cos();
                let re = q[[s, offset + d]];
                let im = q[[s, offset + d + half]];
                q[[s, offset + d]] = re * cos_a - im * sin_a;
                q[[s, offset + d + half]] = re * sin_a + im * cos_a;
            }
        }
    }
}

fn apply_rope_partial_inplace(q: &mut Array2<f32>, head_dim: usize, num_heads: usize, base: f32, start_pos: usize, rotary_dim: usize) {
    let seq = q.shape()[0];
    let half = rotary_dim / 2;
    for s in 0..seq {
        let pos = (start_pos + s) as f32;
        for h in 0..num_heads {
            let offset = h * head_dim;
            for d in 0..half {
                let freq = 1.0 / base.powf(2.0 * d as f32 / rotary_dim as f32);
                let angle = pos * freq;
                let (sin_a, cos_a) = angle.sin_cos();
                let re = q[[s, offset + d]];
                let im = q[[s, offset + d + half]];
                q[[s, offset + d]] = re * cos_a - im * sin_a;
                q[[s, offset + d + half]] = re * sin_a + im * cos_a;
            }
        }
    }
}

fn geglu_silu(gate: &Array2<f32>, up: &Array2<f32>) -> Array2<f32> {
    let mut out = Array2::zeros(gate.raw_dim());
    ndarray::Zip::from(&mut out).and(gate).and(up).for_each(|o, &g, &u| {
        *o = (g / (1.0 + (-g).exp())) * u;
    });
    out
}

fn attention_reference(
    q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>,
    num_q: usize, num_kv: usize, head_dim: usize, seq: usize,
) -> Array2<f32> {
    let mut out = Array2::zeros((seq, num_q * head_dim));
    let scale = 1.0 / (head_dim as f32).sqrt();
    for s in 0..seq {
        for h in 0..num_q {
            let kv_h = h / (num_q / num_kv);
            let q_off = h * head_dim;
            let k_off = kv_h * head_dim;
            // Scores
            let mut scores = vec![f32::NEG_INFINITY; seq];
            for t in 0..=s {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[[s, q_off + d]] * k[[t, k_off + d]];
                }
                scores[t] = dot * scale;
            }
            // Softmax
            let max = scores[..=s].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = scores[..=s].iter().map(|&sc| (sc - max).exp()).sum();
            // V-weighted sum
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for t in 0..=s {
                    let w = (scores[t] - max).exp() / exp_sum;
                    acc += w * v[[t, k_off + d]];
                }
                out[[s, q_off + d]] = acc;
            }
        }
    }
    out
}
