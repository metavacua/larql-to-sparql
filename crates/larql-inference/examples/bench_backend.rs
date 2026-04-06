//! Matmul backend benchmark — CPU (Accelerate BLAS) vs Metal GPU.
//!
//! Benchmarks at transformer-realistic dimensions for Gemma-3 4B:
//!   hidden=2560, head_dim=256, intermediate=10240, vocab=262144
//!
//! Usage:
//!   cargo run --release -p larql-inference --example bench_backend
//!   cargo run --release -p larql-inference --example bench_backend --features metal

use ndarray::Array2;
use std::time::Instant;

use larql_compute::CpuBackend;
use larql_compute::{default_backend, ComputeBackend, MatMulOp};

/// Deterministic f32 matrix.
fn synth_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut state = seed;
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

fn bench<F: FnMut()>(name: &str, iters: usize, mut f: F) {
    // Warmup
    for _ in 0..3.min(iters) {
        f();
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = t0.elapsed();
    let per_iter = elapsed.as_micros() as f64 / iters as f64;

    if per_iter > 10_000.0 {
        println!("  {name:<50} {:.2} ms  ({iters} iters)", per_iter / 1000.0);
    } else {
        println!("  {name:<50} {:.1} us  ({iters} iters)", per_iter);
    }
}

fn bench_backend(label: &str, backend: &dyn ComputeBackend) {
    println!("\n--- {label}: {} ---\n", backend.name());

    // ── Attention projections ──
    // Q/K/V: [seq, hidden] x [hidden, head_dim*num_heads]^T
    let seq = 6;
    let hidden = 2560;
    let head_dim = 256;
    let num_heads = 10;
    let intermediate = 10240;

    let h = synth_matrix(seq, hidden, 1);
    let w_q = synth_matrix(num_heads * head_dim, hidden, 2);

    bench(&format!("Q proj [{seq},{hidden}] x [{},{hidden}]^T", num_heads * head_dim), 50, || {
        let _ = backend.matmul_transb(h.view(), w_q.view());
    });

    // QK^T per head: [seq, head_dim] x [seq, head_dim]^T
    let q = synth_matrix(seq, head_dim, 10);
    let k = synth_matrix(seq, head_dim, 11);

    bench(&format!("QK^T [{seq},{head_dim}] x [{seq},{head_dim}]^T"), 200, || {
        let _ = backend.matmul_transb(q.view(), k.view());
    });

    // scores @ V: [seq, seq] x [seq, head_dim]
    let scores = synth_matrix(seq, seq, 20);
    let v = synth_matrix(seq, head_dim, 21);

    bench(&format!("scores*V [{seq},{seq}] x [{seq},{head_dim}]"), 200, || {
        let _ = backend.matmul(scores.view(), v.view());
    });

    // O projection: [seq, num_heads*head_dim] x [hidden, num_heads*head_dim]^T
    let attn_out = synth_matrix(seq, num_heads * head_dim, 30);
    let w_o = synth_matrix(hidden, num_heads * head_dim, 31);

    bench(&format!("O proj [{seq},{}] x [{hidden},{}]^T", num_heads * head_dim, num_heads * head_dim), 50, || {
        let _ = backend.matmul_transb(attn_out.view(), w_o.view());
    });

    // ── FFN projections ──
    let x = synth_matrix(seq, hidden, 40);
    let w_gate = synth_matrix(intermediate, hidden, 41);

    bench(&format!("FFN gate [{seq},{hidden}] x [{intermediate},{hidden}]^T"), 20, || {
        let _ = backend.matmul_transb(x.view(), w_gate.view());
    });

    let act = synth_matrix(seq, intermediate, 50);
    let w_down = synth_matrix(hidden, intermediate, 51);

    bench(&format!("FFN down [{seq},{intermediate}] x [{hidden},{intermediate}]^T"), 20, || {
        let _ = backend.matmul_transb(act.view(), w_down.view());
    });

    // ── Batched attention heads ──
    let ops: Vec<MatMulOp> = (0..num_heads)
        .map(|h| MatMulOp {
            a: synth_matrix(seq, head_dim, 100 + h as u64),
            b: synth_matrix(seq, head_dim, 200 + h as u64),
            transpose_b: true,
        })
        .collect();

    bench(&format!("Batch QK^T ({num_heads} heads, 1 dispatch)"), 100, || {
        let _ = backend.matmul_batch(&ops);
    });

    bench(&format!("Serial QK^T ({num_heads} heads, {num_heads} calls)"), 100, || {
        for op in &ops {
            let _ = backend.matmul_transb(op.a.view(), op.b.view());
        }
    });

    // ── Logits projection (the big one) ──
    let vocab = 262144;
    let last = synth_matrix(1, hidden, 300);
    let lm_head = synth_matrix(vocab, hidden, 301);

    bench(&format!("Logits [1,{hidden}] x [{vocab},{hidden}]^T"), 5, || {
        let _ = backend.matmul_transb(last.view(), lm_head.view());
    });

    // ── Sequence length scaling ──
    println!("\n  Sequence length scaling (Q projection):");
    for &s in &[1, 6, 12, 24, 48] {
        let h_s = synth_matrix(s, hidden, 400 + s as u64);
        bench(&format!("  seq={s:<4} [{s},{hidden}] x [{},{hidden}]^T", num_heads * head_dim), 20, || {
            let _ = backend.matmul_transb(h_s.view(), w_q.view());
        });
    }
}

fn main() {
    println!("=== MatMul Backend Benchmark ===");
    println!("Gemma-3 4B dimensions: hidden=2560, heads=10, head_dim=256, inter=10240, vocab=262144");

    // Always benchmark CPU
    let cpu = CpuBackend;
    bench_backend("CPU", &cpu);

    // Benchmark default (may be Metal if feature enabled)
    let default = default_backend();
    if default.name() != cpu.name() {
        bench_backend("Default", &*default);

        // ── Head-to-head comparison ──
        println!("\n--- Head-to-head: CPU vs {} ---\n", default.name());

        let cases: Vec<(&str, usize, usize, usize, bool)> = vec![
            ("Q projection", 6, 2560, 2560, true),
            ("QK^T", 6, 256, 6, true),
            ("FFN gate", 6, 2560, 10240, true),
            ("FFN down", 6, 10240, 2560, true),
        ];

        for (name, m, k, n, transb) in cases {
            let a = synth_matrix(m, k, 600);
            let b = if transb {
                synth_matrix(n, k, 601)
            } else {
                synth_matrix(k, n, 601)
            };

            // CPU
            let t0 = Instant::now();
            for _ in 0..20 {
                if transb {
                    let _ = cpu.matmul_transb(a.view(), b.view());
                } else {
                    let _ = cpu.matmul(a.view(), b.view());
                }
            }
            let cpu_us = t0.elapsed().as_micros() as f64 / 20.0;

            // Default
            let t0 = Instant::now();
            for _ in 0..20 {
                if transb {
                    let _ = default.matmul_transb(a.view(), b.view());
                } else {
                    let _ = default.matmul(a.view(), b.view());
                }
            }
            let def_us = t0.elapsed().as_micros() as f64 / 20.0;

            let ratio = cpu_us / def_us.max(0.1);
            let winner = if ratio > 1.0 {
                format!("{} wins {ratio:.1}x", default.name())
            } else {
                format!("CPU wins {:.1}x", 1.0 / ratio)
            };
            println!("  {name:<20} CPU: {cpu_us:>8.0} us  {}: {def_us:>8.0} us  ({winner})",
                default.name());
        }
    } else {
        println!("\n  (Metal not available — default is CPU)");
        println!("  Run with --features metal to compare GPU backend.");
    }

    println!("\n=== Done ===");
}
