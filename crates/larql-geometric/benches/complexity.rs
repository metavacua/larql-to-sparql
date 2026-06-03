// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
//
// Complexity benchmark: O(n) complex-linear vs O(n²) standard-softmax attention.
// Run: cargo bench -p larql-geometric --bench complexity
// Expected: complex-linear scales ~2x per seq_len doubling; standard scales ~4x.

use std::time::Instant;
use larql_geometric::attention::{AttnBackend, AttnInput, AttnOutput};
use larql_geometric::attention::linear_complex::ComplexLinearAttn;
use larql_geometric::attention::reference::standard_attn;

fn time_ms(f: impl Fn() -> AttnOutput, iters: usize) -> f64 {
    // warm-up
    f();
    let t = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(f());
    }
    t.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

struct StandardAttn;
impl AttnBackend for StandardAttn {
    fn name(&self) -> &str {
        "standard"
    }
    fn forward(&self, inp: &AttnInput) -> AttnOutput {
        standard_attn(inp)
    }
}

fn bench_pair(n: usize, d: usize, iters: usize) -> (f64, f64) {
    let q: Vec<f32> = (0..n * d)
        .map(|i| ((i as f32) * 0.07).sin())
        .collect();
    let k: Vec<f32> = (0..n * d)
        .map(|i| ((i as f32) * 0.11 + 0.3).cos())
        .collect();
    let v: Vec<f32> = (0..n * d)
        .map(|i| ((i as f32) * 0.05).sin() * 0.5)
        .collect();

    let linear = ComplexLinearAttn::new();
    let std_b = StandardAttn;

    let t_lin = time_ms(
        || {
            let inp = AttnInput {
                q: &q,
                k: &k,
                v: &v,
                seq_len: n,
                head_dim: d,
            };
            linear.forward(&inp)
        },
        iters,
    );
    let t_std = time_ms(
        || {
            let inp = AttnInput {
                q: &q,
                k: &k,
                v: &v,
                seq_len: n,
                head_dim: d,
            };
            std_b.forward(&inp)
        },
        iters,
    );
    (t_lin, t_std)
}

fn main() {
    let d = 32usize;
    let iters = 8usize;
    let seq_lens: &[usize] = &[16, 32, 64, 128, 256];

    println!("\n=== Attention complexity benchmark (d={d}) ===");
    println!(
        "{:>8} | {:>20} | {:>20} | {:>8}",
        "seq_len", "complex-linear (ms)", "standard (ms)", "ratio"
    );
    println!(
        "{:-<8}-+-{:-<20}-+-{:-<20}-+-{:-<8}",
        "", "", "", ""
    );

    let mut prev: Option<(f64, f64)> = None;
    for &n in seq_lens {
        let (t_lin, t_std) = bench_pair(n, d, iters);
        let ratio = if t_lin > 1e-9 { t_std / t_lin } else { 0.0 };
        print!(
            "{n:>8} | {t_lin:>20.4} | {t_std:>20.4} | {ratio:>7.2}x"
        );
        if let Some((p_lin, p_std)) = prev {
            let lin_scale = if p_lin > 1e-9 { t_lin / p_lin } else { 0.0 };
            let std_scale = if p_std > 1e-9 { t_std / p_std } else { 0.0 };
            println!(
                "  (lin×{lin_scale:.2} std×{std_scale:.2})"
            );
        } else {
            println!();
        }
        prev = Some((t_lin, t_std));
    }
    println!("\nExpected: complex-linear ≈2x per doubling, standard ≈4x per doubling.");
}
