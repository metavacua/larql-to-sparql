//! Criterion benchmarks for `memit_solve` — the vanilla MEMIT
//! decomposition that powers `COMPACT MAJOR` cycles.
//!
//! Wraps `larql_compute::cpu::ops::linalg::ridge_decomposition_solve`
//! and additionally walks every fact to compute `decomposed`,
//! `reconstruction_cos`, and `max_off_diagonal`. The end-to-end timing
//! here is what COMPACT MAJOR sees per layer.
//!
//! Run: `cargo bench -p larql-vindex --bench memit_solve`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use larql_vindex::memit_solve;
use ndarray::Array2;

fn synth(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut state = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

fn bench_memit_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("memit_solve");
    group.sample_size(20);
    // Realistic shapes — N facts × hidden_dim.
    // d=2560 = Gemma 3 4B; d=576 = v11 TinyStories.
    let configs = [
        (10usize, 576usize),
        (30, 576),
        (60, 576),
        (10, 2560),
        (30, 2560),
        (60, 2560),
    ];
    for &(n, d) in &configs {
        let keys = synth(n, d, 1);
        let targets = synth(n, d, 2);
        let label = format!("N={n}_d={d}");
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &(&keys, &targets),
            |b, (k, t)| {
                b.iter(|| memit_solve(k, t, 1e-3).unwrap());
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_memit_solve);
criterion_main!(benches);
