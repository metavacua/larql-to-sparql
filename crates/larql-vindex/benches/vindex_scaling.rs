//! Production-dimension scaling benchmarks for vindex.
//!
//! Tests gate KNN and walk at the dimensions of real models (Gemma 3 4B,
//! Llama 3 8B, Llama 3 70B, Mixtral 8x22B). Synthetic data — no real
//! model needed.
//!
//! Run: `cargo bench -p larql-vindex --bench vindex_scaling`
//!
//! These were previously printed by the hand-rolled `profile_scaling`
//! example. They now live as proper criterion benches.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use larql_vindex::VectorIndex;
use ndarray::{Array1, Array2};

fn random_query(hidden: usize) -> Array1<f32> {
    let mut state = 0xdeadbeefu64;
    Array1::from_shape_fn(hidden, |_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

fn synth_matrix(rows: usize, cols: usize) -> Array2<f32> {
    let mut state = 42u64;
    Array2::from_shape_fn((rows, cols), |_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

/// Single-layer gate KNN at production dimensions for the 4 representative
/// model families.
fn bench_production_knn(c: &mut Criterion) {
    let mut group = c.benchmark_group("production_knn_per_layer");
    // (label, intermediate_size, hidden_size)
    let configs: &[(&str, usize, usize)] = &[
        ("gemma-3-4b", 10240, 2560),
        ("llama-3-8b", 14336, 4096),
        ("mixtral-8x22b-1expert", 16384, 6144),
        // Llama 3 70B and 405B are intentionally excluded — they need
        // multi-GB allocations and would slow the bench suite to a crawl.
        // Add them locally if you want to validate scaling.
    ];

    for &(name, features, hidden) in configs {
        let gate = synth_matrix(features, hidden);
        let index = VectorIndex::new(vec![Some(gate)], vec![None], 1, hidden);
        let query = random_query(hidden);
        group.bench_with_input(BenchmarkId::from_parameter(name), &index, |b, idx| {
            b.iter(|| {
                let _ = idx.gate_knn(0, &query, 10);
            });
        });
    }
    group.finish();
}

/// MoE gate KNN — single layer with growing total feature counts. Tests
/// the regime where MoE models have many small experts vs dense models
/// with one large feature bank.
fn bench_moe_production(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_production_knn");
    let hidden = 2560;
    let configs: &[(&str, usize)] = &[
        ("dense-10240", 10240),
        ("8experts-2048", 16384),
        ("16experts-2048", 32768),
        ("64experts-2048", 131072),
    ];

    for &(label, total_features) in configs {
        let gate = synth_matrix(total_features, hidden);
        let index = VectorIndex::new(vec![Some(gate)], vec![None], 1, hidden);
        let query = random_query(hidden);
        group.bench_with_input(BenchmarkId::from_parameter(label), &index, |b, idx| {
            b.iter(|| {
                let _ = idx.gate_knn(0, &query, 10);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_production_knn, bench_moe_production);
criterion_main!(benches);
