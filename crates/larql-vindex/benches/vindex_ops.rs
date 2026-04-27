//! Criterion benchmarks for the core vindex operations.
//!
//! Measures: gate KNN (per-layer + walk), walk, feature lookup, save,
//! load, mutate, and MoE scaling. All against synthetic in-memory
//! indexes — no real model needed.
//!
//! Run: `cargo bench -p larql-vindex --bench vindex_ops`
//!
//! These were previously printed via the hand-rolled `profile_operations`
//! example. They now live as proper criterion benches so we get
//! statistically valid timings, change detection across runs, and HTML
//! reports under `target/criterion/`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use larql_models::TopKEntry;
use larql_vindex::{FeatureMeta, VectorIndex, VindexConfig};
use ndarray::{Array1, Array2};

// ── Synthetic data builders ─────────────────────────────────────────────

fn random_query(hidden: usize) -> Array1<f32> {
    let mut state = 0xc0ffeeu64;
    Array1::from_shape_fn(hidden, |_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

fn synth_gate(features: usize, hidden: usize) -> Array2<f32> {
    let mut state = 42u64;
    Array2::from_shape_fn((features, hidden), |_| {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

fn build_synthetic_index(
    num_layers: usize,
    features: usize,
    hidden: usize,
    top_k_meta: usize,
) -> VectorIndex {
    let mut gate_vectors = Vec::with_capacity(num_layers);
    let mut down_meta = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        gate_vectors.push(Some(synth_gate(features, hidden)));
        let metas: Vec<Option<FeatureMeta>> = (0..features)
            .map(|i| {
                Some(FeatureMeta {
                    top_token: format!("tok{i}"),
                    top_token_id: i as u32,
                    c_score: 0.5 + (i as f32 * 0.001) % 0.5,
                    top_k: (0..top_k_meta)
                        .map(|k| TopKEntry {
                            token: format!("tok{}", i + k),
                            token_id: (i + k) as u32,
                            logit: 1.0 - k as f32 * 0.1,
                        })
                        .collect(),
                })
            })
            .collect();
        down_meta.push(Some(metas));
    }
    VectorIndex::new(gate_vectors, down_meta, num_layers, hidden)
}

// ── Bench groups ────────────────────────────────────────────────────────

/// Per-layer gate KNN at three index sizes — small/medium/large.
/// Throughput is "1 KNN call per iteration".
fn bench_gate_knn(c: &mut Criterion) {
    let mut group = c.benchmark_group("gate_knn_per_layer");
    for &(features, hidden) in &[
        (1024usize, 256usize),
        (4096, 512),
        (10240, 2560), // Gemma 3 4B production size
    ] {
        let index = build_synthetic_index(1, features, hidden, 5);
        let query = random_query(hidden);
        let label = format!("{features}f×{hidden}h");
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(&label), &index, |b, idx| {
            b.iter(|| {
                let _ = idx.gate_knn(0, &query, 10);
            });
        });
    }
    group.finish();
}

/// Multi-layer walk — measures "1 walk across N layers".
fn bench_walk(c: &mut Criterion) {
    let mut group = c.benchmark_group("walk_all_layers");
    for &(num_layers, features, hidden) in &[
        (8usize, 1024usize, 256usize),
        (14, 4096, 512), // Gemma knowledge band
        (8, 10240, 2560),
    ] {
        let index = build_synthetic_index(num_layers, features, hidden, 5);
        let query = random_query(hidden);
        let layers: Vec<usize> = (0..num_layers).collect();
        let label = format!("{num_layers}L×{features}f×{hidden}h");
        group.bench_with_input(BenchmarkId::from_parameter(&label), &index, |b, idx| {
            b.iter(|| {
                let _ = idx.walk(&query, &layers, 10);
            });
        });
    }
    group.finish();
}

/// Feature meta lookup — read-only, hot path.
fn bench_feature_meta_lookup(c: &mut Criterion) {
    let index = build_synthetic_index(8, 1024, 256, 5);
    c.bench_function("feature_meta_lookup", |b| {
        b.iter(|| {
            // Use varying indices to defeat any internal cache.
            let mut sum = 0u32;
            for i in 0..16 {
                if let Some(m) = index.feature_meta(i % 8, (i * 37) % 1024) {
                    sum = sum.wrapping_add(m.top_token_id);
                }
            }
            sum
        });
    });
}

/// Mutation — `set_feature_meta` + `set_gate_vector` per call.
fn bench_mutate(c: &mut Criterion) {
    let mut group = c.benchmark_group("mutate");
    let hidden = 256;
    let features = 1024;
    let num_layers = 8;
    let gate_vec = random_query(hidden);
    let meta = FeatureMeta {
        top_token: "test".into(),
        top_token_id: 42,
        c_score: 0.99,
        top_k: vec![TopKEntry {
            token: "test".into(),
            token_id: 42,
            logit: 0.99,
        }],
    };

    group.bench_function("set_meta_plus_gate", |b| {
        let mut index = build_synthetic_index(num_layers, features, hidden, 5);
        let mut counter = 0usize;
        b.iter(|| {
            let layer = counter % num_layers;
            let feat = counter % features;
            index.set_feature_meta(layer, feat, meta.clone());
            index.set_gate_vector(layer, feat, &gate_vec);
            counter = counter.wrapping_add(1);
        });
    });
    group.finish();
}

/// Save + load round-trip — mostly disk-bound, useful as a regression
/// detector for serialisation overhead.
fn bench_save_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("save_load");
    group.sample_size(20); // disk I/O is slow — fewer samples

    let hidden = 256;
    let features = 1024;
    let num_layers = 8;
    let index = build_synthetic_index(num_layers, features, hidden, 5);

    group.bench_function("save_gate_vectors", |b| {
        let dir = std::env::temp_dir().join("larql_vindex_bench_save_gate");
        b.iter(|| {
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let _ = index.save_gate_vectors(&dir).unwrap();
        });
        let _ = std::fs::remove_dir_all(&dir);
    });

    group.bench_function("save_down_meta", |b| {
        let dir = std::env::temp_dir().join("larql_vindex_bench_save_meta");
        b.iter(|| {
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            let _ = index.save_down_meta(&dir).unwrap();
        });
        let _ = std::fs::remove_dir_all(&dir);
    });

    // Load benchmark needs a saved vindex on disk.
    let load_dir = std::env::temp_dir().join("larql_vindex_bench_load");
    let _ = std::fs::remove_dir_all(&load_dir);
    std::fs::create_dir_all(&load_dir).unwrap();
    let layer_infos = index.save_gate_vectors(&load_dir).unwrap();
    index.save_down_meta(&load_dir).unwrap();
    let config = VindexConfig {
        version: 2,
        model: "bench-load".into(),
        family: "bench".into(),
        source: None,
        checksums: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: features,
        vocab_size: 100,
        embed_scale: 1.0,
        extract_level: larql_vindex::ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        layers: layer_infos,
        down_top_k: 5,
        has_model_weights: false,
        model_config: None,
    };
    VectorIndex::save_config(&config, &load_dir).unwrap();
    let tok_json = r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(load_dir.join("tokenizer.json"), tok_json).unwrap();

    group.bench_function("load_vindex", |b| {
        b.iter(|| {
            let mut cb = larql_vindex::SilentLoadCallbacks;
            let _ = VectorIndex::load_vindex(&load_dir, &mut cb).unwrap();
        });
    });

    let _ = std::fs::remove_dir_all(&load_dir);
    group.finish();
}

/// MoE-style scaling — KNN against a single layer with growing feature
/// counts (1, 2, 4, 8 expert equivalents).
fn bench_moe_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_scaling");
    let hidden = 256;
    let base_features = 1024;
    for &n_experts in &[1usize, 2, 4, 8] {
        let total_features = base_features * n_experts;
        let gate = synth_gate(total_features, hidden);
        let index = VectorIndex::new(vec![Some(gate)], vec![None], 1, hidden);
        let query = random_query(hidden);
        group.throughput(Throughput::Elements(total_features as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{n_experts}x_experts")),
            &index,
            |b, idx| {
                b.iter(|| {
                    let _ = idx.gate_knn(0, &query, 10);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_gate_knn,
    bench_walk,
    bench_feature_meta_lookup,
    bench_mutate,
    bench_save_load,
    bench_moe_scaling,
);
criterion_main!(benches);
