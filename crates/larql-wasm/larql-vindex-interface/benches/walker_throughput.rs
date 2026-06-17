//! Walker throughput bench.
//!
//! Times the three build-time graph extractors —
//! [`WeightWalker::walk_layer`], [`AttentionWalker::walk_layer`], and
//! [`VectorExtractor::extract_ffn_down`] — against a moderately-sized
//! synthetic Gemma3-style model. The headline is the per-layer cost of
//! each walker on a fixed weight shape.
//!
//! Synthetic dims: `hidden=64, intermediate=128, vocab=128, head_dim=32,
//! num_q_heads=2, num_kv_heads=2, num_layers=2`. Picked so each walker
//! finishes in milliseconds (criterion's outer loop converges) but the
//! matmuls are realistic (>= one cache line per row, > 1k vocab rows so
//! the embed projection isn't trivial).
//!
//! Run: `cargo bench -p larql-vindex --bench walker_throughput`

use std::path::PathBuf;

use criterion::{criterion_group, criterion_main, Criterion};
use larql_core::Graph;
use larql_vindex::walker::vector_extractor::VectorFileHeader;
use larql_vindex::walker::vector_extractor::COMPONENT_FFN_DOWN;
use larql_vindex::walker::{
    attention_walker::AttentionWalker,
    test_fixture::{create_with_dims, ModelDims},
    vector_extractor::{ExtractConfig, SilentExtractCallbacks, VectorExtractor, VectorWriter},
    weight_walker::{SilentWalkCallbacks, WalkConfig, WeightWalker},
};

fn bench_root() -> PathBuf {
    let p = std::env::temp_dir().join("larql_bench_walker_throughput");
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn dims() -> ModelDims {
    ModelDims {
        hidden: 64,
        intermediate: 128,
        vocab: 128,
        head_dim: 32,
        num_q_heads: 2,
        num_kv_heads: 2,
        num_layers: 2,
    }
}

fn bench_walker_throughput(c: &mut Criterion) {
    let root = bench_root();
    let model_dir = root.join("model");
    create_with_dims(&model_dir, &dims());

    let cfg = WalkConfig {
        top_k: 5,
        min_score: 0.0,
    };

    let weight_walker = WeightWalker::load(model_dir.to_str().unwrap()).expect("load weight");
    let attention_walker =
        AttentionWalker::load(model_dir.to_str().unwrap()).expect("load attention");
    let extractor = VectorExtractor::load(model_dir.to_str().unwrap()).expect("load extractor");

    let mut group = c.benchmark_group("walker_throughput");
    group.sample_size(20);

    group.bench_function("weight_walker_layer0", |b| {
        b.iter(|| {
            let mut g = Graph::new();
            let mut cb = SilentWalkCallbacks;
            weight_walker
                .walk_layer(0, &cfg, &mut g, &mut cb)
                .expect("walk weight");
        });
    });

    group.bench_function("attention_walker_layer0", |b| {
        b.iter(|| {
            let mut g = Graph::new();
            let mut cb = SilentWalkCallbacks;
            attention_walker
                .walk_layer(0, &cfg, &mut g, &mut cb)
                .expect("walk attention");
        });
    });

    let extract_cfg = ExtractConfig {
        components: vec![COMPONENT_FFN_DOWN.into()],
        layers: Some(vec![0]),
        top_k: 5,
    };
    let header = VectorFileHeader {
        _header: true,
        component: COMPONENT_FFN_DOWN.into(),
        model: "bench/walker".into(),
        dimension: dims().hidden,
        extraction_date: "2026-05-09".into(),
    };
    group.bench_function("vector_extractor_ffn_down_layer0", |b| {
        b.iter(|| {
            let path = root.join("bench_ffn_down.jsonl");
            let _ = std::fs::remove_file(&path);
            let mut writer = VectorWriter::create(&path).expect("create writer");
            writer.write_header(&header).expect("header");
            let mut cb = SilentExtractCallbacks;
            extractor
                .extract_ffn_down(0, &extract_cfg, &mut writer, &mut cb)
                .expect("extract down");
        });
    });

    group.finish();
}

criterion_group!(benches, bench_walker_throughput);
criterion_main!(benches);
