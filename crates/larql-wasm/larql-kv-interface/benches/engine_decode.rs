//! Criterion microbenchmarks for the four KV engines on synthetic weights.
//!
//! Times prefill (8-token prompt) and a single decode step on a 2-layer
//! synthetic model. The synthetic fixture is small so these benches run
//! quickly and don't depend on a vindex on disk; for end-to-end real-model
//! numbers see `cargo bench -p kv-cache-benchmark --bench kv_strategies`.

use criterion::{criterion_group, criterion_main, Criterion};
use larql_compute::cpu_backend;
use larql_inference::test_utils::make_test_weights;
use larql_kv::EngineKind;

fn bench_prefill(c: &mut Criterion) {
    let weights = make_test_weights();
    let prompt: Vec<u32> = (0..8).collect();

    let kinds = [
        (
            "markov-rs",
            EngineKind::MarkovResidual { window_size: None },
        ),
        (
            "unlimited-context",
            EngineKind::UnlimitedContext { window_size: 4 },
        ),
        ("turbo-quant-4bit", EngineKind::TurboQuant { bits: 4 }),
        (
            "apollo",
            EngineKind::Apollo {
                injection_layer: 1,
                inject_coefficient: 8.0,
                top_k: 4,
            },
        ),
    ];

    let mut group = c.benchmark_group("prefill");
    for (name, kind) in kinds {
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut engine = kind.clone().build(cpu_backend());
                let _ = engine.prefill(&weights, &prompt);
            });
        });
    }
    group.finish();
}

fn bench_decode_step(c: &mut Criterion) {
    let weights = make_test_weights();
    let prompt: Vec<u32> = (0..8).collect();

    let kinds = [
        (
            "markov-rs",
            EngineKind::MarkovResidual { window_size: None },
        ),
        (
            "unlimited-context",
            EngineKind::UnlimitedContext { window_size: 4 },
        ),
        ("turbo-quant-4bit", EngineKind::TurboQuant { bits: 4 }),
        (
            "apollo",
            EngineKind::Apollo {
                injection_layer: 1,
                inject_coefficient: 8.0,
                top_k: 4,
            },
        ),
    ];

    let mut group = c.benchmark_group("decode_step");
    for (name, kind) in kinds {
        group.bench_function(name, |b| {
            // Pre-warm: prefill once, then time a single decode_step.
            let mut engine = kind.clone().build(cpu_backend());
            let _ = engine.prefill(&weights, &prompt);
            b.iter(|| {
                let _ = engine.decode_step(&weights, 1);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_prefill, bench_decode_step);
criterion_main!(benches);
