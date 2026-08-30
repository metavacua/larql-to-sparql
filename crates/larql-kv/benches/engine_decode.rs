//! Criterion microbenchmarks for the KV engines on synthetic weights.
//!
//! Times prefill (8-token prompt) and a single decode step on the
//! synthetic test model. The fixture is small so these benches run
//! quickly and don't depend on a vindex on disk; for end-to-end
//! real-model numbers run `larql bench <vindex> --engine <spec>` from
//! the CLI. (The retired `kv-cache-benchmark::kv_strategies` synthetic
//! comparator was deprecated in 2026-05-16 — it measured random-vector
//! encode/decode, not real decode steady-state.)
//!
//! Engines covered: whatever [`EngineKind::bench_specs`] lists. That is
//! the single source of truth, pinned against the engine roster by
//! `bench_specs_cover_every_benchable_engine` in the lib tests — so a
//! new engine cannot land without either a bench arm or a written
//! reason it can't have one. This list used to be hand-maintained here
//! and had silently fallen to 7 of 9 engines.

use criterion::{criterion_group, criterion_main, Criterion};
use larql_inference::cpu_engine_backend;
use larql_inference::ffn::WeightFfn;
use larql_inference::test_utils::make_test_weights;
use larql_kv::EngineKind;

/// Engines to bench, from [`EngineKind::bench_specs`]. The spec string
/// doubles as the criterion benchmark id.
///
/// Apollo is excluded by that list (see `EngineKind::bench_excluded_names`):
/// with no store attached its `prefill` fails closed with `RetrievalMiss`
/// before touching the model, so benching it here timed the error return
/// — ~65 ns against ~16 µs for `standard`, which read as a 250x win.
fn all_engines() -> Vec<(&'static str, EngineKind)> {
    EngineKind::bench_specs()
        .iter()
        .map(|spec| {
            let kind = EngineKind::from_name(spec)
                .unwrap_or_else(|| panic!("bench spec {spec:?} failed to parse"));
            (*spec, kind)
        })
        .collect()
}

fn bench_prefill(c: &mut Criterion) {
    let weights = make_test_weights();
    let prompt: Vec<u32> = (0..8).collect();
    let ffn = WeightFfn { weights: &weights };

    let mut group = c.benchmark_group("prefill");
    for (name, kind) in all_engines() {
        group.bench_function(name, |b| {
            b.iter(|| {
                let mut engine = kind.clone().build(cpu_engine_backend());
                // Unwrap, don't discard: a `let _ =` here would happily
                // time an engine that bailed out before doing any work
                // and report the error path as a stellar result.
                engine
                    .prefill(&weights, &ffn, &prompt)
                    .unwrap_or_else(|e| panic!("{name}: prefill failed: {e}"));
            });
        });
    }
    group.finish();
}

fn bench_decode_step(c: &mut Criterion) {
    let weights = make_test_weights();
    let prompt: Vec<u32> = (0..8).collect();
    let ffn = WeightFfn { weights: &weights };

    let mut group = c.benchmark_group("decode_step");
    for (name, kind) in all_engines() {
        group.bench_function(name, |b| {
            // Measure ONE decode step at a fixed (prompt-length) context. A
            // fresh prefilled engine per timed call keeps the K/V cache from
            // growing across iterations — the unbounded `standard` engine
            // otherwise appends ~N positions over criterion's N iterations,
            // making the per-call cost non-stationary (and the result a
            // function of iteration count, not single-step latency). Setup is
            // untimed; only `decode_step` is measured.
            b.iter_batched_ref(
                || {
                    let mut engine = kind.clone().build(cpu_engine_backend());
                    engine
                        .prefill(&weights, &ffn, &prompt)
                        .unwrap_or_else(|e| panic!("{name}: prefill failed: {e}"));
                    engine
                },
                |engine| {
                    engine
                        .decode_step(&weights, &ffn, 1)
                        .unwrap_or_else(|e| panic!("{name}: decode_step failed: {e}"));
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Step-4 parity-relevant bench: end-to-end token generation through
/// `generate_with_engine` (Standard) vs the legacy `generate_cached_backend`.
/// If `Standard` is a parity-preserving wrapper, this benchmark
/// quantifies the dispatch-trait overhead — should be a wash.
fn bench_engine_vs_legacy_generation(c: &mut Criterion) {
    use larql_inference::test_utils::make_test_tokenizer;
    use larql_kv::generation::{generate_cached_backend, generate_with_engine};
    use larql_kv::StandardEngine;

    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let ffn = WeightFfn { weights: &weights };
    let prompt: Vec<u32> = (0..8).collect();
    let max = 8;

    let mut group = c.benchmark_group("generate");

    group.bench_function("legacy_generate_cached_backend", |b| {
        b.iter(|| {
            generate_cached_backend(
                &weights,
                &tokenizer,
                &ffn,
                &prompt,
                max,
                None,
                None,
                |_, _| {},
            );
        });
    });

    group.bench_function("engine_dispatch_standard", |b| {
        b.iter(|| {
            let mut engine = larql_kv::AnyEngine::Kv(Box::new(StandardEngine::new(None)));
            generate_with_engine(
                &mut engine,
                &weights,
                &tokenizer,
                &ffn,
                &prompt,
                max,
                |_, _| {},
            );
        });
    });

    // A5: async dispatch on `CpuBackend` is a degenerate `Ready*` wrapper.
    // Expected: bit-identical token stream + tok/s within criterion noise
    // of the sync path. Confirms the `BackendSlot::Async` branch +
    // `Ready*` handle allocations don't introduce overhead on the CPU
    // path (the only path that matters until A4 lands real Metal
    // deferred dispatch).
    group.bench_function("engine_dispatch_standard_async", |b| {
        use larql_inference::AsyncComputeBackend;
        b.iter(|| {
            let backend: Box<dyn AsyncComputeBackend> = Box::new(larql_compute::CpuBackend);
            let mut engine = larql_kv::AnyEngine::Kv(Box::new(StandardEngine::with_async_backend(
                None, backend,
            )));
            generate_with_engine(
                &mut engine,
                &weights,
                &tokenizer,
                &ffn,
                &prompt,
                max,
                |_, _| {},
            );
        });
    });

    group.finish();
}

/// Compares the per-layer dispatch helpers directly: sync vs async on
/// CpuBackend. Isolates the `attention_*_async` + `read_hidden` + `flush`
/// overhead from the surrounding generate-loop / sampling work.
fn bench_helpers_sync_vs_async(c: &mut Criterion) {
    use larql_inference::kv_dispatch::helpers::{
        kv_decode_step_via_dispatch, kv_decode_step_via_dispatch_async, kv_prefill_via_dispatch,
        kv_prefill_via_dispatch_async,
    };

    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let prompt: Vec<u32> = (0..8).collect();
    let cpu = larql_compute::CpuBackend;

    let mut group = c.benchmark_group("helpers");

    group.bench_function("prefill_sync", |b| {
        b.iter(|| {
            let _ = kv_prefill_via_dispatch(
                &cpu,
                larql_inference::WeightsView::dense(&weights),
                &ffn,
                &prompt,
                None,
                None,
            )
            .unwrap()
            .expect("dispatch produced a result");
        });
    });

    group.bench_function("prefill_async", |b| {
        b.iter(|| {
            let _ = kv_prefill_via_dispatch_async(
                &cpu,
                larql_inference::WeightsView::dense(&weights),
                &ffn,
                &prompt,
                None,
                None,
            )
            .unwrap()
            .expect("dispatch produced a result");
        });
    });

    group.bench_function("decode_step_sync", |b| {
        let (_, mut handles) = kv_prefill_via_dispatch(
            &cpu,
            larql_inference::WeightsView::dense(&weights),
            &ffn,
            &prompt,
            None,
            None,
        )
        .unwrap()
        .expect("dispatch produced a result");
        let mut pos = prompt.len();
        b.iter(|| {
            let _ = kv_decode_step_via_dispatch(
                &cpu,
                larql_inference::WeightsView::dense(&weights),
                &ffn,
                &mut handles,
                1,
                pos,
                None,
                None,
            );
            pos += 1;
        });
    });

    group.bench_function("decode_step_async", |b| {
        let (_, mut handles) = kv_prefill_via_dispatch_async(
            &cpu,
            larql_inference::WeightsView::dense(&weights),
            &ffn,
            &prompt,
            None,
            None,
        )
        .unwrap()
        .expect("dispatch produced a result");
        let mut pos = prompt.len();
        b.iter(|| {
            let _ = kv_decode_step_via_dispatch_async(
                &cpu,
                larql_inference::WeightsView::dense(&weights),
                &ffn,
                &mut handles,
                1,
                pos,
                None,
                None,
            );
            pos += 1;
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_prefill,
    bench_decode_step,
    bench_engine_vs_legacy_generation,
    bench_helpers_sync_vs_async,
);
criterion_main!(benches);
