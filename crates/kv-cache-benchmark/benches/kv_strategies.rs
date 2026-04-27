use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use kv_cache_benchmark::*;
use kv_cache_benchmark::model_config::ModelConfig;
use kv_cache_benchmark::standard_kv::StandardKv;
use kv_cache_benchmark::turboquant::TurboQuant;
use kv_cache_benchmark::markov_residual::MarkovResidual;
use rand::prelude::*;

fn bench_encode(c: &mut Criterion) {
    let config = ModelConfig::gemma_4b();
    let dim = config.kv_dim();

    let mut rng = StdRng::seed_from_u64(42);
    let keys: Vec<Vec<f32>> = (0..200)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
        .collect();
    let values: Vec<Vec<f32>> = (0..200)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
        .collect();

    let mut group = c.benchmark_group("encode");

    group.bench_function("standard_kv", |b| {
        let s = StandardKv;
        b.iter(|| s.encode(&keys, &values))
    });

    group.bench_function("turboquant_4bit", |b| {
        let s = TurboQuant::new(4);
        b.iter(|| s.encode(&keys, &values))
    });

    group.bench_function("turboquant_3bit", |b| {
        let s = TurboQuant::new(3);
        b.iter(|| s.encode(&keys, &values))
    });

    group.bench_function("markov_residual", |b| {
        let s = MarkovResidual::new(512);
        b.iter(|| s.encode(&keys, &values))
    });

    group.finish();
}

fn bench_wht(c: &mut Criterion) {
    let mut group = c.benchmark_group("wht");

    for dim in [128, 256] {
        let x: Vec<f32> = (0..dim).map(|i| (i as f32 - dim as f32 / 2.0) / 100.0).collect();
        group.bench_with_input(BenchmarkId::new("wht", dim), &x, |b, x| {
            b.iter(|| kv_cache_benchmark::turboquant::rotation::wht(x))
        });
    }

    group.finish();
}

fn bench_memory_sweep(c: &mut Criterion) {
    let config = ModelConfig::gemma_4b();
    let standard = StandardKv;
    let tq4 = TurboQuant::new(4);
    let markov = MarkovResidual::new(512);

    let strategies: Vec<&dyn KvStrategy> = vec![&standard, &tq4, &markov];
    let lengths = benchmark::CONTEXT_LENGTHS;

    c.bench_function("memory_sweep", |b| {
        b.iter(|| benchmark::memory_sweep(&config, &strategies, lengths))
    });
}

criterion_group!(benches, bench_encode, bench_wht, bench_memory_sweep);
criterion_main!(benches);
