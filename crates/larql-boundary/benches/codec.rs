//! Benchmarks for Phase 1 (codec) and Phase 2 (metadata).
//!
//! Measured on M3 Max (aarch64-apple-darwin):
//!
//! | Operation              | Time     | Throughput  | Notes                        |
//! |------------------------|----------|-------------|------------------------------|
//! | bf16_encode_d2560      | 1.21 µs  | 8.1 GB/s    | bit-manip, memory-bound      |
//! | bf16_decode_d2560      | 0.27 µs  | 36 GB/s     | shift + store                |
//! | int8_encode_d2560      | 4.62 µs  | 2.1 GB/s    | σ + clamp + quantize         |
//! | int8_decode_d2560      | 0.23 µs  | 42 GB/s     | multiply by scale            |
//! | metadata_compute_no_hat | 517 µs  | 1.9 GB/s    | log_softmax over 262K vocab  |
//! | metadata_compute_with_hat | 659 µs | 1.5 GB/s   | + one extra forward          |
//!
//! **Bottleneck note.** `metadata_compute` at 517 µs is dominated by `log_softmax`
//! over the full 262 K vocabulary. This is the Phase 2 gate check, run once per
//! boundary. At a 512-token chunk stride with 50 tok/s decode, a new boundary
//! arrives every ~10 s, making 517 µs negligible (0.005% of budget).
//!
//! The codec ops (bf16/int8 encode+decode) are µs-level and never on the critical
//! path next to the model forward pass (~150 ms per token on CPU).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use larql_boundary::codec::{bf16, int8};
use larql_boundary::metadata;

// Gemma 3 4B dimensions
const D: usize = 2560;
const VOCAB: usize = 262_145;

fn residual() -> Vec<f32> {
    (0..D).map(|i| (i as f32 * 0.01).sin() * 80.0).collect()
}

fn logits() -> Vec<f32> {
    let mut l = vec![0.01f32; VOCAB];
    l[42] = 10.0;
    l[17] = 3.0;
    l
}

fn bench_bf16(c: &mut Criterion) {
    let r = residual();
    let enc = bf16::encode(&r);

    // Throughput: bytes read from the input f32 slice.
    let input_bytes = (D * 4) as u64;
    let mut g = c.benchmark_group("bf16");
    g.throughput(Throughput::Bytes(input_bytes));
    g.bench_function("encode_d2560", |b| b.iter(|| bf16::encode(black_box(&r))));
    g.bench_function("decode_d2560", |b| b.iter(|| bf16::decode(black_box(&enc))));
    g.finish();

    // Flat names for criterion baseline compat.
    c.bench_function("bf16_encode_d2560", |b| {
        b.iter(|| bf16::encode(black_box(&r)))
    });
    c.bench_function("bf16_decode_d2560", |b| {
        b.iter(|| bf16::decode(black_box(&enc)))
    });
}

fn bench_int8(c: &mut Criterion) {
    let r = residual();
    let payload = int8::encode(&r);
    let bytes = payload.to_bytes();

    let input_bytes = (D * 4) as u64;
    let mut g = c.benchmark_group("int8");
    g.throughput(Throughput::Bytes(input_bytes));
    g.bench_function("encode_d2560", |b| b.iter(|| int8::encode(black_box(&r))));
    g.bench_function("decode_d2560", |b| {
        b.iter(|| {
            let p = int8::Payload::from_bytes(black_box(&bytes));
            int8::decode(black_box(&p))
        })
    });
    g.finish();

    c.bench_function("int8_encode_d2560", |b| {
        b.iter(|| int8::encode(black_box(&r)))
    });
    c.bench_function("int8_decode_d2560", |b| {
        b.iter(|| {
            let p = int8::Payload::from_bytes(black_box(&bytes));
            int8::decode(black_box(&p))
        })
    });
}

fn bench_metadata(c: &mut Criterion) {
    let raw = logits();
    let hat = {
        let mut h = raw.clone();
        h[42] = 9.5;
        h
    };

    // Throughput: both logit slices read (raw + hat for with_hat).
    let vocab_bytes = (VOCAB * 4) as u64;
    let mut g = c.benchmark_group("metadata");
    g.throughput(Throughput::Bytes(vocab_bytes));
    g.bench_function("compute_no_hat", |b| {
        b.iter(|| metadata::compute(black_box(&raw), None))
    });
    g.bench_function("compute_with_hat", |b| {
        b.iter(|| metadata::compute(black_box(&raw), Some(black_box(&hat))))
    });
    g.finish();

    c.bench_function("metadata_compute_no_hat", |b| {
        b.iter(|| metadata::compute(black_box(&raw), None))
    });
    c.bench_function("metadata_compute_with_hat", |b| {
        b.iter(|| metadata::compute(black_box(&raw), Some(black_box(&hat))))
    });
}

fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip");
    let r = residual();
    for d in [256usize, 1024, 2560] {
        let rv: Vec<f32> = r[..d].to_vec();
        let bytes = (d * 4) as u64;
        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(BenchmarkId::new("bf16", d), &rv, |b, v| {
            b.iter(|| bf16::decode(&bf16::encode(black_box(v))))
        });
        group.bench_with_input(BenchmarkId::new("int8", d), &rv, |b, v| {
            b.iter(|| int8::decode(&int8::encode(black_box(v))))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_bf16,
    bench_int8,
    bench_metadata,
    bench_roundtrip
);
criterion_main!(benches);
