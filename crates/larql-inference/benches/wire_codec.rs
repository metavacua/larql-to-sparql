//! Criterion benchmarks for the FFN binary wire codec (ADR-0009).
//!
//! Measures encode and decode throughput (MB/s) for f32, f16, and i8 formats
//! at the hidden sizes and sequence lengths used by production models.
//!
//! Run with:
//!   cargo bench -p larql-inference --bench wire_codec
//!
//! Parameters are read at bench construction — no hardcoded model family names.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use larql_inference::ffn::remote::codec::{
    decode_binary_single, decode_binary_single_f16, encode_binary_request,
};

// Hidden sizes from real models (architecture-agnostic labelling).
const HIDDEN_SIZES: &[(usize, &str)] = &[
    (2560, "h2560"), // typical 4B dense
    (4096, "h4096"), // typical 8B dense / MoE shared
    (5120, "h5120"), // typical 26B MoE
];
const SEQ_LENS: &[(usize, &str)] = &[(1, "seq1"), (32, "seq32"), (256, "seq256")];

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_residual(hidden: usize, seq: usize) -> Vec<f32> {
    (0..hidden * seq)
        .map(|i| (i as f32 * 0.001 - 0.5).clamp(-1.0, 1.0))
        .collect()
}

/// Encode a single-layer f32 request body.
fn encode_f32(hidden: usize, seq: usize) -> Vec<u8> {
    let residual = make_residual(hidden, seq);
    encode_binary_request(Some(0), None, &residual, seq, true, 8092)
}

/// Encode a single-layer f16 response body (mimics server output).
fn encode_f16_response(hidden: usize, seq: usize) -> Vec<u8> {
    use half::f16;
    let residual = make_residual(hidden, seq);
    let mut buf = Vec::with_capacity(12 + hidden * seq * 2);
    buf.extend_from_slice(&0u32.to_le_bytes()); // layer = 0
    buf.extend_from_slice(&(seq as u32).to_le_bytes());
    buf.extend_from_slice(&5.0f32.to_le_bytes()); // latency_ms
    for &v in &residual {
        buf.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    buf
}

/// Encode a single-layer f32 response body.
fn encode_f32_response(hidden: usize, seq: usize) -> Vec<u8> {
    let residual = make_residual(hidden, seq);
    let mut buf = Vec::with_capacity(12 + hidden * seq * 4);
    buf.extend_from_slice(&0u32.to_le_bytes()); // layer = 0
    buf.extend_from_slice(&(seq as u32).to_le_bytes());
    buf.extend_from_slice(&5.0f32.to_le_bytes());
    for &v in &residual {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

fn bench_encode_f32_request(c: &mut Criterion) {
    let mut group = c.benchmark_group("wire_codec/encode_f32_request");
    for &(hidden, hlabel) in HIDDEN_SIZES {
        for &(seq, slabel) in SEQ_LENS {
            let bytes = hidden * seq * 4;
            group.throughput(Throughput::Bytes(bytes as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{hlabel}_{slabel}"), bytes),
                &(hidden, seq),
                |b, &(h, s)| {
                    b.iter(|| encode_f32(h, s));
                },
            );
        }
    }
    group.finish();
}

fn bench_decode_f32_response(c: &mut Criterion) {
    let mut group = c.benchmark_group("wire_codec/decode_f32_response");
    for &(hidden, hlabel) in HIDDEN_SIZES {
        for &(seq, slabel) in SEQ_LENS {
            let body = encode_f32_response(hidden, seq);
            let bytes = body.len() as u64;
            group.throughput(Throughput::Bytes(bytes));
            group.bench_with_input(
                BenchmarkId::new(format!("{hlabel}_{slabel}"), bytes),
                &body,
                |b, body| {
                    b.iter(|| decode_binary_single(body).unwrap());
                },
            );
        }
    }
    group.finish();
}

fn bench_decode_f16_response(c: &mut Criterion) {
    let mut group = c.benchmark_group("wire_codec/decode_f16_response");
    for &(hidden, hlabel) in HIDDEN_SIZES {
        for &(seq, slabel) in SEQ_LENS {
            let body = encode_f16_response(hidden, seq);
            let bytes = body.len() as u64;
            group.throughput(Throughput::Bytes(bytes));
            group.bench_with_input(
                BenchmarkId::new(format!("{hlabel}_{slabel}"), bytes),
                &body,
                |b, body| {
                    b.iter(|| decode_binary_single_f16(body).unwrap());
                },
            );
        }
    }
    group.finish();
}

fn bench_batch_encode_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("wire_codec/batch_30_layers");
    // Simulate a 30-layer batch (one round trip per forward pass).
    let layers: Vec<usize> = (0..30).collect();
    for &(hidden, hlabel) in HIDDEN_SIZES {
        let seq = 1;
        let residual = make_residual(hidden, seq);
        let bytes = residual.len() * 4;
        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("{hlabel}_30layers"), bytes),
            &(hidden,),
            |b, &(h,)| {
                let r = make_residual(h, 1);
                b.iter(|| encode_binary_request(None, Some(&layers), &r, 1, true, 8092));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_encode_f32_request,
    bench_decode_f32_response,
    bench_decode_f16_response,
    bench_batch_encode_decode,
);
criterion_main!(benches);
