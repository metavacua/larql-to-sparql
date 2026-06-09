//! Cross-backend, cross-format quant matvec benchmarks.
//!
//! Each format × shape × backend combination shows up as one Criterion
//! sample so HTML reports under `target/criterion/` give a side-by-side
//! comparison. The 75 %-row drop bug in `q4_matvec_v4` (closed
//! 2026-04-25) would have shown up here as a 4× throughput cliff
//! between CPU and Metal at the lm-head shape, *weeks* before goldens
//! caught it. This is what these benches exist for.
//!
//! Run: `cargo bench -p larql-compute --bench quant_matvec`
//! Or with metal: `cargo bench -p larql-compute --features gpu --bench quant_matvec`
//!
//! ## What's covered
//!
//! - **Formats**: Q4_0, Q4_K, Q4_KF, Q6_K (Q8_0 internally aliases
//!   Q4_0 in `quant_matvec`'s default impl).
//! - **Shapes**: three reference shapes, named after their role in
//!   Gemma 3 4B (hidden=2560):
//!   - `decode_2560`: square N=2560 × K=2560. Per-token, hot path.
//!   - `prefill_10240`: N=10240 × K=2560. FFN gate/up matrix shape.
//!   - `lm_head_262144`: N=262144 × K=2560. Vocab projection — the
//!     row-drop regression-detector shape.
//! - **Backends**: CPU always; Metal under `--features gpu`.

extern crate blas_src;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use larql_compute::cpu::ops::q4_common::{
    quantize_q4_0, quantize_q4_k, quantize_q4_kf, quantize_q6_k,
};
use larql_compute::{ComputeBackend, CpuBackend, QuantFormat};

/// Three reference shapes — see module docs for their roles.
struct Shape {
    name: &'static str,
    n: usize,
    k: usize,
}

const SHAPES: &[Shape] = &[
    Shape {
        name: "decode_2560",
        n: 2_560,
        k: 2_560,
    },
    Shape {
        name: "prefill_10240",
        n: 10_240,
        k: 2_560,
    },
    Shape {
        name: "lm_head_262144",
        n: 262_144,
        k: 2_560,
    },
];

/// Q4_K / Q6_K / Q4_KF require both N×K to be a multiple of the
/// super-block size (256) along K. All shapes here use K=2560 so this
/// holds; Q4_0 also uses K=2560 (multiple of 32).
fn synth_inputs(n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut w = Vec::with_capacity(n * k);
    for i in 0..n * k {
        let f = i as f32;
        w.push(((f * 0.0001).sin() + 0.3 * (f * 0.00037).cos()) * 0.05);
    }
    let x: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
    (w, x)
}

/// Run `bench_fn` for one (format × shape × backend) cell.
fn add_cell<B: ComputeBackend>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    backend: &B,
    backend_label: &str,
    format: QuantFormat,
    shape: &Shape,
    weights: &[u8],
    x: &[f32],
) {
    let id = format!("{}/{}", backend_label, shape.name);
    group.bench_with_input(
        BenchmarkId::from_parameter(&id),
        &(weights, x),
        |b, (w, x)| {
            b.iter(|| backend.quant_matvec(format, w, x, shape.n, shape.k));
        },
    );
}

fn bench_format(
    c: &mut Criterion,
    format: QuantFormat,
    quantize: impl Fn(&[f32]) -> Vec<u8>,
    group_name: &str,
) {
    let mut group = c.benchmark_group(group_name);
    // The lm_head_262144 cell is multi-second; keep sample size modest
    // so the suite finishes in reasonable time.
    group.sample_size(20);

    let cpu = CpuBackend;

    #[cfg(target_os = "macos")]
    let metal = larql_compute_metal::MetalBackend::new();
    #[cfg(target_os = "macos")]
    if let Some(ref m) = metal {
        m.set_flop_threshold(1);
    }

    for shape in SHAPES {
        let (w_f32, x) = synth_inputs(shape.n, shape.k);
        let weights = quantize(&w_f32);

        // Throughput in elements/sec is more useful than time/iter for
        // comparing across shapes.
        group.throughput(Throughput::Elements((shape.n * shape.k) as u64));

        add_cell(&mut group, &cpu, "cpu", format, shape, &weights, &x);

        #[cfg(target_os = "macos")]
        if let Some(ref m) = metal {
            add_cell(&mut group, m, "metal", format, shape, &weights, &x);
        }
    }
    group.finish();
}

fn bench_q4_0(c: &mut Criterion) {
    bench_format(c, QuantFormat::Q4_0, quantize_q4_0, "quant_matvec_q4_0");
}
fn bench_q4_k(c: &mut Criterion) {
    bench_format(c, QuantFormat::Q4_K, quantize_q4_k, "quant_matvec_q4_k");
}
fn bench_q4_kf(c: &mut Criterion) {
    bench_format(c, QuantFormat::Q4_KF, quantize_q4_kf, "quant_matvec_q4_kf");
}
fn bench_q6_k(c: &mut Criterion) {
    bench_format(c, QuantFormat::Q6_K, quantize_q6_k, "quant_matvec_q6_k");
}

/// Head-to-head Q6_K CPU: AVX2 Q8K-input (`q6k_q8k_matvec_into`,
/// dispatched to AVX2 on x86_64 + avx2 feature) vs scalar f32-input
/// (`q6k_matvec::dispatch`, the `CpuBackend::q6k_matvec` trait body).
/// Both reach production callers — the AVX2 path is hot for the
/// walk-ffn-q8k Q6_K branch (#103) and the f32 path is hot for
/// attention V-projection and lm-head KNN.
///
/// The activation quant cost is amortised across `n` rows in a real
/// FFN step (the caller quantises `x` once per layer, not per row),
/// so this bench pre-quantises `x` outside the `iter` closure — same
/// as the production hot path.
fn bench_q6_k_avx2_vs_scalar(c: &mut Criterion) {
    use larql_compute::cpu::ops::q4k_q8k_dot::{q6k_q8k_matvec_into, quantize_x_to_q8k};
    use larql_compute::cpu::ops::q6k_matvec;

    let mut group = c.benchmark_group("q6k_q8k_vs_q6k_f32");
    group.sample_size(20);

    for shape in SHAPES {
        let (w_f32, x) = synth_inputs(shape.n, shape.k);
        let weights = quantize_q6_k(&w_f32);
        let q8k_x = quantize_x_to_q8k(&x);

        group.throughput(Throughput::Elements((shape.n * shape.k) as u64));

        // AVX2 Q8K-input — the walk-ffn-q8k hot path.
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("avx2_q8k_input/{}", shape.name)),
            &(&weights, &q8k_x),
            |b, (w, q8k)| {
                let mut out = vec![0.0f32; shape.n];
                b.iter(|| {
                    q6k_q8k_matvec_into(&mut out, q8k, w, shape.n, shape.k);
                });
            },
        );

        // f32-input scalar — the trait-dispatched path used by attention
        // V projection and lm-head KNN.
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("scalar_f32_input/{}", shape.name)),
            &(&weights, &x),
            |b, (w, xv)| {
                b.iter(|| q6k_matvec::dispatch(w, xv, shape.n, shape.k));
            },
        );
    }
    group.finish();
}

/// Validate that `q4k_q8k_gate_up_into` (the fused FFN gate+up entry
/// point called from `q4k_ffn_forward_layer_q8k::104`) reaches the
/// AVX2 fast path on x86_64 after #112. Compares against two
/// sequential `q4k_q8k_matvec_into` calls — the new fallback should
/// be within noise of the explicit-twice form (both end up running
/// AVX2 twice on x86_64), and both should be ~18× faster than the
/// prior `q4k_q8k_matvec_scalar` ×2 path.
fn bench_q4_k_gate_up_fused_vs_split(c: &mut Criterion) {
    use larql_compute::cpu::ops::q4k_q8k_dot::{
        q4k_q8k_gate_up_into, q4k_q8k_matvec_into, q4k_q8k_matvec_scalar, quantize_x_to_q8k,
    };

    let mut group = c.benchmark_group("q4k_gate_up_fused");
    group.sample_size(20);

    // Only the FFN gate/up shapes make sense here — `decode_2560` is
    // a square attention-style shape, and `lm_head_262144` doesn't
    // have a fused gate/up. Stick to `prefill_10240` (Gemma 3 4B FFN
    // gate/up: 10240 × 2560) — the shape walk-ffn-q8k actually hits.
    let shape = &SHAPES[1]; // prefill_10240
    let (w_f32, x) = synth_inputs(shape.n, shape.k);
    let gate = quantize_q4_k(&w_f32);
    let up = quantize_q4_k(&w_f32);
    let q8k_x = quantize_x_to_q8k(&x);

    group.throughput(Throughput::Elements((2 * shape.n * shape.k) as u64));

    group.bench_with_input(
        BenchmarkId::from_parameter("fused_into"),
        &(&gate, &up, &q8k_x),
        |b, (g, u, q8k)| {
            let mut gate_out = vec![0.0f32; shape.n];
            let mut up_out = vec![0.0f32; shape.n];
            b.iter(|| {
                q4k_q8k_gate_up_into(&mut gate_out, &mut up_out, q8k, g, u, shape.n, shape.k);
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::from_parameter("two_matvec_into"),
        &(&gate, &up, &q8k_x),
        |b, (g, u, q8k)| {
            let mut gate_out = vec![0.0f32; shape.n];
            let mut up_out = vec![0.0f32; shape.n];
            b.iter(|| {
                q4k_q8k_matvec_into(&mut gate_out, q8k, g, shape.n, shape.k);
                q4k_q8k_matvec_into(&mut up_out, q8k, u, shape.n, shape.k);
            });
        },
    );

    // Document the prior scalar baseline that #112 replaced.
    group.bench_with_input(
        BenchmarkId::from_parameter("two_matvec_scalar_pre_pr112"),
        &(&gate, &up, &q8k_x),
        |b, (g, u, q8k)| {
            let mut gate_out = vec![0.0f32; shape.n];
            let mut up_out = vec![0.0f32; shape.n];
            b.iter(|| {
                q4k_q8k_matvec_scalar(&mut gate_out, q8k, g, shape.n, shape.k);
                q4k_q8k_matvec_scalar(&mut up_out, q8k, u, shape.n, shape.k);
            });
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_q4_0,
    bench_q4_k,
    bench_q4_kf,
    bench_q6_k,
    bench_q6_k_avx2_vs_scalar,
    bench_q4_k_gate_up_fused_vs_split,
);
criterion_main!(benches);
