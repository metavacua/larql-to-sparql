//! CPU-1B probe: can BF16-resident weights be consumed fast enough to be
//! worth halving the traffic?
//!
//! Asked BEFORE building residency, because if no BF16 consumption path
//! reaches the f32 BLAS rate, storing BF16 buys nothing and CPU-1B1 would
//! be residency for its own sake.
//!
//! The metric that matters is **effective STORED-weight GB/s** — bytes
//! actually read from RAM. An f32 path reads 4 bytes per weight; a BF16
//! path reads 2. A BF16 kernel at half the f32 GB/s is BREAK-EVEN in
//! time and still halves resident memory.
//!
//! ```text
//! QW_BF16_BENCH=1 cargo test --release bf16_gemv_bench -- --nocapture
//! ```

use std::time::Instant;

use crate::format::vindex3::fixtures::lcg_values;

/// `(u16 as u32) << 16` reinterpreted — bf16 is the top half of f32, so
/// widening is exact and needs no table.
#[inline(always)]
fn widen(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

#[inline(always)]
fn narrow(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

/// Widen a tile of rows into scratch, then one BLAS call over the tile.
///
/// The point of tiling: the bf16 read comes from RAM (half the bytes),
/// while the widened f32 stays in cache. A whole-matrix widen would put
/// the f32 copy back in RAM and lose the entire benefit.
fn tiled_blas(w: &[u16], x: &[f32], out_dim: usize, in_dim: usize, rows: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    let mut scratch = vec![0.0f32; rows * in_dim];
    let mut base = 0;
    while base < out_dim {
        let n = rows.min(out_dim - base);
        let src = &w[base * in_dim..(base + n) * in_dim];
        for (d, s) in scratch[..n * in_dim].iter_mut().zip(src) {
            *d = widen(*s);
        }
        let part =
            larql_compute::cpu::ops::moe::math::matmul_vec(x, &scratch[..n * in_dim], n, in_dim);
        y[base..base + n].copy_from_slice(&part);
        base += n;
    }
    y
}

/// **The candidate.** Fused: load BF16, widen in REGISTERS, multiply by
/// the f32 activation, accumulate in f32. No scratch matrix, so the only
/// bytes crossing RAM are the 2-per-weight actually stored.
///
/// The widen is exact — bf16 is the top half of f32, so `(bits as u32)
/// << 16` reproduces the value with no rounding and no table. The
/// activation stays f32 and the accumulator stays f32, which keeps this
/// a change of REPRESENTATION AND MECHANICS ONLY: no stored weight and
/// no residual-stream value takes a different numerical value than the
/// f32 path would give. Rounding activations to bf16 would be a second,
/// separate precision decision and is deliberately not made here.
///
/// Four accumulators to hide FMA latency; the tail is scalar.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fused_bf16_dot(w: &[u16], x: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = x.len();
    let (wp, xp) = (w.as_ptr(), x.as_ptr());
    let (mut a0, mut a1) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    let (mut a2, mut a3) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
    let mut i = 0usize;
    while i + 16 <= n {
        let w0 = vld1q_u16(wp.add(i));
        let w1 = vld1q_u16(wp.add(i + 8));
        let f0 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(w0)), 16));
        let f1 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(w0)), 16));
        let f2 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(w1)), 16));
        let f3 = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(w1)), 16));
        a0 = vfmaq_f32(a0, f0, vld1q_f32(xp.add(i)));
        a1 = vfmaq_f32(a1, f1, vld1q_f32(xp.add(i + 4)));
        a2 = vfmaq_f32(a2, f2, vld1q_f32(xp.add(i + 8)));
        a3 = vfmaq_f32(a3, f3, vld1q_f32(xp.add(i + 12)));
        i += 16;
    }
    let mut acc = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    while i < n {
        acc += f32::from_bits((*w.get_unchecked(i) as u32) << 16) * *x.get_unchecked(i);
        i += 1;
    }
    acc
}

#[cfg(target_arch = "aarch64")]
fn fused_bf16(w: &[u16], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    (0..out_dim)
        .map(|o| unsafe { fused_bf16_dot(&w[o * in_dim..(o + 1) * in_dim], x) })
        .collect()
}

/// The same kernel with output rows across workers — CPU-1B0 showed
/// threading saturates at two on the f32 path, so this asks whether a
/// kernel reading HALF the bytes has more headroom.
#[cfg(target_arch = "aarch64")]
fn fused_bf16_threaded(
    w: &[u16],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
    workers: usize,
) -> Vec<f32> {
    use rayon::prelude::*;
    let rows = out_dim.div_ceil(workers);
    w.par_chunks(rows * in_dim)
        .flat_map_iter(|slab| {
            (0..slab.len() / in_dim)
                .map(|o| unsafe { fused_bf16_dot(&slab[o * in_dim..(o + 1) * in_dim], x) })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Scalar dot with widening inlined — no scratch at all.
fn direct_scalar(w: &[u16], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    (0..out_dim)
        .map(|o| {
            let row = &w[o * in_dim..(o + 1) * in_dim];
            let mut acc = 0.0f32;
            for (b, v) in row.iter().zip(x) {
                acc += widen(*b) * v;
            }
            acc
        })
        .collect()
}

/// f32 GEMV with OUTPUT ROWS partitioned across workers.
///
/// The decode-shaped parallel axis: rows are independent, each worker
/// streams a contiguous slab of weights, and there is no reduction. The
/// position axis the batch path uses vanishes at batch 1, so this is the
/// axis that replaces it.
fn threaded_f32(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize, workers: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let rows = out_dim.div_ceil(workers);
    w.par_chunks(rows * in_dim)
        .flat_map_iter(|slab| {
            let n = slab.len() / in_dim;
            larql_compute::cpu::ops::moe::math::matmul_vec(x, slab, n, in_dim)
        })
        .collect()
}

fn bench_shape(label: &str, out_dim: usize, in_dim: usize) {
    let f32w = lcg_values(out_dim * in_dim, 11);
    // Round-trip through bf16 so both paths see the SAME values and a
    // difference is the kernel, not the data.
    let bf: Vec<u16> = f32w.iter().map(|v| narrow(*v)).collect();
    let f32w: Vec<f32> = bf.iter().map(|b| widen(*b)).collect();
    let x = lcg_values(in_dim, 22);

    let stored_bytes = (out_dim * in_dim * 2) as f64; // BF16 bytes
    let f32_bytes = (out_dim * in_dim * 4) as f64;
    let iters = (1_500_000_000.0 / (out_dim * in_dim) as f64).clamp(3.0, 200.0) as usize;

    let mut sink = 0.0f32;
    let t = Instant::now();
    for _ in 0..iters {
        sink += larql_compute::cpu::ops::moe::math::matmul_vec(&x, &f32w, out_dim, in_dim)[0];
    }
    let base = t.elapsed().as_secs_f64() / iters as f64;

    let mut rows = [64usize, 256, 1024];
    let mut best = (f64::MAX, 0usize);
    for r in rows.iter_mut() {
        let t = Instant::now();
        for _ in 0..iters {
            sink += tiled_blas(&bf, &x, out_dim, in_dim, *r)[0];
        }
        let dt = t.elapsed().as_secs_f64() / iters as f64;
        if dt < best.0 {
            best = (dt, *r);
        }
    }

    let t = Instant::now();
    for _ in 0..iters.min(5) {
        sink += direct_scalar(&bf, &x, out_dim, in_dim)[0];
    }
    let scal = t.elapsed().as_secs_f64() / iters.min(5) as f64;

    // Is the generic SGEMV already using the machine, or one core?
    let mut thr = Vec::new();
    for workers in [2usize, 4, 8, 12] {
        let t = Instant::now();
        for _ in 0..iters {
            sink += threaded_f32(&f32w, &x, out_dim, in_dim, workers)[0];
        }
        let dt = t.elapsed().as_secs_f64() / iters as f64;
        thr.push((workers, dt, f32_bytes / dt / 1e9));
    }

    println!(
        "  {label:20} {out_dim:>6}x{in_dim:<5}\n     \
         f32 blas      {:7.2} ms   {:6.1} GB/s read\n     \
         bf16 tiled    {:7.2} ms   {:6.1} GB/s read   (tile {:4})   {:5.2}x vs f32\n     \
         bf16 scalar   {:7.2} ms   {:6.1} GB/s read",
        base * 1e3,
        f32_bytes / base / 1e9,
        best.0 * 1e3,
        stored_bytes / best.0 / 1e9,
        best.1,
        base / best.0,
        scal * 1e3,
        stored_bytes / scal / 1e9,
    );
    #[cfg(target_arch = "aarch64")]
    {
        let t = Instant::now();
        for _ in 0..iters {
            sink += fused_bf16(&bf, &x, out_dim, in_dim)[0];
        }
        let f1 = t.elapsed().as_secs_f64() / iters as f64;
        println!(
            "     bf16 fused x1   {:7.2} ms   {:6.1} GB/s read   {:5.2}x vs f32 blas",
            f1 * 1e3,
            stored_bytes / f1 / 1e9,
            base / f1
        );
        for workers in [2usize, 4, 8, 12] {
            let t = Instant::now();
            for _ in 0..iters {
                sink += fused_bf16_threaded(&bf, &x, out_dim, in_dim, workers)[0];
            }
            let dt = t.elapsed().as_secs_f64() / iters as f64;
            println!(
                "     bf16 fused x{workers:<2}  {:7.2} ms   {:6.1} GB/s read   {:5.2}x vs f32 blas",
                dt * 1e3,
                stored_bytes / dt / 1e9,
                base / dt
            );
        }
        // Exactness: the widen must reproduce the f32 path bit-for-bit
        // up to summation order alone.
        let a = fused_bf16(&bf, &x, out_dim, in_dim);
        let b = larql_compute::cpu::ops::moe::math::matmul_vec(&x, &f32w, out_dim, in_dim);
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (p, q) in a.iter().zip(&b) {
            num += (*p as f64 - *q as f64).powi(2);
            den += (*q as f64).powi(2);
        }
        println!(
            "     fused vs f32 blas  rel_rms {:.3e}  (summation order only)",
            (num / den.max(f64::MIN_POSITIVE)).sqrt()
        );
    }
    for (n, dt, gbs) in &thr {
        println!(
            "     f32 x{n:<2}         {:7.2} ms   {:6.1} GB/s read   {:5.2}x vs 1 thread",
            dt * 1e3,
            gbs,
            base / dt
        );
    }
    std::hint::black_box(sink);
}

#[test]
fn bf16_gemv_bench() {
    if std::env::var("QW_BF16_BENCH").is_err() {
        eprintln!("SKIP bf16_gemv_bench: set QW_BF16_BENCH=1");
        return;
    }
    println!(
        "\n  BF16-resident GEMV candidates — 'GB/s read' is bytes actually\n  \
              streamed from RAM, so f32 and bf16 rows are directly comparable.\n"
    );
    for (l, o, i) in [
        ("delta in_proj_qkv", 10240usize, 5120usize),
        ("delta out_proj", 5120, 6144),
        ("ffn gate/up", 17408, 5120),
    ] {
        bench_shape(l, o, i);
    }
    println!();
}
