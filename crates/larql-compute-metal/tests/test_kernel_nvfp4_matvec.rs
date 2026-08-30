//! Does the NVFP4 kernel execute the format, or merely something shaped
//! like it?
//!
//! VINDEX3-Q2 rests on the claim that NVFP4 and MXFP4 differ *only* in
//! the scale geometry. That claim is worth nothing if the kernel reads
//! the scale stream at the wrong offset, drops the tensor scale, or
//! flushes subnormal group scales to zero — each of which produces
//! finite, plausible numbers that a bare CPU-vs-GPU comparison would
//! wave through on a badly chosen fixture.
//!
//! So the arms are:
//!
//! | arm | what it runs | expectation |
//! |---|---|---|
//! | 1 | CPU reference: dequantise then dot | the oracle |
//! | 2 | `nvfp4_gemv` | agrees to fp reduction-order error |
//! | 3 | kernel with a perturbed tensor scale | materially divergent |
//! | 4 | kernel on a matrix whose quiet groups are E4M3 subnormal | agrees |
//!
//! Arm 3 is the load-bearing one. Without it, arms 1 and 2 agreeing shows
//! only that two paths compute *something* consistently — which they
//! would also do if both ignored the tensor scale. Arm 3 is what makes
//! the agreement mean "the scale was read".
//!
//! ## What the fixture does on purpose
//!
//! - **Group amaxes vary by orders of magnitude across a row**, so the
//!   E4M3 scale stream carries genuinely different bytes per group. With
//!   one scale everywhere, a kernel indexing the scale stream wrongly
//!   would still get the right answer and this file would prove nothing.
//! - **`K` spans many groups and rows are not a multiple of
//!   `ROWS_PER_TG`**, so the tail-row guard is exercised rather than
//!   assumed.
//! - **Arm 4's spread pushes quiet groups into E4M3 subnormals.** The
//!   tensor scale pins the loudest group at E4M3's top by construction,
//!   so a wide spread necessarily drives the quietest groups below
//!   `2^-6`. A kernel that flushed those to zero would delete whole
//!   groups of weights and still return finite numbers.

#![cfg(target_os = "macos")]

mod common;

use larql_compute::backend::matmul::MatMul;
use larql_models::quant::nvfp4::{self, NVFP4_GROUP_ELEMS};

/// Deliberately not a multiple of the kernel's 4 rows per threadgroup.
const ROWS: usize = 13;
/// 24 groups per row.
const K: usize = NVFP4_GROUP_ELEMS * 24;

/// Deterministic pseudo-random spread whose per-group amax varies by
/// orders of magnitude across the row.
fn fixture(rows: usize, k: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    (0..rows * k)
        .map(|i| {
            let group = (i % k) / NVFP4_GROUP_ELEMS;
            // Each group two octaves quieter than the last, wrapping, so
            // one row spans a wide dynamic range.
            let octave = (group % 12) as i32;
            next() * (2.0f32).powi(-2 * octave)
        })
        .collect()
}

fn cpu_reference(matrix: &nvfp4::Nvfp4Matrix, rows: usize, k: usize, x: &[f32]) -> Vec<f32> {
    let mut weights = vec![0.0f32; rows * k];
    nvfp4::dequantize_into(matrix, rows, k, &mut weights).expect("dequantise");
    (0..rows)
        .map(|r| {
            weights[r * k..(r + 1) * k]
                .iter()
                .zip(x)
                .map(|(w, v)| w * v)
                .sum()
        })
        .collect()
}

/// Relative error against the reference's own scale, so a row whose dot
/// product is naturally tiny is not judged against an absolute epsilon.
fn rel_error(reference: &[f32], got: &[f32]) -> f32 {
    let scale = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if scale == 0.0 {
        return 0.0;
    }
    reference
        .iter()
        .zip(got)
        .map(|(r, g)| (r - g).abs())
        .fold(0.0f32, f32::max)
        / scale
}

#[test]
fn nvfp4_kernel_matches_the_cpu_reference_and_notices_a_wrong_tensor_scale() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let values = fixture(ROWS, K, 7);
    let x: Vec<f32> = (0..K).map(|i| ((i as f32) * 0.017).sin()).collect();
    let matrix = nvfp4::quantize(&values, ROWS, K).expect("quantise");

    // The fixture must actually exercise distinct scales, or the arms
    // below prove nothing about scale indexing.
    let distinct: std::collections::BTreeSet<u8> = matrix.scales.iter().copied().collect();
    assert!(
        distinct.len() >= 8,
        "fixture must span many E4M3 scales, got {}",
        distinct.len()
    );

    // ── Arm 1: the oracle ────────────────────────────────────────────
    let reference = cpu_reference(&matrix, ROWS, K, &x);

    // ── Arm 2: the kernel ────────────────────────────────────────────
    let got = gpu
        .nvfp4_gemv(
            &matrix.packed,
            &matrix.scales,
            matrix.tensor_scale,
            &x,
            ROWS,
            K,
        )
        .expect("Metal backend must have an NVFP4 kernel");
    assert_eq!(got.len(), ROWS);
    assert!(
        got.iter().all(|v| v.is_finite()),
        "kernel produced non-finite output"
    );

    let err = rel_error(&reference, &got);
    assert!(
        err < 1e-4,
        "kernel disagrees with the CPU reference by {err} (reduction-order error should be ~1e-6)"
    );

    // ── Arm 3: the control ───────────────────────────────────────────
    // Halve the tensor scale. Every output must halve with it; if the
    // kernel ignored the scale, arm 2's agreement would have been empty.
    let perturbed = gpu
        .nvfp4_gemv(
            &matrix.packed,
            &matrix.scales,
            matrix.tensor_scale * 0.5,
            &x,
            ROWS,
            K,
        )
        .expect("kernel");
    let control_err = rel_error(&reference, &perturbed);
    assert!(
        control_err > 0.1,
        "a halved tensor scale must change the result materially; got {control_err} — \
         arm 2's agreement does not demonstrate the scale was read"
    );
    for (r, p) in reference.iter().zip(&perturbed) {
        assert!(
            (r * 0.5 - p).abs() <= r.abs() * 1e-4 + 1e-6,
            "halving the tensor scale must halve the output exactly: {r} vs {p}"
        );
    }
}

/// Arm 4: quiet groups land in E4M3's subnormal range, and the kernel
/// must decode them rather than flush to zero.
#[test]
fn nvfp4_kernel_decodes_subnormal_group_scales() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    // One very loud group forces the tensor scale up, pushing the rest
    // of the row's scales into E4M3 subnormals.
    let mut values = fixture(4, K, 11);
    values[0] = 4096.0;
    let matrix = nvfp4::quantize(&values, 4, K).expect("quantise");

    let subnormals = matrix
        .scales
        .iter()
        .filter(|&&b| (b & 0x78) == 0 && (b & 0x07) != 0)
        .count();
    assert!(
        subnormals > 0,
        "fixture must actually produce subnormal E4M3 scales, or this arm tests nothing"
    );

    let x: Vec<f32> = (0..K).map(|i| ((i as f32) * 0.029).cos()).collect();
    let reference = cpu_reference(&matrix, 4, K, &x);
    let got = gpu
        .nvfp4_gemv(
            &matrix.packed,
            &matrix.scales,
            matrix.tensor_scale,
            &x,
            4,
            K,
        )
        .expect("kernel");

    let err = rel_error(&reference, &got);
    assert!(
        err < 1e-4,
        "kernel disagrees on subnormal group scales by {err} — flushing them to zero \
         would delete whole groups of weights while still returning finite numbers"
    );
}

/// Several matrices in one submission return the same values as one at a
/// time, so the batched path cannot drift from the single-shot one.
#[test]
fn nvfp4_gemv_multi_matches_single_dispatches() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let x: Vec<f32> = (0..K).map(|i| ((i as f32) * 0.011).sin()).collect();
    let matrices: Vec<nvfp4::Nvfp4Matrix> = (0..3)
        .map(|s| nvfp4::quantize(&fixture(ROWS, K, 3 + s), ROWS, K).expect("quantise"))
        .collect();

    let batched = gpu
        .nvfp4_gemv_multi(
            &matrices
                .iter()
                .map(|m| {
                    (
                        m.packed.as_slice(),
                        m.scales.as_slice(),
                        m.tensor_scale,
                        ROWS,
                        K,
                    )
                })
                .collect::<Vec<_>>(),
            &x,
        )
        .expect("kernel");

    for (m, got) in matrices.iter().zip(&batched) {
        let single = gpu
            .nvfp4_gemv(&m.packed, &m.scales, m.tensor_scale, &x, ROWS, K)
            .expect("kernel");
        assert_eq!(
            &single, got,
            "batched and single dispatch must agree exactly"
        );
    }
}

/// Geometry the kernel cannot serve is refused, not silently truncated.
#[test]
fn nvfp4_kernel_refuses_unaligned_geometry() {
    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let matrix = nvfp4::quantize(&fixture(4, K, 5), 4, K).expect("quantise");
    let x = vec![0.0f32; K];

    // K not a whole number of groups.
    assert!(gpu
        .nvfp4_gemv(
            &matrix.packed,
            &matrix.scales,
            matrix.tensor_scale,
            &x,
            4,
            K - 1
        )
        .is_none());
    // Short input vector.
    assert!(gpu
        .nvfp4_gemv(
            &matrix.packed,
            &matrix.scales,
            matrix.tensor_scale,
            &x[..K - NVFP4_GROUP_ELEMS],
            4,
            K
        )
        .is_none());
    // Scale stream too short for the declared rows.
    assert!(gpu
        .nvfp4_gemv(
            &matrix.packed,
            &matrix.scales[..matrix.scales.len() / 2],
            matrix.tensor_scale,
            &x,
            4,
            K
        )
        .is_none());
}
