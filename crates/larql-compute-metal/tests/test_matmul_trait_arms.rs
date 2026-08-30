//! Arm-by-arm contract tests for `impl MatMul for MetalBackend`
//! (`src/trait_impl/matmul.rs`).
//!
//! Each arm is asserted for what it is:
//!
//! | arm | proven by |
//! |---|---|
//! | `f32_gemv` / `f16_gemv` | parity with a CPU loop over the (dequantised) weights; `None` below the FLOP threshold while the `_force` twin runs |
//! | `f16_gemv_multi` | bit-identical to the sequential `f16_gemv_force` calls; empty batch → `Some([])`; short/mismatched operands → `None` |
//! | `wire_resident` | observably a no-op: a gemv after wiring is bit-identical to one before; degenerate inputs return without dispatch |
//! | `mxfp4_gemv(_multi)` / `nvfp4_gemv(_multi)` | parity with the dequantised matrix; `_multi` equals the singles; geometry guards → `None` |
//! | `*_topk1` / `f16_gemv_topk` | equal to the CPU argmax / top-k of the same gemv, on a fixture whose top gaps exceed the tolerance |
//! | `matmul_batch` | each op dispatched through `matmul` or `matmul_transb` by its flag, against ndarray `dot` |
//!
//! Shapes are deliberately awkward (row counts that are not a multiple of
//! the kernels' rows-per-threadgroup) so tail-row guards are exercised.

#![cfg(target_os = "macos")]

extern crate blas_src;

mod matmul_trait_support;

use matmul_trait_support::{
    cpu_argmax, cpu_gemv, cpu_topk, min_gap_between_top, mxfp4_fixture, rel_error, uniform_values,
};

use larql_compute::backend::matmul::{MatMul, MatMulOp};
use larql_compute::prelude::*;
use larql_compute::CpuBackend;
use larql_compute_metal::calibration::MIN_FLOP_FLOOR;
use larql_compute_metal::shaders::f32_gemv::K_TOPK;
use larql_compute_metal::MetalBackend;
use larql_models::quant::half::{decode_f16, encode_f16};
use larql_models::quant::mxfp4::MXFP4_GROUP_ELEMS;
use larql_models::quant::nvfp4::{self, NVFP4_GROUP_ELEMS};
use ndarray::Array2;

/// Row count that is not a multiple of any kernel's rows-per-threadgroup
/// (4 or 8), so the tail-row guard runs.
const ROWS: usize = 37;
/// Reduction length: a multiple of both the MXFP4 (32) and NVFP4 (16)
/// group sizes.
const K: usize = 256;
/// A second row count for the `_multi` batches, so the batch carries
/// matrices of different height.
const ROWS_ALT: usize = 21;
/// Rows needed for `2 * rows * K` to clear `MIN_FLOP_FLOOR` — the lowest
/// threshold `set_flop_threshold` will accept.
const ROWS_ABOVE_FLOOR: usize = MIN_FLOP_FLOOR / (2 * K) + 1;
/// Bytes per f16 element.
const F16_BYTES: usize = 2;
/// Reduction-order tolerance: the GPU accumulates lane-parallel then
/// `simd_sum`; the oracle accumulates left to right. Both in f32.
const REDUCTION_REL_TOL: f32 = 1e-4;
/// Minimum score gap the top-k fixture must exhibit for the argmax /
/// top-k index comparison to be meaningful (well above the tolerance).
const MIN_TOPK_GAP: f32 = 1e-2;
/// `top_k` requested from `f16_gemv_topk` — inside the kernel's
/// per-threadgroup capacity.
const TOPK_REQUESTED: usize = 5;
/// Seeds for the deterministic fixtures.
const SEED_W: u32 = 11;
const SEED_W_ALT: u32 = 13;
const SEED_X: u32 = 17;
/// Matmul geometry for `matmul_batch`.
const MM_M: usize = 3;
const MM_K: usize = 5;
const MM_N: usize = 7;

fn gpu() -> Option<MetalBackend> {
    let gpu = MetalBackend::new();
    if gpu.is_none() {
        eprintln!("no Metal device; skipping");
    }
    gpu
}

fn x_vec() -> Vec<f32> {
    uniform_values(K, SEED_X)
}

fn f32_matrix(rows: usize, seed: u32) -> (Array2<f32>, Vec<f32>) {
    let w = uniform_values(rows * K, seed);
    (Array2::from_shape_vec((rows, K), w.clone()).unwrap(), w)
}

/// f16 weights are compared against their own decoded values, so the
/// comparison isolates the kernel from the f16 rounding of the fixture.
fn f16_matrix(rows: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    let bytes = encode_f16(&uniform_values(rows * K, seed));
    let decoded = decode_f16(&bytes);
    (bytes, decoded)
}

// ───────────────────────── f32 gemv ─────────────────────────

/// `f32_gemv` above the FLOP threshold matches the CPU loop, and the
/// shape used really clears the floor the backend clamps to.
#[test]
fn f32_gemv_above_threshold_matches_cpu_reference() {
    let Some(gpu) = gpu() else { return };
    gpu.set_flop_threshold(MIN_FLOP_FLOOR);
    assert!(2 * ROWS_ABOVE_FLOOR * K >= gpu.flop_threshold());
    let (w, w_flat) = f32_matrix(ROWS_ABOVE_FLOOR, SEED_W);
    let x = x_vec();
    let reference = cpu_gemv(&w_flat, &x, ROWS_ABOVE_FLOOR, K);
    let got = gpu
        .f32_gemv(w.view(), &x)
        .expect("above threshold dispatches");
    assert!(rel_error(&reference, &got) < REDUCTION_REL_TOL);
}

/// Below the threshold `f32_gemv` declines (`None`) while `f32_gemv_force`
/// runs the same kernel and agrees with the CPU loop.
#[test]
fn f32_gemv_declines_below_threshold_while_force_runs() {
    let Some(gpu) = gpu() else { return };
    gpu.set_flop_threshold(MIN_FLOP_FLOOR);
    assert!(2 * ROWS * K < gpu.flop_threshold());
    let (w, w_flat) = f32_matrix(ROWS, SEED_W);
    let x = x_vec();
    assert!(gpu.f32_gemv(w.view(), &x).is_none());
    let reference = cpu_gemv(&w_flat, &x, ROWS, K);
    let got = gpu
        .f32_gemv_force(w.view(), &x)
        .expect("force ignores the threshold");
    assert!(rel_error(&reference, &got) < REDUCTION_REL_TOL);
}

/// Both f32 variants reject an input vector whose length is not `k`.
#[test]
fn f32_gemv_variants_reject_mismatched_input_length() {
    let Some(gpu) = gpu() else { return };
    let (w, _) = f32_matrix(ROWS, SEED_W);
    let short_x = vec![0.0f32; K - 1];
    assert!(gpu.f32_gemv(w.view(), &short_x).is_none());
    assert!(gpu.f32_gemv_force(w.view(), &short_x).is_none());
}

// ───────────────────────── f16 gemv ─────────────────────────

/// `f16_gemv` above threshold and `f16_gemv_force` below it both match
/// the CPU loop over the decoded f16 weights; the gated arm declines on
/// the small shape.
#[test]
fn f16_gemv_matches_decoded_weights_and_honours_threshold_gate() {
    let Some(gpu) = gpu() else { return };
    gpu.set_flop_threshold(MIN_FLOP_FLOOR);
    let x = x_vec();

    let (big, big_dec) = f16_matrix(ROWS_ABOVE_FLOOR, SEED_W);
    let got = gpu
        .f16_gemv(&big, &x, ROWS_ABOVE_FLOOR, K)
        .expect("above threshold dispatches");
    let reference = cpu_gemv(&big_dec, &x, ROWS_ABOVE_FLOOR, K);
    assert!(rel_error(&reference, &got) < REDUCTION_REL_TOL);

    let (small, small_dec) = f16_matrix(ROWS, SEED_W);
    assert!(gpu.f16_gemv(&small, &x, ROWS, K).is_none());
    let forced = gpu
        .f16_gemv_force(&small, &x, ROWS, K)
        .expect("force ignores the threshold");
    let reference = cpu_gemv(&small_dec, &x, ROWS, K);
    assert!(rel_error(&reference, &forced) < REDUCTION_REL_TOL);
}

/// Short weight bytes or a mismatched input length yield `None` on both
/// f16 variants.
#[test]
fn f16_gemv_variants_reject_short_weights_and_mismatched_input() {
    let Some(gpu) = gpu() else { return };
    let (w, _) = f16_matrix(ROWS, SEED_W);
    let x = x_vec();
    let short_w = &w[..w.len() - F16_BYTES];
    let short_x = &x[..K - 1];
    assert!(gpu.f16_gemv(short_w, &x, ROWS, K).is_none());
    assert!(gpu.f16_gemv_force(short_w, &x, ROWS, K).is_none());
    assert!(gpu.f16_gemv(&w, short_x, ROWS, K).is_none());
    assert!(gpu.f16_gemv_force(&w, short_x, ROWS, K).is_none());
}

/// One submission of N dispatches is bit-identical to N sequential
/// `f16_gemv_force` calls, and each agrees with the CPU loop.
#[test]
fn f16_gemv_multi_is_bit_identical_to_sequential_force_gemvs() {
    let Some(gpu) = gpu() else { return };
    let x = x_vec();
    let (w_a, dec_a) = f16_matrix(ROWS, SEED_W);
    let (w_b, dec_b) = f16_matrix(ROWS_ALT, SEED_W_ALT);
    let batch = gpu
        .f16_gemv_multi(&[(&w_a, ROWS, K), (&w_b, ROWS_ALT, K)], &x)
        .expect("valid batch dispatches");
    assert_eq!(batch.len(), 2);
    let single_a = gpu.f16_gemv_force(&w_a, &x, ROWS, K).unwrap();
    let single_b = gpu.f16_gemv_force(&w_b, &x, ROWS_ALT, K).unwrap();
    assert_eq!(batch[0], single_a, "batch[0] must be bit-identical");
    assert_eq!(batch[1], single_b, "batch[1] must be bit-identical");
    assert!(rel_error(&cpu_gemv(&dec_a, &x, ROWS, K), &batch[0]) < REDUCTION_REL_TOL);
    assert!(rel_error(&cpu_gemv(&dec_b, &x, ROWS_ALT, K), &batch[1]) < REDUCTION_REL_TOL);
}

/// An empty batch is `Some(empty)`; any short or mismatched operand
/// fails the whole batch with `None` before anything is dispatched.
#[test]
fn f16_gemv_multi_empty_batch_and_operand_guards() {
    let Some(gpu) = gpu() else { return };
    let x = x_vec();
    assert_eq!(gpu.f16_gemv_multi(&[], &x), Some(Vec::new()));
    let (w, _) = f16_matrix(ROWS, SEED_W);
    let short_w = &w[..w.len() - F16_BYTES];
    assert!(gpu
        .f16_gemv_multi(&[(&w, ROWS, K), (short_w, ROWS, K)], &x)
        .is_none());
    assert!(gpu.f16_gemv_multi(&[(&w, ROWS, K)], &x[..K - 1]).is_none());
}

// ───────────────────────── wire_resident ─────────────────────────

/// Wiring a set of weight buffers changes no number: the gemv over the
/// first buffer is bit-identical before and after, and the kernel's
/// scratch 1×1 dispatch leaks nothing into the caller's bytes.
#[test]
fn wire_resident_is_a_no_op_on_numbers() {
    let Some(gpu) = gpu() else { return };
    let x = x_vec();
    let (w_a, _) = f16_matrix(ROWS, SEED_W);
    let (w_b, _) = f16_matrix(ROWS_ALT, SEED_W_ALT);
    let before = gpu.f16_gemv_force(&w_a, &x, ROWS, K).unwrap();
    let w_a_bytes_before = w_a.clone();
    gpu.wire_resident(&[&w_a, &w_b]);
    assert_eq!(w_a, w_a_bytes_before, "wiring must not write the weights");
    let after = gpu.f16_gemv_force(&w_a, &x, ROWS, K).unwrap();
    assert_eq!(before, after, "gemv must be bit-identical across wiring");
}

/// Degenerate inputs — no buffers, or a first buffer shorter than one
/// f16 element — return without dispatching and without panicking.
#[test]
fn wire_resident_returns_early_on_degenerate_inputs() {
    let Some(gpu) = gpu() else { return };
    gpu.wire_resident(&[]);
    let one_byte = [0u8; 1];
    gpu.wire_resident(&[&one_byte]);
}

// ───────────────────────── MXFP4 ─────────────────────────

/// `mxfp4_gemv` agrees with the CPU loop over the dequantised matrix, on
/// a fixture whose nibble stream spans every code and whose scales differ
/// per group.
#[test]
fn mxfp4_gemv_matches_dequantised_reference() {
    let Some(gpu) = gpu() else { return };
    let x = x_vec();
    let fx = mxfp4_fixture(ROWS, K, SEED_W);
    let reference = cpu_gemv(&fx.dequantised, &x, ROWS, K);
    let got = gpu
        .mxfp4_gemv(&fx.packed, &fx.scales, &x, ROWS, K)
        .expect("aligned geometry dispatches");
    assert!(rel_error(&reference, &got) < REDUCTION_REL_TOL);
}

/// The MXFP4 batch equals the single-matrix calls element for element
/// (same kernel, same arguments) and an empty batch is `Some(empty)`.
#[test]
fn mxfp4_gemv_multi_equals_single_gemvs() {
    let Some(gpu) = gpu() else { return };
    let x = x_vec();
    let fa = mxfp4_fixture(ROWS, K, SEED_W);
    let fb = mxfp4_fixture(ROWS_ALT, K, SEED_W_ALT);
    let batch = gpu
        .mxfp4_gemv_multi(
            &[
                (&fa.packed, &fa.scales, ROWS, K),
                (&fb.packed, &fb.scales, ROWS_ALT, K),
            ],
            &x,
        )
        .expect("valid batch dispatches");
    let single_a = gpu.mxfp4_gemv(&fa.packed, &fa.scales, &x, ROWS, K).unwrap();
    let single_b = gpu
        .mxfp4_gemv(&fb.packed, &fb.scales, &x, ROWS_ALT, K)
        .unwrap();
    assert_eq!(batch, vec![single_a, single_b]);
    assert_eq!(gpu.mxfp4_gemv_multi(&[], &x), Some(Vec::new()));
}

/// Unaligned `k`, a short input, short packed bytes or short scales each
/// yield `None` rather than an out-of-bounds read.
#[test]
fn mxfp4_gemv_rejects_bad_geometry() {
    let Some(gpu) = gpu() else { return };
    let x = x_vec();
    let fx = mxfp4_fixture(ROWS, K, SEED_W);
    let unaligned_k = K - 1;
    assert!(gpu
        .mxfp4_gemv(&fx.packed, &fx.scales, &x, ROWS, unaligned_k)
        .is_none());
    assert!(gpu
        .mxfp4_gemv(&fx.packed, &fx.scales, &x[..K - MXFP4_GROUP_ELEMS], ROWS, K)
        .is_none());
    assert!(gpu
        .mxfp4_gemv(&fx.packed[..fx.packed.len() - 1], &fx.scales, &x, ROWS, K)
        .is_none());
    assert!(gpu
        .mxfp4_gemv(&fx.packed, &fx.scales[..fx.scales.len() - 1], &x, ROWS, K)
        .is_none());
}

// ───────────────────────── NVFP4 ─────────────────────────

fn nvfp4_matrix(rows: usize, seed: u32) -> (nvfp4::Nvfp4Matrix, Vec<f32>) {
    let m = nvfp4::quantize(&uniform_values(rows * K, seed), rows, K).expect("quantise");
    let mut dequantised = vec![0.0f32; rows * K];
    nvfp4::dequantize_into(&m, rows, K, &mut dequantised).expect("dequantise");
    (m, dequantised)
}

/// `nvfp4_gemv` agrees with the dequantised matrix; the batch equals the
/// singles; an empty batch is `Some(empty)`.
#[test]
fn nvfp4_gemv_and_multi_match_dequantised_reference() {
    let Some(gpu) = gpu() else { return };
    let x = x_vec();
    let (ma, da) = nvfp4_matrix(ROWS, SEED_W);
    let (mb, db) = nvfp4_matrix(ROWS_ALT, SEED_W_ALT);
    let single_a = gpu
        .nvfp4_gemv(&ma.packed, &ma.scales, ma.tensor_scale, &x, ROWS, K)
        .expect("aligned geometry dispatches");
    let single_b = gpu
        .nvfp4_gemv(&mb.packed, &mb.scales, mb.tensor_scale, &x, ROWS_ALT, K)
        .expect("aligned geometry dispatches");
    assert!(rel_error(&cpu_gemv(&da, &x, ROWS, K), &single_a) < REDUCTION_REL_TOL);
    assert!(rel_error(&cpu_gemv(&db, &x, ROWS_ALT, K), &single_b) < REDUCTION_REL_TOL);
    let batch = gpu
        .nvfp4_gemv_multi(
            &[
                (&ma.packed, &ma.scales, ma.tensor_scale, ROWS, K),
                (&mb.packed, &mb.scales, mb.tensor_scale, ROWS_ALT, K),
            ],
            &x,
        )
        .expect("valid batch dispatches");
    assert_eq!(batch, vec![single_a, single_b]);
    assert_eq!(gpu.nvfp4_gemv_multi(&[], &x), Some(Vec::new()));
}

/// NVFP4 geometry guards: unaligned `k`, short input, short packed, short
/// scales each yield `None`.
#[test]
fn nvfp4_gemv_rejects_bad_geometry() {
    let Some(gpu) = gpu() else { return };
    let x = x_vec();
    let (m, _) = nvfp4_matrix(ROWS, SEED_W);
    let ts = m.tensor_scale;
    assert!(gpu
        .nvfp4_gemv(&m.packed, &m.scales, ts, &x, ROWS, K - 1)
        .is_none());
    assert!(gpu
        .nvfp4_gemv(
            &m.packed,
            &m.scales,
            ts,
            &x[..K - NVFP4_GROUP_ELEMS],
            ROWS,
            K
        )
        .is_none());
    assert!(gpu
        .nvfp4_gemv(&m.packed[..m.packed.len() - 1], &m.scales, ts, &x, ROWS, K)
        .is_none());
    assert!(gpu
        .nvfp4_gemv(&m.packed, &m.scales[..m.scales.len() - 1], ts, &x, ROWS, K)
        .is_none());
}

// ───────────────────────── top-k ─────────────────────────

/// `f32_gemv_topk1` returns the CPU argmax of the same gemv with its
/// score, on a fixture whose top gap is well above the tolerance.
#[test]
fn f32_gemv_topk1_equals_cpu_argmax() {
    let Some(gpu) = gpu() else { return };
    let (w, w_flat) = f32_matrix(ROWS, SEED_W);
    let x = x_vec();
    let scores = cpu_gemv(&w_flat, &x, ROWS, K);
    assert!(
        min_gap_between_top(&scores, 1) > MIN_TOPK_GAP,
        "fixture gap"
    );
    let (idx, val) = cpu_argmax(&scores);
    let (got_idx, got_val) = MatMul::f32_gemv_topk1(&gpu, w.view(), &x).expect("valid shape");
    assert_eq!(got_idx, idx);
    assert!((got_val - val).abs() < REDUCTION_REL_TOL * val.abs().max(1.0));
}

/// `f32_gemv_topk1` on a non-contiguous view (the transpose of a
/// row-major matrix) materialises a standard-layout copy and returns the
/// same argmax as the contiguous copy of that view.
#[test]
fn f32_gemv_topk1_handles_non_contiguous_view() {
    let Some(gpu) = gpu() else { return };
    // Square so the transpose keeps `k == K` for the shared input vector.
    let (w_square, _) = f32_matrix(K, SEED_W);
    let w_t = w_square.t();
    assert!(w_t.as_slice().is_none(), "fixture: view is non-contiguous");
    let x = x_vec();
    let contiguous = w_t.as_standard_layout().into_owned();
    let w_t_flat = contiguous.as_slice().unwrap();
    let scores = cpu_gemv(w_t_flat, &x, K, K);
    assert!(
        min_gap_between_top(&scores, 1) > MIN_TOPK_GAP,
        "fixture gap"
    );
    let (idx, _) = cpu_argmax(&scores);
    let (got_idx, _) = MatMul::f32_gemv_topk1(&gpu, w_t, &x).expect("valid shape");
    assert_eq!(got_idx, idx);
    let (from_contiguous, _) =
        MatMul::f32_gemv_topk1(&gpu, contiguous.view(), &x).expect("valid shape");
    assert_eq!(got_idx, from_contiguous);
}

/// `f32_gemv_topk1` rejects a mismatched input and an empty matrix.
#[test]
fn f32_gemv_topk1_rejects_mismatched_input_and_zero_rows() {
    let Some(gpu) = gpu() else { return };
    let (w, _) = f32_matrix(ROWS, SEED_W);
    let x = x_vec();
    assert!(MatMul::f32_gemv_topk1(&gpu, w.view(), &x[..K - 1]).is_none());
    let empty = Array2::<f32>::zeros((0, K));
    assert!(MatMul::f32_gemv_topk1(&gpu, empty.view(), &x).is_none());
}

/// `f16_gemv_topk1` equals the CPU argmax over the decoded f16 weights,
/// and `f16_gemv_topk` returns the CPU top-k in descending order with
/// matching indices.
#[test]
fn f16_gemv_topk1_and_topk_equal_cpu_ranking() {
    let Some(gpu) = gpu() else { return };
    let (w, dec) = f16_matrix(ROWS, SEED_W);
    let x = x_vec();
    let scores = cpu_gemv(&dec, &x, ROWS, K);
    assert!(
        min_gap_between_top(&scores, TOPK_REQUESTED) > MIN_TOPK_GAP,
        "fixture gap"
    );
    let (idx, val) = cpu_argmax(&scores);
    let (got_idx, got_val) = MatMul::f16_gemv_topk1(&gpu, &w, &x, ROWS, K).expect("valid shape");
    assert_eq!(got_idx, idx);
    assert!((got_val - val).abs() < REDUCTION_REL_TOL * val.abs().max(1.0));

    let expected = cpu_topk(&scores, TOPK_REQUESTED);
    let got = MatMul::f16_gemv_topk(&gpu, &w, &x, ROWS, K, TOPK_REQUESTED)
        .expect("top_k within capacity");
    assert_eq!(got.len(), TOPK_REQUESTED);
    for (g, e) in got.iter().zip(&expected) {
        assert_eq!(g.0, e.0, "top-k index order");
        assert!((g.1 - e.1).abs() < REDUCTION_REL_TOL * e.1.abs().max(1.0));
    }
}

/// The f16 top-k arms decline on short weights, mismatched input, zero
/// rows, `top_k == 0` and `top_k > K_TOPK`.
#[test]
fn f16_gemv_topk_arms_reject_bad_shapes_and_capacity() {
    let Some(gpu) = gpu() else { return };
    let (w, _) = f16_matrix(ROWS, SEED_W);
    let x = x_vec();
    let short_w = &w[..w.len() - F16_BYTES];
    assert!(MatMul::f16_gemv_topk1(&gpu, short_w, &x, ROWS, K).is_none());
    assert!(MatMul::f16_gemv_topk1(&gpu, &w, &x[..K - 1], ROWS, K).is_none());
    assert!(MatMul::f16_gemv_topk1(&gpu, &w, &x, 0, K).is_none());
    assert!(MatMul::f16_gemv_topk(&gpu, short_w, &x, ROWS, K, 1).is_none());
    assert!(MatMul::f16_gemv_topk(&gpu, &w, &x, ROWS, K, 0).is_none());
    assert!(MatMul::f16_gemv_topk(&gpu, &w, &x, ROWS, K, K_TOPK + 1).is_none());
}

// ───────────────────────── matmul_batch ─────────────────────────

/// Each op is routed by its `transpose_b` flag: the first against
/// `a · b`, the second against `a · bᵀ`. Non-square `b` makes the two
/// routes have different shapes, so a flag ignored would not type-check
/// against the oracle.
#[test]
fn matmul_batch_dispatches_each_op_by_transpose_flag() {
    let Some(gpu) = gpu() else { return };
    let a = Array2::from_shape_vec((MM_M, MM_K), uniform_values(MM_M * MM_K, SEED_W)).unwrap();
    let b = Array2::from_shape_vec((MM_K, MM_N), uniform_values(MM_K * MM_N, SEED_W_ALT)).unwrap();
    let b_t = b.t().to_owned();
    let ops = vec![
        MatMulOp {
            a: a.clone(),
            b: b.clone(),
            transpose_b: false,
        },
        MatMulOp {
            a: a.clone(),
            b: b_t,
            transpose_b: true,
        },
    ];
    let outs = gpu.matmul_batch(&ops);
    assert_eq!(outs.len(), 2);
    let expected = a.dot(&b);
    for out in &outs {
        assert_eq!(out.shape(), &[MM_M, MM_N]);
        let err = rel_error(expected.as_slice().unwrap(), out.as_slice().unwrap());
        assert!(err < REDUCTION_REL_TOL, "matmul_batch arm error {err}");
    }
}

// ───────────────────────── Q4_K stride-32 ─────────────────────────

/// Hidden size that is a multiple of the Q4_K super-block (256).
const Q4K_HIDDEN: usize = 512;
/// Q4_K-vs-Q4_K (same quantised weights) disagreement is reduction
/// order only; the CPU path dequantises in a different grouping, so the
/// bound is looser than the f32 arms but far below any Q4_K ulp.
const Q4K_ABS_TOL: f32 = 0.5;

/// The stride-32 Q4_K matvec matches the CPU Q4_K matvec over the same
/// bytes, and declines a hidden size that is not a super-block multiple.
#[test]
fn q4k_matvec_stride32_matches_cpu_and_rejects_unaligned_hidden() {
    let Some(gpu) = gpu() else { return };
    let w = uniform_values(ROWS * Q4K_HIDDEN, SEED_W);
    let x = uniform_values(Q4K_HIDDEN, SEED_X);
    let q4k = larql_compute::cpu::ops::q4_common::quantize_q4_k(&w);
    let cpu = CpuBackend
        .q4k_matvec(&q4k, &x, ROWS, Q4K_HIDDEN)
        .expect("CPU Q4_K matvec");
    let got = MetalBackend::q4k_matvec_stride32(&gpu, &q4k, &x, ROWS, Q4K_HIDDEN)
        .expect("aligned hidden dispatches");
    for (i, (c, g)) in cpu.iter().zip(&got).enumerate() {
        assert!((c - g).abs() < Q4K_ABS_TOL, "row {i}: cpu={c} gpu={g}");
    }
    assert!(MetalBackend::q4k_matvec_stride32(&gpu, &q4k, &x, ROWS, Q4K_HIDDEN - 1).is_none());
    assert!(MetalBackend::q4k_matvec_stride32(&gpu, &q4k, &x, ROWS, 0).is_none());
}
