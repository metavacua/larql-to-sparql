//! Numeric primitives used by the MoE forward pass.
//!
//! `pub(super)` keeps these module-private — `cpu_moe_forward` and the
//! per-expert helpers share them, nothing outside `moe/` should.

/// Dequantize a BF16 byte slice to f32.
#[inline]
pub(super) fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| {
            f32::from_bits(
                (u32::from(u8::from_le_bytes([b[0]]))
                    | (u32::from(u8::from_le_bytes([b[1]])) << 8))
                    << 16,
            )
        })
        .collect()
}

// `extract_expert_weights` was the pre-cache code path (eager BF16→f32 on
// every token). Replaced by `super::cache::cached_dequant` in both
// `forward.rs` and `expert.rs` — keeping `bf16_to_f32` as the underlying
// conversion helper, but the bulk-extract shim is no longer needed.

/// RMSNorm: out[i] = x[i] / rms(x) * (w[i] + offset)
pub(super) fn rms_norm(x: &[f32], w: &[f32], eps: f32, offset: f32) -> Vec<f32> {
    if w.is_empty() || x.is_empty() {
        return x.to_vec();
    }
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter()
        .zip(w.iter())
        .map(|(&xi, &wi)| xi / rms * (wi + offset))
        .collect()
}

/// Parameter-free RMSNorm (HF `Gemma4RMSNorm(with_scale=False)`): scales
/// `x` by `1/sqrt(mean(x²) + eps)` with no learned weight. Used by the
/// Gemma 4 router, whose norm has no `.weight` tensor on disk.
pub(super) fn rms_norm_no_weight(x: &[f32], eps: f32) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt();
    x.iter().map(|v| v / rms).collect()
}

/// SiLU activation: x * sigmoid(x)
#[inline]
pub(super) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GELU with tanh approximation (Gemma 4 expert FFN activation).
#[inline]
pub(super) fn gelu_tanh(x: f32) -> f32 {
    let c = 0.797_884_6_f32;
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

/// Compute y = W · x  (W is [out_rows, in_cols] row-major, x is [in_cols]).
///
/// Uses BLAS sgemv via the workspace-level `ndarray` BLAS feature (Accelerate
/// on macOS, OpenBLAS on Linux). For the 26B A4B MoE this replaces a scalar
/// loop that dominated decode time: each expert call is roughly
/// `out_rows × in_cols` multiplies, repeated 8 experts × 60 layers per token,
/// and BLAS sgemv hits the AMX tiles + SIMD fused-multiply-add pipeline that
/// the scalar path misses entirely.
pub(super) fn matmul_vec(x: &[f32], w: &[f32], out_rows: usize, in_cols: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), out_rows * in_cols);
    debug_assert_eq!(x.len(), in_cols);
    if out_rows == 0 || in_cols == 0 {
        return vec![0.0f32; out_rows];
    }
    let w_view = ndarray::ArrayView2::from_shape((out_rows, in_cols), w)
        .expect("matmul_vec: weight shape mismatch");
    let x_view = ndarray::ArrayView1::from(x);
    // `Array2.dot(&Array1)` dispatches to BLAS sgemv when the ndarray blas
    // feature is enabled at the workspace level (larql-compute owns that).
    w_view.dot(&x_view).to_vec()
}

/// Same as `matmul_vec` but writes into a caller-provided output buffer
/// instead of allocating.  Reuse a per-call scratch (`gate_scratch`,
/// `up_scratch`, `out_scratch` in `run_single_expert`) to avoid 360+ heap
/// allocations per token per shard at Gemma 4 26B-A4B sizes.
///
/// `out` must have length exactly `out_rows`; existing contents are
/// overwritten.  Panics in debug builds on size mismatch (matches
/// `matmul_vec`'s assertion semantics).
pub(super) fn matmul_vec_into(
    out: &mut [f32],
    x: &[f32],
    w: &[f32],
    out_rows: usize,
    in_cols: usize,
) {
    debug_assert_eq!(w.len(), out_rows * in_cols);
    debug_assert_eq!(x.len(), in_cols);
    debug_assert_eq!(out.len(), out_rows);
    if out_rows == 0 || in_cols == 0 {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let w_view = ndarray::ArrayView2::from_shape((out_rows, in_cols), w)
        .expect("matmul_vec_into: weight shape mismatch");
    let x_view = ndarray::ArrayView1::from(x);
    // `assign_to` writes the gemv result into `out` without allocating an
    // intermediate Array1 — the same code path as `Array2.dot(&Array1)` but
    // landing in caller memory.
    let dst = ndarray::ArrayViewMut1::from(out);
    w_view.dot(&x_view).assign_to(dst);
}

/// Softmax in-place.
pub(super) fn softmax(v: &mut [f32]) {
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

/// Top-k indices by value (descending). Returns (indices, values).
pub(super) fn top_k(v: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let k = k.min(v.len());
    let mut indexed: Vec<(usize, f32)> = v.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);
    let indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
    let values: Vec<f32> = indexed.iter().map(|(_, v)| *v).collect();
    (indices, values)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// BF16 round-trip on the standard handful of "easy" floats —
    /// catches an endianness flip or a bit-shift typo.
    #[test]
    fn bf16_to_f32_known_values() {
        // 1.0 in BF16 = 0x3F80
        let bytes = vec![0x80u8, 0x3F];
        assert_eq!(bf16_to_f32(&bytes), vec![1.0]);
        // 0.0
        assert_eq!(bf16_to_f32(&[0x00, 0x00]), vec![0.0]);
        // -1.0 in BF16 = 0xBF80
        assert_eq!(bf16_to_f32(&[0x80, 0xBF]), vec![-1.0]);
        // 5.0 in BF16 = 0x40A0
        assert_eq!(bf16_to_f32(&[0xA0, 0x40]), vec![5.0]);
        // Multiple values in one call
        let bytes = vec![0x80, 0x3F, 0x80, 0xBF, 0xA0, 0x40];
        assert_eq!(bf16_to_f32(&bytes), vec![1.0, -1.0, 5.0]);
    }

    /// `rms_norm(constant_x, weight=1, offset=0)` — RMS of [c,c,…] is
    /// |c|, so out[i] = c / |c| * 1 = sign(c).
    #[test]
    fn rms_norm_constant_input() {
        let x = vec![2.0; 8];
        let w = vec![1.0; 8];
        let out = rms_norm(&x, &w, 0.0, 0.0);
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-5, "expected 1.0, got {v}");
        }
    }

    /// `rms_norm` with empty weight slice returns the input unchanged
    /// (defensive guard for "weight tensor not present").
    #[test]
    fn rms_norm_empty_weight_passthrough() {
        let x = vec![3.0, 4.0, 5.0];
        let out = rms_norm(&x, &[], 1e-6, 0.0);
        assert_eq!(out, x);
    }

    /// Parameter-free RMSNorm: scales `x` so that `mean(out²) ≈ 1`.
    #[test]
    fn rms_norm_no_weight_normalises_to_unit_rms() {
        let x = vec![2.0, 4.0, 6.0, 8.0];
        let out = rms_norm_no_weight(&x, 1e-6);
        let mean_sq: f32 = out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32;
        assert!(
            (mean_sq - 1.0).abs() < 1e-4,
            "mean(out²)={mean_sq:.5} ≠ 1.0"
        );
    }

    /// SiLU(0) = 0, SiLU(x) → x as x → ∞, SiLU(x) → 0 as x → -∞.
    #[test]
    fn silu_known_values() {
        assert_eq!(silu(0.0), 0.0);
        assert!(silu(10.0) > 9.99);
        assert!(silu(-10.0).abs() < 1e-3);
    }

    /// `top_k` returns the largest k values in descending order.
    #[test]
    fn top_k_descending_with_k_capped_at_len() {
        let (idx, val) = top_k(&[0.1, 0.5, 0.3, 0.9, 0.2], 3);
        assert_eq!(idx, vec![3, 1, 2]); // values 0.9, 0.5, 0.3
        assert_eq!(val, vec![0.9, 0.5, 0.3]);

        // k > len — get all in descending order.
        let (idx, _) = top_k(&[0.1, 0.5, 0.3], 99);
        assert_eq!(idx, vec![1, 2, 0]);
    }

    /// `softmax` produces a probability distribution.
    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.0f32, 2.0, 3.0, 4.0];
        softmax(&mut v);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum={sum} ≠ 1");
        // Largest input → largest output.
        let max_idx = v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 3, "max input index should be max output index");
    }

    /// `matmul_vec` agrees with a hand-rolled scalar reference.
    #[test]
    fn matmul_vec_matches_scalar_reference() {
        let w = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0,
        ]; // row 1
        let x = vec![1.0, 1.0, 1.0];
        let out = matmul_vec(&x, &w, 2, 3);
        // Hand-computed: row0 = 1+2+3 = 6; row1 = 4+5+6 = 15.
        assert_eq!(out, vec![6.0, 15.0]);
    }

    /// Empty input dimensions return a zero-filled output of the
    /// requested length — defensive guard, not a panic.
    #[test]
    fn matmul_vec_zero_dimensions_returns_zeros() {
        let out = matmul_vec(&[], &[], 4, 0);
        assert_eq!(out, vec![0.0, 0.0, 0.0, 0.0]);
    }
}
