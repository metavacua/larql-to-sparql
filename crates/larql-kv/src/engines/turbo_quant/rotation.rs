//! Walsh–Hadamard Transform (WHT) for the turbo-quant codec.
//!
//! The WHT converts coordinates to a near-Gaussian distribution
//! (Beta(d/2, d/2) → approximates N(0, 1/d)). It is self-inverse up to
//! a 1/√d scaling factor.
//!
//! Complexity: O(d log d) — d/2 butterfly operations per stage, log₂(d)
//! stages. For d=256: 8 stages × 128 butterflies = 1024 add/sub ops.
//!
//! **2026-05-19 SIMD pass** (turbo-quant codec hot path):
//!
//! Diagnostic measurement showed `recompute_hot` consumed 64.5% of
//! turbo-quant's decode step (19.8 ms / 30.7 ms) at the default
//! head_dim=256 + bits=4 config. The WHT butterfly is the largest
//! single contributor inside that bucket. This file now ships two
//! paths:
//!
//! - **Scalar fallback** ([`wht`], [`wht_inplace`]) — portable; the
//!   reference for tests and non-aarch64 targets.
//! - **NEON path** ([`wht_inplace_neon`]) — aarch64 only; processes
//!   four butterflies per instruction at stages where `half ≥ 4`
//!   (covers 6 of 8 stages for head_dim=256). Falls back to scalar
//!   at the two innermost stages (half = 1, 2) where the pair layout
//!   doesn't fit a `vld1q_f32` cleanly.
//!
//! Both paths produce bit-equivalent output up to floating-point
//! re-association of the scaling multiply. The `wht_inplace_matches_scalar`
//! test pins this within 1e-5 tolerance.

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

#[inline(always)]
fn apply_sign_flips(y: &mut [f32]) {
    for (i, v) in y.iter_mut().enumerate() {
        if (i.wrapping_mul(2654435761) >> 16) & 1 == 1 {
            *v = -*v;
        }
    }
}

/// Forward WHT with sign flips: `D · H · D · x`. Self-inverse because
/// `(DHD)² = DH·(DD)·HD = DH·I·HD = D·(HH)·D = D·I·D = I`.
///
/// Returns a freshly-allocated `Vec<f32>`. Callers on the codec hot
/// path should prefer [`wht_inplace`] (or
/// [`wht_inplace_neon`] on aarch64) to skip the per-call allocation.
pub fn wht(x: &[f32]) -> Vec<f32> {
    let mut y = x.to_vec();
    wht_inplace(&mut y);
    y
}

/// In-place WHT. Same transform as [`wht`] but writes back into the
/// caller's buffer; the hot path uses this with a scratch buffer
/// reused across encode_vector calls.
pub fn wht_inplace(y: &mut [f32]) {
    let d = y.len();
    assert!(
        d.is_power_of_two(),
        "WHT requires power-of-2 dimension, got {d}"
    );

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 always has NEON; we're inside the cfg gate.
        unsafe { wht_inplace_neon(y) };
        return;
    }

    #[allow(unreachable_code)]
    {
        wht_inplace_scalar(y);
    }
}

#[doc(hidden)]
pub fn wht_inplace_scalar(y: &mut [f32]) {
    let d = y.len();
    apply_sign_flips(y);

    let mut half = 1;
    while half < d {
        let mut i = 0;
        while i < d {
            for j in i..i + half {
                let a = y[j];
                let b = y[j + half];
                y[j] = a + b;
                y[j + half] = a - b;
            }
            i += half * 2;
        }
        half *= 2;
    }

    let scale = 1.0 / (d as f32).sqrt();
    for v in &mut *y {
        *v *= scale;
    }

    apply_sign_flips(y);
}

/// NEON-accelerated in-place WHT for aarch64. Processes 4 butterflies
/// per `vaddq_f32`/`vsubq_f32` pair at stages where `half ≥ 4`.
///
/// # Safety
/// Requires aarch64. Caller must pass a power-of-two-length slice.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[doc(hidden)]
pub unsafe fn wht_inplace_neon(y: &mut [f32]) {
    use std::arch::aarch64::*;

    let d = y.len();
    apply_sign_flips(y);

    let mut half = 1;
    while half < d {
        if half >= 4 {
            // SIMD path: pairs are (y[i+j], y[i+j+half]); for `half ≥ 4`
            // we load 4 of each side with one `vld1q_f32` and butterfly
            // them as f32x4 add/sub.
            let mut i = 0;
            while i < d {
                let mut j = 0;
                while j < half {
                    let pa = y.as_mut_ptr().add(i + j);
                    let pb = y.as_mut_ptr().add(i + j + half);
                    let a = vld1q_f32(pa);
                    let b = vld1q_f32(pb);
                    vst1q_f32(pa, vaddq_f32(a, b));
                    vst1q_f32(pb, vsubq_f32(a, b));
                    j += 4;
                }
                i += half * 2;
            }
        } else {
            // Scalar fallback for half = 1, 2.
            let mut i = 0;
            while i < d {
                for j in i..i + half {
                    let a = y[j];
                    let b = y[j + half];
                    y[j] = a + b;
                    y[j + half] = a - b;
                }
                i += half * 2;
            }
        }
        half *= 2;
    }

    // Normalize: multiply each element by 1/√d. NEON 4-at-a-time.
    let scale = 1.0 / (d as f32).sqrt();
    let scale_v = vdupq_n_f32(scale);
    let mut i = 0;
    while i + 4 <= d {
        let p = y.as_mut_ptr().add(i);
        let v = vld1q_f32(p);
        vst1q_f32(p, vmulq_f32(v, scale_v));
        i += 4;
    }
    while i < d {
        y[i] *= scale;
        i += 1;
    }

    apply_sign_flips(y);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wht_self_inverse() {
        let x: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 100.0).collect();
        let y = wht(&x);
        let x_recon = wht(&y);

        for (a, b) in x.iter().zip(x_recon.iter()) {
            assert!((a - b).abs() < 1e-4, "WHT not self-inverse: {a} vs {b}");
        }
    }

    #[test]
    fn test_wht_preserves_norm() {
        let x: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01) - 1.28).collect();
        let norm_x: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let y = wht(&x);
        let norm_y: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();

        let err = (norm_x - norm_y).abs() / norm_x;
        assert!(err < 1e-4, "WHT changed norm by {err}: {norm_x} → {norm_y}");
    }

    #[test]
    fn wht_inplace_matches_wht() {
        let x: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 64.0).collect();
        let y_alloc = wht(&x);
        let mut y_inplace = x.clone();
        wht_inplace(&mut y_inplace);
        for (a, b) in y_alloc.iter().zip(y_inplace.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "wht_inplace diverged from wht: {a} vs {b}"
            );
        }
    }

    #[test]
    fn wht_inplace_scalar_matches_scalar_reference_at_sizes() {
        for d in [4usize, 8, 16, 32, 64, 128, 256] {
            let x: Vec<f32> = (0..d).map(|i| (i as f32 + 1.0) / (d as f32)).collect();
            let mut y = x.clone();
            wht_inplace_scalar(&mut y);
            // Self-inverse check confirms the scalar path is consistent.
            wht_inplace_scalar(&mut y);
            for (a, b) in x.iter().zip(y.iter()) {
                assert!(
                    (a - b).abs() < 1e-4,
                    "scalar WHT not self-inverse at d={d}: {a} vs {b}"
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn wht_neon_matches_scalar() {
        for d in [4usize, 8, 16, 32, 64, 128, 256] {
            let x: Vec<f32> = (0..d)
                .map(|i| ((i as f32) - 0.5 * d as f32) / 10.0)
                .collect();
            let mut y_scalar = x.clone();
            wht_inplace_scalar(&mut y_scalar);
            let mut y_neon = x.clone();
            // SAFETY: aarch64 always has NEON.
            unsafe { wht_inplace_neon(&mut y_neon) };
            for (a, b) in y_scalar.iter().zip(y_neon.iter()) {
                assert!(
                    (a - b).abs() < 1e-5,
                    "NEON diverged from scalar at d={d}: {a} vs {b}"
                );
            }
        }
    }
}
