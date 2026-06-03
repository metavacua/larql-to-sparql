//! ¬L∧¬M-portable kernels.
//!
//! Ported from `larql-compute-wasm32v1-none/src/cpu/ops/geglu.rs`.
//! Algorithm is identical; only the exp() call dispatches differently:
//!   - wasm32-wasip1 / native (std available): `f32::exp` inherent method
//!   - wasm32v1-none / nvptx64 / spirv (no_std): `libm::expf`
//!
//! This mirrors how `larql-compute-wasm32v1-none` uses `larql-wasm-math`
//! (which is itself libm-backed) for the no_std wasm32 targets.
//!
//! **Feature parity test**: the test values here match the expected values in
//! `larql-compute-wasm32v1-none`'s geglu tests. Any divergence is a GPU-port bug.

// no_std targets: use libm for transcendental math.
#[cfg(any(target_os = "none", target_arch = "nvptx64", target_arch = "spirv"))]
#[inline(always)]
fn neg_exp(x: f32) -> f32 {
    libm::expf(-x)
}

// std targets (wasm32-wasip1, native): use the inherent f32 method.
#[cfg(not(any(target_os = "none", target_arch = "nvptx64", target_arch = "spirv")))]
#[inline(always)]
fn neg_exp(x: f32) -> f32 {
    (-x).exp()
}

/// SiLU (Swish) activation: `x / (1 + exp(-x))`.
///
/// GPU-portable: no heap allocation, no recursion, no host imports.
/// Ported from `larql-compute-wasm32v1-none::cpu::ops::geglu::silu`.
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + neg_exp(x))
}

/// In-place GEGLU: `out[i] = silu(gate[i]) × up[i]`.
///
/// Operates on caller-provided slices — no heap growth. This is the
/// ¬L∧¬M-portable form of the GEGLU activation in the compute pipeline.
///
/// Ported from `larql-compute-wasm32v1-none::cpu::ops::geglu::geglu_silu`.
pub fn geglu_silu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len(), "gate and up must be same length");
    debug_assert_eq!(gate.len(), out.len(), "gate and out must be same length");
    for i in 0..gate.len() {
        out[i] = silu(gate[i]) * up[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Feature parity tests ──────────────────────────────────────────────────
    // These tests replicate the exact checks in larql-compute-wasm32v1-none's
    // geglu tests. If any value diverges, this GPU port has a bug.

    #[test]
    fn silu_zero() {
        // silu(0) = 0 / (1 + exp(0)) = 0 / 2 = 0.0
        assert!((silu(0.0) - 0.0).abs() < 1e-6, "silu(0) must equal 0");
    }

    #[test]
    fn silu_large_positive_approaches_identity() {
        // silu(x) ≈ x for large positive x (exp(-x) → 0)
        assert!(
            silu(10.0) > 9.9,
            "silu(10) must be close to 10, got {}",
            silu(10.0)
        );
    }

    #[test]
    fn silu_large_negative_approaches_zero() {
        // silu(x) ≈ 0 for large negative x
        assert!(
            silu(-10.0).abs() < 0.001,
            "silu(-10) must be near 0, got {}",
            silu(-10.0)
        );
    }

    #[test]
    fn silu_known_value() {
        // silu(1.0) = 1 / (1 + exp(-1)) ≈ 0.7310586
        // Must match larql-compute-wasm32v1-none's geglu::silu result.
        let expected = 0.7310586f32;
        let got = silu(1.0);
        assert!(
            (got - expected).abs() < 1e-5,
            "silu(1.0) parity failure: expected {expected}, got {got}"
        );
    }

    #[test]
    fn geglu_silu_zero_gate_kills_output() {
        // silu(0) = 0, so gate=0 → out=0 regardless of up.
        let gate = [0.0f32; 4];
        let up = [5.0f32; 4];
        let mut out = [0.0f32; 4];
        geglu_silu(&gate, &up, &mut out);
        for &v in &out {
            assert!(v.abs() < 1e-6, "zero gate must produce zero output, got {v}");
        }
    }

    #[test]
    fn geglu_silu_unit_gate() {
        // gate=1, up=2 → silu(1)*2 ≈ 0.7310586 * 2 ≈ 1.4621172
        let gate = [1.0f32; 4];
        let up = [2.0f32; 4];
        let mut out = [0.0f32; 4];
        geglu_silu(&gate, &up, &mut out);
        let expected = silu(1.0) * 2.0;
        for &v in &out {
            assert!(
                (v - expected).abs() < 1e-5,
                "unit-gate parity failure: expected {expected}, got {v}"
            );
        }
    }

    #[test]
    fn geglu_silu_in_place_does_not_alias() {
        // Verify out is completely overwritten (not accumulated).
        let gate = [1.0f32; 4];
        let up = [1.0f32; 4];
        let mut out = [999.0f32; 4]; // poison value
        geglu_silu(&gate, &up, &mut out);
        for &v in &out {
            assert!(
                (v - silu(1.0)).abs() < 1e-5,
                "in-place must overwrite, got {v}"
            );
        }
    }
}
