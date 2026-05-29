//! `Fp4FfnAccess` — FP4 / FP8 FFN row access (exp 26).

/// FP4 / FP8 FFN storage access (exp 26).
pub trait Fp4FfnAccess: Send + Sync {
    // ── FP4 / FP8 FFN storage (exp 26) ─────────────────────────────────────
    //
    // These mirror the `q4k_ffn_row_*` family for the FP4 block format. All
    // default to "no data" so overlays / FFN impls that don't carry
    // FP4 storage work unchanged.

    /// Whether this index has FP4/FP8 FFN storage attached.
    fn has_fp4_storage(&self) -> bool {
        false
    }

    /// FP4/FP8 fused dequant + dot. `component`: 0=gate, 1=up, 2=down.
    fn fp4_ffn_row_dot(
        &self,
        _layer: usize,
        _component: usize,
        _feat: usize,
        _x: &[f32],
    ) -> Option<f32> {
        None
    }

    /// FP4/FP8 fused dequant + scaled-add: `out += alpha * dequant(row)`.
    fn fp4_ffn_row_scaled_add(
        &self,
        _layer: usize,
        _component: usize,
        _feat: usize,
        _alpha: f32,
        _out: &mut [f32],
    ) -> bool {
        false
    }

    /// FP4/FP8 dequantise one row into `out`.
    fn fp4_ffn_row_into(
        &self,
        _layer: usize,
        _component: usize,
        _feat: usize,
        _out: &mut [f32],
    ) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoOpFp4;
    impl Fp4FfnAccess for NoOpFp4 {}

    #[test]
    fn defaults_decline_every_row_op() {
        let n = NoOpFp4;
        let x = [1.0_f32; 4];
        let mut out = [0.0_f32; 4];
        assert!(!n.has_fp4_storage());
        assert!(n.fp4_ffn_row_dot(0, 0, 0, &x).is_none());
        assert!(!n.fp4_ffn_row_scaled_add(0, 0, 0, 1.0, &mut out));
        assert!(!n.fp4_ffn_row_into(0, 0, 0, &mut out));
        // Defaults must not write to `out`.
        assert_eq!(out, [0.0_f32; 4]);
    }
}
