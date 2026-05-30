//! `impl Fp4FfnAccess for VectorIndex`.
//!
//! Delegation shim for the FP4 (exp-26) FFN access paths.

use super::VectorIndex;
use crate::index::types::Fp4FfnAccess;

impl Fp4FfnAccess for VectorIndex {
    fn has_fp4_storage(&self) -> bool {
        VectorIndex::has_fp4_storage(self)
    }

    fn fp4_ffn_row_dot(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        x: &[f32],
    ) -> Option<f32> {
        VectorIndex::fp4_ffn_row_dot(self, layer, component, feat, x)
    }

    fn fp4_ffn_row_scaled_add(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        VectorIndex::fp4_ffn_row_scaled_add(self, layer, component, feat, alpha, out)
    }

    fn fp4_ffn_row_into(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        out: &mut [f32],
    ) -> bool {
        VectorIndex::fp4_ffn_row_into(self, layer, component, feat, out)
    }
}

#[cfg(test)]
mod tests {
    //! Trait-impl shim coverage. The inherent FP4 methods live in
    //! `index/storage/fp4_store.rs` and are exercised by the FP4
    //! integration tests; here we just pin the dispatch.
    use super::*;

    fn fresh() -> VectorIndex {
        VectorIndex::empty(2, 8)
    }

    #[test]
    fn has_fp4_storage_false_on_empty() {
        let v = fresh();
        assert!(!<VectorIndex as Fp4FfnAccess>::has_fp4_storage(&v));
    }

    #[test]
    fn fp4_ffn_row_dot_none_on_empty() {
        let v = fresh();
        let x = [1.0_f32; 8];
        assert!(<VectorIndex as Fp4FfnAccess>::fp4_ffn_row_dot(&v, 0, 0, 0, &x).is_none());
    }

    #[test]
    fn fp4_ffn_row_scaled_add_false_on_empty() {
        let v = fresh();
        let mut out = [0.0_f32; 8];
        assert!(!<VectorIndex as Fp4FfnAccess>::fp4_ffn_row_scaled_add(
            &v, 0, 0, 0, 1.0, &mut out
        ));
    }

    #[test]
    fn fp4_ffn_row_into_false_on_empty() {
        let v = fresh();
        let mut out = [0.0_f32; 8];
        assert!(!<VectorIndex as Fp4FfnAccess>::fp4_ffn_row_into(
            &v, 0, 0, 0, &mut out
        ));
    }
}
