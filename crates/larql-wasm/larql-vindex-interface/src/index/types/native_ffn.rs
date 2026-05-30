//! `NativeFfnAccess` — f32/f16 FFN row access.

/// Native f32/f16 FFN storage access.
pub trait NativeFfnAccess: Send + Sync {
    fn down_feature_vector(&self, _layer: usize, _feature: usize) -> Option<&[f32]> {
        None
    }
    fn has_down_features(&self) -> bool {
        false
    }
    fn down_layer_matrix(&self, _layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        None
    }
    fn up_layer_matrix(&self, _layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        None
    }
    fn has_full_mmap_ffn(&self) -> bool {
        false
    }
    fn has_interleaved(&self) -> bool {
        false
    }
    fn interleaved_gate(&self, _layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        None
    }
    fn interleaved_up(&self, _layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        None
    }
    fn interleaved_down(&self, _layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        None
    }
    fn prefetch_interleaved_layer(&self, _layer: usize) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All-defaults stub — every native-FFN method falls through to
    /// the no-op default body. Lets those bodies run under coverage.
    struct NoOpNative;
    impl NativeFfnAccess for NoOpNative {}

    #[test]
    fn all_defaults_are_safely_empty() {
        let n = NoOpNative;
        assert!(n.down_feature_vector(0, 0).is_none());
        assert!(!n.has_down_features());
        assert!(n.down_layer_matrix(0).is_none());
        assert!(n.up_layer_matrix(0).is_none());
        assert!(!n.has_full_mmap_ffn());
        assert!(!n.has_interleaved());
        assert!(n.interleaved_gate(0).is_none());
        assert!(n.interleaved_up(0).is_none());
        assert!(n.interleaved_down(0).is_none());
        // prefetch is a no-op; just make sure it doesn't panic.
        n.prefetch_interleaved_layer(0);
    }
}
