//! `PatchOverrides` — overlay-vector hooks.

/// Patch overlay vectors installed above a readonly base vindex.
pub trait PatchOverrides: Send + Sync {
    fn down_override(&self, _layer: usize, _feature: usize) -> Option<&[f32]> {
        None
    }
    /// Up vector override at (layer, feature). Used by INSERT to write
    /// the slot's up component when installing a constellation fact.
    /// `walk_ffn_sparse` checks this before reading from `up_layer_matrix`,
    /// matching the parallel pattern for `down_override`.
    fn up_override(&self, _layer: usize, _feature: usize) -> Option<&[f32]> {
        None
    }
    /// Gate vector override at (layer, feature). Lives in the patch
    /// overlay (`PatchedVindex.overrides_gate`). Used by the sparse
    /// inference fallback to recompute `silu(gate_override · x)` so
    /// the strong installed gate actually drives the activation —
    /// without this, gather-from-dense reads the original weak slot.
    fn gate_override(&self, _layer: usize, _feature: usize) -> Option<&[f32]> {
        None
    }
    /// Check if any down vector overrides or gate overrides exist at this layer.
    fn has_overrides_at(&self, _layer: usize) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All-defaults stub — exercises the no-op trait bodies so the
    /// default-method lines actually run under coverage.
    struct NoOpPatch;
    impl PatchOverrides for NoOpPatch {}

    #[test]
    fn defaults_return_none_or_false() {
        let n = NoOpPatch;
        assert!(n.down_override(0, 0).is_none());
        assert!(n.up_override(0, 0).is_none());
        assert!(n.gate_override(0, 0).is_none());
        assert!(!n.has_overrides_at(0));
    }
}
