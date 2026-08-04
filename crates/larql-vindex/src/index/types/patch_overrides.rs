//! `PatchOverrides` — overlay-vector hooks.

/// One patched slot at a layer, as enumerated by
/// [`PatchOverrides::override_slots_at`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverrideSlot {
    /// Feature index of the patched slot.
    pub feature: usize,
    /// `true` when the slot is tombstoned (DELETE): its post-patch
    /// contribution is zero and only the base row remains to subtract.
    pub tombstoned: bool,
}

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
    /// Enumerate every patched slot at `layer`: each feature carrying a
    /// gate/up/down override plus each tombstoned (deleted) feature,
    /// in ascending feature order without duplicates.
    ///
    /// `None` means the implementation supports only the point queries
    /// above — callers that need the complete patch set (the base+delta
    /// FFN path, 2026-07-30 review item 16) must then decline. The
    /// default is `None` so partial mocks stay honest: an implementation
    /// answering `Some` promises the list is exhaustive, because
    /// base+delta subtracts/adds exactly these slots on top of an
    /// unpatched dense base.
    fn override_slots_at(&self, _layer: usize) -> Option<Vec<OverrideSlot>> {
        None
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
        assert!(
            n.override_slots_at(0).is_none(),
            "default must decline enumeration — point-query-only mocks \
             cannot promise an exhaustive patch set"
        );
    }

    #[test]
    fn override_slot_is_plain_data() {
        let a = OverrideSlot {
            feature: 3,
            tombstoned: false,
        };
        let b = OverrideSlot {
            feature: 3,
            tombstoned: true,
        };
        assert_ne!(a, b);
        assert_eq!(a, a);
    }
}
