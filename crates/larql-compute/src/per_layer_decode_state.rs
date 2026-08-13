//! Per-layer pre-attention residual + new K/V state buffer.
//!
//! Used by KvDispatch backends' decode-with-state variants so engines
//! can capture per-layer intermediates without re-running the
//! attention pass. Moved here from `larql-inference/src/kv_dispatch/mod.rs`
//! (ADR-0022 Step 3c) so the moved-down `kquant_forward::cached.rs`
//! can take `Option<&mut crate::PerLayerDecodeState>` parameters.
//!
//! **W10 Phase A (2026-05-18):** Fields now hold
//! `Vec<Box<dyn StateHandle>>` instead of `Vec<Array2<f32>>`. The
//! handle abstraction lets producers defer materialisation cost and
//! lets engines consume entries by value without an extra copy on the
//! CPU happy path. See [`crate::state_handle`] for the trait surface.

use crate::state_handle::StateHandle;
use crate::StateDumpMask;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Captured per-layer state at a single decode step (or per-position
/// state across a prefill).
///
/// **Shape duality.** Decode-step paths populate each layer's entry
/// as a `[1, dim]` chunk; prefill paths populate each entry as
/// `[seq_len, dim]`. Consumers must read the actual shape from the
/// handle, not assume single-row.
pub struct PerLayerDecodeState {
    /// Pre-attention residual entering each layer's attention block.
    /// Each entry has `cols = hidden_size`; rows depend on caller.
    pub h_in_per_layer: Vec<Box<dyn StateHandle>>,
    /// New K rows projected this step (or this prefill), per layer.
    /// Each entry has `cols = kv_dim_for_layer`.
    pub k_new_per_layer: Vec<Box<dyn StateHandle>>,
    /// New V rows projected this step (or this prefill), per layer.
    /// Each entry has `cols = kv_dim_for_layer`.
    pub v_new_per_layer: Vec<Box<dyn StateHandle>>,
}

impl Default for PerLayerDecodeState {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl core::fmt::Debug for PerLayerDecodeState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PerLayerDecodeState")
            .field("h_in_per_layer.len", &self.h_in_per_layer.len())
            .field("k_new_per_layer.len", &self.k_new_per_layer.len())
            .field("v_new_per_layer.len", &self.v_new_per_layer.len())
            .finish()
    }
}

impl PerLayerDecodeState {
    /// Pre-allocate vectors sized for `num_layers`. Caller should
    /// invoke this before passing `Some(&mut state)` to a decode
    /// step; backends `.push()` per layer.
    pub fn with_capacity(num_layers: usize) -> Self {
        Self {
            h_in_per_layer: Vec::with_capacity(num_layers),
            k_new_per_layer: Vec::with_capacity(num_layers),
            v_new_per_layer: Vec::with_capacity(num_layers),
        }
    }

    /// `true` when the state was populated for every layer (one
    /// entry per layer in each vector). Backends MUST guarantee
    /// this on success, but engines may double-check.
    pub fn is_complete_for(&self, num_layers: usize) -> bool {
        self.h_in_per_layer.len() == num_layers
            && self.k_new_per_layer.len() == num_layers
            && self.v_new_per_layer.len() == num_layers
    }

    /// `is_complete_for` variant that respects the capture mask. Under
    /// [`StateDumpMask::HOnly`] only `h_in_per_layer` is required to
    /// be populated; under [`StateDumpMask::Full`] all three vectors
    /// must be.
    pub fn is_complete_under(&self, num_layers: usize, mask: StateDumpMask) -> bool {
        match mask {
            StateDumpMask::Full => self.is_complete_for(num_layers),
            StateDumpMask::HOnly => self.h_in_per_layer.len() == num_layers,
            StateDumpMask::None => true,
        }
    }

    /// Drop all per-layer entries without freeing capacity. Use
    /// before re-passing to the next decode step.
    pub fn reset(&mut self) {
        self.h_in_per_layer.clear();
        self.k_new_per_layer.clear();
        self.v_new_per_layer.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_handle::CpuStateHandle;
    use ndarray::Array2;

    #[test]
    fn with_capacity_returns_empty_vectors() {
        let s = PerLayerDecodeState::with_capacity(4);
        assert!(s.h_in_per_layer.is_empty());
        assert!(s.k_new_per_layer.is_empty());
        assert!(s.v_new_per_layer.is_empty());
        assert!(!s.is_complete_for(4));
    }

    #[test]
    fn is_complete_for_pins_per_layer_count() {
        let mut s = PerLayerDecodeState::with_capacity(2);
        s.h_in_per_layer
            .push(CpuStateHandle::boxed(Array2::zeros((1, 4))));
        s.k_new_per_layer
            .push(CpuStateHandle::boxed(Array2::zeros((1, 2))));
        s.v_new_per_layer
            .push(CpuStateHandle::boxed(Array2::zeros((1, 2))));
        assert!(!s.is_complete_for(2));
        s.h_in_per_layer
            .push(CpuStateHandle::boxed(Array2::zeros((1, 4))));
        s.k_new_per_layer
            .push(CpuStateHandle::boxed(Array2::zeros((1, 2))));
        s.v_new_per_layer
            .push(CpuStateHandle::boxed(Array2::zeros((1, 2))));
        assert!(s.is_complete_for(2));
    }

    #[test]
    fn reset_clears_without_freeing_capacity() {
        let mut s = PerLayerDecodeState::with_capacity(4);
        for _ in 0..4 {
            s.h_in_per_layer
                .push(CpuStateHandle::boxed(Array2::zeros((1, 8))));
        }
        s.reset();
        assert!(s.h_in_per_layer.is_empty());
        assert!(s.h_in_per_layer.capacity() >= 4);
    }

    #[test]
    fn default_yields_empty_state() {
        let s = PerLayerDecodeState::default();
        assert!(s.h_in_per_layer.is_empty());
        assert!(s.k_new_per_layer.is_empty());
        assert!(s.v_new_per_layer.is_empty());
    }

    #[test]
    fn debug_includes_per_layer_lengths() {
        let mut s = PerLayerDecodeState::with_capacity(2);
        s.h_in_per_layer
            .push(CpuStateHandle::boxed(Array2::zeros((1, 4))));
        let dbg = format!("{s:?}");
        assert!(dbg.contains("h_in_per_layer.len"));
        assert!(dbg.contains("k_new_per_layer.len"));
        assert!(dbg.contains("v_new_per_layer.len"));
        // Length values are present in the rendered output.
        assert!(dbg.contains("1") && dbg.contains("0"));
    }

    #[test]
    fn is_complete_under_full_matches_is_complete_for() {
        let mut s = PerLayerDecodeState::with_capacity(1);
        s.h_in_per_layer
            .push(CpuStateHandle::boxed(Array2::zeros((1, 4))));
        // K/V missing → Full fails.
        assert!(!s.is_complete_under(1, StateDumpMask::Full));
        s.k_new_per_layer
            .push(CpuStateHandle::boxed(Array2::zeros((1, 2))));
        s.v_new_per_layer
            .push(CpuStateHandle::boxed(Array2::zeros((1, 2))));
        assert!(s.is_complete_under(1, StateDumpMask::Full));
    }

    #[test]
    fn is_complete_under_h_only_ignores_kv() {
        let mut s = PerLayerDecodeState::with_capacity(1);
        s.h_in_per_layer
            .push(CpuStateHandle::boxed(Array2::zeros((1, 4))));
        assert!(s.is_complete_under(1, StateDumpMask::HOnly));
        assert!(!s.is_complete_under(1, StateDumpMask::Full));
    }

    #[test]
    fn is_complete_under_none_is_trivially_true() {
        let s = PerLayerDecodeState::default();
        assert!(s.is_complete_under(8, StateDumpMask::None));
    }
}
