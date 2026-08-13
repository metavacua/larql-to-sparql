//! `impl PatchOverrides for VectorIndex`.
//!
//! The patch-overlay capability — direct HashMap lookups against the
//! per-feature override tables in `metadata`. Every other capability
//! impl on `VectorIndex` is a delegation shim; this one is the actual
//! implementation, because there's no inherent method to delegate to.

use super::VectorIndex;
use crate::index::types::{OverrideSlot, PatchOverrides};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

impl PatchOverrides for VectorIndex {
    fn down_override(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.metadata
            .down_overrides
            .get(&(layer, feature))
            .map(|v| v.as_slice())
    }

    fn up_override(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        self.metadata
            .up_overrides
            .get(&(layer, feature))
            .map(|v| v.as_slice())
    }

    fn has_overrides_at(&self, layer: usize) -> bool {
        self.metadata
            .down_overrides
            .keys()
            .any(|(l, _)| *l == layer)
            || self.metadata.up_overrides.keys().any(|(l, _)| *l == layer)
    }

    /// The base index carries only up/down vector overrides (gate
    /// overrides and tombstones live on `PatchedVindex`'s overlay), so
    /// the enumeration is the sorted union of both override maps'
    /// features at `layer` — never tombstoned.
    fn override_slots_at(&self, layer: usize) -> Option<Vec<OverrideSlot>> {
        let feats: std::collections::BTreeSet<usize> = self
            .metadata
            .up_overrides
            .keys()
            .chain(self.metadata.down_overrides.keys())
            .filter(|(l, _)| *l == layer)
            .map(|&(_, f)| f)
            .collect();
        Some(
            feats
                .into_iter()
                .map(|feature| OverrideSlot {
                    feature,
                    tombstoned: false,
                })
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> VectorIndex {
        VectorIndex::empty(3, 8)
    }

    #[test]
    fn down_override_returns_inserted_slice() {
        let mut v = fresh();
        v.metadata
            .down_overrides
            .insert((1, 5), vec![0.1, 0.2, 0.3, 0.4]);
        let slice = v.down_override(1, 5).expect("override present");
        assert_eq!(slice, &[0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn down_override_returns_none_for_missing_key() {
        let v = fresh();
        assert!(v.down_override(0, 0).is_none());
    }

    #[test]
    fn up_override_is_keyed_separately_from_down() {
        let mut v = fresh();
        v.metadata.up_overrides.insert((2, 0), vec![9.0; 4]);
        v.metadata.down_overrides.insert((2, 0), vec![1.0; 4]);
        assert_eq!(v.up_override(2, 0).unwrap(), &[9.0; 4]);
        assert_eq!(v.down_override(2, 0).unwrap(), &[1.0; 4]);
        // Cross-key lookup must miss.
        assert!(v.up_override(2, 1).is_none());
    }

    #[test]
    fn has_overrides_at_returns_true_for_down_only() {
        let mut v = fresh();
        v.metadata.down_overrides.insert((4, 0), vec![1.0]);
        assert!(v.has_overrides_at(4));
        assert!(!v.has_overrides_at(5));
    }

    #[test]
    fn has_overrides_at_returns_true_for_up_only() {
        let mut v = fresh();
        v.metadata.up_overrides.insert((7, 2), vec![1.0]);
        assert!(v.has_overrides_at(7));
        assert!(!v.has_overrides_at(0));
    }

    #[test]
    fn has_overrides_at_short_circuits_when_down_present() {
        // Both maps populated — function should still report true.
        let mut v = fresh();
        v.metadata.down_overrides.insert((1, 0), vec![1.0]);
        v.metadata.up_overrides.insert((1, 0), vec![2.0]);
        assert!(v.has_overrides_at(1));
    }

    #[test]
    fn has_overrides_at_returns_false_on_empty_index() {
        let v = fresh();
        for layer in 0..3 {
            assert!(!v.has_overrides_at(layer));
        }
    }

    /// Enumeration is the sorted, deduplicated union of the up and down
    /// override maps at the layer — and never tombstoned (the base
    /// index has no delete vocabulary).
    #[test]
    fn override_slots_at_unions_up_and_down_maps_sorted() {
        let mut v = fresh();
        v.metadata.down_overrides.insert((1, 7), vec![1.0]);
        v.metadata.down_overrides.insert((1, 2), vec![1.0]);
        v.metadata.up_overrides.insert((1, 2), vec![2.0]); // same slot as a down override
        v.metadata.up_overrides.insert((1, 5), vec![2.0]);
        v.metadata.up_overrides.insert((0, 9), vec![2.0]); // other layer — excluded

        let slots = v.override_slots_at(1).expect("base index enumerates");
        assert_eq!(
            slots,
            vec![
                super::OverrideSlot {
                    feature: 2,
                    tombstoned: false
                },
                super::OverrideSlot {
                    feature: 5,
                    tombstoned: false
                },
                super::OverrideSlot {
                    feature: 7,
                    tombstoned: false
                },
            ]
        );
        assert_eq!(v.override_slots_at(2), Some(Vec::new()));
    }
}
