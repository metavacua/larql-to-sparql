//! `impl PatchOverrides for VectorIndex`.
//!
//! The patch-overlay capability — direct HashMap lookups against the
//! per-feature override tables in `metadata`. Every other capability
//! impl on `VectorIndex` is a delegation shim; this one is the actual
//! implementation, because there's no inherent method to delegate to.

use super::VectorIndex;
use crate::index::types::PatchOverrides;

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
}
