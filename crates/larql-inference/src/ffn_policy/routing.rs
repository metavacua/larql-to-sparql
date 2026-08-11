//! [`RoutingPredicate`] — where in the per-layer routing space a
//! [`crate::ffn_policy::FfnLayerPolicy`] binding applies.

use core::ops::Range;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Where in the per-layer routing space a binding applies.
///
/// Today only [`Self::All`], [`Self::Layers`], and [`Self::Otherwise`]
/// — but the slot is structured so future predicates
/// (`ConfidenceAbove(f32)`, `Dispatcher(DispatcherKind)`, etc.) extend
/// without changing the outer `{ffn}@pred` syntax.
///
/// Exhaustive — **no `#[non_exhaustive]`**. Adding a variant is a
/// deliberate schema change; consumers handle every predicate kind
/// explicitly so the silent-drop pattern doesn't reproduce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutingPredicate {
    /// Every layer. Sugar for "no predicate" — useful for the
    /// single-uniform-backend form (the `dense` spec without braces
    /// resolves to `{dense}@all` internally).
    All,
    /// Explicit half-open layer ranges. Multiple ranges in one
    /// predicate are unioned (e.g. `layers=0-13,28-33`).
    Layers(Vec<Range<usize>>),
    /// Every layer not covered by an earlier binding in the same
    /// policy. There can be at most one `Otherwise` per policy, and
    /// it must appear after at least one `Layers` binding.
    Otherwise,
}

impl RoutingPredicate {
    /// Parse a predicate clause. Accepts `all`, `otherwise`, or
    /// `layers=A-B[,C-D,...]`. Returns `None` on syntactic errors
    /// (malformed ranges, unknown predicate names) so the caller can
    /// surface them as part of the policy-level error.
    ///
    /// Named `parse_clause` rather than `from_str` to avoid clippy's
    /// `manual_contains` lint against the inherent method shadowing
    /// the `std::str::FromStr::from_str` trait method — implementing
    /// `FromStr` would force an error type that doesn't fit the
    /// `None`-on-syntactic-error convention this module shares with
    /// [`super::FfnBackendKind::from_name`] and
    /// [`larql_kv::EngineKind::from_name`].
    pub fn parse_clause(spec: &str) -> Option<Self> {
        let trimmed = spec.trim();
        if trimmed.eq_ignore_ascii_case("all") {
            return Some(RoutingPredicate::All);
        }
        if trimmed.eq_ignore_ascii_case("otherwise") {
            return Some(RoutingPredicate::Otherwise);
        }
        let rest = trimmed.strip_prefix("layers=")?;
        let mut ranges = Vec::new();
        for piece in rest.split(',') {
            let piece = piece.trim();
            if piece.is_empty() {
                return None;
            }
            let (start_s, end_s) = piece.split_once('-')?;
            let start: usize = start_s.trim().parse().ok()?;
            let end: usize = end_s.trim().parse().ok()?;
            // Inclusive parse — `14-27` means L14..L27 inclusive,
            // which becomes the half-open range 14..28.
            if end < start {
                return None;
            }
            ranges.push(start..(end + 1));
        }
        if ranges.is_empty() {
            return None;
        }
        Some(RoutingPredicate::Layers(ranges))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_and_otherwise() {
        assert_eq!(
            RoutingPredicate::parse_clause("all"),
            Some(RoutingPredicate::All)
        );
        assert_eq!(
            RoutingPredicate::parse_clause("ALL"),
            Some(RoutingPredicate::All)
        );
        assert_eq!(
            RoutingPredicate::parse_clause("otherwise"),
            Some(RoutingPredicate::Otherwise)
        );
    }

    #[test]
    fn single_range() {
        let p = RoutingPredicate::parse_clause("layers=14-27").unwrap();
        match p {
            RoutingPredicate::Layers(ranges) => {
                assert_eq!(ranges, vec![14..28]);
            }
            other => panic!("expected Layers, got {other:?}"),
        }
    }

    #[test]
    fn multiple_ranges() {
        let p = RoutingPredicate::parse_clause("layers=0-13,28-33").unwrap();
        match p {
            RoutingPredicate::Layers(ranges) => {
                assert_eq!(ranges, vec![0..14, 28..34]);
            }
            other => panic!("expected Layers, got {other:?}"),
        }
    }

    #[test]
    fn malformed_range_returns_none() {
        assert!(RoutingPredicate::parse_clause("layers=abc-27").is_none());
        assert!(RoutingPredicate::parse_clause("layers=14-").is_none());
        assert!(RoutingPredicate::parse_clause("layers=").is_none());
        // end < start
        assert!(RoutingPredicate::parse_clause("layers=27-14").is_none());
    }

    #[test]
    fn unknown_returns_none() {
        assert!(RoutingPredicate::parse_clause("unknown=42").is_none());
        assert!(RoutingPredicate::parse_clause("").is_none());
    }
}
