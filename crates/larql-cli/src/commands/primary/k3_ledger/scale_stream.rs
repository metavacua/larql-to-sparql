//! Fold the e8m0 scale channel into the census alphabet.
//!
//! Pure. The conditional census works over a 16-symbol alphabet, but MXFP4
//! scales are raw e8m0 bytes with 256 codepoints. Folding is safe here for a
//! measured reason rather than a convenient one: the K3 expert scale field was
//! scanned over 1,032,192 superblocks and every exponent landed in **119..122**,
//! a span of four. A 16-wide window centred on the observed minimum therefore
//! has twelve codepoints of headroom.
//!
//! It is still a *reduction*, so it is instrumented: [`FoldedScales::escapes`]
//! counts bytes that did not fit, and any non-zero count invalidates the arm
//! rather than being clipped away. That is the difference between a fold and a
//! clamp.
//!
//! ## What the scale channel can be worth, before it is measured
//!
//! Raw e8m0 is 8 bits per 32 weights = 0.25 bpw, but `AdaptiveSuperblockDelta`
//! already reaches **0.0673 bpw** with no conditioning at all. So the entire
//! prize available to conditional scale coding is 0.0673 bpw — third-order
//! against a payload term of 4.0. Measuring it closes the question; it was never
//! going to decide it.

use serde::Serialize;

use super::symbol_census::FP4_CODES;
use super::transcode::MXFP4_GROUP;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Scale bytes the checkpoint spends per weight, unconditioned.
pub const RAW_SCALE_BPW: f64 = 8.0 / MXFP4_GROUP as f64;

/// Sentinel e8m0 values the CPU reference treats specially; they carry no
/// exponent and must not enter the window.
pub const ZERO_SENTINEL: u8 = 0;
pub const NAN_SENTINEL: u8 = 255;

#[derive(Debug, Clone, Serialize)]
pub struct FoldedScales {
    /// Symbols in `0..16`, one per MXFP4 group, relative to `base`.
    #[serde(skip)]
    pub symbols: Vec<u8>,
    pub base: u8,
    /// Bytes outside the window. Non-zero invalidates the fold.
    pub escapes: usize,
    pub sentinels: usize,
}

impl FoldedScales {
    pub fn is_valid(&self) -> bool {
        self.escapes == 0
    }
}

/// Smallest non-sentinel exponent present, the natural window base.
pub fn window_base(scales: &[u8]) -> u8 {
    scales
        .iter()
        .copied()
        .filter(|&b| b != ZERO_SENTINEL && b != NAN_SENTINEL)
        .min()
        .unwrap_or(0)
}

/// Fold to `0..16` relative to `base`. Sentinels map to the top codepoint so
/// they stay distinguishable from a real exponent of `base + 15`, which the
/// measured four-wide range leaves unoccupied.
pub fn fold(scales: &[u8], base: u8) -> FoldedScales {
    let top = (FP4_CODES - 1) as u8;
    let mut symbols = Vec::with_capacity(scales.len());
    let (mut escapes, mut sentinels) = (0, 0);
    for &b in scales {
        if b == ZERO_SENTINEL || b == NAN_SENTINEL {
            sentinels += 1;
            symbols.push(top);
            continue;
        }
        match b.checked_sub(base) {
            Some(d) if d < top => symbols.push(d),
            _ => {
                escapes += 1;
                symbols.push(top);
            }
        }
    }
    FoldedScales {
        symbols,
        base,
        escapes,
        sentinels,
    }
}

/// Conditional scale entropy expressed per *weight* rather than per group —
/// the unit the transport ledger adds up.
pub fn bits_per_weight(conditional_entropy_per_group: f64) -> f64 {
    conditional_entropy_per_group / MXFP4_GROUP as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "expected ~{b}, got {a}");
    }

    #[test]
    fn the_measured_k3_range_folds_with_room_to_spare() {
        let scales = [119u8, 120, 121, 122, 119, 122];
        let base = window_base(&scales);
        assert_eq!(base, 119);
        let f = fold(&scales, base);
        assert!(f.is_valid());
        assert_eq!(f.symbols, vec![0, 1, 2, 3, 0, 3]);
        assert_eq!(f.sentinels, 0);
    }

    #[test]
    fn an_exponent_outside_the_window_escapes_rather_than_clipping_silently() {
        let scales = [100u8, 140];
        let f = fold(&scales, 100);
        assert_eq!(f.escapes, 1, "140 is 40 above the base");
        assert!(!f.is_valid(), "an escape must invalidate the fold");
    }

    #[test]
    fn an_exponent_below_the_base_also_escapes() {
        let f = fold(&[118u8], 120);
        assert_eq!(f.escapes, 1);
        assert!(!f.is_valid());
    }

    #[test]
    fn sentinels_are_counted_separately_and_do_not_invalidate() {
        let f = fold(&[ZERO_SENTINEL, 120, NAN_SENTINEL], 120);
        assert_eq!(f.sentinels, 2);
        assert_eq!(f.escapes, 0);
        assert!(f.is_valid());
        assert_eq!(f.symbols, vec![15, 0, 15]);
    }

    #[test]
    fn the_window_base_ignores_sentinels() {
        assert_eq!(window_base(&[ZERO_SENTINEL, 121, 119, NAN_SENTINEL]), 119);
        assert_eq!(window_base(&[]), 0);
        assert_eq!(window_base(&[ZERO_SENTINEL]), 0);
    }

    #[test]
    fn the_boundary_codepoint_is_reserved_for_escapes_not_for_data() {
        // base + 14 is the last usable exponent; base + 15 is the escape slot.
        assert!(fold(&[134u8], 120).is_valid());
        assert!(!fold(&[135u8], 120).is_valid());
    }

    #[test]
    fn the_raw_scale_rate_is_a_quarter_bit_and_the_conversion_matches_it() {
        approx(RAW_SCALE_BPW, 0.25, 1e-12);
        // 8 bits of conditional entropy per group is the unconditioned rate.
        approx(bits_per_weight(8.0), RAW_SCALE_BPW, 1e-12);
        approx(bits_per_weight(0.0), 0.0, 1e-12);
    }

    #[test]
    fn folding_an_empty_stream_is_valid_and_empty() {
        let f = fold(&[], 0);
        assert!(f.is_valid());
        assert!(f.symbols.is_empty());
    }
}
