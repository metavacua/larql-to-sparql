//! Is there an **exact variable-rate** MXFP4 code below the 4-bit fixed floor?
//!
//! Pure. `serving_format` closed the fixed-width question by counting the
//! alphabet: 15 distinct values, so 4 bits, so MXFP4 is already optimal *for a
//! fixed-width code*. That left one loophole — four bits is the fixed-rate
//! floor, not the average-entropy floor. If the symbols were locally skewed, a
//! palette, escape or residual-plane format could be exact and cheaper.
//!
//! This module prices those candidates against a measured symbol distribution.
//! The formulas are here, tested, and separate from the fetch that feeds them.
//!
//! ## The trap this module exists to avoid
//!
//! Plug-in entropy over a small tile is **biased downward**. A `B`-weight tile
//! drawn from a 16-symbol alphabet reports entropy about `(K-1)/(2·B·ln2)` bits
//! below the truth *however uniform the source is* — 0.34 bits at `B = 32`.
//!
//! That is large enough to look like a 10% compression opportunity, and it
//! scales as exactly `1/B`. **A real local concentration would not.** So the
//! diagnostic is not "is local entropy below global" — it always is — but
//! "does the gap exceed the bias, and does it survive as `B` grows".
//! `local_skew_is_bias_shaped` is that test.

use serde::Serialize;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// fp4 code width: the alphabet is 16 nibbles (15 distinct values).
pub const FP4_CODES: usize = 16;
/// Payload bits a fixed-width exact code spends — `serving_format`'s floor.
pub const FIXED_PAYLOAD_BITS: f64 = 4.0;
/// Bits a tile spends naming which mode it uses.
pub const MODE_TAG_BITS: f64 = 2.0;
/// Bits to store one palette entry (a raw MXFP4 nibble).
pub const PALETTE_ENTRY_BITS: f64 = 4.0;

/// `+0` and `-0` reconstruct to the same value, so an exact *value* code may
/// merge them. Bit patterns differ; products and sums do not.
pub const NEG_ZERO_CODE: usize = 8;
pub const POS_ZERO_CODE: usize = 0;

/// Symbol frequencies over some population of MXFP4 nibbles.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SymbolCounts {
    pub counts: [u64; FP4_CODES],
}

impl SymbolCounts {
    /// Accumulate both nibbles of each packed byte, low then high.
    pub fn add_packed(&mut self, packed: &[u8]) {
        for &b in packed {
            self.counts[(b & 0xF) as usize] += 1;
            self.counts[(b >> 4) as usize] += 1;
        }
    }

    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }

    /// Symbols that actually occur.
    pub fn distinct(&self) -> usize {
        self.counts.iter().filter(|&&c| c > 0).count()
    }

    fn probs(&self) -> Vec<f64> {
        let n = self.total() as f64;
        if n == 0.0 {
            return Vec::new();
        }
        self.counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| c as f64 / n)
            .collect()
    }

    /// Shannon entropy in bits — the floor for *any* exact code, variable-rate
    /// included, and reachable only by a serial arithmetic coder.
    pub fn entropy(&self) -> f64 {
        -self.probs().iter().map(|p| p * p.log2()).sum::<f64>()
    }

    /// Entropy once `+0`/`-0` are treated as one value. Free and value-exact.
    pub fn entropy_zero_merged(&self) -> f64 {
        let mut c = self.clone();
        c.counts[POS_ZERO_CODE] += c.counts[NEG_ZERO_CODE];
        c.counts[NEG_ZERO_CODE] = 0;
        c.entropy()
    }

    /// Probability mass of the `k` most common symbols.
    pub fn top_k_coverage(&self, k: usize) -> f64 {
        let mut p = self.probs();
        p.sort_by(|a, b| b.partial_cmp(a).unwrap());
        p.into_iter().take(k).sum()
    }

    /// Smallest achievable "rare member" mass when symbols are paired into
    /// `FP4_CODES / 2` classes — pair each common symbol with a rare one, so the
    /// minimum is the sum of the smaller half.
    pub fn optimal_pairing_rare_mass(&self) -> f64 {
        let mut p = self.probs();
        p.sort_by(|a, b| b.partial_cmp(a).unwrap());
        p.into_iter().skip(FP4_CODES / 2).sum()
    }
}

/// Cardinality and local-entropy summary over fixed-size tiles.
#[derive(Debug, Clone, Default, Serialize)]
pub struct TileStats {
    pub block: usize,
    pub tiles: u64,
    pub min_cardinality: usize,
    pub median_cardinality: usize,
    pub max_cardinality: usize,
    /// Tiles a 2-bit / 3-bit palette mode could actually encode.
    pub fits_two_bit: u64,
    pub fits_three_bit: u64,
    /// Weight-averaged plug-in entropy per tile. Compare against the global
    /// figure *only* through `local_skew_is_bias_shaped`.
    pub mean_local_entropy: f64,
}

/// Tile a nibble stream row by row. Tiles never straddle a row, because each
/// row is an independent weight vector — the same constraint `transcode` keeps.
pub fn tile_stats(nibbles: &[u8], row_len: usize, block: usize) -> TileStats {
    let mut hist = [0u64; FP4_CODES + 1];
    let (mut tiles, mut ent_sum) = (0u64, 0.0);
    if row_len == 0 || block == 0 || block > row_len {
        return TileStats {
            block,
            ..Default::default()
        };
    }
    for row in nibbles.chunks(row_len) {
        for tile in row.chunks(block) {
            if tile.len() < block {
                continue; // partial trailing tile: not encodable as one
            }
            let mut c = SymbolCounts::default();
            for &n in tile {
                c.counts[n as usize] += 1;
            }
            hist[c.distinct()] += 1;
            ent_sum += c.entropy();
            tiles += 1;
        }
    }
    let nth = |target: u64| {
        let mut seen = 0;
        hist.iter()
            .position(|&h| {
                seen += h;
                seen > target
            })
            .unwrap_or(0)
    };
    TileStats {
        block,
        tiles,
        min_cardinality: hist.iter().position(|&h| h > 0).unwrap_or(0),
        median_cardinality: nth(tiles / 2),
        max_cardinality: hist.iter().rposition(|&h| h > 0).unwrap_or(0),
        fits_two_bit: hist[..=4].iter().sum(),
        fits_three_bit: hist[..=8].iter().sum(),
        mean_local_entropy: if tiles == 0 {
            0.0
        } else {
            ent_sum / tiles as f64
        },
    }
}

/// Downward bias of plug-in entropy over `n` samples of `k` symbols.
pub fn miller_madow_bias(k: usize, n: u64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    (k.saturating_sub(1)) as f64 / (2.0 * n as f64 * core::f64::consts::LN_2)
}

/// Does an observed global-minus-local entropy gap look like estimator bias
/// rather than real local concentration?
///
/// True when the gap is within `tolerance` bits of the predicted bias. A gap
/// that tracks `(K-1)/(2B ln2)` across block sizes is the estimator talking.
pub fn local_skew_is_bias_shaped(gap_bits: f64, k: usize, block: u64, tolerance: f64) -> bool {
    (gap_bits - miller_madow_bias(k, block)).abs() <= tolerance
}

/// Payload bits for a per-tile palette holding `cardinality` symbols.
///
/// `None` when the tile needs the full 4-bit code, i.e. the mode cannot fire.
pub fn palette_bpw(cardinality: usize, block: u64) -> Option<f64> {
    if cardinality == 0 || block == 0 {
        return None;
    }
    let width = (cardinality as f64).log2().ceil().max(1.0);
    if width >= FIXED_PAYLOAD_BITS {
        return None;
    }
    let overhead = (PALETTE_ENTRY_BITS * cardinality as f64 + MODE_TAG_BITS) / block as f64;
    Some(width + overhead)
}

/// Payload bits for a `hot_bits`-wide code: `2^hot_bits - 1` hot symbols plus
/// one escape, escaped values carried as raw nibbles in a side stream.
pub fn escape_bpw(hot_bits: u32, coverage: f64, block: u64) -> f64 {
    let hot = (1u32 << hot_bits).saturating_sub(1) as f64;
    let overhead = (PALETTE_ENTRY_BITS * hot + MODE_TAG_BITS) / block.max(1) as f64;
    hot_bits as f64 + (1.0 - coverage) * FIXED_PAYLOAD_BITS + overhead
}

/// Payload bits for a 3-bit primary code plus a sparse residual position list.
pub fn residual_plane_bpw(rare_mass: f64, block: u64) -> f64 {
    let primary = (FP4_CODES as f64 / 2.0).log2();
    primary + rare_mass * (block.max(2) as f64).log2()
}

/// Every exact code, fixed or variable, must clear this to be worth building.
pub fn beats_fixed_floor(payload_bpw: f64) -> bool {
    payload_bpw < FIXED_PAYLOAD_BITS
}

/// Efficiency a denser format must sustain to beat an incumbent on speed.
///
/// Speed goes as `eta / bpw`, so bytes alone never decide it — the score is the
/// ratio, which is why a 4.5-bpw format at 0.89 beats a 4.1-bpw format at 0.70.
pub fn break_even_eta(candidate_bpw: f64, incumbent_bpw: f64, incumbent_eta: f64) -> f64 {
    incumbent_eta * candidate_bpw / incumbent_bpw
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "expected ~{b}, got {a}");
    }

    fn uniform() -> SymbolCounts {
        SymbolCounts {
            counts: [64; FP4_CODES],
        }
    }

    /// The measured K3 expert distribution (7.86 M weights, layer 49, 9 tensors).
    fn measured_k3() -> SymbolCounts {
        SymbolCounts {
            counts: [
                576, 1097, 956, 768, 776, 529, 249, 50, // +0 .. +6
                576, 1098, 957, 767, 774, 528, 249, 50, // -0 .. -6
            ],
        }
    }

    #[test]
    fn packed_bytes_split_into_low_then_high_nibbles() {
        let mut c = SymbolCounts::default();
        c.add_packed(&[0x37]);
        assert_eq!(c.counts[7], 1, "low nibble");
        assert_eq!(c.counts[3], 1, "high nibble");
        assert_eq!(c.total(), 2);
    }

    #[test]
    fn uniform_sixteen_symbols_carry_exactly_four_bits() {
        approx(uniform().entropy(), 4.0, 1e-12);
        assert_eq!(uniform().distinct(), 16);
        assert!(!beats_fixed_floor(uniform().entropy()));
    }

    #[test]
    fn the_measured_bank_sits_just_under_the_fixed_floor() {
        // 3.75 bits: the entire variable-rate prize is 0.25 bits, and only an
        // ideal serial arithmetic coder collects it.
        let c = measured_k3();
        approx(c.entropy(), 3.7525, 0.005);
        assert!(
            c.entropy() > 3.7,
            "fp4's log grid is already near-entropy-matched"
        );
    }

    #[test]
    fn merging_the_two_zeros_is_free_and_worth_a_tenth_of_a_bit() {
        let c = measured_k3();
        approx(c.entropy() - c.entropy_zero_merged(), 0.115, 0.01);
    }

    #[test]
    fn merging_zeros_is_a_no_op_when_one_of_them_is_absent() {
        let mut c = uniform();
        c.counts[NEG_ZERO_CODE] = 0;
        approx(c.entropy(), c.entropy_zero_merged(), 1e-12);
    }

    #[test]
    fn measured_top_k_coverage_kills_both_escape_widths() {
        let c = measured_k3();
        // 3-bit code: 7 hot symbols cover 64%, so 36% escape to 4-bit nibbles.
        approx(c.top_k_coverage(7), 0.643, 0.01);
        assert!(!beats_fixed_floor(escape_bpw(3, c.top_k_coverage(7), 256)));
        // 2-bit code: 3 hot symbols cover only 32%.
        approx(c.top_k_coverage(3), 0.315, 0.01);
        assert!(!beats_fixed_floor(escape_bpw(2, c.top_k_coverage(3), 256)));
    }

    #[test]
    fn an_escape_format_needs_implausibly_high_coverage_to_win() {
        // Solve the bar rather than asserting the measured miss: a 3-bit code
        // must cover ~78% with 7 symbols. The measured bank gives 64%.
        let need = (1..100)
            .map(|i| i as f64 / 100.0)
            .find(|&cov| beats_fixed_floor(escape_bpw(3, cov, 256)))
            .unwrap();
        approx(need, 0.78, 0.02);
        assert!(measured_k3().top_k_coverage(7) < need);
    }

    #[test]
    fn the_residual_plane_needs_a_rare_side_the_bank_does_not_have() {
        let c = measured_k3();
        // Optimal pairing still leaves 28% on the rare side; the bar is ~12.5%.
        approx(c.optimal_pairing_rare_mass(), 0.281, 0.01);
        assert!(!beats_fixed_floor(residual_plane_bpw(
            c.optimal_pairing_rare_mass(),
            256
        )));
        assert!(beats_fixed_floor(residual_plane_bpw(0.10, 256)));
    }

    #[test]
    fn a_palette_mode_cannot_fire_once_a_tile_holds_nine_symbols() {
        // Measured minimum tile cardinality was 10 at B=64, 13 at B=256.
        assert!(palette_bpw(4, 256).is_some());
        assert!(palette_bpw(8, 256).is_some());
        assert!(
            palette_bpw(9, 256).is_none(),
            "9 symbols needs the full nibble"
        );
        assert!(palette_bpw(13, 256).is_none());
    }

    #[test]
    fn palette_overhead_shrinks_with_the_tile_but_the_code_width_does_not() {
        let small = palette_bpw(8, 64).unwrap();
        let large = palette_bpw(8, 512).unwrap();
        assert!(large < small);
        assert!(large > 3.0, "the 3-bit code width is irreducible");
    }

    #[test]
    fn the_entropy_bias_is_large_at_small_tiles_and_scales_as_one_over_b() {
        // 0.34 bits at B=32 — big enough to masquerade as a 10% opportunity.
        approx(miller_madow_bias(FP4_CODES, 32), 0.338, 0.005);
        approx(
            miller_madow_bias(FP4_CODES, 32) / miller_madow_bias(FP4_CODES, 64),
            2.0,
            1e-9,
        );
    }

    #[test]
    fn the_measured_local_skew_is_bias_shaped_at_every_block_size() {
        // Measured global-minus-local gaps. All within 0.06 bits of the
        // predicted bias, and themselves shrinking as 1/B — so there is no
        // scale-invariant local structure to encode.
        for (block, gap) in [(32u64, 0.3931), (64, 0.1991), (256, 0.0518), (512, 0.0259)] {
            assert!(
                local_skew_is_bias_shaped(gap, FP4_CODES, block, 0.06),
                "B={block} gap {gap} should be explained by bias"
            );
        }
    }

    #[test]
    fn a_genuine_local_concentration_would_not_look_like_bias() {
        // Guard the guard: half a bit at B=512 is real structure, not estimator
        // noise, and the test must be able to say so.
        assert!(!local_skew_is_bias_shaped(0.5, FP4_CODES, 512, 0.06));
    }

    #[test]
    fn break_even_eta_prices_the_entropy_floor_against_the_fixed_floor() {
        // Even a free arithmetic coder must hold eta 0.81 to beat the fixed
        // exact format — while decoding serially.
        approx(break_even_eta(3.731, 4.09375, 0.89), 0.811, 0.005);
        // And a denser-but-slower format loses: the score is eta/bpw, not bpw.
        assert!(break_even_eta(4.1, 4.5, 0.89) > 0.70);
    }

    #[test]
    fn empty_counts_do_not_divide_by_zero() {
        let c = SymbolCounts::default();
        assert_eq!(c.total(), 0);
        approx(c.entropy(), 0.0, 1e-12);
        approx(c.top_k_coverage(7), 0.0, 1e-12);
        approx(miller_madow_bias(FP4_CODES, 0), 0.0, 1e-12);
        assert!(palette_bpw(0, 256).is_none());
    }
}
