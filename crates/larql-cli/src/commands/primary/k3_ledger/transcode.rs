//! Can MXFP4 be transcoded to Q6_K *exactly*, or only with a loss label?
//!
//! Pure. The stack currently pays every cost of format conversion (full f32
//! materialisation) and collects none of the benefits, which is what produced
//! the 16-61x traffic amplification. If storage and compute formats decouple —
//! MXFP4 as the archive tier, something the kernels actually eat as the serving
//! tier — then the question is what that transcode costs in fidelity.
//!
//! ## Why it might cost nothing
//!
//! fp4 (e2m1) has value set `±{0, 0.5, 1, 1.5, 2, 3, 4, 6}`. Multiply by 2 and
//! every codepoint is an integer: `±{0, 1, 2, 3, 4, 6, 8, 12}`, all inside
//! Q6_K's signed 6-bit range (`-32..31`). So the *values* transfer exactly. What
//! has to also transfer is the **scale structure**:
//!
//! ```text
//!   MXFP4:  w = 2^(e-127) · lut[code]          e8m0 per 32 weights
//!   Q6_K:   w = d · s_sub · q                  f16 d + int8 s_sub per 16, 256/superblock
//! ```
//!
//! Setting `q = 2·lut[code]` requires `d · s_sub = 2^(e-128)`. A Q6_K sub-block
//! (16 weights) always sits inside one MXFP4 group (32), so each sub-block has
//! exactly one exponent — no straddling. Choosing `d = 2^(e_min - 128)` for the
//! super-block gives `s_sub = 2^(e_i - e_min)`, and exactness needs both:
//!
//!   **C1** `e_i - e_min <= 6` within every 256-weight super-block, so the
//!          int8 sub-scale `2^spread` stays <= 127 (2^7 = 128 overflows).
//!   **C2** `2^(e_min - 128)` representable in f16 — normals reach 2^-14, and
//!          subnormals 2^-24, so `e_min` must not sit too low.
//!
//! Both are empirical properties of the checkpoint, decided by reading scale
//! bytes. Neither is assumable: C1 fails on any super-block whose weights span
//! more than a 64x dynamic range, which is exactly what an outlier channel
//! looks like.
//!
//! If both hold, the transcode is **exact, verifiable value-by-value against
//! the banked codec** — a stronger chain of custody than a bounded-KL label.

use serde::Serialize;

use crate::collections::BTreeMap;

/// MXFP4 scale group, in weights.
pub const MXFP4_GROUP: usize = 32;
/// Q6_K super-block, in weights.
pub const Q6K_SUPERBLOCK: usize = 256;
/// MXFP4 groups spanned by one Q6_K super-block.
pub const GROUPS_PER_SUPERBLOCK: usize = Q6K_SUPERBLOCK / MXFP4_GROUP;

/// Largest exponent spread an int8 Q6_K sub-scale can express: `2^6 = 64 <= 127`,
/// `2^7 = 128` overflows.
pub const MAX_EXPONENT_SPREAD: u32 = 6;

/// f16 reaches 2^-24 via subnormals; below that `d` cannot be represented.
pub const MIN_F16_EXPONENT: i32 = -24;
/// f16's largest finite power of two.
pub const MAX_F16_EXPONENT: i32 = 15;

#[derive(Debug, Clone, Serialize)]
pub struct TranscodeScan {
    pub tensors_scanned: usize,
    pub groups_scanned: usize,
    pub superblocks: usize,
    /// Super-blocks whose exponent spread exceeds what an int8 sub-scale holds.
    pub superblocks_over_spread: usize,
    pub max_spread: u32,
    pub spread_histogram: BTreeMap<u32, usize>,
    pub e_min: u8,
    pub e_max: u8,
    /// `e_min - 128` per super-block, the exponent `d` must represent.
    pub d_exponent_min: i32,
    pub d_exponent_max: i32,
    pub superblocks_over_f16_range: usize,
    /// Sentinels the CPU reference treats specially.
    pub zero_scales: usize,
    pub nan_scales: usize,
    pub c1_spread_ok: bool,
    pub c2_f16_ok: bool,
}

impl TranscodeScan {
    /// Exact transcode is available only if every super-block satisfies both.
    pub fn exact_possible(&self) -> bool {
        self.c1_spread_ok && self.c2_f16_ok && self.nan_scales == 0
    }

    pub fn verdict(&self) -> &'static str {
        if self.exact_possible() {
            "EXACT — every MXFP4 codepoint has a Q6_K representation; \
             verify value-by-value, no KL label needed"
        } else if !self.c1_spread_ok {
            "NOT EXACT — exponent spread exceeds the int8 sub-scale range in \
             some super-blocks; Q6_K needs a loss label, or a smaller super-block"
        } else if !self.c2_f16_ok {
            "NOT EXACT — some super-block d falls outside f16's representable range"
        } else {
            "NOT EXACT — NaN scale sentinels present"
        }
    }
}

/// Scan a run of e8m0 scale bytes laid out as `[rows, groups]`.
///
/// `groups_per_row` matters: super-blocks must not straddle a row boundary,
/// because each row is an independent weight vector.
pub fn scan_scales(scales: &[u8], groups_per_row: usize) -> TranscodeScan {
    let mut hist: BTreeMap<u32, usize> = BTreeMap::new();
    let (mut e_min, mut e_max) = (u8::MAX, u8::MIN);
    let (mut d_lo, mut d_hi) = (i32::MAX, i32::MIN);
    let (mut over_spread, mut over_f16, mut zeros, mut nans, mut sbs) = (0, 0, 0, 0, 0);
    let mut max_spread = 0u32;

    for row in scales.chunks(groups_per_row) {
        for sb in row.chunks(GROUPS_PER_SUPERBLOCK) {
            if sb.len() < GROUPS_PER_SUPERBLOCK {
                continue; // partial trailing super-block: not transcodable as one
            }
            sbs += 1;
            let mut lo = u8::MAX;
            let mut hi = u8::MIN;
            for &b in sb {
                match b {
                    0 => zeros += 1,
                    255 => nans += 1,
                    _ => {}
                }
                lo = lo.min(b);
                hi = hi.max(b);
            }
            e_min = e_min.min(lo);
            e_max = e_max.max(hi);

            let spread = (hi - lo) as u32;
            max_spread = max_spread.max(spread);
            *hist.entry(spread).or_default() += 1;
            if spread > MAX_EXPONENT_SPREAD {
                over_spread += 1;
            }

            let d_exp = lo as i32 - 128;
            d_lo = d_lo.min(d_exp);
            d_hi = d_hi.max(d_exp);
            if !(MIN_F16_EXPONENT..=MAX_F16_EXPONENT).contains(&d_exp) {
                over_f16 += 1;
            }
        }
    }

    TranscodeScan {
        tensors_scanned: 1,
        groups_scanned: scales.len(),
        superblocks: sbs,
        superblocks_over_spread: over_spread,
        max_spread,
        spread_histogram: hist,
        e_min,
        e_max,
        d_exponent_min: if d_lo == i32::MAX { 0 } else { d_lo },
        d_exponent_max: if d_hi == i32::MIN { 0 } else { d_hi },
        superblocks_over_f16_range: over_f16,
        zero_scales: zeros,
        nan_scales: nans,
        c1_spread_ok: over_spread == 0,
        c2_f16_ok: over_f16 == 0,
    }
}

/// Merge scans over several tensors into one verdict.
pub fn merge(scans: &[TranscodeScan]) -> TranscodeScan {
    let mut out = TranscodeScan {
        tensors_scanned: 0,
        groups_scanned: 0,
        superblocks: 0,
        superblocks_over_spread: 0,
        max_spread: 0,
        spread_histogram: BTreeMap::new(),
        e_min: u8::MAX,
        e_max: u8::MIN,
        d_exponent_min: i32::MAX,
        d_exponent_max: i32::MIN,
        superblocks_over_f16_range: 0,
        zero_scales: 0,
        nan_scales: 0,
        c1_spread_ok: true,
        c2_f16_ok: true,
    };
    for s in scans {
        out.tensors_scanned += s.tensors_scanned;
        out.groups_scanned += s.groups_scanned;
        out.superblocks += s.superblocks;
        out.superblocks_over_spread += s.superblocks_over_spread;
        out.max_spread = out.max_spread.max(s.max_spread);
        for (k, v) in &s.spread_histogram {
            *out.spread_histogram.entry(*k).or_default() += v;
        }
        out.e_min = out.e_min.min(s.e_min);
        out.e_max = out.e_max.max(s.e_max);
        out.d_exponent_min = out.d_exponent_min.min(s.d_exponent_min);
        out.d_exponent_max = out.d_exponent_max.max(s.d_exponent_max);
        out.superblocks_over_f16_range += s.superblocks_over_f16_range;
        out.zero_scales += s.zero_scales;
        out.nan_scales += s.nan_scales;
    }
    out.c1_spread_ok = out.superblocks_over_spread == 0;
    out.c2_f16_ok = out.superblocks_over_f16_range == 0;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flat_exponent_field_transcodes_exactly() {
        let scales = vec![127u8; GROUPS_PER_SUPERBLOCK * 4];
        let s = scan_scales(&scales, GROUPS_PER_SUPERBLOCK * 4);
        assert_eq!(s.max_spread, 0);
        assert!(s.exact_possible(), "{}", s.verdict());
    }

    #[test]
    fn spread_of_exactly_six_still_fits_the_int8_sub_scale() {
        // 2^6 = 64 <= 127. The boundary is worth pinning: 2^7 = 128 does not.
        let mut scales = vec![120u8; GROUPS_PER_SUPERBLOCK];
        scales[3] = 126;
        let s = scan_scales(&scales, GROUPS_PER_SUPERBLOCK);
        assert_eq!(s.max_spread, 6);
        assert!(s.c1_spread_ok);
    }

    #[test]
    fn spread_of_seven_breaks_exactness() {
        let mut scales = vec![120u8; GROUPS_PER_SUPERBLOCK];
        scales[5] = 127;
        let s = scan_scales(&scales, GROUPS_PER_SUPERBLOCK);
        assert_eq!(s.max_spread, 7);
        assert!(!s.c1_spread_ok);
        assert!(!s.exact_possible());
        assert!(s.verdict().contains("exponent spread"));
    }

    #[test]
    fn a_very_low_exponent_escapes_f16_range_for_d() {
        // e_min = 100 -> d = 2^-28, below f16's 2^-24 subnormal floor.
        let scales = vec![100u8; GROUPS_PER_SUPERBLOCK];
        let s = scan_scales(&scales, GROUPS_PER_SUPERBLOCK);
        assert!(s.c1_spread_ok, "spread is zero here");
        assert!(
            !s.c2_f16_ok,
            "d exponent {} should be out of range",
            s.d_exponent_min
        );
        assert!(!s.exact_possible());
    }

    #[test]
    fn nan_sentinels_block_exactness() {
        let mut scales = vec![127u8; GROUPS_PER_SUPERBLOCK];
        scales[0] = 255;
        let s = scan_scales(&scales, GROUPS_PER_SUPERBLOCK);
        assert_eq!(s.nan_scales, 1);
        assert!(!s.exact_possible());
    }

    #[test]
    fn superblocks_do_not_straddle_rows() {
        // Two rows of 8 groups each: 2 super-blocks, not 1 spanning both.
        let mut scales = vec![127u8; GROUPS_PER_SUPERBLOCK * 2];
        for v in scales.iter_mut().take(GROUPS_PER_SUPERBLOCK) {
            *v = 120;
        }
        let s = scan_scales(&scales, GROUPS_PER_SUPERBLOCK);
        assert_eq!(s.superblocks, 2);
        assert_eq!(s.max_spread, 0, "row-local spreads are both zero");
    }

    #[test]
    fn partial_trailing_superblocks_are_not_counted() {
        let scales = vec![127u8; GROUPS_PER_SUPERBLOCK + 3];
        let s = scan_scales(&scales, GROUPS_PER_SUPERBLOCK + 3);
        assert_eq!(s.superblocks, 1);
    }

    #[test]
    fn merge_takes_the_worst_case_across_tensors() {
        let good = scan_scales(&[127u8; GROUPS_PER_SUPERBLOCK], GROUPS_PER_SUPERBLOCK);
        let mut bad_scales = vec![120u8; GROUPS_PER_SUPERBLOCK];
        bad_scales[1] = 130;
        let bad = scan_scales(&bad_scales, GROUPS_PER_SUPERBLOCK);
        let m = merge(&[good, bad]);
        assert_eq!(m.superblocks, 2);
        assert_eq!(m.max_spread, 10);
        assert!(!m.exact_possible());
    }
}
