//! What is the smallest **exact** serving container for a K3 MXFP4 expert bank?
//!
//! Pure. `transcode` answered "does Q6_K hold MXFP4 exactly" by *scanning* real
//! scale bytes, because the answer depended on an empirical property of the
//! checkpoint (exponent spread). This module answers the sibling question —
//! *which containers could hold it at all, and at what width* — by **counting**,
//! because that answer does not depend on the checkpoint at all.
//!
//! The distinction matters: a search over candidate formats would spend a
//! fortnight rediscovering a bound that falls out of the source alphabet in one
//! paragraph.
//!
//! ## The payload floor is 4 bits, and it is not improvable
//!
//! K3's experts are stored MXFP4. Inside one 32-weight group every weight is
//! `2^(e-127) * L[c]` for a 4-bit code `c`, so the set of values a group can
//! reconstruct has **at most 15 distinct members** (the LUT's 16 entries, less
//! the `+0`/`-0` collapse). Any fixed-width exact code therefore needs
//! `ceil(log2 15) = 4` bits — which is exactly what MXFP4 already spends.
//!
//! **There is no exact format below 4.0 payload bits.** Not "none was found":
//! none exists, short of entropy-coding the code histogram, which trades the
//! constant-time indexed load a GPU kernel needs for a serial dependency.
//!
//! ## Which containers can hold a *non-uniform* alphabet
//!
//! Doubled, the alphabet is `±{0,1,2,3,4,6,8,12}` — integers, but **not an
//! arithmetic progression** (gaps run 1,1,1,2,2,4). K-quants dequantise on an
//! affine grid `w = A*q + B`, so they can only represent an AP. The smallest AP
//! containing the alphabet has step `gcd(differences) = 1` and spans `-12..12`,
//! so it needs **25 levels**. That single number decides every container:
//!
//! | container | levels | exact? | all-in bpw | kernel today |
//! |---|---|---|---|---|
//! | Q4_0 / Q4_K | 16 | **no** | 4.50 | yes |
//! | Q5_K | 32 | yes | 5.50 | **no** |
//! | Q6_K | 64 | yes | 6.5625 | yes |
//! | Q8_0 | 256 | yes | 8.50 | yes |
//! | MXFP4 (LUT) | — | yes by construction | 4.25 | **no** |
//!
//! Two consequences the ladder should not have to measure:
//!
//! 1. **Q4_K can never be exact for MXFP4.** Not marginally — it is 9 levels
//!    short, so no scan, no scale trick and no sub-block size rescues it. A
//!    "Q4-style integer candidate" is a dead rung.
//! 2. **Q6_K is the cheapest exact container that has a kernel today**, which is
//!    why the current path is right and why it costs 1.544x the bytes.
//!
//! Q5_K is exact and 1.06 bpw cheaper than Q6_K, but has no Metal kernel — so it
//! costs the same kernel-writing work as native MXFP4 while shipping 1.25 bpw
//! more. It is **strictly dominated**: it should never be built.
//!
//! ## What is actually left to win
//!
//! Only the scale stream. MXFP4 spends 8 bits per 32 weights = 0.25 bpw on
//! e8m0, and the measured census (`transcode-scan`) says that stream is nearly
//! constant. Compressing it exactly is real but **small**: 0.25 -> 0.067 bpw is
//! a 4.3% cut to expert bytes, worth ~1.6% end-to-end. Priced here so it is not
//! mistaken for a lever of the same class as the container choice.
//!
//! The width is measured, not assumed: **97.12% of superblocks have exponent
//! spread <= 1**, so a 2-bit delta everywhere overpays for a 2.88% tail. See
//! `AdaptiveSuperblockDelta`.

use serde::Serialize;

use super::transcode::{GROUPS_PER_SUPERBLOCK, MXFP4_GROUP};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// fp4 (e2m1) magnitudes. Mirrors `larql_models::quant::mxfp4::MXFP4_TABLE`.
pub const FP4_MAGNITUDES: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// The alphabet scaled by 2 so every codepoint is an integer, sorted ascending,
/// with `+0`/`-0` collapsed. This is the set any exact container must represent.
pub const FP4_INTEGER_ALPHABET: [i32; 15] = [-12, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 12];

/// Measured over 1,032,192 superblocks in 24 real K3 expert scale tensors
/// (`larql k3-ledger transcode-scan`, 2026-07-31). Held here so the format
/// arithmetic states its empirical inputs rather than assuming them.
pub const MEASURED_MAX_SPREAD: u32 = 2;
/// Observed e8m0 range across the same population: 4 distinct exponents.
pub const MEASURED_DISTINCT_EXPONENTS: u32 = 4;

fn gcd(a: i32, b: i32) -> i32 {
    if b == 0 {
        a.abs()
    } else {
        gcd(b, a % b)
    }
}

/// Distinct values one MXFP4 group can reconstruct.
pub fn distinct_codepoints() -> usize {
    FP4_INTEGER_ALPHABET.len()
}

/// Payload bits any fixed-width exact code must spend. The floor, not a target.
pub fn min_payload_bits() -> u32 {
    (distinct_codepoints() as f64).log2().ceil() as u32
}

/// Levels an affine grid `w = A*q + B` needs to contain the alphabet.
///
/// The alphabet is symmetric about zero, so this also bounds the *linear*
/// containers (`B = 0`, signed `q`) that Q6_K and Q8_0 use.
pub fn affine_levels_required() -> u32 {
    let step = FP4_INTEGER_ALPHABET
        .windows(2)
        .fold(0, |acc, p| gcd(acc, p[1] - p[0]));
    let span = FP4_INTEGER_ALPHABET[FP4_INTEGER_ALPHABET.len() - 1] - FP4_INTEGER_ALPHABET[0];
    (span / step) as u32 + 1
}

/// A candidate serving container for the expert bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum Container {
    /// Native MXFP4: a 4-bit LUT index plus an e8m0 group scale.
    Mxfp4,
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
}

/// How far a container's Metal support actually reaches.
///
/// "Has a kernel" was too coarse and produced a wrong claim: MXFP4 *does* have
/// Metal kernels (`mxfp4_matvec`, plus the K2 grouped tournament arms) but is
/// absent from `QuantMatVec`'s format dispatch, so it cannot serve. Ordered so
/// that "cheapest exact container that can actually serve today" is expressible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum KernelMaturity {
    /// No Metal kernel at all.
    None,
    /// A standalone or diagnostic kernel exists.
    ///
    /// No container sits here today — MXFP4, the one that would, has already
    /// reached `Grouped`. The rung is kept because it is what separates "a
    /// kernel exists" from "a kernel serves", and collapsing the ladder back
    /// towards the coarse "has a kernel" question is precisely what produced
    /// the wrong MXFP4 claim recorded above. Ordering is pinned by
    /// `kernel_maturity_ladder_ascends_and_servability_splits_it_once`.
    #[allow(dead_code)]
    Standalone,
    /// A grouped-expert kernel exists (may still be a candidate, not a path).
    Grouped,
    /// Reachable through `QuantMatVec`'s format dispatch.
    Dispatched,
    /// Wired into production inference.
    Production,
}

impl KernelMaturity {
    /// Can this container serve a real forward pass today?
    pub fn is_servable(self) -> bool {
        self >= Self::Dispatched
    }
}

/// Provenance, re-grepped 2026-08-01: `QuantMatVec` dispatches
/// Q4_K/Q4_KF/Q6_K/Q4_0/Q8_0. `larql-compute-metal` additionally carries
/// `shaders::mxfp4_matvec` (K1) and `shaders::mxfp4_grouped_experts` (K2, four
/// tournament arms), neither reachable from the dispatch. No `q5` kernel exists.
impl Container {
    pub fn kernel_maturity(self) -> KernelMaturity {
        match self {
            // K1 standalone + K2 grouped candidates, but no serving dispatch.
            Self::Mxfp4 => KernelMaturity::Grouped,
            Self::Q4K | Self::Q6K => KernelMaturity::Production,
            Self::Q8_0 => KernelMaturity::Dispatched,
            Self::Q5K => KernelMaturity::None,
        }
    }

    /// Levels the container's dequant grid offers, or `None` for a LUT codec,
    /// which is not a grid at all and holds any 16-entry alphabet by fiat.
    pub fn grid_levels(self) -> Option<u32> {
        match self {
            Self::Mxfp4 => None,
            Self::Q4K => Some(16),
            Self::Q5K => Some(32),
            Self::Q6K => Some(64),
            Self::Q8_0 => Some(256),
        }
    }

    /// All-in bits per weight: block bytes over block elements.
    pub fn all_in_bits(self) -> f64 {
        match self {
            // 4-bit codeword + one 8-bit e8m0 scale per 32.
            Self::Mxfp4 => min_payload_bits() as f64 + ScaleEncoding::RawE8M0.bits_per_weight(),
            Self::Q4K => 144.0 * 8.0 / 256.0,
            Self::Q5K => 176.0 * 8.0 / 256.0,
            Self::Q6K => 210.0 * 8.0 / 256.0,
            Self::Q8_0 => 34.0 * 8.0 / 32.0,
        }
    }

    /// Can this container reconstruct every MXFP4 codepoint bit-exactly?
    ///
    /// A LUT codec can by construction. A grid codec can iff it has the levels;
    /// whether the *scales* also transfer is a separate, empirical question
    /// answered by `transcode::scan_scales`.
    pub fn holds_fp4_exactly(self) -> bool {
        self.grid_levels()
            .is_none_or(|levels| levels >= affine_levels_required())
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Mxfp4 => "MXFP4 (native LUT)",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8_0 => "Q8_0",
        }
    }
}

/// Exact encodings for the e8m0 scale stream, which is all that is left to
/// compress once the container is MXFP4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum ScaleEncoding {
    /// One e8m0 byte per group, as the checkpoint stores it.
    RawE8M0,
    /// One e8m0 base per superblock, plus a 2-bit delta per group.
    SuperblockBaseDelta2,
    /// Per-superblock base, plus a **1-bit** delta where the measured spread
    /// allows it and a 2-bit delta elsewhere, selected by one mode bit.
    ///
    /// The 2-bit width was never needed by most of the bank: **97.12% of
    /// superblocks have spread <= 1** (measured, 30,720 superblocks). Paying two
    /// bits everywhere to cover a 2.88% tail is the mistake this variant fixes.
    AdaptiveSuperblockDelta,
    /// A model-wide 4-entry exponent table, plus a 2-bit selector per group.
    GlobalTable2Bit,
}

/// Fraction of superblocks whose exponent spread fits a 1-bit delta.
/// Measured over 30,720 superblocks of K3 layer 49 (`symbol-census` corpus).
pub const MEASURED_SPREAD_LE1: f64 = 0.9712;
/// Bits per superblock naming which delta width it uses.
pub const DELTA_MODE_BITS: f64 = 1.0;

impl ScaleEncoding {
    /// Amortised scale bits per 32-weight group, base included.
    pub fn bits_per_group(self) -> f64 {
        match self {
            Self::RawE8M0 => 8.0,
            Self::SuperblockBaseDelta2 => 8.0 / GROUPS_PER_SUPERBLOCK as f64 + 2.0,
            Self::AdaptiveSuperblockDelta => {
                let delta = MEASURED_SPREAD_LE1 + 2.0 * (1.0 - MEASURED_SPREAD_LE1);
                (8.0 + DELTA_MODE_BITS) / GROUPS_PER_SUPERBLOCK as f64 + delta
            }
            Self::GlobalTable2Bit => 2.0,
        }
    }

    pub fn bits_per_weight(self) -> f64 {
        self.bits_per_group() / MXFP4_GROUP as f64
    }

    /// Largest within-superblock exponent spread this encoding survives.
    /// `None` = unbounded.
    pub fn max_spread(self) -> Option<u32> {
        match self {
            Self::RawE8M0 => None,
            Self::SuperblockBaseDelta2 => Some(3),
            // The 2-bit fallback arm covers the tail, so the reach is unchanged.
            Self::AdaptiveSuperblockDelta => Some(3),
            Self::GlobalTable2Bit => Some(3),
        }
    }

    /// Does exactness depend on a *global* property no local base can absorb?
    ///
    /// `GlobalTable2Bit` fits the measured 4-exponent range with **zero
    /// headroom**: one tensor anywhere outside 119..122 breaks it, so an
    /// exception path stops being optional. `SuperblockBaseDelta2` re-bases per
    /// superblock and so has no global exposure at all.
    pub fn needs_exception_path(self) -> bool {
        matches!(self, Self::GlobalTable2Bit)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::RawE8M0 => "raw e8m0 per group",
            Self::SuperblockBaseDelta2 => "superblock base + 2-bit delta",
            Self::AdaptiveSuperblockDelta => "superblock base + adaptive 1/2-bit delta",
            Self::GlobalTable2Bit => "global 4-entry table + 2-bit selector",
        }
    }
}

/// A fully-specified candidate: container plus, for MXFP4, its scale encoding.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ServingFormat {
    pub container: Container,
    /// Consulted only for `Container::Mxfp4`; K-quant blocks carry their scale
    /// structure inside a fixed block layout that cannot be re-encoded piecemeal.
    pub scale: ScaleEncoding,
}

impl ServingFormat {
    pub fn new(container: Container, scale: ScaleEncoding) -> Self {
        Self { container, scale }
    }

    pub fn all_in_bits(&self) -> f64 {
        match self.container {
            Container::Mxfp4 => min_payload_bits() as f64 + self.scale.bits_per_weight(),
            other => other.all_in_bits(),
        }
    }

    /// Exact against the K3 checkpoint: the container must hold the alphabet,
    /// and the scale encoding must survive the measured spread.
    pub fn is_exact_for_k3(&self) -> bool {
        let scale_ok = match self.container {
            Container::Mxfp4 => self
                .scale
                .max_spread()
                .is_none_or(|m| m >= MEASURED_MAX_SPREAD),
            _ => true,
        };
        self.container.holds_fp4_exactly() && scale_ok
    }
}

/// The cheapest format that is exact for K3 **and** carries no global exposure:
/// native MXFP4 payload with a per-superblock re-based scale stream.
///
/// This is the bar below which no encoding result can go. Compose it against the
/// class census to turn it into a tok/s ceiling rather than quoting one.
pub fn exact_floor() -> ServingFormat {
    ServingFormat::new(Container::Mxfp4, ScaleEncoding::AdaptiveSuperblockDelta)
}

/// Every candidate worth pricing, cheapest exact first.
pub fn candidates() -> Vec<ServingFormat> {
    use Container::*;
    use ScaleEncoding::*;
    vec![
        ServingFormat::new(Mxfp4, GlobalTable2Bit),
        ServingFormat::new(Mxfp4, AdaptiveSuperblockDelta),
        ServingFormat::new(Mxfp4, SuperblockBaseDelta2),
        ServingFormat::new(Mxfp4, RawE8M0),
        ServingFormat::new(Q4K, RawE8M0),
        ServingFormat::new(Q5K, RawE8M0),
        ServingFormat::new(Q6K, RawE8M0),
        ServingFormat::new(Q8_0, RawE8M0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "expected {b}, got {a}");
    }

    #[test]
    fn the_integer_alphabet_is_the_fp4_table_doubled() {
        // Guards the hand-written constant against the real LUT drifting.
        let mut expected: Vec<i32> = FP4_MAGNITUDES
            .iter()
            .flat_map(|m| [(m * 2.0) as i32, -((m * 2.0) as i32)])
            .collect();
        expected.sort_unstable();
        expected.dedup();
        assert_eq!(expected, FP4_INTEGER_ALPHABET);
    }

    #[test]
    fn the_alphabet_is_sorted_which_the_step_calculation_assumes() {
        assert!(FP4_INTEGER_ALPHABET.windows(2).all(|p| p[0] < p[1]));
    }

    #[test]
    fn plus_and_minus_zero_collapse_to_fifteen_codepoints() {
        assert_eq!(distinct_codepoints(), 15);
    }

    #[test]
    fn the_payload_floor_is_four_bits_and_mxfp4_already_pays_exactly_it() {
        // The whole of "Experiment 1" in one assertion: no exact fixed-width
        // code beats 4 payload bits, and the archive format already is one.
        assert_eq!(min_payload_bits(), 4);
        approx(Container::Mxfp4.all_in_bits(), 4.25);
    }

    #[test]
    fn the_alphabet_is_not_an_arithmetic_progression() {
        // If it were, a 4-bit affine grid would suffice and Q4_K would be exact.
        let step = FP4_INTEGER_ALPHABET[1] - FP4_INTEGER_ALPHABET[0];
        assert!(FP4_INTEGER_ALPHABET.windows(2).any(|p| p[1] - p[0] != step));
    }

    #[test]
    fn twenty_five_levels_are_required_and_that_number_decides_everything() {
        assert_eq!(affine_levels_required(), 25);
    }

    #[test]
    fn q4k_is_nine_levels_short_so_no_scan_can_rescue_it() {
        assert!(!Container::Q4K.holds_fp4_exactly());
        assert!(Container::Q4K.grid_levels().unwrap() < affine_levels_required());
    }

    #[test]
    fn q5k_q6k_q8_and_native_mxfp4_are_all_exact() {
        for c in [
            Container::Q5K,
            Container::Q6K,
            Container::Q8_0,
            Container::Mxfp4,
        ] {
            assert!(c.holds_fp4_exactly(), "{} should be exact", c.label());
        }
    }

    #[test]
    fn q6k_is_the_cheapest_exact_container_that_can_actually_serve() {
        // Why the current serving path is Q6_K. The filter is `is_servable`,
        // not "a kernel exists" — MXFP4 has kernels and is cheaper but cannot
        // serve, and conflating those produced a wrong claim once already.
        let cheapest = [
            Container::Mxfp4,
            Container::Q4K,
            Container::Q5K,
            Container::Q6K,
            Container::Q8_0,
        ]
        .into_iter()
        .filter(|c| c.holds_fp4_exactly() && c.kernel_maturity().is_servable())
        .min_by(|a, b| a.all_in_bits().partial_cmp(&b.all_in_bits()).unwrap());
        assert_eq!(cheapest, Some(Container::Q6K));
    }

    #[test]
    fn mxfp4_has_kernels_but_cannot_serve() {
        // The corrected claim, pinned: kernels exist (K1 standalone, K2
        // grouped) yet nothing routes a forward pass through them.
        let m = Container::Mxfp4.kernel_maturity();
        assert!(m >= KernelMaturity::Standalone, "K1 exists");
        assert!(!m.is_servable(), "but it is not in QuantMatVec dispatch");
        assert!(Container::Mxfp4.all_in_bits() < Container::Q6K.all_in_bits());
    }

    #[test]
    fn q5k_is_strictly_dominated_by_native_mxfp4() {
        // Exact, but has NO kernel at all while MXFP4 already has two rungs
        // built — so Q5_K costs strictly more work AND ships more bytes.
        assert!(Container::Q5K.holds_fp4_exactly());
        assert_eq!(Container::Q5K.kernel_maturity(), KernelMaturity::None);
        assert!(Container::Mxfp4.kernel_maturity() > KernelMaturity::None);
        assert!(Container::Q5K.all_in_bits() > Container::Mxfp4.all_in_bits());
    }

    #[test]
    fn the_q6k_byte_tax_is_the_known_factor() {
        approx(
            (Container::Q6K.all_in_bits() / Container::Mxfp4.all_in_bits() * 1e3).round() / 1e3,
            1.544,
        );
    }

    #[test]
    fn scale_encodings_price_out_as_documented() {
        approx(ScaleEncoding::RawE8M0.bits_per_weight(), 0.25);
        approx(
            ScaleEncoding::SuperblockBaseDelta2.bits_per_weight(),
            0.09375,
        );
        approx(ScaleEncoding::GlobalTable2Bit.bits_per_weight(), 0.0625);
    }

    #[test]
    fn both_compact_scale_encodings_clear_the_measured_spread() {
        for s in [
            ScaleEncoding::SuperblockBaseDelta2,
            ScaleEncoding::GlobalTable2Bit,
        ] {
            assert!(
                s.max_spread().unwrap() >= MEASURED_MAX_SPREAD,
                "{}",
                s.label()
            );
        }
    }

    #[test]
    fn only_the_global_table_carries_headroom_free_global_exposure() {
        // 2 bits addresses exactly the 4 exponents observed. Zero slack.
        assert_eq!(1u32 << 2, MEASURED_DISTINCT_EXPONENTS);
        assert!(ScaleEncoding::GlobalTable2Bit.needs_exception_path());
        assert!(!ScaleEncoding::SuperblockBaseDelta2.needs_exception_path());
        assert!(!ScaleEncoding::RawE8M0.needs_exception_path());
    }

    #[test]
    fn the_exact_floor_is_just_under_four_and_a_half_percent_below_mxfp4() {
        // The entire remaining exact prize, so it cannot be mistaken for the
        // container-sized lever it sits next to.
        let best = exact_floor();
        approx(best.all_in_bits(), 4.06730625);
        let cut = 1.0 - best.all_in_bits() / Container::Mxfp4.all_in_bits();
        assert!((0.04..0.05).contains(&cut), "cut {cut} should be ~4.3%");
    }

    #[test]
    fn k_quant_containers_ignore_the_scale_encoding_field() {
        // Their block layout is fixed; re-encoding scales piecemeal is not a
        // thing you can do to a Q6_K block.
        let a = ServingFormat::new(Container::Q6K, ScaleEncoding::RawE8M0);
        let b = ServingFormat::new(Container::Q6K, ScaleEncoding::GlobalTable2Bit);
        approx(a.all_in_bits(), b.all_in_bits());
    }

    #[test]
    fn every_candidate_but_q4k_is_exact_for_k3() {
        for f in candidates() {
            let want = f.container != Container::Q4K;
            assert_eq!(f.is_exact_for_k3(), want, "{}", f.container.label());
        }
    }

    #[test]
    fn the_exact_floor_is_the_cheapest_candidate_without_global_exposure() {
        // Not simply candidates()[0] — that one is 0.03 bpw cheaper but fits the
        // observed exponent range with no headroom, so it owes an exception path.
        let floor = exact_floor();
        assert!(floor.is_exact_for_k3());
        assert!(!floor.scale.needs_exception_path());
        let cheaper_and_safe = candidates()
            .into_iter()
            .filter(|f| f.is_exact_for_k3() && !f.scale.needs_exception_path())
            .any(|f| f.all_in_bits() < floor.all_in_bits());
        assert!(!cheaper_and_safe);
    }

    #[test]
    fn candidates_are_listed_cheapest_first() {
        let bits: Vec<f64> = candidates()
            .iter()
            .map(ServingFormat::all_in_bits)
            .collect();
        assert!(bits.windows(2).all(|p| p[0] <= p[1]), "{bits:?}");
    }

    #[test]
    fn kernel_maturity_ladder_ascends_and_servability_splits_it_once() {
        // `is_servable` is a threshold on the ordering, so the ordering is the
        // load-bearing part: a rung inserted out of order would silently move
        // the servable boundary without touching `is_servable` itself.
        let ladder = [
            KernelMaturity::None,
            KernelMaturity::Standalone,
            KernelMaturity::Grouped,
            KernelMaturity::Dispatched,
            KernelMaturity::Production,
        ];
        assert!(
            ladder.windows(2).all(|p| p[0] < p[1]),
            "the ladder must be declared in ascending maturity: {ladder:?}"
        );
        let servable: Vec<bool> = ladder.iter().map(|m| m.is_servable()).collect();
        // Monotone: once servable, always servable further up the ladder.
        assert!(servable.windows(2).all(|p| p[0] <= p[1]), "{servable:?}");
        assert!(!KernelMaturity::Standalone.is_servable());
        assert!(KernelMaturity::Dispatched.is_servable());
    }
}
