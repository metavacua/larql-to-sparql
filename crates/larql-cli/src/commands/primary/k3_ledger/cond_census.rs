//! ETC-0A — is a K3 expert cheaper to send *given another expert*?
//!
//! Pure. [`symbol_census`](super::symbol_census) closed the question of whether
//! one expert's nibble stream compresses on its own: measured entropy 3.7525
//! bits against 4.0 spent, no tile with a small enough alphabet to palette, and
//! the apparent local skew was entirely plug-in estimator bias. **That census
//! conditioned on nothing.** It says only that fp4's logarithmic grid is already
//! matched to the marginal weight distribution.
//!
//! This module asks the joint question the marginal one cannot reach:
//!
//! ```text
//!   I(S_e ; S_p) = H(S_e) - H(S_e | S_p)
//! ```
//!
//! where `S_e` is the requested expert's packed MXFP4 symbol stream and `S_p` is
//! a shared layer prototype or an already-resident expert, **aligned by weight
//! coordinate**. Nothing is subtracted, requantised or reconstructed — the
//! statistic is computed on the bytes the checkpoint already stores, so it
//! carries no distortion of its own and needs no fidelity label.
//!
//! ## Three traps this module is built around
//!
//! **1. Plug-in mutual information is biased UPWARD.** The joint table has more
//! occupied cells than either marginal, so it absorbs more of the downward
//! plug-in bias, and the difference shows up as redundancy that is not there.
//! Expected size under independence is `(K_t-1)(K_c-1)/(2N ln2)`. Every arm
//! reports [`ConditionalArm::mi_bias_corrected`] beside the raw figure, and the
//! shuffled null measures the residual empirically rather than trusting either.
//!
//! **2. A prototype built from the whole bank leaks its target.** Include expert
//! `e` in the mode that `e` is then scored against and the conditional entropy
//! falls for a reason that has nothing to do with transport. [`BankProfile`] is
//! leave-one-out by construction — there is no non-LOO accessor to call by
//! mistake.
//!
//! **3. An entropy is not a kernel.** `H(S_e | S_p)` is reachable only by a
//! serial arithmetic coder conditioned on the parent, which is the same
//! unrealisable floor the fixed-format ladder already refused to bank (R11).
//! Every arm therefore also carries [`ConditionalArm::match_escape_bpw`], a code
//! a Metal kernel could actually run, and the gap between the two is the
//! realisability tax — read them as a pair or not at all.

use serde::Serialize;

use super::rng::{self, SplitMix64};
use super::symbol_census::{
    miller_madow_bias, SymbolCounts, FIXED_PAYLOAD_BITS, FP4_CODES, NEG_ZERO_CODE, POS_ZERO_CODE,
};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// One bit per weight naming "same symbol as the parent" or "escape".
pub const MATCH_FLAG_BITS: f64 = 1.0;

/// Conditional-entropy bands, as pre-registered before the fetch. Reading a
/// result against a bar written afterwards is how an uncalibrated threshold
/// convicts a correct hypothesis, so these are constants, not prose.
pub const BAND_USELESS_BITS: f64 = 3.0;
pub const BAND_PROGRAMME_BITS: f64 = 1.8;
pub const BAND_MAJOR_BITS: f64 = 1.0;

/// Joint symbol table over aligned (target, context) nibble pairs.
#[derive(Debug, Clone)]
pub struct JointCounts {
    cells: [[u64; FP4_CODES]; FP4_CODES],
}

impl Default for JointCounts {
    fn default() -> Self {
        Self {
            cells: [[0; FP4_CODES]; FP4_CODES],
        }
    }
}

impl JointCounts {
    /// Accumulate coordinate-aligned pairs. Pairs stop at the shorter stream —
    /// the *shape* check that makes truncation impossible belongs at the fetch,
    /// where a mismatch means the wrong tensor was read.
    pub fn add_aligned(&mut self, target: &[u8], context: &[u8]) {
        for (&t, &c) in target.iter().zip(context) {
            if (t as usize) < FP4_CODES && (c as usize) < FP4_CODES {
                self.cells[t as usize][c as usize] += 1;
            }
        }
    }

    pub fn total(&self) -> u64 {
        self.cells.iter().flatten().sum()
    }

    pub fn occupied_cells(&self) -> usize {
        self.cells.iter().flatten().filter(|&&c| c > 0).count()
    }

    pub fn marginal_target(&self) -> SymbolCounts {
        let mut m = SymbolCounts::default();
        for (t, row) in self.cells.iter().enumerate() {
            m.counts[t] = row.iter().sum();
        }
        m
    }

    pub fn marginal_context(&self) -> SymbolCounts {
        let mut m = SymbolCounts::default();
        for row in &self.cells {
            for (c, &n) in row.iter().enumerate() {
                m.counts[c] += n;
            }
        }
        m
    }

    pub fn entropy_joint(&self) -> f64 {
        entropy_bits(self.cells.iter().flatten().copied(), self.total())
    }

    /// `H(target | context)` — the conditional coding floor.
    pub fn conditional_entropy(&self) -> f64 {
        (self.entropy_joint() - self.marginal_context().entropy()).max(0.0)
    }

    /// `I(target ; context)`, plug-in. Biased upward; pair with
    /// [`Self::mutual_information_corrected`].
    pub fn mutual_information(&self) -> f64 {
        self.marginal_target().entropy() - self.conditional_entropy()
    }

    /// Miller-Madow correction applied to all three terms.
    ///
    /// The sign is not fixed, and assuming it is would be its own error. Under
    /// **independence** the joint spreads across `K_t · K_c` cells while the
    /// marginals stay narrow, so the correction is strongly *negative* — and
    /// that is precisely the regime where plug-in mutual information invents
    /// redundancy that is not there. Under a near-deterministic relation the
    /// joint is sparse and the correction is mildly *positive*.
    ///
    /// The nulls are scored on this figure rather than the plug-in one, because
    /// a null is by construction the independent case.
    pub fn mutual_information_corrected(&self) -> f64 {
        let n = self.total();
        let (t, c) = (self.marginal_target(), self.marginal_context());
        let h_t = t.entropy() + miller_madow_bias(t.distinct(), n);
        let h_c = c.entropy() + miller_madow_bias(c.distinct(), n);
        let h_tc = self.entropy_joint() + miller_madow_bias(self.occupied_cells(), n);
        h_t + h_c - h_tc
    }

    /// Mutual information split by whether either side is a `±0` code.
    ///
    /// Zeros are 11.5% of the bank and co-located zeros are genuinely codeable,
    /// so this is not a nuisance term to remove — but a result that is *entirely*
    /// zero-driven is "the sparsity patterns agree", which is a much weaker and
    /// much more layout-dependent claim than "the experts resemble each other".
    pub fn mi_zero_split(&self) -> (f64, f64) {
        let n = self.total() as f64;
        if n == 0.0 {
            return (0.0, 0.0);
        }
        let (tm, cm) = (self.marginal_target(), self.marginal_context());
        let (mut zero, mut nonzero) = (0.0, 0.0);
        for (t, row) in self.cells.iter().enumerate() {
            for (c, &count) in row.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                let p_tc = count as f64 / n;
                let p_t = tm.counts[t] as f64 / n;
                let p_c = cm.counts[c] as f64 / n;
                let term = p_tc * (p_tc / (p_t * p_c)).log2();
                if is_zero_code(t) || is_zero_code(c) {
                    zero += term;
                } else {
                    nonzero += term;
                }
            }
        }
        (zero, nonzero)
    }

    /// Probability the target symbol equals its context symbol — the quantity a
    /// realisable match/escape code actually spends.
    pub fn agreement(&self) -> f64 {
        let n = self.total();
        if n == 0 {
            return 0.0;
        }
        (0..FP4_CODES).map(|i| self.cells[i][i]).sum::<u64>() as f64 / n as f64
    }

    /// Agreement once `+0` and `-0` count as the same value. A value-exact codec
    /// may merge them; a bit-exact one may not.
    pub fn agreement_zero_merged(&self) -> f64 {
        let n = self.total();
        if n == 0 {
            return 0.0;
        }
        let diagonal: u64 = (0..FP4_CODES).map(|i| self.cells[i][i]).sum();
        let cross =
            self.cells[POS_ZERO_CODE][NEG_ZERO_CODE] + self.cells[NEG_ZERO_CODE][POS_ZERO_CODE];
        (diagonal + cross) as f64 / n as f64
    }
}

fn is_zero_code(code: usize) -> bool {
    code == POS_ZERO_CODE || code == NEG_ZERO_CODE
}

fn entropy_bits(counts: impl Iterator<Item = u64>, total: u64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let n = total as f64;
    -counts
        .filter(|&c| c > 0)
        .map(|c| {
            let p = c as f64 / n;
            p * p.log2()
        })
        .sum::<f64>()
}

/// Bits per weight for a code that flags "same as parent" and escapes to a raw
/// nibble otherwise. Realisable as a bitmask plus a compacted residual stream —
/// no serial dependency, which is what separates it from the entropy floor.
pub fn match_escape_bpw(agreement: f64) -> f64 {
    MATCH_FLAG_BITS + (1.0 - agreement.clamp(0.0, 1.0)) * FIXED_PAYLOAD_BITS
}

/// Per-coordinate symbol histogram across a bank of aligned expert streams.
///
/// Exists only to serve leave-one-out modes. There is deliberately no accessor
/// that returns the mode *including* the target: a prototype that has seen the
/// expert it is scored against reports redundancy that no decoder could use.
pub struct BankProfile {
    per_coord: Vec<[u16; FP4_CODES]>,
}

impl BankProfile {
    /// `None` if the streams are ragged — that is a fetch bug, not a data
    /// property, and silently truncating it would corrupt every coordinate
    /// past the first short stream.
    pub fn build(streams: &[Vec<u8>]) -> Option<Self> {
        let len = streams.first()?.len();
        if streams.iter().any(|s| s.len() != len) {
            return None;
        }
        let mut per_coord = vec![[0u16; FP4_CODES]; len];
        for s in streams {
            for (coord, &sym) in s.iter().enumerate() {
                if (sym as usize) < FP4_CODES {
                    per_coord[coord][sym as usize] =
                        per_coord[coord][sym as usize].saturating_add(1);
                }
            }
        }
        Some(Self { per_coord })
    }

    pub fn coords(&self) -> usize {
        self.per_coord.len()
    }

    /// Modal symbol at `coord` with `own` removed from the tally. Ties break to
    /// the lowest code so the prototype is reproducible run to run.
    pub fn mode_excluding(&self, coord: usize, own: u8) -> u8 {
        let mut counts = self.per_coord[coord];
        if (own as usize) < FP4_CODES {
            counts[own as usize] = counts[own as usize].saturating_sub(1);
        }
        let mut best = 0usize;
        for (code, &n) in counts.iter().enumerate() {
            if n > counts[best] {
                best = code;
            }
        }
        best as u8
    }

    /// The leave-one-out prototype stream for one expert.
    pub fn prototype_for(&self, own: &[u8]) -> Vec<u8> {
        own.iter()
            .enumerate()
            .take(self.coords())
            .map(|(coord, &sym)| self.mode_excluding(coord, sym))
            .collect()
    }
}

/// Joint table over (target, prototype, parent), for the question that decides
/// whether ETC-0B is worth running: **does a specific resident parent tell you
/// anything the bank-wide prototype has not already told you?**
///
/// If `H(t | proto, parent)` matches `H(t | proto)`, graph coding buys nothing
/// over a resident dictionary and the routing-trace work is unnecessary.
#[derive(Debug, Clone)]
pub struct TripleCounts {
    cells: Vec<u64>,
}

impl Default for TripleCounts {
    fn default() -> Self {
        Self {
            cells: vec![0; FP4_CODES * FP4_CODES * FP4_CODES],
        }
    }
}

impl TripleCounts {
    pub fn add_aligned(&mut self, target: &[u8], proto: &[u8], parent: &[u8]) {
        for ((&t, &p), &f) in target.iter().zip(proto).zip(parent) {
            let (t, p, f) = (t as usize, p as usize, f as usize);
            if t < FP4_CODES && p < FP4_CODES && f < FP4_CODES {
                self.cells[(t * FP4_CODES + p) * FP4_CODES + f] += 1;
            }
        }
    }

    pub fn total(&self) -> u64 {
        self.cells.iter().sum()
    }

    /// `H(target | prototype, parent)`.
    pub fn conditional_entropy(&self) -> f64 {
        let total = self.total();
        let joint = entropy_bits(self.cells.iter().copied(), total);
        let context: Vec<u64> = (0..FP4_CODES * FP4_CODES)
            .map(|pf| {
                (0..FP4_CODES)
                    .map(|t| self.cells[t * FP4_CODES * FP4_CODES + pf])
                    .sum()
            })
            .collect();
        (joint - entropy_bits(context.into_iter(), total)).max(0.0)
    }
}

/// A marginal-preserving null: shuffle one stream's positions so its histogram
/// is untouched and its coordinate correspondence is destroyed.
///
/// Any conditional gain that survives this is not aligned structure. Any gain
/// that does *not* survive was estimator bias or an alignment bug — the control
/// cannot tell you which, but it does refuse the result either way.
pub fn shuffled(stream: &[u8], seed: u64) -> Vec<u8> {
    let mut out = stream.to_vec();
    rng::shuffle(&mut out, &mut SplitMix64(seed));
    out
}

/// One conditioning arm, scored.
#[derive(Debug, Clone, Serialize)]
pub struct ConditionalArm {
    pub label: String,
    pub samples: u64,
    pub occupied_cells: usize,
    /// `H(target)` — the marginal control. Should reproduce the banked 3.7525.
    pub marginal_entropy: f64,
    /// `H(target | context)` — the conditional coding floor, serial decode only.
    pub conditional_entropy: f64,
    pub mutual_information: f64,
    pub mi_bias_corrected: f64,
    pub mi_zero_involved: f64,
    pub mi_nonzero: f64,
    pub agreement: f64,
    pub agreement_zero_merged: f64,
    /// A code a kernel could run, priced on the same table (R11).
    pub match_escape_bpw: f64,
}

pub fn score(label: impl Into<String>, joint: &JointCounts) -> ConditionalArm {
    let (zero, nonzero) = joint.mi_zero_split();
    let agreement_merged = joint.agreement_zero_merged();
    ConditionalArm {
        label: label.into(),
        samples: joint.total(),
        occupied_cells: joint.occupied_cells(),
        marginal_entropy: joint.marginal_target().entropy(),
        conditional_entropy: joint.conditional_entropy(),
        mutual_information: joint.mutual_information(),
        mi_bias_corrected: joint.mutual_information_corrected(),
        mi_zero_involved: zero,
        mi_nonzero: nonzero,
        agreement: joint.agreement(),
        agreement_zero_merged: agreement_merged,
        match_escape_bpw: match_escape_bpw(agreement_merged),
    }
}

/// Where a measured conditional entropy lands against the pre-registered bands.
pub fn band(conditional_entropy: f64) -> &'static str {
    if conditional_entropy >= BAND_USELESS_BITS {
        "STOP — statistically interesting, useless as a transport codec"
    } else if conditional_entropy >= BAND_PROGRAMME_BITS {
        "MARGINAL — real but below the band a codec was scoped against"
    } else if conditional_entropy >= BAND_MAJOR_BITS {
        "PROGRAMME — enough to justify ETC-0B and a codec design"
    } else {
        "MAJOR — under one bit per weight given a parent"
    }
}

/// Does the null still show a gain? If so the arm is measuring its estimator,
/// not the bank. `tolerance` is in bits.
pub fn null_is_clean(null_mi: f64, tolerance: f64) -> bool {
    null_mi.abs() <= tolerance
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "expected ~{b}, got {a}");
    }

    /// A stream cycling every code, so the marginal is exactly uniform (4 bits).
    fn cycling(len: usize, offset: u8) -> Vec<u8> {
        (0..len)
            .map(|i| ((i as u8).wrapping_add(offset)) % FP4_CODES as u8)
            .collect()
    }

    fn joint_of(target: &[u8], context: &[u8]) -> JointCounts {
        let mut j = JointCounts::default();
        j.add_aligned(target, context);
        j
    }

    #[test]
    fn an_identical_context_leaves_no_uncertainty() {
        let s = cycling(4096, 0);
        let j = joint_of(&s, &s);
        approx(j.marginal_target().entropy(), 4.0, 1e-9);
        approx(j.conditional_entropy(), 0.0, 1e-9);
        approx(j.mutual_information(), 4.0, 1e-9);
        approx(j.agreement(), 1.0, 1e-12);
        approx(j.match_escape_bpw_of(), MATCH_FLAG_BITS, 1e-12);
    }

    #[test]
    fn a_permuted_context_is_just_as_informative_as_an_identical_one() {
        // Redundancy is not similarity: a fixed relabelling is fully codeable.
        let target = cycling(4096, 0);
        let context: Vec<u8> = target.iter().map(|&s| (s + 3) % 16).collect();
        let j = joint_of(&target, &context);
        approx(j.conditional_entropy(), 0.0, 1e-9);
        // ...but the *realisable* match/escape code sees none of it.
        approx(j.agreement(), 0.0, 1e-12);
        approx(
            j.match_escape_bpw_of(),
            MATCH_FLAG_BITS + FIXED_PAYLOAD_BITS,
            1e-12,
        );
    }

    #[test]
    fn an_independent_context_carries_no_information() {
        // Coprime periods 16 and 17 decorrelate the two streams.
        let target: Vec<u8> = (0..4352u32).map(|i| (i % 16) as u8).collect();
        let context: Vec<u8> = (0..4352u32).map(|i| (i % 17 % 16) as u8).collect();
        let j = joint_of(&target, &context);
        assert!(
            j.mutual_information() < 0.05,
            "MI {} should be near zero",
            j.mutual_information()
        );
    }

    #[test]
    fn mutual_information_obeys_its_own_identity() {
        let j = joint_of(&cycling(3333, 0), &cycling(3333, 5));
        let (t, c) = (j.marginal_target(), j.marginal_context());
        approx(
            j.mutual_information(),
            t.entropy() + c.entropy() - j.entropy_joint(),
            1e-9,
        );
    }

    /// Two streams with no aligned relationship, so the joint fills all 256
    /// cells — the regime the correction exists for.
    fn independent(len: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
        let s = cycling(len, 0);
        (s.clone(), shuffled(&s, seed))
    }

    #[test]
    fn the_correction_removes_the_redundancy_that_independence_invents() {
        // Plug-in MI on independent streams is positive purely from the joint
        // occupying more cells than the marginals. That is the whole failure
        // mode, and the correction has to bite here or nowhere.
        let (t, c) = independent(8192, 1);
        let j = joint_of(&t, &c);
        assert!(j.mutual_information() > 0.01, "the bias should be visible");
        assert!(
            j.mutual_information_corrected() < j.mutual_information(),
            "correction must lower an independent joint"
        );
        assert!(j.mutual_information_corrected().abs() < 0.005);
    }

    #[test]
    fn a_deterministic_relation_is_not_penalised_by_the_correction() {
        // The sign is not fixed: a sparse joint gets a mildly positive
        // correction. Asserting one direction everywhere would be wrong.
        let j = joint_of(&cycling(512, 0), &cycling(512, 1));
        assert!(j.mutual_information_corrected() >= j.mutual_information());
    }

    #[test]
    fn the_bias_is_negligible_at_census_scale_and_visible_at_toy_scale() {
        let gap = |len: usize| {
            let (t, c) = independent(len, 2);
            joint_of(&t, &c).mutual_information()
        };
        // 225/(2N ln2): 0.02 bits at 8k samples, 1.2e-5 at the ~14M the real
        // census collects. This is why the null is a control and not a worry.
        approx(gap(8192), 0.0198, 0.006);
        assert!(gap(1 << 20) < 2e-3, "{}", gap(1 << 20));
        assert!(gap(1 << 20) < gap(8192));
    }

    #[test]
    fn the_zero_split_partitions_the_whole_mutual_information() {
        let j = joint_of(&cycling(4096, 0), &cycling(4096, 8));
        let (zero, nonzero) = j.mi_zero_split();
        approx(zero + nonzero, j.mutual_information(), 1e-9);
    }

    #[test]
    fn a_purely_zero_driven_result_is_visible_as_such() {
        // Both streams agree only on where their zeros are; elsewhere they are
        // decorrelated. The split must attribute the information to the zeros.
        let target: Vec<u8> = (0..4096u32)
            .map(|i| if i % 4 == 0 { 0 } else { (i % 7 + 1) as u8 })
            .collect();
        let context: Vec<u8> = (0..4096u32)
            .map(|i| if i % 4 == 0 { 0 } else { (i % 5 + 9) as u8 })
            .collect();
        let (zero, nonzero) = joint_of(&target, &context).mi_zero_split();
        assert!(zero > nonzero, "zero {zero} vs nonzero {nonzero}");
    }

    #[test]
    fn merging_the_two_zeros_can_only_raise_agreement() {
        let mut j = JointCounts::default();
        j.add_aligned(&[POS_ZERO_CODE as u8; 10], &[NEG_ZERO_CODE as u8; 10]);
        approx(j.agreement(), 0.0, 1e-12);
        approx(j.agreement_zero_merged(), 1.0, 1e-12);
    }

    #[test]
    fn the_shuffled_null_destroys_alignment_and_keeps_the_histogram() {
        let target = cycling(8192, 0);
        let real = joint_of(&target, &target);
        let null = joint_of(&target, &shuffled(&target, 42));

        approx(real.mutual_information(), 4.0, 1e-9);
        // Scored on the CORRECTED figure, as the arms are: plug-in MI on this
        // null is ~0.02 bits of pure estimator bias at 8k samples and would
        // fail a 0.02 bar for reasons that have nothing to do with the data.
        assert!(
            null_is_clean(null.mutual_information_corrected(), 0.02),
            "null MI {} should collapse",
            null.mutual_information_corrected()
        );
        // The marginal is untouched — that is what makes it a null and not a
        // different dataset.
        assert_eq!(
            real.marginal_context().counts,
            null.marginal_context().counts
        );
    }

    #[test]
    fn the_null_is_reproducible_from_its_seed() {
        let s = cycling(1024, 3);
        assert_eq!(shuffled(&s, 7), shuffled(&s, 7));
        assert_ne!(shuffled(&s, 7), shuffled(&s, 8));
    }

    #[test]
    fn out_of_range_nibbles_are_skipped_rather_than_indexing_past_the_table() {
        let mut j = JointCounts::default();
        j.add_aligned(&[0, 99, 3], &[0, 1, 200]);
        assert_eq!(j.total(), 1, "only the in-range pair should count");
    }

    #[test]
    fn misaligned_lengths_stop_at_the_shorter_stream() {
        let mut j = JointCounts::default();
        j.add_aligned(&[1, 2, 3, 4], &[1, 2]);
        assert_eq!(j.total(), 2);
    }

    #[test]
    fn an_empty_table_reports_zeros_rather_than_dividing_by_zero() {
        let j = JointCounts::default();
        approx(j.entropy_joint(), 0.0, 1e-12);
        approx(j.conditional_entropy(), 0.0, 1e-12);
        approx(j.agreement(), 0.0, 1e-12);
        approx(j.agreement_zero_merged(), 0.0, 1e-12);
        assert_eq!(j.mi_zero_split(), (0.0, 0.0));
        assert_eq!(j.occupied_cells(), 0);
    }

    #[test]
    fn the_prototype_never_sees_the_expert_it_is_scored_against() {
        // Two experts hold 5, one holds 9. Leave-one-out must hand the 9-holder
        // a prototype of 5, and each 5-holder a prototype of 5 (the other one).
        let streams = vec![vec![5u8], vec![5u8], vec![9u8]];
        let p = BankProfile::build(&streams).unwrap();
        assert_eq!(p.mode_excluding(0, 9), 5);
        assert_eq!(p.mode_excluding(0, 5), 5);
        assert_eq!(p.coords(), 1);
    }

    #[test]
    fn leave_one_out_actually_changes_the_answer_when_it_should() {
        // A lone expert conditioned on a bank of one has nothing to learn from:
        // removing itself empties the tally, and the mode falls back to code 0.
        let streams = vec![vec![7u8; 4]];
        let p = BankProfile::build(&streams).unwrap();
        assert_eq!(p.prototype_for(&[7, 7, 7, 7]), vec![0, 0, 0, 0]);
    }

    #[test]
    fn prototype_ties_break_to_the_lowest_code_for_reproducibility() {
        // {2:2, 11:1}. Removing a 2 leaves 1-1, a genuine tie, and the lower
        // code must win so the prototype replays run to run.
        let tied = vec![vec![2u8], vec![2u8], vec![11u8]];
        let p = BankProfile::build(&tied).unwrap();
        assert_eq!(p.mode_excluding(0, 2), 2, "tie at 1-1 breaks low");
        assert_eq!(p.mode_excluding(0, 11), 2, "2 outright with the 11 removed");
    }

    #[test]
    fn removing_the_target_can_change_which_symbol_leads() {
        // {2:2, 11:2}: removing a 2 hands the mode to 11. Leave-one-out is not
        // cosmetic — it moves the answer, which is why it cannot be skipped.
        let even = vec![vec![2u8], vec![11u8], vec![2u8], vec![11u8]];
        let p = BankProfile::build(&even).unwrap();
        assert_eq!(p.mode_excluding(0, 2), 11);
        assert_eq!(p.mode_excluding(0, 11), 2);
    }

    #[test]
    fn ragged_streams_are_refused_rather_than_silently_truncated() {
        assert!(BankProfile::build(&[vec![1u8, 2], vec![1u8]]).is_none());
        assert!(BankProfile::build(&[]).is_none());
        assert!(BankProfile::build(&[vec![1u8, 2], vec![3u8, 4]]).is_some());
    }

    #[test]
    fn a_parent_cannot_make_the_prototype_worse() {
        // Conditioning on more never raises conditional entropy.
        let target = cycling(4096, 0);
        let proto = cycling(4096, 1);
        let parent = cycling(4096, 2);
        let mut triple = TripleCounts::default();
        triple.add_aligned(&target, &proto, &parent);
        let pair = joint_of(&target, &proto);
        assert!(triple.conditional_entropy() <= pair.conditional_entropy() + 1e-9);
        assert_eq!(triple.total(), 4096);
    }

    #[test]
    fn a_parent_that_adds_nothing_leaves_the_conditional_entropy_where_it_was() {
        // parent is a deterministic function of proto, so it is redundant. This
        // is the ETC-0B kill: if the real bank looks like this, graph coding
        // buys nothing over a resident dictionary.
        let target: Vec<u8> = (0..4096u32).map(|i| (i % 13) as u8).collect();
        let proto: Vec<u8> = (0..4096u32).map(|i| (i % 4) as u8).collect();
        let parent: Vec<u8> = proto.iter().map(|&s| (s * 2) % 16).collect();
        let mut triple = TripleCounts::default();
        triple.add_aligned(&target, &proto, &parent);
        approx(
            triple.conditional_entropy(),
            joint_of(&target, &proto).conditional_entropy(),
            1e-9,
        );
    }

    #[test]
    fn an_empty_triple_does_not_divide_by_zero() {
        let t = TripleCounts::default();
        assert_eq!(t.total(), 0);
        approx(t.conditional_entropy(), 0.0, 1e-12);
    }

    #[test]
    fn the_match_escape_code_is_bounded_by_its_flag_and_its_escape() {
        approx(match_escape_bpw(1.0), MATCH_FLAG_BITS, 1e-12);
        approx(
            match_escape_bpw(0.0),
            MATCH_FLAG_BITS + FIXED_PAYLOAD_BITS,
            1e-12,
        );
        // Out-of-range agreements are clamped, not extrapolated.
        approx(match_escape_bpw(1.5), MATCH_FLAG_BITS, 1e-12);
        approx(
            match_escape_bpw(-0.5),
            MATCH_FLAG_BITS + FIXED_PAYLOAD_BITS,
            1e-12,
        );
    }

    #[test]
    fn a_realisable_code_needs_agreement_well_past_a_half_to_beat_four_bits() {
        // The bar the census has to clear for a kernel, as opposed to for a
        // statistic: match/escape only wins below 4 bits once agreement > 0.25.
        assert!(match_escape_bpw(0.20) > FIXED_PAYLOAD_BITS);
        assert!(match_escape_bpw(0.30) < FIXED_PAYLOAD_BITS);
    }

    #[test]
    fn the_bands_are_ordered_and_cover_the_line() {
        assert!(band(3.7).starts_with("STOP"));
        assert!(band(3.0).starts_with("STOP"));
        assert!(band(2.4).starts_with("MARGINAL"));
        assert!(band(1.5).starts_with("PROGRAMME"));
        assert!(band(0.6).starts_with("MAJOR"));
    }

    #[test]
    fn scoring_an_arm_carries_both_the_floor_and_the_realisable_code() {
        let s = cycling(2048, 0);
        let arm = score("identical", &joint_of(&s, &s));
        assert_eq!(arm.label, "identical");
        assert_eq!(arm.samples, 2048);
        approx(arm.conditional_entropy, 0.0, 1e-9);
        // The entropy floor says zero; the code a kernel can run says one bit.
        // Reporting only the first is the R11 error this pair exists to block.
        approx(arm.match_escape_bpw, MATCH_FLAG_BITS, 1e-12);
        approx(arm.marginal_entropy, 4.0, 1e-9);
    }

    impl JointCounts {
        fn match_escape_bpw_of(&self) -> f64 {
            match_escape_bpw(self.agreement_zero_merged())
        }
    }
}
