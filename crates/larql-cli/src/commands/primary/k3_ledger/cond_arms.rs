//! ETC-0A's arm design — what gets conditioned on what, and against which null.
//!
//! Pure. [`cond_census`](super::cond_census) supplies the statistics; this
//! module decides which comparisons are worth making and, more importantly,
//! which controls have to run beside them for the answer to mean anything.
//!
//! ## Six arms, and why none of them is optional
//!
//! | arm | answers |
//! |---|---|
//! | `prototype` | ETC-0A: does the *bank* carry aligned common structure? |
//! | `column profile` | the same question, invariant to the expert row permutation |
//! | `parent: random` | the uninformed control — would any expert have done? |
//! | `parent: selected` | ETC-0B's ceiling: the best parent, chosen out of sample |
//! | `CONTROL: self` | the pipeline registers perfect structure in these bytes |
//! | `CONTROL: prev row` | is there any column structure at all, within one expert? |
//! | `NULL: *` | is a gain estimator bias, or real alignment? |
//!
//! ## The row axis is not aligned, and that changes what a null result means
//!
//! Experts are permutation-invariant in their intermediate dimension, so row `r`
//! of one expert has no relationship to row `r` of another. Every arm that
//! aligns on full coordinates therefore tests one arbitrary pairing out of
//! `3072!`. [`ARM_COLUMN`] exists because the *column* axis carries no such
//! freedom — both experts read the same latent vector — so it is the only arm
//! here whose negative result constrains the general question. See
//! [`column_profile`](super::column_profile).
//!
//! Plus a triple, `H(target | prototype, parent)`, which is the **ETC-0B kill
//! switch**: if a specific parent adds nothing once the bank-wide prototype is
//! known, graph coding cannot beat a resident dictionary and the routing-trace
//! work is unnecessary. That is a result available from these same bytes, before
//! any trace is opened.
//!
//! ## Selection happens out of sample
//!
//! Picking the parent that codes `e` best and then reporting how well it codes
//! `e` is an oracle, and this ladder has already paid for one of those. The
//! coordinate range is split: parents are chosen on the **fit** slice and every
//! arm is scored on the **held-out** slice, so `parent: selected` is an estimate
//! rather than an upper bound on itself.
//!
//! ## The control that decides whether ETC-0B exists
//!
//! `parent: random` is not a formality. If an arbitrary expert codes as well as
//! a chosen one, there is no compression *topology* — only a common component
//! the prototype already captures — and the interesting version of the graph
//! idea is dead regardless of what the routing traces say.

use serde::Serialize;

use super::column_profile::ColumnProfile;
use super::cond_census::{
    band, null_is_clean, score, BankProfile, ConditionalArm, JointCounts, TripleCounts,
};
use super::rng::SplitMix64;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Fraction of coordinates used to choose parents; the rest are scored.
pub const DEFAULT_FIT_FRACTION: f64 = 0.5;
/// Seed for the marginal-preserving nulls.
pub const DEFAULT_NULL_SEED: u64 = 0x00E7_C000_0A00;
/// Bits of residual mutual information a null may show before the arm it
/// controls is refused.
pub const NULL_TOLERANCE_BITS: f64 = 0.02;
/// Conditional entropy the self control must come in under. It is zero by
/// construction, so any failure is a broken pipeline, not a finding.
pub const SELF_CONTROL_TOLERANCE_BITS: f64 = 1e-9;

pub const ARM_PROTOTYPE: &str = "prototype (leave-one-out)";
pub const ARM_COLUMN: &str = "column profile (perm-invariant)";
pub const ARM_PARENT_RANDOM: &str = "parent: random (control)";
pub const ARM_PARENT_SELECTED: &str = "parent: selected out-of-sample";
pub const ARM_SELF: &str = "CONTROL: self (must read 0.0000)";
pub const ARM_PREV_ROW: &str = "CONTROL: same expert, prev row";
pub const ARM_NULL_PROTOTYPE: &str = "NULL: prototype, shuffled";
pub const ARM_NULL_COLUMN: &str = "NULL: column profile, shuffled";
pub const ARM_NULL_PARENT: &str = "NULL: selected parent, shuffled";

/// Diagnostic arms never compete for the headline: a positive control reads
/// zero by construction and a null reads zero by design, so letting either into
/// `best_conditional_entropy` would report the instrument as the result.
pub fn is_diagnostic(label: &str) -> bool {
    label.starts_with("NULL") || label.starts_with("CONTROL")
}

/// One expert's coordinate-aligned MXFP4 nibble stream.
#[derive(Debug, Clone)]
pub struct ExpertStream {
    pub id: usize,
    pub nibbles: Vec<u8>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CondCensus {
    pub experts: usize,
    pub coords_total: usize,
    pub coords_fit: usize,
    pub coords_scored: usize,
    pub arms: Vec<ConditionalArm>,
    /// `H(target | prototype, parent)` — the ETC-0B kill switch.
    pub triple_conditional_entropy: f64,
    /// What a specific parent adds once the prototype is known. Near zero means
    /// graph coding is redundant with a resident dictionary.
    pub parent_gain_over_prototype: f64,
    /// What selecting the parent bought over taking any expert at all. Near zero
    /// means there is no compression topology to exploit.
    pub selection_gain_over_random: f64,
    /// `(target, chosen parent)` pairs, so a hub structure is visible.
    pub parent_choices: Vec<(usize, usize)>,
    pub distinct_parents: usize,
    pub nulls_clean: bool,
    /// The self arm read zero, so the pipeline can see structure in these exact
    /// bytes. A negative result is only interpretable once this holds.
    pub self_control_ok: bool,
}

impl CondCensus {
    pub fn arm(&self, label: &str) -> Option<&ConditionalArm> {
        self.arms.iter().find(|a| a.label == label)
    }

    /// The headline: the best conditional entropy any arm achieved, and where it
    /// lands against the pre-registered bands.
    pub fn best_conditional_entropy(&self) -> f64 {
        self.arms
            .iter()
            .filter(|a| !is_diagnostic(&a.label))
            .map(|a| a.conditional_entropy)
            .fold(f64::INFINITY, f64::min)
    }

    /// Cheapest code a kernel could actually run, over the non-diagnostic arms.
    pub fn best_realisable_bpw(&self) -> f64 {
        self.arms
            .iter()
            .filter(|a| !is_diagnostic(&a.label))
            .map(|a| a.match_escape_bpw)
            .fold(f64::INFINITY, f64::min)
    }

    pub fn verdict(&self) -> &'static str {
        if !self.self_control_ok {
            return "REFUSED — the self control did not read zero; the pipeline \
                    cannot see structure in these bytes, so a null result is \
                    uninterpretable";
        }
        if !self.nulls_clean {
            return "REFUSED — a null still shows mutual information; the arms are \
                    measuring their estimator or a misalignment, not the bank";
        }
        band(self.best_conditional_entropy())
    }
}

fn split_at(len: usize, fit_fraction: f64) -> Option<usize> {
    if len < 2 {
        return None;
    }
    let cut = (len as f64 * fit_fraction.clamp(0.0, 1.0)).round() as usize;
    Some(cut.clamp(1, len - 1))
}

/// Conditional entropy of `target` given `context` over one coordinate range.
fn pair_cost(target: &[u8], context: &[u8]) -> f64 {
    let mut j = JointCounts::default();
    j.add_aligned(target, context);
    j.conditional_entropy()
}

/// Cheapest coding parent for `target`, chosen on the fit slice only.
fn choose_parent(streams: &[ExpertStream], target: usize, fit: usize) -> Option<usize> {
    streams
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != target)
        .map(|(i, s)| {
            let cost = pair_cost(&streams[target].nibbles[..fit], &s.nibbles[..fit]);
            (i, cost)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal))
        .map(|(i, _)| i)
}

/// An arbitrary other expert — the uninformed control. Deterministic in `seed`.
fn arbitrary_parent(n: usize, target: usize, seed: u64) -> Option<usize> {
    if n < 2 {
        return None;
    }
    let mut rng = SplitMix64(seed ^ target as u64);
    let pick = rng.below(n as u64 - 1) as usize;
    Some(if pick >= target { pick + 1 } else { pick })
}

/// Run every arm. `row_len` is the stream's row length in symbols, which the
/// permutation-invariant column arm needs; pass 0 to skip that arm.
///
/// `None` when the streams are ragged, too few, or too short to split — all of
/// which are fetch bugs rather than data properties.
pub fn run(
    streams: &[ExpertStream],
    row_len: usize,
    fit_fraction: f64,
    seed: u64,
) -> Option<CondCensus> {
    if streams.len() < 2 {
        return None;
    }
    let len = streams.first()?.nibbles.len();
    if streams.iter().any(|s| s.nibbles.len() != len) {
        return None;
    }
    let fit = split_at(len, fit_fraction)?;

    let bank: Vec<Vec<u8>> = streams.iter().map(|s| s.nibbles.clone()).collect();
    let profile = BankProfile::build(&bank)?;
    let columns = ColumnProfile::build(&bank, row_len);

    let mut proto_j = JointCounts::default();
    let mut column_j = JointCounts::default();
    let mut random_j = JointCounts::default();
    let mut selected_j = JointCounts::default();
    let mut self_j = JointCounts::default();
    let mut prev_row_j = JointCounts::default();
    let mut null_proto_j = JointCounts::default();
    let mut null_column_j = JointCounts::default();
    let mut null_parent_j = JointCounts::default();
    let mut triple = TripleCounts::default();
    let mut parent_choices = Vec::with_capacity(streams.len());

    for (idx, s) in streams.iter().enumerate() {
        let scored = &s.nibbles[fit..];
        let proto_full = profile.prototype_for(&s.nibbles);
        let proto = &proto_full[fit..];

        let chosen = choose_parent(streams, idx, fit)?;
        let random = arbitrary_parent(streams.len(), idx, seed)?;
        parent_choices.push((s.id, streams[chosen].id));

        let chosen_scored = &streams[chosen].nibbles[fit..];
        proto_j.add_aligned(scored, proto);
        selected_j.add_aligned(scored, chosen_scored);
        random_j.add_aligned(scored, &streams[random].nibbles[fit..]);
        triple.add_aligned(scored, proto, chosen_scored);

        // Positive controls, on the same real bytes every other arm reads.
        // `self` proves the pipeline can register perfect structure; `prev row`
        // asks whether the column axis carries any structure at all, and unlike
        // `self` it is free to come back empty.
        self_j.add_aligned(scored, scored);
        if row_len > 0 && scored.len() > row_len {
            prev_row_j.add_aligned(&scored[row_len..], &scored[..scored.len() - row_len]);
        }

        if let Some(cols) = &columns {
            let col_full = cols.profile_for(&s.nibbles);
            let col = &col_full[fit..];
            column_j.add_aligned(scored, col);
            null_column_j.add_aligned(
                scored,
                &super::cond_census::shuffled(col, seed ^ (idx as u64) << 3 ^ 2),
            );
        }

        // Distinct seeds per arm and per target: one shared shuffle would
        // correlate the two nulls and hide a failure in either.
        null_proto_j.add_aligned(
            scored,
            &super::cond_census::shuffled(proto, seed ^ (idx as u64) << 1),
        );
        null_parent_j.add_aligned(
            scored,
            &super::cond_census::shuffled(chosen_scored, seed ^ (idx as u64) << 2 ^ 1),
        );
    }

    let mut arms = vec![score(ARM_PROTOTYPE, &proto_j)];
    if columns.is_some() {
        arms.push(score(ARM_COLUMN, &column_j));
    }
    arms.push(score(ARM_PARENT_RANDOM, &random_j));
    arms.push(score(ARM_PARENT_SELECTED, &selected_j));
    arms.push(score(ARM_SELF, &self_j));
    if prev_row_j.total() > 0 {
        arms.push(score(ARM_PREV_ROW, &prev_row_j));
    }
    arms.push(score(ARM_NULL_PROTOTYPE, &null_proto_j));
    if columns.is_some() {
        arms.push(score(ARM_NULL_COLUMN, &null_column_j));
    }
    arms.push(score(ARM_NULL_PARENT, &null_parent_j));

    let nulls_clean = arms
        .iter()
        .filter(|a| a.label.starts_with("NULL"))
        .all(|a| null_is_clean(a.mi_bias_corrected, NULL_TOLERANCE_BITS));
    let self_control_ok = arms
        .iter()
        .find(|a| a.label == ARM_SELF)
        .is_some_and(|a| a.conditional_entropy < SELF_CONTROL_TOLERANCE_BITS);

    let mut distinct: Vec<usize> = parent_choices.iter().map(|(_, p)| *p).collect();
    distinct.sort_unstable();
    distinct.dedup();

    let proto_cost = proto_j.conditional_entropy();
    Some(CondCensus {
        experts: streams.len(),
        coords_total: len,
        coords_fit: fit,
        coords_scored: len - fit,
        triple_conditional_entropy: triple.conditional_entropy(),
        parent_gain_over_prototype: proto_cost - triple.conditional_entropy(),
        selection_gain_over_random: random_j.conditional_entropy()
            - selected_j.conditional_entropy(),
        arms,
        parent_choices,
        distinct_parents: distinct.len(),
        nulls_clean,
        self_control_ok,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::primary::k3_ledger::symbol_census::FP4_CODES;

    fn stream(id: usize, nibbles: Vec<u8>) -> ExpertStream {
        ExpertStream { id, nibbles }
    }

    /// Tests that predate the column arm run without one; `run` is exercised
    /// with a real `row_len` by the permutation tests below.
    fn run_rows(streams: &[ExpertStream], fit_fraction: f64, seed: u64) -> Option<CondCensus> {
        run(streams, 0, fit_fraction, seed)
    }

    /// A bank with a strong shared per-coordinate profile plus per-expert noise.
    fn structured_bank(experts: usize, coords: usize) -> Vec<ExpertStream> {
        (0..experts)
            .map(|e| {
                let n = (0..coords)
                    .map(|c| {
                        let common = (c % 16) as u8;
                        if (c + e) % 5 == 0 {
                            (common + 1 + e as u8) % 16
                        } else {
                            common
                        }
                    })
                    .collect();
                stream(e, n)
            })
            .collect()
    }

    /// A bank with no aligned structure at all: each expert has its own phase.
    fn unstructured_bank(experts: usize, coords: usize) -> Vec<ExpertStream> {
        (0..experts)
            .map(|e| {
                let period = 7 + e;
                let n = (0..coords).map(|c| ((c % period) % 16) as u8).collect();
                stream(e, n)
            })
            .collect()
    }

    #[test]
    fn a_structured_bank_shows_a_large_prototype_gain() {
        let c = run_rows(&structured_bank(8, 8192), DEFAULT_FIT_FRACTION, 1).unwrap();
        let proto = c.arm(ARM_PROTOTYPE).unwrap();
        assert!(
            proto.conditional_entropy < 1.5,
            "conditional entropy {} should collapse on a shared profile",
            proto.conditional_entropy
        );
        assert!(proto.mutual_information > 2.0);
        assert!(c.nulls_clean, "the nulls must stay clean here");
    }

    #[test]
    fn an_unstructured_bank_shows_essentially_none() {
        let c = run_rows(&unstructured_bank(8, 8192), DEFAULT_FIT_FRACTION, 1).unwrap();
        let proto = c.arm(ARM_PROTOTYPE).unwrap();
        assert!(
            proto.conditional_entropy > proto.marginal_entropy - 0.9,
            "cond {} vs marginal {}",
            proto.conditional_entropy,
            proto.marginal_entropy
        );
    }

    #[test]
    fn every_null_collapses_and_is_reported_as_clean() {
        let c = run_rows(&structured_bank(6, 8192), DEFAULT_FIT_FRACTION, 3).unwrap();
        for label in [ARM_NULL_PROTOTYPE, ARM_NULL_PARENT] {
            let arm = c.arm(label).unwrap();
            assert!(
                arm.mi_bias_corrected.abs() <= NULL_TOLERANCE_BITS,
                "{label} leaked {} bits",
                arm.mi_bias_corrected
            );
        }
        assert!(c.nulls_clean);
    }

    #[test]
    fn a_dirty_null_refuses_the_whole_census_rather_than_ranking_the_arms() {
        // Forge the state a broken alignment would produce and check the verdict
        // refuses rather than reporting whichever arm happened to look best.
        let mut c = run_rows(&structured_bank(4, 4096), DEFAULT_FIT_FRACTION, 1).unwrap();
        c.nulls_clean = false;
        assert!(c.verdict().starts_with("REFUSED"));
    }

    #[test]
    fn parents_are_chosen_on_the_fit_slice_and_scored_on_the_rest() {
        let c = run_rows(&structured_bank(5, 4000), 0.25, 1).unwrap();
        assert_eq!(c.coords_total, 4000);
        assert_eq!(c.coords_fit, 1000);
        assert_eq!(c.coords_scored, 3000);
        // Every arm is scored on the same held-out count, so they compare.
        for a in &c.arms {
            assert_eq!(a.samples, 3000 * c.experts as u64);
        }
    }

    #[test]
    fn an_expert_never_selects_itself_as_its_own_parent() {
        let c = run_rows(&structured_bank(8, 4096), DEFAULT_FIT_FRACTION, 1).unwrap();
        assert_eq!(c.parent_choices.len(), 8);
        for (target, parent) in &c.parent_choices {
            assert_ne!(target, parent, "expert {target} chose itself");
        }
    }

    #[test]
    fn the_random_control_also_never_picks_the_target() {
        for n in 2..12usize {
            for target in 0..n {
                for seed in 0..8u64 {
                    let p = arbitrary_parent(n, target, seed).unwrap();
                    assert_ne!(p, target);
                    assert!(p < n);
                }
            }
        }
        assert!(arbitrary_parent(1, 0, 0).is_none());
    }

    #[test]
    fn the_triple_can_only_help_and_its_gain_is_reported() {
        let c = run_rows(&structured_bank(6, 8192), DEFAULT_FIT_FRACTION, 1).unwrap();
        let proto = c.arm(ARM_PROTOTYPE).unwrap();
        assert!(c.triple_conditional_entropy <= proto.conditional_entropy + 1e-9);
        assert!(c.parent_gain_over_prototype >= -1e-9);
    }

    #[test]
    fn a_bank_whose_experts_are_interchangeable_shows_no_selection_gain() {
        // Every expert is an equally good parent for every other, so choosing
        // one cannot beat taking any. This is the shape that kills ETC-0B.
        let identical: Vec<ExpertStream> = (0..6)
            .map(|e| stream(e, (0..4096).map(|c| (c % 16) as u8).collect()))
            .collect();
        let c = run_rows(&identical, DEFAULT_FIT_FRACTION, 1).unwrap();
        assert!(
            c.selection_gain_over_random.abs() < 1e-9,
            "selection gain {} should vanish",
            c.selection_gain_over_random
        );
    }

    #[test]
    fn a_hub_bank_is_visible_in_the_distinct_parent_count() {
        // Experts 1..5 each resemble expert 0 and nothing else, so all of them
        // should choose it — a hub, which is a prototype wearing a graph's hat.
        let mut bank = vec![stream(0, (0..4096).map(|c| (c % 16) as u8).collect())];
        for e in 1..6usize {
            let n = (0..4096)
                .map(|c| {
                    if c % 32 == e {
                        ((c + 1) % 16) as u8
                    } else {
                        (c % 16) as u8
                    }
                })
                .collect();
            bank.push(stream(e, n));
        }
        let c = run_rows(&bank, DEFAULT_FIT_FRACTION, 1).unwrap();
        assert!(
            c.distinct_parents <= 3,
            "expected a hub, got {} distinct parents",
            c.distinct_parents
        );
    }

    #[test]
    fn ragged_or_tiny_banks_are_refused() {
        assert!(run_rows(&[], DEFAULT_FIT_FRACTION, 1).is_none());
        assert!(run_rows(&[stream(0, vec![1, 2, 3])], DEFAULT_FIT_FRACTION, 1).is_none());
        let ragged = vec![stream(0, vec![1, 2, 3]), stream(1, vec![1, 2])];
        assert!(run_rows(&ragged, DEFAULT_FIT_FRACTION, 1).is_none());
    }

    #[test]
    fn a_stream_too_short_to_split_is_refused_rather_than_scored_on_nothing() {
        let one_coord = vec![stream(0, vec![1]), stream(1, vec![2])];
        assert!(run_rows(&one_coord, DEFAULT_FIT_FRACTION, 1).is_none());
        assert_eq!(split_at(1, 0.5), None);
        assert_eq!(split_at(10, 0.5), Some(5));
    }

    #[test]
    fn degenerate_fit_fractions_still_leave_both_slices_non_empty() {
        for f in [0.0, 1.0, -3.0, 7.0] {
            let cut = split_at(100, f).expect("clamped, not refused");
            assert!((1..=99).contains(&cut), "fit_fraction {f} gave cut {cut}");
        }
        let c = run_rows(&structured_bank(4, 2048), 0.0, 1).unwrap();
        assert!(c.coords_fit >= 1 && c.coords_scored >= 1);
    }

    #[test]
    fn the_best_arm_ignores_the_nulls_when_forming_a_verdict() {
        let c = run_rows(&structured_bank(8, 8192), DEFAULT_FIT_FRACTION, 1).unwrap();
        let best = c.best_conditional_entropy();
        let null_cost = c.arm(ARM_NULL_PROTOTYPE).unwrap().conditional_entropy;
        assert!(best < null_cost, "a null must never win the headline");
        assert!(!c.verdict().starts_with("REFUSED"));
    }

    /// A bank whose only shared structure is a per-COLUMN signature, buried in
    /// noise, with each expert's rows independently permuted — the real geometry in
    /// miniature. The noise is what separates the two arms: a row-aligned
    /// prototype pools `experts - 1` samples per coordinate, while a column
    /// profile pools `(experts - 1) * rows`, so only the second one recovers a
    /// weak signature. With the noise removed both arms find it and the test
    /// proves nothing.
    fn column_structured_permuted_bank(
        experts: usize,
        rows: usize,
        row_len: usize,
    ) -> Vec<ExpertStream> {
        (0..experts)
            .map(|e| {
                let mut noise = SplitMix64(0x00AB_CDEF ^ e as u64);
                let mut by_row: Vec<Vec<u8>> = (0..rows)
                    .map(|_| {
                        (0..row_len)
                            .map(|c| {
                                if noise.next_unit() < COLUMN_SIGNAL {
                                    (c % FP4_CODES) as u8
                                } else {
                                    noise.below(FP4_CODES as u64) as u8
                                }
                            })
                            .collect()
                    })
                    .collect();
                let mut order = SplitMix64(0x00C0_FFEE ^ e as u64);
                super::super::rng::shuffle(&mut by_row, &mut order);
                ExpertStream {
                    id: e,
                    nibbles: by_row.concat(),
                }
            })
            .collect()
    }

    /// Fraction of entries carrying the column signature in the fixture above.
    const COLUMN_SIGNAL: f64 = 0.45;

    #[test]
    fn the_self_control_reads_zero_and_gates_the_verdict() {
        let c = run(&structured_bank(4, 4096), 0, DEFAULT_FIT_FRACTION, 1).unwrap();
        let s = c.arm(ARM_SELF).unwrap();
        assert!(s.conditional_entropy < SELF_CONTROL_TOLERANCE_BITS);
        assert!(c.self_control_ok);

        let mut broken = c.clone();
        broken.self_control_ok = false;
        assert!(broken.verdict().starts_with("REFUSED"));
    }

    #[test]
    fn the_self_control_never_wins_the_headline() {
        // It reads 0.0000 by construction. If it competed, every census would
        // report a major result regardless of the data.
        let c = run(&unstructured_bank(6, 8192), 0, DEFAULT_FIT_FRACTION, 1).unwrap();
        assert!(c.best_conditional_entropy() > 1.0);
        assert!(is_diagnostic(ARM_SELF) && is_diagnostic(ARM_NULL_PARENT));
        assert!(!is_diagnostic(ARM_PROTOTYPE) && !is_diagnostic(ARM_COLUMN));
    }

    #[test]
    fn the_column_arm_finds_structure_the_row_aligned_arms_cannot() {
        // This is the whole reason the column arm exists: with rows permuted
        // per expert, coordinate-aligned conditioning sees nothing and the
        // permutation-invariant arm sees the shared column signature.
        let bank = column_structured_permuted_bank(4, 256, 32);
        let c = run(&bank, 32, DEFAULT_FIT_FRACTION, 1).unwrap();

        let proto = c.arm(ARM_PROTOTYPE).unwrap().conditional_entropy;
        let column = c.arm(ARM_COLUMN).unwrap().conditional_entropy;
        let parent = c.arm(ARM_PARENT_SELECTED).unwrap().conditional_entropy;
        assert!(
            column < proto - 0.1,
            "column {column} should beat row-aligned prototype {proto}"
        );
        assert!(
            column < parent - 0.1,
            "column {column} should beat the best row-aligned parent {parent}"
        );
        assert!(c.nulls_clean && c.self_control_ok);
    }

    #[test]
    fn the_column_arm_is_skipped_when_no_row_length_is_supplied() {
        let c = run(&structured_bank(4, 4096), 0, DEFAULT_FIT_FRACTION, 1).unwrap();
        assert!(c.arm(ARM_COLUMN).is_none());
        assert!(c.arm(ARM_NULL_COLUMN).is_none());
        assert!(c.arm(ARM_PREV_ROW).is_none());
    }

    #[test]
    fn the_previous_row_control_reports_column_structure_within_one_expert() {
        let bank = column_structured_permuted_bank(4, 256, 32);
        let c = run(&bank, 32, DEFAULT_FIT_FRACTION, 1).unwrap();
        let prev = c.arm(ARM_PREV_ROW).unwrap();
        assert!(
            prev.mutual_information > 0.1,
            "adjacent rows share the column signature: MI {}",
            prev.mutual_information
        );
    }

    #[test]
    fn the_column_null_collapses_like_the_others() {
        let bank = column_structured_permuted_bank(6, 128, 32);
        let c = run(&bank, 32, DEFAULT_FIT_FRACTION, 5).unwrap();
        let null = c.arm(ARM_NULL_COLUMN).unwrap();
        assert!(
            null.mi_bias_corrected.abs() <= NULL_TOLERANCE_BITS,
            "column null leaked {} bits",
            null.mi_bias_corrected
        );
    }

    #[test]
    fn the_census_is_reproducible_from_its_seed() {
        let bank = structured_bank(6, 4096);
        let a = run_rows(&bank, DEFAULT_FIT_FRACTION, 9).unwrap();
        let b = run_rows(&bank, DEFAULT_FIT_FRACTION, 9).unwrap();
        assert_eq!(a.parent_choices, b.parent_choices);
        assert_eq!(
            a.arm(ARM_NULL_PROTOTYPE).unwrap().mutual_information,
            b.arm(ARM_NULL_PROTOTYPE).unwrap().mutual_information
        );
    }
}
