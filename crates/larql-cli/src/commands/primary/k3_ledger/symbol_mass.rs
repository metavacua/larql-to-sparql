//! Per-symbol routing mass — the measured `p_{l,e}` vector, exported raw.
//!
//! The precision allocator (DEC-8.4) minimises expected traffic
//! `sum_{l,e} p_{l,e} * b(q_{l,e})` subject to a memory bound and a fidelity
//! bound. It needs `p` per symbol, not a cumulative coverage curve: recovering
//! per-symbol mass by differencing coverage points loses the tie structure and
//! the sample size, and cannot be recombined across captures.
//!
//! **Counts are emitted alongside probabilities on purpose.** A probability
//! alone cannot distinguish a genuinely rare expert from a short trace, carries
//! no weight for shrinkage, and cannot be pooled with another capture without
//! inventing pseudo-counts. `events` is the sufficient statistic; the
//! probability is a convenience derived from it.
//!
//! ## Rank convention, and what it costs (R0)
//!
//! `rank` here orders the **pooled** distribution over the whole capture. The
//! coverage curve's static arm ranks leave-one-out, per scored session, so a
//! resident set built from this rank and then scored on the same capture is
//! **optimistic**: it has seen the events it is judged on.
//!
//! Naming the difference is not enough, so here is its size. Measured on the
//! DEC-0 Gemma pool (8-of-128, 64 sessions x 16 steps, 1920 cells) under the
//! shipped estimator's protocol — eval on each session's held-out second half:
//!
//! | resident set | pooled-self | static leave-one-out | optimism |
//! |---|---|---|---|
//! | C=4  (3.1% of bank)  | 0.2609 | 0.2585 | +0.0024 |
//! | C=8  (6.2%)          | 0.4252 | 0.4226 | +0.0026 |
//! | C=16 (12.5%)         | 0.6248 | 0.6204 | +0.0044 |
//! | C=32 (25.0%)         | 0.8266 | 0.8206 | +0.0060 |
//!
//! So the bound is **+0.000 to +0.006 mass, and the sign is always positive**.
//! Small, but a slice built from this export must not be quoted at the coverage
//! curve's number — the rule is: *pooled mass proposes the serving image,
//! held-out routing evaluates it.* The magnitude is capture-specific; the
//! direction is not.
//!
//! ## Absent symbols are a measurement, not a serialisation detail
//!
//! A symbol with no row was **unobserved in this capture**. That is not the
//! same claim as zero routing probability, and the two must not be conflated:
//! `alphabet - observed` is the measured size of the cold tail, and it is one
//! of the numbers DEC-8.4's precision allocator is actually asking for. It is
//! reported per stratum and in aggregate whether or not the rows are emitted.
//!
//! On the DEC-0 pool the count is far too large to be a sampling artifact: at
//! 1024 steps and top-8, an expert routed at the uniform rate would go
//! unobserved with probability ~1e-29. By the rule of three an unobserved
//! expert sits below 3/1024 per step against a uniform 0.0625 — at least
//! twentyfold below uniform.
//!
//! ## The domain-mixture caveat has a sign, and it differs per statistic
//!
//! This capture is 64 unrelated single-turn prompts — a domain **mixture**. The
//! standing habit is to attach that caveat once, at capture level, which applies
//! it with the wrong sign to half the outputs. It depends on whether the
//! statistic rewards breadth or narrowness:
//!
//! - **Coverage curve — anti-conservative.** A mixture is the anti-case for
//!   domain locality, so a domain-specific slice would do *better* than measured.
//! - **Unobserved count / participation ratio — conservative.** A
//!   domain-selective layer would show a *wide* union across 64 unrelated
//!   prompts. Measuring a narrow one anyway is stronger evidence of genuine
//!   narrowness than the same number on a single-domain pool would be.
//!
//! Same pool, opposite directions. State the sign with the statistic.
//!
//! ## Frequency identifies pruning candidates; it does not license pruning
//!
//! An expert unobserved here may still matter on an unrepresented input. Only
//! mean-ablation under DEC-8.1's protocol — held-out KL, top-1 agreement,
//! replacement controls — can decide deletion. The R4 zero-out is the precedent:
//! a reduction real in one currency (row count) converted at 23-40% into the one
//! that mattered (throughput). Frequency-to-fidelity is the same conversion with
//! different units.
//!
//! One structural point in pruning's favour, worth recording beside the hold:
//! removing whole experts leaves the survivors running **grouped dense**, so it
//! composes with compiled compact-dense instead of fighting it. R4 refuted the
//! runtime-gather form of the sparsity thesis, not this one — if mean-ablation
//! clears it, it lands on the surviving route rather than reopening a closed one.

use serde::Serialize;

use super::selection_trace::SelectionTrace;

/// One symbol's measured share of one stratum's selection events.
#[derive(Debug, Clone, Serialize)]
pub struct SymbolMass {
    pub stratum: usize,
    pub symbol_id: u32,
    /// Raw selection count — the sufficient statistic.
    pub events: u64,
    /// `events / stratum_events`.
    pub probability: f64,
    /// Position in the pooled descending order, 0-based. Ties break by id.
    pub rank: usize,
    /// Mass covered by every symbol up to and including this one.
    pub cumulative_mass: f64,
    pub stratum_events: u64,
}

/// How much of one stratum's bank this capture ever routed to.
///
/// `unobserved` is the cold tail's measured size for that stratum. It varies
/// far more than a single aggregate suggests — 3 to 68 of 128 across the DEC-0
/// pool's 30 layers — which is why precision tiering is a per-stratum decision
/// rather than one global cutoff.
#[derive(Debug, Clone, Serialize)]
pub struct StratumCoverage {
    pub stratum: usize,
    pub observed: usize,
    /// `alphabet - observed`: symbols this capture never routed to. NOT a claim
    /// that their probability is zero.
    pub unobserved: usize,
}

/// The capture-level conventions every row above is conditional on.
#[derive(Debug, Clone, Serialize)]
pub struct MassCensus {
    /// What a symbol is — never assume expert.
    pub selection_unit: &'static str,
    /// How `rank` was formed, distinct from the coverage curve's static arm.
    pub rank_convention: &'static str,
    pub strata: usize,
    pub routing_strata: usize,
    /// Symbols per stratum, inferred from the trace where the source cannot
    /// state it.
    pub alphabet: usize,
    pub selection_width: usize,
    pub activation_fraction: f64,
    pub sessions: usize,
    pub steps: usize,
    pub total_events: u64,
    /// Symbol-stratum pairs this capture routed to at least once.
    pub observed_symbols: usize,
    /// `routing_strata * alphabet - observed_symbols` — the measured cold tail.
    /// Emitted whether or not `rows` is, because this is the datum DEC-8.4
    /// wants and it costs one integer.
    pub unobserved_symbols: usize,
    pub coverage_by_stratum: Vec<StratumCoverage>,
    /// Omitted when the caller did not ask for the full vector; the counts
    /// above stand on their own.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub rows: Vec<SymbolMass>,
}

impl MassCensus {
    /// Drop the per-symbol vector, keeping every aggregate. At K3's 92x896 the
    /// rows dominate the document, but the cold-tail counts must survive.
    pub fn without_rows(mut self) -> Self {
        self.rows = Vec::new();
        self
    }
}

const RANK_CONVENTION: &str = "pooled_descending_over_capture";

/// Census every symbol that a stratum ever selected.
///
/// Symbols never routed to emit no row — they carry no mass and would dominate
/// the row count at K3's 896-wide bank — but they are *counted*, per stratum
/// and in aggregate, because their number is the cold tail's measured size.
pub fn census(trace: &SelectionTrace) -> MassCensus {
    let mut rows = Vec::new();
    let mut coverage_by_stratum = Vec::new();
    let mut total_events = 0u64;

    for &stratum in trace.active_strata() {
        let mut counts = vec![0.0f64; trace.alphabet()];
        for session in 0..trace.sessions() {
            trace.accumulate(session, stratum, 0..trace.steps(), &mut counts);
        }
        let stratum_events: u64 = counts.iter().map(|&c| c as u64).sum();
        if stratum_events == 0 {
            continue;
        }
        total_events += stratum_events;

        let mut order: Vec<usize> = (0..counts.len()).filter(|&i| counts[i] > 0.0).collect();
        order.sort_by(|&a, &b| counts[b].total_cmp(&counts[a]).then(a.cmp(&b)));
        coverage_by_stratum.push(StratumCoverage {
            stratum,
            observed: order.len(),
            unobserved: trace.alphabet() - order.len(),
        });

        let denom = stratum_events as f64;
        let mut cumulative = 0.0;
        for (rank, idx) in order.into_iter().enumerate() {
            let events = counts[idx] as u64;
            let probability = events as f64 / denom;
            cumulative += probability;
            rows.push(SymbolMass {
                stratum,
                symbol_id: idx as u32,
                events,
                probability,
                rank,
                cumulative_mass: cumulative,
                stratum_events,
            });
        }
    }

    let observed_symbols: usize = coverage_by_stratum.iter().map(|c| c.observed).sum();
    MassCensus {
        selection_unit: trace.unit().as_str(),
        rank_convention: RANK_CONVENTION,
        strata: trace.strata(),
        routing_strata: coverage_by_stratum.len(),
        alphabet: trace.alphabet(),
        selection_width: trace.width(),
        activation_fraction: trace.activation_fraction(),
        sessions: trace.sessions(),
        steps: trace.steps(),
        total_events,
        observed_symbols,
        unobserved_symbols: coverage_by_stratum.len() * trace.alphabet() - observed_symbols,
        coverage_by_stratum,
        rows,
    }
}

#[cfg(test)]
mod tests {
    use super::super::selection_trace::{SelectionUnit, SENTINEL};
    use super::*;

    /// 2 sessions x 2 steps x 2 strata x width 2, bank 4; stratum 1 is silent.
    /// Stratum 0 selects: {0,1}, {0,2}, {3,1}, {3,3}
    /// => counts 0:2, 1:2, 2:1, 3:3 over 8 events.
    fn fixture() -> SelectionTrace {
        let s = SENTINEL;
        SelectionTrace::new(
            2,
            2,
            2,
            2,
            4,
            vec![
                0, 1, s, s, //
                0, 2, s, s, //
                3, 1, s, s, //
                3, 3, s, s, //
            ],
        )
        .unwrap()
    }

    #[test]
    fn counts_and_probabilities_are_both_emitted() {
        let c = census(&fixture());
        let by_id = |id: u32| c.rows.iter().find(|r| r.symbol_id == id).unwrap();
        assert_eq!(by_id(3).events, 3);
        assert_eq!(by_id(0).events, 2);
        assert_eq!(by_id(2).events, 1);
        assert!((by_id(3).probability - 3.0 / 8.0).abs() < 1e-12);
        assert_eq!(c.total_events, 8);
        assert!(c.rows.iter().all(|r| r.stratum_events == 8));
    }

    #[test]
    fn rank_is_pooled_descending_with_ties_broken_by_id() {
        let c = census(&fixture());
        let ranked: Vec<u32> = c.rows.iter().map(|r| r.symbol_id).collect();
        // 3 (three events), then 0 and 1 tied at two — id order — then 2.
        assert_eq!(ranked, vec![3, 0, 1, 2]);
        assert_eq!(c.rows[0].rank, 0);
        assert_eq!(c.rows[3].rank, 3);
    }

    #[test]
    fn cumulative_mass_is_monotone_and_reaches_one() {
        let c = census(&fixture());
        let mut last = 0.0;
        for r in &c.rows {
            assert!(r.cumulative_mass >= last - 1e-12);
            last = r.cumulative_mass;
        }
        assert!((last - 1.0).abs() < 1e-12);
    }

    #[test]
    fn unobserved_symbols_are_counted_even_though_they_emit_no_row() {
        // Bank of 8, only symbol 0 ever routed, one routing stratum.
        let sparse = SelectionTrace::new(1, 2, 1, 1, 8, vec![0, 0]).unwrap();
        let c = census(&sparse);
        assert_eq!(c.observed_symbols, 1);
        assert_eq!(c.unobserved_symbols, 7, "alphabet 8 minus the one observed");
        assert_eq!(c.coverage_by_stratum.len(), 1);
        assert_eq!(c.coverage_by_stratum[0].unobserved, 7);
        assert_eq!(c.coverage_by_stratum[0].observed, 1);
    }

    #[test]
    fn the_cold_tail_counts_survive_dropping_the_rows() {
        let full = census(&fixture());
        let thin = census(&fixture()).without_rows();
        assert!(thin.rows.is_empty());
        assert_eq!(thin.observed_symbols, full.observed_symbols);
        assert_eq!(thin.unobserved_symbols, full.unobserved_symbols);
        assert_eq!(thin.total_events, full.total_events);
        assert_eq!(
            thin.coverage_by_stratum.len(),
            full.coverage_by_stratum.len()
        );
    }

    #[test]
    fn silent_strata_do_not_inflate_the_unobserved_count() {
        // The fixture's stratum 1 never routes at all. It must not contribute
        // `alphabet` phantom cold symbols — an absent stratum is not a cold
        // bank, and conflating them would overstate the tail by a whole layer.
        let c = census(&fixture());
        assert_eq!(c.strata, 2);
        assert_eq!(c.routing_strata, 1);
        assert_eq!(c.observed_symbols, 4);
        assert_eq!(c.unobserved_symbols, 0, "all 4 of stratum 0's bank routed");
    }

    #[test]
    fn silent_strata_and_unselected_symbols_are_omitted() {
        let c = census(&fixture());
        assert!(c.rows.iter().all(|r| r.stratum == 0), "stratum 1 is silent");
        assert_eq!(c.rows.len(), 4, "all four symbols were selected here");

        // A symbol the trace never picks leaves no row, but `alphabet` records
        // that it exists.
        let s = SENTINEL;
        let sparse = SelectionTrace::new(1, 2, 1, 1, 8, vec![0, 0]).unwrap();
        let c = census(&sparse);
        assert_eq!(c.rows.len(), 1);
        assert_eq!(c.alphabet, 8);
        let _ = s;
    }

    #[test]
    fn the_census_carries_its_conventions() {
        let c = census(&fixture().with_unit(SelectionUnit::Expert));
        assert_eq!(c.selection_unit, "expert");
        assert_eq!(c.rank_convention, RANK_CONVENTION);
        assert_eq!((c.strata, c.routing_strata), (2, 1));
        assert_eq!((c.alphabet, c.selection_width), (4, 2));
        assert!((c.activation_fraction - 0.5).abs() < 1e-12);
        assert_eq!((c.sessions, c.steps), (2, 2));
    }

    #[test]
    fn an_unlabelled_trace_does_not_claim_to_be_experts() {
        assert_eq!(census(&fixture()).selection_unit, "symbol");
        let feat = fixture().with_unit(SelectionUnit::Feature);
        assert_eq!(census(&feat).selection_unit, "feature");
    }

    #[test]
    fn probabilities_within_a_stratum_sum_to_one() {
        let c = census(&fixture());
        let total: f64 = c.rows.iter().map(|r| r.probability).sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_trace_with_no_selections_censuses_to_nothing() {
        let s = SENTINEL;
        let silent = SelectionTrace::new(1, 2, 1, 1, 2, vec![s, s]).unwrap();
        let c = census(&silent);
        assert!(c.rows.is_empty());
        assert_eq!(c.total_events, 0);
        assert_eq!(c.routing_strata, 0);
    }
}
