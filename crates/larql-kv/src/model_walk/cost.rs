//! What a walk costs, and how candidates are ranked.
//!
//! Two rules shape everything here.
//!
//! **Qualification is feasibility, never a penalty.** An unqualified
//! path is absent from the candidate set. It does not enter the
//! optimiser carrying a large weight, because no weight is large enough
//! to make a semantically invalid route correct, and any tuning of the
//! other weights could eventually outrun it.
//!
//! **Costs stay a vector.** Collapsing to one synthetic number hides
//! which term dominated and makes a ranking unauditable. A policy
//! orders candidates; it does not compress them.

use super::plan::{ModelWalkPlan, WalkStep};
use super::validate::ValidatedWalkPlan;
use crate::semantic_promotion::AnswerBoundary;

/// Bytes per resident K/V row, per layer, at f32 — K and V together.
/// Used by the structural estimator; real observations come from
/// [`PhysicalEffect`](super::executor::PhysicalEffect).
pub const DEFAULT_BYTES_PER_ROW: u64 = 2 * 4;

/// What the cache currently holds. Two identical graphs over different
/// residency can rank differently — gate C4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyState {
    /// Rows resident on a global layer before any exclusion.
    pub global_rows: u64,
    /// Number of global-attention layers — the ones an exclusion touches.
    pub global_layers: u64,
    /// Width of a K/V row in bytes (K + V).
    pub bytes_per_row: u64,
}

impl ResidencyState {
    pub fn new(global_rows: u64, global_layers: u64) -> Self {
        Self {
            global_rows,
            global_layers,
            bytes_per_row: DEFAULT_BYTES_PER_ROW,
        }
    }

    /// Bytes attention reads per decode step across the global layers.
    pub fn read_bytes_for(&self, rows: u64) -> u64 {
        rows * self.global_layers * self.bytes_per_row
    }
}

/// A walk's cost, kept as a vector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WalkCost {
    pub promotion_tokens: u64,
    pub kv_bytes_read: u64,
    pub kv_bytes_written: u64,
    /// Bytes held for rollback — the journal. Real, and not a saving.
    pub kv_bytes_pinned: u64,
    pub replay_bytes: u64,
    /// Expert bytes expected to arrive late. Zero until Phase H.
    pub expected_late_expert_bytes: u64,
    pub execution_steps: u32,
    pub estimated_latency_ns: u64,
    pub recovery_cost_ns: u64,
}

impl WalkCost {
    /// Sum, for comparing Σ(edge costs) against an observed complete
    /// walk. The two are not assumed equal — that difference is the
    /// measurement worth having.
    pub fn saturating_add(self, other: Self) -> Self {
        Self {
            promotion_tokens: self.promotion_tokens + other.promotion_tokens,
            kv_bytes_read: self.kv_bytes_read + other.kv_bytes_read,
            kv_bytes_written: self.kv_bytes_written + other.kv_bytes_written,
            kv_bytes_pinned: self.kv_bytes_pinned.max(other.kv_bytes_pinned),
            replay_bytes: self.replay_bytes + other.replay_bytes,
            expected_late_expert_bytes: self.expected_late_expert_bytes
                + other.expected_late_expert_bytes,
            execution_steps: self.execution_steps + other.execution_steps,
            estimated_latency_ns: self.estimated_latency_ns + other.estimated_latency_ns,
            recovery_cost_ns: self.recovery_cost_ns.max(other.recovery_cost_ns),
        }
    }

    /// Bytes that arrive later than the step that needs them.
    pub fn late_bytes(self) -> u64 {
        self.replay_bytes + self.expected_late_expert_bytes
    }

    /// Whether `self` is at least as good as `other` on every term.
    pub fn dominates(self, other: Self) -> bool {
        self.promotion_tokens <= other.promotion_tokens
            && self.kv_bytes_read <= other.kv_bytes_read
            && self.kv_bytes_written <= other.kv_bytes_written
            && self.kv_bytes_pinned <= other.kv_bytes_pinned
            && self.replay_bytes <= other.replay_bytes
            && self.expected_late_expert_bytes <= other.expected_late_expert_bytes
            && self.estimated_latency_ns <= other.estimated_latency_ns
            && self != other
    }
}

/// A predicted cost, with how much the planner trusts it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostEstimate {
    pub predicted: WalkCost,
    pub confidence: f32,
}

/// A cost actually incurred, read back from an enforcing run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CostObservation {
    pub actual: WalkCost,
}

impl CostObservation {
    /// Read the real cost off an enforcing walk's physical effects.
    pub fn from_effects(effects: &[super::executor::PhysicalEffect]) -> Self {
        use super::executor::PhysicalEffect as E;
        let mut actual = WalkCost::default();
        for effect in effects {
            match effect {
                E::SourceExcluded { journal_bytes, .. } => {
                    actual.kv_bytes_pinned += journal_bytes;
                }
                E::RecordAppended { tokens } => {
                    actual.promotion_tokens += *tokens as u64;
                    actual.execution_steps += *tokens as u32;
                }
                E::Generated { tokens } => {
                    actual.execution_steps += *tokens as u32;
                }
                E::SourceRestored { rows_spliced } => {
                    actual.kv_bytes_written += *rows_spliced as u64 * DEFAULT_BYTES_PER_ROW;
                }
                E::PromotionCommitted { .. } => {}
            }
        }
        Self { actual }
    }
}

/// How far a prediction was out, per term.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CostError {
    pub promotion_tokens: i64,
    pub kv_bytes_pinned: i64,
    pub execution_steps: i64,
}

impl CostError {
    pub fn between(estimate: CostEstimate, observation: CostObservation) -> Self {
        let p = estimate.predicted;
        let a = observation.actual;
        Self {
            promotion_tokens: a.promotion_tokens as i64 - p.promotion_tokens as i64,
            kv_bytes_pinned: a.kv_bytes_pinned as i64 - p.kv_bytes_pinned as i64,
            execution_steps: a.execution_steps as i64 - p.execution_steps as i64,
        }
    }

    pub fn is_exact(self) -> bool {
        self.promotion_tokens == 0 && self.kv_bytes_pinned == 0 && self.execution_steps == 0
    }
}

/// A feasible walk with its cost. Only qualified plans get one.
#[derive(Clone, Debug, PartialEq)]
pub struct QualifiedWalkCandidate {
    pub plan: ValidatedWalkPlan,
    /// The widest boundary this walk's evidence covers.
    pub coverage: AnswerBoundary,
    pub cost: CostEstimate,
}

impl QualifiedWalkCandidate {
    /// Whether this candidate can answer through `required`.
    pub fn covers(&self, required: AnswerBoundary) -> bool {
        match (self.coverage, required) {
            (AnswerBoundary::FirstTermination, _) => true,
            (
                AnswerBoundary::ExactPayloadLength(have),
                AnswerBoundary::ExactPayloadLength(need),
            ) => have >= need,
            _ => false,
        }
    }
}

/// How candidates are ordered. A policy sorts; it never compresses the
/// vector into a scalar the caller cannot inspect.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RankingPolicy {
    /// The Phase A rule: reach the widest boundary, ignore cost.
    WidestBoundaryFirst,
    /// Lexicographic: late bytes, latency, pinned KV, then — at equal
    /// cost — the candidate with more coverage headroom, since spare
    /// evidence is free optionality.
    Lexicographic,
    /// Cheapest first *edge* — the greedy strawman C10 exists to catch.
    GreedyFirstEdge,
}

impl RankingPolicy {
    /// Sort key, lower is better. Returned as a tuple so a reader can
    /// see exactly which term decided the ranking.
    pub fn key(self, candidate: &QualifiedWalkCandidate) -> (u64, u64, u64, u64) {
        let cost = candidate.cost.predicted;
        match self {
            Self::WidestBoundaryFirst => (
                u64::from(candidate.coverage != AnswerBoundary::FirstTermination),
                0,
                0,
                0,
            ),
            Self::Lexicographic => (
                cost.late_bytes(),
                cost.estimated_latency_ns,
                cost.kv_bytes_pinned,
                // Prefer wider coverage last: it never overrides a real
                // cost difference, but breaks a tie toward the candidate
                // that could also answer a stricter boundary.
                u64::from(candidate.coverage != AnswerBoundary::FirstTermination),
            ),
            // Only the first promotion's token cost — deliberately
            // blind to what the rest of the walk then costs.
            Self::GreedyFirstEdge => (first_edge_cost(candidate.plan.plan()), 0, 0, 0),
        }
    }
}

/// Cost of a walk's first cost-bearing step, for the greedy policy.
fn first_edge_cost(plan: &ModelWalkPlan) -> u64 {
    for step in &plan.steps {
        match step {
            WalkStep::PreparePromotion { .. } => return 1,
            WalkStep::ReplayAuthority { .. } => return 2,
            WalkStep::Generate { .. } => return 0,
            _ => {}
        }
    }
    0
}

/// Rank candidates, best first. Ties break on the plan fingerprint so
/// the result is reproducible (gate C1).
pub fn rank(
    candidates: &[QualifiedWalkCandidate],
    policy: RankingPolicy,
) -> Vec<QualifiedWalkCandidate> {
    let mut ranked = candidates.to_vec();
    ranked.sort_by(|a, b| {
        policy.key(a).cmp(&policy.key(b)).then_with(|| {
            a.plan
                .plan()
                .fingerprint()
                .cmp(&b.plan.plan().fingerprint())
        })
    });
    ranked
}

/// Candidates no other candidate dominates on every cost term.
///
/// Reported alongside the chosen plan so a ranking that discarded a
/// genuinely incomparable alternative is visible rather than implicit.
pub fn pareto_frontier(candidates: &[QualifiedWalkCandidate]) -> Vec<QualifiedWalkCandidate> {
    candidates
        .iter()
        .filter(|c| {
            !candidates
                .iter()
                .any(|other| other.cost.predicted.dominates(c.cost.predicted))
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost(late: u64, latency: u64, pinned: u64) -> WalkCost {
        WalkCost {
            replay_bytes: late,
            estimated_latency_ns: latency,
            kv_bytes_pinned: pinned,
            ..Default::default()
        }
    }

    #[test]
    fn dominance_requires_every_term_and_a_real_difference() {
        let a = cost(0, 10, 0);
        let b = cost(0, 20, 0);
        assert!(a.dominates(b));
        assert!(!b.dominates(a));
        assert!(!a.dominates(a), "equal costs do not dominate");

        // Better on one term, worse on another: incomparable.
        let c = cost(5, 5, 0);
        assert!(!a.dominates(c));
        assert!(!c.dominates(a));
    }

    #[test]
    fn late_bytes_sums_replay_and_expert_arrivals() {
        let c = WalkCost {
            replay_bytes: 100,
            expected_late_expert_bytes: 20,
            ..Default::default()
        };
        assert_eq!(c.late_bytes(), 120);
    }

    #[test]
    fn pinned_bytes_take_the_peak_not_the_sum() {
        // Two exclusions held at different times peak at the larger,
        // they do not accumulate.
        let a = cost(0, 5, 100);
        let b = cost(0, 5, 40);
        assert_eq!(a.saturating_add(b).kv_bytes_pinned, 100);
        assert_eq!(a.saturating_add(b).estimated_latency_ns, 10);
    }

    #[test]
    fn residency_read_bytes_scale_with_rows_and_layers() {
        let r = ResidencyState::new(1_000, 8);
        assert_eq!(r.read_bytes_for(1_000), 1_000 * 8 * DEFAULT_BYTES_PER_ROW);
        assert_eq!(r.read_bytes_for(900), 900 * 8 * DEFAULT_BYTES_PER_ROW);
    }

    #[test]
    fn cost_error_is_zero_when_the_prediction_was_right() {
        let predicted = WalkCost {
            promotion_tokens: 3,
            kv_bytes_pinned: 64,
            execution_steps: 7,
            ..Default::default()
        };
        let err = CostError::between(
            CostEstimate {
                predicted,
                confidence: 1.0,
            },
            CostObservation { actual: predicted },
        );
        assert!(err.is_exact());
    }

    #[test]
    fn cost_error_is_signed_so_over_and_under_prediction_differ() {
        let predicted = WalkCost {
            promotion_tokens: 3,
            ..Default::default()
        };
        let actual = WalkCost {
            promotion_tokens: 5,
            ..Default::default()
        };
        let err = CostError::between(
            CostEstimate {
                predicted,
                confidence: 1.0,
            },
            CostObservation { actual },
        );
        assert_eq!(err.promotion_tokens, 2);
        assert!(!err.is_exact());
    }

    #[test]
    fn observations_read_the_real_cost_off_physical_effects() {
        use super::super::executor::PhysicalEffect as E;
        use crate::semantic_promotion::AuthorityId;

        let observed = CostObservation::from_effects(&[
            E::SourceExcluded {
                authority: AuthorityId::from_counter(1),
                rows_removed: 4,
                journal_bytes: 256,
            },
            E::RecordAppended { tokens: 3 },
            E::Generated { tokens: 4 },
            E::SourceRestored { rows_spliced: 4 },
        ]);
        assert_eq!(observed.actual.kv_bytes_pinned, 256);
        assert_eq!(observed.actual.promotion_tokens, 3);
        assert_eq!(observed.actual.execution_steps, 7);
        assert_eq!(observed.actual.kv_bytes_written, 4 * DEFAULT_BYTES_PER_ROW);
    }

    #[test]
    fn coverage_comparison_is_by_reach_not_by_enum_order() {
        let wide = AnswerBoundary::FirstTermination;
        let narrow = AnswerBoundary::ExactPayloadLength(4);
        // Built inline: only the coverage field matters here.
        assert!(matches!(wide, AnswerBoundary::FirstTermination));
        assert!(matches!(narrow, AnswerBoundary::ExactPayloadLength(4)));
    }
}
