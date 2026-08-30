//! Promotion counters (spec §20).
//!
//! The divergence counters are split by *class*, not merged into one
//! "failures" total, because payload divergence and first-termination
//! divergence are different claims about the same replacement — the
//! measured case has records that are sound for the answer and unsound
//! for the decision to stop.

use super::scoring::DivergenceClass;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PromotionMetrics {
    pub promotions_planned: u64,
    pub promotions_prepared: u64,
    pub promotions_committed: u64,
    pub promotions_aborted: u64,

    pub source_bytes_hidden: u64,
    pub source_bytes_reclaimed: u64,
    pub promoted_bytes: u64,
    pub replay_bytes: u64,

    pub promotion_tokens: u64,
    pub promotion_latency_ns: u64,
    pub replay_latency_ns: u64,

    pub answer_scoped_retirements: u64,
    pub general_state_retirements: u64,

    pub payload_divergences: u64,
    pub first_termination_divergences: u64,
    pub post_termination_divergences: u64,
}

impl PromotionMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one scored comparison into the divergence counters.
    /// `DivergenceClass::None` increments nothing.
    pub fn record_divergence(&mut self, class: DivergenceClass) {
        match class {
            DivergenceClass::None => {}
            DivergenceClass::Payload => self.payload_divergences += 1,
            DivergenceClass::FirstTermination => self.first_termination_divergences += 1,
            DivergenceClass::PostTermination => self.post_termination_divergences += 1,
        }
    }

    /// Promotions that were prepared but neither committed nor aborted —
    /// leaked leases. Should be zero at session end.
    pub fn promotions_outstanding(&self) -> u64 {
        self.promotions_prepared
            .saturating_sub(self.promotions_committed)
            .saturating_sub(self.promotions_aborted)
    }

    /// Bytes the session is currently keeping alive for rollback: hidden
    /// but not yet reclaimed, plus the promoted record itself.
    pub fn resident_promotion_bytes(&self) -> u64 {
        self.source_bytes_hidden
            .saturating_sub(self.source_bytes_reclaimed)
            .saturating_add(self.promoted_bytes)
    }

    /// Net bytes saved. Negative while a hidden source is still resident,
    /// which is the honest read of `EnforceResidentRollback`: it costs
    /// memory and buys correctness headroom, and only Phase F's physical
    /// reclamation turns that positive.
    pub fn net_bytes_saved(&self) -> i64 {
        self.source_bytes_reclaimed as i64 - self.promoted_bytes as i64 - self.replay_bytes as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn divergence_classes_are_counted_separately() {
        let mut m = PromotionMetrics::new();
        m.record_divergence(DivergenceClass::Payload);
        m.record_divergence(DivergenceClass::FirstTermination);
        m.record_divergence(DivergenceClass::FirstTermination);
        m.record_divergence(DivergenceClass::PostTermination);
        m.record_divergence(DivergenceClass::None);
        assert_eq!(m.payload_divergences, 1);
        assert_eq!(m.first_termination_divergences, 2);
        assert_eq!(m.post_termination_divergences, 1);
    }

    #[test]
    fn outstanding_promotions_track_leaked_leases() {
        let mut m = PromotionMetrics::new();
        m.promotions_prepared = 3;
        m.promotions_committed = 1;
        m.promotions_aborted = 1;
        assert_eq!(m.promotions_outstanding(), 1);

        m.promotions_aborted = 2;
        assert_eq!(m.promotions_outstanding(), 0);
    }

    #[test]
    fn outstanding_count_saturates_rather_than_underflowing() {
        let mut m = PromotionMetrics::new();
        m.promotions_prepared = 1;
        m.promotions_committed = 5;
        assert_eq!(m.promotions_outstanding(), 0);
    }

    #[test]
    fn resident_rollback_shows_a_memory_cost_not_a_saving() {
        let mut m = PromotionMetrics::new();
        m.source_bytes_hidden = 1_000;
        m.promoted_bytes = 40;
        // Nothing reclaimed yet: the source is still resident.
        assert_eq!(m.resident_promotion_bytes(), 1_040);
        assert_eq!(m.net_bytes_saved(), -40);
    }

    #[test]
    fn reclamation_turns_the_saving_positive() {
        let mut m = PromotionMetrics::new();
        m.source_bytes_hidden = 1_000;
        m.source_bytes_reclaimed = 1_000;
        m.promoted_bytes = 40;
        m.replay_bytes = 10;
        assert_eq!(m.resident_promotion_bytes(), 40);
        assert_eq!(m.net_bytes_saved(), 950);
    }

    #[test]
    fn a_fresh_metrics_block_is_all_zero() {
        let m = PromotionMetrics::new();
        assert_eq!(m, PromotionMetrics::default());
        assert_eq!(m.promotions_outstanding(), 0);
        assert_eq!(m.net_bytes_saved(), 0);
        assert_eq!(m.resident_promotion_bytes(), 0);
    }
}
