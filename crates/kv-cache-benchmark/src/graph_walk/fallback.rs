/// Tier A/B/C fallback routing for graph walk.
///
/// Tier A: cached template walk — known template, entity KNN only (<0.1ms)
/// Tier B: dynamic graph walk — full routing table lookup (~1-5ms)
/// Tier C: Markov RS fallback — full RS forward pass for free-form generation (~200ms)
///
/// The benchmark reports what % of queries resolve at each tier
/// and the accuracy per tier vs full forward pass baseline.

use super::walk_state::{WalkState, WalkTier};

/// Result of tier-based routing.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TierResult {
    pub tier: WalkTier,
    pub latency_us: f64,
    pub resolved: bool,
    pub description: &'static str,
}

/// Route a walk state to its resolution tier.
pub fn route_to_tier(state: &WalkState) -> TierResult {
    match state.tier {
        WalkTier::CachedTemplate => TierResult {
            tier: WalkTier::CachedTemplate,
            latency_us: state.estimated_latency_us(),
            resolved: true,
            description: "Cached template walk: entity KNN only",
        },
        WalkTier::DynamicWalk => TierResult {
            tier: WalkTier::DynamicWalk,
            latency_us: state.estimated_latency_us(),
            resolved: true,
            description: "Dynamic graph walk: full routing table lookup",
        },
        WalkTier::MarkovFallback => TierResult {
            tier: WalkTier::MarkovFallback,
            latency_us: state.estimated_latency_us(),
            resolved: false,
            description: "Markov RS fallback: full RS forward pass",
        },
    }
}

/// Tier distribution statistics for a set of queries.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct TierDistribution {
    pub tier_a_count: usize,
    pub tier_b_count: usize,
    pub tier_c_count: usize,
    pub total: usize,
    pub mean_latency_us: f64,
}

impl TierDistribution {
    /// Compute tier distribution from a set of walk states.
    pub fn from_states(states: &[WalkState]) -> Self {
        let mut dist = Self {
            total: states.len(),
            ..Default::default()
        };
        let mut total_latency = 0.0;

        for state in states {
            match state.tier {
                WalkTier::CachedTemplate => dist.tier_a_count += 1,
                WalkTier::DynamicWalk => dist.tier_b_count += 1,
                WalkTier::MarkovFallback => dist.tier_c_count += 1,
            }
            total_latency += state.estimated_latency_us();
        }

        if dist.total > 0 {
            dist.mean_latency_us = total_latency / dist.total as f64;
        }
        dist
    }

    pub fn tier_a_pct(&self) -> f64 {
        if self.total == 0 { 0.0 } else { self.tier_a_count as f64 / self.total as f64 * 100.0 }
    }

    pub fn tier_b_pct(&self) -> f64 {
        if self.total == 0 { 0.0 } else { self.tier_b_count as f64 / self.total as f64 * 100.0 }
    }

    pub fn tier_c_pct(&self) -> f64 {
        if self.total == 0 { 0.0 } else { self.tier_c_count as f64 / self.total as f64 * 100.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::walk_state::WalkMode;

    #[test]
    fn test_tier_routing() {
        let factual = WalkState {
            last_entity: Some("France".to_string()),
            current_relation: Some("capital-of".to_string()),
            mode: WalkMode::Factual,
            tier: WalkTier::CachedTemplate,
        };
        let result = route_to_tier(&factual);
        assert!(result.resolved);
        assert!(result.latency_us < 1000.0);

        let fallback = WalkState {
            last_entity: None,
            current_relation: None,
            mode: WalkMode::Conversation,
            tier: WalkTier::MarkovFallback,
        };
        let result = route_to_tier(&fallback);
        assert!(!result.resolved);
        assert!(result.latency_us > 100_000.0);
    }

    #[test]
    fn test_tier_distribution() {
        let states = vec![
            WalkState::from_tokens(&["capital", "of", "France"]),
            WalkState::from_tokens(&["capital", "of", "Germany"]),
            WalkState::from_tokens(&["tell", "me", "a", "joke"]),
        ];
        let dist = TierDistribution::from_states(&states);
        assert_eq!(dist.tier_a_count, 2); // Two factual queries
        assert_eq!(dist.tier_c_count, 1); // One fallback
        assert_eq!(dist.total, 3);
    }
}
