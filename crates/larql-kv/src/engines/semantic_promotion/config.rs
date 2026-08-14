//! Engine configuration (spec §19).

use super::error::PromotionError;
use super::mode::PromotionMode;
use super::qualification::{QualificationPolicy, DEFAULT_TOLERANCE_BITS};
use super::scope::RetirementScopePolicy;

/// Cap on how many tokens one promoted record may occupy. A record that
/// needs more than this is not compact enough to be replacing anything.
pub const DEFAULT_MAX_PROMOTED_TOKENS: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticPromotionConfig {
    pub mode: PromotionMode,
    /// Which retirement scopes this engine will issue at all.
    pub default_scope: RetirementScopePolicy,
    pub max_promoted_tokens: usize,
    /// Keep a hidden source's KV resident until the operation's answer
    /// boundary, so rollback stays exact.
    pub keep_hidden_source_until_termination: bool,
    /// Refuse physical reclamation unless a durable replay route exists
    /// (invariant I7).
    pub require_durable_replay_before_reclaim: bool,
    pub qualification_policy: QualificationPolicy,
    /// Per-position KL ceiling used when scoring, in bits.
    pub kl_tolerance_bits: f32,
    pub emit_phase_events: bool,
    pub collect_peak_position: bool,
}

impl Default for SemanticPromotionConfig {
    /// Spec §19's suggested defaults: observe-only, answer-scoped,
    /// exact certificates, durable replay required before any reclaim.
    fn default() -> Self {
        Self {
            mode: PromotionMode::Observe,
            default_scope: RetirementScopePolicy::AnswerScopedOnly,
            max_promoted_tokens: DEFAULT_MAX_PROMOTED_TOKENS,
            keep_hidden_source_until_termination: true,
            require_durable_replay_before_reclaim: true,
            qualification_policy: QualificationPolicy::ExactCertificate,
            kl_tolerance_bits: DEFAULT_TOLERANCE_BITS,
            emit_phase_events: true,
            collect_peak_position: true,
        }
    }
}

impl SemanticPromotionConfig {
    /// Builder shorthand used by tests and the CLI parser.
    pub fn with_mode(mut self, mode: PromotionMode) -> Self {
        self.mode = mode;
        self
    }

    /// Internally consistent? Checked at construction.
    pub fn validate(&self) -> Result<(), PromotionError> {
        if self.max_promoted_tokens == 0 {
            return Err(PromotionError::InvariantViolation(
                "max_promoted_tokens must be non-zero",
            ));
        }
        if !self.kl_tolerance_bits.is_finite() || self.kl_tolerance_bits < 0.0 {
            return Err(PromotionError::InvariantViolation(
                "kl_tolerance_bits must be finite and non-negative",
            ));
        }
        // Reclaiming pages with the durable-replay requirement switched
        // off is a request to destroy the only copy of source authority.
        if self.mode.reclaims_pages() && !self.require_durable_replay_before_reclaim {
            return Err(PromotionError::InvariantViolation(
                "cold-replay mode requires require_durable_replay_before_reclaim",
            ));
        }
        Ok(())
    }

    /// Short config string for [`EngineInfo`](crate::EngineInfo).
    pub fn info_string(&self) -> String {
        format!(
            "mode={},scope={},tol={:.3}b,max_promoted={}",
            self.mode.as_str(),
            match self.default_scope {
                RetirementScopePolicy::AnswerScopedOnly => "answer",
                RetirementScopePolicy::AllowGeneralState => "general",
            },
            self.kl_tolerance_bits,
            self.max_promoted_tokens,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_conservative_ones() {
        let c = SemanticPromotionConfig::default();
        assert_eq!(c.mode, PromotionMode::Observe);
        assert_eq!(c.default_scope, RetirementScopePolicy::AnswerScopedOnly);
        assert_eq!(
            c.qualification_policy,
            QualificationPolicy::ExactCertificate
        );
        assert!(c.keep_hidden_source_until_termination);
        assert!(c.require_durable_replay_before_reclaim);
        assert_eq!(c.max_promoted_tokens, DEFAULT_MAX_PROMOTED_TOKENS);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn cold_replay_may_not_switch_off_the_durable_replay_requirement() {
        let c = SemanticPromotionConfig::default().with_mode(PromotionMode::EnforceColdReplay);
        assert!(c.validate().is_ok());
        let c = SemanticPromotionConfig {
            require_durable_replay_before_reclaim: false,
            ..c
        };
        assert!(matches!(
            c.validate(),
            Err(PromotionError::InvariantViolation(_))
        ));
    }

    #[test]
    fn non_reclaiming_modes_may_relax_the_replay_requirement() {
        let c = SemanticPromotionConfig {
            require_durable_replay_before_reclaim: false,
            ..SemanticPromotionConfig::default().with_mode(PromotionMode::EnforceResidentRollback)
        };
        assert!(c.validate().is_ok());
    }

    #[test]
    fn degenerate_numeric_settings_are_rejected() {
        for c in [
            SemanticPromotionConfig {
                max_promoted_tokens: 0,
                ..Default::default()
            },
            SemanticPromotionConfig {
                kl_tolerance_bits: -1.0,
                ..Default::default()
            },
            SemanticPromotionConfig {
                kl_tolerance_bits: f32::NAN,
                ..Default::default()
            },
        ] {
            assert!(c.validate().is_err(), "{c:?} should be rejected");
        }
    }

    #[test]
    fn info_string_names_mode_and_scope_policy() {
        let s = SemanticPromotionConfig::default().info_string();
        assert!(s.contains("mode=observe"), "{s}");
        assert!(s.contains("scope=answer"), "{s}");

        let c = SemanticPromotionConfig {
            default_scope: RetirementScopePolicy::AllowGeneralState,
            ..Default::default()
        };
        assert!(c.info_string().contains("scope=general"));
    }
}
