//! Per-layer positional-encoding policy.
//!
//! Absence of positional rotation is an **intentional execution property**,
//! not a parameter value. Muse-Glimmer's global layers carry no position
//! encoding at all ("RoPE, local layers only" per the release card), and the
//! checkpoint spells that as `layer_rope_theta[i] == 0` — a sentinel that is
//! only meaningful at the parse boundary. Internally a zero theta must never
//! circulate: `1/0^(i/d)` is degenerate, and a resolver that stores `0.0`
//! where it means "none" has re-invented the magic value this type exists to
//! remove.
//!
//! The sentinel is honoured exactly once, in
//! [`PositionPolicy::from_declared_theta`]; everything downstream matches on
//! the variant.

use serde::{Deserialize, Serialize};

/// The HF `layer_rope_theta` sentinel for "no positional encoding on this
/// layer". Consumed at the parse boundary only.
const NOPE_THETA_SENTINEL: f64 = 0.0;

/// How a layer encodes position.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PositionPolicy {
    /// Rotary position embedding at the given base frequency.
    Rope { theta: f64 },
    /// No positional encoding — the layer attends position-agnostically.
    None,
}

impl PositionPolicy {
    /// Interpret one declared per-layer theta, honouring the upstream
    /// zero-as-NoPE sentinel at this boundary and nowhere else.
    pub fn from_declared_theta(theta: f64) -> Self {
        if theta == NOPE_THETA_SENTINEL {
            Self::None
        } else {
            Self::Rope { theta }
        }
    }

    /// The rope base when the policy is rotary; `None` for a NoPE layer.
    /// Callers that need a theta must handle absence — there is no default.
    pub fn rope_theta(self) -> Option<f64> {
        match self {
            Self::Rope { theta } => Some(theta),
            Self::None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_the_nope_sentinel_at_the_boundary() {
        assert_eq!(
            PositionPolicy::from_declared_theta(0.0),
            PositionPolicy::None
        );
        assert_eq!(
            PositionPolicy::from_declared_theta(500000.0),
            PositionPolicy::Rope { theta: 500000.0 }
        );
    }

    #[test]
    fn nope_layers_have_no_theta_to_offer() {
        assert_eq!(PositionPolicy::None.rope_theta(), None);
        assert_eq!(
            PositionPolicy::Rope { theta: 10000.0 }.rope_theta(),
            Some(10000.0)
        );
    }

    #[test]
    fn serialises_tagged() {
        assert_eq!(
            serde_json::to_string(&PositionPolicy::None).unwrap(),
            "{\"kind\":\"none\"}"
        );
        assert_eq!(
            serde_json::to_string(&PositionPolicy::Rope { theta: 500000.0 }).unwrap(),
            "{\"kind\":\"rope\",\"theta\":500000.0}"
        );
    }

    #[test]
    fn round_trips() {
        for policy in [PositionPolicy::None, PositionPolicy::Rope { theta: 1e6 }] {
            let json = serde_json::to_string(&policy).unwrap();
            let back: PositionPolicy = serde_json::from_str(&json).unwrap();
            assert_eq!(back, policy);
        }
    }
}
