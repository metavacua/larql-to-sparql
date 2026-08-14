//! Reachable-trajectory scoring (spec §9 metrics, invariant I10).
//!
//! A promotion is judged by teacher-forcing the *reference* continuation
//! through the candidate branch and comparing distributions position by
//! position. Two things about that comparison are load-bearing.
//!
//! **What gets scored (I10).** The scored object ends at the first
//! emitted termination token — payload tokens plus that one position.
//! Repeated termination probes past it are recorded but excluded, because
//! forcing a model to re-emit EOT eight times measures the harness, not
//! the replacement.
//!
//! **Why the boundary matters.** In EXP-25 the canonical record's entire
//! residual divergence landed on exactly that first-termination position:
//! 0.147 bits on `transform_inc` and 0.061 on `transform_rev` against a
//! 0.05 tolerance, while every payload position stayed under 0.023. So
//! excluding repeated probes does *not* rescue those rows — the peak is
//! inside the reachable window, not past it. A record can be sound for a
//! payload and unsound for the decision to stop. [`ScoredObject`] keeps
//! those two claims apart so a scope can never quietly inherit the wrong
//! one.

use super::operation::AnswerBoundary;

/// Default per-position KL ceiling, in bits. Matches the tolerance the
/// EXP-25 matrix was scored at.
pub const DEFAULT_KL_TOLERANCE_BITS: f32 = 0.05;

/// Which object a set of metrics actually covers.
///
/// Ordered by strength: a certificate scored over the reachable
/// trajectory also covers a payload-only claim, never the reverse.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ScoredObject {
    /// Payload positions only. Says nothing about when the model stops.
    PayloadOnly { payload_tokens: u32 },
    /// Payload positions plus the first termination token.
    ReachableTrajectory { payload_tokens: u32 },
}

impl ScoredObject {
    pub fn payload_tokens(self) -> u32 {
        match self {
            Self::PayloadOnly { payload_tokens } | Self::ReachableTrajectory { payload_tokens } => {
                payload_tokens
            }
        }
    }

    /// Whether the first termination position was inside what was scored.
    pub fn covers_first_termination(self) -> bool {
        matches!(self, Self::ReachableTrajectory { .. })
    }

    /// Whether evidence over `self` licenses a scope ending at `boundary`.
    ///
    /// This is the congruence rule: the boundary names the tokens the
    /// replacement must survive, and the scored object names the tokens
    /// it was actually tested over. The first must fit inside the second.
    pub fn covers_boundary(self, boundary: AnswerBoundary) -> bool {
        match boundary {
            AnswerBoundary::ExactPayloadLength(n) => self.payload_tokens() >= n,
            AnswerBoundary::FirstTermination => self.covers_first_termination(),
            // Unbounded by construction — no finite scoring covers it.
            AnswerBoundary::ExternalCommit => false,
        }
    }
}

/// Where a divergence peak sits relative to the answer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DivergenceClass {
    /// Nothing anywhere in the trace breached tolerance or flipped top-1.
    None,
    /// Peak on a payload position — the answer itself is at risk.
    Payload,
    /// Peak on the first termination token — the answer is reproduced
    /// but the decision to stop is not.
    FirstTermination,
    /// Peak past the first termination — excluded from scoring by I10,
    /// reported so it is not mistaken for a clean run.
    PostTermination,
}

/// Metrics from one reference-vs-candidate comparison.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QualificationMetrics {
    pub payload_top1_equal: bool,
    pub payload_max_kl_bits: f32,
    pub first_termination_top1_equal: bool,
    pub first_termination_kl_bits: f32,
    /// Max KL over payload + first termination (the I10 window).
    pub reachable_max_kl_bits: f32,
    /// Argmax over the *full* trace, including excluded probes.
    pub peak_position: u32,
    pub divergence_class: DivergenceClass,
    /// Number of positions inside the I10 window.
    pub m_effective: u32,
}

impl QualificationMetrics {
    /// Whether these metrics qualify the given object at `tolerance`.
    pub fn qualifies(&self, object: ScoredObject, tolerance: f32) -> bool {
        match object {
            ScoredObject::PayloadOnly { .. } => {
                self.payload_top1_equal && self.payload_max_kl_bits <= tolerance
            }
            ScoredObject::ReachableTrajectory { .. } => {
                self.payload_top1_equal
                    && self.first_termination_top1_equal
                    && self.reachable_max_kl_bits <= tolerance
            }
        }
    }
}

/// One scored position of the forced comparison.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionScore {
    /// KL(reference ‖ candidate) at this position, in bits.
    pub kl_bits: f32,
    /// Whether both branches' argmax agree here.
    pub top1_equal: bool,
}

/// Reasons a trace cannot be scored at all.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ScoringError {
    #[error("scored trace is empty")]
    EmptyTrace,
    #[error("payload length exceeds the scored trace")]
    PayloadExceedsTrace,
    #[error("trace has no first-termination position")]
    NoTerminationPosition,
}

/// Score a forced comparison.
///
/// `positions` is the full per-position trace in emission order.
/// `payload_tokens` is the count of payload positions before the first
/// termination token, so position `payload_tokens` *is* that token.
pub fn score_trajectory(
    positions: &[PositionScore],
    payload_tokens: u32,
    tolerance: f32,
) -> Result<QualificationMetrics, ScoringError> {
    if positions.is_empty() {
        return Err(ScoringError::EmptyTrace);
    }
    let payload_len = payload_tokens as usize;
    if payload_len > positions.len() {
        return Err(ScoringError::PayloadExceedsTrace);
    }
    if payload_len == positions.len() {
        // Nothing follows the payload, so the model's termination
        // decision was never observed. Refuse rather than score a
        // trajectory claim over a payload-only trace.
        return Err(ScoringError::NoTerminationPosition);
    }

    let payload = &positions[..payload_len];
    let payload_top1_equal = payload.iter().all(|p| p.top1_equal);
    let payload_max_kl_bits = max_kl(payload);

    let termination = positions[payload_len];
    let m_effective = payload_len + 1;
    let reachable = &positions[..m_effective];
    let reachable_max_kl_bits = max_kl(reachable);

    // Peak is located over the FULL trace: excluded-from-scoring is not
    // the same as unobserved, and a peak sitting past termination is a
    // materially different report from a clean one.
    let peak_position = positions
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bk), (i, p)| {
            if p.kl_bits > bk {
                (i, p.kl_bits)
            } else {
                (bi, bk)
            }
        })
        .0;

    let clean = positions
        .iter()
        .all(|p| p.top1_equal && p.kl_bits <= tolerance);
    let divergence_class = if clean {
        DivergenceClass::None
    } else if peak_position < payload_len {
        DivergenceClass::Payload
    } else if peak_position == payload_len {
        DivergenceClass::FirstTermination
    } else {
        DivergenceClass::PostTermination
    };

    Ok(QualificationMetrics {
        payload_top1_equal,
        payload_max_kl_bits,
        first_termination_top1_equal: termination.top1_equal,
        first_termination_kl_bits: termination.kl_bits,
        reachable_max_kl_bits,
        peak_position: peak_position as u32,
        divergence_class,
        m_effective: m_effective as u32,
    })
}

fn max_kl(positions: &[PositionScore]) -> f32 {
    positions
        .iter()
        .map(|p| p.kl_bits)
        .fold(0.0f32, |a, b| if b > a { b } else { a })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = DEFAULT_KL_TOLERANCE_BITS;

    /// Build a trace from raw KL bits; top-1 agrees wherever KL is small.
    /// The threshold used here is only a fixture convenience — real
    /// callers supply measured top-1 equality.
    fn trace(kls: &[f32]) -> Vec<PositionScore> {
        kls.iter()
            .map(|&kl_bits| PositionScore {
                kl_bits,
                top1_equal: kl_bits < 1.0,
            })
            .collect()
    }

    /// EXP-25 `copy` / canonical: peak 0.0441 on the first termination
    /// token, everything else clean. Qualifies on both objects.
    #[test]
    fn exp25_copy_canonical_qualifies_on_the_reachable_trajectory() {
        let m = score_trajectory(
            &trace(&[0.0001, 0.0, 0.0, 0.0, 0.0441, 0.0, 0.0, 0.0004]),
            4,
            TOL,
        )
        .unwrap();
        assert_eq!(m.m_effective, 5);
        assert_eq!(m.peak_position, 4);
        assert_eq!(m.divergence_class, DivergenceClass::None);
        assert!(m.qualifies(ScoredObject::ReachableTrajectory { payload_tokens: 4 }, TOL));
        assert!(m.qualifies(ScoredObject::PayloadOnly { payload_tokens: 4 }, TOL));
    }

    /// EXP-25 `transform_inc` / canonical: payload immaculate, 0.1472 on
    /// the first termination token. Payload-only qualifies; the reachable
    /// trajectory does not. Excluding repeated probes cannot rescue it,
    /// because the peak is inside the window.
    #[test]
    fn exp25_inc_canonical_qualifies_on_payload_but_not_trajectory() {
        let m = score_trajectory(
            &trace(&[
                0.0008, 0.0003, 0.0001, 0.001, 0.1472, 0.0044, 0.0001, 0.0011,
            ]),
            4,
            TOL,
        )
        .unwrap();
        assert_eq!(m.divergence_class, DivergenceClass::FirstTermination);
        assert!((m.first_termination_kl_bits - 0.1472).abs() < 1e-6);
        assert!(m.qualifies(ScoredObject::PayloadOnly { payload_tokens: 4 }, TOL));
        assert!(!m.qualifies(ScoredObject::ReachableTrajectory { payload_tokens: 4 }, TOL));
    }

    /// EXP-25 `transform_inc` / derived: peak at position 9, past
    /// termination. Scored clean, but the peak location is still
    /// reported as post-termination rather than hidden.
    #[test]
    fn exp25_inc_derived_peaks_past_termination_and_still_qualifies() {
        let m = score_trajectory(
            &trace(&[
                0.0001, 0.0, 0.0, 0.0008, 0.0003, 0.0117, 0.0001, 0.0012, 0.0062, 0.0218,
            ]),
            4,
            TOL,
        )
        .unwrap();
        assert_eq!(m.peak_position, 9);
        assert_eq!(m.divergence_class, DivergenceClass::None);
        assert!(m.reachable_max_kl_bits <= TOL);
        assert!(m.qualifies(ScoredObject::ReachableTrajectory { payload_tokens: 4 }, TOL));
    }

    /// EXP-25 `pad_only` control: divergence on a payload digit.
    #[test]
    fn exp25_pad_only_control_diverges_on_payload() {
        let m = score_trajectory(
            &trace(&[0.3036, 3.2582, 7.6657, 9.5763, 0.0151, 0.0005]),
            4,
            TOL,
        )
        .unwrap();
        assert_eq!(m.divergence_class, DivergenceClass::Payload);
        assert!(!m.payload_top1_equal);
        assert!(!m.qualifies(ScoredObject::PayloadOnly { payload_tokens: 4 }, TOL));
    }

    #[test]
    fn reachable_window_excludes_repeated_termination_probes() {
        // A huge post-termination probe must not enter the reachable max.
        let m = score_trajectory(&trace(&[0.0, 0.0, 0.01, 99.0]), 2, TOL).unwrap();
        assert_eq!(m.m_effective, 3);
        assert!((m.reachable_max_kl_bits - 0.01).abs() < 1e-6);
        assert_eq!(m.divergence_class, DivergenceClass::PostTermination);
    }

    #[test]
    fn trajectory_object_covers_payload_object_but_not_the_reverse() {
        let traj = ScoredObject::ReachableTrajectory { payload_tokens: 4 };
        let payload = ScoredObject::PayloadOnly { payload_tokens: 4 };
        assert!(traj.covers_boundary(AnswerBoundary::FirstTermination));
        assert!(!payload.covers_boundary(AnswerBoundary::FirstTermination));
        assert!(traj.covers_boundary(AnswerBoundary::ExactPayloadLength(4)));
        assert!(payload.covers_boundary(AnswerBoundary::ExactPayloadLength(4)));
    }

    #[test]
    fn a_longer_payload_boundary_is_not_covered_by_shorter_evidence() {
        let payload = ScoredObject::PayloadOnly { payload_tokens: 4 };
        assert!(!payload.covers_boundary(AnswerBoundary::ExactPayloadLength(5)));
    }

    #[test]
    fn external_commit_is_covered_by_no_finite_scoring() {
        for object in [
            ScoredObject::PayloadOnly {
                payload_tokens: 999,
            },
            ScoredObject::ReachableTrajectory {
                payload_tokens: 999,
            },
        ] {
            assert!(!object.covers_boundary(AnswerBoundary::ExternalCommit));
        }
    }

    #[test]
    fn scoring_rejects_traces_it_cannot_read() {
        assert_eq!(
            score_trajectory(&[], 0, TOL).unwrap_err(),
            ScoringError::EmptyTrace
        );
        assert_eq!(
            score_trajectory(&trace(&[0.0]), 4, TOL).unwrap_err(),
            ScoringError::PayloadExceedsTrace
        );
        assert_eq!(
            score_trajectory(&trace(&[0.0, 0.0]), 2, TOL).unwrap_err(),
            ScoringError::NoTerminationPosition
        );
    }

    #[test]
    fn payload_tokens_accessor_reads_both_variants() {
        assert_eq!(
            ScoredObject::PayloadOnly { payload_tokens: 3 }.payload_tokens(),
            3
        );
        assert_eq!(
            ScoredObject::ReachableTrajectory { payload_tokens: 7 }.payload_tokens(),
            7
        );
        assert!(!ScoredObject::PayloadOnly { payload_tokens: 3 }.covers_first_termination());
    }

    #[test]
    fn a_top1_flip_at_termination_fails_the_trajectory_within_tolerance() {
        // Small KL but a flipped argmax still fails: top-1 equality is a
        // separate condition, not implied by the bits.
        let positions = vec![
            PositionScore {
                kl_bits: 0.0,
                top1_equal: true,
            },
            PositionScore {
                kl_bits: 0.001,
                top1_equal: false,
            },
        ];
        let m = score_trajectory(&positions, 1, TOL).unwrap();
        assert!(!m.first_termination_top1_equal);
        assert!(!m.qualifies(ScoredObject::ReachableTrajectory { payload_tokens: 1 }, TOL));
        assert!(m.qualifies(ScoredObject::PayloadOnly { payload_tokens: 1 }, TOL));
    }
}
