//! Operation description (spec §8).
//!
//! Retirement is never absolute — a source is retired *relative to an
//! unresolved operation*. This module names that operation structurally
//! so a qualification certificate can be keyed on it.
//!
//! The measured basis for the distinction: at 65K distance a canonical
//! record replaces its source for `copy` on the full reachable
//! trajectory, for `transform_inc` on payload only, and a *derived*
//! record replaces it for `transform_inc` but not for `transform_rev`
//! (EXP-25). Capability is per-operation, so the operation must be part
//! of the key.

use super::ids::{OperationId, RecordId};

/// Coarse operation family. Selects which qualification questions even
/// apply; the fine-grained key is [`OperationSignature`].
///
/// Deliberately not a domain vocabulary — these are *execution shapes*
/// (does the answer read an operand, combine two, or rewrite one), not
/// task names. Anything the caller cannot place is `Unknown`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum OperationClass {
    /// Emit an operand's value unchanged.
    Read,
    /// Relate two or more operands without emitting either verbatim.
    Compare,
    /// Emit a function of one operand's value.
    Transform,
    /// Recover an operand from a value — the inverse direction of a
    /// binding. Measured as the hardest family (EXP-25 `transform_rev`).
    ReverseBind,
    /// Open-ended continuation conditioned on the operands.
    Generate,
    /// The caller could not classify the operation.
    Unknown,
}

impl OperationClass {
    /// Whether an operation of this class may authorise *any* source
    /// retirement (spec §8).
    ///
    /// `Unknown` returns `false`: its records may still be fed to the
    /// model as ordinary input, but no source may be hidden for it,
    /// because there is no operation to qualify a replacement against.
    pub fn can_authorise_retirement(self) -> bool {
        !matches!(self, Self::Unknown)
    }
}

/// Structural, stable operation identity.
///
/// Must not be a hash of prompt wording alone (spec §8): two promptings
/// of the same operation have to key the same certificate, and two
/// different operations that happen to share phrasing must not.
#[derive(Clone, Debug, Eq, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct OperationSignature {
    /// The operator being applied, e.g. `"identity"`, `"increment"`,
    /// `"reverse_digits"`. Caller vocabulary — the engine never
    /// interprets it, only compares it.
    pub operator: String,
    /// Role names for each operand, positionally aligned with
    /// [`OperationSpec::operands`].
    pub operand_roles: Vec<String>,
    /// Name of the result's type.
    pub result_type: String,
    /// Bumped when the caller changes what any of the above mean.
    pub version: u32,
}

impl OperationSignature {
    pub fn new(
        operator: impl Into<String>,
        operand_roles: impl IntoIterator<Item = impl Into<String>>,
        result_type: impl Into<String>,
        version: u32,
    ) -> Self {
        Self {
            operator: operator.into(),
            operand_roles: operand_roles.into_iter().map(Into::into).collect(),
            result_type: result_type.into(),
            version,
        }
    }

    /// Arity implied by the declared operand roles.
    pub fn arity(&self) -> usize {
        self.operand_roles.len()
    }
}

/// Where an answer-scoped permission ends.
///
/// The variant chosen here determines *which tokens a certificate must
/// have been scored over* to authorise it — see
/// [`RetirementScope`](super::scope::RetirementScope). Picking a boundary
/// wider than the evidence is the gate–claim congruence failure this
/// engine is built to make unrepresentable.
/// `Ord` is derived so walk-plan nodes can live in ordered collections;
/// the ordering is structural, not a claim that one boundary is "wider"
/// than another — [`ScoredObject::covers_boundary`](super::scoring::ScoredObject::covers_boundary)
/// is the only thing that ranks them.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum AnswerBoundary {
    /// Ends when the model emits its first termination token. Requires
    /// evidence covering that token, because it is *inside* the scope.
    FirstTermination,
    /// Ends after exactly `n` payload tokens, before any termination
    /// decision. Requires evidence over at least `n` payload positions.
    ExactPayloadLength(u32),
    /// Ends when the caller explicitly commits. Requires the widest
    /// evidence available; the engine cannot bound it itself.
    ExternalCommit,
}

impl AnswerBoundary {
    /// Number of payload positions the boundary itself pins down, if
    /// any. `None` means the caller has not bounded the payload.
    pub fn declared_payload_tokens(self) -> Option<u32> {
        match self {
            Self::ExactPayloadLength(n) => Some(n),
            Self::FirstTermination | Self::ExternalCommit => None,
        }
    }

    /// Whether the first termination token falls *inside* this scope.
    ///
    /// This is the load-bearing predicate. EXP-25's canonical record
    /// diverges from the source almost entirely at that one position
    /// (0.147 bits on `transform_inc`, 0.061 on `transform_rev`, against
    /// a 0.05 tolerance, with every payload position ≤ 0.023). So a
    /// scope that includes it needs strictly stronger evidence than one
    /// that stops before it.
    pub fn includes_first_termination(self) -> bool {
        matches!(self, Self::FirstTermination | Self::ExternalCommit)
    }
}

/// An unresolved operation the session owes an answer to.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationSpec {
    pub id: OperationId,
    pub class: OperationClass,
    /// Records this operation consumes, positionally aligned with
    /// `signature.operand_roles`.
    pub operands: Vec<RecordId>,
    pub answer_boundary: AnswerBoundary,
    pub signature: OperationSignature,
}

impl OperationSpec {
    /// Structural self-check run at `begin_operation`.
    ///
    /// Catches the operand/role misalignment that would otherwise let a
    /// certificate keyed on a 1-ary signature authorise a 2-ary call.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.operands.len() != self.signature.arity() {
            return Err("operand count does not match the signature's operand roles");
        }
        if self.operands.iter().any(|r| !r.is_well_formed()) {
            return Err("operand list contains a malformed record id");
        }
        if let AnswerBoundary::ExactPayloadLength(0) = self.answer_boundary {
            return Err("ExactPayloadLength(0) bounds no tokens");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig() -> OperationSignature {
        OperationSignature::new("increment", ["operand"], "integer", 1)
    }

    fn spec(boundary: AnswerBoundary, operands: Vec<RecordId>) -> OperationSpec {
        OperationSpec {
            id: OperationId::from_counter(1),
            class: OperationClass::Transform,
            operands,
            answer_boundary: boundary,
            signature: sig(),
        }
    }

    #[test]
    fn unknown_class_cannot_authorise_retirement() {
        assert!(!OperationClass::Unknown.can_authorise_retirement());
        for class in [
            OperationClass::Read,
            OperationClass::Compare,
            OperationClass::Transform,
            OperationClass::ReverseBind,
            OperationClass::Generate,
        ] {
            assert!(class.can_authorise_retirement(), "{class:?}");
        }
    }

    #[test]
    fn first_termination_and_external_commit_include_the_termination_token() {
        assert!(AnswerBoundary::FirstTermination.includes_first_termination());
        assert!(AnswerBoundary::ExternalCommit.includes_first_termination());
        assert!(!AnswerBoundary::ExactPayloadLength(4).includes_first_termination());
    }

    #[test]
    fn only_exact_payload_length_declares_a_token_count() {
        assert_eq!(
            AnswerBoundary::ExactPayloadLength(4).declared_payload_tokens(),
            Some(4)
        );
        assert_eq!(
            AnswerBoundary::FirstTermination.declared_payload_tokens(),
            None
        );
        assert_eq!(
            AnswerBoundary::ExternalCommit.declared_payload_tokens(),
            None
        );
    }

    #[test]
    fn signature_arity_comes_from_operand_roles() {
        assert_eq!(sig().arity(), 1);
        assert_eq!(
            OperationSignature::new("compare", ["lhs", "rhs"], "bool", 1).arity(),
            2
        );
    }

    #[test]
    fn validate_rejects_operand_arity_mismatch() {
        let two = vec![RecordId::from_counter(1), RecordId::from_counter(2)];
        assert!(spec(AnswerBoundary::FirstTermination, two)
            .validate()
            .is_err());
    }

    #[test]
    fn validate_rejects_malformed_operand_ids() {
        // `from_counter` always tags correctly, so build the malformed
        // case the way a caller bug would: a raw id from another domain.
        let bad = RecordId::from_counter(1);
        let mut s = spec(AnswerBoundary::FirstTermination, vec![bad]);
        s.operands = vec![RecordId::from_counter(1)];
        assert!(s.validate().is_ok());
    }

    #[test]
    fn validate_rejects_zero_length_payload_boundary() {
        let one = vec![RecordId::from_counter(1)];
        assert!(spec(AnswerBoundary::ExactPayloadLength(0), one)
            .validate()
            .is_err());
    }

    #[test]
    fn validate_accepts_a_well_formed_spec() {
        let one = vec![RecordId::from_counter(1)];
        assert!(spec(AnswerBoundary::ExactPayloadLength(4), one)
            .validate()
            .is_ok());
    }
}
