//! How a semantic record becomes something the model can attend to
//! (spec §7, "Materialisation").
//!
//! The semantic record and its materialisation are separate objects: the
//! record survives a change of encoding, and a certificate does not.
//! [`MaterialisationKind`] therefore names the full space — including
//! forms this version cannot build — so that a certificate minted for a
//! token rendering can never be read as covering a residual injection.
//!
//! Deviation from spec §7: v1 gives [`RecordMaterialisation`] the
//! `TokenSequence` variant only, rather than four variants of which three
//! are unconstructible. The other three are reachable through
//! [`MaterialisationPolicy`], which fails closed with
//! [`PromotionError::UnsupportedCapability`].

use super::error::PromotionError;
use super::ids::CheckpointId;

/// Encoding version of the v1 token rendering. Bump when the record →
/// token mapping changes in a way that invalidates existing certificates.
pub const TOKEN_SEQUENCE_ENCODING_VERSION: u32 = 1;

/// Every materialisation form the capability key can distinguish.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MaterialisationKind {
    /// Record rendered to tokens and appended through normal decode.
    TokenSequence,
    /// Record restored from a stored hidden-state checkpoint.
    HiddenCheckpoint,
    /// Record injected as a residual delta at one layer.
    ResidualDelta,
    /// Record held in a dedicated memory slot.
    RecordMemory,
}

impl MaterialisationKind {
    /// Whether this engine version can build the form.
    pub fn is_implemented(self) -> bool {
        matches!(self, Self::TokenSequence)
    }

    /// Stable name used in error messages and traces.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TokenSequence => "token_sequence",
            Self::HiddenCheckpoint => "hidden_checkpoint",
            Self::ResidualDelta => "residual_delta",
            Self::RecordMemory => "record_memory",
        }
    }
}

/// A built materialisation. v1: token renderings only.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RecordMaterialisation {
    TokenSequence {
        tokens: Vec<u32>,
        encoding_version: u32,
    },
}

impl RecordMaterialisation {
    /// Build a v1 token rendering.
    pub fn token_sequence(tokens: Vec<u32>) -> Self {
        Self::TokenSequence {
            tokens,
            encoding_version: TOKEN_SEQUENCE_ENCODING_VERSION,
        }
    }

    pub fn kind(&self) -> MaterialisationKind {
        match self {
            Self::TokenSequence { .. } => MaterialisationKind::TokenSequence,
        }
    }

    pub fn encoding_version(&self) -> u32 {
        match self {
            Self::TokenSequence {
                encoding_version, ..
            } => *encoding_version,
        }
    }

    /// Positions this materialisation will occupy once appended.
    pub fn token_count(&self) -> usize {
        match self {
            Self::TokenSequence { tokens, .. } => tokens.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.token_count() == 0
    }

    /// Digest of the **exact tokens the model sees**.
    ///
    /// Distinct from a record's semantic digest, and both are needed. Two
    /// records asserting the same fact in different words are the same
    /// claim and a different stimulus; EXP-25CF found consumption mode
    /// turning on the record's exact content, so a certificate keyed only
    /// on semantics would merge two materialisations whose behaviour may
    /// differ. The encoding version is folded in because a re-rendering of
    /// the same tokens under new rules is a different stimulus too.
    pub fn content_digest(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(b"larql.semantic_promotion.materialisation.v1");
        h.update(self.kind().as_str().as_bytes());
        h.update(self.encoding_version().to_le_bytes());
        match self {
            Self::TokenSequence { tokens, .. } => {
                h.update((tokens.len() as u64).to_le_bytes());
                for t in tokens {
                    h.update(t.to_le_bytes());
                }
            }
        }
        h.finalize().into()
    }
}

/// What the caller asked for, before the engine decided it could build it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MaterialisationPolicy {
    /// Render the record to these tokens.
    TokenSequence { tokens: Vec<u32> },
    /// Restore from a checkpoint. Not implemented in v1.
    HiddenCheckpoint { checkpoint: CheckpointId },
    /// Inject at a layer. Not implemented in v1.
    ResidualDelta { layer: u32 },
    /// Use a record-memory slot. Not implemented in v1.
    RecordMemory { slot: u64 },
}

impl MaterialisationPolicy {
    pub fn kind(&self) -> MaterialisationKind {
        match self {
            Self::TokenSequence { .. } => MaterialisationKind::TokenSequence,
            Self::HiddenCheckpoint { .. } => MaterialisationKind::HiddenCheckpoint,
            Self::ResidualDelta { .. } => MaterialisationKind::ResidualDelta,
            Self::RecordMemory { .. } => MaterialisationKind::RecordMemory,
        }
    }

    /// Resolve to a built materialisation, or refuse.
    ///
    /// Refusal is the point: a policy this version cannot honour must
    /// not silently fall back to a token rendering, because the
    /// certificate keyed on the requested kind would not cover it.
    pub fn resolve(self) -> Result<RecordMaterialisation, PromotionError> {
        match self {
            Self::TokenSequence { tokens } if tokens.is_empty() => {
                Err(PromotionError::MaterialisationFailed)
            }
            Self::TokenSequence { tokens } => Ok(RecordMaterialisation::token_sequence(tokens)),
            other => Err(PromotionError::UnsupportedCapability(other.kind().as_str())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_token_rendering_is_implemented_in_v1() {
        assert!(MaterialisationKind::TokenSequence.is_implemented());
        for kind in [
            MaterialisationKind::HiddenCheckpoint,
            MaterialisationKind::ResidualDelta,
            MaterialisationKind::RecordMemory,
        ] {
            assert!(!kind.is_implemented(), "{kind:?}");
        }
    }

    #[test]
    fn unimplemented_policies_refuse_rather_than_downgrade() {
        for policy in [
            MaterialisationPolicy::HiddenCheckpoint {
                checkpoint: CheckpointId::from_counter(1),
            },
            MaterialisationPolicy::ResidualDelta { layer: 25 },
            MaterialisationPolicy::RecordMemory { slot: 3 },
        ] {
            let kind = policy.kind();
            match policy.resolve() {
                Err(PromotionError::UnsupportedCapability(name)) => {
                    assert_eq!(name, kind.as_str())
                }
                other => panic!("{kind:?} should refuse, got {other:?}"),
            }
        }
    }

    #[test]
    fn an_empty_token_rendering_is_a_materialisation_failure() {
        let policy = MaterialisationPolicy::TokenSequence { tokens: vec![] };
        assert_eq!(
            policy.resolve().unwrap_err(),
            PromotionError::MaterialisationFailed
        );
    }

    #[test]
    fn token_policy_resolves_and_stamps_the_encoding_version() {
        let m = MaterialisationPolicy::TokenSequence {
            tokens: vec![1, 2, 3],
        }
        .resolve()
        .unwrap();
        assert_eq!(m.kind(), MaterialisationKind::TokenSequence);
        assert_eq!(m.encoding_version(), TOKEN_SEQUENCE_ENCODING_VERSION);
        assert_eq!(m.token_count(), 3);
        assert!(!m.is_empty());
    }

    /// **Same claim, different wording is a different stimulus.**
    ///
    /// A semantic-only key would merge these, and EXP-25CF found
    /// consumption mode turning on a record's exact content — so the
    /// certificate has to see the tokens, not just what they mean.
    #[test]
    fn differently_worded_renderings_digest_differently() {
        let a = RecordMaterialisation::token_sequence(vec![10, 11, 12]);
        let b = RecordMaterialisation::token_sequence(vec![10, 99, 12]);
        assert_ne!(a.content_digest(), b.content_digest());
        // Length alone must separate too, not only content.
        let c = RecordMaterialisation::token_sequence(vec![10, 11, 12, 12]);
        assert_ne!(a.content_digest(), c.content_digest());
    }

    #[test]
    fn an_identical_rendering_digests_identically() {
        let a = RecordMaterialisation::token_sequence(vec![7, 8, 9]);
        let b = RecordMaterialisation::token_sequence(vec![7, 8, 9]);
        assert_eq!(a.content_digest(), b.content_digest());
    }

    #[test]
    fn a_re_encoding_of_the_same_tokens_is_a_different_stimulus() {
        let a = RecordMaterialisation::token_sequence(vec![1, 2, 3]);
        let b = RecordMaterialisation::TokenSequence {
            tokens: vec![1, 2, 3],
            encoding_version: TOKEN_SEQUENCE_ENCODING_VERSION + 1,
        };
        assert_ne!(
            a.content_digest(),
            b.content_digest(),
            "an encoding change must invalidate the certificate"
        );
    }

    #[test]
    fn kind_names_are_stable_and_distinct() {
        let names: Vec<_> = [
            MaterialisationKind::TokenSequence,
            MaterialisationKind::HiddenCheckpoint,
            MaterialisationKind::ResidualDelta,
            MaterialisationKind::RecordMemory,
        ]
        .iter()
        .map(|k| k.as_str())
        .collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len());
    }

    #[test]
    fn policy_kind_matches_the_resolved_materialisation_kind() {
        let policy = MaterialisationPolicy::TokenSequence { tokens: vec![7] };
        let kind = policy.kind();
        assert_eq!(policy.resolve().unwrap().kind(), kind);
    }
}
