//! Source authorities and stable references to them (spec §6).
//!
//! A source authority is a span of context that carries original truth.
//! References to it are *logical*: an [`AuthorityId`] plus a logical
//! token range, a session epoch and a content digest. Nothing here is a
//! physical KV offset, so compaction, eviction and cache growth cannot
//! invalidate a stored reference (invariant I5) — and a span whose
//! content changed underneath the reference fails the digest check
//! rather than silently resolving to different tokens.

use std::ops::Range;

use sha2::{Digest, Sha256};

use super::error::PromotionError;
use super::ids::{AuthorityId, CheckpointId};

/// Domain separator mixed into every content digest, so a digest over
/// context tokens can never collide with one taken over some other byte
/// string elsewhere in the system.
const DIGEST_DOMAIN: &[u8] = b"larql.semantic_promotion.source_authority.v1";

/// Digest of a token span's content.
pub fn content_digest(tokens: &[u32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(DIGEST_DOMAIN);
    hasher.update((tokens.len() as u64).to_le_bytes());
    for t in tokens {
        hasher.update(t.to_le_bytes());
    }
    hasher.finalize().into()
}

/// What the caller registers: a span of already-prefilled context.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceAuthority {
    /// Epoch of the session the span belongs to. Bumped on re-prefill.
    pub session_epoch: u64,
    /// Half-open logical token range within that session.
    pub logical_tokens: Range<u64>,
    pub content_digest: [u8; 32],
    /// Durable route back to this span's content. `None` means the only
    /// copy is the resident KV, so physical reclamation is refused
    /// (invariant I7).
    pub replay_checkpoint: Option<CheckpointId>,
}

impl SourceAuthority {
    /// Register a span from its tokens, digesting them.
    pub fn from_tokens(session_epoch: u64, start: u64, tokens: &[u32]) -> Self {
        Self {
            session_epoch,
            logical_tokens: start..start + tokens.len() as u64,
            content_digest: content_digest(tokens),
            replay_checkpoint: None,
        }
    }

    pub fn with_replay_checkpoint(mut self, checkpoint: CheckpointId) -> Self {
        self.replay_checkpoint = Some(checkpoint);
        self
    }

    pub fn token_count(&self) -> u64 {
        self.logical_tokens.end - self.logical_tokens.start
    }

    /// Whether this span may ever be physically reclaimed.
    pub fn has_durable_replay(&self) -> bool {
        self.replay_checkpoint.is_some()
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.logical_tokens.start >= self.logical_tokens.end {
            return Err("source authority span is empty");
        }
        Ok(())
    }
}

/// A stable handle to a registered authority.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceAuthorityRef {
    pub authority_id: AuthorityId,
    pub session_epoch: u64,
    pub logical_tokens: Range<u64>,
    pub content_digest: [u8; 32],
    pub replay_checkpoint: Option<CheckpointId>,
}

impl SourceAuthorityRef {
    pub fn new(authority_id: AuthorityId, authority: &SourceAuthority) -> Self {
        Self {
            authority_id,
            session_epoch: authority.session_epoch,
            logical_tokens: authority.logical_tokens.clone(),
            content_digest: authority.content_digest,
            replay_checkpoint: authority.replay_checkpoint,
        }
    }

    /// Confirm the reference still names what it was minted against.
    ///
    /// Epoch first, then digest: a stale epoch means the whole context
    /// was rebuilt, which is a different failure from the same context
    /// having been edited in place.
    pub fn check_resolves(
        &self,
        live_epoch: u64,
        live: &SourceAuthority,
    ) -> Result<(), PromotionError> {
        if self.session_epoch != live_epoch {
            return Err(PromotionError::StaleAuthorityEpoch);
        }
        if self.logical_tokens != live.logical_tokens || self.content_digest != live.content_digest
        {
            return Err(PromotionError::ContentDigestMismatch);
        }
        Ok(())
    }

    pub fn token_count(&self) -> u64 {
        self.logical_tokens.end - self.logical_tokens.start
    }

    /// Whether this span overlaps `other`. Two promotions may not hide
    /// overlapping spans concurrently — the second's rollback would
    /// restore rows the first still has hidden.
    pub fn overlaps(&self, other: &Range<u64>) -> bool {
        self.logical_tokens.start < other.end && other.start < self.logical_tokens.end
    }
}

/// Physical pages an authority currently occupies, resolved at execution
/// time from its logical reference.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedSourcePages {
    pub authority_id: AuthorityId,
    /// Physical row range in the live cache. Valid only until the next
    /// compaction — never store it.
    pub physical_rows: Range<usize>,
}

/// Resolves logical authority references to physical pages.
pub trait AuthorityResolver {
    fn resolve_source(
        &self,
        authority: &SourceAuthorityRef,
    ) -> Result<ResolvedSourcePages, PromotionError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn authority() -> SourceAuthority {
        SourceAuthority::from_tokens(1, 16_389, &[10, 11, 12, 13])
    }

    fn reference(a: &SourceAuthority) -> SourceAuthorityRef {
        SourceAuthorityRef::new(AuthorityId::from_counter(1), a)
    }

    #[test]
    fn digest_is_stable_and_content_sensitive() {
        assert_eq!(content_digest(&[1, 2, 3]), content_digest(&[1, 2, 3]));
        assert_ne!(content_digest(&[1, 2, 3]), content_digest(&[1, 2, 4]));
    }

    #[test]
    fn digest_distinguishes_length_from_content() {
        // Length is folded in explicitly so [1,2] and [1,2,0] differ
        // even where a naive byte concat could be made to collide.
        assert_ne!(content_digest(&[1, 2]), content_digest(&[1, 2, 0]));
    }

    #[test]
    fn span_derives_from_the_token_count() {
        let a = authority();
        assert_eq!(a.logical_tokens, 16_389..16_393);
        assert_eq!(a.token_count(), 4);
        assert!(a.validate().is_ok());
    }

    #[test]
    fn an_empty_span_is_rejected() {
        let mut a = authority();
        a.logical_tokens = 5..5;
        assert!(a.validate().is_err());
    }

    #[test]
    fn replay_is_absent_until_a_checkpoint_is_attached() {
        let a = authority();
        assert!(!a.has_durable_replay());
        let a = a.with_replay_checkpoint(CheckpointId::from_counter(1));
        assert!(a.has_durable_replay());
        assert_eq!(
            reference(&a).replay_checkpoint,
            Some(CheckpointId::from_counter(1))
        );
    }

    #[test]
    fn a_reference_from_another_epoch_is_stale() {
        let a = authority();
        let r = reference(&a);
        assert_eq!(
            r.check_resolves(2, &a).unwrap_err(),
            PromotionError::StaleAuthorityEpoch
        );
    }

    #[test]
    fn edited_content_fails_the_digest_rather_than_resolving() {
        let a = authority();
        let r = reference(&a);
        let edited = SourceAuthority::from_tokens(1, 16_389, &[10, 11, 12, 99]);
        assert_eq!(
            r.check_resolves(1, &edited).unwrap_err(),
            PromotionError::ContentDigestMismatch
        );
    }

    #[test]
    fn a_moved_span_fails_even_with_identical_content() {
        let a = authority();
        let r = reference(&a);
        let moved = SourceAuthority::from_tokens(1, 20_000, &[10, 11, 12, 13]);
        assert_eq!(
            r.check_resolves(1, &moved).unwrap_err(),
            PromotionError::ContentDigestMismatch
        );
    }

    #[test]
    fn an_unchanged_reference_resolves() {
        let a = authority();
        assert!(reference(&a).check_resolves(1, &a).is_ok());
        assert_eq!(reference(&a).token_count(), 4);
    }

    #[test]
    fn overlap_detection_is_half_open() {
        let r = reference(&authority()); // 16389..16393
        assert!(r.overlaps(&(16_392..16_400)));
        assert!(r.overlaps(&(16_380..16_390)));
        assert!(!r.overlaps(&(16_393..16_400)), "end is exclusive");
        assert!(!r.overlaps(&(16_380..16_389)), "start is inclusive");
    }
}
