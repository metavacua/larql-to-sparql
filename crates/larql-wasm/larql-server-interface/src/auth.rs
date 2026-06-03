//! API key authentication middleware.
//!
//! # Threat model
//!
//! `==` on `&str` is short-circuit: it stops at the first differing byte.
//! That leaks bytewise progress to a network attacker who can measure
//! request timing — they can guess one byte at a time and recover the
//! full key in linear (rather than exponential) time. This middleware
//! gates every authenticated request, so the leak is reachable.
//!
//! Mitigation: hash both the provided token and the configured key with
//! SHA-256, then compare the fixed-length digests with
//! [`subtle::ConstantTimeEq`]. The hash step removes any length-based
//! leak; the constant-time compare removes the bytewise leak. SHA-256
//! preimage resistance means a digest match implies the inputs match.

use std::sync::Arc;

use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::middleware::Next;
use axum::response::Response;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::http::{BEARER_PREFIX, HEALTH_PATH};
use crate::state::AppState;

/// Constant-time equality on bearer tokens.
///
/// Hashes both inputs with SHA-256 and compares the 32-byte digests with
/// [`subtle::ConstantTimeEq`]. See module docs for the threat model.
fn tokens_match(provided: &str, expected: &str) -> bool {
    let a = Sha256::digest(provided.as_bytes());
    let b = Sha256::digest(expected.as_bytes());
    bool::from(a.ct_eq(&b))
}

/// Middleware that validates the Authorization: Bearer <api_key> header.
/// If no api_key is configured, all requests pass through.
pub async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let required_key = match &state.api_key {
        Some(key) => key,
        None => return Ok(next.run(request).await),
    };

    // Allow health checks without auth.
    if request.uri().path() == HEALTH_PATH {
        return Ok(next.run(request).await);
    }

    let auth_header = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(header) if header.starts_with(BEARER_PREFIX) => {
            let token = &header[BEARER_PREFIX.len()..];
            if tokens_match(token, required_key) {
                Ok(next.run(request).await)
            } else {
                Err(StatusCode::UNAUTHORIZED)
            }
        }
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

#[cfg(test)]
mod tests {
    use super::tokens_match;

    #[test]
    fn matching_tokens_compare_equal() {
        assert!(tokens_match("secret123", "secret123"));
    }

    #[test]
    fn non_matching_tokens_compare_unequal() {
        assert!(!tokens_match("secret123", "wrongkey"));
    }

    #[test]
    fn different_lengths_do_not_match() {
        assert!(!tokens_match("short", "longer-key"));
        assert!(!tokens_match("longer-key", "short"));
    }

    #[test]
    fn empty_tokens_match_each_other() {
        // Defensible behaviour: empty == empty. The middleware never
        // reaches `tokens_match` with an empty `required_key` in
        // practice (api_key is `Option<String>`), but the helper itself
        // shouldn't pretend two empty inputs differ.
        assert!(tokens_match("", ""));
    }

    #[test]
    fn empty_provided_does_not_match_real_key() {
        assert!(!tokens_match("", "secret123"));
        assert!(!tokens_match("secret123", ""));
    }

    #[test]
    fn single_byte_difference_does_not_match() {
        // The bytewise-timing leak this fix targets: under naive `==`,
        // these two strings would have differed only at the last byte
        // and an attacker could measure that. Functional correctness
        // here is the same as the naive impl; the property we care
        // about (constant-time) isn't directly assertable in unit
        // tests, but compiling against `subtle::ConstantTimeEq`
        // documents the choice.
        assert!(!tokens_match("aaaaaaaaaa", "aaaaaaaaab"));
        assert!(!tokens_match("aaaaaaaaab", "aaaaaaaaaa"));
    }
}
