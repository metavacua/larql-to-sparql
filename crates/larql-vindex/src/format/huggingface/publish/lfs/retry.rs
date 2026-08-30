//! Bounded retry for transient upload failures.
//!
//! A large publish issues hundreds of PUTs against HF's S3 backend, and
//! that backend rate-limits. Without a retry, one `503 SlowDown` on part
//! 35 of 109 aborts the whole publish — which is exactly what happened on
//! 2026-08-07 and left `gemma-3-4b-it-vindex-server` holding a stale weight
//! file with no replacement. A partially-uploaded repo is worse than a
//! failed one, because it still serves.
//!
//! Scope: transport errors and the transient HTTP statuses below. A 4xx
//! that is not 408/429 is a real rejection (bad signature, expired URL) and
//! is returned immediately — retrying it would just burn the backoff budget
//! and delay the error the caller needs to see.

use std::time::Duration;

/// Attempts per operation, including the first. Five attempts with the
/// backoff below spans ~15 s, comfortably past the short shaping windows
/// S3 applies while still failing fast on a genuine outage.
pub(super) const TRANSIENT_MAX_ATTEMPTS: u32 = 5;

/// First backoff step; doubles per attempt (1s, 2s, 4s, 8s).
pub(super) const TRANSIENT_BACKOFF_BASE: Duration = Duration::from_secs(1);

/// Upper bound on a single sleep, including a server-supplied
/// `Retry-After`. Stops a hostile or mistaken header from parking the
/// upload for minutes.
pub(super) const TRANSIENT_BACKOFF_CAP: Duration = Duration::from_secs(30);

/// HTTP statuses worth retrying: request timeout, explicit rate limit, and
/// the 5xx family S3 uses for shaping and transient unavailability.
pub(super) fn is_transient_status(status: u16) -> bool {
    matches!(status, 408 | 429 | 500 | 502 | 503 | 504)
}

/// Backoff for `attempt` (1-based), honouring a server `Retry-After` in
/// seconds when it is present and shorter than the cap.
pub(super) fn backoff_for(attempt: u32, retry_after_secs: Option<u64>) -> Duration {
    let exponential = TRANSIENT_BACKOFF_BASE
        .saturating_mul(1u32 << attempt.saturating_sub(1).min(16))
        .min(TRANSIENT_BACKOFF_CAP);
    match retry_after_secs {
        Some(s) => Duration::from_secs(s).clamp(exponential, TRANSIENT_BACKOFF_CAP),
        None => exponential,
    }
}

/// Read `Retry-After` as whole seconds. The HTTP-date form is not parsed —
/// S3 and HF both send the delta-seconds form, and guessing wrong on a
/// date is worse than falling back to the exponential schedule.
pub(super) fn retry_after_secs(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_statuses_are_exactly_the_retryable_family() {
        for s in [408, 429, 500, 502, 503, 504] {
            assert!(is_transient_status(s), "{s} should be transient");
        }
        // 403 is an expired/incorrect pre-signed URL and 404 a missing
        // target: retrying either just delays a real error.
        for s in [200, 400, 401, 403, 404, 409, 422, 501] {
            assert!(!is_transient_status(s), "{s} should not be transient");
        }
    }

    #[test]
    fn backoff_doubles_and_is_capped() {
        assert_eq!(backoff_for(1, None), Duration::from_secs(1));
        assert_eq!(backoff_for(2, None), Duration::from_secs(2));
        assert_eq!(backoff_for(3, None), Duration::from_secs(4));
        assert_eq!(backoff_for(4, None), Duration::from_secs(8));
        // Far past the schedule, the cap holds rather than overflowing.
        assert_eq!(backoff_for(30, None), TRANSIENT_BACKOFF_CAP);
    }

    #[test]
    fn retry_after_raises_but_never_exceeds_the_cap() {
        // Server asks for longer than the exponential step — honour it.
        assert_eq!(backoff_for(1, Some(5)), Duration::from_secs(5));
        // Server asks for less than we already intended — keep our own
        // floor, so a "Retry-After: 0" cannot spin the loop.
        assert_eq!(backoff_for(4, Some(0)), Duration::from_secs(8));
        // Absurd value is clamped.
        assert_eq!(backoff_for(1, Some(86_400)), TRANSIENT_BACKOFF_CAP);
    }

    #[test]
    fn retry_after_parses_only_delta_seconds() {
        let mut h = reqwest::header::HeaderMap::new();
        h.insert(reqwest::header::RETRY_AFTER, "12".parse().unwrap());
        assert_eq!(retry_after_secs(&h), Some(12));

        // HTTP-date form is deliberately ignored.
        h.insert(
            reqwest::header::RETRY_AFTER,
            "Wed, 21 Oct 2026 07:28:00 GMT".parse().unwrap(),
        );
        assert_eq!(retry_after_secs(&h), None);

        assert_eq!(retry_after_secs(&reqwest::header::HeaderMap::new()), None);
    }
}
