//! The timebase every session record shares.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Milliseconds in a second — the conversion between the stored offset
/// and the wall-clock projection.
const MILLIS_PER_SEC: u64 = 1_000;

/// One timebase shared by every session record a
/// [`SessionManager`](super::SessionManager) owns.
///
/// Lifetime policy (the idle TTL) must be **monotonic**: an NTP step must
/// never expire a live session nor resurrect a dead one. The HTTP surface,
/// meanwhile, has to report `created_at` / `last_used_at` as ordinary unix
/// seconds. Keeping two independently-updated stamps would let the policy
/// clock and the reported clock diverge — the classic "two authorities for
/// one fact" bug — so a session stores exactly **one** quantity,
/// milliseconds since [`SessionClock::new`], and the wall-clock value is
/// *derived* from it. A monotonic offset is also `AtomicU64`-friendly,
/// which is what lets [`SessionLease`](super::SessionLease) refresh the
/// idle clock without taking any lock.
#[derive(Clone, Copy, Debug)]
pub struct SessionClock {
    origin: Instant,
    origin_unix: u64,
}

impl SessionClock {
    /// Anchor a new timebase at "now".
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
            origin_unix: unix_now(),
        }
    }

    /// Offset of `at` from the origin, in milliseconds. Saturates at 0 for
    /// instants captured before this clock existed (tests hand-build such
    /// instants; production never does).
    pub fn millis_at(&self, at: Instant) -> u64 {
        u64::try_from(at.saturating_duration_since(self.origin).as_millis()).unwrap_or(u64::MAX)
    }

    /// Project a stored offset back onto the wall clock, in unix seconds.
    pub fn unix_at(&self, millis: u64) -> u64 {
        self.origin_unix.saturating_add(millis / MILLIS_PER_SEC)
    }
}

impl Default for SessionClock {
    fn default() -> Self {
        Self::new()
    }
}

/// Current unix time in seconds; 0 if the host clock predates the epoch.
///
/// Single implementation for the whole crate — the OpenAI surface's
/// `created` stamps re-export this one rather than keeping their own.
pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_anchors_a_clock_like_new() {
        let clock = SessionClock::default();
        assert!(clock.unix_at(0) > 0, "anchored on a real wall clock");
        assert_eq!(clock.millis_at(clock.origin), 0);
    }

    #[test]
    fn unix_now_is_after_the_2020s_began() {
        // 2020-01-01T00:00:00Z — a sanity floor, not a precise assertion.
        const TWENTY_TWENTY: u64 = 1_577_836_800;
        assert!(unix_now() > TWENTY_TWENTY);
    }
}
