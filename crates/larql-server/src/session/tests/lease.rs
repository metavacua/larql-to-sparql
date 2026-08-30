//! The lock-free identity record: liveness, idle clock, counters, and the
//! monotonic → wall-clock projection.

use super::*;

fn lease_at(at: Instant, clock: SessionClock) -> std::sync::Arc<SessionLease> {
    SessionLease::new("s", TEST_MODEL, clock, at)
}

#[test]
fn a_fresh_lease_is_alive_and_zeroed() {
    let clock = SessionClock::new();
    let lease = lease_at(Instant::now(), clock);
    assert!(lease.is_alive());
    assert_eq!(lease.resumptions(), 0);
    assert_eq!(lease.reused_tokens_total(), 0);
}

#[test]
fn kill_is_permanent_and_idempotent() {
    let clock = SessionClock::new();
    let lease = lease_at(Instant::now(), clock);
    lease.kill();
    assert!(!lease.is_alive());
    lease.kill();
    assert!(!lease.is_alive());
}

#[test]
fn idle_grows_with_the_clock_and_resets_on_touch() {
    let clock = SessionClock::new();
    let t0 = Instant::now();
    let lease = lease_at(t0, clock);
    assert_eq!(lease.idle_at(t0), Duration::ZERO);
    assert_eq!(
        lease.idle_at(t0 + Duration::from_secs(5)),
        Duration::from_secs(5)
    );

    lease.touch_at(t0 + Duration::from_secs(5));
    assert_eq!(lease.idle_at(t0 + Duration::from_secs(5)), Duration::ZERO);
    assert_eq!(
        lease.idle_at(t0 + Duration::from_secs(7)),
        Duration::from_secs(2)
    );
}

#[test]
fn idle_never_goes_backwards_for_a_stale_instant() {
    // Saturating arithmetic: an instant before the last touch reads as
    // zero idle, never as an enormous duration that would evict a live
    // session.
    let clock = SessionClock::new();
    let t0 = Instant::now();
    let lease = lease_at(t0, clock);
    lease.touch_at(t0 + Duration::from_secs(10));
    assert_eq!(lease.idle_at(t0), Duration::ZERO);
}

#[test]
fn wall_clock_stamps_track_the_one_stored_offset() {
    // created_at is fixed at birth; last_used_at follows touches; both are
    // derived from the same monotonic offset, so they cannot disagree.
    let clock = SessionClock::new();
    let t0 = Instant::now();
    let lease = lease_at(t0, clock);
    let created = lease.created_at_unix();
    assert_eq!(lease.last_used_at_unix(), created);

    lease.touch_at(t0 + Duration::from_secs(30));
    assert_eq!(lease.created_at_unix(), created, "birth stamp is immutable");
    assert_eq!(lease.last_used_at_unix(), created + 30);
}

#[test]
fn expires_at_is_last_use_plus_ttl() {
    let clock = SessionClock::new();
    let t0 = Instant::now();
    let lease = lease_at(t0, clock);
    let ttl = Duration::from_secs(TEST_TTL_SECS);
    assert_eq!(
        lease.expires_at_unix(ttl),
        lease.last_used_at_unix() + TEST_TTL_SECS
    );

    lease.touch_at(t0 + Duration::from_secs(4));
    assert_eq!(
        lease.expires_at_unix(ttl),
        lease.created_at_unix() + 4 + TEST_TTL_SECS
    );
}

#[test]
fn resumption_counters_accumulate() {
    let clock = SessionClock::new();
    let lease = lease_at(Instant::now(), clock);
    lease.record_resumption(12);
    lease.record_resumption(30);
    assert_eq!(lease.resumptions(), 2);
    assert_eq!(lease.reused_tokens_total(), 42);
}

#[test]
fn clock_projects_offsets_onto_unix_seconds() {
    let clock = SessionClock::new();
    let origin_unix = clock.unix_at(0);
    assert_eq!(clock.unix_at(1_500), origin_unix + 1);
    assert_eq!(clock.unix_at(999), origin_unix, "sub-second offsets floor");
}

#[test]
fn clock_saturates_for_instants_that_predate_it() {
    let before = Instant::now();
    let clock = SessionClock::new();
    assert_eq!(clock.millis_at(before), 0);
}
