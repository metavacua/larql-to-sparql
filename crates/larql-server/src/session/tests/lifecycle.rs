//! TTL eviction and the idle clock.

use super::*;

#[test]
fn zero_ttl_falls_back_to_default() {
    let sm = SessionManager::new(0);
    assert_eq!(sm.ttl(), Duration::from_secs(DEFAULT_SESSION_TTL_SECS));
}

#[test]
fn explicit_ttl_is_honored() {
    let sm = SessionManager::new(TEST_TTL_SECS);
    assert_eq!(sm.ttl(), Duration::from_secs(TEST_TTL_SECS));
}

#[test]
fn evict_expired_removes_only_idle_sessions() {
    let sm = SessionManager::new(TEST_TTL_SECS);
    let t0 = Instant::now();
    insert_session(&sm, "idle", t0);
    insert_session(&sm, "fresh", t0 + Duration::from_secs(TEST_TTL_SECS));

    // At t0 + TTL + 1s the first session is past its TTL, the second is not.
    let removed = block_on(sm.evict_expired_at(t0 + Duration::from_secs(TEST_TTL_SECS + 1)));
    assert_eq!(removed, 1);
    assert_eq!(block_on(sm.session_count()), 1);

    let map = sm.sessions_blocking_read();
    assert!(map.contains_key("fresh"));
    assert!(!map.contains_key("idle"));
}

#[test]
fn evict_expired_keeps_sessions_exactly_at_ttl() {
    // The boundary is inclusive: a session idle for exactly TTL survives.
    let sm = SessionManager::new(TEST_TTL_SECS);
    let t0 = Instant::now();
    insert_session(&sm, "boundary", t0);

    let removed = block_on(sm.evict_expired_at(t0 + Duration::from_secs(TEST_TTL_SECS)));
    assert_eq!(removed, 0);
    assert_eq!(block_on(sm.session_count()), 1);
}

#[test]
fn evict_expired_on_empty_map_is_noop() {
    let sm = SessionManager::new(TEST_TTL_SECS);
    assert_eq!(block_on(sm.evict_expired_at(Instant::now())), 0);
    assert_eq!(block_on(sm.session_count()), 0);
}

#[test]
fn touch_refreshes_the_idle_clock() {
    let sm = SessionManager::new(TEST_TTL_SECS);
    let t0 = Instant::now();
    insert_session(&sm, "touched", t0);

    let touch_at = t0 + Duration::from_secs(TEST_TTL_SECS - 1);
    {
        let map = sm.sessions_blocking_write();
        map.get("touched").expect("present").touch(touch_at);
    }

    // Past the original creation's TTL but within the touched TTL: kept.
    let removed = block_on(sm.evict_expired_at(t0 + Duration::from_secs(TEST_TTL_SECS + 1)));
    assert_eq!(removed, 0);

    // Past the touched TTL: evicted.
    let removed = block_on(sm.evict_expired_at(touch_at + Duration::from_secs(TEST_TTL_SECS + 1)));
    assert_eq!(removed, 1);
    assert_eq!(block_on(sm.session_count()), 0);
}

#[test]
fn rebinding_an_existing_session_touches_it_without_replacing_the_lease() {
    let sm = SessionManager::new(TEST_TTL_SECS);
    let t0 = Instant::now();
    insert_session(&sm, "s", t0);
    let first = {
        let map = sm.sessions_blocking_read();
        std::sync::Arc::clone(map.get("s").expect("present").lease())
    };

    insert_session(&sm, "s", t0 + Duration::from_secs(TEST_TTL_SECS - 1));
    let second = {
        let map = sm.sessions_blocking_read();
        std::sync::Arc::clone(map.get("s").expect("present").lease())
    };

    assert!(
        std::sync::Arc::ptr_eq(&first, &second),
        "rebinding must not mint a new lease — in-flight owners hold the old one"
    );
    assert!(first.is_alive());
}

#[test]
fn evicting_a_session_kills_its_lease() {
    // The lease outlives the map entry: whoever still holds it (an
    // in-flight generation) must be able to see that it died.
    let sm = SessionManager::new(TEST_TTL_SECS);
    let t0 = Instant::now();
    insert_session(&sm, "doomed", t0);
    let lease = {
        let map = sm.sessions_blocking_read();
        std::sync::Arc::clone(map.get("doomed").expect("present").lease())
    };
    assert!(lease.is_alive());

    block_on(sm.evict_expired_at(t0 + Duration::from_secs(TEST_TTL_SECS + 1)));
    assert!(!lease.is_alive());
}

#[test]
fn expired_sessions_are_unobservable_before_the_sweeper_runs() {
    // Reads never evict (they must not queue behind an in-flight forward
    // pass), so expiry has to be applied at read time instead.
    let sm = SessionManager::new(TEST_TTL_SECS);
    let t0 = Instant::now();
    insert_session(&sm, "stale", t0);
    let after = t0 + Duration::from_secs(TEST_TTL_SECS + 1);

    assert!(block_on(sm.get_at("stale", after)).is_none());
    assert!(block_on(sm.list_at(after)).is_empty());
    assert_eq!(block_on(sm.live_count_at(after)), 0);
    // Still physically present: the read path did not take a writer.
    assert_eq!(sm.sessions_blocking_read().len(), 1);
}

#[test]
fn evict_expired_uses_current_time() {
    // The wall-clock entry point: freshly created sessions survive it.
    let sm = SessionManager::new(TEST_TTL_SECS);
    insert_session(&sm, "now", Instant::now());
    assert_eq!(block_on(sm.evict_expired()), 0);
    assert_eq!(block_on(sm.session_count()), 1);
}

#[test]
fn bind_creates_without_an_overlay() {
    // Binding for identity alone must not clone a model index.
    let sm = SessionManager::new(TEST_TTL_SECS);
    let lease = block_on(sm.bind("s", TEST_MODEL));
    assert_eq!(lease.id(), "s");
    assert_eq!(lease.model_id(), TEST_MODEL);

    let map = sm.sessions_blocking_read();
    let session = map.get("s").expect("bound");
    assert!(session.patched().is_none());
    assert_eq!(session.num_patches(), 0);
    assert!(session.patches().is_empty());
}

#[test]
fn bind_evicts_expired_sessions_opportunistically() {
    let sm = SessionManager::new(TEST_TTL_SECS);
    let t0 = Instant::now();
    insert_session(&sm, "stale", t0);
    assert_eq!(sm.sessions_blocking_read().len(), 1);

    let later = t0 + Duration::from_secs(TEST_TTL_SECS + 1);
    block_on(sm.bind_at("fresh", TEST_MODEL, later));
    let map = sm.sessions_blocking_read();
    assert!(!map.contains_key("stale"), "write paths still evict");
    assert!(map.contains_key("fresh"));
}

#[test]
fn drop_sessions_bound_to_removes_only_the_matching_model() {
    let sm = SessionManager::new(TEST_TTL_SECS);
    let t0 = Instant::now();
    block_on(sm.bind_at("a", "model-x", t0));
    block_on(sm.bind_at("b", "model-x", t0));
    block_on(sm.bind_at("c", "model-y", t0));

    assert_eq!(block_on(sm.drop_sessions_bound_to("model-x")), 2);
    assert_eq!(block_on(sm.session_count()), 1);
    let map = sm.sessions_blocking_read();
    assert!(map.contains_key("c"));
    assert!(!map.contains_key("a"));
    assert!(!map.contains_key("b"));
}

#[test]
fn drop_sessions_bound_to_an_unknown_model_frees_nothing() {
    let sm = SessionManager::new(TEST_TTL_SECS);
    insert_session(&sm, "s", Instant::now());
    assert_eq!(block_on(sm.drop_sessions_bound_to("never-bound")), 0);
    assert_eq!(block_on(sm.session_count()), 1);
}

#[test]
fn drop_sessions_bound_to_kills_the_lease_of_every_session_it_removes() {
    // Same reason `delete`/eviction kill the lease: an in-flight
    // generation holding the old lease must observe the model it was
    // generating under is gone, not silently keep working as if the
    // session were still live.
    let sm = SessionManager::new(TEST_TTL_SECS);
    let lease = block_on(sm.bind("doomed", "model-x"));
    assert!(lease.is_alive());

    assert_eq!(block_on(sm.drop_sessions_bound_to("model-x")), 1);
    assert!(!lease.is_alive());
}
