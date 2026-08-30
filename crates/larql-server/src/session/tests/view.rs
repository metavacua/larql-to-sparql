//! The read-only projection `/v1/sessions` renders: summaries, ordering,
//! and deletion.

use super::*;

#[test]
fn summary_reports_identity_lifecycle_and_patch_names() {
    let sm = SessionManager::new(TEST_TTL_SECS);
    let t0 = Instant::now();
    insert_session(&sm, "s", t0);
    give_overlay(&sm, "s", &["alpha", "beta"]);

    let summary = block_on(sm.get_at("s", t0)).expect("live");
    assert_eq!(summary.id, "s");
    assert_eq!(summary.model, TEST_MODEL);
    assert_eq!(summary.patch_names, vec!["alpha", "beta"]);
    assert_eq!(summary.expires_at, summary.last_used_at + TEST_TTL_SECS);
    assert_eq!(summary.resumptions, 0);
    assert_eq!(summary.reused_tokens_total, 0);
}

#[test]
fn summary_carries_the_sessions_own_resumption_counters() {
    let sm = SessionManager::new(TEST_TTL_SECS);
    let t0 = Instant::now();
    insert_session(&sm, "s", t0);
    {
        let map = sm.sessions_blocking_read();
        map.get("s").expect("present").lease().record_resumption(7);
    }
    let summary = block_on(sm.get_at("s", t0)).expect("live");
    assert_eq!(summary.resumptions, 1);
    assert_eq!(summary.reused_tokens_total, 7);
}

#[test]
fn list_orders_most_recently_used_first() {
    let sm = SessionManager::new(TEST_TTL_SECS);
    let t0 = Instant::now();
    insert_session(&sm, "older", t0);
    insert_session(&sm, "newer", t0 + Duration::from_secs(2));

    let listed = block_on(sm.list_at(t0 + Duration::from_secs(2)));
    let ids: Vec<&str> = listed.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(ids, vec!["newer", "older"]);
}

#[test]
fn get_unknown_session_is_absent() {
    let sm = SessionManager::new(TEST_TTL_SECS);
    assert!(block_on(sm.get("nope")).is_none());
}

#[test]
fn delete_frees_the_overlay_and_kills_the_lease() {
    let sm = SessionManager::new(TEST_TTL_SECS);
    let t0 = Instant::now();
    insert_session(&sm, "s", t0);
    give_overlay(&sm, "s", &["alpha"]);
    let lease = {
        let map = sm.sessions_blocking_read();
        std::sync::Arc::clone(map.get("s").expect("present").lease())
    };

    assert_eq!(block_on(sm.delete("s")), Some(1), "one patch freed");
    assert!(!lease.is_alive());
    assert!(block_on(sm.get("s")).is_none());
    assert_eq!(sm.sessions_blocking_read().len(), 0);
}

#[test]
fn delete_is_idempotent() {
    let sm = SessionManager::new(TEST_TTL_SECS);
    insert_session(&sm, "s", Instant::now());
    assert_eq!(block_on(sm.delete("s")), Some(0));
    assert_eq!(
        block_on(sm.delete("s")),
        None,
        "repeat delete is not an error"
    );
    assert_eq!(block_on(sm.delete("never-existed")), None);
}

#[test]
fn delete_leaves_unrelated_sessions_untouched() {
    let sm = SessionManager::new(TEST_TTL_SECS);
    let t0 = Instant::now();
    insert_session(&sm, "victim", t0);
    insert_session(&sm, "bystander", t0);
    let bystander = {
        let map = sm.sessions_blocking_read();
        std::sync::Arc::clone(map.get("bystander").expect("present").lease())
    };

    block_on(sm.delete("victim"));
    assert!(bystander.is_alive());
    assert!(block_on(sm.get("bystander")).is_some());
}

#[test]
fn a_deleted_session_id_can_be_bound_again_as_a_new_incarnation() {
    let sm = SessionManager::new(TEST_TTL_SECS);
    insert_session(&sm, "s", Instant::now());
    let first = {
        let map = sm.sessions_blocking_read();
        std::sync::Arc::clone(map.get("s").expect("present").lease())
    };
    block_on(sm.delete("s"));

    let second = block_on(sm.bind("s", TEST_MODEL));
    assert!(!first.is_alive(), "the old incarnation stays dead");
    assert!(second.is_alive());
    assert!(!std::sync::Arc::ptr_eq(&first, &second));
}
