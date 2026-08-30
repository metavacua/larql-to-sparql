//! Session ownership of continuation states: explicit freeing, orphan
//! collection, and the delete-vs-late-insert resurrection race.
//!
//! The race gate is [`race`]: the cache mutex serialises a concurrent
//! delete and insert into one of exactly two orderings, so driving both
//! across real threads — with a channel handoff, not a sleep — covers
//! the interleaving without ever depending on the scheduler. Both must
//! end with the deleted session holding nothing.

use super::*;
use crate::session::SessionManager;
use std::sync::Arc;

/// TTL long enough that nothing in these tests expires by accident.
const OWNER_TTL_SECS: u64 = 3_600;

/// A live lease for `id`, obtained the way a request obtains one.
fn owner(manager: &SessionManager, id: &str) -> Arc<crate::session::SessionLease> {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("current-thread runtime")
        .block_on(manager.bind(id, TEST_MODEL))
}

fn cache() -> ResponseKvCache {
    ResponseKvCache::new(8, TEST_TTL_SECS)
}

#[test]
fn owned_by_reports_only_this_sessions_states() {
    let manager = SessionManager::new(OWNER_TTL_SECS);
    let cache = cache();
    let mine = owner(&manager, "mine");
    let theirs = owner(&manager, "theirs");

    cache.insert(
        "resp_1",
        TEST_MODEL,
        Some(Arc::clone(&mine)),
        handoff(&[1, 2, 3]),
    );
    cache.insert(
        "resp_2",
        TEST_MODEL,
        Some(Arc::clone(&mine)),
        handoff(&[4, 5]),
    );
    cache.insert("resp_3", TEST_MODEL, Some(theirs), handoff(&[6]));
    cache.insert("resp_4", TEST_MODEL, None, handoff(&[7, 8, 9, 10]));

    let summary = cache.owned_by("mine");
    assert!(summary.available());
    assert_eq!(summary.entries, 2);
    assert_eq!(summary.input_tokens, 5, "absorbed prompt ids, summed");

    assert_eq!(cache.owned_by("theirs").input_tokens, 1);
    let none = cache.owned_by("never-bound");
    assert!(!none.available());
    assert_eq!(none, SessionContinuation::default());
}

#[test]
fn drop_owned_by_frees_only_that_session() {
    let manager = SessionManager::new(OWNER_TTL_SECS);
    let cache = cache();
    let mine = owner(&manager, "mine");
    let theirs = owner(&manager, "theirs");

    cache.insert("resp_1", TEST_MODEL, Some(Arc::clone(&mine)), handoff(&[1]));
    cache.insert("resp_2", TEST_MODEL, Some(mine), handoff(&[2]));
    cache.insert("resp_3", TEST_MODEL, Some(theirs), handoff(&[3]));
    cache.insert("resp_4", TEST_MODEL, None, handoff(&[4]));

    assert_eq!(cache.drop_owned_by("mine"), 2);
    assert_eq!(cache.len(), 2);
    assert!(cache.take("resp_3", TEST_MODEL).is_some(), "other session");
    assert!(cache.take("resp_4", TEST_MODEL).is_some(), "unowned");
}

#[test]
fn drop_owned_by_an_unknown_session_frees_nothing() {
    let cache = cache();
    cache.insert("resp_1", TEST_MODEL, None, handoff(&[1]));
    assert_eq!(cache.drop_owned_by("never-bound"), 0);
    assert_eq!(cache.len(), 1);
}

#[test]
fn drop_owned_by_model_frees_every_entry_regardless_of_session_ownership() {
    // Unload's cache sweep — unlike `drop_owned_by`, this doesn't care
    // who (if anyone) owns an entry; only which runtime produced it.
    let manager = SessionManager::new(OWNER_TTL_SECS);
    let cache = cache();
    let owned = owner(&manager, "s");
    cache.insert("resp_1", "model-x", Some(owned), handoff(&[1]));
    cache.insert("resp_2", "model-x", None, handoff(&[2]));
    cache.insert("resp_3", "model-y", None, handoff(&[3]));

    assert_eq!(cache.drop_owned_by_model("model-x"), 2);
    assert_eq!(cache.len(), 1);
    assert!(
        cache.take("resp_3", "model-y").is_some(),
        "other model kept"
    );
}

#[test]
fn drop_owned_by_an_unknown_model_frees_nothing() {
    let cache = cache();
    cache.insert("resp_1", TEST_MODEL, None, handoff(&[1]));
    assert_eq!(cache.drop_owned_by_model("never-loaded"), 0);
    assert_eq!(cache.len(), 1);
}

#[test]
fn drop_owned_by_model_keeps_capacity_accounting_consistent() {
    let cache = ResponseKvCache::new(2, TEST_TTL_SECS);
    cache.insert("resp_1", "model-x", None, handoff(&[1]));
    assert_eq!(cache.drop_owned_by_model("model-x"), 1);

    cache.insert("resp_2", TEST_MODEL, None, handoff(&[2]));
    cache.insert("resp_3", TEST_MODEL, None, handoff(&[3]));
    cache.insert("resp_4", TEST_MODEL, None, handoff(&[4]));
    assert_eq!(cache.len(), 2);
    assert!(
        cache.take("resp_2", TEST_MODEL).is_none(),
        "resp_2 was the oldest live entry"
    );
    assert!(cache.take("resp_3", TEST_MODEL).is_some());
    assert!(cache.take("resp_4", TEST_MODEL).is_some());
}

#[test]
fn drop_owned_by_keeps_capacity_accounting_consistent() {
    // The order queue must be compacted alongside the map, or capacity
    // eviction later targets an id that is no longer resident.
    let manager = SessionManager::new(OWNER_TTL_SECS);
    let cache = ResponseKvCache::new(2, TEST_TTL_SECS);
    let mine = owner(&manager, "mine");
    cache.insert("resp_1", TEST_MODEL, Some(mine), handoff(&[1]));
    assert_eq!(cache.drop_owned_by("mine"), 1);

    cache.insert("resp_2", TEST_MODEL, None, handoff(&[2]));
    cache.insert("resp_3", TEST_MODEL, None, handoff(&[3]));
    cache.insert("resp_4", TEST_MODEL, None, handoff(&[4]));
    assert_eq!(cache.len(), 2);
    assert!(
        cache.take("resp_2", TEST_MODEL).is_none(),
        "resp_2 was the oldest live entry"
    );
    assert!(cache.take("resp_3", TEST_MODEL).is_some());
    assert!(cache.take("resp_4", TEST_MODEL).is_some());
}

#[test]
fn a_late_insert_after_deletion_cannot_resurrect_the_session() {
    // The sequential shadow of the race the concurrency gate covers:
    //   take KV → DELETE session → generation finishes → insert.
    // The lease is dead by insert time, so the state is dropped rather
    // than re-populating a session that no longer exists.
    let manager = SessionManager::new(OWNER_TTL_SECS);
    let cache = cache();
    let lease = owner(&manager, "doomed");
    cache.insert(
        "resp_1",
        TEST_MODEL,
        Some(Arc::clone(&lease)),
        handoff(&[1]),
    );

    // The in-flight request takes the previous turn's state...
    assert!(cache.take("resp_1", TEST_MODEL).is_some());
    // ...the operator deletes the session while it generates...
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("current-thread runtime");
    assert_eq!(rt.block_on(manager.delete("doomed")), Some(0));
    assert_eq!(cache.drop_owned_by("doomed"), 0);
    // ...and the finished generation tries to retain its new state.
    cache.insert("resp_2", TEST_MODEL, Some(lease), handoff(&[1, 2]));

    assert!(cache.is_empty(), "a dead owner's state is never retained");
    assert_eq!(cache.owned_by("doomed"), SessionContinuation::default());
}

#[test]
fn an_expired_session_also_refuses_late_inserts() {
    // TTL eviction kills the lease for the same reason deletion does.
    let manager = SessionManager::new(1);
    let cache = cache();
    let lease = owner(&manager, "s");
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("current-thread runtime");
    rt.block_on(manager.evict_expired_at(Instant::now() + Duration::from_secs(2)));

    cache.insert("resp_1", TEST_MODEL, Some(lease), handoff(&[1]));
    assert!(cache.is_empty());
}

#[test]
fn taking_an_orphan_is_a_miss_that_drops_it() {
    // An entry whose session died between insert and take has no
    // rightful chain left, so — unlike a foreign model id — it is
    // consumed rather than preserved.
    let manager = SessionManager::new(OWNER_TTL_SECS);
    let cache = cache();
    let lease = owner(&manager, "s");
    cache.insert("resp_1", TEST_MODEL, Some(lease), handoff(&[1]));

    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("current-thread runtime");
    rt.block_on(manager.delete("s"));

    assert!(cache.take("resp_1", TEST_MODEL).is_none());
    assert_eq!((cache.hits(), cache.misses()), (0, 1));
    assert!(cache.is_empty(), "the orphan is collected on the way out");
}

#[test]
fn the_sweeper_collects_orphans_the_delete_path_missed() {
    // Opportunistic TTL eviction inside a write path kills leases
    // without touching the cache, so orphan collection has to be a
    // property of the cache's own sweep — not of the delete route.
    let manager = SessionManager::new(OWNER_TTL_SECS);
    let cache = cache();
    let lease = owner(&manager, "s");
    cache.insert("resp_1", TEST_MODEL, Some(lease), handoff(&[1]));
    cache.insert("resp_2", TEST_MODEL, None, handoff(&[2]));

    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("current-thread runtime");
    rt.block_on(manager.delete("s"));

    assert_eq!(cache.evict_expired(), 1, "only the orphan");
    assert_eq!(cache.len(), 1);
    assert!(cache.take("resp_2", TEST_MODEL).is_some());
}

#[test]
fn a_live_owner_does_not_block_normal_reuse() {
    // Ownership must not change the resumption path for a healthy
    // session: take still hits, and still consumes exactly once.
    let manager = SessionManager::new(OWNER_TTL_SECS);
    let cache = cache();
    let lease = owner(&manager, "s");
    cache.insert("resp_1", TEST_MODEL, Some(lease), handoff(&[1, 2, 3]));

    let taken = cache.take("resp_1", TEST_MODEL).expect("live owner hits");
    assert_eq!(taken.absorbed_ids, vec![1, 2, 3]);
    assert_eq!((cache.hits(), cache.misses()), (1, 0));
}

/// Which side of the race runs first. The cache mutex serialises a
/// concurrent delete and insert into exactly one of these two orderings,
/// so covering both is covering the race.
enum Ordering {
    DeleteFirst,
    InsertFirst,
}

/// Drive one ordering across two real threads, handing off through a
/// channel rather than a sleep, and assert the invariant: a deleted
/// session never ends up holding a resident continuation.
fn race(ordering: Ordering) {
    let manager = Arc::new(SessionManager::new(OWNER_TTL_SECS));
    let cache = Arc::new(cache());
    let id = "raced".to_string();
    let lease = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("current-thread runtime")
        .block_on(manager.bind(&id, TEST_MODEL));

    // The in-flight request has already taken the previous turn's state
    // and is generating; it will retain a new one when it finishes.
    let (to_inserter, inserter_go) = std::sync::mpsc::channel::<()>();
    let (to_deleter, deleter_go) = std::sync::mpsc::channel::<()>();

    let kick_inserter = to_inserter.clone();
    let kick_deleter = to_deleter.clone();

    let inserter = {
        let cache = Arc::clone(&cache);
        let lease = Arc::clone(&lease);
        std::thread::spawn(move || {
            inserter_go.recv().expect("start signal");
            cache.insert("resp_new", TEST_MODEL, Some(lease), handoff(&[1, 2]));
            let _ = to_deleter.send(());
        })
    };
    let deleter = {
        let cache = Arc::clone(&cache);
        let manager = Arc::clone(&manager);
        let id = id.clone();
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("current-thread runtime");
            deleter_go.recv().expect("start signal");
            rt.block_on(manager.delete(&id));
            cache.drop_owned_by(&id);
            let _ = to_inserter.send(());
        })
    };

    match ordering {
        // Deletion wins: the late insert must be refused outright.
        Ordering::DeleteFirst => kick_deleter.send(()).expect("kick the deleter"),
        // The insert wins: the purge that follows must remove it.
        Ordering::InsertFirst => kick_inserter.send(()).expect("kick the inserter"),
    }
    inserter.join().expect("inserter");
    deleter.join().expect("deleter");

    assert_eq!(
        cache.owned_by(&id),
        SessionContinuation::default(),
        "a deleted session must hold no continuation"
    );
    assert!(cache.is_empty(), "and nothing may be left resident");
}

#[test]
fn deletion_before_a_late_insert_refuses_it() {
    race(Ordering::DeleteFirst);
}

#[test]
fn deletion_after_an_insert_purges_it() {
    race(Ordering::InsertFirst);
}
