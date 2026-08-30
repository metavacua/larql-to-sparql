use super::*;

fn stored(id: &str) -> StoredResponse {
    StoredResponse {
        id: id.to_string(),
        model_id: "m".to_string(),
        conversation: vec![StoredMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }],
        envelope: serde_json::json!({"id": id}),
    }
}

#[test]
fn insert_then_get_round_trips() {
    let store = ResponseStore::new();
    store.insert(stored("resp_a"));
    let got = store.get("resp_a").expect("stored");
    assert_eq!(got.model_id, "m");
    assert_eq!(got.conversation.len(), 1);
    assert_eq!(got.envelope["id"], "resp_a");
    assert_eq!(store.len(), 1);
    assert!(!store.is_empty());
}

#[test]
fn get_unknown_is_none() {
    let store = ResponseStore::new();
    assert!(store.get("resp_missing").is_none());
    assert!(store.is_empty());
}

#[test]
fn remove_reports_existence() {
    let store = ResponseStore::new();
    store.insert(stored("resp_a"));
    assert!(store.remove("resp_a"));
    assert!(!store.remove("resp_a"));
    assert!(store.get("resp_a").is_none());
}

#[test]
fn reinsert_same_id_replaces_without_growth() {
    let store = ResponseStore::new();
    store.insert(stored("resp_a"));
    let mut updated = stored("resp_a");
    updated.model_id = "m2".to_string();
    store.insert(updated);
    assert_eq!(store.len(), 1);
    assert_eq!(store.get("resp_a").unwrap().model_id, "m2");
}

#[test]
fn eviction_drops_oldest_first() {
    let store = ResponseStore::new();
    for i in 0..(MAX_STORED_RESPONSES + 3) {
        store.insert(stored(&format!("resp_{i}")));
    }
    assert_eq!(store.len(), MAX_STORED_RESPONSES);
    // The three oldest are gone; the newest survive.
    assert!(store.get("resp_0").is_none());
    assert!(store.get("resp_1").is_none());
    assert!(store.get("resp_2").is_none());
    assert!(store.get("resp_3").is_some());
    assert!(store
        .get(&format!("resp_{}", MAX_STORED_RESPONSES + 2))
        .is_some());
}

#[test]
fn poisoned_lock_recovers() {
    let store = std::sync::Arc::new(ResponseStore::new());
    store.insert(stored("resp_a"));
    let s2 = std::sync::Arc::clone(&store);
    let _ = std::thread::spawn(move || {
        let _guard = s2.inner.lock().unwrap();
        panic!("poison the store lock");
    })
    .join();
    // Reads and writes still work after the panic.
    assert!(store.get("resp_a").is_some());
    store.insert(stored("resp_b"));
    assert_eq!(store.len(), 2);
}
