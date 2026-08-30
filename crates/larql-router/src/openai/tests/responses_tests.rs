//! Unit tests for the Responses API sticky machinery: the bounded
//! route store, the id-capture scanner, and backend resolution.

use axum::body::Bytes;

use crate::openai::responses::{
    extract_response_id, previous_response_id_of, resolve_sticky, IdCapture, ResponseRouteStore,
    MAX_ROUTED_RESPONSES,
};
use crate::openai::OpenAIBackend;

fn backend(model: &str, url: &str) -> OpenAIBackend {
    OpenAIBackend {
        model_id: model.to_string(),
        listen_url: url.to_string(),
        requests_in_flight: 0,
    }
}

// ── ResponseRouteStore ───────────────────────────────────────────────

#[test]
fn store_round_trips_and_removes() {
    let store = ResponseRouteStore::default();
    assert!(store.is_empty());
    store.insert("resp_1", "http://a");
    assert_eq!(store.get("resp_1").as_deref(), Some("http://a"));
    assert_eq!(store.len(), 1);
    assert!(store.remove("resp_1"));
    assert!(!store.remove("resp_1"));
    assert_eq!(store.get("resp_1"), None);
}

#[test]
fn store_insert_same_id_replaces_without_growing() {
    let store = ResponseRouteStore::default();
    store.insert("resp_1", "http://a");
    store.insert("resp_1", "http://b");
    assert_eq!(store.len(), 1);
    assert_eq!(store.get("resp_1").as_deref(), Some("http://b"));
}

#[test]
fn store_evicts_fifo_at_capacity() {
    let store = ResponseRouteStore::default();
    for i in 0..=MAX_ROUTED_RESPONSES {
        store.insert(&format!("resp_{i}"), "http://a");
    }
    assert_eq!(store.len(), MAX_ROUTED_RESPONSES);
    assert_eq!(store.get("resp_0"), None, "oldest route must be evicted");
    assert!(store.get(&format!("resp_{MAX_ROUTED_RESPONSES}")).is_some());
}

// ── extract_response_id / IdCapture ──────────────────────────────────

#[test]
fn extract_finds_the_envelope_id() {
    let body = br#"{"id":"resp_abc123","object":"response","output":[]}"#;
    assert_eq!(extract_response_id(body).as_deref(), Some("resp_abc123"));
}

#[test]
fn extract_finds_the_id_inside_an_sse_event() {
    let body = b"event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_sse9\",\"object\":\"response\"}}\n\n";
    assert_eq!(extract_response_id(body).as_deref(), Some("resp_sse9"));
}

#[test]
fn extract_ignores_non_response_ids() {
    let body = br#"{"id":"msg_1","object":"message"}"#;
    assert_eq!(extract_response_id(body), None);
}

#[test]
fn extract_incomplete_id_is_none_until_closed() {
    assert_eq!(extract_response_id(br#"{"id":"resp_par"#), None);
}

#[test]
fn capture_handles_marker_split_across_chunks() {
    let mut capture = IdCapture::new();
    assert_eq!(capture.observe(br#"{"id":"re"#), None);
    assert_eq!(
        capture
            .observe(br#"sp_split42","object":"response"}"#)
            .as_deref(),
        Some("resp_split42")
    );
    // Only reported once.
    assert_eq!(capture.observe(br#"{"id":"resp_other"}"#), None);
}

#[test]
fn capture_gives_up_past_the_scan_limit() {
    let mut capture = IdCapture::new();
    let filler = vec![b'x'; 8192];
    assert_eq!(capture.observe(&filler), None);
    // Even a real id afterwards is ignored — the response did not lead
    // with one, so it is not a response envelope.
    assert_eq!(capture.observe(br#"{"id":"resp_late"}"#), None);
}

// ── previous_response_id_of / resolve_sticky ─────────────────────────

#[test]
fn previous_response_id_is_read_when_present() {
    let body = Bytes::from(r#"{"input":"x","previous_response_id":"resp_p1"}"#);
    assert_eq!(previous_response_id_of(&body).as_deref(), Some("resp_p1"));
    assert_eq!(
        previous_response_id_of(&Bytes::from(r#"{"input":"x"}"#)),
        None
    );
    assert_eq!(previous_response_id_of(&Bytes::from("not json")), None);
}

#[test]
fn resolve_sticky_prefers_the_recorded_backend() {
    let backends = vec![backend("m", "http://a"), backend("m", "http://b")];
    let chosen = resolve_sticky(&backends, Some("http://b".to_string())).unwrap();
    assert_eq!(chosen.listen_url, "http://b");
}

#[test]
fn resolve_sticky_departed_backend_falls_to_single_server() {
    let backends = vec![backend("m", "http://a")];
    let chosen = resolve_sticky(&backends, Some("http://gone".to_string())).unwrap();
    assert_eq!(chosen.listen_url, "http://a");
}

#[test]
fn resolve_sticky_miss_with_multiple_servers_is_none() {
    let backends = vec![backend("m", "http://a"), backend("m", "http://b")];
    assert_eq!(resolve_sticky(&backends, None), None);
}

#[test]
fn resolve_sticky_miss_with_no_servers_is_none() {
    assert_eq!(resolve_sticky(&[], None), None);
}
