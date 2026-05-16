//! Serial (single-threaded) integration tests.
//!
//! These run under Node.js via `wasm-pack test --node`.

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_node_experimental);

use larql_wasm::GraphSession;

#[wasm_bindgen_test]
fn new_session_starts_empty() {
    let s = GraphSession::new();
    assert_eq!(s.edge_count(), 0);
    assert_eq!(s.entity_count(), 0);
}

#[wasm_bindgen_test]
fn load_minimal_json() {
    let mut s = GraphSession::new();
    // Minimal valid larql-core JSON graph with one edge.
    let json = r#"{
        "larql_version": "0.1.0",
        "metadata": {},
        "schema": {},
        "edges": [
            {
                "subject": "alice",
                "relation": "knows",
                "object":   "bob",
                "confidence": 1.0,
                "source": "Unknown"
            }
        ]
    }"#;
    s.load_json(json).expect("load_json failed");
    assert_eq!(s.edge_count(), 1);
    assert_eq!(s.entity_count(), 2);
}

#[wasm_bindgen_test]
fn load_invalid_json_returns_error() {
    let mut s = GraphSession::new();
    assert!(s.load_json("not valid json {{{{").is_err());
}

#[wasm_bindgen_test]
fn stats_returns_valid_json() {
    let s = GraphSession::new();
    let json_str = s.stats().expect("stats failed");
    // Just check it parses as JSON and has expected fields.
    let v: serde_json::Value =
        serde_json::from_str(&json_str).expect("stats is not valid JSON");
    assert!(v["edges"].is_number());
    assert!(v["entities"].is_number());
}

#[wasm_bindgen_test]
fn pagerank_on_empty_graph_returns_empty_array() {
    let s = GraphSession::new();
    let json_str = s.pagerank(10, 5).expect("pagerank failed");
    let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert_eq!(v.as_array().map(|a| a.len()), Some(0));
}

#[wasm_bindgen_test]
fn bfs_on_empty_graph_returns_empty_array() {
    let s = GraphSession::new();
    let json_str = s.bfs("nonexistent", 3).expect("bfs failed");
    let v: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert_eq!(v.as_array().map(|a| a.len()), Some(0));
}

#[wasm_bindgen_test]
fn benchmark_serial_runs_without_panic() {
    // Smoke test: ensure the serial benchmark helpers don't panic.
    let ms = larql_wasm::benchmark_pagerank_serial(200, 3);
    assert!(ms >= 0.0);
    let ms = larql_wasm::benchmark_bfs_serial(200, 3);
    assert!(ms >= 0.0);
}
