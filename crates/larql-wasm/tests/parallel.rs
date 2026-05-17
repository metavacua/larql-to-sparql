//! Parallel (multi-threaded) integration tests.
//!
//! Only compiled when the `parallel` feature is active.
//! Runs under Node.js 20+ (SharedArrayBuffer supported without flags).
//!
//! `wasm-pack test crates/larql-wasm --node --features parallel`

#![cfg(feature = "parallel")]

use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_node_experimental);

use larql_wasm::GraphSession;

/// Node.js tests cannot call async JS to init the thread pool, so parallel
/// tests use a single-thread rayon fallback (rayon detects no workers and
/// falls back to serial execution).  The important thing tested here is that
/// the `parallel` feature compiles and the exported symbols are callable.
#[wasm_bindgen_test]
fn parallel_session_new_and_edge_count() {
    let s = GraphSession::new();
    assert_eq!(s.edge_count(), 0);
}

#[wasm_bindgen_test]
fn parallel_benchmark_pagerank_runs() {
    let ms = larql_wasm::benchmark_pagerank_parallel(100, 2);
    assert!(ms >= 0.0);
}

#[wasm_bindgen_test]
fn parallel_benchmark_bfs_runs() {
    let ms = larql_wasm::benchmark_bfs_parallel(100, 2);
    assert!(ms >= 0.0);
}
