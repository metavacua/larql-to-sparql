//! WASM-exported benchmark helpers.
//!
//! These functions let the Node.js benchmark script (`scripts/wasm-bench.mjs`)
//! drive timing comparisons between the serial and parallel build variants.
//!
//! The "parallel" variant uses rayon (backed by wasm-bindgen-rayon Web Workers)
//! to run N independent pagerank/BFS computations concurrently, establishing a
//! CPU-thread scaling baseline ahead of WebGPU offloading work.

use wasm_bindgen::prelude::*;

// ── Synthetic graph builder ───────────────────────────────────────────────────

/// Build a synthetic graph with `n_edges` directed edges for benchmarking.
///
/// Uses a deterministic LCG so results are reproducible across serial and
/// parallel runs.
fn synthetic_graph(n_edges: u32) -> larql_core::Graph {
    let mut graph = larql_core::Graph::new();
    let mut rng: u64 = 0xDEAD_BEEF_CAFE_1234;

    let entities = (n_edges / 4).max(4) as u64;
    let relations = ["knows", "contains", "relates_to", "precedes", "follows"];

    for _ in 0..n_edges {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005)
                 .wrapping_add(1_442_695_040_888_963_407);
        let s = format!("e{}", rng % entities);
        rng = rng.wrapping_mul(6_364_136_223_846_793_005)
                 .wrapping_add(1_442_695_040_888_963_407);
        let o = format!("e{}", rng % entities);
        let rel = relations[(rng % relations.len() as u64) as usize];
        graph.add_edge(larql_core::Edge::new(s, rel, o));
    }
    graph
}

// ── Serial benchmark exports ──────────────────────────────────────────────────

/// Run PageRank `rounds` times on a synthetic graph with `n_edges` edges.
///
/// Returns elapsed wall-clock milliseconds (using `Date.now()`).
#[wasm_bindgen]
pub fn benchmark_pagerank_serial(n_edges: u32, rounds: u32) -> f64 {
    let graph = synthetic_graph(n_edges);
    let start = js_sys::Date::now();
    for _ in 0..rounds {
        let _ = larql_core::pagerank(&graph, 0.85, 20, 1e-4);
    }
    js_sys::Date::now() - start
}

/// Run BFS from the first entity `rounds` times on a synthetic graph.
///
/// Returns elapsed wall-clock milliseconds.
#[wasm_bindgen]
pub fn benchmark_bfs_serial(n_edges: u32, rounds: u32) -> f64 {
    let graph = synthetic_graph(n_edges);
    let start = js_sys::Date::now();
    for _ in 0..rounds {
        let _ = larql_core::bfs_traversal(&graph, "e0", 6);
    }
    js_sys::Date::now() - start
}

// ── Parallel benchmark exports ────────────────────────────────────────────────

/// Run PageRank `rounds` times in parallel using rayon's thread pool.
///
/// larql_core::Graph contains a RefCell for lazy node caching and is therefore
/// !Sync — it cannot be shared across rayon threads.  Each parallel worker
/// builds its own independent Graph instead, modelling N concurrent independent
/// graph queries rather than N queries over a single shared graph.
///
/// Requires the caller to first `await initThreadPool(n)`.
///
/// Returns elapsed wall-clock milliseconds.
#[cfg(feature = "parallel")]
#[wasm_bindgen]
pub fn benchmark_pagerank_parallel(n_edges: u32, rounds: u32) -> f64 {
    use rayon::prelude::*;
    let start = js_sys::Date::now();
    (0..rounds as usize).into_par_iter().for_each(|_| {
        let graph = synthetic_graph(n_edges);
        let _ = larql_core::pagerank(&graph, 0.85, 20, 1e-4);
    });
    js_sys::Date::now() - start
}

/// Run BFS `rounds` times in parallel.
///
/// Each worker builds its own Graph for the same reason as benchmark_pagerank_parallel.
/// Requires the caller to first `await initThreadPool(n)`.
///
/// Returns elapsed wall-clock milliseconds.
#[cfg(feature = "parallel")]
#[wasm_bindgen]
pub fn benchmark_bfs_parallel(n_edges: u32, rounds: u32) -> f64 {
    use rayon::prelude::*;
    let start = js_sys::Date::now();
    (0..rounds as usize).into_par_iter().for_each(|_| {
        let graph = synthetic_graph(n_edges);
        let _ = larql_core::bfs_traversal(&graph, "e0", 6);
    });
    js_sys::Date::now() - start
}
