//! Browser/Node.js WASM bindings for larql-core.
//!
//! Two build variants:
//! - **serial** (default) — single-threaded, standard wasm32-unknown-unknown.
//! - **parallel** (`--features parallel`) — multi-threaded via wasm-bindgen-rayon,
//!   backed by Web Workers and SharedArrayBuffer. Requires nightly + atomics flags.

use wasm_bindgen::prelude::*;

mod bench;
pub use bench::*;

/// Re-export the thread-pool initialiser for the parallel variant.
/// JS callers must `await initThreadPool(n)` before using parallel operations.
#[cfg(feature = "parallel")]
pub use wasm_bindgen_rayon::init_thread_pool;

/// An in-memory graph session for browser / Node.js consumers.
///
/// Build a session, load JSON, then query.
#[wasm_bindgen]
pub struct GraphSession {
    graph: larql_core::Graph,
}

#[wasm_bindgen]
impl GraphSession {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            graph: larql_core::Graph::new(),
        }
    }

    /// Load a graph from its JSON representation (larql-core `.larql.json` format).
    ///
    /// Accepts the full JSON string produced by `larql-core`'s serialiser.
    pub fn load_json(&mut self, json_str: &str) -> Result<(), JsValue> {
        self.graph = larql_core::io::json::from_json_bytes(json_str.as_bytes())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(())
    }

    /// Number of edges in the loaded graph.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Number of distinct entity nodes in the loaded graph.
    pub fn entity_count(&self) -> usize {
        self.graph.list_entities().len()
    }

    /// Return graph statistics as a JSON string.
    pub fn stats(&self) -> Result<String, JsValue> {
        let s = self.graph.stats();
        serde_json::to_string(&s).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Run PageRank and return the top-`k` entities as a JSON array.
    ///
    /// Uses the serial (single-threaded) larql-core implementation regardless of
    /// which feature variant is active.  The *parallel* variant benchmarks N
    /// independent computations running concurrently — see `benchmark_pagerank`.
    pub fn pagerank(&self, iterations: u32, top_k: usize) -> Result<String, JsValue> {
        let result =
            larql_core::pagerank(&self.graph, 0.85, iterations as usize, 1e-6);
        let top = result.top_k(top_k);
        let json: Vec<_> = top
            .iter()
            .map(|(entity, rank)| serde_json::json!({ "entity": entity, "rank": rank }))
            .collect();
        serde_json::to_string(&json).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// BFS traversal from `start` up to `depth` hops.
    ///
    /// Returns a JSON array of visited entity names in visit order.
    pub fn bfs(&self, start: &str, depth: u32) -> Result<String, JsValue> {
        let result = larql_core::bfs_traversal(&self.graph, start, depth as usize);
        serde_json::to_string(&result.nodes)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

impl Default for GraphSession {
    fn default() -> Self {
        Self::new()
    }
}
