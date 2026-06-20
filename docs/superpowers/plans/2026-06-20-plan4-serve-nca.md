# larql graph-serve + NCA Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `larql graph-serve` (a lightweight HTTP server that exposes a `.larql.json` graph over OpenAI-compat endpoints), add `HttpProvider::larql_graph(port)` to `larql-core` so NCA and other tools can query it as an inference backend, and implement a `VindexProvider` for direct in-process vindex queries (Phase 2).

**Architecture:** `graph-serve` reuses the existing `larql-server` HTTP infrastructure and listens on a configurable port. It implements two routes: `GET /v1/graph/describe?entity=X` and `POST /v1/completions` (OpenAI-compat, uses LQL WALK as the "completion"). `HttpProvider::larql_graph(port)` points at these routes. `VindexProvider` is a zero-HTTP alternative that loads a `.larql.json` or vindex directory directly. Cloud providers (Anthropic, MiniMax) become optional via a `--provider` flag in NCA; the default changes to `larql_graph` if a local graph-serve is detected.

**Tech Stack:** Rust stable, `axum 0.7`, `tokio`, `larql-core`, `larql-lql`, `larql-vindex`, existing `HttpProvider` pattern in `larql-core/src/engine/http_provider.rs`.

## Global Constraints

- `larql graph-serve` is Tier 2 (network) — it must NOT be added to the wasm32v1-none CI job.
- Cloud providers (Anthropic, MiniMax) remain available; they must be explicitly selected with `--provider anthropic` or `--provider minimax`.
- `VindexProvider` is Tier 1 (filesystem) — no network access.
- All new types must implement `serde::Serialize + serde::Deserialize` where they cross HTTP boundaries.
- Run `cargo test --workspace` before every commit.

---

### Task 1: `larql graph-serve` command

**Files:**
- Create: `crates/larql-cli/src/commands/primary/graph_serve_cmd.rs`
- Modify: `crates/larql-cli/src/commands/primary/mod.rs`
- Modify: `crates/larql-cli/src/main.rs`
- Modify: `crates/larql-cli/Cargo.toml`

**Interfaces:**
- Produces:
  - `larql graph-serve <LARQL_JSON> [--port 8181]` CLI command.
  - Route `GET /v1/graph/describe?entity=<name>` → JSON array of edges.
  - Route `POST /v1/completions` with body `{"model":"...", "prompt":"DESCRIBE <entity>"}` → OpenAI-compat JSON.

- [ ] **Step 1: Add `axum` + `tokio` to `larql-cli/Cargo.toml`**

```toml
axum = "0.7"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

- [ ] **Step 2: Write an integration test**

Create `crates/larql-cli/src/commands/primary/graph_serve_cmd.rs` with a test block:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use larql_core::core::{edge::Edge, graph::Graph};
    use std::sync::Arc;

    #[tokio::test]
    async fn describe_route_returns_edges() {
        let mut g = Graph::new();
        g.add_edge(Edge::new("module_a", "calls", "module_b"));
        let shared = Arc::new(g);
        let app = build_router(shared);

        let req = axum::http::Request::builder()
            .uri("/v1/graph/describe?entity=module_a")
            .body(axum::body::Body::empty())
            .unwrap();
        let response = axum::Router::oneshot(app, req).await.unwrap();
        assert_eq!(response.status(), 200);
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert!(json.is_array());
        assert!(!json.as_array().unwrap().is_empty());
    }
}
```

- [ ] **Step 3: Run to confirm failure**

```bash
cargo test -p larql-cli describe_route_returns_edges
```

Expected: `error[E0425]: cannot find function 'build_router'`

- [ ] **Step 4: Implement `build_router` and the `run` function**

```rust
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use clap::Args;
use larql_core::core::graph::Graph;
use larql_core::io::load_graph;
use serde::{Deserialize, Serialize};

#[derive(Args)]
pub struct GraphServeArgs {
    /// Path to a .larql.json graph file.
    graph: PathBuf,

    /// Port to listen on.
    #[arg(long, default_value_t = 8181)]
    port: u16,
}

#[derive(Deserialize)]
struct DescribeQuery {
    entity: String,
}

#[derive(Serialize)]
struct EdgeView {
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub confidence: f64,
}

#[derive(Deserialize)]
struct CompletionRequest {
    pub prompt: String,
    #[allow(dead_code)]
    pub model: Option<String>,
}

#[derive(Serialize)]
struct CompletionResponse {
    pub choices: Vec<CompletionChoice>,
}

#[derive(Serialize)]
struct CompletionChoice {
    pub text: String,
}

pub fn build_router(graph: Arc<Graph>) -> Router {
    Router::new()
        .route("/v1/graph/describe", get(handle_describe))
        .route("/v1/completions", post(handle_completions))
        .with_state(graph)
}

async fn handle_describe(
    State(graph): State<Arc<Graph>>,
    Query(q): Query<DescribeQuery>,
) -> (StatusCode, Json<Vec<EdgeView>>) {
    let edges: Vec<EdgeView> = graph
        .describe(&q.entity)
        .into_iter()
        .map(|e| EdgeView {
            subject: e.subject.clone(),
            relation: e.relation.clone(),
            object: e.object.clone(),
            confidence: e.confidence,
        })
        .collect();
    (StatusCode::OK, Json(edges))
}

async fn handle_completions(
    State(graph): State<Arc<Graph>>,
    axum::Json(req): axum::Json<CompletionRequest>,
) -> (StatusCode, Json<CompletionResponse>) {
    // Parse "DESCRIBE <entity>" from prompt; fall back to first word.
    let entity = req
        .prompt
        .trim()
        .strip_prefix("DESCRIBE ")
        .unwrap_or(req.prompt.split_whitespace().next().unwrap_or(""))
        .trim();
    let edges = graph.describe(entity);
    let text = edges
        .iter()
        .map(|e| format!("{} {} {}", e.subject, e.relation, e.object))
        .collect::<Vec<_>>()
        .join("\n");
    let resp = CompletionResponse {
        choices: vec![CompletionChoice { text }],
    };
    (StatusCode::OK, Json(resp))
}

pub fn run(args: GraphServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let graph = load_graph(&args.graph).map_err(|e| format!("{e}"))?;
    let graph = Arc::new(graph);
    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
    eprintln!("larql graph-serve listening on http://{addr}");

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async {
            let app = build_router(graph);
            let listener = tokio::net::TcpListener::bind(addr).await?;
            axum::serve(listener, app).await?;
            Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
        })?;
    Ok(())
}
```

Note: `graph.describe(&entity)` must be verified against `larql-core`'s actual `Graph::describe` signature — it may return `Vec<Edge>` or `Vec<&Edge>`; adjust the `EdgeView` construction accordingly.

- [ ] **Step 5: Register in mod.rs and main.rs**

```rust
// primary/mod.rs:
pub mod graph_serve_cmd;

// Commands enum:
#[command(next_help_heading = "Server")]
/// Serve a .larql.json graph over HTTP (OpenAI-compat + graph describe API).
GraphServe(graph_serve_cmd::GraphServeArgs),

// match arm:
Commands::GraphServe(args) => graph_serve_cmd::run(args),
```

- [ ] **Step 6: Run tests**

```bash
cargo test -p larql-cli describe_route_returns_edges
```

Expected: `test commands::primary::graph_serve_cmd::tests::describe_route_returns_edges ... ok`

- [ ] **Step 7: Manual smoke test**

```bash
# Terminal 1
cat > /tmp/test-graph.larql.json << 'EOF'
{"larql_version":"0.1.0","metadata":{},"schema":{},"edges":[
  {"s":"module_a","r":"calls","o":"module_b","c":1.0,"src":"ast"},
  {"s":"module_b","r":"calls","o":"module_c","c":1.0,"src":"ast"}
]}
EOF
cargo run -p larql-cli -- graph-serve /tmp/test-graph.larql.json --port 8181 &

# Terminal 2 (or after brief pause)
curl -s "http://localhost:8181/v1/graph/describe?entity=module_a" | python3 -m json.tool
```

Expected: JSON array with one edge `{"subject":"module_a","relation":"calls","object":"module_b","confidence":1.0}`.

- [ ] **Step 8: Commit**

```bash
git add crates/larql-cli/src/commands/primary/graph_serve_cmd.rs \
        crates/larql-cli/src/commands/primary/mod.rs \
        crates/larql-cli/src/main.rs \
        crates/larql-cli/Cargo.toml
git commit -m "feat(cli): larql graph-serve — OpenAI-compat HTTP server for .larql.json graphs"
```

---

### Task 2: `HttpProvider::larql_graph(port)` in `larql-core`

**Files:**
- Modify: `crates/larql-core/src/engine/http_provider.rs`

**Interfaces:**
- Consumes: existing `HttpProvider` struct (line 32 has `pub fn ollama`, line 37 has `pub fn llama_cpp`)
- Produces: `pub fn larql_graph(port: u16) -> Self` — sets base URL to `http://127.0.0.1:{port}` with endpoint `/v1/completions`.

- [ ] **Step 1: Read the existing constructor pattern**

```bash
sed -n '25,55p' crates/larql-core/src/engine/http_provider.rs
```

Note the exact struct fields set by `ollama()` and `llama_cpp()`. The new `larql_graph()` follows the same pattern.

- [ ] **Step 2: Write the failing test**

In `crates/larql-core/src/engine/http_provider.rs` (add to existing test block):

```rust
#[test]
fn larql_graph_url_uses_port() {
    let p = HttpProvider::larql_graph(8181);
    assert!(p.base_url().contains("8181"), "base_url should contain port 8181");
}
```

- [ ] **Step 3: Run to confirm failure**

```bash
cargo test -p larql-core larql_graph_url_uses_port
```

Expected: `error[E0599]: no function named 'larql_graph' found`

- [ ] **Step 4: Implement `larql_graph`**

Mirror the `ollama` or `llama_cpp` constructor — only the base URL differs:

```rust
pub fn larql_graph(port: u16) -> Self {
    Self {
        base_url: format!("http://127.0.0.1:{port}/v1"),
        model: "larql-graph".to_string(),
        // Copy other fields from the ollama() constructor (api_key, timeout, etc.)
        ..Self::default()
    }
}
```

Adjust field names to match the actual `HttpProvider` struct fields found in Step 1.

- [ ] **Step 5: Run tests**

```bash
cargo test -p larql-core larql_graph_url_uses_port
cargo test -p larql-core
```

Expected: both pass.

- [ ] **Step 6: Commit**

```bash
git add crates/larql-core/src/engine/http_provider.rs
git commit -m "feat(core): HttpProvider::larql_graph(port) — local graph-serve backend preset"
```

---

### Task 3: `VindexProvider` — direct in-process graph query (Phase 2)

**Files:**
- Create: `crates/larql-core/src/engine/vindex_provider.rs`
- Modify: `crates/larql-core/src/engine/mod.rs`

**Interfaces:**
- Consumes: `larql_core::io::load_graph`, `Graph::describe`, `Graph::walk`
- Produces:
  - `pub struct VindexProvider { graph: Graph }`
  - `pub fn VindexProvider::from_file(path: &Path) -> Result<Self, GraphError>`
  - `pub fn VindexProvider::complete(&self, prompt: &str) -> String` — same semantics as `handle_completions` in Task 1 but in-process.

- [ ] **Step 1: Write the failing test**

Create `crates/larql-core/src/engine/vindex_provider.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use larql_core::core::{edge::Edge, graph::Graph};

    #[test]
    fn vindex_provider_describe_returns_text() {
        let mut g = Graph::new();
        g.add_edge(Edge::new("alpha", "calls", "beta"));
        let provider = VindexProvider::from_graph(g);
        let result = provider.complete("DESCRIBE alpha");
        assert!(result.contains("alpha"), "response should mention alpha");
        assert!(result.contains("calls"), "response should include relation");
    }

    #[test]
    fn vindex_provider_empty_response_for_unknown() {
        let g = Graph::new();
        let provider = VindexProvider::from_graph(g);
        let result = provider.complete("DESCRIBE nonexistent");
        // Should return empty string (no edges) without panicking
        assert_eq!(result, "");
    }
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
cargo test -p larql-core vindex_provider_describe_returns_text
```

Expected: `error[E0412]: cannot find type 'VindexProvider'`

- [ ] **Step 3: Implement `VindexProvider`**

```rust
use std::path::Path;
use larql_core::core::graph::{Graph, GraphError};
use larql_core::io::load_graph;

pub struct VindexProvider {
    graph: Graph,
}

impl VindexProvider {
    pub fn from_graph(graph: Graph) -> Self {
        Self { graph }
    }

    pub fn from_file(path: &Path) -> Result<Self, GraphError> {
        let graph = load_graph(path)?;
        Ok(Self { graph })
    }

    pub fn complete(&self, prompt: &str) -> String {
        let entity = prompt
            .trim()
            .strip_prefix("DESCRIBE ")
            .unwrap_or(prompt.split_whitespace().next().unwrap_or(""))
            .trim();
        self.graph
            .describe(entity)
            .iter()
            .map(|e| format!("{} {} {}", e.subject, e.relation, e.object))
            .collect::<Vec<_>>()
            .join("\n")
    }
}
```

- [ ] **Step 4: Expose from `larql-core/src/engine/mod.rs`**

```rust
pub mod vindex_provider;
pub use vindex_provider::VindexProvider;
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p larql-core vindex_provider_describe_returns_text vindex_provider_empty_response_for_unknown
```

Expected: both pass.

- [ ] **Step 6: Verify `larql-core` still passes wasm32v1-none**

```bash
cargo build --target wasm32v1-none -p larql-core
```

Expected: `Finished` (the `io` module is already gated; `VindexProvider::from_file` uses `load_graph` which is also gated — verify that the `cfg` gate propagates correctly; if not, wrap `from_file` in `#[cfg(not(target_arch = "wasm32"))]`).

- [ ] **Step 7: Commit**

```bash
git add crates/larql-core/src/engine/vindex_provider.rs crates/larql-core/src/engine/mod.rs
git commit -m "feat(core): VindexProvider — direct in-process graph queries (Phase 2)"
```

---

### Task 4: Make cloud providers opt-in in the NCA integration

> **Note:** `metavacua/native-cli-ai` is a separate repository. The changes in this task apply to that repo. The larql-main codebase provides the `HttpProvider::larql_graph(port)` and `VindexProvider` APIs that NCA consumes. Do NOT modify chrishayuk/larql — that is read-only.

**Files (in `native-cli-ai` repo):**
- Modify: `crates/core/src/provider/mod.rs` (or wherever the provider enum is)
- Modify: NCA's config / CLI flags to add `--provider larql_graph[:<port>]` and default to local graph-serve detection

**Interfaces:**
- Consumes: `larql_core::engine::HttpProvider::larql_graph(port)` (from the `larql-core` dep)
- Produces: `nca run --prompt "..." --provider larql_graph` uses the local graph-serve; `--provider anthropic` uses Anthropic; no `--provider` flag tries local first, then falls back to cloud.

- [ ] **Step 1: Find the provider enum in native-cli-ai**

```bash
find . -name "*.rs" | xargs grep -l "Anthropic\|Provider\|minimax" | head -5
```

Read the provider enum definition to understand the current shape.

- [ ] **Step 2: Write a failing unit test for the provider selection**

In `crates/core/src/provider/mod.rs` (add to test block):

```rust
#[test]
fn larql_graph_provider_is_selectable() {
    let p = ProviderConfig::from_str("larql_graph:8181").unwrap();
    assert!(matches!(p, ProviderConfig::LarqlGraph { port: 8181 }));
}
```

- [ ] **Step 3: Run to confirm failure**

```bash
cargo test -p nca-core larql_graph_provider_is_selectable
```

Expected: `error[E0599]: no variant 'LarqlGraph'`

- [ ] **Step 4: Add `LarqlGraph` variant to the provider enum**

Find the provider enum (likely `ProviderConfig` or `Provider`) and add:

```rust
LarqlGraph { port: u16 },
```

Add parsing:

```rust
fn from_str(s: &str) -> Result<Self, Self::Err> {
    if let Some(rest) = s.strip_prefix("larql_graph:") {
        let port: u16 = rest.parse()?;
        return Ok(Self::LarqlGraph { port });
    }
    if s == "larql_graph" {
        return Ok(Self::LarqlGraph { port: 8181 });
    }
    // existing cases follow ...
}
```

- [ ] **Step 5: Wire LarqlGraph to HttpProvider::larql_graph**

In the provider dispatch (wherever the provider is instantiated from the config):

```rust
ProviderConfig::LarqlGraph { port } => {
    // larql-core must be a dependency of native-cli-ai
    larql_core::engine::HttpProvider::larql_graph(port)
}
```

Add `larql-core = { path = "../larql-main/crates/larql-core" }` (or the published version) to `native-cli-ai/Cargo.toml`.

- [ ] **Step 6: Change default provider to try local first**

In the NCA entrypoint where the provider is selected when `--provider` is not given:

```rust
// Try larql-graph-serve first; fall back to Anthropic if not reachable.
let provider = match args.provider {
    Some(p) => p,
    None => {
        if is_larql_graph_reachable(8181) {
            ProviderConfig::LarqlGraph { port: 8181 }
        } else {
            ProviderConfig::Anthropic  // or the existing default
        }
    }
};

fn is_larql_graph_reachable(port: u16) -> bool {
    std::net::TcpStream::connect_timeout(
        &format!("127.0.0.1:{port}").parse().unwrap(),
        std::time::Duration::from_millis(200),
    ).is_ok()
}
```

- [ ] **Step 7: Run test**

```bash
cargo test -p nca-core larql_graph_provider_is_selectable
```

Expected: `test larql_graph_provider_is_selectable ... ok`

- [ ] **Step 8: Commit (to metavacua/larql-to-sparql or native-cli-ai only)**

```bash
git add crates/core/src/provider/
git commit -m "feat(nca): add LarqlGraph provider — local graph-serve default, cloud opt-in"
```

---

### Task 5: Full integration test — end-to-end pipeline

- [ ] **Step 1: Full workspace build**

```bash
cargo build --workspace
```

Expected: `Finished`, no errors.

- [ ] **Step 2: Full workspace test**

```bash
cargo test --workspace
```

Expected: all tests pass.

- [ ] **Step 3: End-to-end pipeline smoke test**

```bash
# 1. Extract this codebase into a .larql.json graph
cargo run -p larql-cli -- extract-codebase . --output /tmp/larql-main.vindex
# (extract-codebase currently writes a vindex, not .larql.json — this step
#  exercises the vindex path)

# 2. Create a small test graph and serve it
cat > /tmp/pipe-test.larql.json << 'EOF'
{"larql_version":"0.1.0","metadata":{},"schema":{},"edges":[
  {"s":"larql_core","r":"implements","o":"graph-algorithms","c":1.0,"src":"ast"},
  {"s":"larql_cli","r":"depends_on","o":"larql_core","c":1.0,"src":"ast"}
]}
EOF

# 3. Start graph-serve in background
cargo run -p larql-cli -- graph-serve /tmp/pipe-test.larql.json --port 8181 &
SERVE_PID=$!
sleep 1

# 4. Describe an entity
curl -s "http://localhost:8181/v1/graph/describe?entity=larql_core"

# 5. Query via completions endpoint
curl -s -X POST "http://localhost:8181/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"DESCRIBE larql_core","model":"larql-graph"}'

# 6. Kill graph-serve
kill $SERVE_PID
```

Expected output from Step 4: JSON array with `implements` edge. Step 5: `choices[0].text` containing `larql_core implements graph-algorithms`.

- [ ] **Step 4: make ci**

```bash
make ci
```

Expected: fmt + clippy + test all pass.

- [ ] **Step 5: Final commit (if any fmt/clippy fixes needed)**

```bash
git add -p
git commit -m "chore(serve-nca): fix fmt/clippy issues"
```
