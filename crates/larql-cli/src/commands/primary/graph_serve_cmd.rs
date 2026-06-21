use std::{
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
};

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

/// Arguments for `larql graph-serve` — serves a `.larql.json` graph over HTTP.
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

/// Build the axum Router with state already applied — returns `Router<()>`
/// so it can be used directly with `oneshot()` in tests.
///
/// `Graph` contains `RefCell` (for lazy node caching), so `Graph: !Sync`.
/// We wrap it in `Mutex` to satisfy axum's `State: Clone + Send + Sync` bound.
/// `RwLock` would NOT work here: `RwLock<T>: Sync` requires `T: Sync`, which
/// `Graph` doesn't satisfy. `Mutex<T>: Sync` only requires `T: Send`. ✓
pub fn build_router(graph: Arc<Mutex<Graph>>) -> Router {
    Router::new()
        .route("/v1/graph/describe", get(handle_describe))
        .route("/v1/completions", post(handle_completions))
        .with_state(graph)
}

async fn handle_describe(
    State(graph): State<Arc<Mutex<Graph>>>,
    Query(q): Query<DescribeQuery>,
) -> (StatusCode, Json<Vec<EdgeView>>) {
    // Take the describe result out of the lock scope so the guard is dropped
    // before any await point, keeping the future Send.
    let result = {
        let g = graph.lock().unwrap();
        g.describe(&q.entity)
    };
    let edges: Vec<EdgeView> = result
        .outgoing
        .iter()
        .chain(result.incoming.iter())
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
    State(graph): State<Arc<Mutex<Graph>>>,
    axum::Json(req): axum::Json<CompletionRequest>,
) -> (StatusCode, Json<CompletionResponse>) {
    // Parse "DESCRIBE <entity>" from prompt; fall back to first word.
    let entity = req
        .prompt
        .trim()
        .strip_prefix("DESCRIBE ")
        .unwrap_or(req.prompt.split_whitespace().next().unwrap_or(""))
        .trim()
        .to_string();

    let result = {
        let g = graph.lock().unwrap();
        g.describe(&entity)
    };
    let text = result
        .outgoing
        .iter()
        .chain(result.incoming.iter())
        .map(|e| format!("{} {} {}", e.subject, e.relation, e.object))
        .collect::<Vec<_>>()
        .join("\n");
    let resp = CompletionResponse {
        choices: vec![CompletionChoice { text }],
    };
    (StatusCode::OK, Json(resp))
}

/// Load the graph and start the axum HTTP server, blocking until killed.
pub fn run(args: GraphServeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let graph = load_graph(&args.graph).map_err(|e| format!("{e}"))?;
    let graph = Arc::new(Mutex::new(graph));
    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
    eprintln!("larql graph-serve listening on http://{addr}");

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async {
            let app = build_router(graph);
            let listener = tokio::net::TcpListener::bind(addr).await?;
            axum::serve(listener, app).await?;
            Ok::<(), std::io::Error>(())
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_core::core::{edge::Edge, graph::Graph};
    use std::sync::{Arc, Mutex};
    use tower::ServiceExt; // provides .oneshot()

    #[tokio::test]
    async fn describe_route_returns_edges() {
        let mut g = Graph::new();
        g.add_edge(Edge::new("module_a", "calls", "module_b"));
        let shared = Arc::new(Mutex::new(g));
        let app = build_router(shared);

        let req = axum::http::Request::builder()
            .uri("/v1/graph/describe?entity=module_a")
            .body(axum::body::Body::empty())
            .unwrap();
        let response = app.oneshot(req).await.unwrap();
        assert_eq!(response.status(), 200);
        let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert!(json.is_array());
        assert!(!json.as_array().unwrap().is_empty());
    }
}
