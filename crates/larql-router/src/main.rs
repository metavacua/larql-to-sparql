//! larql-router — transparent layer-sharding proxy for larql-server.
//!
//! Two dispatch modes:
//!   --shards  "0-16=http://host-a:8080,17-33=http://host-b:8081"
//!             Static shard map (ADR-0003, backwards-compatible).
//!   --grid-port 50052
//!             Self-assembling grid (ADR-0004). Servers connect via gRPC
//!             and announce their capabilities. No static configuration.
//!
//! Both modes can coexist. Grid takes priority; static shards are fallback.
//!
//! # Wire format
//!
//! The router is wire-transparent for both JSON (`application/json`) and binary
//! (`application/x-larql-ffn`) requests. For single-shard routes the body is
//! forwarded byte-for-byte with no intermediate parsing. Multi-shard fan-out
//! is supported for JSON only; binary multi-shard requests are rejected with
//! HTTP 400 (use the batched JSON format or route per-shard manually).

#[cfg(not(target_arch = "wasm32"))]
use larql_router::routing::{parse_shards, peek_binary, Shard};

#[cfg(not(target_arch = "wasm32"))]
use larql_router::grid;
#[cfg(not(target_arch = "wasm32"))]
use larql_router::rebalancer;

#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::net::SocketAddr;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use axum::body::Bytes;
#[cfg(not(target_arch = "wasm32"))]
use axum::extract::State;
#[cfg(not(target_arch = "wasm32"))]
use axum::http::{header, StatusCode};
#[cfg(not(target_arch = "wasm32"))]
use axum::response::Response;
#[cfg(not(target_arch = "wasm32"))]
use axum::routing::post;
#[cfg(not(target_arch = "wasm32"))]
use axum::{Json, Router};
#[cfg(not(target_arch = "wasm32"))]
use clap::Parser;
#[cfg(not(target_arch = "wasm32"))]
use serde_json::Value;
#[cfg(not(target_arch = "wasm32"))]
use tokio::sync::RwLock;
#[cfg(not(target_arch = "wasm32"))]
use tonic::transport::Server as GrpcServer;
#[cfg(not(target_arch = "wasm32"))]
use tracing::{info, warn};

#[cfg(not(target_arch = "wasm32"))]
use grid::{GridServiceImpl, GridState};
#[cfg(not(target_arch = "wasm32"))]
use larql_router_protocol::GridServiceServer;

// ── Binary wire format constants ───────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
const BINARY_CT: &str = "application/x-larql-ffn";

// ── CLI ────────────────────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
#[derive(Parser)]
#[command(
    name = "larql-router",
    version,
    about = "Layer-sharding proxy for larql-server"
)]
struct Cli {
    /// Static shard map: comma-separated "START-END=URL" entries (inclusive bounds).
    /// Example: "0-16=http://host-a:8080,17-33=http://host-b:8081"
    /// Optional when --grid-port is provided.
    #[arg(long)]
    shards: Option<String>,

    /// Enable the self-assembling grid gRPC server on this port.
    /// Servers connect here with --join grpc://router:PORT.
    #[arg(long)]
    grid_port: Option<u16>,

    /// HTTP listen port.
    #[arg(long, default_value = "9090")]
    port: u16,

    /// Bind address.
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Per-request timeout to backend shards, in seconds.
    #[arg(long, default_value = "120")]
    timeout_secs: u64,

    /// Log level.
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Shared secret for the self-assembling grid.
    /// Servers must pass the same key via --grid-key to be accepted.
    /// If not set, the grid port is open to any server (development only).
    #[arg(long, env = "LARQL_GRID_KEY")]
    grid_key: Option<String>,

    /// GT6: seconds between rebalancer checks (default: 30).
    /// Set to 0 to disable dynamic rebalancing.
    #[arg(long, default_value = "30")]
    rebalance_interval: u64,

    /// GT6: latency ratio threshold to trigger rebalancing (default: 2.0).
    /// The slowest replica must be this many times slower than the fastest
    /// for the same layer before the rebalancer acts.
    #[arg(long, default_value = "2.0")]
    rebalance_threshold: f32,
}

// ── App state ──────────────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
struct AppState {
    /// Static shards from --shards (may be empty).
    static_shards: Vec<Shard>,
    /// Grid state from --grid-port (None if grid mode not enabled).
    grid: Option<Arc<RwLock<GridState>>>,
    client: reqwest::Client,
}

#[cfg(not(target_arch = "wasm32"))]
impl AppState {
    /// Resolve all layers in one lock acquisition.
    /// Returns Ok(layer → url) or Err(first missing layer).
    async fn resolve_all(
        &self,
        model_id: Option<&str>,
        layers: &[usize],
    ) -> Result<HashMap<usize, String>, usize> {
        if let Some(grid) = &self.grid {
            let guard = grid.read().await;
            let mut out = HashMap::with_capacity(layers.len());
            let mut static_needed: Vec<usize> = Vec::new();
            for &layer in layers {
                match guard.route(model_id, layer as u32) {
                    Some(url) => {
                        out.insert(layer, url);
                    }
                    None => static_needed.push(layer),
                }
            }
            drop(guard);
            for layer in static_needed {
                match self.static_shards.iter().find(|s| s.owns(layer)) {
                    Some(s) => {
                        out.insert(layer, s.url.clone());
                    }
                    None => return Err(layer),
                }
            }
            return Ok(out);
        }
        let mut out = HashMap::with_capacity(layers.len());
        for &layer in layers {
            match self.static_shards.iter().find(|s| s.owns(layer)) {
                Some(s) => {
                    out.insert(layer, s.url.clone());
                }
                None => return Err(layer),
            }
        }
        Ok(out)
    }
}

// ── Route handler ──────────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
async fn handle_walk_ffn(
    State(state): State<Arc<AppState>>,
    request: axum::extract::Request,
) -> Response {
    match handle_walk_ffn_inner(state, request).await {
        Ok(r) => r,
        Err((status, msg)) => {
            // Always return errors as JSON regardless of input content-type.
            let body = format!(r#"{{"error":{}}}"#, serde_json::Value::String(msg));
            Response::builder()
                .status(status)
                .header(header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(body))
                .unwrap()
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
async fn handle_walk_ffn_inner(
    state: Arc<AppState>,
    request: axum::extract::Request,
) -> Result<Response, (StatusCode, String)> {
    let is_binary = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|ct| ct.starts_with(BINARY_CT))
        .unwrap_or(false);

    let body_bytes = axum::body::to_bytes(request.into_body(), 64 * 1024 * 1024)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("read body: {e}")))?;

    let (layers, model_id_owned): (Vec<usize>, Option<String>) = if is_binary {
        let layers = peek_binary(&body_bytes).ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "binary: truncated or malformed header".to_string(),
            )
        })?;
        (layers, None)
    } else {
        let peek: Value = serde_json::from_slice(&body_bytes)
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")))?;
        let layers: Vec<usize> = if let Some(arr) = peek.get("layers").and_then(|v| v.as_array()) {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect()
        } else if let Some(n) = peek.get("layer").and_then(|v| v.as_u64()) {
            vec![n as usize]
        } else {
            return Err((
                StatusCode::BAD_REQUEST,
                "must provide 'layer' or 'layers'".to_string(),
            ));
        };
        let model_id = peek
            .get("model_id")
            .and_then(|v| v.as_str())
            .map(str::to_owned);
        (layers, model_id)
    };

    if layers.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "empty layer list".to_string()));
    }

    let mid = model_id_owned.as_deref();
    let layer_urls = state.resolve_all(mid, &layers).await.map_err(|missing| {
        (
            StatusCode::BAD_REQUEST,
            format!("layer {missing} has no owning shard in this router"),
        )
    })?;

    // Determine unique shards.
    let unique_urls: std::collections::HashSet<&String> = layer_urls.values().collect();

    if unique_urls.len() == 1 || layers.len() == 1 {
        // All layers on the same shard — proxy raw bytes unchanged.
        let url = layer_urls.values().next().unwrap();
        let ct = if is_binary {
            BINARY_CT
        } else {
            "application/json"
        };
        return proxy_raw(&state.client, url, body_bytes, ct).await;
    }

    // Multi-shard dispatch.
    if is_binary {
        return Err((
            StatusCode::BAD_REQUEST,
            "binary fan-out across multiple shards is not supported; use JSON or split by shard"
                .to_string(),
        ));
    }

    // JSON fan-out: group layers by URL, dispatch in parallel, merge.
    let body_value: Value = serde_json::from_slice(&body_bytes)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")))?;

    let mut by_url: HashMap<String, Vec<usize>> = HashMap::new();
    for (&layer, url) in &layer_urls {
        by_url.entry(url.clone()).or_default().push(layer);
    }

    let mut handles = Vec::new();
    for (url, shard_layers) in &by_url {
        let mut sub_body = body_value.clone();
        if shard_layers.len() == 1 {
            sub_body["layer"] = Value::from(shard_layers[0]);
            sub_body.as_object_mut().unwrap().remove("layers");
        } else {
            sub_body["layers"] =
                Value::Array(shard_layers.iter().map(|&l| Value::from(l)).collect());
            sub_body.as_object_mut().unwrap().remove("layer");
        }
        let client = state.client.clone();
        let target = format!("{url}/v1/walk-ffn");
        handles.push(tokio::spawn(async move {
            client
                .post(&target)
                .json(&sub_body)
                .send()
                .await
                .map_err(|e| e.to_string())?
                .json::<Value>()
                .await
                .map_err(|e| e.to_string())
        }));
    }

    let responses: Vec<Value> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|jh| jh.map_err(|e| e.to_string()).and_then(|r| r))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("shard error: {e}")))?;

    let mut all_results: Vec<Value> = Vec::new();
    let mut max_latency: f64 = 0.0;
    for resp in responses {
        if let Some(arr) = resp.get("results").and_then(|v| v.as_array()) {
            all_results.extend(arr.iter().cloned());
        } else if resp.get("layer").is_some() {
            all_results.push(resp.clone());
        }
        if let Some(ms) = resp.get("latency_ms").and_then(|v| v.as_f64()) {
            if ms > max_latency {
                max_latency = ms;
            }
        }
    }
    all_results.sort_by_key(|r| r.get("layer").and_then(|v| v.as_u64()).unwrap_or(0));

    let merged = serde_json::json!({
        "results": all_results,
        "latency_ms": (max_latency * 10.0).round() / 10.0,
    });
    let json_bytes = serde_json::to_vec(&merged)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(json_bytes))
        .unwrap())
}

/// Forward raw bytes to a shard, passing the Content-Type header through.
/// The shard's response status and Content-Type are preserved unchanged.
#[cfg(not(target_arch = "wasm32"))]
async fn proxy_raw(
    client: &reqwest::Client,
    base_url: &str,
    body: Bytes,
    ct: &str,
) -> Result<Response, (StatusCode, String)> {
    let url = format!("{base_url}/v1/walk-ffn");
    let resp = client
        .post(&url)
        .header(reqwest::header::CONTENT_TYPE, ct)
        .body(body.to_vec())
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("shard {base_url}: {e}")))?;

    let status = resp.status();
    let resp_ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let resp_bytes = resp
        .bytes()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("read shard response: {e}")))?;

    Ok(Response::builder()
        .status(status.as_u16())
        .header(header::CONTENT_TYPE, resp_ct)
        .body(axum::body::Body::from(resp_bytes))
        .unwrap())
}

#[cfg(not(target_arch = "wasm32"))]
async fn handle_health() -> Json<Value> {
    Json(serde_json::json!({"status": "ok"}))
}

/// Proxy /v1/stats to the first reachable shard so that clients connecting
/// via RemoteWalkBackend (which reads hidden_size from /v1/stats) work
/// transparently through the router.
#[cfg(not(target_arch = "wasm32"))]
async fn handle_stats(State(state): State<Arc<AppState>>) -> Response {
    // Collect candidate shard URLs: grid shards first, then static.
    let mut candidates: Vec<String> = Vec::new();
    if let Some(grid) = &state.grid {
        let guard = grid.read().await;
        for url in guard.all_shard_urls() {
            candidates.push(url);
        }
    }
    for shard in &state.static_shards {
        if !candidates.contains(&shard.url) {
            candidates.push(shard.url.clone());
        }
    }
    for url in candidates {
        let stats_url = format!("{url}/v1/stats");
        if let Ok(resp) = state.client.get(&stats_url).send().await {
            if resp.status().is_success() {
                if let Ok(bytes) = resp.bytes().await {
                    return Response::builder()
                        .status(StatusCode::OK)
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(axum::body::Body::from(bytes))
                        .unwrap();
                }
            }
        }
    }
    // No shard reachable — return minimal synthetic stats so callers don't fail hard.
    Response::builder()
        .status(StatusCode::SERVICE_UNAVAILABLE)
        .header(header::CONTENT_TYPE, "application/json")
        .body(axum::body::Body::from(r#"{"error":"no shard reachable"}"#))
        .unwrap()
}

// ── Main ───────────────────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Accept both `larql-router <args>` and `larql-router route <args>`.
    let args: Vec<String> = std::env::args().collect();
    let filtered: Vec<String> = if args.len() > 1 && args[1] == "route" {
        std::iter::once(args[0].clone())
            .chain(args[2..].iter().cloned())
            .collect()
    } else {
        args
    };
    let cli = Cli::parse_from(filtered);

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&cli.log_level)),
        )
        .init();

    info!("larql-router v{}", env!("CARGO_PKG_VERSION"));

    if cli.shards.is_none() && cli.grid_port.is_none() {
        eprintln!("error: must provide --shards or --grid-port (or both)");
        std::process::exit(1);
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(cli.timeout_secs))
        .tcp_keepalive(std::time::Duration::from_secs(30))
        .pool_idle_timeout(std::time::Duration::from_secs(90))
        .pool_max_idle_per_host(16)
        .build()?;

    let static_shards = if let Some(spec) = &cli.shards {
        let shards = parse_shards(spec).map_err(|e| format!("--shards: {e}"))?;
        info!("Static shard map:");
        for shard in &shards {
            let status_url = format!("{}/v1/stats", shard.url);
            let healthy = client
                .get(&status_url)
                .send()
                .await
                .map(|r| r.status().is_success())
                .unwrap_or(false);
            let marker = if healthy { "✓" } else { "✗ UNREACHABLE" };
            info!(
                "  layers {}-{}: {}  {}",
                shard.layer_start,
                shard.layer_end - 1,
                shard.url,
                marker
            );
            if !healthy {
                warn!("  Shard {} is not reachable", shard.url);
            }
        }
        shards
    } else {
        Vec::new()
    };

    let grid_state: Option<Arc<RwLock<GridState>>> = if cli.grid_port.is_some() {
        Some(Arc::new(RwLock::new(GridState::default())))
    } else {
        None
    };

    if let (Some(grid_port), Some(state)) = (cli.grid_port, &grid_state) {
        let svc = GridServiceServer::new(GridServiceImpl::new_with_key(
            state.clone(),
            cli.grid_key.clone(),
        ));
        let grpc_addr: SocketAddr = format!("{}:{}", cli.host, grid_port).parse()?;
        info!("Grid gRPC server listening: {grpc_addr}");
        tokio::spawn(async move {
            if let Err(e) = GrpcServer::builder()
                .add_service(svc)
                .serve(grpc_addr)
                .await
            {
                tracing::error!("gRPC server error: {e}");
            }
        });

        // GT6: spawn dynamic rebalancer (disabled when interval == 0).
        if cli.rebalance_interval > 0 {
            let rebalance_cfg = rebalancer::RebalancerConfig::from_cli(
                cli.rebalance_interval,
                cli.rebalance_threshold,
            );
            info!(
                interval_s = cli.rebalance_interval,
                threshold = cli.rebalance_threshold,
                "Rebalancer: enabled"
            );
            rebalancer::spawn(state.clone(), rebalance_cfg);
        }
    }

    let state = Arc::new(AppState {
        static_shards,
        grid: grid_state,
        client,
    });

    let app = Router::new()
        .route("/v1/walk-ffn", post(handle_walk_ffn))
        .route("/v1/stats", axum::routing::get(handle_stats))
        .route("/v1/health", axum::routing::get(handle_health))
        .with_state(state);

    let addr = format!("{}:{}", cli.host, cli.port);
    info!("HTTP listening: http://{}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}

// Tests live in crates/larql-router/src/routing.rs (the library module).
