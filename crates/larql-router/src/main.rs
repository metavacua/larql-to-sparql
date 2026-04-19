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

mod grid;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::post;
use axum::{Json, Router};
use clap::Parser;
use serde_json::Value;
use tokio::sync::RwLock;
use tonic::transport::Server as GrpcServer;
use tracing::{info, warn};

use grid::{GridServiceImpl, GridState};
use larql_router_protocol::GridServiceServer;

// ── CLI ────────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "larql-router", version, about = "Layer-sharding proxy for larql-server")]
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
}

// ── Static shard map ───────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct Shard {
    layer_start: usize, // inclusive
    layer_end: usize,   // exclusive
    url: String,
}

impl Shard {
    fn owns(&self, layer: usize) -> bool {
        layer >= self.layer_start && layer < self.layer_end
    }
}

fn parse_shards(spec: &str) -> Result<Vec<Shard>, String> {
    let mut shards = Vec::new();
    for entry in spec.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let (range, url) = entry
            .split_once('=')
            .ok_or_else(|| format!("expected 'START-END=URL', got '{entry}'"))?;
        let (start_s, end_s) = range
            .split_once('-')
            .ok_or_else(|| format!("expected 'START-END', got '{range}'"))?;
        let start: usize = start_s
            .trim()
            .parse()
            .map_err(|_| format!("invalid start '{start_s}'"))?;
        let end: usize = end_s
            .trim()
            .parse()
            .map_err(|_| format!("invalid end '{end_s}'"))?;
        if end < start {
            return Err(format!("end ({end}) must be >= start ({start})"));
        }
        shards.push(Shard {
            layer_start: start,
            layer_end: end + 1,
            url: url.trim().to_string(),
        });
    }
    if shards.is_empty() {
        return Err("no shards specified".into());
    }
    Ok(shards)
}

// ── App state ──────────────────────────────────────────────────────────────────

struct AppState {
    /// Static shards from --shards (may be empty).
    static_shards: Vec<Shard>,
    /// Grid state from --grid-port (None if grid mode not enabled).
    grid: Option<Arc<RwLock<GridState>>>,
    client: reqwest::Client,
}

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
            // Try grid first; fall through to static shards for any misses.
            let mut out = HashMap::with_capacity(layers.len());
            let mut static_needed: Vec<usize> = Vec::new();
            for &layer in layers {
                match guard.route(model_id, layer as u32) {
                    Some(url) => { out.insert(layer, url); }
                    None => static_needed.push(layer),
                }
            }
            drop(guard); // release grid lock before static scan
            for layer in static_needed {
                match self.static_shards.iter().find(|s| s.owns(layer)) {
                    Some(s) => { out.insert(layer, s.url.clone()); }
                    None => return Err(layer),
                }
            }
            return Ok(out);
        }
        // Grid not enabled — static shards only.
        let mut out = HashMap::with_capacity(layers.len());
        for &layer in layers {
            match self.static_shards.iter().find(|s| s.owns(layer)) {
                Some(s) => { out.insert(layer, s.url.clone()); }
                None => return Err(layer),
            }
        }
        Ok(out)
    }
}

// ── Route handler ──────────────────────────────────────────────────────────────

async fn handle_walk_ffn(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, (StatusCode, String)> {
    // Collect the requested layers.
    let layers: Vec<usize> = if let Some(arr) = body.get("layers").and_then(|v| v.as_array()) {
        arr.iter()
            .filter_map(|v| v.as_u64().map(|n| n as usize))
            .collect()
    } else if let Some(n) = body.get("layer").and_then(|v| v.as_u64()) {
        vec![n as usize]
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            r#"{"error":"must provide 'layer' or 'layers'"}"#.into(),
        ));
    };

    if layers.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            r#"{"error":"empty layer list"}"#.into(),
        ));
    }

    // Optional model_id in the request body (multi-model grids).
    let model_id = body.get("model_id").and_then(|v| v.as_str()).map(str::to_owned);
    let mid = model_id.as_deref();

    // Resolve all layers in one lock acquisition (validates + maps in a single read).
    let layer_urls = state.resolve_all(mid, &layers).await.map_err(|missing| {
        (
            StatusCode::BAD_REQUEST,
            format!(r#"{{"error":"layer {missing} has no owning shard in this router"}}"#),
        )
    })?;

    // Single layer: proxy the body unchanged.
    if layers.len() == 1 {
        return proxy_to(&state.client, &layer_urls[&layers[0]], body).await;
    }

    // Batched: group layers by URL, fan out in parallel, merge.
    let mut by_url: HashMap<String, Vec<usize>> = HashMap::new();
    for (&layer, url) in &layer_urls {
        by_url.entry(url.clone()).or_default().push(layer);
    }

    let mut handles = Vec::new();
    for (url, shard_layers) in &by_url {
        let mut sub_body = body.clone();
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

    Ok(Json(serde_json::json!({
        "results": all_results,
        "latency_ms": (max_latency * 10.0).round() / 10.0,
    })))
}

async fn proxy_to(
    client: &reqwest::Client,
    base_url: &str,
    body: Value,
) -> Result<Json<Value>, (StatusCode, String)> {
    let url = format!("{base_url}/v1/walk-ffn");
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("shard {base_url}: {e}")))?;

    let status = resp.status();
    let json: Value = resp
        .json()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("decode response: {e}")))?;

    if !status.is_success() {
        let msg = json
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown shard error")
            .to_string();
        return Err((
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            msg,
        ));
    }
    Ok(Json(json))
}

async fn handle_health() -> Json<Value> {
    Json(serde_json::json!({"status": "ok"}))
}

// ── Main ───────────────────────────────────────────────────────────────────────

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

    // Must have at least one of --shards or --grid-port.
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

    // Static shards.
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

    // Grid state (shared with gRPC service and HTTP handler).
    let grid_state: Option<Arc<RwLock<GridState>>> = if cli.grid_port.is_some() {
        Some(Arc::new(RwLock::new(GridState::default())))
    } else {
        None
    };

    // Start gRPC grid server if --grid-port is set.
    if let (Some(grid_port), Some(state)) = (cli.grid_port, &grid_state) {
        let svc = GridServiceServer::new(GridServiceImpl::new_with_key(state.clone(), cli.grid_key.clone()));
        let grpc_addr: SocketAddr = format!("{}:{}", cli.host, grid_port).parse()?;
        info!("Grid gRPC server listening: {grpc_addr}");
        tokio::spawn(async move {
            if let Err(e) = GrpcServer::builder().add_service(svc).serve(grpc_addr).await {
                tracing::error!("gRPC server error: {e}");
            }
        });
    }

    let state = Arc::new(AppState {
        static_shards,
        grid: grid_state,
        client,
    });

    let app = Router::new()
        .route("/v1/walk-ffn", post(handle_walk_ffn))
        .route("/v1/health", axum::routing::get(handle_health))
        .with_state(state);

    let addr = format!("{}:{}", cli.host, cli.port);
    info!("HTTP listening: http://{}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
