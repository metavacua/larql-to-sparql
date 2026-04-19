//! larql-router — transparent layer-sharding proxy for larql-server.
//!
//! Sits between clients and a set of layer-sharded larql-server instances.
//! The client sees one endpoint and uses RemoteWalkBackend unchanged.
//! The router owns the sharding map and routes each layer to the right server.
//!
//! Usage:
//!   larql-router --shards "0-9=http://host-a:8080,10-19=http://host-b:8081" --port 9090
//!
//! Batched requests (layers=[0,1,...]) are split by shard, fanned out in
//! parallel, and merged before returning. Single-layer requests are proxied
//! directly.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::post;
use axum::{Json, Router};
use clap::Parser;
use serde_json::Value;
use tracing::{info, warn};

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "larql-router", version, about = "Layer-sharding proxy for larql-server")]
struct Cli {
    /// Shard map: comma-separated "START-END=URL" entries (inclusive bounds).
    /// Example: "0-16=http://host-a:8080,17-33=http://host-b:8081"
    #[arg(long)]
    shards: String,

    /// Listen port.
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
}

// ── Shard map ─────────────────────────────────────────────────────────────────

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

/// Parse "START-END=URL" (inclusive bounds → exclusive end internally).
fn parse_shards(spec: &str) -> Result<Vec<Shard>, String> {
    let mut shards = Vec::new();
    for entry in spec.split(',') {
        let entry = entry.trim();
        if entry.is_empty() { continue; }
        let (range, url) = entry.split_once('=')
            .ok_or_else(|| format!("expected 'START-END=URL', got '{entry}'"))?;
        let (start_s, end_s) = range.split_once('-')
            .ok_or_else(|| format!("expected 'START-END', got '{range}'"))?;
        let start: usize = start_s.trim().parse()
            .map_err(|_| format!("invalid start '{start_s}'"))?;
        let end: usize = end_s.trim().parse()
            .map_err(|_| format!("invalid end '{end_s}'"))?;
        if end < start {
            return Err(format!("end ({end}) must be >= start ({start})"));
        }
        shards.push(Shard { layer_start: start, layer_end: end + 1, url: url.trim().to_string() });
    }
    if shards.is_empty() {
        return Err("no shards specified".into());
    }
    Ok(shards)
}

// ── App state ─────────────────────────────────────────────────────────────────

struct AppState {
    shards: Vec<Shard>,
    client: reqwest::Client,
}

impl AppState {
    fn find_shard(&self, layer: usize) -> Option<&Shard> {
        self.shards.iter().find(|s| s.owns(layer))
    }
}

// ── Route handler ─────────────────────────────────────────────────────────────

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
        return Err((StatusCode::BAD_REQUEST, "must provide 'layer' or 'layers'".into()));
    };

    if layers.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "empty layer list".into()));
    }

    // Validate all layers have an owner before dispatching anything.
    for &layer in &layers {
        if state.find_shard(layer).is_none() {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("layer {layer} has no owning shard in this router"),
            ));
        }
    }

    // Single layer: proxy the body unchanged.
    if layers.len() == 1 {
        let shard = state.find_shard(layers[0]).unwrap();
        return proxy_to_shard(&state.client, shard, body).await;
    }

    // Batched: group layers by shard, fan out in parallel, merge.
    let mut shard_layers: HashMap<String, Vec<usize>> = HashMap::new();
    for &layer in &layers {
        let shard = state.find_shard(layer).unwrap();
        shard_layers.entry(shard.url.clone()).or_default().push(layer);
    }

    // Build per-shard sub-requests and dispatch in parallel.
    let mut handles = Vec::new();
    for (url, shard_layer_list) in &shard_layers {
        let mut sub_body = body.clone();
        if shard_layer_list.len() == 1 {
            sub_body["layer"] = Value::from(shard_layer_list[0]);
            sub_body.as_object_mut().unwrap().remove("layers");
        } else {
            sub_body["layers"] = Value::Array(
                shard_layer_list.iter().map(|&l| Value::from(l)).collect()
            );
            sub_body.as_object_mut().unwrap().remove("layer");
        }
        let client = state.client.clone();
        let url = format!("{url}/v1/walk-ffn");
        handles.push(tokio::spawn(async move {
            client.post(&url)
                .json(&sub_body)
                .send().await
                .map_err(|e| e.to_string())?
                .json::<Value>().await
                .map_err(|e| e.to_string())
        }));
    }

    let responses: Vec<Value> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|jh| jh.map_err(|e| e.to_string()).and_then(|r| r))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("shard error: {e}")))?;

    // Merge results arrays into a single ordered array.
    // Each shard may return {"results": [...]} or a single {"layer": N, ...}.
    let mut all_results: Vec<Value> = Vec::new();
    let mut total_latency: f64 = 0.0;
    for resp in responses {
        if let Some(arr) = resp.get("results").and_then(|v| v.as_array()) {
            all_results.extend(arr.iter().cloned());
        } else if resp.get("layer").is_some() {
            all_results.push(resp.clone());
        }
        if let Some(ms) = resp.get("latency_ms").and_then(|v| v.as_f64()) {
            // Track max latency (parallel shards, wall clock = max not sum).
            if ms > total_latency { total_latency = ms; }
        }
    }

    // Sort by layer to match the original request order.
    all_results.sort_by_key(|r| r.get("layer").and_then(|v| v.as_u64()).unwrap_or(0));

    Ok(Json(serde_json::json!({
        "results": all_results,
        "latency_ms": (total_latency * 10.0).round() / 10.0,
    })))
}

async fn proxy_to_shard(
    client: &reqwest::Client,
    shard: &Shard,
    body: Value,
) -> Result<Json<Value>, (StatusCode, String)> {
    let url = format!("{}/v1/walk-ffn", shard.url);
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("shard {}: {e}", shard.url)))?;

    let status = resp.status();
    let json: Value = resp.json().await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("decode response: {e}")))?;

    if !status.is_success() {
        let msg = json.get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown shard error")
            .to_string();
        return Err((StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY), msg));
    }

    Ok(Json(json))
}

async fn handle_health() -> Json<Value> {
    Json(serde_json::json!({"status": "ok"}))
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Accept both `larql-router <args>` and `larql-router route <args>`.
    let args: Vec<String> = std::env::args().collect();
    let filtered: Vec<String> = if args.len() > 1 && args[1] == "route" {
        std::iter::once(args[0].clone()).chain(args[2..].iter().cloned()).collect()
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

    let shards = parse_shards(&cli.shards)
        .map_err(|e| format!("--shards: {e}"))?;

    // Log shard map and health-check each server.
    info!("Shard map:");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(cli.timeout_secs))
        .build()?;

    for shard in &shards {
        let status_url = format!("{}/v1/stats", shard.url);
        let healthy = client.get(&status_url).send().await
            .map(|r| r.status().is_success())
            .unwrap_or(false);
        let marker = if healthy { "✓" } else { "✗ UNREACHABLE" };
        info!(
            "  layers {}-{}: {}  {}",
            shard.layer_start, shard.layer_end - 1, shard.url, marker
        );
        if !healthy {
            warn!("  Shard {} is not reachable — requests to its layers will fail", shard.url);
        }
    }

    let state = Arc::new(AppState { shards, client });

    let app = Router::new()
        .route("/v1/walk-ffn", post(handle_walk_ffn))
        .route("/v1/health", axum::routing::get(handle_health))
        .with_state(state);

    let addr = format!("{}:{}", cli.host, cli.port);
    info!("Listening: http://{}", addr);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
