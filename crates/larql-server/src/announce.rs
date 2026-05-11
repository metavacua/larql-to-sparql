//! Grid announce task — keeps a persistent gRPC stream to the router.
//!
//! On startup, if --join is provided, this module spawns a background task
//! that connects to the router, sends an AnnounceMsg, and then sends
//! Heartbeats every 10 seconds. On disconnect it reconnects with backoff.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Duration;

use std::sync::Arc;

use larql_router_protocol::{
    AnnounceMsg, DroppingMsg, GridServiceClient, HeartbeatMsg, RouterPayload, ServerMessage,
    ServerPayload,
};

use crate::metrics::LayerLatencyTracker;
use tokio_stream::StreamExt;
use tonic::metadata::AsciiMetadataValue;
use tracing::{error, info, warn};

// ── Tunables ───────────────────────────────────────────────────────────────────

const RECONNECT_INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const RECONNECT_MAX_BACKOFF: Duration = Duration::from_secs(60);
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(10);
/// Maximum time to wait for in-flight requests to drain before sending DroppingMsg (GT6).
const DRAIN_TIMEOUT: Duration = Duration::from_secs(30);

// ── Config ─────────────────────────────────────────────────────────────────────

pub struct AnnounceConfig {
    /// gRPC endpoint of the router, e.g. "http://router:50052".
    pub join_url: String,
    /// Model identifier, e.g. "gemma3-4b-q4k".
    pub model_id: String,
    /// First owned layer (inclusive).
    pub layer_start: u32,
    /// Last owned layer (inclusive).
    pub layer_end: u32,
    /// URL clients should use to send requests here, e.g. "http://host:8080".
    pub listen_url: String,
    /// Approximate resident RAM for this shard in bytes.
    pub ram_bytes: u64,
    /// Shared secret that the router expects. None = open grid (dev only).
    pub grid_key: Option<String>,
    /// Stable identity hash of the vindex (model_id + num_layers).
    pub vindex_hash: String,
    /// Per-layer latency tracker — populates HeartbeatMsg.layer_stats.
    pub latency_tracker: Arc<LayerLatencyTracker>,
    /// Active request counter — used for drain (GT6) and heartbeat.requests_in_flight.
    pub requests_in_flight: Arc<std::sync::atomic::AtomicU32>,
}

// ── Mode B config ──────────────────────────────────────────────────────────────

/// Config for Mode B announce: server has capacity but no shard loaded.
pub struct AvailableConfig {
    pub join_url: String,
    pub listen_url: String,
    pub ram_bytes: u64,
    pub disk_bytes: u64,
    pub store_path: String,
    pub grid_key: Option<String>,
}

// ── Public entry points ────────────────────────────────────────────────────────

/// Spawn a background task that keeps the grid connection alive.
/// Returns immediately; the task runs for the process lifetime.
pub fn run_announce(config: AnnounceConfig) {
    tokio::spawn(async move {
        let mut backoff = RECONNECT_INITIAL_BACKOFF;
        loop {
            info!(
                join_url = %config.join_url,
                model_id = %config.model_id,
                layers = %format!("{}-{}", config.layer_start, config.layer_end),
                "Connecting to router grid..."
            );
            match try_once(&config).await {
                Ok(()) => {
                    info!("Grid stream closed cleanly — reconnecting");
                    backoff = RECONNECT_INITIAL_BACKOFF;
                }
                Err(e) => {
                    warn!(
                        "Grid stream error: {e} — retrying in {}s",
                        backoff.as_secs()
                    );
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(RECONNECT_MAX_BACKOFF);
                }
            }
        }
    });
}

/// Spawn a Mode B background task: advertise available capacity, wait for
/// `AssignMsg`, download the assigned shard, send `ReadyMsg`, then
/// re-enter Mode A serving loop.
///
/// Returns immediately; the task runs for the process lifetime.
pub fn run_announce_available(config: AvailableConfig) {
    tokio::spawn(async move {
        let mut backoff = RECONNECT_INITIAL_BACKOFF;
        loop {
            info!(
                join_url = %config.join_url,
                ram_gb = config.ram_bytes / (1024 * 1024 * 1024),
                "Connecting to router grid (Mode B — available)..."
            );
            match try_once_available(&config).await {
                Ok(()) => {
                    info!("Mode B stream closed cleanly — reconnecting");
                    backoff = RECONNECT_INITIAL_BACKOFF;
                }
                Err(e) => {
                    warn!(
                        "Mode B stream error: {e} — retrying in {}s",
                        backoff.as_secs()
                    );
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(RECONNECT_MAX_BACKOFF);
                }
            }
        }
    });
}

/// Stable hash of the vindex identity (not a security primitive — for version checks).
pub fn vindex_identity_hash(model_id: &str, num_layers: usize) -> String {
    let mut h = DefaultHasher::new();
    model_id.hash(&mut h);
    num_layers.hash(&mut h);
    format!("{:016x}", h.finish())
}

fn grid_bearer_value(
    grid_key: Option<&str>,
) -> Result<Option<AsciiMetadataValue>, Box<dyn std::error::Error + Send + Sync>> {
    grid_key
        .map(|k| format!("Bearer {k}").parse())
        .transpose()
        .map_err(Into::into)
}

fn announce_message(cfg: &AnnounceConfig) -> ServerMessage {
    ServerMessage {
        payload: Some(ServerPayload::Announce(AnnounceMsg {
            model_id: cfg.model_id.clone(),
            layer_start: cfg.layer_start,
            layer_end: cfg.layer_end,
            ram_bytes: cfg.ram_bytes,
            listen_url: cfg.listen_url.clone(),
            vindex_hash: cfg.vindex_hash.clone(),
        })),
    }
}

fn heartbeat_message(
    tracker: &LayerLatencyTracker,
    requests_in_flight: &std::sync::atomic::AtomicU32,
) -> ServerMessage {
    use std::sync::atomic::Ordering;
    ServerMessage {
        payload: Some(ServerPayload::Heartbeat(HeartbeatMsg {
            cpu_pct: 0.0,
            ram_used: 0,
            requests_in_flight: requests_in_flight.load(Ordering::Relaxed),
            layer_stats: tracker.snapshot(),
        })),
    }
}

fn dropping_message(model_id: String, layer_start: u32, layer_end: u32) -> ServerMessage {
    ServerMessage {
        payload: Some(ServerPayload::Dropping(DroppingMsg {
            model_id,
            layer_start,
            layer_end,
            reason: "reassigned".into(),
        })),
    }
}

/// Wait until `counter` reaches zero or `timeout` expires.
/// Polls every 100 ms. Used by GT6 drain to ensure no requests are
/// mid-flight before sending DroppingMsg.
async fn drain_requests(counter: &std::sync::atomic::AtomicU32, timeout: Duration) {
    use std::sync::atomic::Ordering;
    let deadline = tokio::time::Instant::now() + timeout;
    let mut interval = tokio::time::interval(std::time::Duration::from_millis(100));
    loop {
        if counter.load(Ordering::Acquire) == 0 {
            return;
        }
        if tokio::time::Instant::now() >= deadline {
            warn!(
                "drain timeout ({:.0}s) elapsed with {} requests still in flight",
                timeout.as_secs_f64(),
                counter.load(Ordering::Relaxed)
            );
            return;
        }
        interval.tick().await;
    }
}

// ── Single connection lifecycle ────────────────────────────────────────────────

async fn try_once(cfg: &AnnounceConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let channel = tonic::transport::Channel::from_shared(cfg.join_url.clone())?
        .connect()
        .await?;

    // Inject the grid key into every outgoing RPC as "Authorization: Bearer <key>".
    let bearer = grid_bearer_value(cfg.grid_key.as_deref())?;
    let mut client =
        GridServiceClient::with_interceptor(channel, move |mut req: tonic::Request<()>| {
            if let Some(val) = &bearer {
                req.metadata_mut().insert("authorization", val.clone());
            }
            Ok(req)
        });

    // Channel for messages we send to the router.
    let (tx, rx) = tokio::sync::mpsc::channel::<ServerMessage>(32);
    let outbound = tokio_stream::wrappers::ReceiverStream::new(rx);

    let response = client.join(outbound).await?;
    let mut inbound = response.into_inner();

    // Send the announce message immediately.
    tx.send(announce_message(cfg)).await?;

    // Spawn the heartbeat sender.
    let tx_hb = tx.clone();
    let tracker = cfg.latency_tracker.clone();
    let rif = cfg.requests_in_flight.clone();
    let hb_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(HEARTBEAT_INTERVAL);
        loop {
            interval.tick().await;
            if tx_hb.send(heartbeat_message(&tracker, &rif)).await.is_err() {
                break;
            }
        }
    });

    // Process incoming router messages.
    while let Some(msg) = inbound.next().await {
        match msg {
            Err(e) => {
                hb_handle.abort();
                return Err(e.into());
            }
            Ok(rm) => match rm.payload {
                Some(RouterPayload::Ack(ack)) => {
                    info!(
                        server_id = %ack.server_id,
                        model_id = %cfg.model_id,
                        layers = %format!("{}-{}", cfg.layer_start, cfg.layer_end),
                        "Registered with router. Serving."
                    );
                }
                Some(RouterPayload::Reject(r)) => {
                    error!(reason = %r.reason, "Router rejected registration");
                    hb_handle.abort();
                    return Err(format!("router rejected: {}", r.reason).into());
                }
                Some(RouterPayload::Assign(_)) => {
                    warn!("Received AssignMsg but Mode B not implemented — ignoring");
                }
                Some(RouterPayload::Unassign(u)) => {
                    info!(
                        model_id = %u.model_id,
                        layers = %format!("{}-{}", u.layer_start, u.layer_end),
                        reason = %u.reason,
                        "Router unassigned shard — draining in-flight requests…"
                    );
                    // GT6 drain: wait up to DRAIN_TIMEOUT for active requests
                    // to finish before sending DroppingMsg.
                    drain_requests(&cfg.requests_in_flight, DRAIN_TIMEOUT).await;
                    // Send dropping notice then let the stream close.
                    let _ = tx
                        .send(dropping_message(
                            u.model_id.clone(),
                            u.layer_start,
                            u.layer_end,
                        ))
                        .await;
                    break;
                }
                None => {}
            },
        }
    }

    hb_handle.abort();
    Ok(())
}

// ── Mode B connection lifecycle ────────────────────────────────────────────────

async fn try_once_available(
    cfg: &AvailableConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use larql_router_protocol::{AvailableMsg, ReadyMsg};

    let channel = tonic::transport::Channel::from_shared(cfg.join_url.clone())?
        .connect()
        .await?;

    let bearer = grid_bearer_value(cfg.grid_key.as_deref())?;
    let mut client =
        GridServiceClient::with_interceptor(channel, move |mut req: tonic::Request<()>| {
            if let Some(val) = &bearer {
                req.metadata_mut().insert("authorization", val.clone());
            }
            Ok(req)
        });

    let (tx, rx) = tokio::sync::mpsc::channel::<ServerMessage>(32);
    let outbound = tokio_stream::wrappers::ReceiverStream::new(rx);
    let response = client.join(outbound).await?;
    let mut inbound = response.into_inner();

    // Send AvailableMsg — advertise capacity.
    tx.send(ServerMessage {
        payload: Some(ServerPayload::Available(AvailableMsg {
            ram_bytes: cfg.ram_bytes,
            disk_bytes: cfg.disk_bytes,
            store_path: cfg.store_path.clone(),
        })),
    })
    .await?;

    info!(
        join_url = %cfg.join_url,
        ram_gb = cfg.ram_bytes / (1024 * 1024 * 1024),
        "Mode B: sent AvailableMsg — waiting for assignment…"
    );

    // Wait for AssignMsg from router.
    while let Some(msg) = inbound.next().await {
        match msg {
            Err(e) => return Err(e.into()),
            Ok(rm) => match rm.payload {
                Some(RouterPayload::Assign(assign)) => {
                    info!(
                        model_id = %assign.model_id,
                        layers = %format!("{}-{}", assign.layer_start, assign.layer_end),
                        origin_url = %assign.origin_url,
                        "Mode B: received AssignMsg — downloading shard…"
                    );

                    // Download the assigned shard.
                    match crate::shard_loader::download_and_load_shard(
                        &assign.origin_url,
                        &cfg.store_path,
                        &assign.shard_hash,
                        &assign.model_id,
                        assign.layer_start,
                        assign.layer_end,
                    )
                    .await
                    {
                        Ok(()) => {
                            // Send ReadyMsg — shard is loaded and serving.
                            let _ = tx
                                .send(ServerMessage {
                                    payload: Some(ServerPayload::Ready(ReadyMsg {
                                        model_id: assign.model_id.clone(),
                                        layer_start: assign.layer_start,
                                        layer_end: assign.layer_end,
                                        listen_url: cfg.listen_url.clone(),
                                    })),
                                })
                                .await;
                            info!(
                                model_id = %assign.model_id,
                                "Mode B: shard ready — sent ReadyMsg"
                            );
                        }
                        Err(e) => {
                            use larql_router_protocol::RefuseMsg;
                            warn!("Mode B: shard download failed: {e} — sending RefuseMsg");
                            let _ = tx
                                .send(ServerMessage {
                                    payload: Some(ServerPayload::Refuse(RefuseMsg {
                                        model_id: assign.model_id,
                                        layer_start: assign.layer_start,
                                        layer_end: assign.layer_end,
                                        reason: "download_failed".into(),
                                    })),
                                })
                                .await;
                        }
                    }
                }
                Some(RouterPayload::Ack(ack)) => {
                    info!(server_id = %ack.server_id, "Mode B: registered as serving");
                    break;
                }
                Some(RouterPayload::Reject(r)) => {
                    return Err(format!("Mode B rejected: {}", r.reason).into());
                }
                _ => {}
            },
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> AnnounceConfig {
        AnnounceConfig {
            join_url: "http://router:50052".into(),
            model_id: "gemma-test".into(),
            layer_start: 3,
            layer_end: 7,
            listen_url: "http://server:8080".into(),
            ram_bytes: 42,
            grid_key: Some("secret".into()),
            vindex_hash: "abc123".into(),
            latency_tracker: Arc::new(LayerLatencyTracker::new()),
            requests_in_flight: Arc::new(std::sync::atomic::AtomicU32::new(0)),
        }
    }

    #[test]
    fn vindex_identity_hash_is_stable_and_hex() {
        let a = vindex_identity_hash("model-a", 30);
        let b = vindex_identity_hash("model-a", 30);
        let c = vindex_identity_hash("model-a", 31);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_eq!(a.len(), 16);
        assert!(a.chars().all(|ch| ch.is_ascii_hexdigit()));
    }

    #[test]
    fn grid_bearer_value_formats_authorization() {
        let val = grid_bearer_value(Some("secret")).unwrap().unwrap();
        assert_eq!(val.to_str().unwrap(), "Bearer secret");
        assert!(grid_bearer_value(None).unwrap().is_none());
    }

    #[test]
    fn announce_message_copies_config_fields() {
        let cfg = config();
        let msg = announce_message(&cfg);
        let Some(ServerPayload::Announce(announce)) = msg.payload else {
            panic!("expected announce payload");
        };
        assert_eq!(announce.model_id, "gemma-test");
        assert_eq!(announce.layer_start, 3);
        assert_eq!(announce.layer_end, 7);
        assert_eq!(announce.ram_bytes, 42);
        assert_eq!(announce.listen_url, "http://server:8080");
        assert_eq!(announce.vindex_hash, "abc123");
    }

    #[test]
    fn heartbeat_message_uses_zeroed_metrics() {
        let tracker = LayerLatencyTracker::new();
        let rif = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let msg = heartbeat_message(&tracker, &rif);
        let Some(ServerPayload::Heartbeat(heartbeat)) = msg.payload else {
            panic!("expected heartbeat payload");
        };
        assert_eq!(heartbeat.cpu_pct, 0.0);
        assert_eq!(heartbeat.ram_used, 0);
        assert_eq!(heartbeat.requests_in_flight, 0);
        assert!(heartbeat.layer_stats.is_empty());
    }

    #[test]
    fn heartbeat_includes_layer_stats_after_recording() {
        let tracker = LayerLatencyTracker::new();
        tracker.record(5, 3.0);
        tracker.record(5, 5.0);
        let rif = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let msg = heartbeat_message(&tracker, &rif);
        let Some(ServerPayload::Heartbeat(hb)) = msg.payload else {
            panic!("expected heartbeat");
        };
        assert_eq!(hb.layer_stats.len(), 1);
        assert_eq!(hb.layer_stats[0].layer, 5);
        assert!(hb.layer_stats[0].avg_ms > 0.0);
    }

    #[test]
    fn dropping_message_marks_reassigned() {
        let msg = dropping_message("model".into(), 1, 2);
        let Some(ServerPayload::Dropping(dropping)) = msg.payload else {
            panic!("expected dropping payload");
        };
        assert_eq!(dropping.model_id, "model");
        assert_eq!(dropping.layer_start, 1);
        assert_eq!(dropping.layer_end, 2);
        assert_eq!(dropping.reason, "reassigned");
    }
}
