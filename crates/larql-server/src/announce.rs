//! Grid announce task — keeps a persistent gRPC stream to the router.
//!
//! On startup, if --join is provided, this module spawns a background task
//! that connects to the router, sends an AnnounceMsg, and then sends
//! Heartbeats every 10 seconds. On disconnect it reconnects with backoff.

use std::time::Duration;

use larql_router_protocol::{
    AnnounceMsg, DroppingMsg, GridServiceClient, HeartbeatMsg, RouterPayload, ServerMessage,
    ServerPayload,
};
use tokio_stream::StreamExt;
use tracing::{error, info, warn};

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
}

// ── Public entry point ─────────────────────────────────────────────────────────

/// Spawn a background task that keeps the grid connection alive.
/// Returns immediately; the task runs for the process lifetime.
pub fn run_announce(config: AnnounceConfig) {
    tokio::spawn(async move {
        let mut backoff = Duration::from_secs(1);
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
                    backoff = Duration::from_secs(1);
                }
                Err(e) => {
                    warn!("Grid stream error: {e} — retrying in {}s", backoff.as_secs());
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(Duration::from_secs(60));
                }
            }
        }
    });
}

// ── Single connection lifecycle ────────────────────────────────────────────────

async fn try_once(cfg: &AnnounceConfig) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut client = GridServiceClient::connect(cfg.join_url.clone()).await?;

    // Channel for messages we send to the router.
    let (tx, rx) = tokio::sync::mpsc::channel::<ServerMessage>(32);
    let outbound = tokio_stream::wrappers::ReceiverStream::new(rx);

    let response = client.join(outbound).await?;
    let mut inbound = response.into_inner();

    // Send the announce message immediately.
    tx.send(ServerMessage {
        payload: Some(ServerPayload::Announce(AnnounceMsg {
            model_id: cfg.model_id.clone(),
            layer_start: cfg.layer_start,
            layer_end: cfg.layer_end,
            ram_bytes: cfg.ram_bytes,
            listen_url: cfg.listen_url.clone(),
            vindex_hash: String::new(),
        })),
    })
    .await?;

    // Spawn the heartbeat sender.
    let tx_hb = tx.clone();
    let hb_handle = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(10));
        loop {
            interval.tick().await;
            let msg = ServerMessage {
                payload: Some(ServerPayload::Heartbeat(HeartbeatMsg {
                    cpu_pct: 0.0,
                    ram_used: 0,
                    requests_in_flight: 0,
                })),
            };
            if tx_hb.send(msg).await.is_err() {
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
                        "Router unassigned shard"
                    );
                    // Send dropping notice then let the stream close.
                    let _ = tx
                        .send(ServerMessage {
                            payload: Some(ServerPayload::Dropping(DroppingMsg {
                                model_id: u.model_id.clone(),
                                layer_start: u.layer_start,
                                layer_end: u.layer_end,
                                reason: "reassigned".into(),
                            })),
                        })
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
