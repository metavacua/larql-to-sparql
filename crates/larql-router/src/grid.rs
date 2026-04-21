//! Grid state and gRPC service implementation for the self-assembling FFN grid.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use tokio::sync::{mpsc, RwLock};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

use larql_router_protocol::{
    AckMsg, AnnounceMsg, Gap, GridService, ModelCoverage, RejectMsg, RouterMessage,
    RouterPayload, ServerInfo, ServerMessage, ServerPayload, ShardInfo, StatusRequest,
    StatusResponse,
};

// ── Per-server record ─────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ServerEntry {
    pub server_id: String,
    pub listen_url: String,
    pub model_id: String,
    pub layer_start: u32, // inclusive
    pub layer_end: u32,   // inclusive
    pub cpu_pct: f32,
    pub ram_used: u64,
    pub requests_in_flight: u32,
    pub last_seen: Instant,
}

// ── Grid state ────────────────────────────────────────────────────────────────

#[derive(Default)]
pub struct GridState {
    servers: HashMap<String, ServerEntry>,
    // Pre-built: (model_id, layer) → server_ids; rebuilt only on topology change.
    route_table: HashMap<(String, u32), Vec<String>>,
    // Pre-built: layer → server_ids for model_id=None (single-model) queries.
    any_model_table: HashMap<u32, Vec<String>>,
}

impl GridState {
    pub fn register(&mut self, entry: ServerEntry) {
        tracing::info!(
            server_id = %entry.server_id,
            listen_url = %entry.listen_url,
            model_id = %entry.model_id,
            layers = %format!("{}-{}", entry.layer_start, entry.layer_end),
            "Grid: server joined"
        );
        self.servers.insert(entry.server_id.clone(), entry);
        self.rebuild_route_table();
        self.log_coverage();
    }

    pub fn deregister(&mut self, server_id: &str) {
        if let Some(entry) = self.servers.remove(server_id) {
            tracing::info!(
                server_id = %server_id,
                model_id = %entry.model_id,
                layers = %format!("{}-{}", entry.layer_start, entry.layer_end),
                "Grid: server left"
            );
            self.rebuild_route_table();
            self.log_coverage();
        }
    }

    pub fn update_heartbeat(
        &mut self,
        server_id: &str,
        cpu_pct: f32,
        ram_used: u64,
        requests_in_flight: u32,
    ) {
        if let Some(entry) = self.servers.get_mut(server_id) {
            entry.cpu_pct = cpu_pct;
            entry.ram_used = ram_used;
            entry.requests_in_flight = requests_in_flight;
            entry.last_seen = Instant::now();
        }
        // Heartbeats don't change topology — no table rebuild needed.
    }

    /// Route one layer. O(1) table lookup + O(replicas) least-loaded scan.
    pub fn route(&self, model_id: Option<&str>, layer: u32) -> Option<String> {
        let ids = match model_id {
            Some(m) => self.route_table.get(&(m.to_owned(), layer)),
            None => self.any_model_table.get(&layer),
        };
        ids.and_then(|server_ids| {
            server_ids
                .iter()
                .filter_map(|id| self.servers.get(id))
                .min_by_key(|s| s.requests_in_flight)
                .map(|s| s.listen_url.clone())
        })
    }

    /// Resolve all layers in one call — one lock acquisition covers the whole batch.
    /// Returns Ok(layer → url) or Err(first layer with no owning shard).
    pub fn route_all(
        &self,
        model_id: Option<&str>,
        layers: &[usize],
    ) -> Result<HashMap<usize, String>, usize> {
        let mut out = HashMap::with_capacity(layers.len());
        for &layer in layers {
            match self.route(model_id, layer as u32) {
                Some(url) => { out.insert(layer, url); }
                None => return Err(layer),
            }
        }
        Ok(out)
    }

    /// Rebuild layer→servers index. Called only on join/leave (cold path).
    fn rebuild_route_table(&mut self) {
        let mut rt: HashMap<(String, u32), Vec<String>> = HashMap::new();
        let mut any: HashMap<u32, Vec<String>> = HashMap::new();
        for entry in self.servers.values() {
            for layer in entry.layer_start..=entry.layer_end {
                rt.entry((entry.model_id.clone(), layer))
                    .or_default()
                    .push(entry.server_id.clone());
                any.entry(layer).or_default().push(entry.server_id.clone());
            }
        }
        self.route_table = rt;
        self.any_model_table = any;
    }

    fn log_coverage(&self) {
        // Group by model_id
        let mut by_model: HashMap<&str, Vec<&ServerEntry>> = HashMap::new();
        for entry in self.servers.values() {
            by_model.entry(&entry.model_id).or_default().push(entry);
        }
        for (model_id, entries) in &by_model {
            let layer_count: u32 = entries.iter().map(|e| e.layer_end - e.layer_start + 1).sum();
            tracing::info!(
                model_id = model_id,
                servers = entries.len(),
                total_layers_covered = layer_count,
                "Grid coverage updated"
            );
        }
    }

    pub fn status_response(&self) -> StatusResponse {
        // Build per-model coverage
        let mut by_model: HashMap<String, Vec<&ServerEntry>> = HashMap::new();
        for entry in self.servers.values() {
            by_model.entry(entry.model_id.clone()).or_default().push(entry);
        }

        let models: Vec<ModelCoverage> = by_model
            .iter()
            .map(|(model_id, entries)| {
                let mut shards: Vec<ShardInfo> = entries
                    .iter()
                    .map(|e| ShardInfo {
                        layer_start: e.layer_start,
                        layer_end: e.layer_end,
                        server_ids: vec![e.server_id.clone()],
                        replica_count: 1,
                    })
                    .collect();
                shards.sort_by_key(|s| s.layer_start);

                // Find gaps
                let mut gaps: Vec<Gap> = Vec::new();
                let mut prev_end: Option<u32> = None;
                for shard in &shards {
                    if let Some(end) = prev_end {
                        if shard.layer_start > end + 1 {
                            gaps.push(Gap {
                                layer_start: end + 1,
                                layer_end: shard.layer_start - 1,
                            });
                        }
                    }
                    prev_end = Some(shard.layer_end);
                }

                ModelCoverage {
                    model_id: model_id.clone(),
                    num_layers: 0, // not known to router without vindex
                    shards,
                    gaps,
                }
            })
            .collect();

        let servers: Vec<ServerInfo> = self
            .servers
            .values()
            .map(|e| ServerInfo {
                server_id: e.server_id.clone(),
                listen_url: e.listen_url.clone(),
                state: "serving".into(),
                model_id: e.model_id.clone(),
                layer_start: e.layer_start,
                layer_end: e.layer_end,
                cpu_pct: e.cpu_pct,
                ram_used: e.ram_used,
                requests_in_flight: e.requests_in_flight,
                rtt_ms: 0,
            })
            .collect();

        StatusResponse { models, servers }
    }
}

// ── gRPC service impl ─────────────────────────────────────────────────────────

pub struct GridServiceImpl {
    pub state: Arc<RwLock<GridState>>,
    next_id: AtomicU64,
    /// If set, every incoming Join stream must present "Authorization: Bearer <key>".
    grid_key: Option<String>,
}

impl GridServiceImpl {
    pub fn new(state: Arc<RwLock<GridState>>) -> Self {
        Self { state, next_id: AtomicU64::new(1), grid_key: None }
    }

    pub fn new_with_key(state: Arc<RwLock<GridState>>, key: Option<String>) -> Self {
        Self { state, next_id: AtomicU64::new(1), grid_key: key }
    }

    fn alloc_server_id(&self) -> String {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let n = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("srv-{ts}-{n}")
    }
}

type JoinStream = Pin<Box<dyn futures_core::Stream<Item = Result<RouterMessage, Status>> + Send>>;

#[tonic::async_trait]
impl GridService for GridServiceImpl {
    type JoinStream = JoinStream;

    async fn join(
        &self,
        request: Request<Streaming<ServerMessage>>,
    ) -> Result<Response<Self::JoinStream>, Status> {
        // Auth check — reject streams that don't carry the correct grid key.
        if let Some(expected) = &self.grid_key {
            let token = request
                .metadata()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "));
            if token.map(|t| t != expected).unwrap_or(true) {
                return Err(Status::unauthenticated("invalid grid key"));
            }
        }

        let state = self.state.clone();
        let server_id = self.alloc_server_id();
        let (tx, rx) = mpsc::channel::<Result<RouterMessage, Status>>(32);
        let mut inbound = request.into_inner();

        let sid = server_id.clone();
        tokio::spawn(async move {
            let mut registered_model: Option<(String, u32, u32)> = None; // (model_id, start, end)

            while let Some(msg) = inbound.next().await {
                match msg {
                    Err(e) => {
                        tracing::warn!(server_id = %sid, "Stream error: {e}");
                        break;
                    }
                    Ok(ServerMessage { payload: None }) => {}
                    Ok(ServerMessage { payload: Some(p) }) => match p {
                        ServerPayload::Announce(AnnounceMsg {
                            model_id,
                            layer_start,
                            layer_end,
                            ram_bytes,
                            listen_url,
                            ..
                        }) => {
                            let entry = ServerEntry {
                                server_id: sid.clone(),
                                listen_url: listen_url.clone(),
                                model_id: model_id.clone(),
                                layer_start,
                                layer_end,
                                cpu_pct: 0.0,
                                ram_used: ram_bytes,
                                requests_in_flight: 0,
                                last_seen: Instant::now(),
                            };
                            state.write().await.register(entry);
                            registered_model = Some((model_id, layer_start, layer_end));

                            let ack = RouterMessage {
                                payload: Some(RouterPayload::Ack(AckMsg {
                                    server_id: sid.clone(),
                                })),
                            };
                            if tx.send(Ok(ack)).await.is_err() {
                                break;
                            }
                        }

                        ServerPayload::Heartbeat(hb) => {
                            state.write().await.update_heartbeat(
                                &sid,
                                hb.cpu_pct,
                                hb.ram_used,
                                hb.requests_in_flight,
                            );
                        }

                        ServerPayload::Dropping(d) => {
                            tracing::info!(
                                server_id = %sid,
                                model_id = %d.model_id,
                                layers = %format!("{}-{}", d.layer_start, d.layer_end),
                                reason = %d.reason,
                                "Server dropping shard"
                            );
                            state.write().await.deregister(&sid);
                            registered_model = None;
                        }

                        ServerPayload::Available(_) => {
                            // Phase 2: Mode B assignment
                            tracing::info!(server_id = %sid, "Server is available (Mode B — not yet implemented)");
                            let reject = RouterMessage {
                                payload: Some(RouterPayload::Reject(RejectMsg {
                                    reason: "available mode not yet implemented".into(),
                                })),
                            };
                            let _ = tx.send(Ok(reject)).await;
                        }

                        ServerPayload::Ready(_) | ServerPayload::Refuse(_) => {
                            tracing::debug!(server_id = %sid, "Ignored message (not in assignment flow)");
                        }
                    },
                }
            }

            // Stream closed — clean up
            if registered_model.is_some() {
                state.write().await.deregister(&sid);
            }
            tracing::info!(server_id = %sid, "Connection closed");
        });

        let stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream)))
    }

    async fn status(
        &self,
        _request: Request<StatusRequest>,
    ) -> Result<Response<StatusResponse>, Status> {
        let resp = self.state.read().await.status_response();
        Ok(Response::new(resp))
    }
}
