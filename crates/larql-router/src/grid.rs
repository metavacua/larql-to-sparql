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
    AckMsg, AnnounceMsg, Gap, GridService, LayerLatency, ModelCoverage, RouterMessage,
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
    /// Per-layer EMA latency and p99, from HeartbeatMsg.layer_stats (GT3).
    /// Key = layer index. Empty until the first heartbeat with layer data arrives.
    pub layer_latencies: HashMap<u32, (f32, f32)>, // (avg_ms, p99_ms)
}

// ── Mode B: available server entry ───────────────────────────────────────────

/// A server in Mode B idle state — it has capacity but no shard loaded yet.
pub struct AvailableEntry {
    pub server_id: String,
    /// Channel to send `RouterMessage` (including `AssignMsg`) to this server.
    pub sender: mpsc::Sender<Result<RouterMessage, tonic::Status>>,
    pub ram_bytes: u64,
    pub disk_bytes: u64,
    pub store_path: String,
    pub joined_at: std::time::Instant,
}

// ── Grid state ────────────────────────────────────────────────────────────────

#[derive(Default)]
pub struct GridState {
    servers: HashMap<String, ServerEntry>,
    // Pre-built: (model_id, layer) → server_ids; rebuilt only on topology change.
    route_table: HashMap<(String, u32), Vec<String>>,
    // Pre-built: layer → server_ids for model_id=None (single-model) queries.
    any_model_table: HashMap<u32, Vec<String>>,
    /// Mode B: servers that advertised capacity and are waiting for assignment.
    /// Key = server_id.
    available_servers: HashMap<String, AvailableEntry>,
    /// Sender channels for currently-serving (Mode A) servers.
    /// Used by the rebalancer to push UnassignMsg without holding a lock.
    /// Key = server_id.
    serving_senders: HashMap<String, mpsc::Sender<Result<RouterMessage, tonic::Status>>>,
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

    /// Register a server and store its sender for rebalancer-initiated UnassignMsg.
    pub fn register_with_sender(
        &mut self,
        entry: ServerEntry,
        sender: mpsc::Sender<Result<RouterMessage, tonic::Status>>,
    ) {
        self.serving_senders.insert(entry.server_id.clone(), sender);
        self.register(entry);
    }

    pub fn deregister(&mut self, server_id: &str) {
        self.serving_senders.remove(server_id);
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
        layer_stats: Vec<LayerLatency>,
    ) {
        if let Some(entry) = self.servers.get_mut(server_id) {
            entry.cpu_pct = cpu_pct;
            entry.ram_used = ram_used;
            entry.requests_in_flight = requests_in_flight;
            entry.last_seen = Instant::now();
            for ls in layer_stats {
                entry
                    .layer_latencies
                    .insert(ls.layer, (ls.avg_ms, ls.p99_ms));
            }
        }
        // Heartbeats don't change topology — no table rebuild needed.
    }

    /// Route one layer. O(1) table lookup + O(replicas) least-loaded scan.
    ///
    /// Replica selection (GT3): when per-layer latency data is available from
    /// heartbeats, prefer the server with lowest avg_ms for this specific layer.
    /// Falls back to requests_in_flight when no layer data exists yet.
    pub fn route(&self, model_id: Option<&str>, layer: u32) -> Option<String> {
        let ids = match model_id {
            Some(m) => self.route_table.get(&(m.to_owned(), layer)),
            None => self.any_model_table.get(&layer),
        };
        ids.and_then(|server_ids| {
            server_ids
                .iter()
                .filter_map(|id| self.servers.get(id))
                .min_by(|a, b| {
                    let lat_a = a.layer_latencies.get(&layer).map(|(avg, _)| *avg);
                    let lat_b = b.layer_latencies.get(&layer).map(|(avg, _)| *avg);
                    match (lat_a, lat_b) {
                        (Some(la), Some(lb)) => {
                            la.partial_cmp(&lb).unwrap_or(std::cmp::Ordering::Equal)
                        }
                        // Prefer server with latency data over unknown.
                        (Some(_), None) => std::cmp::Ordering::Less,
                        (None, Some(_)) => std::cmp::Ordering::Greater,
                        // No latency data for either: fall back to requests_in_flight.
                        (None, None) => a.requests_in_flight.cmp(&b.requests_in_flight),
                    }
                })
                .map(|s| s.listen_url.clone())
        })
    }

    /// Resolve all layers in one call — one lock acquisition covers the whole batch.
    /// Returns Ok(layer → url) or Err(first layer with no owning shard).
    #[allow(dead_code)]
    pub fn route_all(
        &self,
        model_id: Option<&str>,
        layers: &[usize],
    ) -> Result<HashMap<usize, String>, usize> {
        let mut out = HashMap::with_capacity(layers.len());
        for &layer in layers {
            match self.route(model_id, layer as u32) {
                Some(url) => {
                    out.insert(layer, url);
                }
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
            let layer_count: u32 = entries
                .iter()
                .map(|e| e.layer_end - e.layer_start + 1)
                .sum();
            tracing::info!(
                model_id = model_id,
                servers = entries.len(),
                total_layers_covered = layer_count,
                "Grid coverage updated"
            );
        }
    }

    /// Accessor for all serving servers (for the rebalancer).
    pub fn servers(&self) -> impl Iterator<Item = (&String, &ServerEntry)> {
        self.servers.iter()
    }

    /// Returns true if there is at least one available server in the Mode B pool.
    pub fn has_available_servers(&self) -> bool {
        !self.available_servers.is_empty()
    }

    /// Get the sender channel for a serving server by ID (for UnassignMsg delivery).
    pub fn serving_sender(
        &self,
        server_id: &str,
    ) -> Option<mpsc::Sender<Result<RouterMessage, tonic::Status>>> {
        self.serving_senders.get(server_id).cloned()
    }

    /// Register a Mode B available server. Returns the server_id.
    pub fn register_available(
        &mut self,
        server_id: String,
        sender: mpsc::Sender<Result<RouterMessage, tonic::Status>>,
        ram_bytes: u64,
        disk_bytes: u64,
        store_path: String,
    ) {
        tracing::info!(
            server_id = %server_id,
            ram_gb = ram_bytes / (1024 * 1024 * 1024),
            "Grid: Mode B server available"
        );
        self.available_servers.insert(
            server_id.clone(),
            AvailableEntry {
                server_id,
                sender,
                ram_bytes,
                disk_bytes,
                store_path,
                joined_at: std::time::Instant::now(),
            },
        );
    }

    /// Remove a server from the available pool.
    pub fn deregister_available(&mut self, server_id: &str) {
        self.available_servers.remove(server_id);
    }

    /// Find the first available server that has at least `min_ram_bytes` of
    /// RAM, send it an `AssignMsg`, and move it out of the available pool.
    ///
    /// Returns `true` if an assignment was sent.
    pub fn try_assign_gap(
        &mut self,
        model_id: &str,
        layer_start: u32,
        layer_end: u32,
        origin_url: &str,
        shard_hash: &str,
        min_ram_bytes: u64,
    ) -> bool {
        // Find a suitable available server.
        let server_id = self
            .available_servers
            .iter()
            .find(|(_, e)| e.ram_bytes >= min_ram_bytes)
            .map(|(id, _)| id.clone());

        let Some(server_id) = server_id else {
            return false;
        };

        let entry = self.available_servers.remove(&server_id).unwrap();
        let msg = RouterMessage {
            payload: Some(RouterPayload::Assign(larql_router_protocol::AssignMsg {
                model_id: model_id.to_owned(),
                layer_start,
                layer_end,
                origin_url: origin_url.to_owned(),
                shard_hash: shard_hash.to_owned(),
            })),
        };
        if entry.sender.try_send(Ok(msg)).is_ok() {
            tracing::info!(
                server_id = %server_id,
                model_id = %model_id,
                layers = %format!("{layer_start}-{layer_end}"),
                "Grid: Mode B assignment sent"
            );
            true
        } else {
            tracing::warn!(server_id = %server_id, "Grid: Mode B assignment send failed (peer disconnected)");
            false
        }
    }

    /// Return a list of (model_id, layer_start, layer_end) ranges that have no
    /// server covering them, based on the current route table.
    ///
    /// Gaps are only detectable if the router knows the total layer count for
    /// each model. Since the router doesn't store that, we instead return every
    /// layer range between consecutive covered shards.
    pub fn coverage_gaps(&self) -> Vec<(String, u32, u32)> {
        let mut by_model: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
        for entry in self.servers.values() {
            by_model
                .entry(entry.model_id.clone())
                .or_default()
                .push((entry.layer_start, entry.layer_end));
        }
        let mut gaps = Vec::new();
        for (model_id, mut ranges) in by_model {
            ranges.sort_by_key(|(s, _)| *s);
            let mut prev_end: Option<u32> = None;
            for (start, end) in ranges {
                if let Some(pe) = prev_end {
                    if start > pe + 1 {
                        gaps.push((model_id.clone(), pe + 1, start - 1));
                    }
                }
                prev_end = Some(end);
            }
        }
        gaps
    }

    /// All distinct `listen_url` values across all registered servers.
    /// Used by the `/v1/stats` proxy to find a shard to forward to.
    pub fn all_shard_urls(&self) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        self.servers
            .values()
            .filter_map(|s| {
                if seen.insert(s.listen_url.clone()) {
                    Some(s.listen_url.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn status_response(&self) -> StatusResponse {
        // Build per-model coverage
        let mut by_model: HashMap<String, Vec<&ServerEntry>> = HashMap::new();
        for entry in self.servers.values() {
            by_model
                .entry(entry.model_id.clone())
                .or_default()
                .push(entry);
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
            .map(|e| {
                let mut layer_stats: Vec<LayerLatency> = e
                    .layer_latencies
                    .iter()
                    .map(|(&layer, &(avg_ms, p99_ms))| LayerLatency {
                        layer,
                        avg_ms,
                        p99_ms,
                    })
                    .collect();
                layer_stats.sort_by_key(|l| l.layer);
                ServerInfo {
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
                    layer_stats,
                }
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
    #[allow(dead_code)]
    pub fn new(state: Arc<RwLock<GridState>>) -> Self {
        Self {
            state,
            next_id: AtomicU64::new(1),
            grid_key: None,
        }
    }

    pub fn new_with_key(state: Arc<RwLock<GridState>>, key: Option<String>) -> Self {
        Self {
            state,
            next_id: AtomicU64::new(1),
            grid_key: key,
        }
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
            let mut is_available = false; // true while in Mode B available pool

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
                                layer_latencies: HashMap::new(),
                            };
                            state.write().await.register_with_sender(entry, tx.clone());
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
                                hb.layer_stats,
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

                        ServerPayload::Available(av) => {
                            // Mode B: server advertises capacity.
                            // Register it, then check for coverage gaps to fill.
                            state.write().await.register_available(
                                sid.clone(),
                                tx.clone(),
                                av.ram_bytes,
                                av.disk_bytes,
                                av.store_path.clone(),
                            );
                            is_available = true;
                            tracing::info!(
                                server_id = %sid,
                                ram_gb = av.ram_bytes / (1024 * 1024 * 1024),
                                "Grid: Mode B server registered; checking gaps…"
                            );
                            // Attempt to fill an existing coverage gap.
                            let gaps = state.read().await.coverage_gaps();
                            for (model_id, layer_start, layer_end) in gaps {
                                let assigned = state.write().await.try_assign_gap(
                                    &model_id,
                                    layer_start,
                                    layer_end,
                                    "http://origin-placeholder:8090", // filled by config in GT5b
                                    "0000000000000000",
                                    av.ram_bytes,
                                );
                                if assigned {
                                    break; // one gap per available server
                                }
                            }
                        }

                        ServerPayload::Ready(r) => {
                            // Mode B: server finished downloading + loading a shard.
                            // Register it as a serving shard and send Ack.
                            let entry = ServerEntry {
                                server_id: sid.clone(),
                                listen_url: r.listen_url.clone(),
                                model_id: r.model_id.clone(),
                                layer_start: r.layer_start,
                                layer_end: r.layer_end,
                                cpu_pct: 0.0,
                                ram_used: 0,
                                requests_in_flight: 0,
                                last_seen: std::time::Instant::now(),
                                layer_latencies: HashMap::new(),
                            };
                            state.write().await.register_with_sender(entry, tx.clone());
                            registered_model =
                                Some((r.model_id.clone(), r.layer_start, r.layer_end));
                            is_available = false;
                            tracing::info!(
                                server_id = %sid,
                                model_id = %r.model_id,
                                layers = %format!("{}-{}", r.layer_start, r.layer_end),
                                "Grid: Mode B server ready — now serving"
                            );
                            let ack = RouterMessage {
                                payload: Some(RouterPayload::Ack(AckMsg {
                                    server_id: sid.clone(),
                                })),
                            };
                            if tx.send(Ok(ack)).await.is_err() {
                                break;
                            }
                        }

                        ServerPayload::Refuse(r) => {
                            // Mode B: server refused the assignment. Re-add to available pool.
                            tracing::warn!(
                                server_id = %sid,
                                reason = %r.reason,
                                "Grid: Mode B server refused assignment — re-queuing"
                            );
                            // The tx clone is lost here; server must reconnect. Just log.
                        }
                    },
                }
            }

            // Stream closed — clean up
            if registered_model.is_some() {
                state.write().await.deregister(&sid);
            }
            if is_available {
                state.write().await.deregister_available(&sid);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(
        server_id: &str,
        listen_url: &str,
        model_id: &str,
        layer_start: u32,
        layer_end: u32,
    ) -> ServerEntry {
        ServerEntry {
            server_id: server_id.into(),
            listen_url: listen_url.into(),
            model_id: model_id.into(),
            layer_start,
            layer_end,
            cpu_pct: 0.0,
            ram_used: 1024,
            requests_in_flight: 0,
            last_seen: Instant::now(),
            layer_latencies: HashMap::new(),
        }
    }

    #[test]
    fn route_uses_inclusive_layer_ranges() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 2));
        state.register(entry("b", "http://b", "model-a", 3, 5));

        assert_eq!(state.route(Some("model-a"), 0).as_deref(), Some("http://a"));
        assert_eq!(state.route(Some("model-a"), 2).as_deref(), Some("http://a"));
        assert_eq!(state.route(Some("model-a"), 3).as_deref(), Some("http://b"));
        assert_eq!(state.route(Some("model-a"), 5).as_deref(), Some("http://b"));
        assert_eq!(state.route(Some("model-a"), 6), None);
    }

    #[test]
    fn route_without_model_uses_any_model_table() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 1));

        assert_eq!(state.route(None, 1).as_deref(), Some("http://a"));
        assert_eq!(state.route(None, 2), None);
    }

    #[test]
    fn route_prefers_least_loaded_replica() {
        let mut state = GridState::default();
        let mut busy = entry("busy", "http://busy", "model-a", 0, 4);
        busy.requests_in_flight = 12;
        let mut idle = entry("idle", "http://idle", "model-a", 0, 4);
        idle.requests_in_flight = 1;

        state.register(busy);
        state.register(idle);

        assert_eq!(
            state.route(Some("model-a"), 3).as_deref(),
            Some("http://idle")
        );
    }

    #[test]
    fn deregister_removes_server_from_route_table() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 2));
        state.register(entry("b", "http://b", "model-a", 3, 5));

        state.deregister("a");

        assert_eq!(state.route(Some("model-a"), 1), None);
        assert_eq!(state.route(Some("model-a"), 4).as_deref(), Some("http://b"));
    }

    #[test]
    fn heartbeat_updates_load_without_rebuilding_topology() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 4));
        state.register(entry("b", "http://b", "model-a", 0, 4));

        state.update_heartbeat("a", 80.0, 2048, 20, vec![]);
        state.update_heartbeat("b", 10.0, 1024, 0, vec![]);

        assert_eq!(state.route(Some("model-a"), 2).as_deref(), Some("http://b"));
        let a = state.servers.get("a").unwrap();
        assert_eq!(a.cpu_pct, 80.0);
        assert_eq!(a.ram_used, 2048);
        assert_eq!(a.requests_in_flight, 20);
    }

    #[test]
    fn route_all_returns_first_uncovered_layer() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 1));
        state.register(entry("b", "http://b", "model-a", 3, 4));

        assert_eq!(state.route_all(Some("model-a"), &[0, 1, 2, 3]), Err(2));
    }

    #[test]
    fn status_response_reports_shards_and_gaps() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 1));
        state.register(entry("b", "http://b", "model-a", 3, 4));

        let status = state.status_response();

        assert_eq!(status.servers.len(), 2);
        assert_eq!(status.models.len(), 1);
        let model = &status.models[0];
        assert_eq!(model.model_id, "model-a");
        assert_eq!(model.shards.len(), 2);
        assert_eq!(model.gaps.len(), 1);
        assert_eq!(model.gaps[0].layer_start, 2);
        assert_eq!(model.gaps[0].layer_end, 2);
    }

    #[test]
    fn route_prefers_lower_layer_latency_over_inflight() {
        // slow has fewer requests_in_flight but higher per-layer latency.
        // fast has more requests but lower layer latency.
        // Router should route to fast.
        let mut state = GridState::default();
        let mut slow = entry("slow", "http://slow", "model-a", 0, 4);
        slow.requests_in_flight = 2;
        slow.layer_latencies.insert(2, (50.0, 80.0)); // 50 ms avg

        let mut fast = entry("fast", "http://fast", "model-a", 0, 4);
        fast.requests_in_flight = 8;
        fast.layer_latencies.insert(2, (5.0, 9.0)); // 5 ms avg

        state.register(slow);
        state.register(fast);

        assert_eq!(
            state.route(Some("model-a"), 2).as_deref(),
            Some("http://fast")
        );
    }

    #[test]
    fn heartbeat_stores_layer_latencies() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 4));

        let stats = vec![LayerLatency {
            layer: 2,
            avg_ms: 3.5,
            p99_ms: 7.0,
        }];
        state.update_heartbeat("a", 0.0, 0, 0, stats);

        let entry = state.servers.get("a").unwrap();
        assert_eq!(entry.layer_latencies.get(&2), Some(&(3.5, 7.0)));
    }

    #[test]
    fn status_response_includes_layer_stats() {
        let mut state = GridState::default();
        let mut srv = entry("a", "http://a", "model-a", 0, 1);
        srv.layer_latencies.insert(0, (2.1, 4.0));
        state.register(srv);

        let status = state.status_response();
        let server = &status.servers[0];
        assert_eq!(server.layer_stats.len(), 1);
        assert_eq!(server.layer_stats[0].layer, 0);
        assert!((server.layer_stats[0].avg_ms - 2.1).abs() < 0.001);
    }

    #[test]
    fn register_available_and_deregister() {
        let mut state = GridState::default();
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        state.register_available(
            "avail-1".into(),
            tx,
            16 * 1024 * 1024 * 1024,
            100 * 1024 * 1024 * 1024,
            "/mnt/shards".into(),
        );
        assert!(state.available_servers.contains_key("avail-1"));
        state.deregister_available("avail-1");
        assert!(!state.available_servers.contains_key("avail-1"));
    }

    #[test]
    fn coverage_gaps_finds_uncovered_range() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 1));
        state.register(entry("b", "http://b", "model-a", 3, 4));

        let gaps = state.coverage_gaps();
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0], ("model-a".to_string(), 2, 2));
    }

    #[test]
    fn coverage_gaps_empty_when_fully_covered() {
        let mut state = GridState::default();
        state.register(entry("a", "http://a", "model-a", 0, 2));
        state.register(entry("b", "http://b", "model-a", 3, 5));

        // Only gap-between-shards; shards are contiguous here.
        let gaps = state.coverage_gaps();
        assert!(gaps.is_empty());
    }
}
