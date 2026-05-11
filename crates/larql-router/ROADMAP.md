# Roadmap — larql-router / larql-router-protocol

---

## Current state (2026-05-07)

Mode A (self-assembling grid via announce + heartbeat) is implemented and
live-validated on a 2-shard grid running Gemma 4 26B-A4B at 19.7 tok/s
on M3 Max. Static shard maps (`--shards`) also work as a fallback.

The codebase is architecture-agnostic: routing logic reads layer ranges,
model_id, and server state from the grid protocol — no model-family constants
are hardcoded.

### What works today

- `GridService.Join` bidirectional gRPC stream.
- `AnnounceMsg` → `AckMsg` registration flow (Mode A).
- `HeartbeatMsg.layer_stats` (`LayerLatency[]`) — per-layer EMA + p99 routing (GT3 ✅).
- `HeartbeatMsg` → `update_heartbeat()` → per-layer latency routing; falls back to `requests_in_flight`.
- `DroppingMsg` → deregistration on graceful shutdown.
- Static `--shards` mode with layer-range routing and per-shard parallel fan-out.
- Grid + static fallback via `AppState::resolve_all()`.
- `GET /grid-status` (served by `StatusResponse` with `layer_stats` per server).
- Auth: optional shared `--grid-key` Bearer token in gRPC metadata.
- Library crate (`larql_router::grid`) for tests and benchmarks.
- Criterion benchmarks: `routing.rs` (route, route_all, heartbeat, rebuild) (GT9 ✅).

### What is not yet implemented

- Mode B (server advertises capacity → router assigns shard) — GT5.
- `AssignMsg` / `UnassignMsg` — defined in proto, router never sends — GT5/GT6.
- Dynamic rebalancing — GT6.
- QUIC transport — GT7.

---

## Live perf snapshot (2026-05-01, M3 Max, 1 local gRPC shard, 26B-A4B)

| Path | tok/s |
|---|---|
| gRPC unary | 17.7 |
| gRPC streaming + SPLIT overlap (default) | 19.7 |

Per-call transport RTT (loopback):
- TCP HTTP: ~660 µs
- UDS HTTP: ~510 µs
- gRPC streaming (multiplexed): ~460 µs

---

## P1 — Active work (ordered by dependency)

### GT3 — Per-layer latency in HeartbeatMsg ✅ shipped 2026-05-07

**Spec**: ADR-0011 §HeartbeatMsg Extension.

**What shipped:**
- `grid.proto`: `LayerLatency { layer, avg_ms, p99_ms }` message; `HeartbeatMsg.layer_stats = 4`; `ServerInfo.layer_stats = 11`.
- `ServerEntry.layer_latencies: HashMap<u32, (f32, f32)>` (avg_ms, p99_ms).
- `update_heartbeat()` now accepts `Vec<LayerLatency>` and stores them.
- `route()` prefers server with lowest `layer_latencies[layer].avg_ms` when data exists; falls back to `requests_in_flight` otherwise.
- `status_response()` populates `ServerInfo.layer_stats` sorted by layer.
- New tests: `route_prefers_lower_layer_latency_over_inflight`, `heartbeat_stores_layer_latencies`, `status_response_includes_layer_stats`.

**Acceptance**: ✅ `/grid-status` `StatusResponse.servers[*].layer_stats` populated after first heartbeat with layer data.

---

### GT5 — Mode B: gap-fill assignment

**Spec**: ADR-0011 §Phase B1 Protocol.

New state in `GridState`:
```rust
available_servers: HashMap<String, AvailableEntry>,
pending_assignments: HashMap<String, AssignmentRecord>,
target_replicas: u32,  // CLI: --target-replicas, default 1
```

New method `assign_available(model_id, layer_start, layer_end) -> bool`:
1. Find `AvailableEntry` with `ram_bytes >= required_ram_estimate`.
2. Insert into `pending_assignments`.
3. Send `AssignMsg` via the server's `sender` channel.

Triggered from:
- `GridServiceImpl::join()` when `AvailableMsg` arrives and a gap exists.
- `GridServiceImpl::join()` when `DroppingMsg` arrives and a gap is now exposed.
- A new `check_gaps_and_assign()` call after any `deregister()`.

New `rebalancer.rs`:
```rust
pub async fn rebalancer_task(state: Arc<RwLock<GridState>>, cfg: RebalancerConfig)
```
- Spawned by `main.rs` when `--target-replicas > 0`.
- Interval: `--rebalance-interval` (default 30s).
- Calls `state.read().check_gaps_and_assign()` each tick.

New CLI flags:
```
--target-replicas N      default 1
--rebalance-interval S   seconds (default 30)
--rebalance-threshold X  latency ratio (default 2.0)
--assignment-timeout S   seconds (default 600)
```

**Files**:
- `src/grid.rs` — add available_servers, pending_assignments, assign_available, check_gaps_and_assign
- `src/rebalancer.rs` — NEW background task
- `src/main.rs` — spawn rebalancer, add CLI flags

---

### GT6 — Dynamic rebalancing

**Spec**: ADR-0011 §Phase B2 Protocol.

Extends `rebalancer.rs`:
- After gap-fill check, run `check_imbalance()`:
  - Requires GT3 data (`layer_latencies` from heartbeats).
  - Triggers if `max/min > threshold` sustained over `--rebalance-window` (60s default).
  - Only fires when a spare `AvailableEntry` exists for reassignment.
- Sends `UnassignMsg` via server's sender channel.
- Tracks drain state in `pending_assignments` (same map, different phase).
- On `DroppingMsg`: deregister server, move to available state, re-run gap-fill.

Router accepts `RefuseMsg` after `UnassignMsg` (e.g. drain timeout): logs,
deregisters anyway, marks server unavailable for 5 minutes.

---

### GT7 — QUIC transport

**Spec**: ADR-0010 (full spec).

Feature-gated. Additive — TCP gRPC is never removed.

New crate dependency (behind `quic` feature):
```toml
[features]
quic = ["dep:quinn", "dep:rustls"]
quinn = { version = "0.11", optional = true }
rustls = { version = "0.23", optional = true, features = ["ring"] }
```

Router spawns a second listener when `--quic-port PORT` is given:
```rust
// main.rs
if let Some(quic_port) = cli.quic_port {
    let quic_endpoint = transport::quic::make_server_endpoint(quic_addr, &tls)?;
    tokio::spawn(quic_grid_accept_loop(quic_endpoint, grid_state.clone()));
}
```

`quic_grid_accept_loop`: accepts QUIC connections, opens a bidirectional
stream, hands it to the existing `GridServiceImpl::join()` handler by wrapping
the stream as a tonic `Streaming<ServerMessage>`.

**Files**:
- `src/main.rs` — spawn QUIC listener, add `--quic-port`, `--quic-cert`, `--quic-key`
- `../larql-router-protocol/src/transport/quic.rs` — NEW: quinn endpoint, stream wrapper
- `../larql-router-protocol/src/transport/mod.rs` — NEW: re-export (feature-gated)
- `../larql-router-protocol/Cargo.toml` — add quinn optional dep + quic feature

---

### GT9 — Criterion routing benchmarks ✅ shipped 2026-05-07

**Spec**: ADR-0012 §Layer 2.

**What shipped:**
- `crates/larql-router/benches/routing.rs`: `bench_route_single_layer`, `bench_route_all`, `bench_heartbeat_update`, `bench_rebuild_route_table` at 1/10/100 servers × 30/62 layers. No hardcoded model names.
- `src/lib.rs` added to expose `pub mod grid` — required for benches to link against.
- `Cargo.toml`: `criterion = "0.5"`, `[[bench]] name = "routing" harness = false`.
- Makefile target: `make bench-routing`.
- `make bench-all` runs `quant_matvec` + `wire_codec` + `routing`.

New file: `crates/larql-router/benches/routing.rs`

```rust
criterion_group!(
    benches,
    bench_route_single_layer,    // ns/op at 1/10/100 servers
    bench_route_all,             // 30-layer batch at 1/10/100 servers
    bench_heartbeat_update,      // update_heartbeat() per server count
    bench_rebuild_route_table,   // cold-path rebuild at 10/50/100 servers
);
```

No hardcoded model shapes — uses parameterised layer counts and server counts
as `BenchmarkId`.

Add to `Cargo.toml`:
```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "routing"
harness = false
```

---

## P2 — Forward-looking

### Cross-router federation

Multiple routers cover different geographic regions. A client request is
forwarded to the regional router that owns the model shard. Requires a
router-to-router protocol (reuse `GridService.Join` with a `RouterMsg`
variant, or a separate `FederationService`). No implementation planned until
Act 2 multi-host demo is complete.

### Expert-level routing

Current routing is at the layer granularity (server owns a layer range). For
MoE models, a server could own a subset of experts within a layer, not a range
of layers. This requires the router to know expert IDs, not just layer IDs.
Proto messages already have `model_id` and `layer_start/end`; extending to
expert ranges is additive. ADR-0003 §Phase 2 covers this.

### RTT-based routing

`ServerInfo.rtt_ms` is defined in `StatusResponse` but never populated. Adding
active RTT probes (ICMP or HTTP HEAD) from the router to each server would
let the router prefer geographically closer servers as a tie-breaker after
latency-based routing from GT3.
