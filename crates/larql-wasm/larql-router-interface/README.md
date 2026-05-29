# larql-router

Layer-sharding router for distributed `larql-server` deployments.

## What it does

Fans out `POST /v1/walk-ffn` calls across multiple `larql-server`
shards, each owning a contiguous range of transformer layers, and
aggregates their results. The router is intentionally narrow — it
exposes only the endpoints needed for layer-fanout operation, not a
full transparent reverse proxy:

- `POST /v1/walk-ffn` — single-layer or multi-layer fan-out across
  the shard map. Multi-layer requests are dispatched in parallel
  to each owning shard and the results merged.
- `GET /v1/health` — liveness + grid coverage summary.

Other endpoints (`/v1/stats`, `/v1/walk`, `/v1/models`, etc.) live on
the individual shards — clients can call them directly on a shard's
HTTP port. The router exists to coordinate the fan-out, not to be
a full server.

## Two topologies

### Static `--shards` map

Router knows all shards' URLs at boot. Simplest ops; routes are
fixed for the router's lifetime.

```bash
larql-router \
    --shards 0-14=http://shard-a:9181,15-29=http://shard-b:9182 \
    --port 9090
```

### Self-assembling `--grid-port` + `--join`

Router exposes a gRPC port; shards register themselves with `--join
http://router:50052 --public-url http://shard:port`. The router
tracks coverage live and can accept / drop shards without a
restart.

```bash
# Router with HTTP on 9090 + grid gRPC on 50052
larql-router --grid-port 50052 --grid-key <secret> --port 9090

# Each shard joins (see larql-server docs for the full flag list)
larql-server <vindex> --port 9181 --layers 0-14 \
    --join http://router:50052 --grid-key <secret> \
    --public-url http://shard-a:9181
```

When a shard exits cleanly its announce stream closes; the router
logs `Grid: server left layers=N-M` and updates coverage. Requests
for now-uncovered layers return `HTTP 400 "layer N has no owning
shard in this router"` — clean error, not a hang. When the shard
restarts and re-joins, coverage automatically returns.

Both topologies serve the same HTTP API; clients don't need to know
which the operator picked.

## Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--shards <SPEC>` | Comma-separated `START-END=URL` (inclusive bounds). Optional when `--grid-port` is set. | — |
| `--grid-port <PORT>` | gRPC server port for self-assembling grid. Servers connect with `--join`. | — |
| `--grid-key <KEY>` | Shared secret enforced on `--join` registrations. Reads `LARQL_GRID_KEY` env. Without it, the grid port is open (development only). | — |
| `--port <PORT>` | HTTP listen port. | 9090 |
| `--host <HOST>` | Bind address. | 0.0.0.0 |
| `--timeout-secs <N>` | Per-request timeout to backend shards. | 120 |
| `--log-level <LEVEL>` | Logging level. | info |

## Live perf snapshot (M3 Max, 2-shard grid, Gemma 26B-A4B)

Static `--shards` topology:

| Operation | Cold | Warm |
|---|---|---|
| `walk-ffn` 1 layer (router → shard) | 12.8 ms | 0.2–0.3 ms |
| `walk-ffn` 6 layers fan-out | — | 1.3 ms |
| `walk-ffn` 30 layers (full model) | 30 ms | 5.9 ms |
| 8-way concurrent × 15 layers | 112 ms wall | ~1070 layer-evals/sec |

Self-assembling `--grid-port` topology adds a 1–2 ms / request
indirection vs static (gRPC route lookup); negligible for fan-out
calls.

## Validation

Grid routing is covered by focused unit tests for:

- inclusive layer-range routing
- model-specific and default single-model route tables
- least-loaded replica selection from heartbeat load
- deregistration on shard leave
- first uncovered layer reporting for batched requests
- status response shard and gap reporting

```bash
cargo test -p larql-router
```

Current local check: 20 router tests passing, including 7 grid-state tests.

## See also

- `crates/larql-server/README.md` — shard configuration, recommended
  setups, the `--join` / `--public-url` / `--grid-key` flags.
- `crates/larql-server/ROADMAP.md` — perf wins (G1/G2/G3) and live
  validation results.
- `crates/larql-router-protocol/` — the gRPC schema for grid
  announce + heartbeat.
