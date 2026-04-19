# Vindex Server — Remote Knowledge & Inference

**Version:** 0.1  
**Date:** 2026-04-01  
**Status:** Draft  
**Implementation:** `larql-server` crate (Rust)  
**Depends on:** `larql-vindex`, `larql-inference`

---

## 1. Purpose

A lightweight Rust HTTP server that loads a vindex and serves knowledge queries and inference over the network. No GPU, no ML framework, no Python. One binary.

```bash
larql serve gemma3-4b.vindex --port 8080
# Serving google/gemma-3-4b-it (3.0 GB loaded, browse-only, 0 GPU)
# Endpoints: http://localhost:8080/v1/describe, /v1/walk, /v1/select, ...
```

```bash
larql> USE REMOTE "http://localhost:8080";
larql> DESCRIBE "France";
# capital → Paris (probe) — 47ms round-trip
```

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────┐
│                    larql serve                         │
│                                                       │
│  ┌─────────────────────────────────────────────────┐ │
│  │  HTTP Layer (axum)                               │ │
│  │  Routes → handlers → JSON responses              │ │
│  └────────────────────┬────────────────────────────┘ │
│                       │                               │
│  ┌────────────────────▼────────────────────────────┐ │
│  │  Vindex Layer                                    │ │
│  │  VectorIndex (readonly) + PatchedVindex (overlay)│ │
│  │  gate_knn, walk, describe, feature_meta          │ │
│  └────────────────────┬────────────────────────────┘ │
│                       │                               │
│  ┌────────────────────▼────────────────────────────┐ │
│  │  Weight Layer (optional, lazy-loaded)            │ │
│  │  attn_weights.bin → attention forward pass       │ │
│  │  Only loaded on first INFER request              │ │
│  └─────────────────────────────────────────────────┘ │
│                                                       │
│  Files on disk:                                       │
│  gate_vectors.bin (3 GB) — always loaded              │
│  embeddings.bin (2.5 GB) — always loaded              │
│  down_meta.bin (2 MB) — always loaded                 │
│  attn_weights.bin (2 GB) — loaded on first INFER      │
│  feature_labels.json — always loaded                  │
└──────────────────────────────────────────────────────┘
```

### Design principles

- **Stateless queries.** Every request is independent. No session state on the server. Same input → same output, every time.
- **Lazy weight loading.** Browse endpoints load only gate + embed + down_meta. Attention weights load on first INFER request. Compile weights never load.
- **Readonly base.** The server never modifies the vindex files. Patches are per-session or per-request.
- **No framework dependencies.** No PyTorch, no MLX, no ONNX runtime. Pure Rust with BLAS for the dot products.

---

## 3. CLI

```bash
# Serve a single vindex
larql serve <vindex_path> [OPTIONS]

# Serve multiple vindexes
larql serve --dir <directory> [OPTIONS]

# Serve from HuggingFace (downloads on startup)
larql serve "hf://chrishayuk/gemma-3-4b-it-vindex" [OPTIONS]
```

| Flag | Description | Default |
|------|-------------|---------|
| `--port <PORT>` | Listen port | 8080 |
| `--host <HOST>` | Bind address | 0.0.0.0 |
| `--dir <DIR>` | Serve all .vindex directories in this folder | — |
| `--no-infer` | Disable INFER endpoint (browse-only, reduces memory) | false |
| `--cors` | Enable CORS for browser access | false |
| `--max-concurrent <N>` | Max concurrent requests | 100 |
| `--api-key <KEY>` | Require Bearer token auth (health exempt) | — |
| `--log-level <LEVEL>` | Logging level | info |
| `--tls-cert <PATH>` | TLS certificate for HTTPS | — |
| `--tls-key <PATH>` | TLS private key | — |

**Examples:**

```bash
# Development
larql serve output/gemma3-4b-v2.vindex --port 8080

# Production: browse-only, HTTPS, CORS for web clients
larql serve output/gemma3-4b-v2.vindex \
    --port 443 --no-infer --cors \
    --tls-cert cert.pem --tls-key key.pem

# Multi-model server
larql serve --dir ./vindexes/ --port 8080
# Serves: /v1/gemma-3-4b-it/describe, /v1/llama-3-8b/describe, ...

# From HuggingFace
larql serve "hf://chrishayuk/gemma-3-4b-it-vindex" --port 8080
```

**Startup output:**

```
larql-server v0.1.0
Loading: output/gemma3-4b-v2.vindex
  Model: google/gemma-3-4b-it
  Features: 348,160 (34 layers × 10,240)
  Labels: 1,967 probe-confirmed
  Browse: 5.8 GB loaded (gate + embed + down_meta)
  Infer: available (attn_weights.bin detected, will lazy-load)
Listening: http://0.0.0.0:8080
```

---

## 4. API Endpoints

### 4.1 Knowledge Endpoints (browse-only, always available)

#### GET /v1/describe

Query all knowledge edges for an entity.

```
GET /v1/describe?entity=France
GET /v1/describe?entity=France&band=knowledge
GET /v1/describe?entity=France&band=all&verbose=true
GET /v1/describe?entity=France&limit=10
```

| Param | Type | Description | Default |
|-------|------|-------------|---------|
| `entity` | string | Entity name (required) | — |
| `band` | string | Layer band: `syntax`, `knowledge`, `output`, `all` | `knowledge` |
| `verbose` | bool | Include TF-IDF labels, also-tokens, layer ranges | false |
| `limit` | int | Max edges to return | 20 |
| `min_score` | float | Minimum gate score threshold | 5.0 |

**Response:**

```json
{
  "entity": "France",
  "model": "google/gemma-3-4b-it",
  "edges": [
    {
      "relation": "capital",
      "target": "Paris",
      "gate_score": 1436.9,
      "layer": 27,
      "source": "probe",
      "also": ["Berlin", "Tokyo"]
    },
    {
      "relation": "language",
      "target": "French",
      "gate_score": 35.2,
      "layer": 24,
      "layer_max": 32,
      "count": 4,
      "source": "probe"
    },
    {
      "relation": "continent",
      "target": "Europe",
      "gate_score": 14.4,
      "layer": 25,
      "source": "probe",
      "also": ["Spain", "Australia"]
    }
  ],
  "latency_ms": 12.3
}
```

#### GET /v1/walk

Feature scan — which features fire for a prompt.

```
GET /v1/walk?prompt=The+capital+of+France+is&top=10
GET /v1/walk?prompt=Einstein&top=5&layers=24-33
```

| Param | Type | Description | Default |
|-------|------|-------------|---------|
| `prompt` | string | Prompt text (required) | — |
| `top` | int | Top-K features per layer | 5 |
| `layers` | string | Layer range (e.g. `24-33` or `14,26,27`) | all |

**Response:**

```json
{
  "prompt": "The capital of France is",
  "hits": [
    {"layer": 24, "feature": 4532, "gate_score": 26.1, "target": "French"},
    {"layer": 25, "feature": 4207, "gate_score": 14.4, "target": "Europe"},
    {"layer": 27, "feature": 9515, "gate_score": 1436.9, "target": "Paris"}
  ],
  "latency_ms": 0.4
}
```

#### POST /v1/select

SQL-style edge query.

```json
POST /v1/select
{
  "relation": "capital",
  "limit": 10,
  "order_by": "gate_score",
  "order": "desc"
}
```

```json
POST /v1/select
{
  "entity": "France",
  "min_confidence": 0.5
}
```

**Response:**

```json
{
  "edges": [
    {"entity": "France", "relation": "capital", "target": "Paris", "gate_score": 1436.9},
    {"entity": "Germany", "relation": "capital", "target": "Berlin", "gate_score": 1289.3},
    {"entity": "Japan", "relation": "capital", "target": "Tokyo", "gate_score": 1156.7}
  ],
  "total": 94,
  "latency_ms": 45.2
}
```

#### GET /v1/relations

List all known relation types.

```
GET /v1/relations
GET /v1/relations?source=probe
```

**Response:**

```json
{
  "relations": [
    {"name": "capital", "count": 94, "source": "probe", "example": "France→Paris"},
    {"name": "language", "count": 51, "source": "probe", "example": "France→French"},
    {"name": "birthplace", "count": 91, "source": "probe", "example": "Mozart→Salzburg"}
  ],
  "total_probe": 1967,
  "total_cluster": 512,
  "latency_ms": 0.1
}
```

#### GET /v1/stats

Model and index statistics.

```
GET /v1/stats
```

**Response:**

```json
{
  "model": "google/gemma-3-4b-it",
  "family": "gemma3",
  "layers": 34,
  "features": 348160,
  "features_per_layer": 10240,
  "hidden_size": 2560,
  "vocab_size": 262208,
  "extract_level": "all",
  "dtype": "f32",
  "probe_confirmed": 1967,
  "cluster_types": 512,
  "layer_bands": {
    "syntax": [0, 13],
    "knowledge": [14, 27],
    "output": [28, 33]
  },
  "loaded": {
    "browse": true,
    "inference": true,
    "compile": false
  },
  "memory_mb": 5800
}
```

### 4.2 Inference Endpoint (requires attention weights)

#### POST /v1/infer

Full forward pass with attention. Returns next-token predictions.

```json
POST /v1/infer
{
  "prompt": "The capital of France is",
  "top": 5,
  "mode": "walk"
}
```

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `prompt` | string | Prompt text (required) | — |
| `top` | int | Top-K predictions | 5 |
| `mode` | string | `walk` (vindex FFN) or `dense` (original FFN) or `compare` | `walk` |

**Response:**

```json
{
  "prompt": "The capital of France is",
  "predictions": [
    {"token": "Paris", "probability": 0.9791},
    {"token": "the", "probability": 0.0042},
    {"token": "a", "probability": 0.0031}
  ],
  "mode": "walk",
  "latency_ms": 210
}
```

**Compare mode:**

```json
{
  "prompt": "The capital of France is",
  "walk": [
    {"token": "Paris", "probability": 0.9791}
  ],
  "dense": [
    {"token": "Paris", "probability": 0.9801}
  ],
  "latency_ms": 420
}
```

### 4.3 Patch Endpoints

#### POST /v1/patches/apply

Apply a patch to the server's in-memory vindex. Per-session — does not modify base files.

```json
POST /v1/patches/apply
{
  "url": "hf://medical-ai/drug-interactions@2.1.0"
}
```

Or upload a patch directly:

```json
POST /v1/patches/apply
{
  "patch": {
    "version": 1,
    "operations": [
      {"op": "insert", "layer": 26, "feature": 8821, ...}
    ]
  }
}
```

**Response:**

```json
{
  "applied": "drug-interactions@2.1.0",
  "operations": 5000,
  "active_patches": 1
}
```

#### GET /v1/patches

List active patches.

```json
{
  "patches": [
    {"name": "drug-interactions@2.1.0", "operations": 5000, "source": "hf://medical-ai/drug-interactions@2.1.0"}
  ]
}
```

#### DELETE /v1/patches/{name}

Remove a patch.

```
DELETE /v1/patches/drug-interactions@2.1.0
```

### 4.4 Management Endpoints

#### GET /v1/health

```json
{"status": "ok", "uptime_seconds": 3600, "requests_served": 12450}
```

#### GET /v1/models

List loaded models (multi-model server).

```json
{
  "models": [
    {
      "id": "gemma-3-4b-it",
      "path": "/v1/gemma-3-4b-it",
      "features": 348160,
      "probe_confirmed": 1967,
      "loaded": true
    }
  ]
}
```

---

## 5. Multi-Model Serving

When serving a directory, each vindex gets its own namespace:

```bash
larql serve --dir ./vindexes/ --port 8080
# Found: gemma-3-4b-it, llama-3-8b, mistral-7b
```

```
GET /v1/gemma-3-4b-it/describe?entity=France
GET /v1/llama-3-8b/describe?entity=France
GET /v1/mistral-7b/describe?entity=France
```

Single-model serving uses the root path:

```
GET /v1/describe?entity=France
```

---

## 6. Client-Side Patches

The server hosts the immutable base. The client brings patches. Two modes:

### 6.1 Session Patches (server-side)

Client sends a patch, server applies it in-memory for that session. Good for web apps where the client can't run gate KNN locally.

```
POST /v1/patches/apply  {"url": "..."}
GET  /v1/describe?entity=aspirin
# Returns edges from base + patch
```

The patch stays in memory until the server restarts or the patch is removed. Each client session can have different patches — the server maintains per-session PatchedVindex instances.

### 6.2 Client-Side Overlay (client-side)

Client has the patch locally. Sends queries to the server, overlays patch results locally. Good for privacy — patch contents never leave the client.

```
Client:
  1. GET /v1/describe?entity=aspirin → base edges from server
  2. Check local patch for aspirin edges → local overrides
  3. Merge: server edges + local overrides → display
```

The REPL handles this transparently:

```sql
USE REMOTE "http://localhost:8080";
APPLY PATCH "private-facts.vlp";        -- stays local
DESCRIBE "aspirin";
-- Server provides base edges
-- REPL overlays local patch edges
-- User sees combined result
-- Server never sees patch contents
```

---

## 7. Performance

### 7.1 Expected Latency

| Endpoint | Server Processing | Network (LAN) | Network (Internet) | Total |
|----------|------------------|---------------|-------------------|-------|
| /v1/describe | 12ms | 1ms | 50ms | 13-62ms |
| /v1/walk | 0.4ms | 1ms | 50ms | 1-50ms |
| /v1/select | 5ms | 1ms | 50ms | 6-55ms |
| /v1/relations | 0.1ms | 1ms | 50ms | 1-50ms |
| /v1/stats | 0.01ms | 1ms | 50ms | 1-50ms |
| /v1/infer | 200ms | 1ms | 50ms | 200-250ms |

Browse queries are dominated by network latency, not computation. The server processes DESCRIBE in 12ms — the rest is round-trip time.

### 7.2 Memory Usage

| Mode | Memory | What's loaded |
|------|--------|---------------|
| Browse-only | ~6 GB (f32) / ~3 GB (f16) | gate + embed + down_meta + labels |
| Browse + Infer | ~8 GB (f32) / ~4 GB (f16) | + attn_weights |
| Browse + Infer + Patches | ~8 GB + patches | + per-session PatchedVindex |

### 7.3 Throughput

The server is stateless for browse queries — each request is an independent gate KNN. This means:

- **Horizontal scaling:** Run N instances behind a load balancer. Each loads the same vindex (or uses mmap to share pages).
- **CDN caching:** For popular entities, cache DESCRIBE responses at the edge. TTL-based invalidation when labels update.
- **Concurrent requests:** axum handles async I/O. Gate KNN is CPU-bound but completes in 12ms. 100 concurrent requests = 1.2 seconds of CPU per batch.

At 12ms per DESCRIBE, a single server handles ~80 queries/second. With 4 instances: ~320 queries/second. More than enough for most use cases.

---

## 8. Security

### 8.1 Authentication (implemented)

Optional API key authentication. `/v1/health` is exempt (always accessible).

```bash
larql serve gemma3-4b.vindex --port 8080 --api-key "sk-abc123"
```

```
GET /v1/describe?entity=France
Authorization: Bearer sk-abc123
```

Requests without a valid token receive 401 Unauthorized.

### 8.2 Concurrency Limit (implemented)

Max concurrent requests via tower middleware:

```bash
larql serve gemma3-4b.vindex --max-concurrent 100
```

### 8.3 Rate Limiting (implemented)

Per-IP token bucket rate limiting. Supports `N/sec`, `N/min`, `N/hour` formats. `/v1/health` is exempt. Respects `X-Forwarded-For` for proxied clients.

```bash
larql serve gemma3-4b.vindex --rate-limit "100/min"
```

Excess requests receive `429 Too Many Requests`.

### 8.4 DESCRIBE Cache (implemented)

In-memory TTL cache for DESCRIBE results. Keys include model ID, entity, band, limit, min_score.

```bash
larql serve gemma3-4b.vindex --cache-ttl 300  # 5 minute TTL
```

### 8.5 Per-Session Isolation (implemented)

Patches can be scoped to a session via the `X-Session-Id` header. Each session gets its own PatchedVindex overlay. Sessions expire after 1 hour of inactivity.

```
POST /v1/patches/apply
X-Session-Id: sess-medical-team
{"patch": {...}}
```

Without the header, patches go to the global shared state.

### 8.6 Patch Validation

Patches uploaded via API are validated before application:
- Base model checksum must match
- Operations must reference valid layers and features
- Gate vectors must have correct dimensions

### 8.4 No Write Access

The server never modifies vindex files on disk. All mutations are in-memory via PatchedVindex. The base vindex is readonly. There is no endpoint to modify the base files.

---

## 9. REPL Integration

The REPL connects to remote servers transparently:

```sql
USE REMOTE "http://localhost:8080";
-- Connected: google/gemma-3-4b-it (348K features, 1967 probe-confirmed)

DESCRIBE "France";
-- Sends GET /v1/describe?entity=France
-- Displays same format as local DESCRIBE

WALK "Einstein" TOP 10;
-- Sends GET /v1/walk?prompt=Einstein&top=10

SELECT entity, target FROM EDGES WHERE relation = "capital" LIMIT 5;
-- Sends POST /v1/select

INFER "The capital of France is" TOP 5;
-- Sends POST /v1/infer

APPLY PATCH "local-facts.vlp";
-- Patch stays local, overlaid on remote results

SHOW PATCHES;
-- Shows local patches only
```

The user doesn't know or care whether the vindex is local or remote. Same statements, same output format.

**Implementation status:** `USE REMOTE` is fully implemented. Supported: DESCRIBE, WALK, INFER, SELECT, STATS, SHOW RELATIONS, INSERT, DELETE, UPDATE. Local patches (APPLY PATCH) stay client-side and overlay on remote DESCRIBE results.

---

## 10. Deployment

### 10.1 Docker

```dockerfile
FROM rust:1.82-slim AS builder
WORKDIR /build
COPY . .
RUN cargo build --release -p larql-server

FROM debian:bookworm-slim
COPY --from=builder /build/target/release/larql-server /usr/local/bin/
EXPOSE 8080
ENTRYPOINT ["larql-server"]
```

```bash
docker run -v ./vindexes:/data -p 8080:8080 larql-server /data/gemma3-4b.vindex
```

### 10.2 Fly.io / Railway

```toml
# fly.toml
[build]
  builder = "dockerfile"

[env]
  VINDEX_PATH = "/data/gemma3-4b.vindex"

[mounts]
  source = "vindex_data"
  destination = "/data"

[[services]]
  internal_port = 8080
  protocol = "tcp"
  [services.concurrency]
    hard_limit = 100
```

Browse-only deployment: 3 GB RAM (f16). $5-10/month on Fly.io.

### 10.3 Bare Metal / VPS

```bash
# Install
cargo install larql-server

# Run
larql-server /path/to/model.vindex --port 8080

# Systemd service
[Unit]
Description=LARQL Vindex Server
After=network.target

[Service]
ExecStart=/usr/local/bin/larql-server /data/gemma3-4b.vindex --port 8080
Restart=always
MemoryMax=8G

[Install]
WantedBy=multi-user.target
```

$5-20/month VPS. No GPU. No Python. No CUDA drivers.

---

## 11. Crate Structure

```
larql-server/
├── Cargo.toml
├── examples/
│   ├── server_demo.rs          Synthetic vindex API demo
│   └── server_bench.rs         Endpoint latency benchmarks
├── tests/
│   └── test_api.rs             Integration tests (76 tests)
└── src/
    ├── main.rs                 CLI parsing, server startup
    ├── state.rs                AppState: loaded models, probe labels, lazy weights
    ├── auth.rs                 API key Bearer token middleware
    ├── ratelimit.rs            Per-IP token bucket rate limiting
    ├── cache.rs                TTL cache for DESCRIBE results
    ├── session.rs              Per-session PatchedVindex isolation
    ├── error.rs                ServerError → HTTP status codes
    ├── routes/
    │   ├── mod.rs              Router setup (single + multi-model)
    │   ├── describe.rs         GET /v1/describe (cached, relation labels)
    │   ├── walk.rs             GET /v1/walk (with relation labels)
    │   ├── select.rs           POST /v1/select (relation filter)
    │   ├── relations.rs        GET /v1/relations
    │   ├── stats.rs            GET /v1/stats
    │   ├── infer.rs            POST /v1/infer
    │   ├── patches.rs          POST/GET/DELETE /v1/patches
    │   ├── health.rs           GET /v1/health
    │   └── models.rs           GET /v1/models
    ├── session.rs              Per-session PatchedVindex management
    ├── auth.rs                 API key validation middleware
    └── error.rs                Error types → HTTP status codes
```

**Dependencies:** `larql-vindex`, `larql-inference` (for INFER), `axum`, `tokio`, `serde_json`, `tower-http` (CORS, logging)

---

## 12. Advanced Endpoints

### 12.1 WebSocket Streaming (implemented)

Layer-by-layer streaming for DESCRIBE via `WS /v1/stream`:

```
→ {"type": "describe", "entity": "France", "band": "all"}
← {"type": "layer", "layer": 14, "edges": []}
← {"type": "layer", "layer": 15, "edges": [{"target": "French", "gate_score": 35.2}]}
← {"type": "layer", "layer": 27, "edges": [{"relation": "capital", "target": "Paris", "gate_score": 1436.9, "source": "probe"}]}
← {"type": "done", "entity": "France", "total_edges": 6, "latency_ms": 12.3}
```

Edges include probe relation labels when available. Error messages: `{"type": "error", "message": "..."}`.

### 12.2 gRPC (implemented)

All endpoints available over gRPC via tonic + prost. Enable with `--grpc-port`:

```bash
larql serve gemma3-4b.vindex --port 8080 --grpc-port 50051
```

Proto definition: `proto/vindex.proto`. Services mirror the REST API: `Describe`, `Walk`, `Select`, `Infer`, `GetRelations`, `GetStats`, `WalkFfn`, `Health`. Includes `StreamDescribe` for server-streaming layer-by-layer results.

gRPC runs alongside HTTP — both ports active simultaneously. Uses bundled protoc (no system install required).

### 12.3 Edge Caching (implemented)

DESCRIBE responses include ETag and Cache-Control headers for CDN edge caching:

```
Cache-Control: public, max-age=86400
ETag: "a1b2c3d4"
```

Clients can send `If-None-Match` to receive `304 Not Modified` when the response hasn't changed. Combined with `--cache-ttl` for server-side caching.

### 12.4 Decoupled Inference Protocol (implemented)

`POST /v1/walk-ffn` — client sends a residual vector, server runs gate KNN and returns selected features + scores.

**Single layer:**

```json
POST /v1/walk-ffn
{"layer": 26, "residual": [0.12, -0.34, ...], "top_k": 8092}
→ {"layer": 26, "features": [9515, 4532, ...], "scores": [1436.9, 26.1, ...], "latency_ms": 0.01}
```

**Batched (all layers in one round-trip):**

```json
POST /v1/walk-ffn
{"layers": [0, 1, ..., 33], "residual": [0.12, -0.34, ...]}
→ {"results": [{"layer": 0, "features": [...], "scores": [...]}, ...], "latency_ms": 0.3}
```

Single-layer mode: 34 round-trips per token. Batched mode: 1 round-trip (~200ms per token anywhere). Validates that residual length matches hidden_size.

---

## 13. FFN-Service Mode and Layer Sharding

### 13.1 FFN-Service Mode (`--ffn-only`)

Run the server as a dedicated FFN backend. The client holds attention weights locally;
this server holds only the FFN weights and answers `/v1/walk-ffn` requests.

```bash
larql-server output/gemma3-4b-q4k.vindex --ffn-only --port 8080
```

**What changes under `--ffn-only`:**

- `/v1/infer` is disabled (`infer_disabled = true`)
- `/v1/stats` advertises `"mode": "ffn-service"` — `RemoteWalkBackend` on the client
  uses this to confirm it has connected to the right endpoint
- f16→f32 gate warmup is skipped; gate decode is lazy per layer on first request
- Attention + embed + lm_head tensors are filtered from the weight manifest at load
  time (pre-mmap filter — no allocation for ~3.4 GB of f32 tensors on 4B, ~14 GB on 31B)

**Client usage:**

```bash
larql walk --ffn-remote http://server:8080 --predict \
    --prompt "The capital of France is"
```

The client runs attention locally and calls the server for every FFN layer.

---

### 13.2 Layer Sharding (`--layers`)

Restrict the server to a contiguous layer range. Only those layers are loaded into
memory; requests for other layers are rejected immediately.

```bash
larql-server output/gemma3-4b-q4k.vindex --ffn-only --layers 0-16  --port 8080
larql-server output/gemma3-4b-q4k.vindex --ffn-only --layers 17-33 --port 8081
```

`START-END` bounds are **inclusive**. A 34-layer model split into two shards:

| Shard | Layers | Approximate RSS |
|-------|--------|-----------------|
| A | 0–16 (17 layers) | ~50% of full vindex |
| B | 17–33 (17 layers) | ~50% of full vindex |

**Memory model:**

Out-of-range layers are never loaded into physical RAM:

- For Q4K vindexes (`synthesize_gate_from_q4k`): the anonymous mmap is sized only
  for owned layers. Unowned layers are skipped entirely during dequantisation.
- For demand-paged files (`gate_vectors.bin`, `interleaved_q4k.bin`): the OS-level
  mmap covers the full file (cheap — virtual address space only). `is_layer_owned()`
  guards every accessor before any byte is read, so out-of-range pages never fault in.

**Startup log:**

```
larql-server v0.1.0
Loading: output/gemma3-4b-q4k.vindex
  Layers: 0–16 (of 34)
  Model: google/gemma-3-4b-it (34 layers, 348160 features)
  Warmup: skipped (--ffn-only, lazy gate decode on first request)
  Mode: ffn-service (--ffn-only)
Listening: http://0.0.0.0:8080
```

**Out-of-range rejection:**

```json
POST /v1/walk-ffn {"layer": 20, "residual": [...]}
→ 400 {"error": "layer 20 not served by this shard (owned: 0–16)"}
```

**CLI options summary:**

| Flag | Description |
|------|-------------|
| `--ffn-only` | FFN-service mode; disables infer, skips warmup, filters attn weights |
| `--layers START-END` | Inclusive layer range to load and serve |
| `--max-gate-cache-layers N` | LRU cap on decoded f16 gate layers in memory (0 = unlimited) |

---

### 13.3 Deployment with a Router

Layer-sharded servers are not meant to be addressed directly. Use `larql-router`
(see `larql-router-spec.md`) as a transparent proxy in front of them. The client
uses `--ffn-remote http://router:9090` and has no knowledge of the shard topology.

```
Client  →  larql-router:9090  →  shard-a:8080  (layers 0–16)
                               →  shard-b:8081  (layers 17–33)
```

The router health-checks each shard on startup and rejects requests for layers that
have no owning shard with a clear error before contacting any backend.

---

## License

Apache-2.0
