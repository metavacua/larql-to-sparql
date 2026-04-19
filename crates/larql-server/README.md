# larql-server

HTTP server for vindex knowledge queries and inference. Loads a vindex and serves it over the network. No GPU, no ML framework, no Python. One binary.

```bash
larql serve output/gemma3-4b.vindex --port 8080
# Serving google/gemma-3-4b-it (348K features, 1967 probe-confirmed)
# Listening: http://0.0.0.0:8080
```

```bash
curl "http://localhost:8080/v1/describe?entity=France"
# {"entity":"France","edges":[{"relation":"capital","target":"Paris","gate_score":1436.9,"layer":27,"source":"probe"}, ...]}
```

## Features

- **Browse endpoints** — DESCRIBE, WALK, SELECT, RELATIONS, STATS (no weights needed)
- **Inference** — full forward pass with WalkFfn (weights lazy-loaded on first request)
- **Relation labels** — probe-confirmed labels from `feature_labels.json` in DESCRIBE responses
- **Patch overlay** — apply knowledge patches via API without modifying base files
- **Multi-model serving** — serve multiple vindexes from a directory
- **HuggingFace support** — load vindexes directly from `hf://` paths
- **API key auth** — optional Bearer token authentication
- **TLS** — native HTTPS via rustls
- **Concurrency limit** — configurable max concurrent requests
- **CORS** — enable for browser-based clients
- **REPL integration** — `USE REMOTE "http://..."` in the LQL REPL

## Quickstart

```bash
# Build
cargo build --release -p larql-server

# Serve a local vindex
larql-server output/gemma3-4b-v2.vindex --port 8080

# Or via the CLI wrapper
larql serve output/gemma3-4b-v2.vindex --port 8080

# From HuggingFace
larql serve "hf://chrishayuk/gemma-3-4b-it-vindex" --port 8080

# Multi-model
larql serve --dir ./vindexes/ --port 8080

# With auth + TLS
larql serve output/gemma3-4b.vindex --api-key "sk-abc123" --tls-cert cert.pem --tls-key key.pem
```

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `<VINDEX_PATH>` | Path to .vindex directory or `hf://` URL | — |
| `--dir <DIR>` | Serve all .vindex directories in folder | — |
| `--port <PORT>` | Listen port | 8080 |
| `--host <HOST>` | Bind address | 0.0.0.0 |
| `--no-infer` | Disable inference (browse-only, saves memory) | false |
| `--ffn-only` | Run as an FFN-service endpoint for `RemoteWalkBackend` clients. Skips the f16→f32 gate warmup (10× smaller startup RSS on 31B Q4_K) | false |
| `--embed-only` | Run as an embed-service endpoint (ADR-0008). Loads only embeddings + lm_head + tokenizer; skips all FFN and attention weights. Enables `/v1/embed`, `/v1/logits`, `/v1/token/*`. Advertises `mode: embed-service`. | false |
| `--layers <START-END>` | Serve only this layer range. Out-of-range requests return HTTP 400. Pages outside the range are never touched. | all |
| `--max-gate-cache-layers <N>` | LRU cap on decoded f16 gate layers. `0` = unlimited. Each decoded layer is ~433 MB on 31B. | 0 |
| `--release-mmap-after-request` | `madvise(MADV_DONTNEED)` on all mmaps after each walk-ffn request. Linux: immediate RSS drop. Darwin: advisory. | false |
| `--cors` | Enable CORS headers | false |
| `--api-key <KEY>` | Require Bearer token auth (health exempt) | — |
| `--rate-limit <SPEC>` | Per-IP rate limit (e.g., "100/min", "10/sec") | — |
| `--max-concurrent <N>` | Max concurrent requests | 100 |
| `--cache-ttl <SECS>` | Cache TTL for DESCRIBE results (0 = disabled) | 0 |
| `--grpc-port <PORT>` | Enable gRPC server on this port | — |
| `--tls-cert <PATH>` | TLS certificate for HTTPS | — |
| `--tls-key <PATH>` | TLS private key for HTTPS | — |
| `--log-level <LEVEL>` | Logging level | info |

### Memory bounds — cheat sheet

Measured on Gemma 4 31B Q4_K (macOS, CPU). See ADR-0005 for details.

| Flags | Startup RSS | After 3 requests |
|---|---|---|
| default | 55 GB | 55 GB |
| `--ffn-only` | 5.6 GB | 23 GB |
| `--ffn-only --max-gate-cache-layers 4` | 5.6 GB | 23 GB |
| `... --release-mmap-after-request` | 5.6 GB | 23 GB stable (Linux: ~6 GB) |
| `... --layers 0-19` (sharding) | 5.6 GB | ~8 GB |

`--layers` is the strict bound. The other flags target individual growth
modes and compose cleanly (`--ffn-only` skips startup warmup,
`--max-gate-cache-layers` caps decoded heap, `--release-mmap-after-request`
hints the kernel to drop mmap pages).

## API Endpoints

### Knowledge Endpoints (browse-only)

#### GET /v1/describe

Query all knowledge edges for an entity. Edges include probe-confirmed relation labels when `feature_labels.json` is present in the vindex.

```
GET /v1/describe?entity=France
GET /v1/describe?entity=France&band=all&verbose=true&limit=10&min_score=5.0
```

```json
{
  "entity": "France",
  "model": "google/gemma-3-4b-it",
  "edges": [
    {"relation": "capital", "target": "Paris", "gate_score": 1436.9, "layer": 27, "source": "probe", "also": ["Berlin", "Tokyo"]},
    {"target": "French", "gate_score": 35.2, "layer": 24},
    {"target": "Europe", "gate_score": 14.4, "layer": 25}
  ],
  "latency_ms": 12.3
}
```

| Param | Default | Description |
|-------|---------|-------------|
| `entity` | required | Entity name |
| `band` | `knowledge` | Layer band: syntax, knowledge, output, all |
| `verbose` | false | Include layer_min, layer_max, count per edge |
| `limit` | 20 | Max edges |
| `min_score` | 5.0 | Minimum gate score |

#### GET /v1/walk

Feature scan — which features fire for a prompt.

```
GET /v1/walk?prompt=The+capital+of+France+is&top=5
GET /v1/walk?prompt=Einstein&top=10&layers=24-33
```

```json
{
  "prompt": "The capital of France is",
  "hits": [
    {"layer": 27, "feature": 9515, "gate_score": 1436.9, "target": "Paris"},
    {"layer": 24, "feature": 4532, "gate_score": 26.1, "target": "French"}
  ],
  "latency_ms": 0.4
}
```

#### POST /v1/select

SQL-style edge query over down-projection metadata.

```json
POST /v1/select
{"entity": "France", "limit": 10, "order_by": "c_score", "order": "desc"}
```

```json
{
  "edges": [
    {"layer": 26, "feature": 8821, "target": "Paris", "c_score": 0.95}
  ],
  "total": 94,
  "latency_ms": 5.2
}
```

#### GET /v1/relations

List top tokens across knowledge layers.

```json
{
  "relations": [
    {"name": "Paris", "count": 94, "example": "Berlin"},
    {"name": "French", "count": 51, "example": "German"}
  ],
  "total": 512
}
```

#### GET /v1/stats

Model and index statistics.

```json
{
  "model": "google/gemma-3-4b-it",
  "family": "gemma3",
  "layers": 34,
  "features": 348160,
  "hidden_size": 2560,
  "layer_bands": {"syntax": [0, 13], "knowledge": [14, 27], "output": [28, 33]},
  "loaded": {"browse": true, "inference": true}
}
```

### Inference Endpoint

#### POST /v1/infer

Full forward pass with attention weights. Model weights are lazy-loaded on first request. Requires a vindex built with `--include-weights` (or extract level `inference`/`all`). Disabled when `--no-infer` is set.

```json
POST /v1/infer
{"prompt": "The capital of France is", "top": 5, "mode": "walk"}
```

| Field | Default | Description |
|-------|---------|-------------|
| `prompt` | required | Input text |
| `top` | 5 | Top-K predictions |
| `mode` | `walk` | `walk` (vindex FFN), `dense` (original FFN), `compare` (both) |

**Walk mode response:**

```json
{
  "prompt": "The capital of France is",
  "predictions": [
    {"token": "Paris", "probability": 0.9791},
    {"token": "the", "probability": 0.0042}
  ],
  "mode": "walk",
  "latency_ms": 210
}
```

**Compare mode response:**

```json
{
  "prompt": "The capital of France is",
  "walk": [{"token": "Paris", "probability": 0.9791}],
  "walk_ms": 210,
  "dense": [{"token": "Paris", "probability": 0.9801}],
  "dense_ms": 180,
  "latency_ms": 420
}
```

### Streaming Endpoint

#### WS /v1/stream

WebSocket endpoint for layer-by-layer streaming DESCRIBE. Client sends a JSON message, server streams per-layer results.

```
→ {"type": "describe", "entity": "France", "band": "all"}
← {"type": "layer", "layer": 14, "edges": []}
← {"type": "layer", "layer": 15, "edges": [{"target": "French", "gate_score": 35.2}]}
← {"type": "layer", "layer": 27, "edges": [{"relation": "capital", "target": "Paris", "gate_score": 1436.9, "source": "probe"}]}
← {"type": "done", "entity": "France", "total_edges": 6, "latency_ms": 12.3}
```

### Decoupled Inference Endpoint

#### POST /v1/walk-ffn

Client sends a residual vector, server runs gate KNN and returns selected features. Enables distributed inference where the client runs attention locally.

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

**Full-output mode** — returns the computed FFN output vector (gate KNN → up
gather → down projection). Requires model weights (`--ffn-only` is sufficient).

```json
POST /v1/walk-ffn
Content-Type: application/json
{"layer": 26, "residual": [...], "seq_len": 1, "full_output": true}
→ {"layer": 26, "output": [...], "seq_len": 1, "latency_ms": 8.1}
```

**Binary wire format** (`Content-Type: application/x-larql-ffn`) — eliminates
JSON float serialization overhead. Only supported with `full_output: true`.

```
Single-layer request:
  [4: layer u32 LE][4: seq_len u32][4: flags u32 (bit0=1)][4: top_k u32][residual f32[] LE]

Single-layer response:
  [4: layer u32 LE][4: seq_len u32][4: latency f32][output f32[] LE]

Batch request:  [4: BATCH_MARKER=0xFFFFFFFF][4: num_layers u32][layers u32[] LE]...
Batch response: [4: BATCH_MARKER][4: num_results u32][4: latency f32]
                per result: [4: layer][4: seq_len][4: num_floats][output f32[] LE]
```

Performance vs JSON (Gemma 3 4B, hidden_size=3072, seq_len=1): ~33% smaller
requests, ~0.5 ms/hop faster.

`RemoteWalkBackend` in `larql-inference` uses binary format automatically and
exposes `forward_all_layers()` for a batched single-round-trip forward pass.

### Embed Service Endpoints (ADR-0008)

Enabled on every server (including `--ffn-only` and default mode). The primary use case is `--embed-only`: offload the static embedding table and lm_head to a dedicated small server, shrinking the attention-only client from ~7 GB to ~1.9 GB on 31B models.

```bash
# Start an embed-only server
larql-server output/gemma3-4b.vindex --embed-only --port 8082

# Serving google/gemma-3-4b-it — mode: embed-service
# Loaded: embeddings (1.3 GB), lm_head (tied), tokenizer
# Listening: http://0.0.0.0:8082
```

#### POST /v1/embed

Convert token IDs to scaled initial residual vectors.

```json
POST /v1/embed
{"token_ids": [1, 5432, 235, 1234]}
```

```json
{
  "residual": [[0.12, -0.03, ...], [0.45, 0.01, ...]],
  "seq_len": 4,
  "hidden_size": 2560,
  "latency_ms": 0.02
}
```

Binary wire format (`Content-Type: application/x-larql-ffn`):

```
Request:  [num_tokens u32 LE][token_id u32 LE × N]
Response: [seq_len u32 LE][hidden_size u32 LE][residuals f32[] LE]
```

**Measured (Gemma 3 4B, hidden=2560):** encode request 17 ns, encode response 1.5 µs.
Binary is 6.7× faster and 3× smaller than JSON for the embed response. Use binary on the decode hot path.

#### POST /v1/logits

Project a final residual through lm_head to get token probabilities. Accepts JSON or binary input.

```json
POST /v1/logits
{"residual": [0.12, -0.03, ...], "top_k": 5, "temperature": 1.0}
```

```json
{
  "top_k": [
    {"token_id": 9515, "token": "Paris", "prob": 0.801},
    {"token_id": 235,  "token": "the",   "prob": 0.042}
  ],
  "latency_ms": 2.1
}
```

Binary input (`Content-Type: application/x-larql-ffn`): raw `[f32 × hidden_size]` little-endian bytes.

Performance (measured, Gemma 3 4B): ~14ms CPU (BLAS), ~0.67ms Metal (Apple Silicon f32_gemv).

#### GET /v1/token/encode

```
GET /v1/token/encode?text=Paris
→ {"token_ids": [9515], "text": "Paris"}
```

#### GET /v1/token/decode

```
GET /v1/token/decode?ids=9515,235,1234
→ {"text": "Paris the model", "token_ids": [9515, 235, 1234]}
```

#### Memory footprint — embed-only server

Measured on Gemma 3 4B Q4K (macOS, release build). See ADR-0008 for full benchmark output.

| Model | Disk (f16) | RSS (f32 heap) | Total RSS (with tokenizer) |
|-------|-----------|----------------|---------------------------|
| Gemma 3 4B | 1.34 GB | 2.69 GB | ~2.9 GB |
| Gemma 4 31B | 2.67 GB | 5.37 GB | ~5.6 GB |
| Llama 3 70B | 2.10 GB | 4.20 GB | ~4.5 GB |

The current implementation decodes f16→f32 at load time (doubles RSS vs disk).
A future f16-at-rest path will halve this — tracked in ADR-0008 open questions.

The tokenizer alone takes ~244 MB for the Gemma 262K-vocab BPE model.

### gRPC

All endpoints are available over gRPC using Protocol Buffers. Enable with `--grpc-port`:

```bash
larql serve output/gemma3-4b.vindex --port 8080 --grpc-port 50051
```

Proto definition: `proto/vindex.proto`. Services: `Describe`, `Walk`, `Select`, `Infer`, `GetRelations`, `GetStats`, `WalkFfn`, `Health`, `StreamDescribe` (server-streaming).

gRPC runs alongside HTTP — both ports active simultaneously.

### Patch Endpoints

#### POST /v1/patches/apply

Apply a patch in-memory (does not modify base files). Session-aware: include `X-Session-Id` header to scope patches to a session.

```json
POST /v1/patches/apply
{"patch": {"version": 1, "base_model": "...", "operations": [...]}}
```

```bash
# Session-scoped patch
curl -H "X-Session-Id: sess-a" -X POST http://localhost:8080/v1/patches/apply -d '{"patch": {...}}'
```

#### GET /v1/patches

List active patches (session-aware).

#### DELETE /v1/patches/{name}

Remove a patch by description (session-aware).

### Management Endpoints

#### GET /v1/health

Always accessible (exempt from API key auth).

```json
{"status": "ok", "uptime_seconds": 3600, "requests_served": 12450}
```

#### GET /v1/models

```json
{"models": [{"id": "gemma-3-4b-it", "path": "/v1", "features": 348160, "loaded": true}]}
```

## Authentication

When `--api-key` is set, all endpoints (except `/v1/health`) require a Bearer token:

```bash
larql serve output/gemma3-4b.vindex --api-key "sk-abc123"
```

```bash
curl -H "Authorization: Bearer sk-abc123" "http://localhost:8080/v1/describe?entity=France"
```

Requests without a valid token receive 401 Unauthorized.

## Rate Limiting

Per-IP token bucket rate limiting. Supports `N/sec`, `N/min`, `N/hour` formats. `/v1/health` is exempt.

```bash
larql serve output/gemma3-4b.vindex --rate-limit "100/min"
```

Excess requests receive `429 Too Many Requests`. The limiter also respects `X-Forwarded-For` headers for clients behind proxies.

## DESCRIBE Cache

Cache DESCRIBE responses in memory with a configurable TTL. Useful for popular entities queried repeatedly.

```bash
larql serve output/gemma3-4b.vindex --cache-ttl 300  # 5 minute cache
```

Cache keys include: model ID, entity, band, limit, min_score. Expired entries are evicted automatically.

## Sessions

Per-session patch isolation via the `X-Session-Id` header. Each session gets its own PatchedVindex overlay — patches applied to one session don't affect others or the global state.

```bash
# Session A applies a medical patch
curl -H "X-Session-Id: sess-a" \
     -X POST http://localhost:8080/v1/patches/apply \
     -d '{"patch": {"version": 1, ...}}'

# Session A sees patched edges
curl -H "X-Session-Id: sess-a" "http://localhost:8080/v1/describe?entity=aspirin"

# Session B (or no header) sees unpatched edges
curl "http://localhost:8080/v1/describe?entity=aspirin"
```

Sessions expire after 1 hour of inactivity. Without an `X-Session-Id` header, patches go to the global (shared) state.

## Error Codes

| Status | When |
|--------|------|
| 200 | Success |
| 400 | Bad request (empty prompt, missing required params) |
| 401 | Unauthorized (missing/invalid API key) |
| 404 | Model not found (multi-model), patch not found |
| 429 | Rate limit exceeded |
| 503 | Inference unavailable (`--no-infer` or no model weights) |
| 500 | Internal server error |

All errors return `{"error": "message"}`.

## Layer Bands

Models are divided into three functional bands based on architecture:

| Band | Gemma 3 4B | What it encodes |
|------|-----------|-----------------|
| `syntax` | L0-13 | Morphology, grammar, function words |
| `knowledge` | L14-27 | Factual relations (capital, language, etc.) |
| `output` | L28-33 | Answer formatting, token selection |

DESCRIBE defaults to `band=knowledge`. Use `band=all` to scan everything.

## Probe Labels

If the vindex contains `feature_labels.json`, probe-confirmed relation labels appear in DESCRIBE and WALK responses:

```json
{"relation": "capital", "target": "Paris", "gate_score": 1436.9, "layer": 27, "source": "probe"}
```

Format of `feature_labels.json`: `{"L27_F9515": "capital", "L24_F4532": "language", ...}`

SELECT also supports filtering by relation: `{"relation": "capital"}` returns only edges with that probe label.

## REPL Integration

The LQL REPL connects to a remote server transparently:

```sql
USE REMOTE "http://localhost:8080";
-- Connected: google/gemma-3-4b-it (34 layers, 348160 features)

DESCRIBE "France";
WALK "Einstein" TOP 10;
INFER "The capital of France is" TOP 5;
STATS;
SHOW RELATIONS;

-- Mutations forwarded to server as patches
INSERT ("aspirin", "treats", "headache");

-- Local patches stay client-side (server never sees them)
APPLY PATCH "private-facts.vlp";
DESCRIBE "aspirin";          -- merges server + local patch edges
SHOW PATCHES;                -- shows local patches only
```

## Multi-Model Serving

When using `--dir`, each vindex gets its own namespace:

```bash
larql serve --dir ./vindexes/ --port 8080
# /v1/gemma-3-4b-it/describe, /v1/llama-3-8b/describe, ...
```

```
GET /v1/gemma-3-4b-it/describe?entity=France
GET /v1/llama-3-8b/describe?entity=France
```

## Crate Structure

```
larql-server/
├── Cargo.toml
├── README.md
├── examples/
│   ├── server_demo.rs          Synthetic vindex API demo
│   └── server_bench.rs         Endpoint latency benchmarks
├── proto/
│   └── vindex.proto            gRPC service definitions
├── build.rs                    Proto compilation (bundled protoc)
├── tests/
│   └── test_api.rs             Integration tests (107 tests)
└── src/
    ├── main.rs                 CLI parsing, vindex loading, server startup
    ├── state.rs                AppState: loaded models, probe labels, lazy weights
    ├── error.rs                ServerError → HTTP status codes
    ├── auth.rs                 API key Bearer token middleware
    ├── ratelimit.rs            Per-IP token bucket rate limiting
    ├── cache.rs                TTL cache for DESCRIBE results
    ├── session.rs              Per-session PatchedVindex isolation
    ├── etag.rs                 ETag generation for CDN caching
    ├── grpc.rs                 gRPC service (tonic, all endpoints)
    └── routes/
        ├── mod.rs              Router setup (single + multi-model)
        ├── describe.rs         GET /v1/describe (cached, ETag, relation labels)
        ├── walk.rs             GET /v1/walk (with relation labels)
        ├── select.rs           POST /v1/select (relation filter)
        ├── relations.rs        GET /v1/relations
        ├── stats.rs            GET /v1/stats
        ├── infer.rs            POST /v1/infer (walk/dense/compare)
        ├── stream.rs           WS /v1/stream (layer-by-layer streaming)
        ├── walk_ffn.rs         POST /v1/walk-ffn (decoupled inference)
        ├── patches.rs          POST/GET/DELETE /v1/patches (session-aware)
        ├── health.rs           GET /v1/health
        └── models.rs           GET /v1/models
```

## Dependencies

- `larql-vindex` — vector index loading, gate KNN, walk, patches
- `larql-inference` — forward pass, WalkFfn, dense FFN
- `axum` — HTTP framework
- `axum-server` — TLS support (rustls)
- `tokio` — async runtime
- `tonic` + `prost` — gRPC server and protobuf
- `tower` — concurrency limit middleware
- `tower-http` — CORS, tracing middleware
- `clap` — CLI argument parsing

## Testing

```bash
# Unit/integration tests
cargo test -p larql-server

# Demo (synthetic data, no real vindex needed)
cargo run -p larql-server --example server_demo

# Benchmarks (synthetic data)
cargo run -p larql-server --example server_bench --release
```

## Deployment

### Docker

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

### Systemd

```ini
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

Browse-only (f16): ~3 GB RAM. No GPU needed.

## License

Apache-2.0
