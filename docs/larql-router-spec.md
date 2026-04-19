# larql-router — Layer-Sharding FFN Proxy

**Version:** 0.1  
**Status:** Implemented (layer sharding, HTTP fan-out)  
**Implementation:** `crates/larql-router`  
**ADR:** `docs/adr/0003-ffn-router.md`  
**See also:** `docs/vindex-server-spec.md §13`, `docs/ffn/distributed.md`

---

## 1. Purpose

`larql-router` is a transparent HTTP proxy that sits between a client running
`RemoteWalkBackend` and a set of layer-sharded `larql-server` instances. The client
connects to one endpoint and sends one layer at a time; the router owns the sharding
map and routes each request to the correct backend.

```
Client  (attention locally, --ffn-remote http://router:9090)
  │
  ▼
larql-router:9090
  │  layers 0–16   →  larql-server A:8080
  │  layers 17–33  →  larql-server B:8081
```

The client has no knowledge of the shard topology. The `RemoteWalkBackend` is
unchanged — same flag, same request format, same response format.

---

## 2. Quickstart

```bash
# Start two layer-sharded FFN servers
larql-server output/gemma3-4b-q4k.vindex --ffn-only --layers 0-16  --port 8080
larql-server output/gemma3-4b-q4k.vindex --ffn-only --layers 17-33 --port 8081

# Start the router
larql-router \
  --shards "0-16=http://127.0.0.1:8080,17-33=http://127.0.0.1:8081" \
  --port 9090

# Client — unchanged
larql walk output/gemma3-4b-q4k.vindex \
    --ffn-remote http://127.0.0.1:9090 \
    --predict --prompt "The capital of France is"
```

**Startup output:**

```
larql-router v0.1.0
Shard map:
  layers 0–16:  http://127.0.0.1:8080  ✓
  layers 17–33: http://127.0.0.1:8081  ✓
Listening: http://0.0.0.0:9090
```

---

## 3. CLI Options

```bash
larql-router --shards <SPEC> [OPTIONS]
# or
larql-router route --shards <SPEC> [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--shards <SPEC>` | required | Comma-separated `START-END=URL` entries (inclusive bounds). Example: `"0-16=http://a:8080,17-33=http://b:8081"` |
| `--port <PORT>` | 9090 | Listen port |
| `--host <ADDR>` | 0.0.0.0 | Bind address |
| `--timeout-secs <N>` | 120 | Per-request timeout to backend shards |
| `--log-level <LEVEL>` | info | Tracing log level |

### Shard spec format

```
"START-END=URL[,START-END=URL,...]"
```

- `START` and `END` are **inclusive** layer indices.
- `URL` is the full base URL of the shard server, without a trailing slash.
- Whitespace around `-`, `=`, and `,` is ignored.

Examples:

```
"0-16=http://host-a:8080"
"0-16=http://host-a:8080,17-33=http://host-b:8081"
"0-20=http://host-a:8080,21-41=http://host-b:8081,42-61=http://host-c:8082"
```

Ranges must not overlap. Gaps (layers with no owning shard) are allowed but any
request hitting a gap layer returns HTTP 400.

---

## 4. Endpoints

### POST /v1/walk-ffn

Proxies the `walk-ffn` protocol. Accepts the same request body as `larql-server`.

**Single-layer request:** proxied directly to the owning shard. The request body is
forwarded unchanged.

```json
POST /v1/walk-ffn
{"layer": 5, "residual": [...], "full_output": true}
→ (proxied to shard owning layer 5)
← {"layer": 5, "output": [...], "latency_ms": 12.3}
```

**Batched request:** layers are grouped by owning shard. Each shard receives a
sub-request containing only its layers. All sub-requests are dispatched in parallel.
Results are merged and sorted by layer index.

```json
POST /v1/walk-ffn
{"layers": [5, 20], "residual": [...]}

→ parallel:
    shard A: {"layer":  5, "residual": [...]}
    shard B: {"layer": 20, "residual": [...]}

← {"results": [{"layer": 5, ...}, {"layer": 20, ...}], "latency_ms": 14.3}
```

Wall-clock latency for a parallel fan-out equals `max(shard_latencies)`, not the sum.

**Error cases:**

| Condition | HTTP | Body |
|-----------|------|------|
| Layer has no owning shard | 400 | `{"error": "layer 99 has no owning shard in this router"}` |
| Neither `layer` nor `layers` provided | 400 | `{"error": "must provide 'layer' or 'layers'"}` |
| `layers` array is empty | 400 | `{"error": "empty layer list"}` |
| Shard unreachable | 502 | `{"error": "shard http://...: ..."}` |
| Shard returns error | forwarded | Shard error body and status code passed through |

### GET /v1/health

```json
GET /v1/health
→ {"status": "ok"}
```

Always returns 200. Does not re-check shard health on each call.

---

## 5. Dispatch Logic

```
Receive POST /v1/walk-ffn

1. Parse layer or layers from request body
2. Validate: all layers have an owning shard → 400 if not
3. If single layer:
     forward body unchanged to owning shard → return response
4. If multiple layers:
     group by owning shard URL
     for each group, build sub-request:
       - single layer in group  → {"layer": N, ...}
       - multiple layers in group → {"layers": [N, M, ...], ...}
     dispatch all sub-requests in parallel (tokio::spawn per shard)
     await all responses
     merge results arrays, sort by layer
     return {"results": [...], "latency_ms": max_shard_latency}
```

The residual is not modified in transit. Each shard receives the same residual the
client sent. Layer-filtered sub-requests share the residual by reference in memory
(the JSON body is cloned per shard group, not per layer).

---

## 6. Health Checks

On startup, the router calls `GET /v1/stats` on each configured shard. Shards that
respond with HTTP 200 are marked healthy in the log (`✓`). Unreachable shards are
logged as warnings (`✗ UNREACHABLE`) but do not block startup.

```
Shard map:
  layers 0–16:  http://127.0.0.1:8080  ✓
  layers 17–33: http://127.0.0.1:8081  ✗ UNREACHABLE
  Warning: Shard http://127.0.0.1:8081 is not reachable — requests to its layers will fail
```

Requests to an unreachable shard return HTTP 502.

There is no periodic re-check after startup. If a shard comes back online, the next
request to it will succeed automatically (reqwest uses a connection pool that retries
on connection failure). If a shard goes down after startup, requests to its layers
will start returning 502 until it recovers or the router is restarted with an updated
`--shards` spec.

---

## 7. Deployment Examples

### Two-shard local (Gemma 3 4B, 34 layers)

```bash
larql-server output/gemma3-4b-q4k.vindex --ffn-only --layers 0-16  --port 8080
larql-server output/gemma3-4b-q4k.vindex --ffn-only --layers 17-33 --port 8081
larql-router --shards "0-16=http://127.0.0.1:8080,17-33=http://127.0.0.1:8081" --port 9090
```

### Three-shard remote (Gemma 4 31B, 62 layers, ~11 GB per shard)

```bash
# On server-a
larql-server output/gemma4-31b-q4k.vindex --ffn-only --layers 0-20  --port 8080

# On server-b
larql-server output/gemma4-31b-q4k.vindex --ffn-only --layers 21-41 --port 8080

# On server-c
larql-server output/gemma4-31b-q4k.vindex --ffn-only --layers 42-61 --port 8080

# Router (any machine, minimal RAM — just a proxy)
larql-router \
  --shards "0-20=http://server-a:8080,21-41=http://server-b:8080,42-61=http://server-c:8080" \
  --port 9090
```

### Systemd service

```ini
[Unit]
Description=LARQL FFN Router
After=network.target

[Service]
ExecStart=/usr/local/bin/larql-router \
    --shards "0-16=http://shard-a:8080,17-33=http://shard-b:8081" \
    --port 9090
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 8. What the Router Does Not Do

These are out of scope for the current implementation and tracked in ADR-0003:

- **Live resharding** — changing the shard map without restarting the router
- **Automatic failover** — redirecting traffic when a shard goes down
- **L2 cache** — the router currently passes every request through to shards; it does
  not cache FFN outputs itself
- **gRPC transport** — fan-out currently uses HTTP/JSON; a future version will use
  raw f32 bytes over gRPC to eliminate serialisation overhead
- **MoE expert dispatch** — routing by expert ID for mixture-of-experts models

---

## 9. Crate Structure

```
crates/larql-router/
├── Cargo.toml
└── src/
    └── main.rs       parse_shards, ShardMap, handle_walk_ffn, proxy_to_shard
```

**Dependencies:** `axum`, `tokio`, `reqwest`, `serde_json`, `clap`, `tracing`,
`futures`. No `larql-*` dependencies — the router is a pure HTTP proxy with no
vindex knowledge.

---

## License

Apache-2.0
