# ADR-0003 — FFN Router: Transparent Dispatch Tier

**Status:** Accepted — Phase 1 (dense HTTP router) implemented  
**Depends on:** ADR-0002 (Three-Tier Cache)

---

## Context

ADR-0002 established the `RemoteWalkBackend` protocol: a client sends a residual, a
server returns an FFN delta. That works for a single server. It breaks down when:

1. **Expert weights exceed one machine** — a 26B-A4B MoE has 128 experts × 2 active
   per token. Holding all experts on one host requires ~60GB; commodity machines top
   out at 16–32GB.

2. **Serial layer latency compounds** — 62 layers × 38ms RTT = 2.4s per token on a
   real network. The serial path cannot improve without a dispatch tier that can
   pipeline or merge requests.

3. **Resharding requires client changes** — if an expert server goes down or a new
   machine is added, today every client must be reconfigured. That is operationally
   brittle.

All three are solved by inserting a router between the client and the expert servers.
The client continues to use the existing `RemoteWalkBackend` protocol unchanged. The
router owns dispatch, merging, caching, and resharding.

---

## Decision

Add a `larql-router` process that sits between clients and expert servers. The router:

- Exposes the same `VindexService.WalkFfn` RPC the client already uses — **no client
  changes**.
- Reads model topology from a stripped "router vindex" (`index.json` +
  `router_weights.bin`) — **no hardcoded model knowledge**.
- For dense models: routes each layer to its owning shard by layer range.
- For MoE models: runs `RouterIndex::route` to select top-K experts, fans out in
  parallel, weighted-sums the deltas.
- Maintains an `Arc<RwLock<ShardMap>>` that can be atomically swapped during live
  resharding without interrupting in-flight requests.
- Acts as the L2 cache — residuals that hit the cache never reach expert servers.

---

## Architecture

```
Client
  │
  │  gRPC: VindexService.WalkFfn (same as today)
  ▼
┌──────────────────────────────────────────┐
│  larql-router                            │
│                                          │
│  1. L2 cache lookup (gate-KNN key)       │
│     hit → return, skip expert servers    │
│                                          │
│  2. RouterIndex::route(layer, residual)  │
│     → expert_ids + probs  (MoE only)     │
│     → layer range lookup  (dense)        │
│                                          │
│  3. Parallel gRPC fan-out to shards      │
│     ExpertService.FfnExpert per shard    │
│                                          │
│  4. Merge: Σ(prob_i × delta_i)          │
│     + shared_expert delta if present     │
│                                          │
│  5. L2 cache insert, return delta        │
└────────┬──────────┬────────────┬─────────┘
         │          │            │
         │ gRPC     │ gRPC       │ gRPC
         ▼          ▼            ▼
   [Expert A]  [Expert B]  [Expert C]
   experts 0–63 experts 64–127  experts 128–191
   layers 0–31  layers 0–31     layers 0–31
```

---

## Proto Definition

Two services on two different ports (or the same port with service routing).

### Client-facing (unchanged)

The router exposes the existing `VindexService.WalkFfn` RPC from `vindex.proto` —
the same message types the `RemoteWalkBackend` already uses. No proto changes needed
on the client side.

### Router ↔ Expert: `expert.proto`

```protobuf
syntax = "proto3";
package larql.expert;

service ExpertService {
  // Single expert FFN forward. Router sends residual + expert selection;
  // expert returns the weighted contribution.
  rpc FfnExpert(ExpertRequest) returns (ExpertResponse);

  // Batch: multiple (layer, expert_id) pairs in one round trip.
  // Eliminates N-1 RTTs when router owns a layer range on one expert server.
  rpc FfnExpertBatch(ExpertBatchRequest) returns (ExpertBatchResponse);

  // Health check for router use.
  rpc Health(HealthRequest) returns (HealthResponse);
}

// Sent by router to one expert server for one (layer, expert) computation.
message ExpertRequest {
  uint32 layer     = 1;
  uint32 expert_id = 2;   // 0 for dense models (no MoE), ignored on server side
  bytes  residual  = 3;   // raw f32 little-endian, length = seq_len * hidden_size * 4
  uint32 seq_len   = 4;
}

// Returned by expert server: the unscaled FFN output for this expert.
// Router applies routing_weight before summing across experts.
message ExpertResponse {
  uint32 layer          = 1;
  uint32 expert_id      = 2;
  bytes  delta          = 3;   // raw f32, same shape as residual
  float  routing_weight = 4;   // echoed from router's softmax for verification
  float  latency_ms     = 5;
}

message ExpertBatchRequest {
  repeated ExpertRequest requests = 1;
}

message ExpertBatchResponse {
  repeated ExpertResponse responses = 1;
  float latency_ms = 2;
}

message HealthRequest {}
message HealthResponse {
  string status           = 1;
  uint32 expert_range_lo  = 2;
  uint32 expert_range_hi  = 3;
  uint32 layer_range_lo   = 4;
  uint32 layer_range_hi   = 5;
  uint64 requests_served  = 6;
}
```

**Wire format for residual/delta:** raw little-endian f32 bytes, not repeated float.
This is ~3× smaller on the wire than proto's `repeated float` varint encoding and
avoids an extra copy on encode/decode.

---

## Dispatch Logic

The router is model-agnostic. All topology is derived from `VindexConfig` at startup.

```rust
fn dispatch_layer(
    &self,
    layer: usize,
    residual: &[f32],
    seq_len: usize,
) -> Result<Vec<f32>, RouterError> {

    // 1. L2 cache lookup
    if let Some(cached) = self.l2_cache.get(layer, residual) {
        return Ok(cached);
    }

    // 2. Route — model-agnostic branch
    let delta = if let Some(ref router_idx) = self.router_index {
        // MoE path: RouterIndex knows num_experts and top_k from VindexConfig
        let residual_1d = ndarray::Array1::from_vec(residual[..self.hidden_size].to_vec());
        let route = router_idx.route(layer, &residual_1d)
            .ok_or(RouterError::RoutingFailed(layer))?;

        // Fan out to top-K expert shards in parallel
        let futures: Vec<_> = route.experts.iter().zip(route.probs.iter())
            .map(|(&expert_id, &prob)| {
                let shard = self.shard_map.read().find_expert(layer, expert_id)?;
                shard.ffn_expert(layer, expert_id, residual, seq_len, prob)
            })
            .collect();

        let expert_deltas = futures::future::join_all(futures).await;

        // Weighted sum: delta = Σ prob_i * expert_delta_i
        let mut merged = vec![0f32; residual.len()];
        for (delta_i, prob_i) in expert_deltas {
            for (d, v) in merged.iter_mut().zip(delta_i.iter()) {
                *d += prob_i * v;
            }
        }

        // Shared expert (e.g. DeepSeek, Qwen MoE): always active, weight=1.0
        if self.config.moe.as_ref().map_or(false, |m| m.shared_expert) {
            let shared_shard = self.shard_map.read().find_shared_expert(layer)?;
            let shared_delta = shared_shard.ffn_expert(layer, SHARED_EXPERT_ID, residual, seq_len, 1.0).await?;
            for (d, v) in merged.iter_mut().zip(shared_delta.iter()) {
                *d += v;
            }
        }

        merged
    } else {
        // Dense path: one shard per layer range, full FFN output
        let shard = self.shard_map.read().find_layer(layer)?;
        shard.ffn_expert(layer, 0, residual, seq_len, 1.0).await?
    };

    // 3. L2 cache insert
    self.l2_cache.insert(layer, residual, delta.clone());

    Ok(delta)
}
```

The `RouterIndex` struct lives in `larql-vindex::index::router` and is already
model-agnostic — it reads `num_experts` and `top_k` from `VindexConfig.model_config.moe`.
The dispatch logic above calls it directly; the router crate takes `larql-vindex` as a
dependency.

---

## Shard Map

```rust
pub struct ShardEntry {
    pub layer_start: usize,       // inclusive
    pub layer_end: usize,         // exclusive
    pub expert_start: Option<usize>, // None = whole expert range (dense or shared)
    pub expert_end: Option<usize>,
    pub url: String,              // grpc://host:port
    pub channel: ExpertServiceClient<Channel>,  // pooled gRPC channel
}

pub struct ShardMap {
    pub entries: Vec<ShardEntry>,
}

impl ShardMap {
    pub fn find_layer(&self, layer: usize) -> Option<&ShardEntry> {
        self.entries.iter().find(|e| layer >= e.layer_start && layer < e.layer_end)
    }

    pub fn find_expert(&self, layer: usize, expert_id: usize) -> Option<&ShardEntry> {
        self.entries.iter().find(|e|
            layer >= e.layer_start && layer < e.layer_end
            && e.expert_start.map_or(true, |s| expert_id >= s)
            && e.expert_end.map_or(true, |end| expert_id < end)
        )
    }
}
```

---

## Router Vindex (Stripped Client Vindex)

The router loads a stripped vindex containing only what it needs:

```
gemma4-26b-a4b.router.vindex/
  index.json             # VindexConfig: num_layers, hidden_size, moe config
  router_weights.bin     # RouterIndex weights (MoE only; absent for dense)
  tokenizer.json         # For future L2 cache seeding by token
```

No `gate_vectors.bin`, no FFN weights. This is ~20MB for a 26B-A4B model
(128 experts × 2560 hidden × 62 layers × 4 bytes = ~80MB at f32; the existing
`router_weights.bin` layout stores `[num_experts × hidden_size + num_experts]` per
layer as already implemented in `RouterIndex::load`).

The router reads this at startup via the existing `VectorIndex::load_vindex_with_range`
path with all layers owned — but since gate/FFN files are absent, only `index.json`
and `router_weights.bin` are mmapped.

---

## Reshard Protocol

The sharding map is wrapped in `Arc<RwLock<ShardMap>>`. Resharding is an atomic
write-lock swap. In-flight requests hold a read guard on the old map until they
complete; no request is interrupted.

```bash
# Add a new expert server at runtime
$ larql-router reshard \
    --add "layers=0-31,experts=192-255,url=grpc://expert-d:50055"

# Remove a failed server (traffic migrates to remaining shards)
$ larql-router reshard \
    --remove "grpc://expert-b:50053" \
    --replace "layers=0-31,experts=0-127,url=grpc://expert-a:50052"
```

The reshard command sends a `ReshardRequest` to the router's admin gRPC port:

```protobuf
service RouterAdmin {
  rpc Reshard(ReshardRequest) returns (ReshardResponse);
  rpc ShardStatus(ShardStatusRequest) returns (ShardStatusResponse);
}

message ReshardRequest {
  repeated ShardSpec add    = 1;
  repeated string    remove = 2;  // URLs to remove
}

message ShardSpec {
  uint32 layer_start   = 1;
  uint32 layer_end     = 2;
  uint32 expert_start  = 3;
  uint32 expert_end    = 4;
  string url           = 5;
}

message ReshardResponse {
  bool   success       = 1;
  uint32 shards_active = 2;
  string error         = 3;
}
```

The router performs a health check on any new shard before adding it to the map.
Removal takes effect immediately; the old channel drains in-flight requests.

---

## Cache Integration

The router is the natural L2 cache position: it sees every residual for every layer
before any expert server is contacted.

Cache key: same scheme as ADR-0002 — `hash(sorted gate-KNN feature IDs)`. For the
router, gate KNN is run against the router's local copy of gate vectors from the
stripped vindex. Wait — the stripped vindex has no gate vectors. Two options:

**Option A (preferred):** Router caches on `hash(quantised_residual)`. Quantise
residual to i8 before hashing (reduces 2560 × 4 = 10KB to 2560 bytes; accepts
small hash collision rate). No gate KNN needed at the router.

**Option B:** Include `gate_vectors.bin` in the router vindex. Adds ~6GB for a 26B
model. Enables the exact ADR-0002 key. Deferred until collision rate on Option A is
measured.

```
Router receives residual for layer L:
  key = hash(quantise_i8(residual))
  L2_CACHE[L].get(key)?
    hit  → return cached delta, skip all expert servers
    miss → dispatch to experts, L2_CACHE[L].insert(key, delta)
```

Cache lives in the router process for its lifetime. On reshard, entries for layers
owned by the removed shard remain valid (the delta is correct regardless of which
server computed it).

---

## Configuration

```toml
# larql-router.toml
vindex  = "output/gemma4-26b-a4b.router.vindex"
port    = 50051
admin_port = 50052

[cache]
max_entries_per_layer = 4096   # L2 cache cap

[[shards]]
layers  = "0-31"
experts = "0-63"
url     = "grpc://expert-a:50052"

[[shards]]
layers  = "0-31"
experts = "64-127"
url     = "grpc://expert-b:50053"

[[shards]]
layers  = "32-61"
experts = "0-63"
url     = "grpc://expert-c:50054"

[[shards]]
layers  = "32-61"
experts = "64-127"
url     = "grpc://expert-d:50055"
```

Dense model example (no MoE, layer sharding only):

```toml
vindex = "output/gemma3-4b.router.vindex"
port   = 50051

[[shards]]
layers = "0-16"
url    = "grpc://shard-a:50052"

[[shards]]
layers = "17-33"
url    = "grpc://shard-b:50053"
```

The router infers from `VindexConfig.model_config.moe` whether expert dispatch is
needed. If `moe` is absent or None, the dense path is used automatically.

---

## Client Usage

```bash
# Client: unchanged except http:// → grpc://
$ larql-cli predict \
    --model google/gemma-4-26B-a4b \
    --vindex output/gemma4-26b-a4b.client.vindex \
    --ffn-remote grpc://router:50051 \
    --prompt "The capital of France is"
```

`RemoteWalkBackend` needs one change: detect `grpc://` prefix and use the tonic
client instead of reqwest. The wire format exposed by the router is `VindexService.WalkFfn`
from the existing proto — the message types are identical.

---

## Expert Server Changes

Expert servers (`larql-server --ffn-only --layers N-M`) need one addition: implement
`ExpertService.FfnExpert` alongside the existing HTTP endpoint. The handler is a thin
wrapper over the existing `run_full_output` logic, selecting the right expert's weight
slice when `expert_id` is provided.

For dense servers (`expert_id == 0`), the handler is identical to the current
`walk_ffn` handler. No behaviour change.

For MoE servers, the server must load the weights for its expert range only. Expert
weight selection uses the existing `--layers` flag for layer range; a new
`--experts N-M` flag selects which expert weight rows to load from the interleaved
weight file. Both flags compose independently.

---

## Implementation Plan

### Phase 1 — Dense router (1 week)

- New `crates/larql-router` binary crate
- Load stripped vindex (index.json + tokenizer only for dense)
- TOML config parser
- `ShardMap` + `find_layer` dispatch
- Forward `VindexService.WalkFfn` requests to the owning layer shard via HTTP (reuse
  existing `RemoteWalkBackend` logic)
- `RouterAdmin.Reshard` endpoint (add/remove shards)
- Health check on shard add

### Phase 2 — gRPC expert protocol (1 week)

- `expert.proto` + tonic codegen
- `ExpertService.FfnExpert` + `FfnExpertBatch` handlers in `larql-server`
- Router uses gRPC to expert servers instead of HTTP
- Raw f32 bytes wire format for residual/delta
- `grpc://` prefix detection in `RemoteWalkBackend`

### Phase 3 — MoE dispatch (1 week)

- Router loads `router_weights.bin` via `RouterIndex::load`
- `find_expert` dispatch for MoE models
- Parallel fan-out + weighted sum
- `shared_expert` support (add per `MoeConfig.shared_expert`)

### Phase 4 — Router L2 cache (3 days)

- i8 quantised residual hash key (Option A)
- `FfnL2Cache` instance in router (same type as ADR-0002)
- Measure hit rate; evaluate whether to include gate vectors in router vindex

### Phase 5 — Live reshard demo (2 days)

- `larql-router reshard --remove / --add` CLI
- Admin RPC handler
- Test: kill one expert server mid-generation, reshard to surviving server, zero
  client interruption

---

## Open Questions

1. **L2 cache key — quantised residual vs gate-KNN.** Option A (i8 residual hash)
   avoids adding gate vectors to the router vindex but has a small collision risk.
   Option B is exact but adds ~6GB. Measure Option A collision rate on a benchmark
   query set before committing.

2. **Streaming vs round-trip for multi-layer batches.** The router could hold all
   layer residuals from the forward pass and send them to each shard in one
   `FfnExpertBatch` call, waiting for the shard to return all deltas. This eliminates
   one RTT per shard but requires the router to buffer residuals. For a 3-shard layer
   partition that's 3 RTTs → 1 RTT + compute(N layers). Worth it on LAN, marginal
   on WAN.

3. **Fault mode during reshard.** The current design lets in-flight requests drain on
   the old map. If an expert server dies without a clean remove, those requests will
   fail. Add a per-request retry that re-reads the shard map on failure.

4. **Expert weight file layout for MoE servers.** The existing `interleaved_q4k.bin`
   interleaves layers sequentially. For an expert-sharded server, the file should
   interleave by expert within each layer. This is a vindex extraction change, not a
   router change, but it must land before Phase 3.

5. **Router vindex vs inline config.** Should the router read topology from a vindex
   directory (reusing all existing loading infrastructure) or from the TOML config
   alone? The vindex approach is cleaner for MoE (router_weights.bin already has the
   right layout) but adds a load step. TOML-only works for dense. Decision: vindex for
   MoE, TOML-only for dense.
