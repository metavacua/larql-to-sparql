---
name: FFN router + layer sharding shipped
description: Layer-sharded vindex loading and transparent HTTP router both working as of 2026-04-19
type: project
---

Layer sharding + router landed on `architecture-b` branch.

**Why:** Memory-limited servers can't hold a full vindex. Sharding keeps RSS proportional to the layer slice served.

**What shipped:**

1. `larql-vindex`: `VectorIndex::load_vindex_with_range(dir, callbacks, Option<(usize, usize)>)` — restricts mmap page touches and anon allocation to owned layers. `is_layer_owned(layer)` guards accessors. `synthesize_gate_from_q4k` only dequantizes owned layers.

2. `larql-server`: `--layers START-END` (inclusive) — passes range to `load_vindex_with_range`. Unowned layer requests return HTTP 400 with `"layer N not served by this shard (owned: X–Y)"`.

3. `larql-router`: new `crates/larql-router` binary. `--shards "0-16=http://host-a:8080,17-33=http://host-b:8081"`. Transparent HTTP proxy — client uses `RemoteWalkBackend` unchanged (`--ffn-remote http://router:9190`). Single-layer requests proxied directly. Batched `layers:[...]` requests split by shard, fanned out in parallel, merged in order. Health-checks each shard on startup.

**How to apply:** Next step per ADR-0003 is Phase 2 (gRPC + ExpertService proto). MoE expert dispatch is Phase 3 — blocked on expert-major layout in vindex extraction.
