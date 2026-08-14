# Roadmap — larql-router / larql-router-protocol

For shipped work, see [CHANGELOG.md](CHANGELOG.md).

## Current state (verified 2026-08-04)

**Tests.** 219 tests passing across `larql-router` and
`larql-router-protocol`.

**Layout.** 23 source files. The library surface is
`larql_router::{grid, tasks, dispatch, shards, http, admin, cli_helpers}`.
`grid/` holds one file per concern (`mod` state core, `routing`, `replication`,
`hot_shard`, `status`, `service` gRPC impl, and a `#[cfg(test)] testing`
helper); `tasks/` holds `rebalancer/` (6 sub-files) alongside `rtt_probe.rs`,
signalling both as long-lived background tasks spawned at router startup.

**Coverage.** Enforced: 90% per file; totals at 91% (router) and 90%
(router-protocol). One debt baseline each — `grid/service.rs` at 88% (the
gRPC streaming join handler; the residual gap is mainly unreachable Mode B
gap-fill code and tx-send-failure races) and, **new since the 2026-05-16
snapshot**, `transport/h3.rs` at 89% in router-protocol. The old snapshot
recorded router-protocol as having zero debt baselines; that is no longer true.

```bash
make larql-router-coverage-summary
make larql-router-protocol-coverage-summary
```

**Performance.** The last full measurement is the 2026-05-16 M3 Max snapshot in
[`CHANGELOG.md`](CHANGELOG.md). Its load-bearing conclusion — `route()` is
essentially flat at ~110 ns across grid sizes, driven only by
`target_replicas`, so a 62-layer forward pass spends ~8 µs picking shards
against a 13.78 ms decode — has not been re-run since. Treat the absolute
numbers as a baseline to compare against, not as current. QUIC still has not
been benched against TCP on real workloads; `quic` is opt-in and outside the
default build.

### Architecture

Self-assembling grid is feature-complete across ADR-0004 Phase 1–5, ADR-0010
(QUIC), ADR-0011 (Mode B + Phase B2 drain-then-reassign + replication),
ADR-0012 Phase 2 (criterion micro-benchmarks, both worst-case + production-shape),
ADR-0013 (three-tier routing comparator + active-probe RTT),
ADR-0014 (hot-shard load-rate replication + two-threshold hysteresis amendment),
ADR-0015 (ShardService.Query KNN endpoint),
ADR-0016 (router module organization),
ADR-0017 (Prometheus `/metrics` endpoint, bounded-cardinality),
ADR-0018 (MoE expert routing — `route_expert` / `route_all_experts`,
per-(layer, expert-range) replication, JSON `experts` / `layer_experts`
HTTP shapes),
ADR-0019 (HTTP/3 shard transport, opt-in via `--http3-shards` and
`--http3-port`),
ADR-0020 (saturation-tier backpressure in `route()` —
`--saturation-ceiling N`, 503 with `Retry-After: 0.5`, distinguished
from 400 via `has_owners_for`, `larql_router_route_saturation_total`
counter), and
ADR-0021 (hedged dispatch — opt-in via `--hedge-after-ms M`,
`route_with_rank` / `route_expert_with_rank` accessors, races a
secondary replica against a slow primary when M ms elapses;
`route_hedge_fires_total` / `route_hedge_wins_total` counters).
Static `--shards` (ADR-0003) remains as a fallback and coexists with
the grid.

The codebase is architecture-agnostic: routing logic reads layer ranges,
`model_id`, and server state from the grid protocol — no model-family
constants are hardcoded.

### What works today

- **Mode A** — `AnnounceMsg` → `AckMsg` registration + heartbeat loop + reconnect.
- **Mode B (Phase B1 + B2)** — `AvailableMsg` → `AssignMsg` → `ReadyMsg`; servers
  re-enter the available pool after an `UnassignMsg`-driven drain on the same
  stream.
- **Replication** — `--target-replicas N`; under-replicated ranges pull spares
  from the available pool, over-replicated ranges drop the least-loaded
  replica via `UnassignMsg`. Origin URLs resolved from any live replica via
  `find_origin_for`.
- **Hot-shard load-rate replication** — `--hot-shard-rps THRESHOLD`; when a
  shard's max `HeartbeatMsg.req_per_sec` across replicas exceeds the
  threshold, the rebalancer treats it as effectively under-replicated
  (`target + 1`) and pulls a spare. The elevated flag clears when the rate
  drops; over-replication then prunes the surplus on the next tick.
- **Stale heartbeat eviction** — rebalancer evicts serving servers whose
  `last_seen` exceeds `stale_heartbeat_timeout` (default 25 s = 2.5 ×
  heartbeat interval).
- **Per-layer latency-aware routing (GT3)** — `route()` prefers the server
  with lowest `layer_latencies[layer].avg_ms`; falls back through
  active-probe RTT, then `requests_in_flight`.
- **Active-probe RTT routing** — opt-in via `--rtt-probe-interval-secs`.
  The probe loop `GET`s `{listen_url}/v1/health` against every serving
  server on the configured cadence and lands the round-trip on
  `ServerEntry.rtt_ms`. Used by `route()` as the middle tier of the
  three-tier comparator.
- **`GridService.Join`** bidirectional gRPC stream over TCP (default) or
  QUIC (`--features quic`).
- **QUIC transport (GT7)** — `--quic-port`, `--quic-cert`/`--quic-key` (or
  auto-generated self-signed cert), SHA-256 fingerprint pinning on the
  client side via `--quic-cert-fingerprint`. HTTP/2 carried over a single
  QUIC bi-stream; 0-RTT reconnect + TLS 1.3.
- **Admin CLI (Phase 5)** — `larql-router status` / `gaps` / `drain --server`
  / `assign --model M --layers A-B [--server S] [--origin-url URL]`. Backed
  by new `DrainServer` + `AssignRange` gRPC RPCs.
- `DroppingMsg` → deregistration + auto gap re-fill + auto re-replication.
- Static `--shards` mode with layer-range routing and per-shard parallel
  fan-out.
- Grid + static fallback via `AppState::resolve_all()`.
- `GET /grid-status` (served by `StatusResponse` with `layer_stats` per
  server).
- Auth: optional shared `--grid-key` Bearer token in gRPC metadata.
- Library crate (`larql_router::{grid, tasks, dispatch, shards, http,
  admin, cli_helpers}`) for tests and external consumers. `tasks` rolls
  up `rebalancer` (6 sub-files) and `rtt_probe`; `grid` rolls up
  `mod`, `routing`, `replication`, `hot_shard`, `status`, `service`,
  and a `#[cfg(test)] testing` helper.
- Examples — `examples/embed_grid.rs`, `examples/fanout_dispatch.rs`,
  `examples/static_shards_server.rs`, `examples/admin_client.rs`,
  `examples/saturation_backpressure.rs` (ADR-0020 — drives a
  `GridState` through five saturation/coverage scenarios and prints
  the routing-layer decision plus the HTTP status the dispatcher
  would emit).
- Criterion benchmarks (GT9 ✅) — `routing.rs` with ten groups:
  `route_single_layer` + `route_all` (worst-case full replication),
  `route_realistic` + `route_all_realistic` (production-shape
  contiguous shards × `target_replicas`), `route_expert_single` +
  `route_all_experts` (ADR-0018 MoE), `heartbeat_update`,
  `single_register` (per-join rebuild cost), `register_cascade` (N
  sequential joins — O(N²) cold-start measurement), and
  `saturation_filter` (ADR-0020 — no-filter / filter-all-unsat /
  filter-all-sat across 5, 10, 20 shards × 2 replicas).

### What is not yet implemented

- **Cross-router federation** — multi-region routing (P2).
  (MoE within-layer expert sharding — previously listed here as P2
  — shipped 2026-05-16 as ADR-0018; see [`CHANGELOG.md`](CHANGELOG.md)
  for `route_expert` / `route_all_experts` and the self-healing
  section below for per-(layer, expert-range) replication.)

---

## Open defects

Both raised by the 2026-05-28 whole-codebase review; neither is fixed.

- **P1 — announced layer ranges are not validated (DoS class).** The announce
  path builds an unbounded route table from gRPC-announced ranges with no
  validation. Confirmed still open 2026-08-04: `grid/routing.rs`'s
  `rebuild_route_table` iterates `entry.layer_start..=entry.layer_end`
  directly, and no clamp or sanity check exists on the insert path either — a
  single announce claiming a `u32`-wide span makes the router build a route
  table with billions of entries. Clamp the span to sane model depth before
  the entry is admitted, not just before `rebuild_route_table`.
- **`larql-router-protocol` — a `None` fingerprint disables TLS verification**
  on a public API. Document the contract or gate it behind an explicit opt-in.

---

## Next work — by theme

Items are tagged **P1** (active or next-up), **P2** (well-defined,
implementation sketch exists, 3-6 month horizon), **P3** (recognized
future work, no concrete plan yet). Everything surfaced during the
2026-05-16 doc/spec review is folded in.

P1 is **empty by default** — items move into P1 only when explicitly
chosen as next work. The candidate pool below is the menu.

---

### Theme: Dense model sharding

The router's bread-and-butter use case — pipeline-parallel models
across many hosts.

**Shipped:** ADR-0003 (static `--shards`), ADR-0004 P1–5
(self-assembling grid), ADR-0011 (Mode B + replication), ADR-0014
(hot-shard load-rate replication), ADR-0013 (routing comparator).

**P2 — well-defined, implementable:**

- **Auto-shard planner.** Given a `vindex` + N hosts with declared
  RAM budgets, compute a layer assignment that minimises shard-size
  variance under the per-host memory cap. Today the operator picks
  `--layers` manually per host; auto-plan would mean the router (or a
  one-shot `larql-router plan` command) emits a recommended map.
- **Heterogeneous-aware routing.** Server announce carries a `host_kind`
  hint (e.g. `gpu_metal`, `gpu_cuda`, `cpu`). The 3-tier comparator
  gains a 0th tier that prefers GPU hosts for compute-heavy layers
  (`lm_head`, attention) and CPU hosts for FFN-only shards. Extends
  ADR-0013 with a layer-kind classifier.
- **Mid-flight resharding without packet drop.** Today's
  drain-then-reassign (ADR-0011 Phase B2) is operator-driven via
  `admin assign`. P2 makes it traffic-driven: if a host's
  `ram_used / ram_total` exceeds a threshold AND a spare with more
  RAM exists in the available pool, the rebalancer initiates a
  drain-and-reassign on the smaller host. Requires safe handover —
  spare needs to be `Ready` before the original is `Unassign`ed.

**P3 — speculative:**

- **Tensor parallelism (single layer split across hosts).** Current
  model is layer-pipeline; tensor-parallel would split a single
  attention head across hosts. Major proto surgery (per-head IDs,
  partial-residual aggregation) — only worth it for models that
  don't fit on a single host even at one-layer-per-host granularity.
- **Cross-host KV cache reuse.** Attention layers cache K/V per
  prefix. If a prefix is shared across requests (system prompt,
  common preamble), routing same-prefix requests to the same host
  reuses the cache. Needs sticky session routing keyed on prefix
  hash.

---

### Theme: MoE model sharding and routing

**Shipped (ADR-0018, 2026-05-16):**

- **Proto extension** — `AnnounceMsg` / `ReadyMsg` / `AssignMsg`
  carry `expert_start` / `expert_end`. Dense servers send `0/0`;
  MoE shards advertise a contiguous expert range.
- **`ServerEntry::owns_expert(expert_id)` + `is_dense()`** — every
  helper that filters by expert ID short-circuits when the server is
  dense, so dense routing pays zero extra cost.
- **`route_expert(model, layer, expert_id)` + `route_all_experts`** —
  three-tier comparator (ADR-0013) over the filtered candidate set.
- **HTTP shape** — `/v1/walk-ffn` accepts `{layer, experts: [...]}`
  or `{layer_experts: [{layer, experts}, ...]}` alongside the
  existing dense shapes. MoE dispatch is grid-only; static `--shards`
  servers see a 503.
- **Replication keys widen to 5-tuples** — `under/over_replicated_ranges`,
  `find_origin_for`, `try_assign_gap`, `effective_target_for`,
  `send_assign_to_named_available`, `least_loaded_in_range` all key
  on `(model, layer_start, layer_end, expert_start, expert_end)`.
  Two shards sharing a layer range but owning different experts are
  treated as distinct slices.
- **Hot-shard elevation set widens** — `hot_layer_ranges`,
  `mark_elevated`, `demote_elevated`, `elevated_ranges_snapshot` all
  emit/take 5-tuples. Hot saturation on one expert-shard elevates
  only that shard, not its sibling.
- **`larql_router_grid_shard_kind{kind=dense|moe}`** — bounded-
  cardinality Prometheus gauge for grid-wide MoE health.
- **Coverage** — 19/20 files at 90%+ post-MoE + ADR-0020, total
  93.21% (`grid/service.rs` at 89.87% — within its 88% debt
  baseline; `main.rs` excluded from per-file).
- **Dense regression** — all 202 pre-MoE tests still green plus the
  post-MoE/ADR-0020/chaos additions (163 lib + 47 integration =
  210 tests, 211 with `--features http3`); bench shows dense
  `route()` within ±10% of the pre-MoE baseline (the expert filter
  is a single boolean check on a dense `ServerEntry`).

**Target deployment scale (per ADR-0018 §Target deployments):**
DeepSeek-V3 (671B / 60 layers × 256 experts), Kimi K2 / K2.6
(~1T-class), DeepSeek-V4 (≥1T). One physical host per (single
layer, expert-subset) shard; route table stays tractable because the
route_table is keyed on `(model, layer)` and expert filtering happens
inline.

**P2 — future MoE extensions:**

- **Expert affinity routing.** Same expert ID routes to same host
  repeatedly so the host's KV/MLP cache stays warm. Adds a 4th tier
  to the routing comparator. Deferred from ADR-0018 — needs real
  workload data showing the cache-warmth signal is meaningful.

**P3 — needs more discovery:**

- **Expert specialization with refusal.** Hosts may load a subset of
  experts and `Refuse` requests for experts they don't own. Today's
  `RefuseMsg` is for Mode B assignment refusal; expert-level refusal
  is a new semantic.
- **Binary wire format v2 with expert IDs.** Today's binary protocol
  is dense-only (ADR-0018 §"Binary protocol stays single-dimension").
  ADR-0009 (wire-format evolution) is the spec hook.
- **Admin RPC `AssignRangeRequest` expert fields.** Today's admin
  `assign` is dense-only. Additive proto change, no design surprises.

---

### Theme: Splitting large models (deployment-time concerns)

How an operator actually gets a 26B / 70B / 405B model running on a
heterogeneous cluster.

**Shipped:** Static `--shards` + Mode B available pool, multi-host
deploy walkthrough ([`crates/larql-router/docs/multi-host-demo.md`](docs/multi-host-demo.md)
— 3-box LAN topology covering router + 2 shards over `--grid-key`,
firewall rules, NTP, MTU gotchas, plus a QUIC variant for ADR-0010
and a MoE variant for V3/V4-scale models), vindex shard-download
endpoint ([`crates/larql-server/docs/router-spec.md`](../larql-server/docs/router-spec.md)
§4 — `GET /v1/shard/{model_id}/{start}-{end}` serves the vindex
directory as a streamed tar, client side at
`crates/larql-server/src/shard_loader.rs` is idempotent + SHA-256
verified + atomic-unpack, exercised end-to-end by
`crates/larql-server/tests/test_grid_mode_b.rs::mode_b_full_vertical_handoff`
against a real donor; 2026-05-16 audit closed the docs gap).

**P2 — extends auto-shard planner:**

- **Large-model bootstrap timeline.** Warm-up loading curve, vindex
  preload, attention buffer allocation under shard ownership. Today's
  Mode B path treats "Ready" as a single event; large models would
  benefit from progress reporting (`LoadingMsg { pct }`) so the
  rebalancer doesn't see a 10-minute load as a 10-minute stall.
- **Disk + RAM constraint solver.** Available-pool advertises
  `ram_bytes` and `disk_bytes` but `try_assign_gap` only checks RAM.
  Add disk gating so spares without enough disk for the vindex slice
  are skipped.

**P3 — future:**

- **Multi-vindex models** — different layers loaded from different
  `.vindex` files. Useful for fine-tuning experiments (swap one
  layer's weights to compare). Today each server loads exactly one
  vindex.

---

### Theme: Self-healing grid

Replication + gap-fill + stale eviction cover the happy reliability
paths. The gaps are in **partial-failure** and **adversarial-load**
scenarios.

**Shipped:** Stale-heartbeat eviction (ADR-0011), replication ticks,
gap-fill on Dropping/disconnect (ADR-0004 P2), Phase B2 drain-then-
reassign (ADR-0011), hot-shard load-rate replication (ADR-0014),
two-threshold hot-shard hysteresis (ADR-0014 amendment, demote at
0.8×T), backpressure filter in `route()` /
`route_expert()` (ADR-0020, `--saturation-ceiling N`,
`larql_router_route_saturation_total` counter, 503 with
`Retry-After: 0.5` on saturated dispatch), long-running chaos test
(`tests/test_grid_chaos.rs`, 5,000 random churn ticks × 2 variants,
asserts ledger consistency + coverage floor + no `route()` panic).

**P1 — reliability gaps surfaced in reviews:**

**P2:**

- **Multi-failure recovery scenarios.** Stress-test with N
  simultaneous failures (3+ servers crash at once). Today's
  rebalancer ticks every 30 s by default; in a 3-server-fail event
  the gap-fill and replicate paths fire in the same tick — verify
  ordering doesn't dispatch two AssignMsgs for the same range.
- **Network partition tolerance.** Router-server unreachable but
  server-server reachable. Today the router would deregister servers
  it can't see. A "partition-suspected" mode could hold deregistration
  for K seconds to avoid mass-eviction on a switch flap.
- **Cascade-failure isolation.** A slow shard backs up requests
  upstream; without a hop-budget circuit-breaker the slow shard's
  upstream peers also slow down. Add a fail-fast hop budget to
  `walk-ffn`.

**P3:**

- **Split-brain protection for multi-router deployments.** Two
  routers both think they're authoritative for the same grid; they
  could send conflicting `AssignMsg` to the same available server.
  Resolution needs either consensus (raft over a small router set)
  or sticky-leader (one router authoritative per `model_id`).

---

### Theme: Latency (router on the hot path)

Today's per-call wire RTT (`README.md` snapshot): TCP HTTP ~660 µs,
UDS HTTP ~510 µs, gRPC streaming ~460 µs. Across a 30-layer model
sharded into 2 hosts (15 hops × 2 = 30 layers serial), wire alone is
~14 ms — a meaningful chunk of decode time.

**Shipped:** 3-tier route() (ADR-0013), GT3 layer-latency in
heartbeats, active-probe RTT, 110 ns route() in production-shape
benches, connection pool tuning, real HTTP/3 shard transport with
per-stream independence (ADR-0019, 2026-05-16 — `--http3-shards` /
`--http3-port` opt-in, `H3Client::post_json` + `serve_axum` in
larql-router-protocol, used by the MoE expert fan-out path when
`h3_client: Some(_)`), hedged dispatch (ADR-0021, 2026-05-16 —
opt-in via `--hedge-after-ms M`; the multi-shard fan-out picks a
secondary replica per sub-request and dispatches it M ms after the
primary if the primary hasn't responded; halves p99 tail latency in
topologies with `--target-replicas ≥ 2`).

**P1 — biggest near-term win:**

- **(Pre-ADR-0021 "speculative next-layer prefetch" — falsified.)**
  An audit during the 2026-05-16 session found that the inference
  side sends one batched `/v1/walk-ffn` per token with the full
  layer list against a single input residual; the router fans every
  sub-request out in parallel against that input. There is no
  layer-N → layer-N+1 dependency at the router boundary, so
  "prefetch layer N+1 while N is in flight" doesn't apply here.
  Cross-token speculation, if it lands, is a client-side
  (`larql-inference`) concern. The legitimate router-layer
  interpretation is hedged dispatch — that shipped as ADR-0021
  (see Shipped above).

**P2:**

- **Wire RTT budget audit.** Real measurement of where the 460 µs is
  going (gRPC framing, TLS, socket queueing, axum middleware).
  Likely yields actionable per-stage optimisations.
- **Connection-pool tuning at scale.** Current
  `pool_max_idle_per_host(16)` was chosen for 2-shard deployments.
  At 20+ shards the pool churn dominates. Auto-size based on observed
  shard count?
- **Native UDS for same-host shards.** When router + server are on
  the same host (single-box dev mode), Unix domain sockets shave
  ~150 µs per call vs loopback TCP. Detect same-host via
  `listen_url` and prefer UDS when available.

**Shipped (was P3):** Real HTTP/3 with per-stream independence — see
the **Latency-shipped** entry above and ADR-0019. Both prerequisites
landed in the same session: MoE expert fan-out (ADR-0018) and the h3
transport (ADR-0019, h3 0.0.8 + h3-quinn 0.0.10 + h3-axum 0.2,
`--http3-shards` / `--http3-port`). The fan-out path branches to h3
when `h3_client: Some(_)` is wired into `AppState`. No HoL benchmark
yet — needs real multi-shard MoE traffic to surface (separate P2
item under Throughput).

---

### Theme: Throughput / speed

Latency is per-request; throughput is requests/sec at p99. The
router rarely bottlenecks throughput on its own (route() is
constant-time), but rebalancer and wire decisions shape what the
fleet can sustain.

**Shipped:** Bench harness (ADR-0012 GT9), production-shape +
worst-case bench scenarios.

**Shipped:** Concurrent-route bench
(`benches/routing.rs::bench_route_concurrent`, 2026-05-16) drives
`route()` from 1 / 4 / 8 / 16 parallel tokio tasks against a single
`Arc<RwLock<GridState>>` — the lock shape `AppState::resolve_all`
actually uses. **Lock primitive swap** (2026-05-16):
`tokio::sync::RwLock<GridState>` → `parking_lot::RwLock<GridState>`
across `larql-router` and its tests. Every grid critical section is
short and sync (no `await` held across the lock), so the
synchronous primitive is correct — and the compiler will catch any
held-across-await pattern as `!Send` guards. Bench-driven
verification:

| Workers | tokio (before) | parking_lot (after) | Δ |
|---|---|---|---|
| 1 | 5.6 Melem/s | 6.4 Melem/s | +14% |
| 4 | 8.7 Melem/s | 11.1 Melem/s | +28% |
| 8 | 4.0 Melem/s | 7.2 Melem/s | **+80%** |
| 16 | 3.6 Melem/s | 6.1 Melem/s | **+70%** |

The pathological 8-worker collapse (worse than 1 worker) is fixed;
all worker counts now stay above the 1-worker baseline. Peak is at
4 workers (M3 Max has 8 performance cores; past that we hit
parking_lot's single-atomic read counter and E-core scheduling).
220 tests still pass. ArcSwap remains a P3 if write traffic ever
drops enough to amortise the copy-on-write cost; today's ~1k
heartbeats/sec on a 100-server grid makes parking_lot the sweet
spot.

**P1 — bench-driven, queued for separate work:**

- **Per-shard concurrency cap.** Hot-shard elevation reacts to
  `req_per_sec` but doesn't *cap* a shard. A misconfigured client
  flooding one shard can knock it over. Per-shard semaphore in the
  client-side dispatch path, or a server-side cap reported back.

**P2:**

- **Batched walk-ffn.** Today each layer is a separate HTTP call to
  its owning shard. If three consecutive layers all live on the
  same shard, batching them into one call halves overhead per
  three-layer run. Existing `route_all` already returns the layer-
  to-url map; the dispatch side needs to group same-URL layers
  before issuing requests.
- **Wire format options (GT8 from ADR-0012).** f16 / i8 residuals
  cut wire bytes proportionally. f16 is the obvious win (2× wire
  reduction, ~no quality loss); i8 is a step further with
  quantisation error to characterise.
- **GT10 from ADR-0012 — CI regression gate.** A shell script that
  runs a stored baseline and fails the build if throughput or tail
  latency regresses beyond thresholds. ADR-0012 sketched this;
  not implemented.

**P3:**

- **GPU-aware throughput tuning.** Once heterogeneous routing exists,
  the rebalancer can pack GPU hosts to a higher utilisation target
  than CPU hosts.
- **FP4 wire format (post-V2 generality).** Quarters wire bytes per
  Exp 26 result; needs the V2 generality work in `larql-vindex`
  before it's safe to ship as default.

---

### Theme: Operability (observability, admin, deployment)

The router currently emits logs and exposes a status RPC. Production
operations need more.

**Shipped:** Admin CLI (ADR-0004 P5) — `status` / `gaps` / `drain` /
`assign`. Hot-shard demo doc.

**Shipped:**

- **Prometheus `/metrics` endpoint** ✅ shipped 2026-05-16
  (ADR-0017). Counters for grid registers/deregisters (split by
  reason), rebalancer-tick outcomes (replicate / drop / elevate /
  demote / evict / unassign_imbalance), RTT probe outcomes
  (success / non_2xx / error), walk-ffn requests (success /
  error_4xx / error_5xx). Histogram for walk-ffn end-to-end
  duration. Gauges (refreshed at each rebalancer tick) for server
  count, distinct models, coverage gaps, elevated ranges,
  configured `--target-replicas`. Bounded cardinality — no
  `model_id` / `server_id` / `layer_id` labels. Unauth, same
  trust model as `/v1/health`.

**P2:**

- **JSON output mode for admin commands.** `larql-router status --json`
  for dashboard ingestion. `format_status` already separates rendering
  from data — slot a JSON serializer alongside the text one.
- **Multi-host deploy walkthrough doc.** Mirrors
  `docs/hot-shard-demo.md` but for a 2-box LAN topology, including
  TLS setup, firewall ports, and the `--quic-cert-fingerprint` flow.
- **`larql-router metrics` admin subcommand.** Dumps current
  Prometheus-style metrics to stdout for one-shot capture in
  scripts. Built on the same endpoint as above.

**P3:**

- **Web dashboard.** Axum-served minimal HTML, live grid state +
  rebalancer event stream. Probably worth it once metrics +
  multi-host docs are out.
- **Per-layer tau in ShardService (ADR-0015 open question).** Today
  tau is per-server; per-layer would mean tuning each layer's cache
  hit rate independently. Wait for usage data showing it matters.

---

### Theme: Cross-router federation (P2 originally)

Stays at P2 — well-defined but no implementation planned until Act 2
multi-host demo is complete. Multiple routers cover different
geographic regions; a client request is forwarded to the regional
router that owns the model shard. Requires either:

- a `RouterMsg` variant on `GridService.Join` so routers join each
  other's grids, or
- a separate `FederationService` for router-to-router routing decisions.

Probably blocks on the multi-host deploy walkthrough (Operability P2)
and on real cross-region perf data (Latency P2).

---

## Cross-references — workspace-level (other crates)

These don't live in the router crate but shape what the grid is asked
to serve. Tracked here so they're visible alongside router work.

- **Decode/prefill perf gap.** Per `crates/larql-router/README.md`
  perf snapshot: local Metal decode 86 tok/s vs ollama 98.7 tok/s
  (1.15× behind on decode). Per memory: prefill 4-14× behind ollama
  depending on prompt length. Lives in `larql-inference` /
  `larql-compute`; router only sees the result.
- **Compute crate split** (in flight, parallel session). Metal lifted
  out into `larql-compute-metal` sibling crate. Brief workspace
  resolver hiccup observed mid-session; resolved by 2026-05-16 EOD.
- **Exp 27 — hash routing across all layers (V1).** Top-2048 mask,
  100% argmax recovered at KL=0.030 at L0 on Gemma 3 4B
  (`ROADMAP_STATUS` item #2). L0 result is interp-validated; scaling
  across layers and architectures is the next step. Router's interest
  is in the resulting vindex shape — FFN rows become sparse-
  addressable, which changes shard-size economics.
- **Exp 26 — FP4 generality (V2). DONE 2026-05-31 — CONFIRMED.**
  `gemma3-4b-fresh` (the live f16 anchor; `gemma3-4b-f16` is a dangling
  symlink) is **99.83% per-feature R<16 natively, no QAT**, `down` the tail
  — and the cross-arch extension landed: Granite 4.1 3B/8B match (≥99.8%),
  and the predictive check (real E2M1 codec) is +0.116 bits/tok vs f32,
  beating Q4-int. See `docs/diagnoses/v2-fp4-generality.md`. Router impact:
  FP4 shards quarter the wire-bytes-per-tok metric tracked by the bench
  harness.
