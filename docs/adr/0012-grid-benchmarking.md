# ADR-0012 — Grid Benchmarking Infrastructure

**Status:** Accepted — **GT9 implemented 2026-05-07** (wire_codec.rs + routing.rs); GT8/GT10 planned  
**Depends on:** ADR-0009 (wire format), ADR-0011 (self-balancing)

---

## Context

The larql grid produces numbers that appear in `crates/larql-server/ROADMAP.md`
as hand-measured snapshots (`17.7 tok/s → 19.7 tok/s after UDS + TCP_NODELAY`).
These are not reproducible via a single command, are not tracked over time,
and have no regression gate. Three categories of measurement are missing:

1. **Grid scaling**: how does tok/s change as shard count increases? What is
   the per-shard efficiency at 1, 2, 4, 8 shards?
2. **Wire format comparison**: how does bandwidth and end-to-end latency
   compare across f32, f16, and i8 residuals?
3. **Transport comparison**: how does TCP vs QUIC latency and throughput
   compare on LAN vs WAN paths?

Without reproducible benchmarks, claims about grid performance cannot be
validated, regressions cannot be detected, and improvement work cannot be
prioritised.

---

## Decision

Three benchmark layers, each independently runnable and CI-compatible:

1. **`larql bench` extensions** — end-to-end grid benchmarks via the existing
   CLI benchmark infrastructure, with new flags for grid/wire/transport modes.
2. **Criterion micro-benchmarks** — sub-millisecond codec and routing hot-path
   benchmarks that run in CI with regression detection.
3. **CI regression gate** — a shell script that runs a stored baseline and
   fails if throughput or tail latency regresses beyond thresholds.

---

## Layer 1: `larql bench` Extensions

### New Flags

```
larql bench <model.vindex> [existing flags...]
  --bench-grid          Run shard-count scaling sweep
  --wire f32,f16,i8     Comma-separated wire formats to compare (requires --ffn or --moe-shards)
  --transport http,quic Comma-separated transports to compare (requires --ffn or --join)
  --concurrent N        Simulate N concurrent clients (default 1)
  --output json         Emit machine-readable JSON in addition to table
  --output-file PATH    Write JSON to file (default: stdout)
```

### `--bench-grid` Mode

Requires `--join grpc://router:PORT`. Runs the model through 1-shard,
2-shard, and 4-shard configurations (stopping at the number of shards
registered in the grid), measuring for each:

- `tok/s` (mean over `--steps`)
- `ms/tok` at p50, p95, p99
- `wire_bytes_per_tok` (bytes sent + received per decode step)
- `shard_efficiency` = `tok/s` / `N_shards` (ideal = constant; degradation
  indicates coordination overhead)
- `per_layer_rtt_ms` from grid status heartbeat

Output table:
```
Grid scaling — <model> (N layers, top_k=K)
  Shards  tok/s   p50ms   p99ms   wire_KB/tok  efficiency
  ──────────────────────────────────────────────────────
  1       17.8    54.3    67.2    1200         1.00×
  2       31.2    31.4    48.7     600         0.88×
  4       51.0    19.1    31.2     300         0.71×
```

The benchmark is architecture-agnostic: it reads `num_layers` and `hidden_size`
from the vindex config and computes `wire_bytes_per_tok` accordingly. No
model-family assumptions are hardcoded.

### `--wire` Mode

Requires `--ffn URL`. For each wire format:
1. Sends 10 warmup tokens (discarded).
2. Runs `--steps` decode tokens.
3. Records: tok/s, wire_bytes_per_tok, encode_ms (client-side), decode_ms
   (client-side), end-to-end latency.
4. Asserts top-1 token is identical across all formats (cross-format parity
   check).

Output table:
```
Wire format comparison — <model> (N layers, hidden=H)
  Format  tok/s   ms/tok  encode_µs  decode_µs  KB/tok  parity
  ───────────────────────────────────────────────────────────────
  f32     19.7    50.8    12.3       14.1        600     —
  f16     20.1    49.7     6.8        7.3        300     ✓ (50/50)
  i8      20.4    49.0     4.2        5.1        150     ✓ (50/50)
```

`KB/tok` is derived from `hidden_size` in the vindex config, not hard-coded.
The parity column compares top-1 tokens across formats for all decode steps.

### `--transport` Mode

Requires both `--join grpc://host:PORT` (TCP) and optionally
`--join quic://host:PORT` (QUIC, requires `--features quic`):

```
Transport comparison — gemma4-26b-a4b
  Transport  tok/s   p50ms   p99ms   reconnects  0RTT_saves
  ─────────────────────────────────────────────────────────
  tcp-grpc   19.7    50.3    67.1    3           —
  quic       21.2    46.8    51.3    3           3
```

### `--concurrent N` Mode

Spawns N tokio tasks, each running the generate loop independently and
sharing the same remote shard. Measures:
- Aggregate tok/s (sum across N clients)
- Per-client p99 ms/tok
- Shard saturation point (N at which per-client tok/s drops below 50% of N=1)

### JSON Output Schema

```json
{
  "timestamp": "2026-05-06T12:00:00Z",
  "model": "gemma4-26b-a4b",
  "grid": {
    "shards": 2,
    "topology": [{"url": "http://s1:8080", "layers": "0-14"}, ...]
  },
  "wire": "f16",
  "transport": "tcp-grpc",
  "concurrent": 1,
  "results": {
    "tok_per_s": 31.2,
    "ms_per_tok": { "mean": 32.1, "p50": 31.4, "p95": 44.1, "p99": 48.7 },
    "wire_bytes_per_tok": 614400,
    "encode_us": { "mean": 6.8, "p99": 12.1 },
    "decode_us": { "mean": 7.3, "p99": 14.2 },
    "per_layer_rtt_ms": [0.82, 0.79, 0.85, ...]
  }
}
```

---

## Layer 2: Criterion Micro-Benchmarks

### `crates/larql-inference/benches/wire_codec.rs`

Benchmarks the encode/decode hot path independently of network:

```
wire_codec/encode_f32/hidden2560_seq1   time: [4.2 µs 4.3 µs 4.4 µs]
wire_codec/encode_f16/hidden2560_seq1   time: [2.1 µs 2.2 µs 2.2 µs]
wire_codec/encode_i8/hidden2560_seq1    time: [1.8 µs 1.9 µs 1.9 µs]
wire_codec/decode_f32/hidden2560_seq1   time: [4.1 µs 4.2 µs 4.3 µs]
wire_codec/decode_f16/hidden2560_seq1   time: [2.0 µs 2.1 µs 2.1 µs]
wire_codec/decode_i8/hidden2560_seq1    time: [1.7 µs 1.8 µs 1.9 µs]
```

Parameters swept: hidden_size ∈ {2560, 4096, 5120}, seq_len ∈ {1, 32, 256}.

Also benchmarks batch codec (multi-layer request encoding).

### `crates/larql-router/benches/routing.rs`

Benchmarks the grid routing hot path:

```
routing/route_single_layer/10_servers    time: [  82 ns   83 ns   85 ns]
routing/route_single_layer/100_servers   time: [ 430 ns  435 ns  441 ns]
routing/route_all/30_layers/10_servers   time: [ 820 ns  827 ns  835 ns]
routing/heartbeat_update/100_servers     time: [  95 ns   97 ns   99 ns]
routing/rebuild_route_table/100_servers  time: [  4.2 µs  4.3 µs  4.4 µs]
```

Ensures routing stays sub-microsecond at grid sizes up to 100 servers.

---

## Layer 3: CI Regression Gate

### `scripts/bench-grid-regress.sh`

```bash
#!/usr/bin/env bash
# Usage: ./scripts/bench-grid-regress.sh [model] [baseline_file]
# Requires: LARQL_BENCH_FFN_URL env var pointing to a running shard

set -euo pipefail
MODEL=${1:-gemma3-4b-q4k}
BASELINE=${2:-bench/baselines/grid-${MODEL}.json}
CURRENT=$(mktemp)

larql bench "${LARQL_BENCH_VINDEX}" \
  --ffn "${LARQL_BENCH_FFN_URL}" \
  --wire f32,f16 \
  --steps 30 \
  --output json \
  --output-file "${CURRENT}"

python3 scripts/bench_compare.py \
  --baseline "${BASELINE}" \
  --current  "${CURRENT}" \
  --tok-per-s-threshold 0.05 \   # fail if tok/s drops >5%
  --p99-threshold       0.10     # fail if p99 rises >10%
```

### `bench/baselines/` Directory

Stores per-model JSON baselines. Updated explicitly via:

```bash
larql bench ... --output json --output-file bench/baselines/grid-gemma3-4b-q4k.json
git add bench/baselines/grid-gemma3-4b-q4k.json
git commit -m "update grid bench baseline after GT1 (f16 wire default)"
```

Baselines are committed to the repo. The CI script fails the build if the
current run regresses beyond thresholds.

### `Makefile` Targets

```makefile
bench-wire:
	cargo bench -p larql-inference --bench wire_codec

bench-routing:
	cargo bench -p larql-router --bench routing

bench-grid:
	./scripts/bench-grid-regress.sh $(MODEL)

bench-all: bench-wire bench-routing bench-grid
```

---

## Metrics Reference

| Metric | Unit | Collected by | Used for |
|---|---|---|---|
| `tok_per_s` | tokens/s | `larql bench` | throughput baseline |
| `ms_per_tok.{mean,p50,p95,p99}` | ms | `larql bench` | latency SLO |
| `wire_bytes_per_tok` | bytes | `larql bench` | bandwidth reduction proof |
| `encode_us`, `decode_us` | µs | `larql bench` | wire codec overhead |
| `per_layer_rtt_ms[]` | ms | grid heartbeat (GT3) | bottleneck layer ID |
| `shard_efficiency` | ratio | `larql bench --bench-grid` | scaling overhead |
| encode/decode throughput | MB/s | criterion `wire_codec.rs` | codec regression |
| `route()` latency | ns | criterion `routing.rs` | router regression |

---

## Implementation Files

| File | Change |
|---|---|
| `crates/larql-cli/src/commands/primary/bench_cmd.rs` | ADD `--bench-grid`, `--wire`, `--transport`, `--concurrent`, `--output json`, `--output-file` |
| `crates/larql-cli/src/commands/primary/bench/grid.rs` | NEW — grid scaling sweep logic |
| `crates/larql-cli/src/commands/primary/bench/wire.rs` | NEW — wire format comparison |
| `crates/larql-cli/src/commands/primary/bench/transport.rs` | NEW — transport comparison |
| `crates/larql-inference/benches/wire_codec.rs` | NEW — criterion codec bench |
| `crates/larql-router/benches/routing.rs` | NEW — criterion routing bench |
| `crates/larql-router/Cargo.toml` | ADD `criterion` dev-dep; `[[bench]]` entry |
| `scripts/bench-grid-regress.sh` | NEW — CI regression gate |
| `scripts/bench_compare.py` | NEW — JSON baseline comparison |
| `bench/baselines/` | NEW directory — committed baseline JSONs |
| `Makefile` | ADD `bench-wire`, `bench-routing`, `bench-grid`, `bench-all` targets |
