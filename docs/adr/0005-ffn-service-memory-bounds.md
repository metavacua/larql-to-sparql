# ADR-0005 — Memory Bounds for the FFN-Service Server

**Status:** Implemented
**Depends on:** ADR-0002 (FFN Activation Cache), ADR-0003 (FFN Router), ADR-0004 (FFN Grid)

---

## Context

A `larql serve --ffn-only` server is the worker tier of the FFN grid: it holds
FFN weights for a slice of a model and responds to `/v1/walk-ffn` requests
from clients running attention locally. For this to be operationally useful on
commodity hardware, a server handling a slice of Gemma 4 31B can't run at
55 GB RSS — the originally observed footprint.

Three layers of growth were diagnosed on 31B Q4_K:

1. **Eager warmup** — `VectorIndex::warmup()` decodes every f16 gate layer into
   f32 at startup. Decoded f32 gates are ~2× the on-disk f16 size. For 31B
   that's `13 GB × 2 ≈ 26 GB` resident *before the first request*. Warmup
   amortises cost for throughput-sensitive deployments but penalises
   cold-start and co-hosted setups.

2. **Unbounded lazy decode** — without warmup, `gate_knn` on f16 mmap data
   populates `f16_decode_cache` on first touch per layer. The cache grew
   monotonically. A full forward pass decoded all 60 layers → same ~26 GB
   heap, just amortised across requests instead of paid at startup.

3. **mmap-resident working set** — `gate_vectors.bin` (~13 GB) and
   `interleaved_q4k.bin` (~11 GB) are demand-paged. Pages that get touched
   during walk + FFN dequant become resident and stay resident until the
   kernel reclaims under pressure. On macOS the kernel rarely reclaims
   file-backed shared mappings; on Linux reclamation is more aggressive but
   still opportunistic.

The aggregate ceiling without any intervention was the sum of all three —
well over system memory on a laptop-class server.

---

## Decision

Three orthogonal, opt-in bounds, each targeting one growth mode. Layer
sharding (ADR-0004) remains the **preferred hard bound** for real
deployments because it prevents out-of-shard pages from ever being touched.
The bounds below are for single-shard or experimental topologies where
sharding isn't practical.

### 1. `--ffn-only` skips eager warmup

The FFN-service mode is declared at startup. When `--ffn-only` is set,
`VectorIndex::warmup()` is not called. Per-layer decode happens lazily on
first `gate_knn` call for that layer. Correctness is unchanged — the f16
path has always had a fallback decode.

Trade-off: a one-request cold cost per layer (~40 ms decode for a
21504 × 5376 × f16 gate matrix on CPU). For an interactive demo this is
invisible behind the existing FFN forward latency.

### 2. `--max-gate-cache-layers N` — LRU on the decode cache

`VectorIndex` gains two fields:

```rust
pub(crate) gate_cache_lru: Mutex<VecDeque<usize>>,
pub(crate) gate_cache_max_layers: AtomicUsize,
```

`set_gate_cache_max_layers(N)` installs the cap. On each cache access
(`resolve_gate` and `gate_knn_mmap_fast` f16 paths), `touch_gate_cache_lru`
moves the accessed layer to the front of the queue. On insert, if the queue
length exceeds the cap, the back (least-recently-used) layer is evicted
from `f16_decode_cache` by setting that slot to `None`.

`N = 0` is unlimited (historical behaviour, max speed). `N = 4` on 31B caps
the decode cache at `4 × 433 MB ≈ 1.7 GB`. Cost: re-decode on LRU miss.

### 3. `--release-mmap-after-request` — madvise(DONTNEED) post-request

Adds `release_mmap_pages()` to `VectorIndex` which iterates all owned mmaps
and calls `Mmap::unchecked_advise(UncheckedAdvice::DontNeed)` on each. The
walk-ffn handler invokes this at the end of each request when the flag is
set.

The call uses `unchecked_advise` (unsafe): safety is preserved by invoking
it only after `run_walk_ffn` has returned, so no slices into the mmap are
live in the current closure. The read lock on `PatchedVindex` is held for
the madvise call but that's just preventing concurrent reshard, not
protecting any derived references.

**Platform behaviour:**

| OS | `MADV_DONTNEED` on shared file-backed mmap | Observed after one request |
|---|---|---|
| Linux | Immediately drops clean pages from RSS | ~23 GB → ~6 GB |
| Darwin | Advisory; kernel may defer until memory pressure | 23 GB → 23 GB (stable) |

macOS's weakness is by design. Darwin reserves `MADV_FREE` for
private-anon mappings; shared mappings have no equivalent release
directive. The flag still prevents unbounded growth across many requests
(page working set stops growing once the forward pass's touched-set is
established); it just doesn't shrink the existing resident set.

---

## Measured Ceilings (Gemma 4 31B Q4_K, macOS, CPU)

| Configuration | Startup RSS | After 3 requests |
|---|---|---|
| Default (no `--ffn-only`) | 55 GB | 55 GB |
| `--ffn-only` | 5.6 GB | 23 GB |
| `--ffn-only --max-gate-cache-layers 4` | 5.6 GB | 23 GB |
| `... --release-mmap-after-request` | 5.6 GB | 23 GB (stable) |
| `... --layers 0-19` (sharding) | 5.6 GB | ~8 GB (shard-proportional) |

Startup RSS improvement: **10×**. The 23 GB floor is the mmap working set
of the whole-model Q4_K forward pass on macOS; it does not grow across
requests with the bounds in place. Layer sharding is the only route below
that floor on macOS; on Linux, `--release-mmap-after-request` would
approximate sharding's RSS profile without the topology.

### Reproducing the table

```bash
# Terminal A — start the server under the scenario being measured.
larql serve gemma4-31b-q4k --port 8088 --ffn-only \
  --max-gate-cache-layers 4 \
  --release-mmap-after-request

# Terminal B — parity driver (also useful as a correctness gate).
cargo run --release -p larql-inference --example q4k_remote_parity -- \
  --vindex /path/to/gemma4-31b-q4k.vindex \
  --server http://127.0.0.1:8088

# Terminal C — sample server RSS. Repeat before/after requests.
ps -o pid,rss,command -p $(pgrep larql-server)
```

The example asserts bit-identical top-5 between local and remote paths;
parity is the correctness half of the story, the RSS measurement is the
bound half. Swap the flag set in Terminal A to fill in other rows.

---

## Implementation Files

| File | Role |
|---|---|
| `crates/larql-vindex/src/index/core.rs` | New fields: `gate_cache_lru`, `gate_cache_max_layers` |
| `crates/larql-vindex/src/index/gate.rs` | `set_gate_cache_max_layers`, `touch_gate_cache_lru`, wired into `resolve_gate` + `gate_knn_mmap_fast` |
| `crates/larql-vindex/src/index/accessors.rs` | `release_mmap_pages` (calls `unchecked_advise(DontNeed)` on every owned mmap) |
| `crates/larql-server/src/main.rs` | CLI flags, skips `warmup()` under `--ffn-only`, wires `set_gate_cache_max_layers` on load |
| `crates/larql-server/src/state.rs` | `LoadedModel.release_mmap_after_request` field |
| `crates/larql-server/src/routes/walk_ffn.rs` | Calls `release_mmap_pages()` inside `spawn_blocking` after `run_walk_ffn` returns |
| `crates/larql-cli/src/main.rs` | Passthrough of `--max-gate-cache-layers` / `--release-mmap-after-request` to `larql-server` |

---

## Trade-offs

- **Startup speed vs sustained RSS.** `--ffn-only` defers decode cost to
  first request. Throughput-first deployments that warm up before serving
  should leave warmup on.
- **Cache hit rate vs heap.** `--max-gate-cache-layers N` evicts layers.
  A 60-layer forward with `N=4` re-decodes 56 layers per pass (vs 0 at
  `N=0`). For single-shot queries on a cold server the overhead is
  invisible; for steady-state throughput, prefer higher `N`.
- **Platform parity.** `--release-mmap-after-request` is a hard bound on
  Linux, a soft hint on Darwin. The primary way to hit a hard RSS target
  on macOS is `--layers`.

---

## Open Questions

1. **f16 gemv without decode.** The root cause of gate-cache growth is that
   the CPU gemv kernel operates on f32. An f16 gemv (Accelerate / NEON or
   Metal) would make the decode cache unnecessary. Metal `f16_gemv` has
   shipped for `lm_head` (see `project_f16_gemv_wiring_todo`); the same
   lift could cover gate KNN.

2. **Darwin `MADV_FREE_REUSABLE`.** Darwin-specific and for private-anon
   mappings only, but worth re-checking whether an anonymous copy of the
   working set could be backed by it. Probably not worth the indirection.

3. **Per-range madvise.** `release_mmap_pages` currently advises the
   whole file. Per-layer `advise_range` would let us keep hot layers
   resident across requests. Complexity is in tracking which ranges were
   last touched; defer until `--release-mmap-after-request` is shown to
   be too aggressive in practice.

4. **Stats endpoint.** `/v1/cache-stats` could expose cache sizes, eviction
   counts, and current RSS. Useful for demo day; not required for
   correctness.
