# Design — dsv4-ondisk-prefix-cache

## Context

`DsV4LayerHcaCache` (`dsv4_kv_cache.rs`) composes:
- `raw: DsV4LayerKvCache` — the per-token SWA KV (`kv_a` rows, `(n_pos, head_dim)`).
- `compressed: DsV4LayerKvCache` — the per-chunk CSA/HCA compressed KV (`(n_comp, head_dim)`).
- `pending_cur: Vec<Array1<f32>>` — `< compress_ratio` tail rows not yet forming a chunk.
- `overlap_state: CompressorOverlapState` — cr=4 compressor shift state.
- indexer compressed KV cache (`Option`, on Indexer layers).

NoCompress layers (0,1) hold only a `DsV4LayerKvCache` (pure SWA). All
KV here is f32 (post-RMSNorm, post-RoPE, FP8-fake-quantized for the SWA
KV). The paper compresses by `m=4` (CSA) / `m'=128` (HCA); a chunk
"completes" every `compress_ratio` tokens.

## Goals / Non-Goals

**Goals (v1):** lossless serialize/deserialize of the compressed cache;
a prefix-hash→blob on-disk store with 128-token-block keying; Zero-SWA
prefill reuse that recomputes only the `n_win·L` tail.

**Non-goals (v1):** server / attention-session integration (DSv4 isn't
wired into the server route yet); the Periodic-checkpoint and Full-SWA
strategies; cross-process / multi-tenant sharing; eviction policy beyond
a simple size/LRU cap; compressing the on-disk blobs further (they are
already the compressed KV).

## Decisions

### D1 — Serialize only the compressed cache (Zero-SWA)

Persist `compressed` (+ indexer compressed) + `compress_ratio` +
`overlap_state`; **omit** `raw`, `pending_cur`. Rationale: the SWA `raw`
cache is ~8× the compressed size (paper §3.5.2) and is cheap to
reconstruct from the compressed entries by recomputing the last
`n_win·L` tokens. Storing it would dominate disk and give an unbalanced
write-heavy access pattern for little benefit. **Alternative considered:**
Full-SWA (store everything) — simpler reuse (no recompute) but ~8× disk
and the paper flags it as SSD-inefficient. Deferred.

### D2 — Versioned, self-describing binary wire format

A small header (`magic`, `version`, `compress_ratio`, `head_dim`,
`n_comp`, `has_indexer`, dims) followed by little-endian f32 rows. New
module `dsv4_kv_persist.rs`, mirroring the server's existing
`kv_snapshot.rs` wire-format style but DSv4-shaped. Round-trip is the
correctness anchor: `deserialize(serialize(cache)) == cache` (bit-exact
on the compressed f32). **No serde/bincode dependency** — hand-rolled
LE, matching the codebase's existing GGUF/snapshot readers.

### D3 — Prefix keying at 128-token block boundaries

The compressor only emits a compressed row at a completed
`compress_ratio`-chunk, and the paper caches at `lcm(m, m') = 128`-token
blocks (so both the `m=4` and `m'=128` streams land on a clean boundary).
Key = `hash(token_ids[0 .. k·128])` for each block boundary `k`. On a
new request, find the **longest** cached block-prefix of the incoming
token ids. Hashing is over token ids only (weights/model are fixed per
store), so the store is salted by a `model_id` (GGUF file hash) to avoid
cross-model collisions.

### D4 — Store layout: content-addressed dir tree

`<root>/<model_id>/<prefix_hash>/layer_{i}.kvz`. A small `index` (the
set of present `prefix_hash`es with token-count + mtime) is kept in
memory and rebuilt by scanning on open. Eviction: a size cap with LRU by
mtime (v1 keeps it simple; a real policy is a follow-up). Writes are
atomic (write to `.tmp` + rename).

### D5 — Zero-SWA reconstruct on hit

On a longest-prefix hit of `H` tokens (a multiple of 128):
1. Load each layer's `compressed` (+ indexer compressed) from disk.
2. Seed `DsV4LayerHcaCache` with the loaded compressed entries; leave
   `raw` empty.
3. Recompute the forward over the **last `min(H, n_win·L)` tokens** of
   the prefix to repopulate `raw`, `pending_cur`, `overlap_state` — using
   the loaded compressed entries for the attention over earlier blocks.
   For an `L`-layer model the recompute window is `n_win·L` tokens
   regardless of `H` (paper §3.5.2), so reuse cost is **O(n_win·L)**, not
   O(H).
4. Continue prefill for the un-cached suffix `token_ids[H..]` normally,
   writing new block boundaries through to the store.

The load-bearing correctness test: logits after a cache-hit prefill
equal (within the documented tolerance) logits from a cold full prefill
of the same prompt — the cache must be **transparent**.

## Risks

- **Recompute correctness** (D5) is the subtle part — the `n_win·L`
  window must exactly restore the SWA tail + compressor `pending`/overlap
  state. Mitigation: a transparency test at several prefix lengths
  (including non-block-aligned suffixes) before wiring it into any
  user-facing path; gate the whole feature behind an explicit opt-in
  (`DsV4PrefixCache::open(root)`) — default off, prefill unchanged.
- **FP8 SWA KV vs f32 compressed**: the SWA `raw` is FP8-fake-quantized;
  recompute reproduces it deterministically, so no extra error vs cold.
- **Disk growth**: bounded by D4's size cap; v1 LRU is coarse.
