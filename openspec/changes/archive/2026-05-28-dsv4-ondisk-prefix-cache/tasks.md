## 1. P0 — Groundwork

- [ ] 1.1 Confirm the exact reconstructable state of `DsV4LayerHcaCache` post-prefill (what `raw`/`pending_cur`/`overlap_state` hold at a 128-block boundary) by instrumenting the existing cached prefill on a real-GGUF prompt — establishes the recompute target for D5.
- [ ] 1.2 Confirm the `lcm(m, m') = 128` block boundary lands both the cr=4 and cr=128 compressed streams on a clean chunk (no `pending_cur` carry) — so a block-aligned prefix is self-contained.
- [ ] 1.3 Decide the `model_id` salt (GGUF file content hash vs path+mtime) — record in design D3.

## 2. P1 — Serialization wire format (no store, no reuse)

- [x] 2.1 New `dsv4_kv_persist.rs`: `serialize_layer_cache(&DsV4LayerCache) -> Vec<u8>` + `deserialize_layer_cache(&[u8]) -> Result<DsV4LayerCache, KvPersistError>`. Operates at the per-layer `DsV4LayerCache` enum (NoCompress | Hca) so it covers both variants uniformly. Serializes the **compressed** cache (+ indexer compressed) + `compress_ratio` + both overlap states; `raw`/`pending_cur` omitted (Zero-SWA). Versioned LE header (magic `D4KV`, version, tag); hand-rolled, no serde.
- [x] 2.2 Round-trip tests (synthetic, no GGUF): `hca_compressed_round_trips_losslessly` + `hca_with_indexer_and_overlap_round_trips` — compressed/indexer-compressed rows bit-exact, compress_ratio/dims/overlap preserved, `raw`+`pending_cur` come back empty.
- [x] 2.3 NoCompress-layer (pure SWA) handling: `no_compress_layer_round_trips_as_empty` (shape-only shell → empty cache) + `empty_compressed_round_trips` (HCA layer with no chunk yet).
- [x] 2.4 Version/format-mismatch handling: typed `KvPersistError` (BadMagic / UnsupportedVersion / UnknownTag / Truncated), never a panic — `bad_magic_is_typed_error`, `unsupported_version_is_typed_error`, `unknown_tag_is_typed_error`, `truncated_blob_is_typed_error` (every truncation length).

## 3. P2 — Prefix-keyed on-disk store

- [x] 3.1 `DsV4PrefixCache::open(root, model_id, max_bytes)` — content-addressed dir tree `<root>/<model_id>/<prefix_hash>/{tokens.bin, layer_{i}.kvz}`; in-memory index rebuilt by scanning on open (`reopen_rebuilds_index`). Sweeps leftover `.tmp.*` dirs.
- [x] 3.2 `put(token_ids, &[DsV4LayerCache])` (the per-layer enum, matching P1's serializer) at a 128-block boundary — atomic write (populate `.tmp.<key>.<nonce>` dir, then `rename`), one `layer_{i}.kvz` per layer + `tokens.bin`. `get_longest_prefix(token_ids) -> Option<(hit_len, Vec<DsV4LayerCache>)>`. Non-aligned `put` → `NotBlockAligned` (`put_rejects_unaligned`).
- [x] 3.3 Prefix hashing at 128-token block boundaries via stable FNV-1a salted by `model_id` (`block_prefix_hashes`, one O(n) pass); longest-prefix match, **verified against `tokens.bin`** so a hash collision can never return the wrong cache (`longest_prefix_wins`, `model_id_isolates`).
- [x] 3.4 Size-capped LRU eviction by `last_used` (`size_cap_evicts_lru`: A touched → B evicted, C survives, survivors still load). Atomicity via tmp-dir rename.
- [x] 3.5 Store round-trip (`put_then_get_longest_prefix_round_trips`) + miss/short-prefix (`no_shared_prefix_misses`). All 7 tests on tempdirs.

## 4. P3 — Full-SWA prefill reuse (the payoff)

**Pivot (2026-05-28):** investigating the cached forward showed Zero-SWA
reuse needs a dedicated "recompute-mode" attention (decouple raw/compressed
position + custom causal mask + suppress re-compress) — large and
correctness-critical. Switched to **Full-SWA first**: serialize the
complete cache (incl. `raw`/`pending`) so a hit just *continues prefill*
at `H` via the existing cached forward — transparent by construction
(lossless serialize + the proven split-prefill==one-shot). Zero-SWA stays
a future storage optimization.

- [x] 4.1 `dsv4_resident_prefill_with_prefix_cache(layers, hp, head, token_ids, &mut DsV4PrefixCache, max_seq_len, backend)` in `dsv4_prefix_reuse.rs`: longest-prefix `get`; on a hit load the complete caches and continue prefill of the suffix at `H` (no recompute); on a miss prefill cold. Write-through `put` of the block-aligned prompt. Opt-in (`PrefixPrefillResult{start_pos, logits, cache_hit}`).
- [x] 4.2 **Transparency test (load-bearing, real-GGUF, ignored):** `prefix_cache_hit_matches_cold_prefill` — cache-hit suffix vs cold one-shot over 4 layers (NoCompress 0,1 · Indexer 2 · Compress 3). **Greedy argmax 16/16** at every continued position (the gate). SWA-only max rel diff ~5e-3; HCA layers add split-batch reduction-order/FP8 drift (max_rel 0.47 < 1.5 documented bound), greedy-transparent throughout — same convention as the resident-quant parity.
- [x] 4.3 **Fixed a latent P8 bug** surfaced by 4.2: `dsv4_compressor_step_coff2` (cr=4 cached step, Indexer layers) was never wired through `proj_wkv`/`proj_wgate` — it would panic on any resident cached decode through an Indexer layer once a chunk completes. P8's parity test ran `layer_caches=None` (prefill path) so missed it. Now guarded + dispatched like `coff1`.
- [x] 4.4 Opt-in gate: the helper is a separate entry point; callers not passing a cache use the unchanged cold `dsv4_resident_model_forward_cached`. Write-through only on a block-aligned prompt.

## 5. P4 — Wire-up & docs

- [x] 5.1 Resident generate entry: `dsv4_resident_generate_with_prefix_cache` (prefill via the reuse helper → decode loop over the returned caches). The reuse helper's `PrefixPrefillResult` now returns the populated `caches` so a decode loop can continue. Opt-in: callers without a cache use the unchanged cold path. (The streaming `dsv4_generate` is left as-is — the prefix cache is a resident-path feature.)
- [x] 5.2 Bench `bench_cold_vs_warm_prefill` (real-GGUF, ignored): **512-token shared prefix, 4 layers → cold 12.2s (528 tok) vs warm 1.67s (16-tok suffix) = 7.3× faster prefill.** Speedup grows with prefix length / layer count (cold scales with prefix; warm is ~constant in the suffix).
- [x] 5.3 `make traceability` regenerated; openspec validate passes; clippy clean; 248 lib tests pass.
