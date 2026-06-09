# dsv4-ondisk-prefix-cache Specification

## Purpose
TBD - created by archiving change dsv4-ondisk-prefix-cache. Update Purpose after archive.
## Requirements
### Requirement: Layer-cache serialization

DSv4 SHALL provide a versioned binary wire format that serializes a
per-layer cache (`DsV4LayerCache`) completely — the `raw` sliding-window
KV, the `compressed` CSA/HCA KV (and, on Indexer layers, the indexer's
compressed KV), the `pending_cur` buffer, the `compress_ratio`, and both
compressor overlap states — and deserializes it back **losslessly**
(the Full-SWA strategy). Because the round-trip is exact, a deserialized
cache equals the post-prefill cache and prefill can simply continue from
it. (Dropping `raw`/`pending_cur` to recompute the SWA tail — Zero-SWA —
is a later storage optimization.)

#### Scenario: Layer cache round-trips losslessly

- **WHEN** a layer's `DsV4LayerCache` (raw + compressed + pending +
  overlap, and the indexer compressed cache on Indexer layers) is
  serialized and then deserialized
- **THEN** every field SHALL equal the original bit-exactly, with the
  same `compress_ratio`, dims, and indexer-present flag
<!-- test: larql_inference::attention::dsv4_kv_persist::hca_full_cache_round_trips_losslessly -->
<!-- test: larql_inference::attention::dsv4_kv_persist::hca_with_indexer_and_overlap_round_trips -->

#### Scenario: No-compress layer round-trips its SWA cache

- **WHEN** a NoCompress (pure-SWA) layer's cache is serialized
- **THEN** its `raw` rows SHALL round-trip in full and deserialize back
  to an equal cache, without error
<!-- test: larql_inference::attention::dsv4_kv_persist::no_compress_layer_round_trips_full -->

#### Scenario: Unknown format version is a typed error

- **WHEN** a blob with a bad magic or unknown version is deserialized
- **THEN** deserialization SHALL return a typed error, not panic
<!-- test: larql_inference::attention::dsv4_kv_persist::unsupported_version_is_typed_error -->
<!-- test: larql_inference::attention::dsv4_kv_persist::bad_magic_is_typed_error -->
<!-- test: larql_inference::attention::dsv4_kv_persist::truncated_blob_is_typed_error -->

### Requirement: Prefix-keyed on-disk store

DSv4 SHALL provide a content-addressed on-disk store mapping a hash of
the prompt token-id prefix — taken at `lcm(m, m')`-token block
boundaries and salted by a model identifier — to the per-layer
serialized compressed-KV blobs. Lookups SHALL return the longest cached
block-prefix of a given token sequence. Writes SHALL be atomic and the
store SHALL enforce a size cap.

#### Scenario: Put then longest-prefix get returns the same caches

- **WHEN** the per-layer compressed caches for a block-aligned prefix
  are written, then a superset token sequence is looked up
- **THEN** the store SHALL return that prefix's hit length and the same
  compressed caches it stored
<!-- test: larql_inference::attention::dsv4_prefix_cache::put_then_get_longest_prefix_round_trips -->
<!-- test: larql_inference::attention::dsv4_prefix_cache::longest_prefix_wins -->
<!-- test: larql_inference::attention::dsv4_prefix_cache::reopen_rebuilds_index -->
<!-- test: larql_inference::attention::dsv4_prefix_cache::model_id_isolates -->

#### Scenario: Miss returns no hit

- **WHEN** a token sequence shares no cached block-prefix
- **THEN** the lookup SHALL return no hit, and prefill SHALL proceed cold
<!-- test: larql_inference::attention::dsv4_prefix_cache::no_shared_prefix_misses -->

#### Scenario: Size cap evicts least-recently-used entries

- **WHEN** writes would exceed the configured size cap
- **THEN** the store SHALL evict least-recently-used prefixes to stay
  under the cap, and surviving entries SHALL still load correctly
<!-- test: larql_inference::attention::dsv4_prefix_cache::size_cap_evicts_lru -->

### Requirement: Full-SWA prefix reuse is transparent

On a prefix hit, DSv4 SHALL load the complete per-layer caches and
**continue prefilling** the uncached suffix at position `H` — the first
`H` tokens are not recomputed. The result SHALL match a cold full
prefill of the identical prompt at the generation level: the greedy next
token SHALL be identical at every continued position (the load-bearing
signal). Raw logits MAY differ within the documented HCA reduction-order
/ FP8 tolerance (the same sensitivity the resident-quant parity
tolerates). The feature SHALL be opt-in; with no cache supplied the cold
prefill path SHALL be unchanged.

#### Scenario: Cache hit matches cold prefill (greedy-transparent)

- **WHEN** a prompt is prefilled cold, and the same prompt is prefilled
  via a prefix-cache hit that continues from a stored prefix
- **THEN** the greedy argmax of every continued-position logit SHALL be
  identical to the cold prefill, across NoCompress, Indexer, and Compress
  layers
<!-- test: larql_inference::attention::dsv4_prefix_reuse::prefix_cache_hit_matches_cold_prefill -->

#### Scenario: Cache hit skips re-prefilling the shared prefix

- **WHEN** a prefix hit of length `H` (a block multiple, `H < N`) is
  reused for an `N`-token prompt
- **THEN** only the `N - H` suffix tokens SHALL be forwarded; the first
  `H` tokens SHALL NOT be re-run, yielding a wall-time speedup that grows
  with the shared-prefix length
<!-- test: larql_inference::attention::dsv4_prefix_reuse::prefix_cache_hit_matches_cold_prefill -->
<!-- test: larql_inference::attention::dsv4_prefix_reuse::bench_cold_vs_warm_prefill -->

#### Scenario: Disabled cache leaves cold prefill unchanged

- **WHEN** no prefix cache is supplied
- **THEN** the prefill path SHALL be byte-for-byte identical to the
  pre-feature cold prefill

