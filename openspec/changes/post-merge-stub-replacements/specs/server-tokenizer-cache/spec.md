## ADDED Requirements

### Requirement: LoadedModel owns a two-tier TokenizerCache

Every `larql_server::state::LoadedModel` SHALL hold an
`Arc<TokenizerCache>` field named `tokenizer_cache`. Production
construction (`bootstrap::load_model`) SHALL initialise the field via
`TokenizerCache::from_env()`, which reads the L0 and L1 sizes from
`LARQL_TOKENIZER_CACHE_L0_SIZE` (default 4096) and
`LARQL_TOKENIZER_CACHE_L1_SIZE` (default 16384). Tests MAY construct
`LoadedModel` with `TokenizerCache::new(0, 0)` to disable both
tiers when the surrounding test does not exercise tokenizer caching.

#### Scenario: Field is present on every LoadedModel
- **WHEN** a `LoadedModel` is constructed via the production
  bootstrap path or either of the two test-only constructors
- **THEN** the value SHALL hold a usable `Arc<TokenizerCache>` and
  the cache's behaviour SHALL match the sizes the constructor
  passed in.
<!-- test: larql_server::state::loaded_model_tests::encode_cached_ids_l0_hit_returns_same_ids -->

### Requirement: encode_cached_ids routes through the two-tier cache

`LoadedModel::encode_cached_ids` SHALL consult
`self.tokenizer_cache.get(text)` before invoking the underlying
tokenizer, with three branches:

- L0 hit (`covered_bytes >= text.len()`): return the cached ids
  directly without touching the tokenizer.
- L1 partial hit (`covered_bytes > 0`): cold-encode the suffix
  `text[covered_bytes..]` with `add_special_tokens = false`,
  concatenate to the cached prefix ids, then `insert(text, combined)`
  back into both tiers so the next identical request hits L0.
- Full miss: cold-encode the entire `text` with the caller's
  `add_special_tokens` flag, `insert` into both tiers, return.

Tokeniser failures (whether on the suffix or the cold-encode path)
SHALL be propagated as `Err(format!("tokenizer encode failed: {e}"))`
so the existing call sites in the OpenAI completions / insert / patches
routes keep their `match`-on-error semantics.

#### Scenario: L0 hit returns the same ids on a repeat call
- **GIVEN** a fresh `LoadedModel` whose tokenizer maps `"hello"` to
  `[1]` and `"world"` to `[2]`
- **WHEN** `encode_cached_ids("hello world", false)` is called twice
- **THEN** both calls SHALL return `Ok(vec![1, 2])` and the second
  call MUST not invoke the cold-encode path.
<!-- test: larql_server::state::loaded_model_tests::encode_cached_ids_l0_hit_returns_same_ids -->

#### Scenario: Distinct texts produce distinct ids
- **WHEN** `encode_cached_ids` is called on two unrelated prompts
- **THEN** the returned ids SHALL match what the cold tokeniser would
  have produced (no cross-prompt contamination from the cache).
<!-- test: larql_server::state::loaded_model_tests::encode_cached_ids_miss_returns_fresh_ids -->

#### Scenario: Cache eviction does not corrupt subsequent reads
- **GIVEN** a `TokenizerCache` with L0 capacity = 1
- **WHEN** three calls hit two distinct prompts then revisit the
  first
- **THEN** every call SHALL return the correct ids — the L0 evict
  of the first entry forces a cold re-encode but the result is
  identical.
<!-- test: larql_server::state::loaded_model_tests::encode_cached_ids_eviction_at_capacity -->

#### Scenario: Disabled cache is a no-op transparent passthrough
- **GIVEN** a `TokenizerCache::new(0, 0)` (both tiers disabled)
- **WHEN** `encode_cached_ids("hello world", false)` is called twice
- **THEN** both calls SHALL return the correct ids; the disabled
  tiers must not corrupt or skip the cold-encode path.
<!-- test: larql_server::state::loaded_model_tests::encode_cached_ids_disabled_cache_still_correct -->
