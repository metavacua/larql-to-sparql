# Changelog — larql-lql

All notable changes to `larql-lql` are documented here.

The format follows the conventions of [Keep a Changelog](https://keepachangelog.com/),
with dated entries (`YYYY-MM-DD`) instead of semantic versions during the
pre-1.0 phase. Forward-looking work lives in [`ROADMAP.md`](ROADMAP.md).

## [2026-05-10] — Coverage push: 38% → 87.7% lines, 96 new tests

Multi-round coverage push on the executor. 584 → 679 tests (+96), full lib
suite green, no regressions. Two real bugs surfaced and fixed along the
way (DIFF and MERGE both silently no-op'd on production-loaded vindexes
because the down-meta accessor was heap-only).

### Aggregate (larql-lql)
- Lines: **38% → 87.67%** (10625 / 12119)
- Regions: **38% → 86.49%** (18188 / 21029)
- Functions: **41% → 80.74%** (834 / 1033)

### Bug fixes
- `executor/lifecycle/diff.rs` — DIFF was using `down_meta_at` (heap
  only). On vindexes loaded via the production `load_vindex` path the
  down metadata lives in `down_meta_mmap`, so DIFF returned
  "(no differences found)" against any real on-disk vindex. Switched
  to `feature_meta` (heap + mmap) and `num_features` (both modes).
  Regression test exercises modified / removed / added statuses
  end-to-end via two on-disk fixtures.
- `executor/mutation/merge.rs` — same bug. MERGE silently merged zero
  features for the same reason. Same fix.

### Files lifted ≥ 90% line coverage
- `relations.rs`: **57% → 97.65%** (+41 pp). 17 new tests covering
  cluster lookup tiers (exact / normalised / substring), classifier
  loading from on-disk fixtures (clusters / probe-only / malformed
  JSONL / malformed keys / empty dir), `typical_layer_for_relation`
  (probe + cluster paths), `classify_direction`, `token_embedding_pub`.
- `executor/lifecycle/stats.rs`: **53% → 94.44%** (+42 pp). New test
  drops `relation_clusters.json` + `feature_clusters.jsonl` +
  `feature_labels.json` into the basic fixture and runs STATS,
  exercising the full classifier-driven coverage breakdown.
- `executor/lifecycle/diff.rs`: **48% → 94.92%** (+47 pp). 7 new tests
  via `make_modified_test_vindex_dir` fixture covering
  modified / removed / added statuses + INTO PATCH op-type
  serialisation + CURRENT / LAYER filter / LIMIT cap branches.
- `executor/remote/mutation.rs`: **46% → 98.67%** (+52 pp). 13 new
  mockito-mocked tests for `remote_insert` / `remote_delete` /
  `remote_update` (both `target=` and target-omitted variants),
  plus `remote_apply_local_patch` / `remote_show_patches` /
  `remote_remove_local_patch` (success, not-found, not-Remote).
- `executor/query/describe/moe.rs`: **8% → 91.74%** (+84 pp). New
  `make_moe_test_vindex_dir` fixture writes `router_weights.bin` +
  patches `model_config.moe` back into `index.json` after
  `write_model_weights` clobbers it via `from_arch`. Drives
  `try_moe_describe` brief + `VERBOSE` + unknown-entity variants.

### Files lifted but still under 90%
- `executor/lifecycle/compile/into_vindex.rs`: **72% → 87.68%**.
  Added MEMIT-enabled solver-path test + `ON CONFLICT
  HIGHEST_CONFIDENCE` test.
- `executor/lifecycle/compile/into_model.rs`: **52% → 77.31%**.
- `executor/lifecycle/extract.rs`: **17% → 51.20%**. Added inline
  `LqlBuildCallbacks` unit tests; the happy path still requires a
  real model fixture.
- `executor/lifecycle/use_cmd.rs`: **61% → 68.89%**. Added corrupt-
  `index.json` / `knn_store.bin` / `memit_store.json` error tests.
- `executor/query/infer.rs`: **61% → 78.81%`. Added `Backend::Weight`
  short-circuit tests + canonical-prompt KNN-override branch.
- `executor/query/infer_trace.rs`: **48% → 63.01%`. Added
  `Backend::Weight` dense-summary path + KNN-override formatter
  branch + 5 EXPLAIN INFER variants (KNOWLEDGE+ATTENTION,
  RELATIONS ONLY+ATTENTION, LAYERS range, SYNTAX, OUTPUT bands).
- `executor/mutation/merge.rs`: **60% → 83.61%`. Mostly via the
  bug fix above.
- `executor/mutation/insert/balance.rs`: **58% → 89.73%`. Added
  three-compose-INSERT test that drives the
  `cross_fact_regression_check` priors loop with multiple entries.
- `executor/remote/query.rs`: **59% → 80.14%`. Added
  `remote_explain_infer` (basic + WITH ATTENTION + KNN-override),
  `remote_infer` knn_override note, `remote_describe` Brief mode,
  `remote_walk` with LAYERS range.
- `executor/trace.rs`: **81% → 83.40%`. Added `Backend::Weight`
  decomposed-forward path + no-model-weights error.
- `repl.rs`: **41% → 62.25%`. Added comment / whitespace handling
  in `run_batch`, single-quote / mixed-quote handling in
  `split_statements`, `print_help` invocation, banner constants,
  `history_path` / `dirs_or_home` smoke. The remaining gap is in
  `run_repl` / `run_repl_basic`, both of which need stdin
  simulation to exercise.

### New test fixtures
- `make_full_test_vindex_dir(tag)` — `TinyModelArch` weights +
  WordLevel tokenizer (vocab 32 + UNK), `has_model_weights = true`,
  hidden=16. Backs INFER / TRACE / EXPLAIN INFER / COMPILE
  exercises against synthetic random weights.
- `make_large_test_vindex_dir(tag)` — hidden=1024 (just clears the
  `COMPACT MAJOR` guard) with intermediate=64 to stay under 6 MB
  on disk and sub-second per forward pass. Drives the full
  MEMIT solve path: residual capture → keys/targets matrices →
  `memit_solve` → cycle persisted to `memit_store.json`.
- `make_moe_test_vindex_dir(tag)` — `MoeConfig` with 4 experts,
  top-K = 2, plus on-disk `router_weights.bin`. Required patching
  `model_config.moe` back into `index.json` post-
  `write_model_weights`, since `VindexModelConfig::from_arch`
  unconditionally rewrites the model_config from the (dense)
  arch. Documented as a fixture-side concern.
- `make_modified_test_vindex_dir(tag)` — sibling of the basic
  fixture with three deliberate divergences (Paris→Madrid,
  French→None, None→Rome) so DIFF surfaces all status arms.
- `weight_backend_session(model_id)` — builds a `Backend::Weight`
  session inline from `larql_inference::test_utils::make_test_weights`
  + tokenizer (with the embed-by-one-row UNK extension) so the
  Weight short-circuits in INFER / TRACE / EXPLAIN INFER /
  diff-CURRENT / RESOLVE-VINDEX-REF are reachable without a real
  model on disk.

### Tooling
- All clippy warnings on larql-lql cleared (assertion-on-constants,
  manual-RangeInclusive::contains, no-effect arithmetic, hex digit
  grouping, const-is-empty).
- `cargo fmt` clean.
- Inline `#[cfg(test)]` modules added/extended in
  `executor/remote/mutation.rs` (mockito mocks for the mutation
  forwarders), `executor/lifecycle/extract.rs`
  (`LqlBuildCallbacks`), `relations.rs` (cluster / probe-label
  fixtures).
