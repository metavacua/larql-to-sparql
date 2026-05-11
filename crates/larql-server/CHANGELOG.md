# Changelog — larql-server

All notable changes to `larql-server` are documented here.

The format follows the conventions of [Keep a Changelog](https://keepachangelog.com/),
with dated entries (`YYYY-MM-DD`) instead of semantic versions during the
pre-1.0 phase. Forward-looking work lives in [`ROADMAP.md`](ROADMAP.md).

## [2026-05-10] — Code-review P0 sweep + coverage scaffolding

Five P0 fixes from the in-tree code review (REV1–REV5 in `ROADMAP.md`)
plus the missing larql-server Makefile coverage targets and a per-file
90% coverage policy.

### Fixed

- **REV1 — gRPC sort panics on NaN scores.** `grpc_describe` and
  `grpc_select` used `partial_cmp(...).unwrap()`, which panics on NaN.
  Replaced both call sites with a shared `cmp_score_desc(a, b)` helper
  that maps NaN → `Ordering::Equal`. A corrupted vindex or a future
  patched-scoring path that produces NaN no longer takes a gRPC worker
  down. Five new unit tests in `grpc.rs` lock the property.
- **REV2 — Non-constant-time API key comparison.** `auth.rs` used
  `==` on `&str`, which short-circuits and leaks bytewise progress
  through request timing. Tokens are now SHA-256-hashed and the digests
  compared via `subtle::ConstantTimeEq`. Module-level doc block names
  the threat model. `subtle` (already in the lockfile via rustls)
  added as a direct dep. Six new unit tests in `auth.rs`; six existing
  `http_auth_*` integration tests still pass with no behavioural
  change.
- **REV3 — `blocking_read` on tokio RwLock inside async path.**
  `SessionManager::apply_patch` previously called
  `model.patched.blocking_read()` while holding `sessions.write().await`
  on a worker thread, which on a multi-thread runtime stalls the
  worker (and risks deadlock against any task acquiring those locks
  in the opposite order). Restructured into fast-path / slow-path:
  the slow path drops the sessions write guard, awaits
  `model.patched.read()`, then re-acquires and uses
  `entry().or_insert_with(...)` to absorb the race where another task
  inserted the same `session_id`. No `blocking_read`/`blocking_write`
  on tokio locks is reachable from an `async fn` in `session.rs`
  anymore. Two new regression tests assert (a) forward progress when
  another task holds a `patched.read()` and (b) 16-way concurrent
  `apply_patch` on the same `session_id` finishes within a bounded
  deadline.
- **REV4 — OpenAI error envelope diverged from spec.** Non-streaming
  responses on `/v1/embeddings`, `/v1/completions`, and
  `/v1/chat/completions` returned `{"error": "msg"}` (flat); the OpenAI
  Python and JS SDKs expect
  `{"error": {"message", "type", "param", "code"}}` (nested) and broke
  on field access against the flat shape. Streaming SSE error chunks
  already used the nested form, so non-stream and stream errors were
  inconsistent. Introduced a new `OpenAIError` type with constructor
  helpers (`invalid_request`, `not_found`, `service_unavailable`,
  `server_error`) and an `IntoResponse` that renders the canonical
  nested envelope with `param`/`code` always present (possibly null).
  `From<ServerError>` lets internal helpers keep `ServerError` and
  propagate via `?`. The three OpenAI handler entry-point return
  types flipped to `Result<_, OpenAIError>` and 16 direct
  `return Err(ServerError::X(...))` sites converted to the matching
  `OpenAIError::Y(...)` constructor. LARQL paradigm endpoints keep the
  flat envelope. Six integration tests assert the nested shape on
  400/503 paths across the three handlers; seven unit tests cover the
  type itself.
- **REV5 — tool-call JSON parser surfaced 500 instead of 400 on
  malformed nested-brace output.** `build_tool_call_message` used
  `find('{')` + `rfind('}')` to extract JSON from constrained-decoder
  output, which silently picked the wrong slice on trailing junk /
  multiple objects / markdown wrappers and surfaced the parse failure
  as `ServerError::Internal` (500). Rewrote as a straight-line
  `serde_json::from_str(text.trim())` with structured diagnostics
  (`invalid JSON: …`, `tool output must be a JSON object`, missing-
  field reports), and flipped the call-site error class from
  `Internal` to `OpenAIError::invalid_request` so the client now sees
  **400 invalid_request_error** with a concrete message. Nine new
  unit tests cover happy path, surrounding whitespace, nested-brace
  arguments, trailing junk, empty/whitespace, non-object top-level,
  missing `name`/`arguments`, and invalid JSON.

### Added

- **Two-envelope error documentation.** `docs/server-spec.md §8.3.1`
  rewritten with the LARQL-flat / OpenAI-nested split and a canonical
  `type` table. README `Error Codes` section updated to match.
- **Makefile coverage targets** for larql-server, mirroring the
  larql-compute / larql-vindex pattern:
  `larql-server-test`, `larql-server-fmt-check`, `larql-server-lint`,
  `larql-server-coverage`, `larql-server-coverage-summary`,
  `larql-server-coverage-html`, `larql-server-coverage-policy`,
  `larql-server-ci`. Threshold variables: `LARQL_SERVER_COVERAGE_MIN`
  (default 65 — current baseline), `LARQL_SERVER_COVERAGE_POLICY`,
  `LARQL_SERVER_COVERAGE_REPORT`.
- **`coverage-policy.json`** with default 90% line floor, 28 per-file
  debt baselines snapshotted from the 2026-05-10 measurement, and the
  total floor at the measured 65.6% baseline. Policy semantics
  ratchet upward only — new / split files automatically inherit the
  90% default.

### Internal

- Cleared 5 pre-existing clippy errors in lib (`bootstrap.rs:230`
  boolean simplification, `metrics.rs:64` missing `Default` for
  `LayerLatencyTracker`, `walk_ffn.rs` doc indentation + needless
  lifetimes + redundant closure). `cargo clippy -p larql-server --lib
  --no-deps -- -D warnings` now clean.
- Updated `tests/test_expert_endpoint.rs` import: `cpu_moe_forward`
  and `MoeLayerWeights` moved from `larql_inference` to
  `larql_compute` in the upstream refactor; the test had a stale
  import that blocked `--tests` builds. Pure plumbing — matches the
  cargo error hint.
- Added `#[derive(Debug)]` to `ChatChoiceMessage`, `ToolCall`,
  `ToolCallFunction` to support `Result::unwrap_err()` in the new
  `build_tool_call_message` tests.

### Coverage snapshot (2026-05-10)

- **TOTAL**: 65.68% line / 72.18% function / 64.90% region.
- **At-or-above 90% default**: `routes/openai/error.rs` (100%),
  `routes/openai/util.rs` (99.6%), `routes/openai/embeddings.rs`
  (93.2%), `session.rs` (96.1%), `state.rs` (85.8% — debt baseline),
  `auth.rs` (98.0%), `wire.rs` (96.9%), `etag.rs` (100%), and 16
  others.
- **Largest debt items** (all carry baselines, must ratchet up):
  `routes/expert/{batch_legacy,multi_layer_batch,single,warmup}.rs`
  at 0% (need a live grid harness),
  `routes/openai/schema/mask.rs` at 0%, `bootstrap.rs` at 29.7%,
  `routes/openai/completions.rs` at 40.3%, `routes/walk_ffn.rs` at
  49.0%, `routes/openai/chat.rs` at 53.4%.
