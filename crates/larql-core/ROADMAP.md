# larql-core Roadmap

`larql-core` owns the in-memory graph model, graph algorithms, lightweight
model-provider extraction helpers, and portable graph serialization formats.
It should stay independent of vindex storage and inference internals: higher
crates can depend on it, but this crate should remain a small, reusable graph
engine.

For shipped work, see [CHANGELOG.md](CHANGELOG.md).

---

## Current state (verified 2026-08-04)

**Shape.** Four modules, no internal dependency on any other `larql-*` crate:

| Module | Contents |
|---|---|
| `core` | `edge`, `enums`, `graph`, `node`, `schema` |
| `algo` | `components`, `diff`, `filter`, `merge`, `pagerank`, `shortest_path`, `traversal`, `walk` |
| `engine` | `bfs`, `chain`, `http_provider`, `mock_provider`, `provider`, `templates` |
| `io` | `checkpoint`, `csv`, `format`, `json`, `msgpack`, `packed` |

**What it does.** `Graph` is an indexed directed multigraph over
`(subject, relation, object)` facts with confidence, source, metadata, and
optional injection hints. Query indexes cover outgoing edges, incoming edges,
exact triples, and keyword search. Algorithms include shortest path/A*,
PageRank, BFS/DFS, components, walks, filtering, merging, and diffing.
Serialization supports JSON, MessagePack, packed binary, CSV, and append-only
checkpoint logs. LLM extraction is provider-agnostic through `ModelProvider`,
`TemplateRegistry`, `chain_tokens`, and BFS extraction.

**Features.** `default = ["http", "msgpack"]`; both are optional, so pure graph
usage carries zero network and zero MessagePack dependencies. The
`--no-default-features` build is a supported configuration and is covered
separately.

**Tests.** 191 tests across 14 binaries (12 integration files in `tests/` plus
unit tests); `cargo test -p larql-core` passes.

**Coverage.** This crate has **no `coverage-policy.json`** — unlike the twelve
crates that do, the per-file 90% floor is not enforced here. The last recorded
measurement was 77.92% lines with default features and 79.84% with
`--no-default-features --features msgpack`; that figure has not been re-taken
and should be treated as a stale reference, not a current number.

**Benchmarks.** A criterion harness exists at `benches/graph.rs` with three
groups (queries, algorithms, serialization) plus a
`examples/bench_graph.rs` snapshot whose release output backs the numbers in
`README.md`. What is still missing is a *regression gate* over the criterion
output, not the harness itself.

**Known open defect.** Confidence is not sanitized at deserialize boundaries,
and five `partial_cmp().unwrap()` sites will panic on a non-finite value —
`core/graph.rs:278`, `algo/pagerank.rs:19`, `algo/walk.rs:35`,
`engine/http_provider.rs:110`, `engine/chain.rs:24`. The 2026-05-28 review
named the first three; the two `engine` sites are the same defect and are
listed here so a fix covers all of them. Detail below under P0.

---

## P0 — correctness and robustness

The original hardening pass shipped; one item is still open. The table is kept
as the record of which regressions are covered.

| Item | Area | Status |
|---|---|---|
| Store exact path edges in shortest path | `algo::shortest_path` | Done. Dijkstra/A* predecessor state now stores the selected edge, so multiedge paths and costs agree. |
| Harden packed binary decoding | `io::packed` | Done. Decoder validates flags, offsets, record bounds, string indexes, checked arithmetic, and metadata ranges. |
| Replace ad hoc CSV parsing/writing | `io::csv` | Done. CSV supports quoted commas, escaped quotes, CRLF/LF records, and multiline quoted fields. |
| Diff all edge attributes | `algo::diff` | Done. Same-triple changes now include confidence, source, metadata, and injection. |
| Clarify traversal edge semantics | `algo::traversal` | Done. `TraversalResult.edges` means edges actually traversed to newly discovered nodes. |
| Sanitize confidence on deserialize | `core::edge`, `core::graph`, `algo`, `engine` | **Open** (raised 2026-05-28, still present 2026-08-04). `CompactEdge -> Edge` stores confidence directly, bypassing `with_confidence`; NaN or out-of-range values can later panic unwrap-based `partial_cmp` sorts. Clamp or reject non-finite confidence at graph format boundaries, and route the five sort sites through the shared NaN-safe helper: `core/graph.rs:278`, `algo/pagerank.rs:19`, `algo/walk.rs:35`, `engine/http_provider.rs:110`, `engine/chain.rs:24`. |

---

## P1 - API polish

| Item | Area | Detail |
|---|---|---|
| Deterministic ordered accessors | `core::graph`, `algo::components` | Done. `list_entities`, `list_relations`, `nodes`, search tie-breaks, and connected component ordering are deterministic. |
| Fallible graph mutation API | `core::graph` | Done. `try_add_edge` reports `Inserted`/`Duplicate` without replacement, `insert_edge` upserts by exact triple and can return `Replaced`, and `add_edge` remains the legacy duplicate-skipping path. |
| Explicit multiedge lookup | `core::graph` | Done. Exact triple lookup is available through `get_edge(subject, relation, object) -> Option<&Edge>`, pair lookup through `edges_between(subject, object)`, and relation discovery through `outgoing_relations`/`incoming_relations`. |
| Configurable keyword tokenizer | `core::graph` | Search lowercases and splits on whitespace/hyphen only. Add a small tokenizer abstraction or normalization options for punctuation, relation aliases, and case/diacritic handling. |
| Error types per subsystem | `core::graph`, `io`, `engine` | `GraphError::Deserialize(String)` is too broad. Split parse, format, unsupported-version, corrupt-offset, and IO context enough for CLI/server diagnostics. |

---

## P2 - Graph features

| Item | Area | Detail |
|---|---|---|
| Relation-aware subgraph extraction | `core::graph`, `algo` | Extend `subgraph` and traversal APIs with relation allow/deny lists, direction modes (`out`, `in`, `both`), confidence thresholds, and source filters. |
| Weighted traversal and path queries | `algo` | Add path APIs for `k_shortest_paths`, all simple paths with bounded depth, and relation-constrained shortest path. These map well to LQL path queries. |
| Stronger graph diff/patch model | `algo::diff` | Provide a stable diff format that can be applied to a graph, serialized, and surfaced as added/removed/updated triples with attribute-level changes. |
| Graph validation | `core::schema` | Validate edges against schema relation metadata: allowed subject/object types, reversible relation declarations, confidence ranges, required metadata keys, and unknown relation warnings. |
| Provenance utilities | `core::edge`, `algo` | Add merge and filter helpers that preserve source precedence, collect source counts per relation, and expose provenance summaries for DESCRIBE/SELECT callers. |
| Graph sampling | `algo` | Add deterministic sampling utilities for large graphs: top confidence per relation, stratified source sampling, random walk sampling with seed control. |

---

## P3 - Performance and scale

| Item | Area | Detail |
|---|---|---|
| Incremental index updates | `core::graph` | `remove_edge` and replacement flows rebuild all indexes. Add index-slot invalidation or swap-remove bookkeeping before large mutation workloads rely on this crate. |
| Memory-efficient string storage | `core::graph` | Edges and indexes clone strings heavily. Consider optional string interning for large graphs while preserving ergonomic `String` APIs. |
| Streaming readers/writers | `io` | JSON and packed paths operate on whole buffers. Add streaming load/save where format allows, especially for checkpoint compaction and large interchange files. |
| Packed format versioning plan | `io::packed` | Add explicit flags handling, forward-compatible unknown flag rejection, metadata/injection section lengths, and upgrade tests before `.larql.pak` becomes a durable format. |
| Bench regression harness | `examples`, benches | Partially done: `benches/graph.rs` is a criterion harness with query/algorithm/serialization groups, and README claims are backed by `bench_graph` release output with fixed generators. Still open: a **regression gate** — record a baseline and fail CI on a threshold breach. The harness exists; nothing checks its output. |

---

## P4 - LLM extraction extensions

| Item | Area | Detail |
|---|---|---|
| Stop-token support in BFS extraction | `engine::bfs` | `PromptTemplate.stop_tokens` exists but `extract_bfs` currently passes `None` to `chain_tokens`. Use template-specific stop tokens. |
| Better multi-token mock provider | `engine::mock_provider` | The mock currently returns only the first token, which under-tests chaining behavior. Add scripted token sequences for realistic multi-pass extraction tests. |
| Provider capability metadata | `engine::provider` | Add optional capability reporting for logprobs, token IDs, timeout behavior, and max top-k so extraction code can fail clearly when a backend cannot supply confidence. |
| Extraction normalization hooks | `engine::bfs` | Add answer cleanup hooks for trimming articles, punctuation, casing, aliases, and entity rejection rules without hardcoding domain policy in BFS. |
| Async provider option | `engine` | Keep blocking APIs for simple callers, but consider an async provider trait behind a feature for server-side extraction and concurrent probing. |

---

## P0 regression coverage

- Shortest path with two `A -> B` edges where the cheaper edge is not the first
  inserted edge; returned path edge and cost must agree.
- Packed files with invalid `string_table_offset`, truncated edge records,
  out-of-range string indexes, unsupported flags, and invalid metadata ranges.
- CSV roundtrip with commas, quotes, and newlines in subject/object fields.
- Diff where confidence is unchanged but `source`, `metadata`, or `injection`
  changes.
- BFS/DFS with `max_depth = 0`, confirming no traversed edges are returned.

---

## Non-goals

- Do not add dependencies on `larql-vindex`, `larql-inference`, or CLI/server
  crates.
- Do not make this crate responsible for mmap vindex storage or tensor patching.
- Do not introduce model-family-specific extraction rules here; keep those in
  higher-level crates or external configuration.
