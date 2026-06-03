# larql-core

Knowledge graph engine for LARQL. Triple-based edges, graph algorithms, LLM extraction, multi-format serialization.

## What it does

Provides the core `Graph` data structure that the rest of LARQL builds on. A graph is a set of `(subject, relation, object)` edges with confidence scores, source attribution, and metadata. The engine supports querying, algorithms, LLM-based extraction, and efficient serialization.

```rust
use larql_core::*;

let mut graph = Graph::new();
graph.add_edge(
    Edge::new("France", "capital", "Paris")
        .with_confidence(0.95)
        .with_source(SourceType::Parametric)
);
graph.add_edge(
    Edge::new("Paris", "river", "Seine")
        .with_confidence(0.88)
);
assert_eq!(
    graph.try_add_edge(Edge::new("France", "capital", "Paris")),
    EdgeInsertResult::Duplicate
);
assert_eq!(
    graph.insert_edge(
        Edge::new("France", "capital", "Paris")
            .with_confidence(0.97)
            .with_source(SourceType::Parametric)
    ),
    EdgeInsertResult::Replaced
);

// Query
let capitals = graph.select("France", Some("capital"));
let capital = graph.get_edge("France", "capital", "Paris").unwrap();
let edges_to_paris = graph.edges_between("France", "Paris");
let outgoing_relations = graph.outgoing_relations("France");
let (dest, path) = graph.walk("France", &["capital", "river"]).unwrap();
assert_eq!(dest, "Seine");

// Algorithms
let result = shortest_path(&graph, "France", "Seine");
let ranks = pagerank(&graph, 0.85, 100, 1e-6);
let comps = connected_components(&graph);

// Serialize
save_json(&graph, "knowledge.larql.json").unwrap();
```

## Core Types

| Type | Purpose |
|------|---------|
| `Graph` | Indexed edge collection with adjacency, reverse, keyword indexes |
| `Edge` | Directed fact: subject --relation--> object, with confidence and metadata |
| `EdgeInsertResult` | Explicit mutation result: Inserted, Duplicate, or Replaced |
| `Schema` | Optional relation type registry and node type inference rules |
| `Node` | Computed entity with degree info and inferred type |
| `SourceType` | Edge origin: Parametric, Document, Installed, Wikidata, Manual, Unknown |

`list_entities()`, `list_relations()`, `nodes()`, search tie-breaks, and
connected components are deterministic. Exact triple lookup is available via
`get_edge(subject, relation, object)`, and multiedge pair lookup is available
via `edges_between(subject, object)`.

`add_edge()` preserves the legacy behavior of silently skipping duplicate
triples. `try_add_edge()` reports `Inserted` or `Duplicate` without replacing,
while `insert_edge()` upserts by exact triple and can return `Replaced` when
confidence, source, metadata, or injection changes.

## Algorithms

| Algorithm | Function | Complexity |
|-----------|----------|------------|
| Shortest path | `shortest_path()`, `astar()` | O((E+V) log V) |
| PageRank | `pagerank()` | O(iterations x (E+V)) |
| BFS/DFS | `bfs_traversal()`, `dfs()` | O(E+V) |
| Components | `connected_components()`, `are_connected()` | O(E+V) |
| Walk | `graph.walk()` (highest confidence), `walk_all_paths()` | O(branching^depth) |
| Filter | `filter_graph()` | O(E) |
| Merge | `merge_graphs()` with Union/MaxConfidence/SourcePriority | O(E) |
| Diff | `diff()` | O(E) |
| Subgraph | `graph.subgraph()` | O(E within depth) |

Shortest path stores the exact edge chosen during Dijkstra/A*, so returned paths
and costs stay consistent for multiedges with different relations or weights.
`TraversalResult.edges` contains edges actually traversed to newly discovered
nodes. `diff()` reports same-triple changes to confidence, source, metadata,
and injection.

`filter_graph()` is metadata-key agnostic. Use `MetadataPredicate` for any
domain-specific edge metadata instead of relying on model-specific field names:

```rust
let filtered = filter_graph(
    &graph,
    &FilterConfig {
        metadata: vec![
            MetadataPredicate::u64_min("layer", 20),
            MetadataPredicate::f64_min("selectivity", 0.7),
        ],
        ..Default::default()
    },
);
```

`MergeStrategy::SourcePriority` uses `default_source_priority()` for backward
compatibility. Call `merge_graphs_with_source_priority()` when a product,
dataset, or import pipeline needs a different source ordering.

## LLM Integration

| Component | Purpose |
|-----------|---------|
| `ModelProvider` trait | Single interface to any LLM backend |
| `HttpProvider` | OpenAI-compatible API (ollama, vLLM, llama.cpp) |
| `MockProvider` | Testing with fixture-based knowledge injection |
| `chain_tokens()` | Multi-token answer extraction |
| `extract_bfs()` | BFS knowledge graph extraction from LLM |
| `TemplateRegistry` | Relation-specific prompt templates |

The engine layer is model-architecture agnostic. `ModelProvider` exposes token
predictions only; it has no dependency on model families, tensor shapes,
vindex internals, or GPU backends. Extraction policy that is domain-specific,
such as edge provenance and whether a generated entity should be followed,
lives in `BfsConfig` and can be replaced by callers.

## Serialization

| Format | Extension | Size (100K edges) | Serialize | Deserialize |
|--------|-----------|-------------------|-----------|-------------|
| JSON | `.larql.json` | 9.9 MB | 135ms | 325ms |
| MessagePack | `.larql.bin` | 4.7 MB (53% smaller) | 126ms | 303ms |
| Packed binary | `.larql.pak` | 4.3 MB (56% smaller) | 22ms | 244ms |
| CSV | `.csv` | (simple interchange) | - | - |
| Checkpoint | (append-only) | (crash-safe log) | - | - |

Packed binary uses string interning — repeated relation names stored once.
Packed decoding validates header offsets, record bounds, string indexes, and
metadata ranges before reading. It also rejects invalid source tags, malformed
metadata JSON, and inconsistent metadata/injection flags instead of silently
dropping flagged data. Packed encoding rejects values that cannot fit in the
on-disk u32 fields instead of truncating them. CSV import/export supports
quoted commas, quotes, CRLF/LF newlines, and multiline fields for the five graph
columns: `subject,relation,object,confidence,source`.

## Crate Structure

```
larql-core/src/
├── lib.rs                  Re-exports
├── core/
│   ├── graph.rs            Graph struct, indexes, queries, walk, search, subgraph, stats
│   ├── edge.rs             Edge struct, builder pattern, compact serialization
│   ├── schema.rs           Schema, RelationMeta, TypeRule, type inference
│   ├── node.rs             Node (computed entity with degree)
│   └── enums.rs            SourceType, MergeStrategy
├── algo/
│   ├── shortest_path.rs    Dijkstra and A* with custom weight functions
│   ├── pagerank.rs         Iterative PageRank
│   ├── traversal.rs        BFS and DFS with depth tracking
│   ├── components.rs       Connected components enumeration, connectivity check
│   ├── walk.rs             Multi-path walk with confidence ranking
│   ├── filter.rs           Predicate-based edge filtering
│   ├── merge.rs            Graph merging with conflict resolution strategies
│   └── diff.rs             Graph diffing (added, removed, changed)
├── engine/
│   ├── provider.rs         ModelProvider trait, PredictionResult
│   ├── http_provider.rs    OpenAI-compatible HTTP provider (feature-gated)
│   ├── mock_provider.rs    Mock provider for testing
│   ├── bfs.rs              BFS knowledge extraction from LLM
│   ├── chain.rs            Multi-token chaining
│   └── templates.rs        Prompt template registry
└── io/
    ├── format.rs           Format enum, auto-detection from extension
    ├── json.rs             JSON serialization (Python-compatible)
    ├── msgpack.rs          MessagePack (feature-gated)
    ├── packed.rs           String-interned binary format with corrupt-input checks
    ├── csv.rs              CSV import/export with quoted-field support
    └── checkpoint.rs       Append-only crash-safe log
```

## Testing

```bash
make larql-core-ci             # fmt + clippy + tests + feature matrix + benches + examples
make larql-core-test           # default feature tests
make larql-core-feature-test   # no-default and msgpack-only tests
make larql-core-lint           # clippy --all-targets -D warnings
make larql-core-coverage       # cargo-llvm-cov summary
make larql-core-bench-test     # compile/smoke benchmark harness
make larql-core-bench          # Criterion benchmark
make larql-core-examples       # run callable examples
```

Equivalent cargo commands:

```bash
cargo test -p larql-core
cargo test -p larql-core --no-default-features
cargo test -p larql-core --no-default-features --features msgpack
cargo clippy -p larql-core --all-targets -- -D warnings
cargo llvm-cov --package larql-core --summary-only
cargo test -p larql-core --benches
cargo bench -p larql-core --bench graph
```

GitHub Actions runs the same core checks on Ubuntu, Windows, and macOS via
`.github/workflows/larql-core.yml`. The workflow is cargo-native rather than
Makefile-driven so it works with the default shell on every runner.

### Benchmarks (release build)

| Operation | Latency |
|-----------|---------|
| select subject + relation (25K edges) | 55.85 ns |
| exists exact triple (25K edges) | 113.23 ns |
| keyword search (25K edges) | 584.95 ns |
| shortest path ring (1K nodes) | 282.92 us |
| connected components ring (1K nodes) | 410.96 us |
| BFS traversal depth 5 (1K nodes) | 3.07 us |
| JSON encode / decode (1K edges) | 635.08 us / 2.49 ms |
| Packed encode / decode (1K edges) | 209.41 us / 1.68 ms |
| JSON encode / decode (25K edges) | 20.94 ms / 93.62 ms |
| Packed encode / decode (25K edges) | 6.55 ms / 50.58 ms |

Criterion reports changes relative to the runner's local baseline under
`target/criterion`; those relative “improved/regressed” labels are not a CI
failure unless a separate regression gate is added. The benchmark harness uses
a longer measurement window for serialization so larger decode cases complete
without noisy sample warnings.

### Test Coverage

- Graph: construction, queries, walk, search, subgraph, stats, dedupe
- Accessors: deterministic entities, relations, nodes, search tie-breaks, exact edge and multiedge lookup
- Mutation: legacy duplicate skipping, explicit duplicate reporting, upsert replacement
- Edge: builder pattern, equality, hashing, compact serialization
- Schema: type rules, inference, JSON roundtrip
- Algorithms: shortest path, multiedge reconstruction, PageRank, BFS/DFS, merge, diff, filter
- Components: enumeration, connectivity, disconnected graphs, edge cases
- Walk: highest-confidence selection, multi-path, all-paths, limits
- Remove edge: index rebuild correctness
- Search: empty query, no match, case insensitive
- Serialization: JSON/MsgPack/Packed roundtrips, metadata preservation, corrupt packed input
- CSV: quoted commas, escaped quotes, multiline fields, confidence/source roundtrips
- Diff: confidence, source, metadata, and injection changes
- BFS extraction: mock provider, depth, multi-seed, max_entities, template stop tokens
- Token chaining: multi-token, stop tokens, probability threshold, model-call accounting
- Templates: registry, JSON load/save
- Checkpoint: append, replay, persistence
- Python compatibility: format interop
- Feature matrix: default features, no default features, msgpack-only
- Examples: edge, graph, algorithm, filter, serialization
- Benches: Criterion graph/query/algorithm/serialization harness compile-smoke

Current default coverage:

| Command | Line coverage | Region coverage |
|---------|---------------|-----------------|
| `cargo llvm-cov --package larql-core --summary-only` | 89.33% | 90.41% |

Coverage should stay high for `larql-core` because the crate is pure Rust and
does not require model weights. Use `make larql-core-coverage` for the current
summary and `make larql-core-coverage-html` for a browsable report.

The default coverage profile includes the optional HTTP provider. The
no-default/msgpack profile is a useful second signal for the graph and
serialization surface when HTTP is intentionally excluded.

## Design Principles

1. **Triple-based** — every fact is (subject, relation, object) with confidence
2. **Indexed** — adjacency, reverse, keyword indexes kept in sync
3. **Schema-optional** — type inference rules loaded externally, not hardcoded
4. **Provider-agnostic** — any LLM backend via ModelProvider trait
5. **Multi-format** — JSON (human), MsgPack (compact), Packed (fast), CSV (interchange)
6. **Crash-safe** — CheckpointLog for long-running extractions
7. **Zero unsafe** — all safe Rust, minimal dependencies
8. **Platform-neutral** — little-endian on-disk formats, standard filesystem APIs, CI on Linux/Windows/macOS
9. **No hidden model assumptions** — graph algorithms and serialization never depend on model architecture, tensor layout, or fixed metadata keys

## Features

- `http` (default) — HTTP model provider for ollama/vLLM/llama.cpp
- `msgpack` (default) — MessagePack serialization

## License

Apache-2.0
