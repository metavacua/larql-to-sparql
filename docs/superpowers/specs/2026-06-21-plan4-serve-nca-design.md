# Plan 4: larql graph-serve + NCA Integration — Design Spec

**Date:** 2026-06-21  
**Status:** Approved for implementation  
**Plan file:** `docs/superpowers/plans/2026-06-20-plan4-serve-nca.md`

## Goal

Close the codebase-to-local-agent pipeline:

```
source code
  → larql extract-codebase   (Plans 1–3, PR #229)
  → larql graph-serve        (Plan 4, Task 1)   ← this spec
  → nca run --provider larql_graph              (Plan 4, Task 4)
```

NCA queries the local graph instead of Anthropic/MiniMax. Works offline.

---

## Architecture

Three independent units, each independently testable:

### Unit 1: `larql graph-serve` CLI command (`larql-cli`)

Lightweight axum 0.8 HTTP server. Loads a `.larql.json` file into memory, then
serves two routes until killed:

| Route | Method | Purpose |
|---|---|---|
| `/v1/graph/describe` | GET `?entity=X` | Returns all edges involving entity X |
| `/v1/completions` | POST JSON | OpenAI-compat; parses `DESCRIBE X` from prompt |

**Key API corrections vs Plan 4 doc (which targeted axum 0.7):**

- Use `axum = "0.8"` (latest: 0.8.9) and `tokio = "1"` (latest: 1.52.3)
- Route path syntax unchanged (no path params in these routes — only query params)
- `into_response()` returns `Response<BoxBody>` — use `Json<T>` which handles this
- `DescribeResult` has `.outgoing: Vec<Edge>` + `.incoming: Vec<Edge>` (not iterable directly)
  → describe handler combines both into a single `Vec<EdgeView>`
- Test pattern: `.with_state(graph).oneshot(req)` via `tower::ServiceExt`
- `axum::body::to_bytes(response.into_body(), usize::MAX)` — available in axum 0.8

**Disk impact:** axum 0.8 + tokio add ~800MB to debug build artifacts.
Mitigated by Task 0 (disk governance, see below).

### Unit 2: `HttpProvider::larql_graph(port)` (`larql-core`)

One new constructor on the existing `#[cfg(feature = "http")] HttpProvider` struct:

```rust
pub fn larql_graph(port: u16) -> Self {
    Self::new(format!("http://127.0.0.1:{port}/v1"), "larql-graph")
}
```

NCA calls `/v1/completions`. The `HttpProvider::predict_next_token` already handles
the text fallback (`choices[0].text`) when logprobs are absent — which is exactly
what graph-serve returns.

### Unit 3: `VindexProvider` (`larql-core`)

Zero-HTTP in-process alternative. Wraps a `Graph` and implements the same
`DESCRIBE X` semantic as graph-serve, but in-process:

```rust
pub struct VindexProvider { graph: Graph }
impl VindexProvider {
    pub fn from_graph(graph: Graph) -> Self
    pub fn from_file(path: &Path) -> Result<Self, GraphError>  // #[cfg(not(target_arch = "wasm32"))]
    pub fn complete(&self, prompt: &str) -> String
}
```

`VindexProvider` must NOT be gated on `feature = "http"` — it is independent of reqwest.
`from_file` must be cfg-gated to preserve wasm32v1-none compatibility.

---

## Task 0 (new): Disk Governance

Added to Plan 4 to address the 50GB build artifact concern.

**Root workspace `Cargo.toml`** — add:
```toml
[profile.dev]
incremental = false
```

This single change eliminates the dominant artifact category (`target/debug/incremental`)
across all workspace crates without requiring per-worktree `.cargo/config.toml` files.

**`Makefile`** — add target:
```makefile
clean-old:
	find . -path '*/target/debug/incremental' -type d -maxdepth 6 -exec rm -rf {} + 2>/dev/null; true
	find . -path '*/.worktrees/*/target' -type d -maxdepth 4 | while read t; do \
	  age=$$(find "$$t" -name "*.d" -newer Cargo.toml 2>/dev/null | wc -l); \
	  [ "$$age" -eq 0 ] && rm -rf "$$t" && echo "pruned stale $$t"; done; true
```

---

## NCA Integration (Task 4)

`metavacua/native-cli-ai` is a separate repo. Task 4 adds:
- `ProviderConfig::LarqlGraph { port: u16 }` variant
- Parsing: `--provider larql_graph` or `--provider larql_graph:8181`
- Auto-detection: if no `--provider` given, try `TcpStream::connect` to 127.0.0.1:8181;
  use `LarqlGraph` if reachable, else existing default

This task is best-effort — if the NCA repo structure differs from the plan assumptions,
adapt to the actual code found.

---

## Constraints

- `larql graph-serve` is Tier 2 (network) — NOT added to `wasm32v1-none` CI
- `VindexProvider` is Tier 1 (filesystem) — wasm32-safe if `from_file` is cfg-gated
- `cargo test --workspace` before every commit
- No tokio in `larql-core` — only `larql-cli` gets async

---

## Testing

| Unit | Test type | Key assertion |
|---|---|---|
| graph-serve | axum `oneshot` (tokio::test) | GET `/v1/graph/describe?entity=X` → 200 + non-empty JSON array |
| graph-serve | axum `oneshot` | POST `/v1/completions` → `choices[0].text` contains relation |
| `larql_graph` | unit | `base_url` contains port number |
| `VindexProvider` | unit | `complete("DESCRIBE alpha")` returns string containing "alpha" |
| `VindexProvider` | unit | `complete("DESCRIBE unknown")` returns `""` without panic |

---

## Spec Self-Review

- No TBDs or placeholders
- `DescribeResult` API correction is explicit
- axum 0.8 path applied throughout
- wasm32 constraint on `VindexProvider::from_file` explicit
- Disk governance task added and scoped
- NCA task marked best-effort (separate repo, adapt on sight)
