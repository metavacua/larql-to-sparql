# Foundation: Resource Tiers + Core Splits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the resource-tier boundary infrastructure: add `SourceType::Ast`, split `larql-lql` into a Tier-0 `larql-lql-core` crate, extract `larql-core` file IO into `larql-core-io`, add two new workspace stub crates (`larql-codebase-core`, `larql-codebase`), and add wasm32v1-none CI enforcement.

**Architecture:** New crates are added as workspace members and verified to compile for `wasm32v1-none` (Tier 0) or standard native (Tier 1+). `larql-lql` re-exports everything from `larql-lql-core` so downstream callers have zero churn. The two new stub crates establish the dependency chain but contain only placeholder `lib.rs` files — Plan 2 fills `larql-codebase-core` and Plan 3 fills `larql-codebase`.

**Tech Stack:** Rust stable, Cargo workspace, GitHub Actions, `wasm32v1-none` target.

## Global Constraints

- All Tier 0 crates must pass `cargo build --target wasm32v1-none` with no `std::fs`, `std::net`, or `libc` imports.
- `larql-lql` public API must remain unchanged — only re-export from `larql-lql-core`.
- No new external dependencies in Tier 0 crates (only `serde`, `thiserror` with `default-features = false`).
- Follow existing workspace `version.workspace = true` pattern in every new `Cargo.toml`.
- Run `make ci` (fmt-check + clippy -D warnings + test) before each commit.

---

### Task 1: Add `SourceType::Ast`

**Files:**
- Modify: `crates/larql-core/src/core/enums.rs`

**Interfaces:**
- Produces: `SourceType::Ast` variant, `"ast"` serialisation string. All downstream tasks use this.

- [ ] **Step 1: Write the failing test**

Add to `crates/larql-core/src/core/enums.rs` at the bottom of the existing test block:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ast_source_type_round_trips() {
        let s = SourceType::Ast;
        assert_eq!(s.as_str(), "ast");
        let json = serde_json::to_string(&s).unwrap();
        let back: SourceType = serde_json::from_str(&json).unwrap();
        assert_eq!(back, SourceType::Ast);
    }
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test -p larql-core ast_source_type_round_trips
```

Expected: `error[E0599]: no variant named 'Ast'`

- [ ] **Step 3: Add the variant**

In `crates/larql-core/src/core/enums.rs`, add `Ast` before `#[default] Unknown`:

```rust
pub enum SourceType {
    Parametric,
    Document,
    Installed,
    Wikidata,
    Manual,
    Ast,        // edges derived from static AST analysis
    #[default]
    Unknown,
}
```

Add `"ast"` arm to `as_str()`:

```rust
Self::Ast => "ast",
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
cargo test -p larql-core ast_source_type_round_trips
```

Expected: `test ast_source_type_round_trips ... ok`

- [ ] **Step 5: Full crate test to check no regressions**

```bash
cargo test -p larql-core
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/larql-core/src/core/enums.rs
git commit -m "feat(core): add SourceType::Ast for AST-derived knowledge edges"
```

---

### Task 2: Create `larql-lql-core` crate (Tier 0 parser split)

**Files:**
- Create: `crates/larql-lql-core/Cargo.toml`
- Create: `crates/larql-lql-core/src/lib.rs`
- Create: `crates/larql-lql-core/src/ast.rs` (copy + trim from `larql-lql`)
- Create: `crates/larql-lql-core/src/error.rs` (copy + trim)
- Create: `crates/larql-lql-core/src/parser/mod.rs` (copy + trim)
- Create: `crates/larql-lql-core/src/lexer.rs` (copy from larql-lql if exists, else from parser internals)
- Create: `crates/larql-lql-core/src/relations.rs` (copy + trim)
- Modify: `crates/larql-lql/src/lib.rs` (re-export from larql-lql-core)
- Modify: `crates/larql-lql/Cargo.toml` (add larql-lql-core dep)
- Modify: `Cargo.toml` (add workspace member)

**Interfaces:**
- Produces: `larql_lql_core::parse(input: &str) -> Result<Statement, LqlError>`, `larql_lql_core::Statement`, `larql_lql_core::LqlError`. These are the Tier-0 parser types consumed by Plan 4's `VindexProvider`.

- [ ] **Step 1: Identify pure parser files**

```bash
grep -rn "^use larql_\|^use reqwest\|^use rustyline\|std::fs\|std::net" \
  crates/larql-lql/src/ast.rs \
  crates/larql-lql/src/parser/ \
  crates/larql-lql/src/error.rs \
  crates/larql-lql/src/relations.rs
```

Expected: zero matches (these files are pure). If any hit, note them — those files stay in `larql-lql`, not `larql-lql-core`.

- [ ] **Step 2: Create `crates/larql-lql-core/Cargo.toml`**

```toml
[package]
name = "larql-lql-core"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true
description = "LQL lexer, parser, and AST — pure Tier-0 (wasm32v1-none safe)"

[dependencies]
serde = { workspace = true, features = ["derive"] }
thiserror = { workspace = true }
```

- [ ] **Step 3: Create `crates/larql-lql-core/src/lib.rs`**

```rust
pub mod ast;
pub mod error;
pub mod parser;
pub mod relations;

pub use ast::Statement;
pub use error::LqlError;
pub use parser::parse;
```

- [ ] **Step 4: Copy pure files**

```bash
mkdir -p crates/larql-lql-core/src/parser
cp crates/larql-lql/src/ast.rs          crates/larql-lql-core/src/ast.rs
cp crates/larql-lql/src/error.rs        crates/larql-lql-core/src/error.rs
cp crates/larql-lql/src/relations.rs    crates/larql-lql-core/src/relations.rs
cp crates/larql-lql/src/parser/*.rs     crates/larql-lql-core/src/parser/
```

In each copied file, replace `crate::` references: anything from `larql-lql`'s executor (e.g., `crate::executor`) must be removed or stubbed. The parser files should only reference `crate::ast`, `crate::error`, `crate::lexer`.

- [ ] **Step 5: Add to workspace `Cargo.toml`**

In root `Cargo.toml`, add to `members` array (after `larql-lql`):

```toml
"crates/larql-lql-core",
```

- [ ] **Step 6: Update `larql-lql/Cargo.toml`**

Add dependency:

```toml
larql-lql-core = { path = "../larql-lql-core" }
```

- [ ] **Step 7: Update `larql-lql/src/lib.rs` to re-export**

Replace the existing `pub mod ast; pub mod error; pub mod parser; pub mod relations;` lines with re-exports so callers have zero churn:

```rust
// Pure parser layer — lives in larql-lql-core (Tier 0).
pub use larql_lql_core::{ast, error, parser, relations};
pub use larql_lql_core::{LqlError, Statement};
pub use larql_lql_core::parser::parse;

// IO-dependent executor layer stays here.
pub mod executor;
pub mod repl;
pub use executor::Session;
pub use repl::{run_batch, run_repl, run_statement};
```

- [ ] **Step 8: Build to confirm no regressions**

```bash
cargo build -p larql-lql-core
cargo build -p larql-lql
cargo test -p larql-lql --lib
```

Expected: all pass.

- [ ] **Step 9: Verify Tier 0 — wasm32v1-none build**

```bash
rustup target add wasm32v1-none
cargo build --target wasm32v1-none -p larql-lql-core
```

Expected: `Finished`. If it fails with a `std::fs` or `std::net` error, trace which file pulled in the dep and either remove it from `larql-lql-core` or replace it with a trait stub.

- [ ] **Step 10: Commit**

```bash
git add crates/larql-lql-core/ crates/larql-lql/Cargo.toml crates/larql-lql/src/lib.rs Cargo.toml
git commit -m "feat(lql): split larql-lql-core as Tier-0 parser crate (wasm32v1-none safe)"
```

---

### Task 3: Extract `larql-core-io` module

**Files:**
- Create: `crates/larql-core/src/io.rs` (extract file IO from wherever `.larql.json` read/write lives)
- Modify: `crates/larql-core/src/lib.rs` (gate `io` behind `#[cfg(not(target_arch = "wasm32"))]`)

**Interfaces:**
- Produces: `larql_core::io::load_graph(path: &Path) -> Result<Graph, GraphError>`, `larql_core::io::save_graph(graph: &Graph, path: &Path) -> Result<(), GraphError>`. These are used by Plan 3's `extract-graphify` and `graph-diff` commands.

- [ ] **Step 1: Find existing .larql.json file IO**

```bash
grep -rn "from_json_value\|to_json_value\|read_to_string\|write_all" \
  crates/larql-core/src/ | grep -v "test"
```

Note which files have `std::fs` calls. These are the ones to move into `io.rs`.

- [ ] **Step 2: Create `crates/larql-core/src/io.rs`**

```rust
//! File-backed Graph IO — Tier 1 (filesystem). Not available on wasm32v1-none.
#![cfg(not(target_arch = "wasm32"))]

use std::path::Path;
use crate::core::{graph::{Graph, GraphError}, schema::Schema};

pub fn load_graph(path: &Path) -> Result<Graph, GraphError> {
    let text = std::fs::read_to_string(path).map_err(GraphError::Io)?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| GraphError::Deserialize(e.to_string()))?;
    Graph::from_json_value(&v)
}

pub fn save_graph(graph: &Graph, path: &Path) -> Result<(), GraphError> {
    let v = graph.to_json_value();
    let text = serde_json::to_string_pretty(&v)
        .map_err(|e| GraphError::Deserialize(e.to_string()))?;
    std::fs::write(path, text).map_err(GraphError::Io)?;
    Ok(())
}
```

- [ ] **Step 3: Expose from `larql-core/src/lib.rs`**

```rust
#[cfg(not(target_arch = "wasm32"))]
pub mod io;
```

- [ ] **Step 4: Write test**

Add to `crates/larql-core/src/io.rs` (inside `#[cfg(test)]`):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{edge::Edge, graph::Graph};
    use tempfile::NamedTempFile;

    #[test]
    fn round_trips_graph_through_file() {
        let mut g = Graph::new();
        g.add_edge(Edge::new("A", "calls", "B"));
        let f = NamedTempFile::new().unwrap();
        save_graph(&g, f.path()).unwrap();
        let g2 = load_graph(f.path()).unwrap();
        assert_eq!(g2.edge_count(), 1);
        assert!(g2.exists("A", "calls", "B"));
    }
}
```

Add `tempfile = "3"` to `larql-core/Cargo.toml` `[dev-dependencies]`.

- [ ] **Step 5: Run test**

```bash
cargo test -p larql-core round_trips_graph_through_file
```

Expected: `test io::tests::round_trips_graph_through_file ... ok`

- [ ] **Step 6: Verify larql-core still passes wasm32v1-none**

```bash
cargo build --target wasm32v1-none -p larql-core
```

Expected: `Finished` (the `io` module is gated behind `cfg(not(target_arch = "wasm32"))`).

- [ ] **Step 7: Commit**

```bash
git add crates/larql-core/src/io.rs crates/larql-core/src/lib.rs crates/larql-core/Cargo.toml
git commit -m "feat(core): add larql_core::io — Tier-1 graph file IO, gated from wasm32"
```

---

### Task 4: Add workspace stub crates for `larql-codebase-core` and `larql-codebase`

**Files:**
- Create: `crates/larql-codebase-core/Cargo.toml`
- Create: `crates/larql-codebase-core/src/lib.rs`
- Create: `crates/larql-codebase/Cargo.toml`
- Create: `crates/larql-codebase/src/lib.rs`
- Modify: `Cargo.toml` (add both to workspace members)

**Interfaces:**
- Produces: two compilable crate stubs. Plan 2 fills `larql-codebase-core`; Plan 3 fills `larql-codebase`.

- [ ] **Step 1: Create `crates/larql-codebase-core/Cargo.toml`**

```toml
[package]
name = "larql-codebase-core"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true
description = "Codebase-to-model weight derivation — pure Tier-0 (wasm32v1-none safe)"

[dependencies]
larql-core = { path = "../larql-core" }
serde = { workspace = true, features = ["derive"] }
thiserror = { workspace = true }
```

- [ ] **Step 2: Create `crates/larql-codebase-core/src/lib.rs`**

```rust
// Tier 0: no filesystem, no network. Must compile for wasm32v1-none.
// Implementation added in Plan 2.
```

- [ ] **Step 3: Create `crates/larql-codebase/Cargo.toml`**

```toml
[package]
name = "larql-codebase"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true
description = "Codebase AST extraction and vindex construction — Tier 1 (filesystem)"

[dependencies]
larql-codebase-core = { path = "../larql-codebase-core" }
larql-core = { path = "../larql-core" }
larql-vindex = { path = "../larql-vindex" }
larql-models = { path = "../larql-models" }
serde = { workspace = true, features = ["derive"] }
serde_json = { workspace = true }
thiserror = { workspace = true }
```

- [ ] **Step 4: Create `crates/larql-codebase/src/lib.rs`**

```rust
// Tier 1: filesystem access allowed. Implementation added in Plan 3.
```

- [ ] **Step 5: Add to root `Cargo.toml` members**

```toml
"crates/larql-codebase-core",
"crates/larql-codebase",
```

(Add after `"crates/larql-lql-core"` for logical ordering.)

- [ ] **Step 6: Build stubs**

```bash
cargo build -p larql-codebase-core -p larql-codebase
cargo build --target wasm32v1-none -p larql-codebase-core
```

Expected: both `Finished`.

- [ ] **Step 7: Commit**

```bash
git add crates/larql-codebase-core/ crates/larql-codebase/ Cargo.toml
git commit -m "chore(workspace): add larql-codebase-core + larql-codebase stub crates"
```

---

### Task 5: Add wasm32v1-none CI enforcement

**Files:**
- Modify: `.github/workflows/larql-server.yml` (or create a new `tier0.yml` workflow file)

**Interfaces:**
- Produces: CI job that fails if any Tier-0 crate gains a forbidden OS/network dependency.

- [ ] **Step 1: Create `.github/workflows/tier0.yml`**

```yaml
name: Tier-0 wasm32v1-none

on:
  push:
    branches: [main, "feat/**", "fix/**"]
  pull_request:

jobs:
  tier0:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32v1-none
      - uses: Swatinem/rust-cache@v2
      - name: Build Tier-0 crates for wasm32v1-none
        run: |
          cargo build --target wasm32v1-none -p larql-codebase-core
          cargo build --target wasm32v1-none -p larql-lql-core
          cargo build --target wasm32v1-none -p larql-core
          cargo build --target wasm32v1-none -p model-compute --features wasm
```

- [ ] **Step 2: Verify the CI file parses**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/tier0.yml'))" && echo "valid"
```

Expected: `valid`

- [ ] **Step 3: Local dry-run of the build matrix**

```bash
cargo build --target wasm32v1-none -p larql-codebase-core
cargo build --target wasm32v1-none -p larql-lql-core
cargo build --target wasm32v1-none -p larql-core
cargo build --target wasm32v1-none -p model-compute --features wasm
```

Expected: all four `Finished`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/tier0.yml
git commit -m "ci: add wasm32v1-none Tier-0 enforcement job"
```

---

### Task 6: Final integration check

- [ ] **Step 1: Full workspace build**

```bash
cargo build --workspace
```

Expected: `Finished` with no errors.

- [ ] **Step 2: Full workspace test**

```bash
cargo test --workspace
```

Expected: all tests pass. Note the count — regression baseline for subsequent plans.

- [ ] **Step 3: make ci**

```bash
make ci
```

Expected: fmt-check + clippy -D warnings + test all pass.

- [ ] **Step 4: Commit (if any fmt/clippy fixes were needed)**

```bash
git add -p
git commit -m "chore(foundation): fix fmt/clippy issues from tier-0 split"
```
