# Codebase Extraction + CLI Commands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `larql-codebase` (Tier 1 AST extraction crate) and four new CLI commands: `larql extract-codebase` (codebase → vindex), `larql extract-graphify` (NetworkX JSON → .larql.json), `larql graph-diff` (consensus merge), and `larql export --format gguf` (vindex → GGUF file).

**Architecture:** `larql-codebase` wraps tree-sitter to produce a `larql_core::Graph` with `SourceType::Ast` edges, then calls `larql_codebase_core::graph_to_weight_repr` and `larql_models::loading::gguf::writer::GgufWriter` to write a GGUF file plus a `vindex.json` manifest (forming a minimal vindex directory). `write_bitnet_artifacts` is NOT used here — it requires a fully constructed `ModelWeights` from the GGUF loader. Each new CLI command follows the existing pattern: `#[derive(Args)] pub struct XxxArgs`, `pub fn run(args: XxxArgs) -> Result<(), Box<dyn std::error::Error>>`, added to `Commands` enum in `main.rs`. The `extract-graphify` command reads NetworkX node-link JSON and performs the φ transform into `larql_core::Graph`. The `export --format gguf` command uses the existing `GgufWriter` from `larql-models`.

**Tech Stack:** Rust stable, `larql-codebase-core` (Plan 2), `larql-core`, `larql-vindex`, `larql-models`, `tree-sitter 0.23`, `tree-sitter-rust 0.23`, `tree-sitter-python 0.23`, `tree-sitter-typescript 0.23`, `clap`, `serde_json`, `walkdir 2`.

## Global Constraints

- `larql-codebase` is Tier 1 (filesystem), not Tier 0 — do not add it to the wasm32v1-none CI job.
- All new CLI commands return `Result<(), Box<dyn std::error::Error>>` and print errors with `eprintln!` before returning `Err`.
- φ transform rules: graphify `node.label` → edge subject/object, `edge.relation` preserved verbatim, `node.kind` → INSERT `("node", "has_kind", kind)`, `node.source_file` (if present) → INSERT `("node", "defined_in", source_file)`.
- Run `cargo test --workspace` before every commit.
- Confidence for all `SourceType::Ast` edges = 1.0 (unit-asserted).

---

### Task 1: tree-sitter Rust language support in `larql-codebase`

**Files:**
- Create: `crates/larql-codebase/src/languages/mod.rs`
- Create: `crates/larql-codebase/src/languages/rust_lang.rs`
- Modify: `crates/larql-codebase/Cargo.toml`
- Modify: `crates/larql-codebase/src/lib.rs`

**Interfaces:**
- Produces:
  - `pub trait LanguageExtractor: Send + Sync { fn extensions(&self) -> &[&'static str]; fn extract(&self, source: &str, path: &str, graph: &mut Graph); }`
  - `pub struct RustExtractor;` implementing `LanguageExtractor`

`RustExtractor::extract` parses the source with tree-sitter-rust and adds edges for:
- `fn foo` in `mod bar` → `("bar::foo", "defined_in", path)`
- `use crate::X` → `("file", "imports", "crate::X")`
- `fn foo()` calls `bar()` → `("foo", "calls", "bar")`

All edges use `SourceType::Ast`, `confidence = 1.0`.

- [ ] **Step 1: Add tree-sitter dependencies to `larql-codebase/Cargo.toml`**

```toml
[dependencies]
larql-codebase-core = { path = "../larql-codebase-core" }
larql-core = { path = "../larql-core" }
larql-vindex = { path = "../larql-vindex" }
larql-models = { path = "../larql-models" }
serde = { workspace = true, features = ["derive"] }
serde_json = { workspace = true }
thiserror = { workspace = true }
walkdir = "2"
tree-sitter = "0.23"
tree-sitter-rust = "0.23"
tree-sitter-python = "0.23"
tree-sitter-typescript = "0.23"

[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 2: Write the failing test**

Create `crates/larql-codebase/src/languages/rust_lang.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use larql_core::core::graph::Graph;

    #[test]
    fn extracts_function_def_edge() {
        let source = r#"
            fn hello_world() {
                println!("hi");
            }
        "#;
        let mut g = Graph::new();
        let extractor = RustExtractor;
        extractor.extract(source, "src/lib.rs", &mut g);
        // Should produce at least one edge involving "hello_world"
        let nodes = g.node_names();
        assert!(
            nodes.iter().any(|n| n.contains("hello_world")),
            "expected hello_world to appear as a node, got: {:?}", nodes
        );
    }

    #[test]
    fn use_statement_produces_imports_edge() {
        let source = "use std::collections::HashMap;";
        let mut g = Graph::new();
        RustExtractor.extract(source, "src/main.rs", &mut g);
        let edges: Vec<_> = g.edges().collect();
        assert!(
            edges.iter().any(|e| e.relation == "imports"),
            "expected an 'imports' edge for use statement"
        );
    }
}
```

- [ ] **Step 3: Run to confirm failure**

```bash
cargo test -p larql-codebase extracts_function_def_edge
```

Expected: `error[E0412]: cannot find type 'RustExtractor'`

- [ ] **Step 4: Implement `LanguageExtractor` trait and `RustExtractor`**

Create `crates/larql-codebase/src/languages/mod.rs`:

```rust
pub mod rust_lang;
pub use rust_lang::RustExtractor;

use larql_core::core::graph::Graph;

pub trait LanguageExtractor: Send + Sync {
    fn extensions(&self) -> &[&'static str];
    fn extract(&self, source: &str, path: &str, graph: &mut Graph);
}
```

Implement `crates/larql-codebase/src/languages/rust_lang.rs`:

```rust
use larql_core::core::{
    edge::Edge,
    enums::SourceType,
    graph::Graph,
};
use tree_sitter::{Node, Parser};
use super::LanguageExtractor;

pub struct RustExtractor;

impl LanguageExtractor for RustExtractor {
    fn extensions(&self) -> &[&'static str] { &["rs"] }

    fn extract(&self, source: &str, path: &str, graph: &mut Graph) {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .expect("tree-sitter-rust load");
        let tree = match parser.parse(source, None) {
            Some(t) => t,
            None => return,
        };
        let bytes = source.as_bytes();
        extract_node(tree.root_node(), bytes, path, graph, None);
    }
}

fn ast_edge(s: &str, r: &str, o: &str) -> Edge {
    let mut e = Edge::new(s, r, o);
    e.source = SourceType::Ast;
    e.confidence = 1.0;
    e
}

fn extract_node(node: Node, src: &[u8], path: &str, graph: &mut Graph, scope: Option<&str>) {
    match node.kind() {
        "function_item" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let fn_name = name_node.utf8_text(src).unwrap_or("?");
                let qualified = match scope {
                    Some(s) => format!("{s}::{fn_name}"),
                    None => fn_name.to_string(),
                };
                graph.add_edge(ast_edge(&qualified, "defined_in", path));
                // Walk body for call_expression children
                for i in 0..node.child_count() {
                    extract_node(node.child(i).unwrap(), src, path, graph, Some(&qualified));
                }
                return;
            }
        }
        "call_expression" => {
            if let Some(func) = node.child_by_field_name("function") {
                let callee = func.utf8_text(src).unwrap_or("?");
                if let Some(s) = scope {
                    graph.add_edge(ast_edge(s, "calls", callee));
                }
            }
        }
        "use_declaration" => {
            let text = node.utf8_text(src).unwrap_or("");
            let path_str = text.trim_start_matches("use ").trim_end_matches(';');
            graph.add_edge(ast_edge(path, "imports", path_str));
        }
        _ => {}
    }
    for i in 0..node.child_count() {
        extract_node(node.child(i).unwrap(), src, path, graph, scope);
    }
}
```

- [ ] **Step 5: Update `larql-codebase/src/lib.rs`**

```rust
pub mod languages;
pub use languages::{LanguageExtractor, RustExtractor};
```

- [ ] **Step 6: Run tests**

```bash
cargo test -p larql-codebase
```

Expected: both extractor tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/larql-codebase/src/languages/ crates/larql-codebase/src/lib.rs crates/larql-codebase/Cargo.toml
git commit -m "feat(codebase): LanguageExtractor trait + RustExtractor (tree-sitter)"
```

---

### Task 2: Python and TypeScript language extractors

**Files:**
- Create: `crates/larql-codebase/src/languages/python_lang.rs`
- Create: `crates/larql-codebase/src/languages/ts_lang.rs`
- Modify: `crates/larql-codebase/src/languages/mod.rs`

**Interfaces:**
- Consumes: `LanguageExtractor` trait (Task 1)
- Produces:
  - `pub struct PythonExtractor;` — handles `.py` files, extracts `def`, `import`, `from X import Y`
  - `pub struct TsExtractor;` — handles `.ts`, `.tsx`, extracts `function`, `import from`

- [ ] **Step 1: Write failing tests**

In `crates/larql-codebase/src/languages/python_lang.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use larql_core::core::graph::Graph;

    #[test]
    fn python_def_produces_defined_in() {
        let src = "def compute(x):\n    return x * 2\n";
        let mut g = Graph::new();
        PythonExtractor.extract(src, "utils.py", &mut g);
        let nodes = g.node_names();
        assert!(nodes.iter().any(|n| n.contains("compute")));
    }

    #[test]
    fn python_import_produces_imports_edge() {
        let src = "import os\nfrom pathlib import Path\n";
        let mut g = Graph::new();
        PythonExtractor.extract(src, "main.py", &mut g);
        let edges: Vec<_> = g.edges().collect();
        assert!(edges.iter().any(|e| e.relation == "imports"));
    }
}
```

In `crates/larql-codebase/src/languages/ts_lang.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use larql_core::core::graph::Graph;

    #[test]
    fn ts_function_produces_defined_in() {
        let src = "function greet(name: string): string { return `Hello ${name}`; }";
        let mut g = Graph::new();
        TsExtractor.extract(src, "hello.ts", &mut g);
        let nodes = g.node_names();
        assert!(nodes.iter().any(|n| n.contains("greet")));
    }
}
```

- [ ] **Step 2: Run to confirm failures**

```bash
cargo test -p larql-codebase python_def_produces_defined_in ts_function_produces_defined_in
```

Expected: both fail with `cannot find type`.

- [ ] **Step 3: Implement `PythonExtractor`**

```rust
use larql_core::core::{edge::Edge, enums::SourceType, graph::Graph};
use tree_sitter::{Node, Parser};
use super::LanguageExtractor;

pub struct PythonExtractor;

fn ast_edge(s: &str, r: &str, o: &str) -> Edge {
    let mut e = Edge::new(s, r, o);
    e.source = SourceType::Ast;
    e.confidence = 1.0;
    e
}

impl LanguageExtractor for PythonExtractor {
    fn extensions(&self) -> &[&'static str] { &["py"] }

    fn extract(&self, source: &str, path: &str, graph: &mut Graph) {
        let mut parser = Parser::new();
        parser.set_language(&tree_sitter_python::LANGUAGE.into()).expect("tree-sitter-python");
        let tree = match parser.parse(source, None) { Some(t) => t, None => return };
        extract_py(tree.root_node(), source.as_bytes(), path, graph);
    }
}

fn extract_py(node: Node, src: &[u8], path: &str, graph: &mut Graph) {
    match node.kind() {
        "function_definition" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = name_node.utf8_text(src).unwrap_or("?");
                graph.add_edge(ast_edge(name, "defined_in", path));
            }
        }
        "import_statement" => {
            let text = node.utf8_text(src).unwrap_or("");
            graph.add_edge(ast_edge(path, "imports", text.trim()));
        }
        "import_from_statement" => {
            let text = node.utf8_text(src).unwrap_or("");
            graph.add_edge(ast_edge(path, "imports", text.trim()));
        }
        _ => {}
    }
    for i in 0..node.child_count() {
        extract_py(node.child(i).unwrap(), src, path, graph);
    }
}
```

- [ ] **Step 4: Implement `TsExtractor`**

```rust
use larql_core::core::{edge::Edge, enums::SourceType, graph::Graph};
use tree_sitter::{Node, Parser};
use super::LanguageExtractor;

pub struct TsExtractor;

fn ast_edge(s: &str, r: &str, o: &str) -> Edge {
    let mut e = Edge::new(s, r, o);
    e.source = SourceType::Ast;
    e.confidence = 1.0;
    e
}

impl LanguageExtractor for TsExtractor {
    fn extensions(&self) -> &[&'static str] { &["ts", "tsx"] }

    fn extract(&self, source: &str, path: &str, graph: &mut Graph) {
        let mut parser = Parser::new();
        parser.set_language(&tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
            .expect("tree-sitter-typescript");
        let tree = match parser.parse(source, None) { Some(t) => t, None => return };
        extract_ts(tree.root_node(), source.as_bytes(), path, graph);
    }
}

fn extract_ts(node: Node, src: &[u8], path: &str, graph: &mut Graph) {
    match node.kind() {
        "function_declaration" | "arrow_function" => {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = name_node.utf8_text(src).unwrap_or("?");
                graph.add_edge(ast_edge(name, "defined_in", path));
            }
        }
        "import_statement" => {
            if let Some(src_node) = node.child_by_field_name("source") {
                let module = src_node.utf8_text(src).unwrap_or("?").trim_matches('"').trim_matches('\'');
                graph.add_edge(ast_edge(path, "imports", module));
            }
        }
        _ => {}
    }
    for i in 0..node.child_count() {
        extract_ts(node.child(i).unwrap(), src, path, graph);
    }
}
```

- [ ] **Step 5: Add to `languages/mod.rs`**

```rust
pub mod python_lang;
pub mod rust_lang;
pub mod ts_lang;
pub use python_lang::PythonExtractor;
pub use rust_lang::RustExtractor;
pub use ts_lang::TsExtractor;
```

- [ ] **Step 6: Run all tests**

```bash
cargo test -p larql-codebase
```

Expected: all 5 language extractor tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/larql-codebase/src/languages/
git commit -m "feat(codebase): PythonExtractor + TsExtractor via tree-sitter"
```

---

### Task 3: File discovery + parallel extraction → `Graph`

**Files:**
- Create: `crates/larql-codebase/src/extractor.rs`
- Modify: `crates/larql-codebase/src/lib.rs`

**Interfaces:**
- Consumes: `LanguageExtractor` implementations (Tasks 1–2)
- Produces:
  - `pub fn extract_codebase(root: &Path) -> Result<Graph, CodebaseError>`

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;

    #[test]
    fn extract_rust_files_from_fixture() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("main.rs"), "fn main() { println!(\"hi\"); }").unwrap();
        fs::write(src.join("lib.rs"), "use std::fmt; fn helper() {}").unwrap();

        let graph = extract_codebase(dir.path()).unwrap();
        assert!(graph.node_count() > 0, "should have nodes from Rust files");
        let edges: Vec<_> = graph.edges().collect();
        assert!(edges.iter().any(|e| e.relation == "defined_in"));
    }
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
cargo test -p larql-codebase extract_rust_files_from_fixture
```

Expected: `error[E0425]: cannot find function 'extract_codebase'`

- [ ] **Step 3: Add `CodebaseError` and implement `extract_codebase`**

Create `crates/larql-codebase/src/extractor.rs`:

```rust
use std::path::Path;
use thiserror::Error;
use walkdir::WalkDir;
use larql_core::core::graph::Graph;
use crate::languages::{LanguageExtractor, RustExtractor, PythonExtractor, TsExtractor};

#[derive(Error, Debug)]
pub enum CodebaseError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

fn extractors() -> Vec<Box<dyn LanguageExtractor>> {
    vec![
        Box::new(RustExtractor),
        Box::new(PythonExtractor),
        Box::new(TsExtractor),
    ]
}

pub fn extract_codebase(root: &Path) -> Result<Graph, CodebaseError> {
    let exts: Vec<Box<dyn LanguageExtractor>> = extractors();
    let mut graph = Graph::new();

    for entry in WalkDir::new(root).follow_links(false).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() { continue; }
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let rel_path = path.strip_prefix(root).unwrap_or(path).to_string_lossy().into_owned();

        // Skip build artifacts
        if rel_path.starts_with("target/") || rel_path.starts_with(".git/") { continue; }

        for extractor in &exts {
            if extractor.extensions().contains(&ext) {
                match std::fs::read_to_string(path) {
                    Ok(source) => extractor.extract(&source, &rel_path, &mut graph),
                    Err(_) => continue, // skip unreadable files (binary, etc.)
                }
                break;
            }
        }
    }
    Ok(graph)
}
```

- [ ] **Step 4: Export from `lib.rs`**

```rust
pub mod extractor;
pub use extractor::{extract_codebase, CodebaseError};
```

- [ ] **Step 5: Run test**

```bash
cargo test -p larql-codebase extract_rust_files_from_fixture
```

Expected: `test extractor::tests::extract_rust_files_from_fixture ... ok`

- [ ] **Step 6: Commit**

```bash
git add crates/larql-codebase/src/extractor.rs crates/larql-codebase/src/lib.rs
git commit -m "feat(codebase): extract_codebase — walkdir file discovery + multi-language extraction"
```

---

### Task 4: `larql extract-codebase` CLI command

**Files:**
- Create: `crates/larql-cli/src/commands/extraction/extract_codebase_cmd.rs`
- Modify: `crates/larql-cli/src/commands/extraction/mod.rs`
- Modify: `crates/larql-cli/src/main.rs`
- Modify: `crates/larql-cli/Cargo.toml`

**Interfaces:**
- Consumes: `larql_codebase::extract_codebase`, `larql_codebase_core::{graph_to_weight_repr, BitNetBasis}`, `larql_vindex::extract::bitnet_writer::{write_bitnet_artifacts, BitnetArchMeta}`
- Produces: `larql extract-codebase <PATH> --output <OUT_DIR>` CLI command that writes a BitNet vindex to `<OUT_DIR>`.

- [ ] **Step 1: Add `larql-codebase` dep to `larql-cli/Cargo.toml`**

```toml
larql-codebase = { path = "../larql-codebase" }
larql-codebase-core = { path = "../larql-codebase-core" }
# larql-models already in larql-cli deps — GgufWriter lives there
```

- [ ] **Step 2: Create the command file**

```rust
use std::path::PathBuf;
use clap::Args;
use larql_codebase::extract_codebase;
use larql_codebase_core::{basis::BitNetBasis, graph_to_weight_repr};
use larql_models::loading::gguf::writer::{GgufTensor, GgufValue, GgufWriter};
use serde_json::json;

#[derive(Args)]
pub struct ExtractCodebaseArgs {
    /// Root directory of the codebase to extract.
    root: PathBuf,

    /// Output vindex directory (created if absent).
    #[arg(short, long, default_value = "codebase.vindex")]
    output: PathBuf,
}

pub fn run(args: ExtractCodebaseArgs) -> Result<(), Box<dyn std::error::Error>> {
    let root = args.root.canonicalize()?;
    eprintln!("Extracting codebase from: {}", root.display());

    let graph = extract_codebase(&root)?;
    eprintln!(
        "  {} nodes, {} edges extracted",
        graph.node_count(),
        graph.edge_count()
    );

    let repr = graph_to_weight_repr(&graph, &BitNetBasis);
    eprintln!("  {} weight tensors synthesised", repr.tensors.len());

    std::fs::create_dir_all(&args.output)?;

    // Write weights as a GGUF file inside the vindex directory.
    let gguf_path = args.output.join("weights.gguf");
    let mut writer = GgufWriter::new();
    writer.meta("general.architecture", GgufValue::String("bitnet".into()));
    writer.meta("general.name", GgufValue::String("larql-codebase".into()));
    writer.meta("larql.hidden_size", GgufValue::U32(repr.arch.hidden_size as u32));
    writer.meta("larql.n_layers", GgufValue::U32(repr.arch.n_layers as u32));
    writer.meta("larql.n_heads", GgufValue::U32(repr.arch.n_heads as u32));
    for t in &repr.tensors {
        writer.tensor(GgufTensor {
            name: t.name.clone(),
            dims: t.dims.clone(),
            ggml_type: t.ggml_type,
            data: t.data.clone(),
        });
    }
    writer.write_to_file(&gguf_path)?;

    // Write a minimal vindex manifest so `larql show` can recognise the dir.
    let manifest = json!({
        "version": 1,
        "kind": "codebase",
        "source": root.to_string_lossy(),
        "weights": "weights.gguf",
        "arch": {
            "hidden_size": repr.arch.hidden_size,
            "n_layers": repr.arch.n_layers,
            "n_heads": repr.arch.n_heads,
            "head_dim": repr.arch.head_dim,
        }
    });
    std::fs::write(
        args.output.join("vindex.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;

    eprintln!("  vindex written to: {}", args.output.display());
    Ok(())
}
```

- [ ] **Step 3: Register in `commands/extraction/mod.rs`**

Add to the existing `mod.rs`:

```rust
pub mod extract_codebase_cmd;
pub use extract_codebase_cmd::ExtractCodebaseArgs;
```

- [ ] **Step 4: Add to `Commands` enum in `main.rs`**

Under `#[command(next_help_heading = "Build")]` section:

```rust
#[command(next_help_heading = "Build")]
/// Build a vindex by extracting AST knowledge from a codebase.
ExtractCodebase(extract_codebase_cmd::ExtractCodebaseArgs),
```

Add to the `match` arm in `main()`:

```rust
Commands::ExtractCodebase(args) => extract_codebase_cmd::run(args),
```

- [ ] **Step 5: Build to check compilation**

```bash
cargo build -p larql-cli
```

Expected: `Finished`. If `ModelWeights`/`TensorData` fields differ, read `crates/larql-models/src/loading/` and adjust.

- [ ] **Step 6: Manual smoke test on the larql-main repo itself**

```bash
cargo run -p larql-cli -- extract-codebase . --output /tmp/larql-main-test.vindex
```

Expected: prints node/edge count and `vindex written to: /tmp/larql-main-test.vindex`. Directory exists with `bitnet/` subdirectory.

- [ ] **Step 7: Commit**

```bash
git add crates/larql-cli/src/commands/extraction/extract_codebase_cmd.rs \
        crates/larql-cli/src/commands/extraction/mod.rs \
        crates/larql-cli/src/main.rs \
        crates/larql-cli/Cargo.toml
git commit -m "feat(cli): larql extract-codebase — AST graph → BitNet vindex"
```

---

### Task 5: `larql extract-graphify` — NetworkX JSON → .larql.json

**Files:**
- Create: `crates/larql-cli/src/commands/extraction/extract_graphify_cmd.rs`
- Modify: `crates/larql-cli/src/commands/extraction/mod.rs`
- Modify: `crates/larql-cli/src/main.rs`

**Interfaces:**
- Produces: `larql extract-graphify <GRAPHIFY_JSON> --output <LARQL_JSON>` command.

The φ transform:
- For each node: if `source_file` is present → add `(label, "defined_in", source_file)` edge. Always add `(label, "has_kind", kind)` if `kind` is set.
- For each edge (link): add `(source_label, relation, target_label)` where relation comes from edge data's `type` or `relation` attribute.
- All edges: `confidence = 1.0`, `source = SourceType::Ast`.

Input JSON format (NetworkX node-link): `{ "nodes": [{"id": ..., "label": ..., "kind": ..., "source_file": ...}], "links": [{"source": ..., "target": ..., "type": ...}] }`.

- [ ] **Step 1: Write the failing test**

Create a test fixture file for the test to use. The test creates an in-memory `serde_json::Value` and calls the transform function directly:

In `crates/larql-cli/src/commands/extraction/extract_graphify_cmd.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_json() -> serde_json::Value {
        serde_json::json!({
            "nodes": [
                {"id": 0, "label": "mymod", "kind": "module", "source_file": "src/lib.rs"},
                {"id": 1, "label": "helper_fn", "kind": "function", "source_file": "src/lib.rs"},
                {"id": 2, "label": "std::fmt", "kind": "external"}
            ],
            "links": [
                {"source": 0, "target": 1, "type": "contains"},
                {"source": 1, "target": 2, "type": "imports"}
            ]
        })
    }

    #[test]
    fn phi_transform_produces_contains_edge() {
        let graph = phi_transform(&fixture_json()).unwrap();
        let edges: Vec<_> = graph.edges().collect();
        assert!(
            edges.iter().any(|e| e.relation == "contains"),
            "expected a 'contains' edge"
        );
    }

    #[test]
    fn phi_transform_produces_defined_in_edge() {
        let graph = phi_transform(&fixture_json()).unwrap();
        let edges: Vec<_> = graph.edges().collect();
        assert!(
            edges.iter().any(|e| e.relation == "defined_in"),
            "expected a 'defined_in' edge for source_file"
        );
    }
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
cargo test -p larql-cli phi_transform_produces_contains_edge
```

Expected: `error[E0425]: cannot find function 'phi_transform'`

- [ ] **Step 3: Implement `phi_transform` and the command**

```rust
use std::{collections::HashMap, path::PathBuf};
use clap::Args;
use larql_core::{
    core::{edge::Edge, enums::SourceType, graph::Graph},
    io::save_graph,
};
use serde_json::Value;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GraphifyError {
    #[error("missing 'nodes' array in graphify JSON")]
    MissingNodes,
    #[error("missing 'links' array in graphify JSON")]
    MissingLinks,
    #[error("IO: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("graph: {0}")]
    Graph(String),
}

pub fn phi_transform(v: &Value) -> Result<Graph, GraphifyError> {
    let nodes_arr = v["nodes"].as_array().ok_or(GraphifyError::MissingNodes)?;
    let links_arr = v["links"].as_array().ok_or(GraphifyError::MissingLinks)?;

    // Build id → label map
    let mut id_to_label: HashMap<String, String> = HashMap::new();
    for node in nodes_arr {
        let id = node["id"].to_string(); // stringify the id (int or str)
        let label = node["label"].as_str().unwrap_or("unknown").to_string();
        id_to_label.insert(id, label.clone());
    }

    let mut graph = Graph::new();

    // Node metadata edges
    for node in nodes_arr {
        let label = node["label"].as_str().unwrap_or("unknown");
        if let Some(kind) = node["kind"].as_str() {
            let mut e = Edge::new(label, "has_kind", kind);
            e.source = SourceType::Ast;
            e.confidence = 1.0;
            graph.add_edge(e);
        }
        if let Some(source_file) = node["source_file"].as_str() {
            let mut e = Edge::new(label, "defined_in", source_file);
            e.source = SourceType::Ast;
            e.confidence = 1.0;
            graph.add_edge(e);
        }
    }

    // Structural edges
    for link in links_arr {
        let src_id = link["source"].to_string();
        let tgt_id = link["target"].to_string();
        let relation = link["type"]
            .as_str()
            .or_else(|| link["relation"].as_str())
            .unwrap_or("references");
        let src_label = id_to_label.get(&src_id).map(|s| s.as_str()).unwrap_or(&src_id);
        let tgt_label = id_to_label.get(&tgt_id).map(|s| s.as_str()).unwrap_or(&tgt_id);
        let mut e = Edge::new(src_label, relation, tgt_label);
        e.source = SourceType::Ast;
        e.confidence = 1.0;
        graph.add_edge(e);
    }

    Ok(graph)
}

#[derive(Args)]
pub struct ExtractGraphifyArgs {
    /// Path to the graphify node-link JSON file.
    input: PathBuf,

    /// Output .larql.json path.
    #[arg(short, long, default_value = "graph.larql.json")]
    output: PathBuf,
}

pub fn run(args: ExtractGraphifyArgs) -> Result<(), Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(&args.input)?;
    let v: Value = serde_json::from_str(&text)?;
    let graph = phi_transform(&v)?;
    eprintln!(
        "φ-transform: {} nodes, {} edges → {}",
        graph.node_count(),
        graph.edge_count(),
        args.output.display()
    );
    save_graph(&graph, &args.output).map_err(|e| format!("{e}"))?;
    Ok(())
}
```

- [ ] **Step 4: Register in mod.rs and main.rs**

In `commands/extraction/mod.rs`:
```rust
pub mod extract_graphify_cmd;
```

In `main.rs` `Commands` enum:
```rust
#[command(next_help_heading = "Build")]
/// Convert a graphify node-link JSON to a .larql.json graph.
ExtractGraphify(extract_graphify_cmd::ExtractGraphifyArgs),
```

And the match arm:
```rust
Commands::ExtractGraphify(args) => extract_graphify_cmd::run(args),
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p larql-cli phi_transform_produces_contains_edge phi_transform_produces_defined_in_edge
```

Expected: both pass.

- [ ] **Step 6: Commit**

```bash
git add crates/larql-cli/src/commands/extraction/extract_graphify_cmd.rs \
        crates/larql-cli/src/commands/extraction/mod.rs \
        crates/larql-cli/src/main.rs
git commit -m "feat(cli): larql extract-graphify — φ-transform from NetworkX JSON to .larql.json"
```

---

### Task 6: `larql graph-diff` — consensus merge of two .larql.json graphs

**Files:**
- Create: `crates/larql-cli/src/commands/extraction/graph_diff_cmd.rs`
- Modify: `crates/larql-cli/src/commands/extraction/mod.rs`
- Modify: `crates/larql-cli/src/main.rs`

**Interfaces:**
- Produces: `larql graph-diff <A.larql.json> <B.larql.json> --output <merged.larql.json>` command.

Consensus rule: edges present in both A and B → confidence 1.0. Edges only in A → confidence 0.7. Edges only in B → confidence 0.7.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use larql_core::core::{edge::Edge, graph::Graph};

    fn graph_a() -> Graph {
        let mut g = Graph::new();
        g.add_edge(Edge::new("A", "calls", "B"));
        g.add_edge(Edge::new("A", "calls", "C")); // unique to A
        g
    }
    fn graph_b() -> Graph {
        let mut g = Graph::new();
        g.add_edge(Edge::new("A", "calls", "B")); // shared
        g.add_edge(Edge::new("B", "calls", "D")); // unique to B
        g
    }

    #[test]
    fn shared_edge_has_confidence_one() {
        let merged = consensus_merge(&graph_a(), &graph_b());
        let edges: Vec<_> = merged.edges().collect();
        let shared = edges.iter().find(|e|
            e.subject == "A" && e.relation == "calls" && e.object == "B"
        ).expect("shared edge missing");
        assert!((shared.confidence - 1.0).abs() < 1e-6);
    }

    #[test]
    fn unique_edge_has_confidence_0_7() {
        let merged = consensus_merge(&graph_a(), &graph_b());
        let edges: Vec<_> = merged.edges().collect();
        let unique_a = edges.iter().find(|e|
            e.subject == "A" && e.object == "C"
        ).expect("unique-A edge missing");
        assert!((unique_a.confidence - 0.7).abs() < 1e-6);
    }
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
cargo test -p larql-cli shared_edge_has_confidence_one
```

Expected: `error[E0425]: cannot find function 'consensus_merge'`

- [ ] **Step 3: Implement**

```rust
use std::{collections::HashSet, path::PathBuf};
use clap::Args;
use larql_core::{core::graph::Graph, io::{load_graph, save_graph}};

pub fn consensus_merge(a: &Graph, b: &Graph) -> Graph {
    let set_a: HashSet<(String, String, String)> = a
        .edges()
        .map(|e| (e.subject.clone(), e.relation.clone(), e.object.clone()))
        .collect();
    let set_b: HashSet<(String, String, String)> = b
        .edges()
        .map(|e| (e.subject.clone(), e.relation.clone(), e.object.clone()))
        .collect();

    let mut out = Graph::new();
    for e in a.edges() {
        let triple = (e.subject.clone(), e.relation.clone(), e.object.clone());
        let mut edge = e.clone();
        edge.confidence = if set_b.contains(&triple) { 1.0 } else { 0.7 };
        out.add_edge(edge);
    }
    for e in b.edges() {
        let triple = (e.subject.clone(), e.relation.clone(), e.object.clone());
        if !set_a.contains(&triple) {
            let mut edge = e.clone();
            edge.confidence = 0.7;
            out.add_edge(edge);
        }
    }
    out
}

#[derive(Args)]
pub struct GraphDiffArgs {
    /// First graph (.larql.json), typically from extract-codebase.
    graph_a: PathBuf,
    /// Second graph (.larql.json), typically from extract-graphify.
    graph_b: PathBuf,
    /// Output merged graph path.
    #[arg(short, long, default_value = "merged.larql.json")]
    output: PathBuf,
}

pub fn run(args: GraphDiffArgs) -> Result<(), Box<dyn std::error::Error>> {
    let a = load_graph(&args.graph_a).map_err(|e| format!("{e}"))?;
    let b = load_graph(&args.graph_b).map_err(|e| format!("{e}"))?;
    let merged = consensus_merge(&a, &b);
    eprintln!(
        "merged: {} edges (was {} + {})",
        merged.edge_count(), a.edge_count(), b.edge_count()
    );
    save_graph(&merged, &args.output).map_err(|e| format!("{e}"))?;
    Ok(())
}
```

- [ ] **Step 4: Register in mod.rs and main.rs**

```rust
// mod.rs addition:
pub mod graph_diff_cmd;

// Commands enum:
#[command(next_help_heading = "Build")]
/// Consensus-merge two .larql.json graphs (shared edges → confidence 1.0).
GraphDiff(graph_diff_cmd::GraphDiffArgs),

// match arm:
Commands::GraphDiff(args) => graph_diff_cmd::run(args),
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p larql-cli shared_edge_has_confidence_one unique_edge_has_confidence_0_7
```

Expected: both pass.

- [ ] **Step 6: Commit**

```bash
git add crates/larql-cli/src/commands/extraction/graph_diff_cmd.rs \
        crates/larql-cli/src/commands/extraction/mod.rs \
        crates/larql-cli/src/main.rs
git commit -m "feat(cli): larql graph-diff — consensus merge with confidence weighting"
```

---

### Task 7: `larql export --format gguf` command

**Files:**
- Create: `crates/larql-cli/src/commands/primary/export_cmd.rs`
- Modify: `crates/larql-cli/src/commands/primary/mod.rs`
- Modify: `crates/larql-cli/src/main.rs`

**Interfaces:**
- Consumes: existing `GgufWriter` + `GgufTensor` from `larql_models::loading::gguf::writer`, `larql_core::io::load_graph`, `larql_codebase_core::{graph_to_weight_repr, BitNetBasis}`
- Produces: `larql export --format gguf <LARQL_JSON_OR_VINDEX> --output <OUT.gguf>` command.

`GgufTensor { name, dims, ggml_type, data }` is the struct used by `GgufWriter::tensor()`.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use larql_core::core::{edge::Edge, graph::Graph};
    use larql_core::io::save_graph;

    #[test]
    fn gguf_header_magic_present() {
        let mut g = Graph::new();
        for i in 0..20 {
            g.add_edge(Edge::new(format!("n{i}"), "calls", format!("n{}", (i+1)%20)));
        }
        let graph_file = NamedTempFile::with_suffix(".larql.json").unwrap();
        save_graph(&g, graph_file.path()).unwrap();

        let out = NamedTempFile::with_suffix(".gguf").unwrap();
        export_to_gguf(graph_file.path(), out.path()).unwrap();

        let bytes = std::fs::read(out.path()).unwrap();
        // GGUF magic: 0x46554747 = "GGUF" in little-endian
        assert_eq!(&bytes[0..4], b"GGUF", "expected GGUF magic bytes");
    }
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
cargo test -p larql-cli gguf_header_magic_present
```

Expected: `error[E0425]: cannot find function 'export_to_gguf'`

- [ ] **Step 3: Implement `export_to_gguf` and the command**

```rust
use std::path::{Path, PathBuf};
use clap::{Args, ValueEnum};
use larql_core::io::load_graph;
use larql_codebase_core::{basis::BitNetBasis, graph_to_weight_repr};
use larql_models::loading::gguf::writer::{GgufTensor, GgufValue, GgufWriter};

#[derive(ValueEnum, Clone)]
pub enum ExportFormat {
    Gguf,
}

#[derive(Args)]
pub struct ExportArgs {
    /// Input: a .larql.json graph or a .vindex directory.
    input: PathBuf,

    /// Output file path.
    #[arg(short, long)]
    output: PathBuf,

    /// Export format.
    #[arg(long, value_enum, default_value = "gguf")]
    format: ExportFormat,
}

pub fn export_to_gguf(input: &Path, output: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let graph = load_graph(input).map_err(|e| format!("load graph: {e}"))?;
    let repr = graph_to_weight_repr(&graph, &BitNetBasis);

    let mut writer = GgufWriter::new();
    writer.meta("general.architecture", GgufValue::String("bitnet".into()));
    writer.meta("general.name", GgufValue::String("larql-codebase-bitnet".into()));
    writer.meta(
        "larql.hidden_size",
        GgufValue::U32(repr.arch.hidden_size as u32),
    );
    writer.meta("larql.n_layers", GgufValue::U32(repr.arch.n_layers as u32));

    for t in repr.tensors {
        writer.tensor(GgufTensor {
            name: t.name,
            dims: t.dims,
            ggml_type: t.ggml_type,
            data: t.data,
        });
    }

    writer.write_to_file(output)?;
    eprintln!(
        "Wrote {} tensors → {}",
        writer.tensor_count(),
        output.display()
    );
    Ok(())
}

pub fn run(args: ExportArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.format {
        ExportFormat::Gguf => export_to_gguf(&args.input, &args.output),
    }
}
```

Note: `GgufValue` variants confirmed: `String(String)`, `U32(u32)`, `Bool(bool)`, `U8(u8)`, `U16(u16)`, `I32(i32)` — from `larql-models/src/loading/gguf/writer.rs`.

- [ ] **Step 4: Register in mod.rs and main.rs**

```rust
// primary/mod.rs addition:
pub mod export_cmd;

// Commands enum:
#[command(next_help_heading = "Build")]
/// Export a .larql.json graph to a model format (GGUF).
Export(export_cmd::ExportArgs),

// match arm:
Commands::Export(args) => export_cmd::run(args),
```

- [ ] **Step 5: Verify `GgufValue` variant names**

```bash
grep "pub enum GgufValue\|String\|Uint32\|Float32" \
  crates/larql-models/src/loading/gguf/writer.rs | head -15
```

Adjust the `GgufValue::` calls in `export_to_gguf` if variant names differ.

- [ ] **Step 6: Run tests**

```bash
cargo test -p larql-cli gguf_header_magic_present
```

Expected: `test commands::primary::export_cmd::tests::gguf_header_magic_present ... ok`

- [ ] **Step 7: Commit**

```bash
git add crates/larql-cli/src/commands/primary/export_cmd.rs \
        crates/larql-cli/src/commands/primary/mod.rs \
        crates/larql-cli/src/main.rs
git commit -m "feat(cli): larql export --format gguf — .larql.json graph → GGUF v3 file"
```

---

### Task 8: Final integration test — full pipeline

- [ ] **Step 1: Build the full workspace**

```bash
cargo build --workspace
```

Expected: `Finished`, no errors.

- [ ] **Step 2: Run all tests**

```bash
cargo test --workspace
```

Expected: all tests pass.

- [ ] **Step 3: End-to-end smoke test**

```bash
# Extract this codebase
cargo run -p larql-cli -- extract-codebase . --output /tmp/larql-test.vindex

# Export graph to .larql.json (uses the same pipeline internally)
cargo run -p larql-cli -- extract-codebase . --output /tmp/larql-tmp && \
  echo "extract-codebase OK"

# Now generate GGUF from a small fixture:
cat > /tmp/test.larql.json << 'EOF'
{"larql_version":"0.1.0","metadata":{},"schema":{},"edges":[
  {"s":"A","r":"calls","o":"B","c":1.0,"src":"ast"},
  {"s":"B","r":"calls","o":"C","c":1.0,"src":"ast"},
  {"s":"C","r":"calls","o":"A","c":1.0,"src":"ast"}
]}
EOF
cargo run -p larql-cli -- export /tmp/test.larql.json --format gguf --output /tmp/test.gguf
file /tmp/test.gguf
```

Expected: `file` reports `GGUF data` or similar; file exists with non-zero size.

- [ ] **Step 4: make ci**

```bash
make ci
```

Expected: fmt + clippy + test all pass.
