# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LARQL decompiles transformer model weights into a **vindex** — a directory of mmap'd files queryable like a graph database. **LQL** (Lazarus Query Language) is the SQL-like surface for browsing, mutating, and recompiling that knowledge.

For the broader architectural context, system design, and workspace layout, see `./AGENTS.md`.

## Development Workflow

### Setup
```bash
cargo build --release                          # Build optimized binaries
cargo test --workspace                         # Run all tests (272+ tests across crates)
make ci                                        # Run CI checks locally (fmt, clippy, test)
make fmt                                       # Format code
make lint                                      # Run clippy linter
```

Python bindings (maturin-built):
```bash
cd crates/larql-python
uv sync --no-install-project --group dev
uv run --no-sync maturin develop --release
uv run --no-sync pytest tests/
```

### Pre-commit Hooks
After cloning, install pre-commit hooks to catch issues before pushing:
```bash
pre-commit install
```

Hooks will automatically run on `git commit` and can be run manually with:
```bash
pre-commit run --all-files
```

### Code Style
- **Formatting**: Rust code must pass `cargo fmt --check` (enforced by CI)
- **Linting**: All code must pass `cargo clippy -- -D warnings` (enforced by CI)
- **Edition**: Workspace uses Rust 2021 edition
- **Configuration**: See `.editorconfig` for cross-IDE consistency; `rustfmt.toml` for Rust-specific formatting

## Workspace Organization

Strict dependency hierarchy (respect this when adding modules):
```
larql-models → larql-compute → larql-vindex → {larql-core, larql-inference}
                                              → larql-lql → {larql-server, larql-cli}
```

**Key invariant**: `model-compute` is portable and never imports `larql-*` modules. It may move to a sibling repository.

## Where to Add Code

**New feature (statement)**: Add AST variant in `crates/larql-lql/src/ast.rs`, then implement in both:
- `crates/larql-lql/src/parser/` (lexing/parsing)
- `crates/larql-lql/src/executor/` (execution logic)

**New command**: Place in `crates/larql-cli/src/commands/extraction/` or `crates/larql-cli/src/commands/query/`, then wire into the `Commands` enum in `crates/larql-cli/src/main.rs`.

**Core algorithm**: Belongs in `crates/larql-core/` (graph algorithms, merge, diff, BFS, etc.).

**Vindex operations**: Modify `crates/larql-vindex/src/` (includes patch overlay system).

## Key Architectural Invariants

1. **Base vindexes are immutable** — all mutation flows through `PatchedVindex` overlay
2. **Storage is mmap-first** — don't load entire tensors into RAM unless necessary
3. **Three extraction levels** — `browse`, `inference`, `all` (gated by `ExtractLevel` enum)
4. **Walk FFN is sparse-by-design** — preserve this invariant even if optimizing FFN code

See `./AGENTS.md` and `docs/specs/` for detailed specifications.

## Testing

Run tests for specific crates:
```bash
cargo test -p larql-lql                       # Test LQL parser/executor
cargo test -p larql-inference --features metal # Test inference with Metal GPU
cargo test -p <crate> <test_name>             # Run specific test
```

## Documentation

- **Architecture & design**: See `./AGENTS.md`
- **LQL language spec**: `docs/specs/lql-spec.md`
- **Vindex format**: `docs/specs/vindex-format-spec.md`
- **Operations & patches**: `docs/specs/vindex-operations-spec.md`
- **Inference internals**: `docs/inference-engine.md`, `docs/ffn-graph-layer.md`
- **Experimental work**: `experiments/` (numbered 01-07, each self-contained)

## Making Changes

- Use the `--help` flag on any CLI command for syntax: `larql extract-index --help`
- All pull requests must pass CI checks (see `.github/workflows/ci.yml`)
- Commit messages should be descriptive and reference any related issues
- See `.github/pull_request_template.md` for PR guidelines