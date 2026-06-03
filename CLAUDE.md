# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Reference the `./AGENTS.md` file for the broader context of this project.

## Commands

```bash
cargo build --release                             # optimized build
cargo build --release --features metal            # macOS Metal GPU
cargo test                                        # whole workspace
cargo test -p larql-lql                           # single crate
cargo test -p larql-core test_name                # single test
cargo fmt --all                                   # format
cargo clippy --workspace --tests -- -D warnings   # lint
make ci                                           # fmt + clippy + test
```

Expert modules — separate nested workspace, builds to WASM:
```bash
cd crates/larql-experts
cargo build --target wasm32-wasip1 --release
cargo test --all
```

Python bindings — maturin/uv, not cargo:
```bash
cd crates/larql-python
uv sync --no-install-project --group dev
uv run --no-sync maturin develop --release
uv run --no-sync pytest tests/
```

## Two-workspace architecture

The repo has two independent Cargo workspaces:

1. **Root workspace** (`Cargo.toml`): 15 crates forming the LARQL stack.
2. **`crates/larql-experts/`**: 20 expert crates that compile to `wasm32-wasip1`. They are *not* in the root workspace — `larql-inference` loads them at runtime as `.wasm` files, not as Rust crate dependencies.

## Crate dependency chain

```
larql-models → larql-compute → larql-vindex
                                     ↓
                          larql-core   larql-inference
                                     ↓
                               larql-lql
                                  ↓
                     larql-server   larql-cli   larql-python
```

`model-compute` has no `larql-*` imports — it is designed to extract into a sibling repo. `kv-cache-benchmark` and `larql-router-protocol` sit outside this chain.

## Storage and mutation invariants

Base vindexes are read-only mmap files. All mutation flows through `PatchedVindex` (overlay). `INSERT`/`DELETE`/`UPDATE` auto-start a patch; base files are never written. `COMPILE CURRENT INTO VINDEX` bakes a patch into a new standalone vindex by hardlinking unchanged base files.

## Adding an LQL statement

1. Add the AST node in `crates/larql-lql/src/ast.rs`.
2. Add matching entries in **both** `src/parser/` and `src/executor/` — they are symmetrical mirrors (`lifecycle.rs`, `query.rs`, `mutation.rs`, `introspection.rs`, `trace.rs`).
3. If the statement requires extraction state, gate it on `ExtractLevel` in `crates/larql-vindex/src/config/types.rs`.
