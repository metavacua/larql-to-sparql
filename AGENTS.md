# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

LARQL decompiles transformer model weights into a **vindex** — a directory of mmap'd files that can be queried like a graph database. **LQL** (Lazarus Query Language) is the SQL-like surface for browsing, mutating, and recompiling that knowledge. The core claim: the model *is* the database, so edits are structural (patch overlays on gate/down matrices), not fine-tuning.

Three extraction levels gate which LQL statements work: `browse` (DESCRIBE/WALK/SELECT), `inference` (+INFER), `all` (+COMPILE). Patches (`.vlp` JSON files) stack onto a readonly base vindex — INSERT/DELETE/UPDATE auto-start a patch; base files are never mutated.

## Workspace layout

Cargo workspace at repo root with a strict dependency chain — respect this when adding modules:

```
# LARQL-specific (depend on vindex, LQL, etc.)
larql-models      model config, architecture traits, weight loading, quant/dequant
    ↓
larql-compute     CPU/Metal matmul backends, pipeline
    ↓
larql-vindex      vindex lifecycle: extract, load, query, mutate, patch, save, Vindexfile
    ↓
larql-core        graph algorithms (merge, diff, BFS, pagerank, shortest-path)
larql-inference   forward pass, BLAS-fused attention, Metal GPU, WalkFfn, trace
    ↓
larql-lql-core    LQL AST + lexer + parser (WASM-pure: only thiserror dep, no cfg guards)
    ↓
larql-lql         LQL executor + REPL + USE REMOTE client (depends on larql-lql-core)
    ↓
larql-server      HTTP + gRPC server serving vindexes
larql-cli         top-level `larql` binary (every subcommand lives in commands/)
larql-python      PyO3 bindings (maturin-built, module name `larql._native`)
kv-cache-benchmark    standalone benchmark crate

# Portable (no LARQL deps; extract to sibling repo later, name stable)
model-compute         bounded native kernels (arithmetic/datetime); wasmi-hosted
                      WASM modules (universal default, all platforms including
                      arm32, features: `native`/`wasm`); original wasmtime JIT
                      backend preserved for REUSE compliance (desktop only,
                      feature: `wasm-jit`)
```

## WASM portability tiers

The workspace is partitioned into three formal tiers using a predicate system:

**Predicates** (properties of a crate's own source and direct Cargo deps):

| Symbol | Meaning |
|--------|---------|
| `Pᵢₒ`   | Requires file I/O (mmap, disk reads/writes) |
| `Pₙₑₜ`  | Requires network I/O (HTTP/gRPC) |
| `P_gpu` | Requires hardware accelerator (Metal, BLAS, CUDA) |
| `P_ffi` | Links a C library or foreign-language FFI (onig, BLAS-C, PyO3) |
| `Pᵣₜ`   | Requires async runtime (Tokio reactor) |
| `P_repl`| Requires interactive terminal (rustyline) |
| `Pᵤ`    | Pure: no side-effects beyond allocation |

**Tier 1 — W_pure** (`Pᵤ ∧ ¬Pᵢₒ ∧ ¬Pₙₑₜ ∧ ¬P_gpu ∧ ¬P_ffi ∧ ¬Pᵣₜ ∧ ¬P_repl`):
No cfg guards in own source. Compiles for `wasm32-unknown-unknown` without `--no-default-features`.
These form a proper closed subcategory W ⊆ C (the full workspace dependency category).

- `larql-boundary` — codec types
- `larql-models` — model config + architecture types
- **`larql-lql-core`** — LQL AST, lexer, parser (dep: thiserror only)
- `larql-wasm` — WASM graph bindings

**Tier 2 — W_compat** (compiles for wasm32 via `#[cfg]` gates in own source):
Not W_pure but usable from wasm32 targets. The cfg gates suppress native-only paths.

- `larql-core` — graph algorithms (`http` feature off by default)
- `larql-vindex` — vindex I/O (`remote` feature off by default)
- `larql-compute` — CPU/Metal backends (BLAS gated)
- `larql-inference` — forward pass (tonic, reqwest gated)
- `larql-kv` — KV store (BLAS gated)
- `larql-lql` — executor + REPL (rustyline, blocking reqwest gated)
- `larql-browser-cli` — WASM cdylib entry point (depends on Tier 2 for LqlSession)
- `model-compute` — bounded kernels (native backends gated)

**Tier 3 — Native-only** (does not compile for wasm32):

| Crate | Class | Blocking predicate(s) |
|-------|-------|-----------------------|
| `larql-server` | Σ_server | `Pₙₑₜ ∧ Pᵣₜ` (axum, tokio, tonic) |
| `larql-router` | Σ_server | `Pₙₑₜ ∧ Pᵣₜ` (axum, tokio, tonic) |
| `larql-router-protocol` | Σ_server | `Pᵣₜ` (tokio, tonic gRPC) |
| `larql-cli` | Σ_cli | binary entry point, `P_repl` (rustyline), `P_ffi` |
| `larql-python` | Σ_ffi | `P_ffi` (PyO3 language binding) |
| `kv-cache-benchmark` | Σ_bench | `Pₙₑₜ` (reqwest blocking), `P_ffi` (onig) |
| `larql-experts` | Σ_domain | domain expert sub-models (no WASM path) |

**Dependency graph (objects = crates, morphisms = Cargo deps):**

```
W_pure subcategory (closed under ←):
  larql-browser-cli ← larql-lql ← larql-lql-core  ← (thiserror)
                                                    ← (no other deps)

W_compat layer (compile for wasm32, have cfg gates):
  larql-lql ← larql-vindex ← larql-models
           ← larql-inference ← larql-compute
           ← larql-core

Native layer (Tier 3 → W_compat → W_pure):
  larql-cli, larql-server, larql-router → larql-lql → larql-lql-core
```

**CI enforcement:** `.github/workflows/larql-lql-core-wasm.yml` runs
`cargo check -p larql-lql-core --target wasm32-unknown-unknown` (no `--no-default-features`).
A native dep or cfg guard in `larql-lql-core` will fail this job.

**`model-compute` never imports `larql-*`.** Dependency flow is one-way:
LARQL may consume it (e.g. for compile-time `sum(1..100)` resolution); it
knows nothing about vindex or LQL. When it moves to a sibling repo, the
name stays the same so imports don't churn. The `install_edge` primitive
that stamps a compiled edge into gate/up/down tensors lives at
[crates/larql-cli/src/commands/extraction/compile_cmd/edge.rs](crates/larql-cli/src/commands/extraction/compile_cmd/edge.rs) —
it's the lowest-level step of the `COMPILE` verb and isn't a separate crate
until a second consumer needs it.

The CLI is a thin dispatcher: each `larql <cmd>` lives in [crates/larql-cli/src/commands/extraction/](crates/larql-cli/src/commands/extraction/) or [crates/larql-cli/src/commands/query/](crates/larql-cli/src/commands/query/) and is wired into the `Commands` enum in [crates/larql-cli/src/main.rs](crates/larql-cli/src/main.rs). `larql serve` exec's into `larql-server`. `larql repl` and `larql lql` delegate to `larql_lql::run_repl`/`run_statement`.

LQL parser and executor are split symmetrically: [crates/larql-lql/src/parser/](crates/larql-lql/src/parser/) and [crates/larql-lql/src/executor/](crates/larql-lql/src/executor/) both have matching `lifecycle.rs`, `query.rs`, `mutation.rs`, `introspection.rs`, `trace.rs`. When adding a statement, touch the AST in [crates/larql-lql/src/ast.rs](crates/larql-lql/src/ast.rs), then both sides.

## Build, test, run

```bash
cargo build --release                             # optimised build
cargo build --release --features metal            # Metal GPU backend (Apple Silicon)
cargo test                                        # entire workspace
cargo test -p larql-lql                           # single crate (272 tests)
cargo test -p larql-inference --features metal    # +Metal GPU tests
cargo test -p <crate> <test_name>                 # single test
make ci                                           # fmt-check + clippy -D warnings + test
make fmt                                          # cargo fmt --all
make lint                                         # cargo clippy --workspace --tests -- -D warnings
```

CLI (after `cargo build --release`): `./target/release/larql extract-index … | repl | lql '…' | convert | hf | build | serve | verify`. See [docs/cli.md](docs/cli.md) for the full surface.

Python bindings are maturin-built under uv (not cargo-run):

```bash
cd crates/larql-python
uv sync --no-install-project --group dev     # create .venv, install dev deps
uv run --no-sync maturin develop --release   # build PyO3 extension into .venv
uv run --no-sync pytest tests/               # run binding tests
```

Or via the Makefile: `make python-setup | python-build | python-test | python-clean`.

## Key architectural invariants

- **Base vindexes are immutable.** All mutation flows through `PatchedVindex` (overlay) — see [crates/larql-vindex/src/patch/core.rs](crates/larql-vindex/src/patch/). `INSERT/DELETE/UPDATE` auto-start a patch; `SAVE PATCH` persists it as `.vlp` JSON. Never write through to base files.
- **`COMPILE CURRENT INTO VINDEX`** bakes patches into a new standalone vindex by hardlinking base weight files (APFS fast path) and rewriting only `down_weights.bin` column-wise. No sidecar at load time.
- **Storage is mmap-first.** Gate vectors, embeddings, down weights are zero-copy `mmap`'d. f16 is the default dtype (`--f16` halves size with negligible accuracy loss). Don't load entire tensors into RAM unless an operation requires it.
- **Three extraction levels, not features.** `browse` (~3 GB), `inference` (~6 GB), `all` (~10 GB) — gated by `ExtractLevel` enum in [crates/larql-vindex/src/config/types.rs](crates/larql-vindex/src/config/types.rs). Check level before attempting an operation; fail loudly if weights aren't present.
- **Walk FFN is sparse-by-design and can beat dense** (517ms vs 535ms on Gemma 4B) because gate KNN (K≈10) skips most of the 10,240 features per layer. If you touch FFN code, preserve this invariant — see [docs/ffn-graph-layer.md](docs/ffn-graph-layer.md).
- **MXFP4 quantized MoE (GPT-OSS) has degraded DESCRIBE/WALK** due to 4-bit precision; `INFER` is the supported path. Don't assume all model families are equivalent — see [docs/specs/vindex-operations-spec.md](docs/specs/vindex-operations-spec.md).

## Where to find things

- LQL language spec: [docs/specs/lql-spec.md](docs/specs/lql-spec.md) (v0.3)
- Vindex file format: [docs/specs/vindex-format-spec.md](docs/specs/vindex-format-spec.md)
- Operations + patches: [docs/specs/vindex-operations-spec.md](docs/specs/vindex-operations-spec.md)
- Ecosystem (HF publish, Vindexfile): [docs/specs/vindex-ecosystem-spec.md](docs/specs/vindex-ecosystem-spec.md)
- Inference engine internals: [docs/inference-engine.md](docs/inference-engine.md), [docs/ffn-graph-layer.md](docs/ffn-graph-layer.md)
- Trace format (.bin/.bndx/.ctxt): [docs/specs/trace-format-spec.md](docs/specs/trace-format-spec.md), [docs/residual-trace.md](docs/residual-trace.md)
- Experimental work: `~/chris-source/chris-experiments/` — numbered 01-45, grouped into foundations, compilation, routing, and shannon series
- Python bindings docs: [crates/larql-python/README.md](crates/larql-python/README.md), [docs/larql-python.md](docs/larql-python.md)
