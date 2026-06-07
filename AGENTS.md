# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

LARQL decompiles transformer model weights into a **vindex** — a directory of mmap'd files that can be queried like a graph database. **LQL** (Lazarus Query Language) is the SQL-like surface for browsing, mutating, and recompiling that knowledge. The core claim: the model *is* the database, so edits are structural (patch overlays on gate/down matrices), not fine-tuning.

Three extraction levels gate which LQL statements work: `browse` (DESCRIBE/WALK/SELECT), `inference` (+INFER), `all` (+COMPILE). Patches (`.vlp` JSON files) stack onto a readonly base vindex — INSERT/DELETE/UPDATE auto-start a patch; base files are never mutated.

## Workspace layout

Cargo workspace at repo root with a strict dependency chain — respect this when adding modules:

```
# LARQL-specific (depend on vindex, LQL, etc.)
larql-models      model config, architecture traits, weight loading, quant/dequant,
                  multi-modal trait surface (ModalEncoder, Connector,
                  MultiModalProtocol, EmbeddingPlan types), vision tower
                  config+weights+loader (encoders/vision_tower.rs), projector
                  weights+loader (connectors/projector.rs), shared
                  test_fixtures (behind `test-utils` feature)
    ↓
larql-compute     CPU substrate: BLAS kernels, residual norms, attention spine
                  (rope/gqa/block/decode/gpu), forward-pass primitives (embed,
                  embed_plan, EmbeddingPlan, ops, hooks, ple, layer, predict/raw),
                  kquant_forward Q4_K/Q6_K decode helpers, FfnBackend trait +
                  dense WeightFfn impl, KvDispatch + AsyncComputeBackend traits +
                  CpuBackend impls, KvIndex trait (abstracts VectorIndex for
                  substrate callers), forward_overrides env-var registry,
                  PerLayerDecodeState, vision encoder CPU forward
                  (encoders/vision_tower.rs), projector CPU forward
                  (connectors/projector.rs). ADR-0022 moved substrate down from
                  larql-inference; the substrate is now self-contained.
    ↓
larql-compute-metal  Metal GPU backend (first-class peer, NOT a thin layer).
                     Implements ComputeBackend / KvDispatch / AsyncComputeBackend
                     for MetalBackend; ships custom MSL shaders, multi-layer
                     pipelining, stage-bisected kernels.
    ↓
larql-vindex      vindex lifecycle: extract, load, query, mutate, patch, save,
                  Vindexfile. Implements `KvIndex for VectorIndex` (Step 3a).
    ↓
larql-core        graph algorithms (merge, diff, BFS, pagerank, shortest-path)
larql-inference   engines (Standard, MarkovResidual, Apollo, etc.), chat,
                  sessions, tokenizer, FFN routing impls (Graph/Remote/MoE),
                  layer_executor, layer_graph orchestration. KvEngine trait
                  (with supports_multimodal + prefill_from_hidden per
                  ADR-0023), AnyEngine dispatch enum (KvEngine | RetrievalEngine).
                  Substrate moves to larql-compute; this crate is the
                  inference-shaped layer that composes substrate primitives +
                  engine state.
    ↓
larql-lql         lexer/parser/executor/REPL + USE REMOTE client
    ↓
larql-server      HTTP + gRPC server serving vindexes
larql-cli         top-level `larql` binary (every subcommand lives in commands/).
                  Multi-modal: `--image` + `--mm-weights` flags on `larql run`,
                  image decode/resize (image_input.rs), plan assembly
                  (run_cmd_image.rs). 3-image regression test in
                  tests/multimodal_e2e.rs (#[ignore], NOT FOR CI).
larql-python      PyO3 bindings (maturin-built, module name `larql._native`)

# Portable (no LARQL deps; extract to sibling repo later, name stable)
model-compute         bounded native kernels (arithmetic/datetime); wasmi-hosted
                      WASM modules (universal default, all platforms including
                      arm32, features: `native`/`wasm`); original wasmtime JIT
                      backend preserved for REUSE compliance (desktop only,
                      feature: `wasm-jit`)
```

**Metal is a first-class peer** (ADR-0022, 2026-05-18). `larql-compute-metal`
is the same shape as a future `larql-compute-vulkan` / `larql-compute-cuda` —
its own crate, implements the same trait surface, owns its kernels. Inference
factories (`default_engine_backend()`, `default_async_engine_backend()`,
`default_compute_backend()` in `larql-inference/src/lib.rs`) compose Metal +
CPU fallback explicitly; engine-level orchestration in `layer_graph/` still
branches on `#[cfg(feature = "gpu", target_os = "macos")]` where the
hybrid + GPU prefill paths take backend-specific actions.

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
cargo build --release --features gpu              # GPU backend (Metal today; Vulkan/CUDA later)
cargo test                                        # entire workspace
cargo test -p larql-lql                           # single crate (272 tests)
cargo test -p larql-inference --features gpu      # +GPU tests (Metal on Apple Silicon)
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
- **Substrate-vs-engine split** (ADR-0022): all CPU forward-pass math + attention + KvDispatch/AsyncComputeBackend traits live in `larql-compute`, not `larql-inference`. When adding a new substrate primitive (a kernel, an attention variant, a new norm), put it in `larql-compute` and re-export from `larql-inference` for back-compat. When adding engine-shaped code (a new session type, an FFN routing impl, a layer-graph dispatcher), it stays in `larql-inference`. The rule of thumb: substrate consumes `&dyn larql_compute::KvIndex` + `ModelWeights`; engines consume sessions, tokenizers, gRPC clients, layer_graphs.
- **VectorIndex is reached through `KvIndex` from substrate.** `larql-compute`'s `KvDispatch` + `AsyncComputeBackend` + `kquant_forward` take `Option<&dyn KvIndex>` parameters. `larql-vindex` impls `KvIndex for VectorIndex` in `kv_index_impl.rs`. Engine callers passing `&VectorIndex` to substrate traits coerce with `.map(|v| v as &dyn larql_compute::KvIndex)`. Don't reach for `larql_vindex::*` from inside `larql-compute` — that's the cycle the trait was created to avoid.

## WASM workspace (`crates/larql-wasm/`)

A **separate nested Cargo workspace** — not a member of the root workspace. It is a
refactoring sandbox: the goal is to verify wasm32v1-none compilability of core crates
without touching the working root workspace implementation.

**Three crate families:**
- `*-interface` — byte-for-byte copies of the native crates. The interface inversion
  step (Phase 1 of the WASM refactoring plan) will replace their bodies with references
  to the `*-wasm32v1-none` crates.
- `*-wasm32v1-none` — the authoritative pure implementations. All `*-wasm32v1-none`
  crates depend only on other `*-wasm32v1-none` crates; zero contamination from native
  or interface crates. Compile with `--target wasm32v1-none`.
- `larql-wasm-certify`, `larql-vindex-mvv-cdylib`, `larql-wasm-math` — certification
  tools and the MVV (Minimal-Viable Vindex) certified kernel artifact.

**Key invariant:** `*-wasm32v1-none` crates must never depend on `-interface` crates.
If you see such a dependency, it is a bug — the entire workspace would need to be
re-derived.

**Build the wasm workspace:**
```bash
cd crates/larql-wasm
cargo build --target wasm32v1-none --release  # wasm32v1-none crates
cargo test --all                               # native tests (interface crates)
```

**Certify the MVV kernel:**
```bash
cd crates/larql-wasm
cargo build -p larql-vindex-mvv-cdylib --target wasm32v1-none --release
cargo run -p larql-wasm-certify -- --strict \
  target/wasm32v1-none/release/larql_vindex_mvv_cdylib.wasm
# Expected: WASM-SAFE [¬L∧¬M]: import-free + call_indirect-free + memory.grow-free + recursion-free
```

See `docs/adr/0013-lql-wasm32v1-none-correspondence.md` for the ¬L∧¬M boundary for
each LQL statement.

## Where to find things

- LQL language spec: [docs/specs/lql-spec.md](docs/specs/lql-spec.md) (v0.3)
- Vindex file format: [docs/specs/vindex-format-spec.md](docs/specs/vindex-format-spec.md)
- Operations + patches: [docs/specs/vindex-operations-spec.md](docs/specs/vindex-operations-spec.md)
- Ecosystem (HF publish, Vindexfile): [docs/specs/vindex-ecosystem-spec.md](docs/specs/vindex-ecosystem-spec.md)
- Inference engine internals: [docs/inference-engine.md](docs/inference-engine.md), [docs/ffn-graph-layer.md](docs/ffn-graph-layer.md)
- Trace format (.bin/.bndx/.ctxt): [docs/specs/trace-format-spec.md](docs/specs/trace-format-spec.md), [docs/residual-trace.md](docs/residual-trace.md)
- Experimental work: `~/chris-source/chris-experiments/` — numbered 01-45, grouped into foundations, compilation, routing, and shannon series
- Python bindings docs: [crates/larql-python/README.md](crates/larql-python/README.md), [docs/larql-python.md](docs/larql-python.md)
