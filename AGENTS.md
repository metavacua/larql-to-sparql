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
larql-lql         lexer/parser/executor/REPL + USE REMOTE client
    ↓
larql-server      HTTP + gRPC server serving vindexes
larql-cli         top-level `larql` binary (every subcommand lives in commands/)
larql-python      PyO3 bindings (maturin-built, module name `larql._native`)
kv-cache-benchmark    standalone benchmark crate

# Portable (no LARQL deps; extract to sibling repo later, name stable)
model-compute         bounded native kernels (arithmetic/datetime) and optional
                      wasmtime-hosted WASM modules (features: `native`/`wasm`)
```

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

## Known build issues

These are pre-existing problems that the CI surfaces as failing checks. They
are NOT acceptable long-term state — the CI gates them as failures so coding
agents see them prominently and prioritise fixing them. Do NOT suppress these
checks with `continue-on-error` or by removing assertions; the right response
to a failing gate is to fix the underlying issue or, if the fix is genuinely
out of scope for the current PR, to leave the failure visible and document it
here so the next agent knows what to tackle.

### Workspace compile errors (blocks: S1 lockfile-consistency-via-compile, S2 clippy, S2 rustdoc, S3 build-test-matrix)
- `crates/larql-compute` does not compile under the pinned 1.88.0 toolchain.
  At the time of writing, `cargo check --workspace` reports ~35 errors
  (`E0282` type annotation needed, `E0432` unresolved import, etc.).
- These are real regressions — not lint nits. Fixing them is a prerequisite
  for the entire S2/S3 stack to turn green.
- Remediation: open a dedicated PR that compiles `larql-compute` cleanly
  (start with `cargo check -p larql-compute`, fix one error at a time,
  prefer narrow per-file commits so the diff is reviewable).
- The `lockfile-consistency` job has been re-architected to use
  `cargo metadata --locked` rather than `cargo check --locked`, so the
  lockfile <-> Cargo.toml gate is independent of the compile-error stack.

### Clippy lint violations on benches/tests (blocks: S2 clippy)
- `cargo clippy --workspace --all-targets -- -D warnings` includes benches
  and tests; benches contain ~20+ unchecked `.unwrap()` calls
  (e.g., `crates/larql-vindex/benches/vindex_ops.rs`,
  `benches/q4k_vs_f32.rs`).
- The `--all-targets` flag is intentional: benches/tests are first-class
  CI surfaces and their `unwrap()` failures hide real correctness bugs.
- Remediation: replace `.unwrap()` with `.expect("explanatory message")` or
  `?`-propagation in benches/tests. Cleanup PR.

### Rustdoc lint warnings (blocks: S2 rustdoc)
- `crates/larql-vindex/src/lib.rs:1-2` carries
  `#![allow(clippy::doc_overindented_list_items)]` and
  `#![allow(clippy::doc_lazy_continuation)]`. These suppressions exist
  because doc comments across multiple crates have indentation /
  continuation issues that `cargo doc -D warnings` would otherwise reject.
- Remediation: run `cargo doc -D warnings` locally, fix each warning,
  remove the `#![allow(...)]` attributes once clean. Cleanup PR.

### cargo-deny failure (blocks: S1 cargo-deny)
- `cargo deny check advisories bans licenses sources` exits with code 3
  on the committed `Cargo.lock`. Prior to lockfile commitment the workflow
  generated a fresh lockfile each run, so any new transitive dependency
  with a license outside the allow-list (or any newly-disclosed advisory
  affecting a current dep) is now caught at this gate where it would
  previously have been masked.
- Remediation: read the failing job log, identify the specific
  `error[license-not-explicitly-allowed]`, `error[advisory]`, or
  `error[ban]` annotation, then either pin the offending dependency to a
  compatible version or — only if the audit justifies it — extend
  `deny.toml :: [licenses] allow` per the procedure in
  `audit/dependency-licenses.md` (D-1 through D-7 are the precedent).

## CI/CD conventions for coding agents

This project's CI is the source of truth. Every constraint below is encoded
in `.github/workflows/ci.yml` and accompanying scripts; the text here exists
so a coding agent does not have to reverse-engineer the workflow before
making a change.

### CHANGELOG.md is derived from commits (NEVER edit manually)
- The `[Unreleased]` block of `CHANGELOG.md` is the deterministic projection
  of the validated Conventional Commits in range `LAST_TAG..HEAD`, computed
  by `git-cliff` using `cliff.toml`.
- BEFORE pushing any commit, run: `bash scripts/check_changelog.sh`.
- If it fails, regenerate with:
  `git-cliff --config cliff.toml --unreleased --strip header --prepend CHANGELOG.md`,
  then manually delete the duplicated `[Unreleased]` block that `--prepend`
  leaves above the file's existing header. The awk extractor in
  `scripts/check_changelog.sh` expects the [Unreleased] section to end with
  exactly two trailing blank lines before the next `## [` heading or EOF.
- **Recursive trap**: each new commit adds a new entry to the projection,
  so CHANGELOG.md must be updated in the SAME commit as the substantive
  change. Otherwise every "fix CHANGELOG" commit triggers another
  CHANGELOG mismatch and the loop never terminates.

### Cargo.lock is committed and must remain reproducible
- The workspace-root `Cargo.lock` is committed and is part of the
  reproducible-build contract. GitHub's Dependency Graph and Dependabot
  (cargo + uv) both rely on it.
- Do NOT regenerate via `cargo update` unless intentionally bumping deps
  (in which case run `cargo update --workspace -p <crate>` for the specific
  bump, not a wholesale refresh).
- If `cargo metadata --locked` fails locally, it is reporting drift
  between `Cargo.toml` and `Cargo.lock`. Investigate the root cause; do
  NOT paper over by regenerating without recording the dependency change
  in a separate `chore: bump <crate>` commit.

### REUSE.toml is the authoritative provenance manifest
- Every file under version control must be covered by an `[[annotations]]`
  block in the root `REUSE.toml`.
- Do NOT add SPDX headers to machine-config files
  (`.github/workflows/*.yml`, `deny.toml`, `cliff.toml`, etc.). Their
  copyright/license metadata lives in `REUSE.toml` only — embedded headers
  break naive YAML/TOML parsers.
- Per-provenance copyright: original LARQL files (from
  https://github.com/chrishayuk/larql) carry "Chris Hay"; fork additions
  carry "Ian Douglas Lawrence Norman McLean"; files with non-trivial fork
  modifications must carry BOTH lines (Apache 2.0 §4(b)).
- Never use a placeholder copyright like "Contributors to ..." — those
  cannot execute a license grant. Name the actual rights-holder.
- `LEGAL_TRANSITION.md` documents the planned forward licensing posture
  (AGPL-3.0-or-later for code, CC-BY-SA-4.0 for docs); do NOT pre-stage
  those identifiers in active `[[annotations]]` blocks until files are
  actually re-licensed.

### Conventional Commits are enforced
- `cog check` runs in CI on the PR commit range and on the newest commit
  on push to main. Pre-existing history is grandfathered.
- Allowed types: `feat`, `fix`, `chore`, `ci`, `docs`, `refactor`, `test`,
  `perf`, `style`, `build` (see `cog.toml`).
- Use `!` after type or a `BREAKING CHANGE:` footer to signal a major bump.
  Be aware: this changes the git-cliff projection (the entry is rendered
  with `**BREAKING:**` prefix), which means CHANGELOG.md must be updated
  in the same commit.

## Where to find things

- LQL language spec: [docs/specs/lql-spec.md](docs/specs/lql-spec.md) (v0.3)
- Vindex file format: [docs/specs/vindex-format-spec.md](docs/specs/vindex-format-spec.md)
- Operations + patches: [docs/specs/vindex-operations-spec.md](docs/specs/vindex-operations-spec.md)
- Ecosystem (HF publish, Vindexfile): [docs/specs/vindex-ecosystem-spec.md](docs/specs/vindex-ecosystem-spec.md)
- Inference engine internals: [docs/inference-engine.md](docs/inference-engine.md), [docs/ffn-graph-layer.md](docs/ffn-graph-layer.md)
- Trace format (.bin/.bndx/.ctxt): [docs/specs/trace-format-spec.md](docs/specs/trace-format-spec.md), [docs/residual-trace.md](docs/residual-trace.md)
- Experimental work: [experiments/](experiments/) — numbered 01-07, each self-contained
- Python bindings docs: [crates/larql-python/README.md](crates/larql-python/README.md), [docs/larql-python.md](docs/larql-python.md)
