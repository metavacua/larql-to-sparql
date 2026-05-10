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

## Cross-OS environment leaks (the actual cascade)

This codebase originated as a macOS-only project (upstream:
<https://github.com/chrishayuk/larql>). Its CI exercises Linux primarily
(containerized — see `.github/docker/linux-ci/Dockerfile`) and macOS
secondarily (native `macos-15` and `macos-13` runners). Windows,
ARM-Linux, Android, and ChromeOS are untested.

**Most CI failures on Linux are host-environment-assumption leaks, not
source-code defects.** Earlier versions of this section, plus the
now-removed `::error` annotations in `.github/workflows/ci.yml`,
asserted that the clippy gate was blocked by "~35 compile errors in
`larql-compute`" or "20+ unchecked `.unwrap()` calls in benches/tests."
Both claims were misframed: clippy halts at layer 1 below and never
reaches either, and `.unwrap()` is not lint-checked by this project's
clippy config (`clippy::unwrap_used` is a `restriction` lint requiring
explicit opt-in, and no `clippy.toml` or `#![warn(clippy::unwrap_used)]`
exists in the tree). Removing every `.unwrap()` would not advance the
gate by one step.

Before "fixing" any failing CI step, run the OS-assumption checklist
below; if any item applies, the source is probably fine and the
*environment or build configuration* is what wants attention.

### OS-assumption checklist (run BEFORE editing a `.rs` file)

1. **Is this lint warn-by-default on Rust 1.88.0 only?** Some lints
   (e.g. `clippy::uninlined_format_args`) were demoted to pedantic in
   newer clippy. Reproduce against the pinned toolchain
   (`rust-toolchain.toml` at repo root pins `1.88.0`), not your local
   one. A "fixed for me / red in CI" cycle almost always means a
   toolchain mismatch between dev machine and CI.
2. **Is a default feature flag activating an optional dependency that
   is target-cfg-gated to macOS?** `crates/larql-cli/Cargo.toml` sets
   `default = ["metal"]`. Workspace feature unification propagates
   `metal=true` to every dependent crate, including `larql-compute` on
   Linux, where the optional `metal` external crate is gated to
   `target_os = "macos"` (`crates/larql-compute/Cargo.toml`). Symptom:
   ~35 compile errors under `cargo check --workspace` on Linux.
3. **Is a system library required at link time?** `openblas-src` with
   `features = ["system"]` (declared by `crates/larql-compute` and
   `crates/larql-inference`) requires `libopenblas-dev` on the host.
   The Linux CI container ships it; macOS uses Accelerate (bundled with
   the OS). If you change Cargo features that affect linking, also
   update `.github/docker/linux-ci/Dockerfile`.
4. **Is the failure on a target nobody has tested?** Windows-specific
   path-handling, ChromeOS-specific containerisation, and Android
   cross-compilation gaps are not bugs in the Rust source — they are
   unfilled platform-support work. Don't "fix" them by editing source
   that compiles fine on Linux/macOS.

### Reproduction recipe

```bash
# Local reproduction with the pinned toolchain (catches lint-set drift):
rustup toolchain install 1.88.0
cargo +1.88.0 clippy --workspace --all-targets -- -D warnings 2>&1 | head -50
```

If running on a newer toolchain shows no warnings — that divergence is
itself the most common reason an agent reports "fixed" while CI stays
red. For a fully faithful reproduction (matches CI's apt packages too):

```bash
docker run --rm -v "$PWD":/work -w /work \
  ghcr.io/metavacua/larql-ci-linux:1.88.0-latest \
  cargo clippy --workspace --all-targets -- -D warnings
```

### The Linux container is the manifest

The Linux build environment is committed to the repo at
`.github/docker/linux-ci/Dockerfile`. Reading that file answers "what
apt packages, what Rust toolchain, what Python interpreter, what uv
version does Linux CI assume?" There is no implicit assumption that
`ubuntu-24.04`'s default runner image happens to have any particular
package preinstalled.

If CI fails with a linker or system-library error, check the Dockerfile
*first*. If the symbol/library isn't installed there, that's the bug —
the Rust source is fine.

macOS has no equivalent. GitHub-hosted macOS runners do not support the
`container:` directive, so macOS jobs run on the bare `macos-15` /
`macos-13` runner; their environment is whatever GitHub's runner image
provides at the time of the run. Treat any macOS-only failure as
potentially environmental until proven otherwise. Do not try to "fix"
this asymmetry — it's a GitHub Actions platform constraint.

### CI ordering rule

Static analysis (rustfmt, clippy, rustdoc) MUST run **downstream** of
dependency audit and MSRV verification. A clippy failure with
unverified deps is not actionable: the failure could be a real lint or
a toolchain-vs-deps mismatch, and you cannot tell which without first
establishing that the toolchain satisfies the dependency tree. The
pipeline architecture in `.github/workflows/ci.yml` enforces this; do
not weaken it. Specifically: `cargo-msrv-verify` is in the `needs:`
graph of `clippy`, `rustfmt`, and `rustdoc` — not the other way around.

### The current cascade (Rust 1.88.0 / Linux, in encounter order)

Each layer documents file:line, the checklist category that explains
*why* it's failing, the command to reproduce, and the *minimum
fix-shape* (not the diff itself).

1. **`clippy::uninlined_format_args` at
   `crates/larql-models/src/loading/gguf.rs:118`** (and ~200 similar
   sites repo-wide). **Category: checklist #1** (toolchain-version-
   gated lint). Two routing options: bulk-inline via
   `cargo +1.88.0 clippy --fix`, or move the toolchain pin off 1.88.0.
   Cross-reference `Cargo.toml :: workspace.package.rust-version`
   (`"1.80"`) and the MSRV pin in `.github/workflows/ci.yml ::
   env.RUST_TOOLCHAIN` (`"1.88.0"`).
2. **35 compile errors** in `crates/larql-compute/src/metal/**`. Only
   surfaces *after* layer 1 is fixed (clippy doesn't reach this far
   today). **Category: checklist #2** (feature-flag activates
   macOS-cfg-gated optional dep). Cause: a 3-way name collision around
   `metal` — the cargo feature
   (`crates/larql-compute/Cargo.toml`), the optional external crate
   gated to `target_os = "macos"` (same file), and the local
   `crate::metal` module gated by the feature
   (`crates/larql-compute/src/lib.rs`). Force-activated by
   `crates/larql-cli/Cargo.toml`'s `default = ["metal"]`. Minimum fix:
   change that default to `[]`. Verification:
   `cargo check -p larql-compute` (no features) exits 0; with
   `--features metal` on Linux it produces the same 35 errors.
3. **`E0433` at `crates/larql-cli/src/commands/primary/bench_cmd.rs:170`**.
   Only surfaces after layer 2. **Category: checklist #2** (same root,
   second site). Reference `larql_compute::metal::MetalBackend::new()`
   is gated by a runtime `if metal {}` boolean, not
   `#[cfg(feature = "metal")]`. Minimum fix: wrap the metal arm in
   `#[cfg]`, mirroring
   `crates/larql-inference/src/layer_graph/hybrid.rs:63-65`.
4. **5 actual clippy lints** (× 2 for lib + tests = 9 messages). Real
   code-cleanup, only visible after layers 1-3 clear. **Category: none
   — actual source nits.** Sites:
   - `crates/larql-vindex/src/format/weights/write.rs:646`
     — `clippy::manual_repeat_n`
   - `crates/larql-inference/src/experts/mask.rs:131`
     — `clippy::if_same_then_else`
   - `crates/larql-inference/src/experts/mask.rs:145`
     — `clippy::ptr_arg`
   - `crates/larql-inference/src/experts/session.rs:205`
     — `clippy::single_char_add_str`
   - `crates/larql-inference/src/experts/mask.rs:278`
     — `clippy::useless_vec`
5. **OpenBLAS system-library — closed by the CI container.** The Linux
   CI container ships `libopenblas-dev` and `pkg-config`. Cargo
   declarations: `crates/larql-compute/Cargo.toml` and
   `crates/larql-inference/Cargo.toml`. If a future PR removes
   `libopenblas-dev` from the Dockerfile without auditing those Cargo
   features, the build-test legs will break with linker errors — which
   would be a Dockerfile bug, not a Rust source bug.

### Rustdoc lint suppressions

`crates/larql-vindex/src/lib.rs:1-2` carries
`#![allow(clippy::doc_overindented_list_items)]` and
`#![allow(clippy::doc_lazy_continuation)]`. These suppressions exist
because doc comments across multiple crates have indentation /
continuation issues that `cargo doc -D warnings` would otherwise reject.
The same caveat as layer 1 applies: some doc lints are toolchain-
version-gated, so reproduce against the pinned toolchain before
deciding whether a fix is in scope. Cleanup PR.

### cargo-deny

`cargo deny check advisories bans licenses sources` runs in S1 against
the committed `Cargo.lock`. If it fails, read the failing job log,
identify the specific `error[license-not-explicitly-allowed]`,
`error[advisory]`, or `error[ban]` annotation, then either pin the
offending dependency to a compatible version or — only if the audit
justifies it — extend `deny.toml :: [licenses] allow` per the procedure
in `audit/dependency-licenses.md` (D-1 through D-7 are the precedent).

### CI image bootstrap (one-time human step)

The Linux CI container referenced by `env.LINUX_CI_IMAGE` in
`.github/workflows/ci.yml` must exist on `ghcr.io` before that workflow
can run. To bootstrap (or to bump the image after editing the
Dockerfile):

1. From the branch containing the desired Dockerfile, manually trigger
   `.github/workflows/publish-ci-image.yml` (Actions tab → Run
   workflow).
2. Read the digest printed in the final step's notice.
3. Update `env.LINUX_CI_IMAGE` in `ci.yml` to
   `ghcr.io/metavacua/larql-ci-linux@sha256:<digest>`. The pin is by
   digest, not tag, so retags on ghcr.io cannot silently change CI
   inputs.
4. Commit + push the `ci.yml` change.

If a CI run fails with "manifest unknown" or "image not found" against
ghcr.io, that's the bootstrap step missing — not a Dockerfile or
ci.yml bug.

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
