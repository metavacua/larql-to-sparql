# larql-cli target-sampling CI workflow — design

Date: 2026-08-13
Branch: `feat/larql-cli-target-sampling` (off `metavacua/larql-to-sparql`, based on
`chrishayuk/larql` main at `60b7fda9`)

## Purpose

Add a GitHub Actions workflow that builds and tests `larql-cli` across a
sampling of Rust compilation targets, to surface where the crate currently
breaks outside its native target. This is a deliberate first pass on raw
upstream code (no target-conditional dependency splitting exists yet on this
branch, unlike the separate `gating/larql-cli-wasm-and-safe` effort). Failures
are expected and are the point: they are the analysis data that later
migration/refactoring work will act on. This workflow does not attempt to fix
anything — it only needs to run every planned target/job to completion and
leave each job's real pass/fail visible.

This branch is independent of `gating/larql-cli-wasm-and-safe` — no code or
CI is reused from it.

## Target graph

```
fmt-check (native, blocking gate)
  ├─→ wasm32v1-none job:              [clippy --target, build --target, test/wasmtime]
  │     └─→ wasm32-unknown-unknown job: [clippy --target, build --target, test/wasmtime]
  │           ├─→ wasm32-wasip1 job:             [clippy, build, test/wasmtime]
  │           ├─→ wasm32-wasip2 job:             [clippy, build, test/wasmtime]
  │           ├─→ wasm32-wasip1-threads job:     [clippy, build, test/wasmtime]
  │           └─→ wasm32-unknown-emscripten job: [clippy, build]   (build-only)
  │
  └─→ kani job (native target only): [kani proof harnesses on larql-cli + its
                                       dependency crates]
```

`wasm32v1-none` is the primary/root target; `wasm32-unknown-unknown` is gated
on it; the WASI targets and Emscripten are gated on `wasm32-unknown-unknown`,
matching the dependency order given for this task. `kani` is a parallel
branch off the same `fmt-check` gate — it verifies the native build and is
not part of the wasm chain.

### Targets explicitly excluded

- **`wasm32-wali-linux-musl`** — the only wasm-family musl target that
  exists (confirmed via `rustc --print target-list`; there is no plural
  "musl targets" family under wasm32). It is Tier 3 with no prebuilt `std`
  (confirmed via a live `rustup target add` attempt, which fails outright on
  stable). Omitted from this workflow.
- **Native `x86_64/aarch64-unknown-linux-musl`** — a different, unrelated
  target family (static-linking portability, not wasm portability). Not in
  scope for this task.

## Gating: fmt vs. clippy

- `fmt-check` runs once, upfront, natively, and blocks everything else.
  Formatting is target-independent (purely syntactic), so one native run is
  sufficient, and a failure here is an ordinary code-quality issue, not
  cross-target analysis signal.
- `clippy` is **target-specific** — lints depend on which `cfg(target_arch =
  ...)` branches are active — so it is not a single upfront job. It runs as
  the first step inside each target job (`cargo clippy --target <T> -p
  larql-cli`), once per target.

## Per-target job steps

Each target job (`wasm32v1-none`, `wasm32-unknown-unknown`, `wasm32-wasip1`,
`wasm32-wasip2`, `wasm32-wasip1-threads`, `wasm32-unknown-emscripten`) runs,
in order:

1. `cargo clippy --target <T> -p larql-cli`
2. `cargo build --target <T> -p larql-cli`
3. `cargo test --target <T> -p larql-cli` (skipped for
   `wasm32-unknown-emscripten` — see below)

Each step has **step-level `continue-on-error: true`**, so a clippy failure
does not prevent the job from still attempting build and test on that same
target — all three signals are wanted per target regardless of whether an
earlier step failed.

### Test execution via wasmtime

For `wasm32v1-none`, `wasm32-unknown-unknown`, `wasm32-wasip1`,
`wasm32-wasip2`, and `wasm32-wasip1-threads`, `cargo test` cross-compiles the
test binaries and executes them under `wasmtime` as the configured test
runner (via `.cargo/config.toml` `runner` entries per target), so real
unit/integration tests run rather than a bare compile check.

`wasm32-unknown-emscripten` is **build-only** — Emscripten's compiled output
expects a JS host environment (glue code) and is not expected to run
directly under wasmtime, so no test-execution step is attempted for it.
Whatever happens when `cargo build` is attempted (including plain link
failures if the toolchain setup is incomplete) is itself part of the
analysis data.

## Kani job

Runs `kani` (formal verification / proof harnesses via CBMC) against
`larql-cli` and its dependency crates, natively only — Kani does not support
cross-compilation to wasm/wasi targets. Its value is as an orthogonal,
low-noise correctness signal: cross-referencing its result against a given
wasm-target job's result (done by a human reading the Actions run page, not
computed by the workflow — see "Explicitly not doing" below) distinguishes
"logic bug, also fails natively" from "portability-only breakage, native
logic is fine."

## Failure handling / job chaining

Every job from `wasm32v1-none` onward (including `kani`) sets
**job-level `continue-on-error: true`**. Verified against GitHub's
documented behavior: with job-level `continue-on-error: true`, a failing
job still reports `needs.<job_id>.result == "success"` to jobs that depend
on it, so downstream jobs with a plain `needs: [...]` (no extra `if:`
condition) run automatically even when the upstream job failed. The failed
job's own status still shows accurately (red/failed) in the Actions run UI
for analysis purposes — only the `needs:` gating is affected, not the
job's own reported outcome.

This preserves the required ordering (`wasm32v1-none` must run before
`wasm32-unknown-unknown`, which must run before the WASI/Emscripten
fan-out) without letting expected breakage at any stage skip or block later
stages.

## Toolchain / tooling setup (GitHub-hosted runners)

Runners are ephemeral fresh VMs per run — nothing installed on this dev
machine carries over. Every job installs what it needs explicitly:

| Job | Installs |
|---|---|
| `fmt-check` | Rust stable + `rustfmt` component |
| every target job | Rust stable + `clippy` component + `rustup target add <T>` |
| `wasm32v1-none`, `wasm32-unknown-unknown`, `wasip1`, `wasip2`, `wasip1-threads` | `wasmtime` binary |
| `wasm32-unknown-emscripten` | emsdk / `emcc` |
| `kani` | `kani-verifier` + its CBMC backend |

No nightly toolchain is required anywhere (only need was for the now-excluded
`wasm32-wali-linux-musl`). Exact action names/versions (e.g. for installing
Rust toolchains, wasmtime, emsdk, Kani) will be verified against current
sources at implementation time rather than asserted from memory here.

## Explicitly not doing

- **No caching** (e.g. `Swatinem/rust-cache`) in this first version. Runner
  minutes are not a constraint (public repo), and there's no data yet on
  what's actually slow. Revisit only if iteration time becomes a real
  problem once this is running.
- **No summary/report job.** Every job's pass/fail is already visible
  natively in the GitHub Actions run's checks list — nothing about that
  needs a dedicated aggregation job. The Kani-vs-target differential
  read is a human glancing at two rows on the same run page, not a
  computed artifact. Adding a job to automate that would be complexity
  without a demonstrated need.
- **No `wasm32-wali-linux-musl` or native musl targets** (see "Targets
  explicitly excluded" above).

## Validation approach

This workflow is validated exclusively by running on GitHub-hosted runners
via a real PR/push against `metavacua/larql-to-sparql` — not by simulating
or reproducing the build/test matrix locally. Local commands (`rustc --print
target-list`, `rustup target add ... `, checking for `emcc`/`wasmtime`) were
used only to inform toolchain-requirement decisions in this design, not to
pre-validate whether `larql-cli` itself builds or tests successfully on any
target.
