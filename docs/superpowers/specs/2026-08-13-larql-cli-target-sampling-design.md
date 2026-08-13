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
migration/refactoring work will act on. The workflow does attempt `clippy
--fix` per target (see "Clippy --fix and cross-job accumulation" below), but
never commits anything — every job's real pass/fail (before and after any fix
attempt) stays visible.

This branch is independent of `gating/larql-cli-wasm-and-safe` — no code or
CI is reused from it.

## Experimentation methodology

This workflow is not written once and pushed whole. It's built up as a
series of standalone, throwaway workflow files (matching the `experiment/*`
branch convention already used elsewhere in this environment), in two
overlapping ways:

- **Incremental build-up**: each job first as its own standalone
  single-job workflow (validates toolchain setup for that target in
  isolation), then small sub-chains (e.g. `wasm32v1-none →
  wasm32-unknown-unknown`, to validate the artifact hand-off mechanism
  itself), before assembling the full graph. Keeps the blast radius of any
  one failure small and legible.
- **Technical-choice variants**: wherever there's a genuinely undecided
  implementation detail (e.g. which action installs `wasmtime`), two or
  more standalone variants are run side by side on real CI and compared;
  the winner carries into the consolidated workflow.

Both of these exist specifically to test GitHub Actions itself, not just
`larql-cli`. Any uncertainty about Actions syntax or behavior — artifact
upload/download semantics between jobs, whether `git apply` behaves
consistently against a fresh Actions checkout, exact `clippy --fix` flags
needed against an already-dirty tree, `needs.<job>.result` edge cases, etc.
— becomes an explicit test case in one of these standalone workflows.
Documentation is not trusted as the source of truth here: this design
process already hit a case (`continue-on-error` at the job level) where
GitHub's own docs page was incomplete and a web search summary
self-contradicted before a specific secondary source resolved it. Where
docs are absent or contradictory, the standalone experiment workflow's
actual observed behavior on a real hosted runner is what's authoritative.

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

1. If this job has a real upstream in the graph (i.e. every job except
   `wasm32v1-none`): download the upstream job's cumulative diff artifact
   and `git apply` it to a fresh checkout.
2. `cargo clippy --target <T> -p larql-cli`
3. If clippy reports fixable findings: `cargo clippy --fix --target <T> -p
   larql-cli` (partial fixes are fine — a `--fix` pass is not expected to
   resolve everything in one go).
4. Upload the current full cumulative diff (original source → now,
   including whatever was inherited in step 1 plus this job's own fix) as
   this job's artifact, so any job downstream of it only ever applies one
   patch file, never a stack of them.
5. `cargo build --target <T> -p larql-cli`
6. `cargo test --target <T> -p larql-cli` (skipped for
   `wasm32-unknown-emscripten` — see below)

Steps 2, 3, 5, and 6 each have **step-level `continue-on-error: true`**, so
a clippy or fix failure does not prevent the job from still attempting
build and test on that same target — all signals are wanted per target
regardless of whether an earlier step failed. Build and test run against
the patched tree (post-step-3), not the original source, so the workflow
also surfaces whether a target's own fixes actually resolve its build/test
failures.

## Clippy --fix and cross-job accumulation

No job ever commits anything — not to the working branch, not anywhere.
All fix propagation is via GitHub Actions artifacts scoped to a single
workflow run, following the real `needs:` edges only:

- `wasm32v1-none` has no upstream fix to inherit; it starts clean from the
  checked-out source.
- `wasm32-unknown-unknown` inherits `wasm32v1-none`'s cumulative diff.
- `wasm32-wasip1`, `wasm32-wasip2`, `wasm32-wasip1-threads`, and
  `wasm32-unknown-emscripten` each independently inherit
  `wasm32-unknown-unknown`'s cumulative diff and extend it with their own
  target-specific fixes. They do **not** see each other's fixes — they are
  genuinely independent siblings in the graph, not a false sequence, so
  their fix output isn't reconciled with each other by the workflow.
- `kani` is untouched by any of this — it stays on its separate
  native-only branch off `fmt-check`, with no artifact exchange with the
  wasm chain.

Expected outcome: as fixes accumulate down the `wasm32v1-none →
wasm32-unknown-unknown → {sibling}` chain, later jobs' clippy passes may
surface issues that only become visible once earlier fixes are applied —
that's useful signal, not something to prevent. Over however many
standalone experiment runs this takes, the accumulated diffs are reviewed
and applied to source by hand — the workflow's job is to produce and
surface the diffs, not to land them.

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
- **No commits from CI, ever.** Not to the working branch, not to a
  side branch, not via a bot. Fix output is a transient workflow-run
  artifact only; landing it in source is a manual, later step.
- **No forced sequencing between genuinely independent jobs.** The
  `wasip1`/`wasip2`/`wasip1-threads`/`emscripten` fan-out stays parallel
  because that's the real dependency graph — it is not restructured into
  a chain merely to make fix accumulation easier.

## Validation approach

This workflow is validated exclusively by running on GitHub-hosted runners
via a real PR/push against `metavacua/larql-to-sparql` — not by simulating
or reproducing the build/test matrix locally. Local commands (`rustc --print
target-list`, `rustup target add ... `, checking for `emcc`/`wasmtime`) were
used only to inform toolchain-requirement decisions in this design, not to
pre-validate whether `larql-cli` itself builds or tests successfully on any
target.

This extends to GitHub Actions mechanics themselves, not just
`larql-cli`'s code: per "Experimentation methodology" above, uncertainty
about Actions syntax/behavior is resolved by observing a standalone
experiment workflow's actual run, not by further documentation research.
