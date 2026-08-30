# ADR-0026 — Tagged Release Binaries via CI (no crates.io publishing)

**Status:** Accepted 2026-07-24 — implemented same day
(`.github/workflows/release.yml`, `[profile.release-dist]` in the
workspace `Cargo.toml`). Crate *names* were separately claimed on
crates.io as empty placeholders — 12 on 2026-07-24, the remaining 6
after the rate-limit window reopened (see addendum below) — that is a
distinct, narrower action from the publishing this ADR still declines.
**Affects:** `.github/workflows/release.yml` (new), `crates/larql-cli`,
`crates/larql-server`, workspace `Cargo.toml` (`[profile.release-dist]`).
**Related:** DEC funnel v0.5 (`docs/dec-funnel.md`), which is the concrete
pain this would remove; ADR-0019 (backend factory / `BackendKind`, relevant
to what a release binary actually contains per platform).

---

## Context

larql has no distribution artifact today — no crates.io publish, no GitHub
Release binaries, nothing `cargo install`-able. The only way to get a
working `larql`/`larql-server` on a machine is `git clone` + `cargo build
--release` from source.

This was fine while development happened on one or two long-lived boxes.
The DEC funnel programme (`docs/dec-funnel.md`) breaks that assumption: it
runs the same serving code on a sequence of **ephemeral, single-use hosts**
— Colab notebooks (arm L), Vast x86 boxes (DEC-0.5, DEC-1A, DEC-2 through
DEC-7, DEC-CV). Every one of them currently starts from zero: install a
Rust toolchain if missing, `git clone`, then a cold `cargo build --release`
with no dependency cache. Concretely, from the DEC-0 arm L run this ADR was
written during: toolchain check → clone → **full cold release build**
(dozens of dependency crates including `wasmtime` transitively) before the
expert server can even start. That's 20–40 minutes of pure setup tax, paid
again on the next box, and the next, for the rest of the programme —
DEC-0.5, DEC-1A's netem host, DEC-2's N clients + server, DEC-2.5's router +
2 servers, DEC-3's metrology box, DEC-4/5/6/7's extraction and serving
hosts. Most of that fleet is architecturally identical (Linux x86_64, CPU
serving) and would be well served by one prebuilt binary, not N rebuilds.

Worse than the wall-clock cost: several of these hosts (Colab T4s in
particular) are GPU allocations, and the build itself is pure CPU
compilation — the GPU sits idle for the entire 20-40 minutes, burning
through a scarce/quota-limited session on work that needed no GPU at all.
A prebuilt binary turns that into near-zero idle GPU time before the
actual (also GPU-idle, since arm L is CPU-attention pre-G-ladder) serving
workload starts.

**Hard policy, not just an optimisation: GPU-provisioned hosts never
build from source.** Any DEC-stage dispatch to a GPU box (Colab, Vast GPU
tiers) must fetch a prebuilt release artifact and skip `cargo build`
entirely — if no release artifact exists yet for the commit/platform in
question, that is a blocker on the dispatch, not something to work around
by building on the box anyway. CPU-only boxes (metrology, extraction,
router hosts with no GPU line item) are unaffected and may still build
from source if no release exists.

## Decision

Add a `release.yml` GitHub Actions workflow, triggered on `v*` tags, that
cross-builds `larql` (the CLI) and `larql-server` for the platforms the
project's CI already validates — `.github/workflows/larql-cli.yml`'s
existing test matrix is `[ubuntu-latest, windows-latest, macos-14]` — and
attaches the binaries to a GitHub Release via `gh release upload` /
`softprops/action-gh-release`.

Implemented as `.github/workflows/release.yml`: a 3-way matrix
(`macos-14`/`aarch64-apple-darwin`, `ubuntu-latest`/`x86_64-unknown-linux-gnu`,
`windows-latest`/`x86_64-pc-windows-msvc`) with the same OpenBLAS/protoc
setup `larql-cli.yml` already has, building `-p larql-cli -p larql-server`
under a dedicated `release-dist` cargo profile (see below), packaging each
platform's binaries into a `.tar.gz`/`.zip`, and publishing them to a
GitHub Release via `softprops/action-gh-release` once every matrix leg
succeeds.

Each release gets one archive per platform containing `larql` +
`larql-server`. A DEC driver script (`dec0-loopback.sh`'s siblings) can
then do `curl -fsSL .../larql-x86_64-unknown-linux-gnu.tar.gz | tar xz`
instead of a full `cargo build`.

**Wired 2026-07-25, once `v0.1.0` existed to fetch from.**
`scripts/dec0-arm-l.sh` now resolves binaries in four ordered steps:
operator-supplied (`DEC0L_LARQL_BIN`/`DEC0L_SERVER_BIN`) → reuse an
already-populated `DEC0L_BIN_DIR` → fetch the release archive for the
detected platform (`DEC0L_LARQL_VERSION`, default `v0.1.0`) → fall back to
`cargo build`. That last step is **gated on the policy above**: if
`nvidia-smi` is present the script refuses and exits non-zero rather than
compiling on a GPU allocation, and says what to do instead (publish a
release for the platform, pass the binaries explicitly, or move to a
CPU-only host). `DEC0L_ALLOW_SOURCE_BUILD=1` overrides for the operator who
knowingly accepts the cost, and the log line then says the policy was
*overridden* rather than claiming it was satisfied. The old
`DEC0L_SKIP_BUILD` knob is gone — "already present" is now detected rather
than asserted.

The logic lives in `scripts/lib/larql-binaries.sh` and is shared, since every
remaining DEC stage provisions a fresh host. `scripts/dec0p5-x86.sh` uses it
too, with one deliberate exemption: DEC-0.5 still compiles the criterion
kernel bench (`cargo bench -p larql-compute`), because a bench target is not
a shipped binary and that kernel is precisely what the stage exists to
measure. Fetching the CLI and server still removes the bulk of the build from
the lease.

### `release-dist` cargo profile

```toml
# Distribution builds only — full strip, no line tables. Kept separate
# from [profile.release] so local profiling builds (samply/flamegraph,
# which need the release profile's line-tables-only debug info) are
# unaffected.
[profile.release-dist]
inherits = "release"
debug = false
strip = true
```

### Why binaries only, not crates.io

Publishing to crates.io means committing to semver and public API stability
across ~13 workspace crates that are still churning weekly (DEC funnel, the
G-ladder backend work, wire codec revisions). A release binary carries no
such commitment — it's a build artifact, versioned by tag, replaceable at
will. If/when individual crates (e.g. `larql-vindex-spec`, whose manifest
schema is already meant to be a stable on-disk contract per ADR-0007)
stabilize enough to be worth other projects depending on via Cargo, that's
a separate, much narrower decision to make crate-by-crate — not bundled
into this one.

### Why the existing CI matrix, not a new one

`larql-cli.yml` already builds and tests on all three platforms with the
correct per-OS feature flags (`--no-default-features` off macOS's Metal
default) and system deps (OpenBLAS, protoc). Reusing that matrix means a
release build exercises a path CI already proves green on every push,
rather than a parallel, less-tested configuration.

### Scope

**In:** `larql` (CLI) + `larql-server` (default build, no
`metal-experts` — README already documents that feature must NOT be in the
CLI's default build; a `metal-experts`-enabled `larql-server` variant is a
possible second artifact, not required for v1 of this ADR).
**Out:** crates.io publishing (see above); Docker images (the existing
`deploy/fly/` Dockerfile already solves that need for fly.io specifically);
Windows/Linux GPU (CUDA) builds — no CUDA backend exists yet (G-ladder is
G0-decided, unimplemented).

## Cost

| Change | Status |
|---|---|
| `release.yml` workflow (matrix build + package + upload) | Done — copied `larql-cli.yml`'s matrix/OpenBLAS/protoc setup |
| `[profile.release-dist]` (strip, no line tables) | Done |
| Update DEC driver scripts to prefer a release binary, falling back to build | Not done — needs a real tag to fetch from first |
| First tag + verify all 3 platform artifacts actually run | Not done — no `v*` tag pushed yet |

## Open questions

- **Tag cadence.** Every merge to main, or hand-picked stable points? DEC's
  own driver scripts pin an exact commit today (`workspec.code.ref`); a
  release tag would need to track commits the DEC programme actually wants
  to measure against, not just "latest."
- **Versioning scheme.** `v0.x.y` calendar-ish tags vs. semver tied to
  something meaningful (no public API to version against yet, since this
  ADR explicitly excludes crates.io).

The workflow exists but is untested end-to-end — it only runs on a `v*`
tag push or manual `workflow_dispatch`, neither of which has happened yet.
First real run will surface whatever the matrix sketch got wrong.

## Addendum: crate-name claiming (2026-07-24, same day)

Separately from the binaries decision above: the 17 workspace crate names
(`larql`, `larql-models`, `larql-compute`, `larql-compute-metal`,
`larql-core`, `larql-vindex`, `larql-vindex-spec`, `larql-inference`,
`larql-kv`, `larql-lql`, `larql-cli`, `larql-server`, `larql-router`,
`larql-router-protocol`, `larql-python`, `larql-boundary`,
`model-compute`) plus `larql-experts` (the nested expert workspace at
`crates/larql-experts/`, excluded from the root `[workspace] members` and
so missed by the first sweep — 18 names total) were confirmed available on
crates.io and claimed as **empty placeholder crates** — `version = "0.0.0"`,
a description pointing back to this repo, no functional code. This is
squatting-prevention, not the crates.io publishing this ADR declines: the
placeholders carry no API surface, so there is nothing to hold
semver-stable, and swapping a placeholder for a real `0.1.0` release later
is an ordinary version bump.

Scope note: only the top-level `larql-experts` name is held. The ~17
*sub*-crate names inside that workspace (`expert-interface`,
`arithmetic`, `date`, `unit`, …) are generic, are not namespaced to this
project, and were deliberately left unclaimed — holding them would be a
much wider land-grab than squatting-prevention warrants.

Mechanically: minimal `Cargo.toml` + `README.md` + doc-comment-only
`src/lib.rs` per name, `cargo publish --allow-dirty` from a scratch
directory (outside the workspace, so `--allow-dirty` just means "no git
repo here" rather than "ignoring real changes"). crates.io's new-crate
rate limit is a **burst of 5, refilling ~1 per 10 minutes** (a rolling
window, not a daily cap), and it shaped the rollout: the first session
landed 12 names and stopped mid-sweep at the burst wall, leaving
`larql-router`, `larql-router-protocol`, `larql-python`, `larql-boundary`
and `model-compute` unclaimed until a follow-up session — a gap this
addendum originally, and wrongly, recorded as complete. Verify the real
state against the registry (`GET https://crates.io/api/v1/crates/<name>`,
404 = unclaimed) rather than against this document.

Retry-parsing gotcha for anyone automating this: the 429 body gives its
retry time as an **HTTP-date** (`Fri, 24 Jul 2026 22:52:09 GMT`), not
RFC3339 — a scraper matching only `\d{4}-\d{2}-\d{2}T…Z` silently
misclassifies a plain rate-limit as a hard failure.

Irreversibility note for future reference: crates.io publishes can be
*yanked* (hidden from new dependents) but never deleted — the name stays
permanently associated with the publishing account's history either way.
That was accepted knowingly here given the placeholder content is
inherently disposable (nothing links to it, nothing depends on it).

**Update 2026-07-29:** `larql-factory` (the Vindex Factory driver crate,
born after this addendum was written) claimed identically — same
`0.0.0` placeholder shape, same scratch-directory mechanics, name
verified unclaimed first via the crates.io API (404, with a descriptive
`User-Agent` — the API's anti-scraping policy 403s a bare `curl`).
**19 names now held.** `larql-vindex-spec` was already among the
original 18; no second claim was needed there.
