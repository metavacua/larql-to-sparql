<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: Apache-2.0
-->

# Local CI/CD Testing Guide

This document describes how to reproduce the project's CI/CD gates on a
developer workstation before pushing to GitHub.

After the upstream-wins merge of `chrishayuk/larql @ main`, the CI/CD
surface is split between **upstream-owned** and **fork-owned** modules
that coexist without redefining each other.

## Boundary: upstream vs. fork

| Concern | Owner | Surface |
|---|---|---|
| Per-crate fmt + lint + test on ubuntu / windows / macos-14 | Upstream | `.github/workflows/larql-*.yml` (11 workflows), driven by upstream's 668-line Makefile (`larql-*-ci` family) |
| Benchmark regression on macos-14 | Upstream | `.github/workflows/bench-regress.yml`, `scripts/bench-regress.sh`, `make bench-check` |
| Nix builds / dev shell / containers | Upstream | `flake.nix`, `nix/*.nix` |
| Conventional Commits, Keep a Changelog, REUSE 3.3, SemVer surface | Fork | `.github/workflows/validate.yml`, `cliff.toml`, `cog.toml`, `REUSE.toml` |
| Workspace-wide cross-cutting quality (audit, deny, msrv, mutants, python, proto-lint) | Fork | `.github/workflows/quality.yml` |
| Android + ChromeOS smoke builds | Fork | `.github/workflows/extra-platforms.yml`, `scripts/ci/build-{android,chromeos}.sh` |
| Dependency-update bot (GitHub Actions only) | Fork | `.github/dependabot.yml` |

Upstream's `make ci` is unchanged (`fmt-check lint test-full`). The fork
adds **separate, dedicated Makefile modules** — they do not redefine
upstream targets.

## Fork-only Makefile modules

```bash
make compliance      # REUSE + Conventional Commits + changelog + first-party license + semver preflight
make quality-fork    # cargo clippy --workspace -- -D warnings + cargo deny check + cargo audit
make platform-test   # Android + ChromeOS smoke (delegates to scripts/ci/build-*.sh)
make ci-fork         # ci (upstream) + compliance + quality-fork  -- run this before pushing a PR
```

Per-target details:

```bash
make platform-test-android     # cross-compile for aarch64-linux-android + armv7-linux-androideabi
make platform-test-chromeos    # Crostini (x86_64-unknown-linux-gnu) build + test + Python bindings
```

Linux, macOS, and Windows are covered by upstream's per-crate workflows
and are not duplicated here.

## Upstream Makefile entry points (selected)

Run upstream's per-crate CI locally:

```bash
make larql-core-ci             # fmt-check + lint + test + feature-test + bench-test + examples
make larql-vindex-ci           # same shape per crate; see Makefile for the full list
make ci                        # workspace-wide: fmt-check + lint + test-full
make test-fast                 # lib/bin tests only (fast path)
make test-models               # model-backed ignored tests (requires real vindexes)
```

See the `Makefile` for the full set of upstream targets.

## What `ci-fork` checks

`ci-fork` is the canonical fork-level pre-flight. It chains:

1. **`ci` (upstream)** — `fmt-check lint test-full` across the workspace.
2. **`compliance` (fork)** — `reuse lint`, `cog check`, `check_changelog.sh`,
   `check_first_party_licenses.sh`, `version_preflight.sh`.
3. **`quality-fork` (fork)** — workspace clippy (`-D warnings`),
   `cargo deny check`, `cargo audit`.

Failures in any step halt the chain. The compliance step uses
`cog check origin/main..HEAD || true`, so unrecognised upstream commit
forms do not block the local run (the same check is enforced strictly in
`.github/workflows/validate.yml` against the PR range).

## Prerequisites

All checks share a pinned Rust toolchain.

```bash
rustup toolchain install 1.92.0
rustup default 1.92.0
```

Additional tooling required by `compliance` and `quality-fork`:

```bash
pip install reuse                                  # REUSE 3.3
cargo install cargo-deny cargo-audit cargo-msrv    # quality scanners
cargo install --locked cocogitto                   # cog (Conventional Commits)
cargo install git-cliff                            # changelog regen
```

Android cross-compilation additionally requires the Android NDK (`r27`
or newer); see `scripts/ci/build-android.sh` for the full setup the CI
workflow performs.

## Regenerating the changelog

```bash
git-cliff --unreleased --strip header > /tmp/unreleased.md
# splice /tmp/unreleased.md into CHANGELOG.md under [Unreleased]
```

`check_changelog.sh` compares the committed `[Unreleased]` block against
the deterministic projection of the PR's commits under `cliff.toml`.

## Related specifications

- `crates/larql-vindex/docs/compliance-pipeline.md` — Foundational Axioms
  A1–A5, validate.yml mapping
- `crates/larql-vindex/docs/code-quality-pipeline.md` — quality.yml
  job inventory and code-scanning policy
- `CONTRIBUTING.md` — inbound license policy and pre-flight steps
- `REUSE.toml` — provenance manifest (centralized; no per-file SPDX
  stamping for fork-authored files beyond the existing 2-line header)
