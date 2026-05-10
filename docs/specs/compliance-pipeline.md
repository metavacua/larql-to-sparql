<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: Apache-2.0
-->

# Compliance Pipeline Specification

This document is the operator's reference for the deterministic, rule-based
validation toolchain. It maps each Foundational Axiom of the project
compliance specification to the concrete artifact and tool that enforces it,
and it defines the contract that the LLM coding agent must satisfy when
remediating a failure.

The pipeline is implemented by `.github/workflows/ci.yml` (see that file
for the canonical job graph). This document is an operator's view; the
workflow file is authoritative.

## Pipeline architecture

```
S0   Meta & License        bare runner, OS-independent
                           (reuse, license headers, commits, changelog)
   ↓
S0.5 Environment Manifest  per OS; Linux containerized via
                           .github/docker/linux-ci/Dockerfile,
                           macOS native; uploads env-manifest-${os}
                           artifact
   ↓
S1   Dependency Audit      cargo-deny, lockfile-consistency, uv-lock
   ↓
S1.5 MSRV Verification     cargo-msrv-verify
                           (gates ALL static analysis; see "CI ordering
                            rule" below)
   ↓
S2   Static Analysis       rustfmt, clippy, rustdoc
                           (downstream of MSRV by design)
   ↓
S3   Build & Test Matrix   ubuntu-24.04 (containerized) ×
                           macos-15 × macos-13 + CodeQL on Linux +
                           Python bindings (maturin + pytest)
   ↓
S5   Post-Build Quality    msrv-discovery, mutants
                           (informational, non-blocking)
S6   Post-Build Validation distribution-notices
                           (informational, non-blocking)
   ↓
ci-passed (aggregate gate, single required check on branch protection)
```

### CI ordering rule

Static analysis (rustfmt, clippy, rustdoc) MUST run downstream of MSRV
verification. A clippy failure with unverified deps is not actionable:
the failure could be a real lint or a toolchain-vs-deps mismatch, and
the two are indistinguishable from the failure message alone. The
pipeline architecture in `ci.yml` enforces this; do not weaken it. See
also AGENTS.md "Cross-OS environment leaks" §"CI ordering rule".

### Linux build environment

The Linux build environment is a single committed manifest:
`.github/docker/linux-ci/Dockerfile`. Reading that file answers "what
apt packages, what Rust toolchain, what Python interpreter, what uv
version does Linux CI assume?" Every Linux job in `ci.yml` runs inside
this container (see the `container: ${{ env.LINUX_CI_IMAGE }}` job-level
directive); GitHub's runner image is used only as the host kernel.

macOS jobs run on bare GitHub-hosted runners (`macos-15`, `macos-13`).
GitHub-hosted macOS runners do not support the `container:` directive,
so the macOS build environment is whatever GitHub provides at run time.
The `env-establishment-macos` job records that environment as an
artifact for retrospective debugging.

## Axiom-to-artifact map

| Axiom | Tool | Configuration | Local hook | CI job |
|---|---|---|---|---|
| **A1: Explicit Provenance** | `reuse` (FSFE REUSE 3.3) | `REUSE.toml`, `LICENSES/` | `pre-commit reuse` | `ci.yml :: reuse-lint` + `reuse-toml-guard` |
| **A1 (first-party licences)** | `scripts/check_first_party_licenses.sh` | `LICENSE`, `NOTICE`, REUSE.toml-derived allow-list | n/a (CI-only) | `ci.yml :: first-party-licenses` |
| **A2: Structured History** | `cog` (cocogitto) | `cog.toml` | `pre-commit cog-verify` (commit-msg) | `ci.yml :: conventional-commits` |
| **A3: Derived Documentation** | `git-cliff` + `scripts/check_changelog.sh` | `cliff.toml`, `CHANGELOG.md` | `pre-commit changelog-consistency` (pre-push) | `ci.yml :: changelog-consistency` |
| **A4: Reproducible Build Environment** | docker base image + Dockerfile | `.github/docker/linux-ci/Dockerfile`, `rust-toolchain.toml` | `docker run ghcr.io/metavacua/larql-ci-linux:1.88.0-latest …` | `ci.yml :: env-establishment-linux` + `env-establishment-macos` |
| **A4 (Verified Compliance)** | aggregate gate | branch protection on `ci-passed` | `make ci` | `ci.yml :: ci-passed` |
| **A5: Candidate Validity Only** | repository policy | branch protection | n/a | `ci-passed` is a non-merging signal |

A4 was previously satisfied implicitly via job-level `rustup install`
calls; it is now satisfied explicitly via the committed Dockerfile.
Treat the Dockerfile as the manifest, not as build scaffolding.

## First-party license requirements

The `first-party-licenses` job reduces the Apache License 2.0 §4
obligations to deterministic file-state predicates, and additionally
enforces that every license actually used in the tree appears in
`REUSE.toml` with its canonical text under `LICENSES/`. Apache-2.0 §4
obligations apply only to portions of the work licensed under
Apache-2.0; the script restricts the §4(b) modification-notice check
accordingly. The audited multi-license posture (see `audit/`) is the
authoritative source of truth.

| § | Requirement | Mechanical check |
|---|---|---|
| §4(a) | Distribute a copy of the License | `LICENSE` and `LICENSES/Apache-2.0.txt` exist and are non-empty |
| §4(b) | Modified files carry a prominent modification notice | Satisfied at the manifest level: `REUSE.toml` `[[annotations]]` covers every tracked file with explicit copyright/license, and `reuse lint` (a `needs:` dependency of this job) verifies coverage. No per-file re-walk. |
| §4(c) | Retain copyright/license notices in source | Every file has `SPDX-FileCopyrightText` and `SPDX-License-Identifier` (delegated to `reuse lint`) |
| §4(d) | Propagate `NOTICE` to redistributions | `NOTICE` exists and is non-empty |
| (REUSE 3.x) | Every license identifier has canonical text | For every SPDX-id in `REUSE.toml`, `LICENSES/<id>.txt` exists and is non-empty |
| (allow-list) | License compatibility | Every `SPDX-License-Identifier:` value declared in source must also appear in `REUSE.toml` (manifest is the single source of truth) |

§§1–3 (definitions, copyright/patent grants), §5 (contribution license),
§6 (trademarks), §7 (warranty disclaimer), §8 (limitation of liability),
and §9 (additional liability) are legal effects rather than file-state
predicates and are not mechanically checkable; the deterministic core does
not attempt to verify them.

## Provenance assignment

| Path scope | Copyright | License | Origin |
|---|---|---|---|
| Pre-existing LARQL codebase on `main` | Copyright (C) 2026 Chris Hay | Apache-2.0 | <https://github.com/chrishayuk/larql> |
| Compliance toolchain (`REUSE.toml`, `cog.toml`, `cliff.toml`, `.github/`, `scripts/`, `docs/specs/*-pipeline.md`, `deny.toml`, `NOTICE`, `CONTRIBUTING.md`, `rust-toolchain.toml`) | Copyright (C) 2026 Ian Douglas Lawrence Norman McLean | Apache-2.0 | toolchain PR series |
| Audit deliverables (`audit/**`) | Copyright (C) 2026 Ian Douglas Lawrence Norman McLean | CC-BY-SA-4.0 | licensing-audit pass |
| Future fork-authored code (new files) | the contributor | AGPL-3.0-or-later | per `CONTRIBUTING.md` |
| Future fork-authored docs (new files) | the contributor | CC-BY-SA-4.0 | per `CONTRIBUTING.md` |
| `LICENSE`, `LICENSES/Apache-2.0.txt`, `LICENSES/CC-BY-SA-4.0.txt`, `LICENSES/AGPL-3.0-or-later.txt`, `knowledge/LICENSE` | `NONE` (license-text boilerplate) | as named | upstream license stewards |

REUSE.toml is the authoritative manifest; per-file SPDX headers are
informational and may be aggregated or overridden by the manifest.

## Forward licensing posture

The fork's outbound posture for *new* contributions is dual-license:
**AGPL-3.0-or-later** for code, **CC-BY-SA-4.0** for documentation.
Existing files retain their inbound license unless deliberately
relicensed in a dedicated PR. This posture is consistent with — and
motivated by — the already-existing AGPL-3.0-only obligation introduced
by the transitive `evalexpr v12.x` dependency (see
`audit/dependency-licenses.md` D-1 and `audit/upstream-report.md` U-1).
Inbound CLA is documented in `CONTRIBUTING.md`.

## File inventory (compliance toolchain)

```
.github/workflows/ci.yml                # unified PR validation pipeline
.github/workflows/publish-ci-image.yml  # manual: build & push CI base image
.github/docker/linux-ci/Dockerfile      # Linux CI environment manifest
rust-toolchain.toml                     # developer-side toolchain pin
.pre-commit-config.yaml                 # local hooks mirroring CI
REUSE.toml                              # bulk SPDX annotations
LICENSES/Apache-2.0.txt                 # canonical Apache-2.0 text
LICENSES/AGPL-3.0-or-later.txt          # canonical AGPL text
LICENSES/CC-BY-SA-4.0.txt               # canonical CC-BY-SA-4.0 text
NOTICE                                  # Apache §4(d) + AGPL transitive obligations
CONTRIBUTING.md                         # inbound dual-license CLA + workflow guidance
cog.toml                                # Conventional Commits grammar + bump rules
cliff.toml                              # commits -> Keep a Changelog projection
CHANGELOG.md                            # Keep a Changelog 1.1.0, with [Unreleased]
deny.toml                               # cargo-deny config (license/bans/sources)
scripts/check_first_party_licenses.sh   # REUSE-driven first-party license gate
scripts/check_changelog.sh              # deterministic [Unreleased] consistency
scripts/version_preflight.sh            # deterministic SemVer preflight
audit/                                  # licensing audit deliverables
docs/specs/compliance-pipeline.md       # this file
```

## Determinism guarantees

Every validator and transformer in the pipeline is reproducible: same
input SHA, same tool versions, same verdict. To preserve this:

1. **Tool versions are pinned** in `.github/workflows/ci.yml` (`env:`)
   and in `.pre-commit-config.yaml` (`rev:`). Bumps go through a
   dedicated PR so the change is explicit in history.
2. **The Linux build environment is pinned by digest.** `env.LINUX_CI_IMAGE`
   in `ci.yml` references the container image by `@sha256:<digest>`,
   not by tag. Retags on `ghcr.io` cannot silently change CI inputs.
3. **No probabilistic logic.** Scripts shell out only to tools listed
   above and standard POSIX utilities. No network calls except tool
   installation.
4. **No LLM-generated metadata.** The pipeline does not invoke any
   model; the LLM coding agent's role is upstream of the gate, not
   inside it.
5. **Idempotent transformers.** `git-cliff` against a fixed range
   produces byte-stable output. `scripts/check_changelog.sh` compares
   to that.

## Remediation contract

When a check fails, the workflow output is the only authoritative
description of what went wrong. The LLM agent must remediate by direct
consequence of the failure message; it must not interpret intent.

| Failing check | Deterministic remediation |
|---|---|
| `reuse-lint` | Add the offending file's path to a matching `[[annotations]]` block in `REUSE.toml`, or insert a per-file SPDX header. Re-run `reuse lint`. |
| `reuse-toml-guard` | A copyright line was stripped from `REUSE.toml`. Restore it. The script names the missing line. |
| `first-party-licenses` (§4(a)) | Restore `LICENSE` and `LICENSES/Apache-2.0.txt` from upstream Apache-2.0 boilerplate. |
| `first-party-licenses` (§4(b)) | Cannot fail directly: §4(b) is satisfied at the manifest level. If `reuse-lint` is green, §4(b) is green. |
| `first-party-licenses` (§4(d)) | Restore `NOTICE` to a non-empty file containing project attribution lines. |
| `first-party-licenses` (REUSE 3.x text missing) | Add the canonical license text to `LICENSES/<id>.txt` (use `reuse download <id>` to fetch). |
| `first-party-licenses` (allow-list) | Add the new SPDX-id to `REUSE.toml` as an `[[annotations]]` block, or change the file's `SPDX-License-Identifier` to one already in the manifest. |
| `conventional-commits` | Amend the commit so its header matches the Conventional Commits grammar declared in `cog.toml`. Force-push to the PR branch. |
| `changelog-consistency` | Run `git-cliff --config cliff.toml --unreleased --output CHANGELOG.md`, commit the result with `docs(changelog): regenerate unreleased`, and re-push. Do not hand-edit the `[Unreleased]` block. |
| `env-establishment-linux` | The CI container is unreachable. Either (a) `env.LINUX_CI_IMAGE` is the bootstrap placeholder — run `publish-ci-image.yml` and update the digest; or (b) `ghcr.io` is having an outage — retry; or (c) the Dockerfile build itself failed and the image was never published. |
| `cargo-deny` | Read the failing log; identify the specific `error[license-not-explicitly-allowed]`, `error[advisory]`, or `error[ban]` annotation; either pin the offending dependency to a compatible version or extend `deny.toml :: [licenses] allow` per `audit/dependency-licenses.md` D-1 through D-7. |
| `lockfile-consistency` | `Cargo.lock` is out of sync with `Cargo.toml`. Run `cargo metadata --locked` locally; if it succeeds, the diff is what to commit; if it fails, fix the underlying dep declaration. |
| `uv-lock-check` | `crates/larql-python/uv.lock` is out of sync with `pyproject.toml`. Run `uv lock` in that directory and commit. |
| `cargo-msrv-verify` | Either bump `Cargo.toml :: workspace.package.rust-version` to the version `cargo-msrv` reports as the actual floor, or pin the offending transitive to a version with a lower MSRV. |
| `rustfmt` | `cargo fmt --all` produced changes; commit them. |
| `clippy` | Run `cargo clippy --workspace --all-targets -- -D warnings` against the **pinned** toolchain (`rust-toolchain.toml` pin, or `docker run ghcr.io/metavacua/larql-ci-linux:1.88.0-latest`). Do **not** rely on a newer local toolchain — clippy's default-warn lint set drifts between versions. See AGENTS.md "Cross-OS environment leaks" for the current cascade. |
| `rustdoc` | Run `RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --exclude larql-python` against the pinned toolchain; fix each reported warning. |
| `build-test-matrix (ubuntu-24.04)` | Linker errors? Almost certainly a missing apt package — update `.github/docker/linux-ci/Dockerfile`, republish the image, update the digest. Compile errors? Real Rust source bug. |
| `build-test-matrix (macos-15 / macos-13)` | macOS-only failure: read the `env-establishment-macos` artifact for that runner before assuming it's a Rust source bug. |

## Out-of-scope (explicit non-goals)

Per Axiom A5 and the Scope section of the formal specification, the
pipeline `ci.yml` **does not** perform or decide:

- merging, closing, or rebasing pull requests
- release tagging, GitHub Releases, or git tag creation
- crate publication to crates.io
- deployment to any environment
- artifact signing or attestation generation

If a future repository policy adopts any of the above, it must live in
a **separate** workflow file. The PR gate `ci-passed` is intentionally
scoped to candidate-validity only.

The first such carrier is `.github/workflows/publish-ci-image.yml`,
which builds and publishes the Linux CI base image to `ghcr.io`. It
runs only on `workflow_dispatch` (never on PR), is gated behind
`packages: write` permission, and does not call into `ci.yml`. PRs
read the published image by digest pinned in `ci.yml :: env`; PRs
cannot publish.

## Local developer workflow

```bash
# One-time setup (per clone).
pipx install pre-commit
pre-commit install --install-hooks
pre-commit install --hook-type commit-msg --hook-type pre-push

# rust-toolchain.toml at the repo root pins Rust 1.88.0; rustup will
# install it automatically on the first cargo invocation. Verify with:
rustc --version    # should print 1.88.0

# For a fully faithful Linux reproduction (matches CI's apt packages):
docker run --rm -v "$PWD":/work -w /work \
  ghcr.io/metavacua/larql-ci-linux:1.88.0-latest \
  cargo clippy --workspace --all-targets -- -D warnings

# Per-change loop.
git checkout -b feat/short-description
# ... edit ...
git add -p
git commit -m "feat(scope): short imperative subject"
# pre-commit runs reuse + fmt + clippy automatically
# commit-msg hook runs cog-verify automatically

# Before pushing:
git-cliff --config cliff.toml --unreleased --strip header --prepend CHANGELOG.md
git add CHANGELOG.md
git commit -m "docs(changelog): regenerate unreleased"
git push -u origin "$(git branch --show-current)"
```
