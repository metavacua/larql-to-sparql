<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: Apache-2.0
-->

# Code Quality and Code Scanning Pipeline

This document is the operator's reference for the code-scanning and
code-quality workflow. It is deliberately separate from the
[compliance pipeline](compliance-pipeline.md), which owns the
deterministic candidate-validity gate (Foundational Axioms A1–A5) and
explicitly excludes security scanning from its scope:

> §Out-of-scope (compliance-pipeline.md): "If a future repository
> policy adopts \[security scanning or vulnerability triage\], it must
> live in a **separate** workflow file and must not call into
> `validate.yml`."

This file describes that separate workflow. The two pipelines run as
independent gates on every PR; neither calls into the other.

## Pipeline-to-artifact map

| Concern | Tool | Configuration | CI job |
|---|---|---|---|
| Rust formatting | `cargo fmt --check` | `rustfmt.toml` (default) | (out-of-workflow; see below) |
| Rust lint (gate + SARIF) | `cargo clippy -D warnings` + `clippy-sarif` | `clippy.toml` (default) | `quality.yml :: clippy` |
| Rust tests | `cargo test --workspace` | n/a | `quality.yml :: test` |
| Rust docs (broken links etc.) | `cargo doc -D warnings` | n/a | `quality.yml :: doc` |
| Rust examples buildability | `cargo build --workspace --examples` | n/a | `quality.yml :: examples` |
| Rust MSRV verification | `cargo-msrv verify` | workspace `Cargo.toml :: rust-version` | `quality.yml :: msrv` (informational) |
| Mutation testing | `cargo-mutants` | n/a | `quality.yml :: mutants` (informational; cron + PR diff) |
| Python (PyO3) bindings | `maturin develop` + `pytest` | `crates/larql-python/pyproject.toml` | `quality.yml :: python` (informational) |
| gRPC schema lint | `buf lint` | per-directory buf detection | `quality.yml :: proto-lint` (informational) |
| Rust dependency vulns | `cargo-audit` | `Cargo.lock`, RustSec advisory-db | `quality.yml :: audit` |
| Rust dep policy (license / bans / sources) | `cargo-deny` | `deny.toml` | `quality.yml :: deny` |
| Semantic code scanning | CodeQL | repository default-setup configuration | (out-of-workflow; see below) |
| Aggregate verdict | n/a | n/a | `quality.yml :: quality-gate` |

### Informational vs. gating

`msrv`, `mutants`, `python`, and `proto-lint` are introduced with
`continue-on-error: true` so they surface as visible signals without
blocking PRs while their baselines stabilize. The path to flipping
each to a hard gate is:

| Job | Pre-condition for gating |
|---|---|
| `msrv` | Workspace `rust-version` is updated to the value `cargo-msrv find` reports (the declared `1.80` is currently incorrect; transitive deps require ≥ 1.85). |
| `mutants` | Surviving-mutant baseline is triaged and either tests added or `.cargo-mutants.toml` carve-outs justified. |
| `python` | `crates/larql-python/tests/` has been run end-to-end on `ubuntu-24.04` to confirm no host-specific assumptions. |
| `proto-lint` | A repo-root or per-directory `buf.yaml` is committed that pins the lint configuration explicitly rather than relying on detection. |

`cargo fmt --check` is intentionally **not** a job in `quality.yml`.
The project already enforces formatting locally via the Makefile
(`make ci` and `make fmt-check`) and via the `cargo-fmt` hook in
`.pre-commit-config.yaml`. Duplicating the check in CI surfaces a
toolchain-version-sensitive baseline gap (rustfmt output drifts across
Rust releases) without adding security or correctness signal. If a
future policy decides to enforce fmt in CI, it should follow a
dedicated `style: cargo fmt --all` baseline-cleanup PR so the gate flip
is a no-op.

CodeQL is intentionally **not** wired into `quality.yml`. The repository
uses GitHub's default-setup CodeQL configured in repository settings,
which already analyses python, rust, actions, and c-cpp. Defining a
parallel CodeQL job here would emit duplicate SARIF for the same
`/language:*` categories and conflict with the default-setup uploads.
Changes to the CodeQL surface (queries, languages) live in the
repository's Code Scanning settings, not in this workflow.

## SARIF and the Security tab

Clippy is the only job in this workflow that uploads SARIF; CodeQL
SARIF is uploaded by the default-setup workflow described above.

| Category | Source | Owner |
|---|---|---|
| `clippy` | `clippy-sarif` over `cargo clippy --message-format=json` | `quality.yml :: clippy` |
| `/language:*` | `github/codeql-action/analyze` | repository default-setup CodeQL |

`audit` and `deny` are gating-only — they fail the workflow on any
finding rather than emitting SARIF. This is intentional: dependency-level
findings have a single deterministic remediation (bump or replace the
crate) and do not need per-finding triage in the Security tab.

## Cargo.lock policy

The repository's `.gitignore` excludes `Cargo.lock` (workspace policy).
The `audit` and `deny` jobs therefore generate a fresh lockfile at job
start with `cargo generate-lockfile` before scanning. This means each
run resolves against the index at run time, not against a frozen
historical snapshot — acceptable for advisory tracking, since the scan
target is "what would be installed today" rather than what was installed
on a specific past commit.

## Pinned versions and floating versions

The Rust toolchain and the SARIF emitter are pinned in
`.github/workflows/quality.yml :: env`. The advisory-feed scanners are
intentionally **not** pinned: they must keep pace with the formats of
their backing databases (a pinned `cargo-audit` rejected CVSS-4.0
entries because its bundled cvss-parser predated CVSS 4.0). Pinning a
scanner against an evolving feed defeats the scanner's purpose.

| Variable | Behaviour | Reason |
|---|---|---|
| `RUST_TOOLCHAIN` | **pinned** | Reproducible dep-tree resolution and clippy/test verdicts. |
| `CLIPPY_SARIF_VERSION` | **pinned** | Pure SARIF format emitter; not advisory-coupled. |
| `cargo-audit` | tracks latest (no pin in `tool:` spec) | Must accept the current advisory-db format. |
| `cargo-deny` | tracks latest (no pin in `tool:` spec) | Same reason. |

## Independence from `validate.yml`

The two workflows are independent gates:

```
pull_request ─┬─> validate.yml :: candidate-validity     (A1–A5)
              └─> quality.yml  :: quality-gate           (this file)
```

There is no `needs:` edge between them and they share no scripts. A red
quality gate does NOT make a branch an invalid candidate extension under
Axiom A5 — it is an orthogonal signal. Branch-protection policy decides
which gate(s) are required for merge; that is a repo-level decision and
not encoded in either workflow.

## Schedule

The advisory-feed scanners (`audit` and `deny`) run on a weekly cron
(Monday 06:00 UTC) so newly-disclosed RustSec advisories surface even
when the tree is dormant. `deny` is included because its `advisories`
section consults the same RustSec database that `audit` does.

`clippy`, `test`, and `quality-gate` are gated with
`if: github.event_name != 'schedule'`. They add no signal on a dormant
tree and would just burn minutes; they run only on `pull_request` to
`main`, `push` to `main`, and `workflow_dispatch`.

## Local mirror

The `make ci` target already mirrors `fmt`, `clippy`, and `test`. To run
the dependency-policy gates locally, substitute the pinned versions from
`.github/workflows/quality.yml :: env` for the placeholders below
(the env vars are workflow-local and are not exported to your shell):

```bash
# Replace ${...} with the literal values from quality.yml.
cargo install cargo-audit --locked --version "${CARGO_AUDIT_VERSION}"
cargo install cargo-deny  --locked --version "${CARGO_DENY_VERSION}"

# A lockfile is required (Cargo.lock is gitignored).
cargo generate-lockfile

cargo audit --deny warnings
cargo deny check advisories bans licenses sources
```

The `cargo deny` invocation deliberately omits `--all-features`: the
graph configuration in `deny.toml` already pins `all-features = false`,
and passing the flag here would evaluate a different dependency graph
than CI evaluates.

CodeQL has no practical local mirror; it runs in CI only and is owned
by the repository's default-setup configuration.

## Remediation contract

When a check fails, the workflow output is the only authoritative
description of what went wrong. The LLM agent must remediate by direct
consequence of the failure message; it must not interpret intent.

| Failing check | Deterministic remediation |
|---|---|
| `clippy` | Address each finding listed in the run log. The Security-tab SARIF view is informational; the gating step is `cargo clippy -- -D warnings`. |
| `test` | Address each test failure named in the run log. |
| `doc` | Address each rustdoc warning printed in the run log. Most are broken intra-doc links resolvable with `[Type]` / `[Type::method]` notation. |
| `examples` | Build the failing example locally (`cargo build -p <crate> --example <name>`) and update its source against the current public API. |
| `msrv` (informational) | Either raise `Cargo.toml :: rust-version` to whatever `cargo-msrv find` reports, or pin the dependency that forces the higher MSRV to an older compatible version. |
| `mutants` (informational) | Inspect the report artifact for surviving mutants; add tests for the corresponding code paths or document a justified carve-out in `.cargo-mutants.toml`. |
| `python` (informational) | Reproduce locally via `make python-test`. Common failures are version-skew in the PyO3 ABI or an out-of-date `crates/larql-python/uv.lock`. |
| `proto-lint` (informational) | Address each `buf lint` finding in the affected `.proto` file. Schema-level fixes are local; wire-compatibility (breaking-change) is a separate concern not yet wired here. |
| `audit` | Bump the affected crate to a fixed version (preferred), or replace it. Do not blanket-ignore advisories without a written rationale. |
| `deny` (licenses) | Either replace the offending dependency, or add the license to `deny.toml :: licenses.allow` if it is genuinely policy-compatible. |
| `deny` (bans) | Resolve the duplicate by aligning versions across the workspace, or add a justified `skip` entry. |
| `deny` (sources) | Either drop the offending source, or add it explicitly to `allow-git` / `allow-registry`. |
| `deny` (advisories) | Same as `audit`. |

CodeQL findings, when they appear, are owned by the repository's
default-setup configuration. Address each finding in the GitHub Security
tab; false positives may be dismissed via the UI with a written
justification. The `quality.yml` aggregate gate does not depend on
CodeQL.
