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
| Rust formatting | `cargo fmt --check` | `rustfmt.toml` (default) | `quality.yml :: fmt` |
| Rust lint (gate + SARIF) | `cargo clippy -D warnings` + `clippy-sarif` | `clippy.toml` (default) | `quality.yml :: clippy` |
| Rust tests | `cargo test --workspace` | n/a | `quality.yml :: test` |
| Rust dependency vulns | `cargo-audit` | `Cargo.lock`, RustSec advisory-db | `quality.yml :: audit` |
| Rust dep policy (license / bans / sources) | `cargo-deny` | `deny.toml` | `quality.yml :: deny` |
| Semantic code scanning | CodeQL | `security-and-quality` queries | `quality.yml :: codeql` |
| Aggregate verdict | n/a | n/a | `quality.yml :: quality-gate` |

## SARIF and the Security tab

Three jobs upload SARIF to GitHub Code Scanning under distinct categories
so that findings can be triaged and dismissed independently:

| Category | Source |
|---|---|
| `clippy` | `clippy-sarif` over `cargo clippy --message-format=json` |
| `/language:python` | `github/codeql-action/analyze` (Python) |

`audit` and `deny` are gating-only — they fail the workflow on any
finding rather than emitting SARIF. This is intentional: dependency-level
findings have a single deterministic remediation (bump or replace the
crate) and do not need per-finding triage in the Security tab.

## Pinned tool versions

All tool versions are pinned in `.github/workflows/quality.yml :: env`.
Bumping any of them is a deliberate change to the scanning surface and
should be done in a dedicated PR so the change is explicit in history.

| Variable | Purpose |
|---|---|
| `RUST_TOOLCHAIN` | Pinned rustc toolchain. Mirrors the value used by `validate.yml`. |
| `CARGO_AUDIT_VERSION` | `cargo-audit` release pinned by `taiki-e/install-action`. |
| `CARGO_DENY_VERSION` | `cargo-deny` release pinned by `taiki-e/install-action`. |
| `CLIPPY_SARIF_VERSION` | `clippy-sarif`/`sarif-fmt` release pinned by `taiki-e/install-action`. |

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

The `audit` job runs on a weekly cron (Monday 06:00 UTC) so newly-disclosed
RustSec advisories surface even when the tree is dormant. The other jobs
run only on `pull_request` to `main`, `push` to `main`, and
`workflow_dispatch`.

## Local mirror

The `make ci` target already mirrors `fmt`, `clippy`, and `test`. To run
the dependency-policy gates locally:

```bash
cargo install cargo-audit --locked --version "${CARGO_AUDIT_VERSION}"
cargo install cargo-deny  --locked --version "${CARGO_DENY_VERSION}"

cargo audit --deny warnings
cargo deny --all-features check advisories bans licenses sources
```

CodeQL has no practical local mirror; it runs in CI only.

## Remediation contract

When a check fails, the workflow output is the only authoritative
description of what went wrong. The LLM agent must remediate by direct
consequence of the failure message; it must not interpret intent.

| Failing check | Deterministic remediation |
|---|---|
| `fmt` | Run `cargo fmt --all`. Commit. Re-push. |
| `clippy` | Address each finding listed in the run log. The Security-tab SARIF view is informational; the gating step is `cargo clippy -- -D warnings`. |
| `test` | Address each test failure named in the run log. |
| `audit` | Bump the affected crate to a fixed version (preferred), or replace it. Do not blanket-ignore advisories without a written rationale. |
| `deny` (licenses) | Either replace the offending dependency, or add the license to `deny.toml :: licenses.allow` if it is genuinely policy-compatible. |
| `deny` (bans) | Resolve the duplicate by aligning versions across the workspace, or add a justified `skip` entry. |
| `deny` (sources) | Either drop the offending source, or add it explicitly to `allow-git` / `allow-registry`. |
| `deny` (advisories) | Same as `audit`. |
| `codeql` | Address each finding in the GitHub Security tab. False positives may be dismissed via the UI with a written justification. |
