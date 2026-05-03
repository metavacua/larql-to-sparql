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

## Axiom-to-artifact map

| Axiom | Tool | Configuration | Local hook | CI job |
|---|---|---|---|---|
| **A1: Explicit Provenance** | `reuse` (FSFE REUSE 3.3) | `REUSE.toml`, `LICENSES/` | `pre-commit` `reuse` | `validate.yml :: provenance` |
| **A1 (first-party licences)** | `scripts/check_first_party_licenses.sh` | `LICENSE`, `NOTICE`, REUSE.toml-derived allow-list | n/a (CI-only) | `validate.yml :: first-party-licenses` |
| **A2: Structured History** | `cog` (cocogitto) | `cog.toml` | `pre-commit` `cog-verify` (commit-msg) | `validate.yml :: commits` |
| **A3: Derived Documentation** | `git-cliff` + `scripts/check_changelog.sh` | `cliff.toml`, `CHANGELOG.md` | `pre-commit` `changelog-consistency` (pre-push) | `validate.yml :: changelog` |
| **A2 (SemVer)** | `scripts/version_preflight.sh` | `cog.toml` (bump rules) | n/a (informational) | `validate.yml :: version-preflight` |
| **A4: Verified Compliance** | aggregate gate | `validate.yml :: candidate-validity` | `make ci` | `validate.yml :: candidate-validity` |
| **A5: Candidate Validity Only** | repository policy | branch protection | n/a | `candidate-validity` is a non-merging signal |

## First-party license requirements

The `first-party-licenses` job (formerly `apache-license`) reduces the
Apache License 2.0 §4 obligations to deterministic file-state predicates,
and additionally enforces that every license actually used in the tree
appears in `REUSE.toml` with its canonical text under `LICENSES/`. Apache-2.0
§4 obligations apply only to portions of the work licensed under
Apache-2.0; the script restricts the §4(b) modification-notice check
accordingly. The audited multi-license posture (see `audit/`) is the
authoritative source of truth.

| § | Requirement | Mechanical check |
|---|---|---|
| §4(a) | Distribute a copy of the License | `LICENSE` and `LICENSES/Apache-2.0.txt` exist and are non-empty |
| §4(b) | Modified files carry a prominent modification notice (Apache-2.0 files only) | For every Apache-2.0 file in `git diff --diff-filter=M $base..$head`, require an `SPDX-FileContributor:` line, a `Modifications:`/`Modified by:` comment, or an explicit path entry in `REUSE.toml` |
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
| Compliance toolchain (`REUSE.toml`, `cog.toml`, `cliff.toml`, `.github/workflows/`, `scripts/`, `docs/specs/*-pipeline.md`, `deny.toml`, `NOTICE`, `CONTRIBUTING.md`) | Copyright (C) 2026 Ian Douglas Lawrence Norman McLean | Apache-2.0 | toolchain PR series |
| Audit deliverables (`audit/**`) | Copyright (C) 2026 Ian Douglas Lawrence Norman McLean | CC-BY-SA-4.0 | licensing-audit pass |
| Future fork-authored code (new files) | the contributor | AGPL-3.0-or-later | per `CONTRIBUTING.md` |
| Future fork-authored docs (new files) | the contributor | CC-BY-SA-4.0 | per `CONTRIBUTING.md` |
| `LICENSE`, `LICENSES/Apache-2.0.txt`, `LICENSES/CC-BY-SA-4.0.txt`, `LICENSES/AGPL-3.0-or-later.txt`, `knowledge/LICENSE` | `NONE` (license-text boilerplate) | as named | upstream license stewards |

REUSE.toml is the authoritative manifest; per-file SPDX headers are
informational and may be aggregated or overridden by the manifest.

## Forward licensing posture

The fork's outbound posture for *new* contributions is dual-license:
**AGPL-3.0-or-later** for code, **CC-BY-SA-4.0** for documentation. Existing
files retain their inbound license unless deliberately relicensed in a
dedicated PR. This posture is consistent with — and motivated by — the
already-existing AGPL-3.0-only obligation introduced by the transitive
`evalexpr v12.x` dependency (see `audit/dependency-licenses.md` D-1 and
`audit/upstream-report.md` U-1). Inbound CLA is documented in
`CONTRIBUTING.md`.

## File inventory (compliance toolchain)

```
.github/workflows/validate.yml          # PR validation workflow (source of truth)
.github/workflows/quality.yml           # independent quality + scanning workflow
.pre-commit-config.yaml                 # local hooks mirroring CI
REUSE.toml                              # bulk SPDX annotations (single source of truth)
LICENSES/Apache-2.0.txt                 # canonical Apache-2.0 text per REUSE 3.x
LICENSES/AGPL-3.0-or-later.txt          # canonical AGPL text per REUSE 3.x
LICENSES/CC-BY-SA-4.0.txt               # canonical CC-BY-SA-4.0 text per REUSE 3.x
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
docs/specs/code-quality-pipeline.md     # quality.yml operator reference
```

## Determinism guarantees

Every validator and transformer in the pipeline is reproducible: same input
SHA, same tool versions, same verdict. To preserve this:

1. **Tool versions are pinned** in `.github/workflows/validate.yml` (`env:`)
   and in `.pre-commit-config.yaml` (`rev:`). Bumps go through a dedicated
   PR so the change is explicit in history.
2. **No probabilistic logic.** Scripts shell out only to tools listed above
   and standard POSIX utilities. No network calls except tool installation.
3. **No LLM-generated metadata.** The pipeline does not invoke any model;
   the LLM coding agent's role is upstream of the gate, not inside it.
4. **Idempotent transformers.** `git-cliff` against a fixed range produces
   byte-stable output. `scripts/check_changelog.sh` compares to that.

## Remediation contract

When a check fails, the workflow output is the only authoritative description
of what went wrong. The LLM agent must remediate by direct consequence of the
failure message; it must not interpret intent.

| Failing check | Deterministic remediation |
|---|---|
| `provenance` | Add the offending file's path to a matching `[[annotations]]` block in `REUSE.toml`, or insert a per-file SPDX header. Re-run `reuse lint`. |
| `first-party-licenses` (§4(a)) | Restore `LICENSE` and `LICENSES/Apache-2.0.txt` from upstream Apache-2.0 boilerplate. |
| `first-party-licenses` (§4(b)) | For each listed modified file, add an `SPDX-FileContributor:` line, a `Modified by:` comment, or an explicit `[[annotations]]` block in `REUSE.toml`. |
| `first-party-licenses` (§4(d)) | Restore `NOTICE` to a non-empty file containing project attribution lines. |
| `first-party-licenses` (REUSE 3.x text missing) | Add the canonical license text to `LICENSES/<id>.txt` (use `reuse download <id>` to fetch). |
| `first-party-licenses` (allow-list) | Add the new SPDX-id to `REUSE.toml` as an `[[annotations]]` block (manifest is the single source of truth) or change the file's `SPDX-License-Identifier` to one already in the manifest. |
| `commits` | Amend the commit so its header matches the Conventional Commits grammar declared in `cog.toml`. Force-push to the PR branch. |
| `changelog` | Run `git-cliff --config cliff.toml --unreleased --output CHANGELOG.md`, commit the result with `docs(changelog): regenerate unreleased`, and re-push. Do not hand-edit the `[Unreleased]` block. |
| `version-preflight` | This job is informational. A non-zero exit indicates an unparseable commit, which is also caught by `commits`; remediate there. |

## Out-of-scope (explicit non-goals)

Per Axiom A5 and the Scope section of the formal specification, the pipeline
**does not** perform or decide:

- merging, closing, or rebasing pull requests
- release tagging, GitHub Releases, or git tag creation
- crate publication to crates.io
- container image publication to any registry
- deployment to any environment
- artifact signing or attestation generation
- security scanning or vulnerability triage

If a future repository policy adopts any of the above, it must live in a
**separate** workflow file and must not call into `validate.yml`. The PR gate
is intentionally scoped to candidate-validity only.

The first such carrier is `.github/workflows/quality.yml`, which owns code
scanning and code quality (lint/format/tests, `cargo-audit`, `cargo-deny`,
CodeQL). It is documented in [code-quality-pipeline.md](code-quality-pipeline.md)
and runs as an independent gate; neither workflow calls into the other.

## Local developer workflow

```bash
# One-time setup (per clone).
pipx install pre-commit
pre-commit install --install-hooks
pre-commit install --hook-type commit-msg --hook-type pre-push

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

## CI workflow shape

```
pull_request -> [provenance] [commits]
                                   \
                                    +--> [changelog]
                                    +--> [version-preflight]
                                                       \
                                                        +--> [candidate-validity]
```

`candidate-validity` is the single required check on branch protection. It is
green iff every prior job is green; nothing more, nothing less.
