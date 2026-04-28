<!--
SPDX-FileCopyrightText: Contributors to the larql-to-sparql project
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
| **A2: Structured History** | `cog` (cocogitto) | `cog.toml` | `pre-commit` `cog-verify` (commit-msg) | `validate.yml :: commits` |
| **A3: Derived Documentation** | `git-cliff` + `scripts/check_changelog.sh` | `cliff.toml`, `CHANGELOG.md` | `pre-commit` `changelog-consistency` (pre-push) | `validate.yml :: changelog` |
| **A2 (SemVer)** | `scripts/version_preflight.sh` | `cog.toml` (bump rules) | n/a (informational) | `validate.yml :: version-preflight` |
| **A4: Verified Compliance** | aggregate gate | `validate.yml :: candidate-validity` | `make ci` | `validate.yml :: candidate-validity` |
| **A5: Candidate Validity Only** | repository policy | branch protection | n/a | `candidate-validity` is a non-merging signal |

## File inventory (compliance toolchain)

```
.github/workflows/validate.yml      # PR validation workflow (source of truth)
.pre-commit-config.yaml             # local hooks mirroring CI
REUSE.toml                          # bulk SPDX annotations
LICENSES/Apache-2.0.txt             # canonical license text per REUSE 3.x
cog.toml                            # Conventional Commits grammar + bump rules
cliff.toml                          # commits -> Keep a Changelog projection
CHANGELOG.md                        # Keep a Changelog 1.1.0, with [Unreleased]
scripts/check_changelog.sh          # deterministic [Unreleased] consistency
scripts/version_preflight.sh        # deterministic SemVer preflight
docs/specs/compliance-pipeline.md   # this file
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
| `commits` | Amend the commit so its header matches the Conventional Commits grammar declared in `cog.toml`. Force-push to the PR branch. |
| `changelog` | Run `git-cliff --config cliff.toml --unreleased --strip header --prepend CHANGELOG.md`, commit the result with `docs(changelog): regenerate unreleased`, and re-push. Do not hand-edit the `[Unreleased]` block. |
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

## Local developer workflow

```bash
# One-time setup (per clone).
pipx install pre-commit
pre-commit install --install-hooks
pre-commit install --hook-type commit-msg

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
