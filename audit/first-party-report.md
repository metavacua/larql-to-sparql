<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: CC-BY-SA-4.0
-->

# First-party licensing audit

Methodology and findings for the first-party portion of the
`metavacua/larql-to-sparql` working tree, evaluated against REUSE 3.3 via the
`reuse` 6.2.0 toolchain. This audit is read-only with respect to source code:
it produces evidence and recommends annotations; it does not change behaviour.

## Method

1. Ran `reuse lint` over the tracked tree (excluding `target/` and `.git/`).
2. Walked the tree for license-bearing files (`LICENSE*`, `COPYING*`,
   `NOTICE*`, `COPYRIGHT*`).
3. Inventoried every `Cargo.toml` `license = ` field across the workspace and
   every `pyproject.toml` `license = ` field across the Python sub-projects.
4. Surveyed the SPDX identifiers actually present in tracked source files.
5. Reconciled findings against the blanket assignments in `REUSE.toml`.

## Coverage

| Metric | Value | Source |
|---|---|---|
| Files under version control | 1102 | `reuse lint :: SUMMARY` |
| Files with copyright information | 1102 / 1102 | `reuse lint` |
| Files with license information | 1102 / 1102 | `reuse lint` |
| Distinct first-party SPDX identifiers in use | 1 (`Apache-2.0`) | `reuse lint :: Used licenses` |
| Bad / deprecated / missing licenses | 0 / 0 / 0 | `reuse lint :: SUMMARY` |
| Invalid SPDX expressions detected | 4 (all in one file, all false positives) | `reuse lint :: INVALID SPDX LICENSE EXPRESSIONS` |

The blanket annotation in `REUSE.toml:25-29` (path `**`, `Apache-2.0`,
`Copyright (C) 2026 Chris Hay`) plus the override block at `REUSE.toml:35-52`
(compliance-toolchain paths, `Apache-2.0`, `Copyright (C) 2026 Ian Douglas
Lawrence Norman McLean`) is sufficient to give every tracked file a copyright
holder and a license. No file is uncovered.

## Findings

### F-1 — `scripts/check_apache_license.sh` triggers four false-positive SPDX-expression errors

`reuse lint` reports four "INVALID SPDX LICENSE EXPRESSIONS" — all four are in
`scripts/check_apache_license.sh` itself. The script greps for the literal
string `SPDX-License-Identifier:` in source files, and `reuse` parses those
shell-string occurrences as SPDX declarations whose value is the next token in
the script (a regex fragment, a backslash, etc.).

Lines that contain the literal `SPDX-License-Identifier:` and trip the parser:

- `grep -RIn 'SPDX-License-Identifier:' \` — the trailing backslash is read as the SPDX expression.
- `grep -vE "SPDX-License-Identifier:[[:space:]]*(${allowed_re})([[:space:]]|$)" \` — `[[:space:]]*(${allowed_re})([[:space:]]|$)" \` is read as the SPDX expression.
- `declaration_re='^[^:]+:[0-9]+:[[:space:]]*([#;*<!/-]|//|<!--)?[[:space:]]*SPDX-License-Identifier:'` — the empty trailing fragment is read as the SPDX expression.
- A documentation comment in the script's preamble whose adjacent text is `are whitespace and standard`.

This is a tooling artefact, not a real provenance defect. The remediation is
to bracket the affected lines with `REUSE-IgnoreStart` / `REUSE-IgnoreEnd`
comments. Phase B (which rewrites this script as `scripts/check_first_party_licenses.sh`)
is the natural place to apply the bracketing.

### F-2 — Sub-project `knowledge/` ships its own duplicate Apache-2.0 LICENSE file

`knowledge/` is a self-contained Python sub-project (`knowledge/pyproject.toml`
declares `name = "larql-knowledge"`, `license = {text = "Apache-2.0"}`,
`version = "0.1.0"`). It has its own `knowledge/LICENSE` file containing
verbatim Apache-2.0 boilerplate.

The `README.md` describes it as the "Knowledge pipeline for LARQL" and
references upstream `chrishayuk/chuk-larql-rs` as the consuming engine —
indicating the sub-project may have separate provenance from the rest of the
workspace, even though the file-level license declaration is the same.

Recommendations:

- Add an explicit `REUSE.toml` annotation pinning `knowledge/LICENSE` to
  `SPDX-FileCopyrightText = "NONE"` (license-text boilerplate, parallel to
  the existing entry for the top-level `LICENSE`).
- Optionally add a sub-project annotation block for `knowledge/**` if upstream
  confirms `knowledge/` was authored separately and should carry a different
  copyright line. Until upstream confirms, the blanket Chris Hay attribution
  is the conservative default.

### F-3 — Workspace `Cargo.toml` declares `license = "Apache-2.0"` and 33 of 35 crate manifests inherit via `license.workspace = true`

Every crate in the Cargo workspace either inherits the workspace
`license = "Apache-2.0"` or declares it directly. The two crates that opt out
of workspace inheritance both declare `license = "Apache-2.0"` directly:

- `crates/larql-experts/Cargo.toml`
- top-level `Cargo.toml` (the workspace root itself)

No crate declares a non-Apache-2.0 license. This is consistent with the
`REUSE.toml` blanket assignment.

### F-4 — Single SPDX identifier in use across all tracked files

The only `SPDX-License-Identifier` value appearing in any tracked file is
`Apache-2.0`. There are no source files declaring MIT, BSD, dual MIT/Apache,
GPL, AGPL, or any non-Apache identifier. The first-party tree is, by every
mechanical signal available, uniformly Apache-2.0.

This is the *outbound* declaration. It does not, by itself, prove that every
file is freely the project's to license that way — but per the Foundational
Axioms (A1: Explicit Provenance) the burden is met when REUSE coverage is
total *and* no contradictory metadata exists, which is the case here.

### F-5 — Copyright attribution is by GitHub-account inference, not by signed CLA

`REUSE.toml` attributes the entire pre-existing tree to "Copyright (C) 2026
Chris Hay" based on the upstream repository owner's GitHub identity. There is
no contributor licensing agreement on file, no `AUTHORS` file, and the git
history (post-merge of `chrishayuk/virtual-experts` PR #33) does not
distinguish individual contributor copyright.

This is acceptable practice for a small upstream maintained by a single
identifiable author, but for completeness the upstream report (Phase A3)
flags this for upstream's confirmation — and recommends upstream adopt an
`AUTHORS` file or in-tree CLA mechanism if they intend to accept third-party
contributions.

## Recommended REUSE.toml amendments (to be applied in Phase B)

| Path | Annotation | Rationale |
|---|---|---|
| `knowledge/LICENSE` | `SPDX-FileCopyrightText = "NONE"`, `SPDX-License-Identifier = "Apache-2.0"`, precedence `override` | License-text boilerplate; same treatment as the top-level `LICENSE`. |
| `audit/**` | `SPDX-FileCopyrightText = "Copyright (C) 2026 Ian Douglas Lawrence Norman McLean"`, `SPDX-License-Identifier = "CC-BY-SA-4.0"`, precedence `override` | New audit deliverables authored by this fork; per the forward licensing posture, fork-authored docs are CC-BY-SA-4.0. |

Phase B will additionally introduce a fork-default annotation block for
*future* code contributions (`SPDX-License-Identifier = "AGPL-3.0-or-later"`)
and docs contributions (`SPDX-License-Identifier = "CC-BY-SA-4.0"`),
both attributed to the fork's contributor copyright line. The existing
upstream-default block stays as the catch-all for paths whose origin is the
upstream repository.

## Conclusion

The first-party tree is mechanically Apache-2.0, fully covered by the
existing `REUSE.toml` manifest, with one tooling false-positive in
`scripts/check_apache_license.sh` (F-1) and one sub-project license-text
duplicate that should be explicitly annotated (F-2). No first-party file
carries a license other than Apache-2.0, and no first-party file is
unattributed.

The mismatch between the project's outbound stance and the *inbound*
licensing of its dependency closure (notably the AGPL-3.0-only `evalexpr`
crate) is a separate finding documented in `audit/dependency-licenses.md`.
