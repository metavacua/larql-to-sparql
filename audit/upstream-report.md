<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: CC-BY-SA-4.0
-->

# Upstream licensing report — for the maintainers of chrishayuk/larql

This is a self-contained report intended to be sent upstream to the original
project at <https://github.com/chrishayuk/larql>. It summarises licensing
findings discovered while auditing the downstream fork
`metavacua/larql-to-sparql`. The fork has elected to relicense its own
forward contributions under AGPL-3.0-or-later (code) and CC-BY-SA-4.0 (docs),
which is one valid response to the obligations summarised below. Upstream is
under no obligation to make the same choice, but should be aware of the facts.

This report describes the *upstream* tree; nothing here requires action by
upstream, but each item below may warrant a deliberate licensing decision.

## TL;DR

1. The upstream tree is uniformly declared `Apache-2.0` and is
   mechanically clean by REUSE 3.3 (1102 of 1102 tracked files have both
   copyright and license metadata; the only `SPDX-License-Identifier` value
   found anywhere in the tree is `Apache-2.0`).
2. **One direct dependency carries strong-copyleft obligations that the
   top-level `Apache-2.0` declaration does not propagate**: `evalexpr v12.x`
   is **AGPL-3.0-only**, consumed by `crates/model-compute` behind the
   feature flag `native`. Pinning `evalexpr` to `^11` (last MIT release)
   restores Apache-2.0-compatible licensing without any API change.
3. Two sub-projects (`knowledge/`, the Python pipeline) ship their own
   `LICENSE` files duplicating the Apache-2.0 boilerplate; either annotate
   them in `REUSE.toml` or remove them in favour of the top-level `LICENSE`.
4. Several precautionary items in any `cargo-deny` / `REUSE.toml` you adopt
   should be calibrated to the targets you actually distribute on; the
   current downstream allow-list lists `OpenSSL` and `Unicode-DFS-2016`
   which never appear in the resolved graph.

## Background

The downstream fork performed a deterministic licensing audit using:

- `reuse 6.2.0` — REUSE 3.3 compliance, file-level coverage check.
- `cargo-deny check licenses` — transitive-closure license-policy check.
- `cargo-license` — per-crate published license expression for the entire
  resolved graph.

Raw outputs are in `audit/cargo-deny-licenses.txt` and
`audit/cargo-license.json` of the fork; both are reproducible from a
`cargo generate-lockfile` snapshot of the upstream tree.

## Findings

### U-1 (BLOCKER) — `evalexpr v12.x` introduces AGPL-3.0-only into an Apache-2.0-declared distribution

`crates/model-compute/Cargo.toml` declares (as it appears in the fork tree,
which is in sync with upstream `main` for this crate):

```toml
[features]
native = ["dep:evalexpr", "dep:chrono"]

[dependencies]
evalexpr = { version = "12", optional = true }
```

**The licensing facts**

- `evalexpr v12.0.0` (released 2025-09-15 by `isibboi`) relicensed from MIT
  to **AGPL-3.0-only**. All v12.x releases are AGPL.
- The previous `^11.x` line (last release v11.3.0) remains MIT-licensed and
  was a strict subset of the v12 API surface — there are no breaking changes
  in v12.0 except the licence change itself.

**The obligations created**

When `model-compute` is built with `--features native` (which is the typical
build path, since `native` covers `chrono` time formatting too), the
resulting binary statically links AGPL-3.0-only code. Per AGPL-3.0:

- §2: distribution of the binary requires offering corresponding source
  *under AGPL-3.0-only*.
- §13: if a modified version of the program is offered to users over a
  network, those users must be offered the AGPL source.

These obligations are *incompatible* with the project's top-level
`Apache-2.0` declaration: an Apache-2.0 distributable cannot impose §13's
network-distribution clause on its consumers. Anyone redistributing the
upstream tree with `model-compute :: native` enabled is — silently — under
AGPL.

**Recommended remediation (one of)**

1. **Pin `evalexpr` to `^11`** (recommended; lowest impact). Edit
   `crates/model-compute/Cargo.toml`:
   ```toml
   evalexpr = { version = "^11", optional = true }
   ```
   No source changes needed; v11→v12 had no API break.

2. Replace `evalexpr` with a permissive equivalent: `meval = "0.2"` (MIT)
   or `fasteval = "0.2"` (MIT/Apache-2.0). Both require small adapter code.

3. Accept the AGPL obligation deliberately: relicense the project to
   AGPL-3.0-only outbound, update `Cargo.toml`'s `license` field, update
   `LICENSE` to the AGPL text, document under `NOTICE`. This is the route
   the downstream fork chose.

### U-2 — Sub-project license-text duplication

`knowledge/LICENSE` is a verbatim copy of the Apache-2.0 boilerplate already
present at the top-level `LICENSE`. `knowledge/pyproject.toml` declares
`license = {text = "Apache-2.0"}` and `name = "larql-knowledge"`,
indicating this directory is intended to be installable as an independent
Python package.

This is correct practice for a separately-distributable Python package
(PyPI requires the licence text to ship with each package). It is
**not** a defect, but two small clarifications would help downstream
auditors:

- Annotate `knowledge/LICENSE` explicitly in any future `REUSE.toml`:
  ```toml
  [[annotations]]
  path = "knowledge/LICENSE"
  precedence = "override"
  SPDX-FileCopyrightText = "NONE"
  SPDX-License-Identifier = "Apache-2.0"
  ```
  (treat it as license-text boilerplate, mirroring the standard treatment
  of the top-level `LICENSE`).

- Confirm whether `knowledge/` originated from a separate codebase. The
  `knowledge/README.md` references `chrishayuk/chuk-larql-rs` as the
  consumer, suggesting `knowledge/` may have been authored separately and
  imported. If so, the file-level copyright attribution (currently inferred
  to be Chris Hay) should be confirmed against your records.

### U-3 — Dependency-tree licensing oddities upstream may want to track

The full per-crate inventory is in `audit/cargo-license.json`; the
non-trivial entries are:

| Crate | License | Path | Action upstream may want |
|---|---|---|---|
| `evalexpr v12.0.3` | AGPL-3.0 | `model-compute` (direct) | See U-1. |
| `webpki-roots v1.0.7`, `webpki-root-certs v1.0.7` | CDLA-Permissive-2.0 | via `ureq` → `hf-hub` → `larql-vindex`; via `openblas-build` | Permissive; safe to adopt. Add to any allow-list you maintain. |
| `option-ext v0.2.0` | MPL-2.0 | via `dirs` → `hf-hub` → `larql-vindex` | Permissive at the consumer's scope (file-scoped weak copyleft). No action. |
| `r-efi v5.x`, `v6.x` | `Apache-2.0 OR LGPL-2.1-or-later OR MIT` | UEFI runtime helper, transitive | Disjunctive; we elect Apache-2.0/MIT. No action. |
| `clipboard-win v5.4.1`, `error-code v3.3.2` | BSL-1.0 | Windows-only; not in current Linux/macOS resolution | If you ship Windows binaries, add `BSL-1.0` to your allow-list. |

### U-4 — Provenance attribution model

The downstream fork attributes the entire upstream tree to "Copyright (C)
2026 Chris Hay" via a blanket REUSE annotation, on the basis that the
upstream repository is owned by your GitHub account. This is conventional
for a small project with a single identifiable maintainer, but is not the
same as a verified per-file copyright record.

If the project ever accepts third-party contributions (or has done so in
the past, e.g. the `chrishayuk/virtual-experts` merge in PR #33), the
blanket attribution may be incomplete. Consider:

- Adding an `AUTHORS` file enumerating contributors who hold copyright in
  any non-trivial portion of the tree.
- Adopting an inbound CLA or DCO (Developer Certificate of Origin) sign-off
  policy for new pull requests, so that contributor copyright is recorded
  in commit metadata.
- Per-file annotation in `REUSE.toml` for any file whose copyright is
  demonstrably not yours.

This is process advice, not a defect finding; the current state is internally
consistent and the fork's manifest correctly delegates to your attribution.

## Suggested artifacts you can adopt verbatim

Should you wish to incorporate any of this work directly, the following files
in the downstream fork are designed to be portable upstream:

- `REUSE.toml` — paths and idiom; remove the
  `Ian Douglas Lawrence Norman McLean` override block (which scopes the
  fork's compliance toolchain) and you are left with a manifest applicable
  to upstream as-is.
- `cog.toml`, `cliff.toml`, `CHANGELOG.md`, `scripts/check_changelog.sh`,
  `scripts/version_preflight.sh` — Conventional Commits + Keep a Changelog
  + SemVer preflight, axiomatically described in
  `docs/specs/compliance-pipeline.md`.
- `deny.toml` — drop our additions for `AGPL-3.0-or-later` /
  `CC-BY-SA-4.0` if upstream stays Apache-2.0-only; otherwise the file is a
  reasonable starting allow-list for the resolved tree.
- `audit/` — the report set this email accompanies; the methodology
  reproduces verbatim against the upstream tree.

All of these files are licensed under either Apache-2.0 (compliance
toolchain) or CC-BY-SA-4.0 (this report and the rest of `audit/`); both
are compatible with upstream's Apache-2.0 outbound for redistribution
purposes.

## Contact

This report was prepared by the maintainer of the
`metavacua/larql-to-sparql` fork as part of routine licensing-pipeline work.
Please raise any factual corrections as issues against that fork; we will
amend `audit/upstream-report.md` and notify upstream of the change.
