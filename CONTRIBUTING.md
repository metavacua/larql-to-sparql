<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: Apache-2.0
-->

# Contributing to larql-to-sparql

## Inbound license

By submitting a contribution to this fork (a pull request, patch, issue
attachment, or any other artefact intended for inclusion in this
repository) you agree that your contribution is licensed under both:

  * **AGPL-3.0-or-later** — for code, scripts, configuration, build
    files, and other functional artefacts (`LICENSES/AGPL-3.0-or-later.txt`).
  * **CC-BY-SA-4.0** — for documentation, design notes, audit reports,
    data, and other non-functional artefacts (`LICENSES/CC-BY-SA-4.0.txt`).

The maintainers select which licence applies to each contributed file
based on its content. New code defaults to AGPL-3.0-or-later; new docs
default to CC-BY-SA-4.0.

Existing files retain whatever inbound licence they carry (`Apache-2.0`
for the upstream-derived tree). Modifying an existing file does not
relicense it; the file keeps its current `SPDX-License-Identifier`
unless the maintainers explicitly relicense it in a separate, deliberate
PR.

This dual-licence inbound policy is consistent with the obligations
already created by the project's transitive dependency on
`evalexpr v12.x` (AGPL-3.0-only); see `audit/dependency-licenses.md`
finding D-1 and `NOTICE`.

## SPDX headers

Every new file must carry a REUSE-compliant header. The example below is
bracketed by `REUSE-IgnoreStart`/`REUSE-IgnoreEnd` markers so that the
illustrative SPDX identifier is not parsed as a real declaration by either
`reuse lint` or `scripts/check_first_party_licenses.sh`:

<!-- REUSE-IgnoreStart -->
```
SPDX-FileCopyrightText: Copyright (C) <year> <Your Name>
SPDX-License-Identifier: <Apache-2.0 | AGPL-3.0-or-later | CC-BY-SA-4.0>
```
<!-- REUSE-IgnoreEnd -->

Alternatively, add an explicit `[[annotations]]` block for the file in
`REUSE.toml`. Either is sufficient; the `reuse-lint` job in
`.github/workflows/ci.yml` (S0) runs `reuse lint` and verifies coverage.

## Conventional Commits

Commit messages must follow the Conventional Commits grammar enforced by
`cog.toml`. The `commits` job runs `cog check` on the PR range. The
admitted types are listed in `cog.toml`; the most common are `feat`,
`fix`, `docs`, `chore`, `refactor`, `test`, `ci`, `build`, `perf`.

## Changelog

The `changelog` job verifies that the `[Unreleased]` block in
`CHANGELOG.md` matches the deterministic projection of the PR's commits
under `cliff.toml`. Regenerate it locally with:

```bash
git-cliff --unreleased --strip header > /tmp/unreleased.md
# then splice /tmp/unreleased.md into CHANGELOG.md under [Unreleased]
```

`docs:`, `chore:`, `ci:`, `build:`, `test:` commits do not produce
changelog entries under the current `cliff.toml` projection.

## Local pre-flight

Before pushing:

```bash
reuse lint
bash scripts/check_first_party_licenses.sh
bash scripts/check_changelog.sh
bash scripts/version_preflight.sh
cog check
```

The `pre-commit` framework wires these in automatically:

```bash
pip install pre-commit && pre-commit install
```

## Adding a new dependency

If a new direct dependency carries a license not already in
`deny.toml :: [licenses] allow`, add an entry there with a one-line
comment justifying it, and reference the supporting analysis in
`audit/dependency-licenses.md`. The `cargo-deny` job in
`.github/workflows/ci.yml` (S1) will reject the build otherwise.

If the new license is copyleft (GPL/AGPL/LGPL), update `NOTICE` to
record the obligation it creates.
