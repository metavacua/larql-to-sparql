<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
SPDX-License-Identifier: CC-BY-SA-4.0
-->

# Dependency licensing audit

Inventory of the inbound license expressions of every crate in the transitive
dependency closure of the workspace, evaluated on the targets enumerated in
`deny.toml :: graph` (`x86_64-unknown-linux-gnu`, `x86_64-unknown-linux-musl`,
`aarch64-unknown-linux-gnu`, `aarch64-apple-darwin`, `x86_64-apple-darwin`).

## Method

1. Generated `Cargo.lock` deterministically with `cargo generate-lockfile`.
2. Captured `cargo-deny check licenses` verdict against the existing
   `deny.toml` allow-list → `audit/cargo-deny-licenses.txt`.
3. Captured the per-crate published license expression for the entire
   resolved graph with `cargo-license --json` → `audit/cargo-license.json`.
4. Aggregated by license expression and surfaced any expression outside the
   trivially-permissive set (MIT, Apache-2.0, BSD-2/3-Clause, ISC, Zlib,
   CC0-1.0).

Both raw outputs are committed verbatim alongside this report so that the
audit is reproducible from a fixed lockfile snapshot.

## Aggregate breakdown

`cargo-license` reports the following license-expression frequencies across
the resolved graph:

| Count | License expression |
|---:|---|
| 288 | `Apache-2.0 OR MIT` |
|  81 | `MIT` |
|  37 | `Apache-2.0 OR Apache-2.0 WITH LLVM-exception OR MIT` |
|  34 | `Apache-2.0 WITH LLVM-exception` |
|  31 | `Apache-2.0` |
|  18 | `Unicode-3.0` |
|   7 | `MIT OR Unlicense` |
|   3 | `Apache-2.0 OR MIT OR Zlib` |
|   3 | `Apache-2.0 OR ISC OR MIT` |
|   3 | `Apache-2.0 OR BSD-2-Clause OR MIT` |
|   2 | `ISC` |
|   2 | **`CDLA-Permissive-2.0`** |
|   2 | **`BSL-1.0`** |
|   2 | `Apache-2.0 OR LGPL-2.1-or-later OR MIT` |
|   1 | `Zlib` |
|   1 | **`MPL-2.0`** |
|   1 | `BSD-3-Clause AND MIT` |
|   1 | `BSD-3-Clause` |
|   1 | `BSD-2-Clause` |
|   1 | `Apache-2.0 OR CC0-1.0 OR MIT-0` |
|   1 | `Apache-2.0 OR BSL-1.0` |
|   1 | `Apache-2.0 AND ISC` |
|   1 | **`AGPL-3.0`** |
|   1 | `0BSD OR Apache-2.0 OR MIT` |
|   1 | `(Apache-2.0 OR MIT) AND Unicode-3.0` |
|   1 | `(Apache-2.0 OR MIT) AND BSD-3-Clause` |
|   1 | `(Apache-2.0 OR ISC) AND ISC` |
|   1 | compound BSD/MIT/ISC/Apache rollup (single crate) |

Bold rows are the expressions that carry obligations beyond the trivially
permissive baseline and are individually itemised below.

## Findings

### D-1 — `evalexpr v12.0.3` is AGPL-3.0-only (BLOCKER for outbound permissive distribution)

```
evalexpr v12.0.3
  license   = "AGPL-3.0"   (cargo-deny normalises to AGPL-3.0-only)
  authors   = "isibboi <isibboi@gmail.com>"
  repository = https://github.com/ISibboI/evalexpr.git
  consumed by = model-compute v0.1.0  (direct dep; feature-gated `native`)
```

**Provenance**

- `crates/model-compute/Cargo.toml:13` — `native = ["dep:evalexpr", "dep:chrono"]`
- `crates/model-compute/Cargo.toml:20` — `evalexpr = { version = "12", optional = true }`
- `crates/model-compute/src/native/arithmetic.rs` — six usage sites
  (`evalexpr::eval`, `evalexpr::Value::{Int,Float,Boolean,String}`).

**Licensing history of `evalexpr`**

The `evalexpr` crate was MIT-licensed up to and including v11.x; it relicensed
to AGPL-3.0-only at v12.0.0. The `model-compute/Cargo.toml` requirement
`version = "12"` deliberately or inadvertently moved across the relicense
boundary. Pinning to `^11` would restore an MIT-permissive inbound license
without changing API (no breaking change at v12.0).

**Impact**

AGPL-3.0 is strong-copyleft and compatible only with AGPL-3.0-or-later
outbound. As long as `model-compute` enables its `native` feature, the
distributable artefact (the binary `larql` produced by `larql-cli`, which
depends transitively on `model-compute`) inherits AGPL-3.0 obligations:

- §2 — must offer corresponding source.
- §13 — if the artefact is "modified" and exposed over a network, the network
  service must offer source to remote users.

Per project decision (this fork's forward licensing posture), AGPL outbound
is acceptable; the obligation is documented and the dependency is retained.

For upstream `chrishayuk/larql`, which presents itself as Apache-2.0, this
is a real conflict: their distributable inherits AGPL obligations the
top-level license does not declare. See `audit/upstream-report.md`.

### D-2 — `webpki-roots v1.0.7` and `webpki-root-certs v1.0.7` are CDLA-Permissive-2.0

```
webpki-roots v1.0.7      license = CDLA-Permissive-2.0
webpki-root-certs v1.0.7 license = CDLA-Permissive-2.0
  consumed by = ureq v3.3.0
                  -> hf-hub v0.5.0
                       -> larql-vindex v0.1.0
                            -> larql-{cli,inference,lql,python,server}
                  -> openblas-build v0.10.15  (build-time only)
```

CDLA-Permissive-2.0 (Community Data License Agreement, Permissive Variant 2.0)
is a Linux-Foundation-stewarded permissive licence designed for *data* (here,
the root certificate bundle published by Mozilla via the WebPKI process). It
imposes no copyleft obligation; it is rejected by current `deny.toml` only
because the allow-list does not enumerate it.

**Recommendation**: add `"CDLA-Permissive-2.0"` to `deny.toml :: [licenses] allow`
in Phase B.

### D-3 — `clipboard-win v5.4.1` and `error-code v3.3.2` are Boost Software License 1.0

```
clipboard-win v5.4.1   license = BSL-1.0      (Windows-only; not in Linux/macOS graph)
error-code v3.3.2      license = BSL-1.0      (transitive of clipboard-win)
```

BSL-1.0 is OSI-approved permissive. Its only obligation beyond MIT is the
"no obligation to redistribute the licence text in machine code" clause,
which simplifies binary-distribution accounting. Not currently triggered on
the targets enumerated in `deny.toml :: graph` (Linux + macOS) — `cargo-deny`
does not surface it as a finding — but listed here for completeness because
`cargo-license --json` resolves the full graph including Windows-only deps.

**Recommendation**: add `"BSL-1.0"` to the allow-list defensively if Windows
ever joins the target list; otherwise no action.

### D-4 — `option-ext v0.2.0` is MPL-2.0 (weak copyleft)

```
option-ext v0.2.0   license = MPL-2.0
  consumed by = dirs-sys v0.5.0
                  -> dirs v6.0.0
                       -> hf-hub v0.5.0
                            -> larql-vindex v0.1.0
                                 -> larql-{cli,inference,lql,python,server}
```

MPL-2.0 is *file-scoped* weak copyleft: only files whose source is the
MPL-licensed crate itself need to retain MPL on redistribution; modifications
to the larger work that links the MPL crate do not need to be MPL. The
project distributes `option-ext`'s source unchanged via `Cargo.lock`-pinned
version; no file-level modification is involved.

**Outbound effect**: none for the consuming binary; *if* this project
redistributed a vendored or modified copy of `option-ext`'s sources, those
sources would need to remain MPL-2.0.

**Recommendation**: add `"MPL-2.0"` to the allow-list (already present in
the existing `deny.toml`, no change needed).

### D-5 — `r-efi v5.3.0` and `r-efi v6.0.0` are tri-licensed including LGPL-2.1-or-later

```
r-efi v5.3.0   license = "Apache-2.0 OR LGPL-2.1-or-later OR MIT"
r-efi v6.0.0   license = same
```

These are tri-licensed at the *crate* level. Cargo / `cargo-deny` evaluates
the SPDX expression and accepts the crate when *any* disjunct is in the
allow-list. With `Apache-2.0` and `MIT` both allowed, the LGPL disjunct is
never selected and creates no obligation.

**Recommendation**: no action. Document the choice (this report).

### D-6 — `Unicode-3.0` (18 crates) and the embedded `Unicode-3.0` conjuncts

`Unicode-3.0` (the Unicode License v3) is OSI-approved permissive; it is
already in the existing `deny.toml` allow-list. The two compound expressions
`(Apache-2.0 OR MIT) AND Unicode-3.0` and the Apache+ICU-derived `icu_*`
crates are accepted without further action.

### D-7 — Allow-list contains two entries that match no crate in the resolved graph

`cargo-deny` warns:

```
warning[license-not-encountered]: license was not encountered
   ┌─ deny.toml:66 — "OpenSSL"
   ┌─ deny.toml:63 — "Unicode-DFS-2016"
```

Both were precautionary inclusions in the original allow-list. Neither is
present in the current resolved graph (`Unicode-3.0` superseded
`Unicode-DFS-2016` upstream). Suggest dropping both in Phase B for
warning-clean output; can be re-added if a future dep introduces them.

## Outbound compatibility

The fork's declared forward licensing posture is **AGPL-3.0-or-later**
(code) **and CC-BY-SA-4.0** (docs/data) for *new* contributions.

| Inbound licence | Outbound combinable with AGPL-3.0-or-later? | Notes |
|---|---|---|
| `Apache-2.0`, `MIT`, `BSD-2/3-Clause`, `ISC`, `Zlib`, `CC0-1.0`, `0BSD`, `Unlicense`, `MIT-0` | Yes | Trivially permissive. |
| `Apache-2.0 OR MIT` and similar disjunctive permissives | Yes | We elect a permissive disjunct that is AGPL-compatible (Apache-2.0 by default, since the FSF has confirmed Apache-2.0 ↔ AGPL-3.0 compatibility; both grant patent rights). |
| `Unicode-3.0` | Yes | Permissive. |
| `MPL-2.0` | Yes | Weak copyleft, file-scoped; AGPL-3.0-compatible per FSF's MPL-2.0 §3.3 analysis. |
| `BSL-1.0` | Yes | Permissive. |
| `CDLA-Permissive-2.0` | Yes | Permissive. |
| `Apache-2.0 WITH LLVM-exception` | Yes | Adds an exception to Apache-2.0; AGPL-compatibility preserved. |
| `LGPL-2.1-or-later` (as disjunct) | Compatible *if* selected; we never select it because `Apache-2.0`/`MIT` are also offered. | No action. |
| **`AGPL-3.0` / `AGPL-3.0-only`** (`evalexpr`) | **Yes — but mandates AGPL-3.0-only outbound** | Strong copyleft. The artefact carrying `evalexpr` cannot be redistributed under a more-permissive licence than AGPL-3.0. |

## Phase B remediation summary

1. Add to `deny.toml :: [licenses] allow`: `"AGPL-3.0"`, `"AGPL-3.0-only"`,
   `"AGPL-3.0-or-later"`, `"CDLA-Permissive-2.0"`. Optionally add `"BSL-1.0"`
   defensively for future Windows targets.
2. Drop unused `"OpenSSL"` and `"Unicode-DFS-2016"` from the allow-list.
3. Add `LICENSES/AGPL-3.0-or-later.txt` and `LICENSES/CC-BY-SA-4.0.txt` per
   REUSE 3.x layout (the existing pipeline's `provenance` job will fail
   without them the first time a file declares either SPDX-id).
4. Update `NOTICE` to enumerate the AGPL transitive obligation
   (`evalexpr v12.x`) and document the source-availability commitment under
   AGPL §2 / §13.
