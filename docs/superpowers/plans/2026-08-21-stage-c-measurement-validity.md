# Stage C Measurement Validity Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the target-analysis-pipeline's Secondary-layer promotion/depth-advancement signal capable of actually reflecting real no_std/no_alloc progress on the mutated crates, instead of being structurally guaranteed to report "no progress" regardless of what the mutation does.

**Architecture:** Eight independently-confirmed defects (real CI artifacts, a minimal isolated Cargo repro, and an exhaustive 7-run/833-target-measurement audit) compound to make the current signal void: the wrong package is measured, no real unmutated-vs-mutated comparison exists anywhere, the recursive cross-round loop has no feedback edge, one mutation stage patches the wrong file, its own postcondition is unsatisfiable even when the mutation is correct, real build failures vanish before being counted, and the pipeline's own spec-required positive control (a golden fixture with a known-in-advance outcome) was never built. This plan fixes each in dependency order — cheapest/most foundational first — and uses the golden fixture as a running acceptance test across the fix sequence, not a bolt-on at the end.

**Tech Stack:** GitHub Actions (`.github/workflows/target-analysis-pipeline.yml`), Python 3.11 (`scripts/target_analysis_*.py`), Rust/Cargo (nightly toolchain, `-Z build-std`).

**Spec:** `docs/superpowers/specs/2026-08-16-target-analysis-pipeline-design.md` (the original design spec this pipeline implements). This plan's own direct evidentiary basis — an independent code review and an exhaustive 7-run CI audit, both dated 2026-08-21 — is recorded in full in `.superpowers/sdd/2026-08-16-target-analysis-pipeline/progress.md`; every finding cited below (C1–C8, I1–I4) refers to that review's own numbering.

## Global Constraints

- No `actions/cache` or `Swatinem/rust-cache` anywhere in the pipeline (spec: Explicitly not doing).
- No `git commit`/`git push` from any CI job, in either layer — mechanically greppable absence (spec: Explicitly not doing, Testing).
- No agent-mediated filtering, summarizing, or relevance-judgment inside the pipeline; every probe's full raw output is preserved and retrievable via `actions/upload-artifact` (Standing Principle 2).
- No exclusion of a probe or tool based on dev-machine availability — every probe targets the GitHub-hosted runner only (Standing Principle 1).
- Every relevant probe runs unconditionally every time; nothing is skipped because an earlier result seems to already explain things (Standing Principle 3).
- No aggregation step collapses one probe's verdict into another's — disagreement between independently mechanically-grounded sources is preserved and surfaced, never merged away (Standing Principle 4).
- `nvptx64-nvidia-cuda` is a standing canary: any probe reporting unexpected clean/success against it is presumptively a bug in that probe, not progress (Standing Principle 5).
- `needs:`/`wait` express only genuine mechanical prerequisites, never judgment-based ordering (Standing Principle 8).
- Every curated (L2) data source is explicitly labeled as curated and checkable against raw, uncurated scan output (Foundational framework).
- All claims about GitHub Actions mechanics or probe behavior are validated by an actual run on a GitHub-hosted runner, never by local simulation (Validation approach) — this plan's own claims about Cargo/rustc/cc-rs behavior are additionally each backed by either a real downloaded CI artifact or a minimal local repro, cited by task.
- Concurrency and least-privilege are standing requirements on every job: `contents: read` as the floor, `actions: read` added only where a job calls `actions/download-artifact`.
- **New for this plan:** every fix must be checked against the golden-fixture canary crate (Task 2) before being considered validated — a fix that cannot be shown to move the canary's own known-in-advance outcome closer to correct is not yet proven, regardless of how plausible its reasoning is.

---

### Task 1: Add `rustup target add` where cross-target Cargo commands run without `-Z build-std`

**Real bug, confirmed via the 2026-08-21 audit's own reproduction:** `grep -rn "rustup target add" .github/workflows/ scripts/` returns zero hits anywhere in this pipeline. `build-attempt` and `dependency-graph` both run real cross-target Cargo commands (`cargo check`, `cargo clippy`, `cargo build`, `cargo tree`, `cargo metadata --filter-platform`) without `-Z build-std` — meaning Cargo needs a prebuilt `core`/`std` sysroot for each target, which only exists after `rustup target add <target>`. Confirmed directly in a real job log: `cargo check --target aarch64-linux-android` (no build-std flag) fails immediately with `error[E0463]: can't find crate for 'core' ... consider downloading the target with 'rustup target add aarch64-linux-android'`. `rustup target list --toolchain nightly` confirms 118 of the 119 real targets in this pipeline's matrix are installable this way; the lone exception is `s390x-unknown-none-softfloat`, which rustup ships no component for at all.

This is the Primary layer's own baseline measurement failing for a reason that has nothing to do with no_std/no_alloc portability — fixing it is a prerequisite for every later task in this plan, since Task 7's new unmutated-vs-mutated Stage C comparison needs the Secondary layer's own baseline pass to be measuring something real too.

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (`build-attempt` job's per-target loop; `dependency-graph` job's per-target loop)

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing new — this is a pure environment-setup fix with no schema change.

- [ ] **Step 1: Add a tolerant `rustup target add` call to `build-attempt`'s per-target loop**

In `.github/workflows/target-analysis-pipeline.yml`, in the `build-attempt` job's "Build-attempt probes for every target/cmd/features combination in this batch" step, immediately after the line `while IFS= read -r TARGET; do`, add:
```bash
            rustup target add "$TARGET" --toolchain nightly 2>&1 || echo "::warning::rustup target add failed for $TARGET (expected for targets with no prebuilt std component, e.g. s390x-unknown-none-softfloat)"
```

- [ ] **Step 2: Add the same tolerant call to `dependency-graph`'s per-target loop**

In the same file, in the `dependency-graph` job's "Dependency-graph probes for every target in this batch" step, immediately after `while IFS= read -r TARGET; do`, add the identical line:
```bash
            rustup target add "$TARGET" --toolchain nightly 2>&1 || echo "::warning::rustup target add failed for $TARGET (expected for targets with no prebuilt std component, e.g. s390x-unknown-none-softfloat)"
```

- [ ] **Step 3: Validate YAML and heredoc syntax locally**

Run:
```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/target-analysis-pipeline.yml')); print('YAML OK')"
actionlint .github/workflows/target-analysis-pipeline.yml
```
Expected: `YAML OK`, zero actionlint findings.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "fix: add missing rustup target add -- Primary layer cross-target commands had no prebuilt std/core sysroot for any non-host target"
```

- [ ] **Step 5: Push and verify on a real runner**

Push and confirm: for a real non-host target in `build-attempt`'s and `dependency-graph`'s real output, the previously-universal `error[E0463]: can't find crate for 'core'` no longer appears for targets `rustup target list --toolchain nightly` reports as available; confirm the one known exception (`s390x-unknown-none-softfloat`) logs the expected `::warning::` and continues rather than aborting the batch.

---

### Task 2: Build the golden-fixture canary crate and its self-test assertion (RED first)

**Why this is Task 2, not the last task:** the spec (Testing section) required exactly this — "a minimal, synthetic crate with a specific, known-in-advance issue, run through the full Stage A→C pipeline, asserting the pipeline reports the known result" — and its absence is the confirmed root cause of why C1–C3 survived 21 tasks, a whole-branch review, and two cleanup passes: nothing ever asserted the pipeline *could* report progress, so a signal that was `false` for every one of 833 real target-measurements across 7 runs was never distinguishable from a correct negative. Building it now, before the remaining fixes, means every later task in this plan re-runs it and reports how the failure reason changes — the actual TDD discipline that was missing the first time, applied to the fix itself.

**Design of the canary.** A crate with zero non-workspace, non-core/alloc dependencies, containing a deliberately planted `std`-only construct that Stage A's clippy lints (`std_instead_of_core`, `std_instead_of_alloc`, `alloc_instead_of_core`) are already proven (this session, real CI) to rewrite, and that Stage B's `insert_no_std_scaffold()` is already proven to complete into a crate buildable under `-Z build-std=core,alloc`. Because it depends on nothing but `core`/`alloc`, its known-in-advance outcome is unconditional: after Stage A+B mutation, `cargo check -p larql-nostd-canary --target <any target with a real core sysroot> -Z build-std=core,alloc` **must** succeed. If it doesn't, the defect is in the pipeline's own measurement machinery, not in an experimental finding — exactly the positive control the spec calls for.

**Files:**
- Create: `crates/larql-nostd-canary/Cargo.toml`
- Create: `crates/larql-nostd-canary/src/lib.rs`
- Modify: `Cargo.toml` (add to `members`)
- Modify: `scripts/target_analysis_promotion.py` (add `GOLDEN_FIXTURE_CRATE` constant, add it to `MUTATED_LIBRARY_CRATES`)
- Modify: `tests/test_target_analysis_promotion.py` (regression test for the new constant)
- Modify: `.github/workflows/target-analysis-pipeline.yml` (`secondary-mutate`'s Stage B3 step: keep the canary crate in the workspace regardless of `STAGE_B3_REACHABLE_CLOSURE`, since it has no real path-dependency relationship to the other four mutated crates and is not part of their transitive closure; `secondary-layer-self-test`: add the golden-fixture check)

**Interfaces:**
- Consumes: `insert_no_std_scaffold()`, `stage_b_lib_rs_filenames()` (Task 16/18, already proven).
- Produces: `GOLDEN_FIXTURE_CRATE = "larql-nostd-canary"`, importable from `scripts.target_analysis_promotion`; a new `secondary-layer-self-test` step whose pass/fail is this plan's own acceptance signal.

- [ ] **Step 1: Create the canary crate's manifest**

`crates/larql-nostd-canary/Cargo.toml`:
```toml
[package]
name = "larql-nostd-canary"
version = "0.1.0"
edition = "2021"
publish = false

[lib]
path = "src/lib.rs"
```

- [ ] **Step 2: Create the canary crate's source, with a deliberately planted std-only construct**

`crates/larql-nostd-canary/src/lib.rs`:
```rust
//! A minimal canary crate with a deliberately planted, known-in-advance
//! outcome. This file uses `std::vec::Vec`/`std::string::String` directly
//! -- Stage A's clippy lints (`std_instead_of_alloc`) are already proven
//! (this repo's own real CI, Task 16) to rewrite these to their
//! `alloc`-crate equivalents, and Stage B's `insert_no_std_scaffold()` is
//! already proven to complete the result into a crate buildable under
//! `-Z build-std=core,alloc`. This crate has zero dependencies beyond
//! core/alloc, so after Stage A+B mutation it MUST compile cleanly under
//! `-Z build-std=core,alloc` on any target with a real core sysroot --
//! there is no legitimate reason for this specific check to ever fail.

pub fn make_greeting(name: &str) -> std::string::String {
    let mut parts: std::vec::Vec<std::string::String> = std::vec::Vec::new();
    parts.push(std::string::String::from("hello, "));
    parts.push(std::string::String::from(name));
    parts.concat()
}
```

- [ ] **Step 3: Add the crate to the real workspace**

In the root `Cargo.toml`, add `"crates/larql-nostd-canary",` to the `members` array (do not add it to `default-members` — it is a Secondary-layer-only fixture, never built by a plain `cargo build` at the workspace root).

- [ ] **Step 4: Write the failing test for the new constant (TDD)**

In `tests/test_target_analysis_promotion.py`, add `GOLDEN_FIXTURE_CRATE` to the existing `from scripts.target_analysis_promotion import (...)` block (alphabetically, before `MUTATED_LIBRARY_CRATES`), then add:
```python
def test_golden_fixture_crate_is_one_of_the_mutated_crates():
    assert GOLDEN_FIXTURE_CRATE in MUTATED_LIBRARY_CRATES
    assert GOLDEN_FIXTURE_CRATE == "larql-nostd-canary"
```

- [ ] **Step 5: Run the test, confirm it fails for the right reason**

Run: `python3 -m pytest tests/test_target_analysis_promotion.py -v -k golden_fixture`
Expected: FAIL — `ImportError: cannot import name 'GOLDEN_FIXTURE_CRATE'`.

- [ ] **Step 6: Add the constant**

In `scripts/target_analysis_promotion.py`, immediately before `MUTATED_LIBRARY_CRATES`, add:
```python
# The spec's required positive control (Testing section): a crate with a
# known-in-advance outcome, run through the real Stage A->C pipeline, so a
# broken measurement is distinguishable from a genuine experimental result.
GOLDEN_FIXTURE_CRATE = "larql-nostd-canary"
```
Then change `MUTATED_LIBRARY_CRATES` to include it:
```python
MUTATED_LIBRARY_CRATES = (
    "larql-boundary",
    "larql-vindex-spec",
    "larql-models",
    "larql-compute",
    GOLDEN_FIXTURE_CRATE,
)
```

- [ ] **Step 7: Run the test, confirm GREEN**

Run: `python3 -m pytest tests/test_target_analysis_promotion.py -v`
Expected: all tests pass, including the new one. (`test_stage_b_lib_rs_filenames_accepts_any_real_mutated_crate`-style tests should still pass unmodified, since they test specific named crates, not the full tuple.)

- [ ] **Step 8: Keep the canary crate in Stage B3's trimmed workspace regardless of the real closure**

In `.github/workflows/target-analysis-pipeline.yml`, in `secondary-mutate`'s "Stage B3: trim workspace members to larql-cli's real tree" step, change:
```python
          reachable = [f"crates/{name}" for name in STAGE_B3_REACHABLE_CLOSURE]
```
to:
```python
          # larql-nostd-canary has no path-dependency relationship to the
          # real 14-crate closure (it depends on nothing), so it is kept
          # explicitly rather than folded into STAGE_B3_REACHABLE_CLOSURE,
          # which must stay an honest, derived-from-real-dependencies set.
          reachable = [f"crates/{name}" for name in STAGE_B3_REACHABLE_CLOSURE] + ["crates/larql-nostd-canary"]
```

- [ ] **Step 9: Add the golden-fixture check to the Secondary-layer self-test**

In the same file, in the `secondary-layer-self-test` job, add a new step after the existing "Blast-radius containment" step:
```yaml
      - name: "Golden fixture: assert the pipeline reports the known planted outcome"
        run: |
          python3 - <<'PYEOF'
          import sys
          sys.path.insert(0, ".")
          from pathlib import Path
          from scripts.target_analysis_promotion import insert_no_std_scaffold, GOLDEN_FIXTURE_CRATE

          lib_rs = Path(f"crates/{GOLDEN_FIXTURE_CRATE}/src/lib.rs")
          lib_rs.write_text(insert_no_std_scaffold(lib_rs.read_text(encoding="utf-8")), encoding="utf-8")
          PYEOF
          cargo +nightly clippy --fix --allow-dirty --allow-staged \
            -p "larql-nostd-canary" --lib -- -W clippy::std_instead_of_core -W clippy::std_instead_of_alloc -W clippy::alloc_instead_of_core
          cargo +nightly check -p larql-nostd-canary --target thumbv6m-none-eabi \
            -Z build-std=core,alloc --message-format=json > canary-check.json
          if grep -q '"level":"error"' canary-check.json; then
            echo "::error::Golden fixture failed -- the pipeline's own known-good canary does not compile under core+alloc. This means the measurement machinery is broken, not that a real experiment found a negative result."
            cat canary-check.json
            exit 1
          fi
          echo "Golden fixture passed: the known-good canary compiles cleanly under -Z build-std=core,alloc."
```
`thumbv6m-none-eabi` is chosen as the check target because it is a genuinely no_std, tier-2, `rustup target add`-installable target (Cortex-M0, no OS) with no native/toolchain blockers of any kind — the cleanest possible target for a crate with zero non-core/alloc dependencies.

- [ ] **Step 10: Commit**

```bash
git add crates/larql-nostd-canary Cargo.toml scripts/target_analysis_promotion.py tests/test_target_analysis_promotion.py .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: add the spec-required golden-fixture canary crate and self-test -- a known-in-advance positive control for the Secondary layer's measurement"
```

- [ ] **Step 11: Push and confirm RED, with the real, specific reason**

Push and confirm on a real runner: the golden-fixture check step fails (this is the expected, correct RED at this point in the plan — Stage C still checks `-p larql-cli`, not `-p larql-nostd-canary`, and even once that's fixed in Task 4, Stage B2's own defect and the missing unmutated-baseline comparison remain). Record the *exact* real failure text in the ledger before proceeding to Task 3 — this is the baseline RED the remaining tasks are measured against.

---

### Task 3: Fix Stage A/B's ordering so `std`→`alloc` rewrites actually persist

**Real, newly discovered defect (found by Task 2's own golden fixture, exactly as it was built to do — not one of the original C1–C8/I1–I4 findings this plan started from).** Task 2's implementer, reproducing this pipeline's real commands locally, found: Stage A (`cargo clippy --fix ... -W clippy::std_instead_of_alloc ...`) runs *before* Stage B ever inserts `extern crate alloc;`. When clippy rewrites `std::collections::BTreeMap` → `alloc::collections::BTreeMap` (or `Arc`, `BTreeSet`, `VecDeque` — any alloc-only path), the crate at that point has no `alloc` crate in scope to reference, so clippy's own re-check of the fixed code hits `error[E0433]: cannot find module or crate 'alloc' in this scope` and **silently reverts the fix**, exiting 0. Confirmed independently on real CI (`secondary-mutate`'s own "Stage A" step log, run `32509146857`): of the 5 crates in `MUTATED_LIBRARY_CRATES`, only `larql-boundary` has clippy findings that avoid an alloc-only path — the other 4 (`larql-vindex-spec`, `larql-models`, `larql-compute`, `larql-nostd-canary`) all hit exactly this revert, on every run of this pipeline to date, and the job still reports success. This has been silently defeating Stage A's own purpose for as long as it has run.

A second, related bug in the same area: `secondary-layer-self-test`'s "Golden fixture" step reproduces Stage A/B in the *opposite* order from `secondary-mutate`'s real sequence (it currently calls `insert_no_std_scaffold` — inserting *both* `#![no_std]` and `extern crate alloc;` — *before* running clippy --fix, not after). This makes the self-test an unfaithful reproduction of the real pipeline regardless of this task's fix, and must be corrected to match.

**The fix, and why it's safe:** split the combined scaffold insertion into two separate insertions, timed differently. `extern crate alloc;` alone (no `#![no_std]`) is valid in a plain, still-`std`-linked crate — `alloc` ships in every target's sysroot regardless of `no_std` — so inserting it *before* Stage A gives clippy's rewrite something real to resolve against, while every `std::` path Stage A does *not* touch keeps compiling exactly as before (std is still fully linked). `#![no_std]` — the attribute that actually removes std — moves to *after* Stage A, once all std-only references have already been rewritten away.

**This task's design carries real uncertainty about clippy's exact lint-firing behavior that this plan's own evidentiary standard requires resolving by a real, local run before touching the pipeline** — Step 1 below is a mandatory verification gate, not optional. If it does not confirm the hypothesis, stop and report BLOCKED with the real observed output; do not proceed to modify the workflow on an unconfirmed guess.

**Files:**
- Modify: `scripts/target_analysis_promotion.py` (replace `insert_no_std_scaffold` with `insert_alloc_extern_crate` + `insert_no_std_attribute`, sharing a new private `_leading_block_end` helper)
- Modify: `tests/test_target_analysis_promotion.py` (replace `insert_no_std_scaffold`'s 2 tests with tests for the split functions and their composition)
- Modify: `.github/workflows/target-analysis-pipeline.yml` (`secondary-mutate`: new "Stage A0" step before Stage A, Stage B's step now inserts only the attribute; `secondary-layer-self-test`: fix both the toolchain gap and the step ordering in "Golden fixture")

**Interfaces:**
- Consumes: `no_std_scaffold_ok()` (Task 1's plan, unchanged — still checks the *final* content has both markers, regardless of which stage inserted which).
- Produces: `insert_alloc_extern_crate(text: str) -> str`, `insert_no_std_attribute(text: str) -> str` (both replace `insert_no_std_scaffold`, which is deleted — nothing else calls it, confirmed by repo-wide grep before writing this task).

- [ ] **Step 1: Mandatory local verification — confirm the hypothesis before writing any workflow change**

In a scratch checkout (nightly + clippy + rust-src installed locally, matching Task 2's implementer's own environment), reproduce against `crates/larql-vindex-spec` (one of the 4 real crates confirmed affected):

```bash
git stash -u  # ensure a clean starting point, restore after
cp crates/larql-vindex-spec/src/lib.rs /tmp/vindex-lib-rs-backup.rs
python3 -c "
import sys
sys.path.insert(0, '.')
from pathlib import Path
lib_rs = Path('crates/larql-vindex-spec/src/lib.rs')
text = lib_rs.read_text(encoding='utf-8')
lines = text.splitlines(keepends=True)
insert_at = 0
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped.startswith('//!') or stripped.startswith('#![') or stripped == '':
        insert_at = i + 1
    else:
        break
lines.insert(insert_at, 'extern crate alloc;\n')
lib_rs.write_text(''.join(lines), encoding='utf-8')
"
cargo +nightly clippy --fix --allow-dirty --allow-staged \
  -p larql-vindex-spec --lib -- -W clippy::std_instead_of_core -W clippy::std_instead_of_alloc -W clippy::alloc_instead_of_core
git diff crates/larql-vindex-spec/src/lib.rs
cp /tmp/vindex-lib-rs-backup.rs crates/larql-vindex-spec/src/lib.rs  # restore
```

Expected (confirms the hypothesis): the `git diff` shows the `std::collections::BTreeMap` → `alloc::collections::BTreeMap`-class rewrite **persisted** this time (not reverted) — the fix compiles now that `alloc` is in scope via the prepended `extern crate alloc;`. If the diff is empty (still reverted) or shows a *different* error, STOP: report BLOCKED with the exact real output, since the fix as designed does not hold and the remaining steps must not be attempted on an unconfirmed premise.

- [ ] **Step 2: Split the scaffold-insertion function**

In `scripts/target_analysis_promotion.py`, replace:

```python
def insert_no_std_scaffold(text: str) -> str:
    lines = text.splitlines(keepends=True)

    insert_at = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("//!") or stripped.startswith("#![") or stripped == "":
            insert_at = i + 1
        else:
            break

    scaffold = "#![no_std]\nextern crate alloc;\n"
    lines.insert(insert_at, scaffold)
    return "".join(lines)
```

with:

```python
def _leading_block_end(text: str) -> int:
    """The line index immediately after any leading //! doc comments, #![...]
    inner attributes, and blank lines -- the correct insertion point for a
    module-level attribute or extern crate declaration, so it lands before
    any real code but after the crate's own existing doc/attributes."""
    lines = text.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("//!") or stripped.startswith("#![") or stripped == "":
            insert_at = i + 1
        else:
            break
    return insert_at


def insert_alloc_extern_crate(text: str) -> str:
    """Runs before Stage A's clippy --fix. A plain, still-std-linked crate
    can declare `extern crate alloc;` without `#![no_std]` -- alloc ships in
    every target's sysroot regardless -- so std:: references Stage A does
    NOT touch keep working exactly as before, while std::X paths clippy
    suggests rewriting to alloc::X now have something to resolve against.
    Real CI evidence (2026-08-21): without this, Stage A's fix for any
    std::->alloc::-only path (BTreeMap, BTreeSet, Arc, VecDeque) fails to
    compile once applied and clippy's own safety check silently reverts it
    -- confirmed for 4 of 5 mutated crates."""
    lines = text.splitlines(keepends=True)
    lines.insert(_leading_block_end(text), "extern crate alloc;\n")
    return "".join(lines)


def insert_no_std_attribute(text: str) -> str:
    """Runs after Stage A. Only #![no_std] remains to insert -- extern
    crate alloc; was already added by insert_alloc_extern_crate() before
    Stage A ran."""
    lines = text.splitlines(keepends=True)
    lines.insert(_leading_block_end(text), "#![no_std]\n")
    return "".join(lines)
```

- [ ] **Step 3: Update the tests**

In `tests/test_target_analysis_promotion.py`, remove `insert_no_std_scaffold` from the import block (add `insert_alloc_extern_crate, insert_no_std_attribute` instead, alphabetically), and replace:

```python
def test_insert_no_std_scaffold_inserts_after_leading_doc_comment_and_attributes():
    text = "//! A crate.\n//! More docs.\n#![allow(dead_code)]\n\npub fn f() {}\n"
    result = insert_no_std_scaffold(text)
    assert result == (
        "//! A crate.\n//! More docs.\n#![allow(dead_code)]\n\n"
        "#![no_std]\nextern crate alloc;\n"
        "pub fn f() {}\n"
    )
    assert no_std_scaffold_ok(result) is True


def test_insert_no_std_scaffold_inserts_at_top_when_no_leading_doc_comment():
    text = "pub fn f() {}\n"
    result = insert_no_std_scaffold(text)
    assert result == "#![no_std]\nextern crate alloc;\npub fn f() {}\n"
```

with:

```python
def test_insert_alloc_extern_crate_inserts_after_leading_doc_comment_and_attributes():
    text = "//! A crate.\n//! More docs.\n#![allow(dead_code)]\n\npub fn f() {}\n"
    result = insert_alloc_extern_crate(text)
    assert result == (
        "//! A crate.\n//! More docs.\n#![allow(dead_code)]\n\n"
        "extern crate alloc;\n"
        "pub fn f() {}\n"
    )


def test_insert_no_std_attribute_inserts_before_a_prior_extern_crate_alloc():
    # The exact post-Stage-A0 shape Stage B receives in the real pipeline:
    # extern crate alloc; already present, #![no_std] still missing.
    text = "//! A crate.\n//! More docs.\n#![allow(dead_code)]\n\nextern crate alloc;\npub fn f() {}\n"
    result = insert_no_std_attribute(text)
    assert result == (
        "//! A crate.\n//! More docs.\n#![allow(dead_code)]\n\n"
        "#![no_std]\nextern crate alloc;\n"
        "pub fn f() {}\n"
    )
    assert no_std_scaffold_ok(result) is True


def test_alloc_then_no_std_attribute_composition_matches_old_combined_scaffold():
    # Real pipeline order (Stage A0 then Stage B) must produce the exact
    # same final scaffold the old single-pass function used to, for a
    # crate with no leading doc comment.
    text = "pub fn f() {}\n"
    step1 = insert_alloc_extern_crate(text)
    step2 = insert_no_std_attribute(step1)
    assert step2 == "#![no_std]\nextern crate alloc;\npub fn f() {}\n"
    assert no_std_scaffold_ok(step2) is True
```

- [ ] **Step 4: Run the tests, confirm pass**

```bash
python3 -m pytest tests/test_target_analysis_promotion.py -v
```
Expected: all tests pass, including the 3 new/changed ones above; no reference to `insert_no_std_scaffold` remains anywhere (`grep -rn insert_no_std_scaffold .` returns nothing).

- [ ] **Step 5: Insert Stage A0 into `secondary-mutate`, retime Stage B**

In `.github/workflows/target-analysis-pipeline.yml`'s `secondary-mutate` job, insert this new step immediately after "Capture pre-mutation lib.rs content per crate and pre-mutation workspace metadata" and immediately before "Stage A: mechanical std->core/alloc rewrite (host target), per crate":

```yaml
      - name: "Stage A0: insert extern crate alloc; (prerequisite for Stage A's alloc-path rewrites to resolve)"
        run: |
          for CRATE in $CRATES; do
            python3 - "$CRATE" <<'PYEOF'
          import sys
          from pathlib import Path

          sys.path.insert(0, ".")
          from scripts.target_analysis_promotion import insert_alloc_extern_crate

          crate = sys.argv[1]
          lib_rs = Path(f"crates/{crate}/src/lib.rs")
          lib_rs.write_text(
              insert_alloc_extern_crate(lib_rs.read_text(encoding="utf-8")),
              encoding="utf-8",
          )
          PYEOF
          done
```

Then change the existing "Stage B: inject `#![no_std]` scaffold, per crate" step's name to `"Stage B: inject #![no_std] attribute, per crate"` and its body's import/call from `insert_no_std_scaffold` to `insert_no_std_attribute`:

```yaml
      - name: "Stage B: inject #![no_std] attribute, per crate"
        run: |
          for CRATE in $CRATES; do
            python3 - "$CRATE" <<'PYEOF'
          import sys
          from pathlib import Path

          sys.path.insert(0, ".")
          from scripts.target_analysis_promotion import insert_no_std_attribute

          crate = sys.argv[1]
          lib_rs = Path(f"crates/{crate}/src/lib.rs")
          lib_rs.write_text(
              insert_no_std_attribute(lib_rs.read_text(encoding="utf-8")),
              encoding="utf-8",
          )
          PYEOF
          done
```

- [ ] **Step 6: Fix `secondary-layer-self-test`'s toolchain gap and step ordering**

In the same workflow file's `secondary-layer-self-test` job, change:
```yaml
      - name: Install nightly Rust
        run: rustup toolchain install nightly --profile minimal
```
to:
```yaml
      - name: Install nightly Rust
        run: |
          rustup toolchain install nightly --profile minimal
          rustup component add clippy rust-src --toolchain nightly
```
(the `minimal` profile excludes both `clippy` and `rust-src`; the "Golden fixture" step below needs `clippy` for its own Stage-A-equivalent call and `rust-src` for `-Z build-std=core,alloc`, matching `secondary-stage-c-and-promotion`'s own already-correct `rustup component add rust-src --toolchain nightly`.)

Then replace the entire "Golden fixture" step (which currently applies the *combined* scaffold *before* clippy --fix — backwards from `secondary-mutate`'s real order) with:
```yaml
      - name: "Golden fixture: assert the pipeline reports the known planted outcome"
        run: |
          python3 - <<'PYEOF'
          import sys
          sys.path.insert(0, ".")
          from pathlib import Path
          from scripts.target_analysis_promotion import insert_alloc_extern_crate, GOLDEN_FIXTURE_CRATE

          lib_rs = Path(f"crates/{GOLDEN_FIXTURE_CRATE}/src/lib.rs")
          lib_rs.write_text(insert_alloc_extern_crate(lib_rs.read_text(encoding="utf-8")), encoding="utf-8")
          PYEOF
          cargo +nightly clippy --fix --allow-dirty --allow-staged \
            -p "larql-nostd-canary" --lib -- -W clippy::std_instead_of_core -W clippy::std_instead_of_alloc -W clippy::alloc_instead_of_core
          python3 - <<'PYEOF'
          import sys
          sys.path.insert(0, ".")
          from pathlib import Path
          from scripts.target_analysis_promotion import insert_no_std_attribute, GOLDEN_FIXTURE_CRATE

          lib_rs = Path(f"crates/{GOLDEN_FIXTURE_CRATE}/src/lib.rs")
          lib_rs.write_text(insert_no_std_attribute(lib_rs.read_text(encoding="utf-8")), encoding="utf-8")
          PYEOF
          cargo +nightly check -p larql-nostd-canary --target thumbv6m-none-eabi \
            -Z build-std=core,alloc --message-format=json > canary-check.json
          if grep -q '"level":"error"' canary-check.json; then
            echo "::error::Golden fixture failed -- the pipeline's own known-good canary does not compile under core+alloc. This means the measurement machinery is broken, not that a real experiment found a negative result."
            cat canary-check.json
            exit 1
          fi
          echo "Golden fixture passed: the known-good canary compiles cleanly under -Z build-std=core,alloc."
```

Note the "Blast-radius containment" step (immediately before "Golden fixture" in this same job) still calls `insert_no_std_scaffold` too — leave it untouched for now; Task 11 (blast-radius containment, formerly Task 10) already has this step in its own scope and will need to account for the split there specifically. This task's scope is the ordering defect Task 2 found, not a blanket sweep of every remaining `insert_no_std_scaffold` reference — confirmed there is exactly one such reference left after this task's own 2 replacements (`secondary-mutate`'s Stage B, `secondary-layer-self-test`'s Golden fixture), in the Blast-radius step, correctly deferred.

- [ ] **Step 7: Validate and commit**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/target-analysis-pipeline.yml')); print('YAML OK')"
actionlint .github/workflows/target-analysis-pipeline.yml
python3 -m pytest tests/ --ignore=tests/test_vindex_bindings.py -v
git add scripts/target_analysis_promotion.py tests/test_target_analysis_promotion.py .github/workflows/target-analysis-pipeline.yml
git commit -m "fix: split Stage A/B's scaffold insertion so std->alloc rewrites actually persist -- Stage A's clippy --fix silently reverted every alloc-only-path rewrite for 4 of 5 mutated crates because alloc wasn't in scope yet when it ran"
```

- [ ] **Step 8: Push and confirm the real failure reason changes**

Push and confirm on real CI: `secondary-mutate`'s "Stage A" step log now shows the previously-reverted fixes (`larql-vindex-spec`, `larql-models`, `larql-compute`, `larql-nostd-canary`) actually persisting (`git diff`/`Fixed ...` output for each, not just `larql-boundary`); the golden-fixture self-test step's real failure reason (if it still fails) has changed from the E0433 ordering error to something else entirely — report the exact new reason in the ledger, since Stage C still checking `-p larql-cli` (fixed next, in Task 4) may still be masking full success at this point. Do not expect a full GREEN yet — this task's acceptance criterion is that the *real, previously-silent* Stage A persistence defect is now visibly gone from the log, not that the golden fixture fully passes (that is Task 9's own gate).

---

### Task 4: Switch Stage C's checked package(s) from `-p larql-cli` to the mutated crates directly

**Real, confirmed defect (C1/C3):** Stage C's real invocation is `cargo +nightly check -p larql-cli --target "$TARGET" -Z build-std=core,alloc --keep-going --message-format=json`. Traced via `cargo tree -i` against the real workspace: `-p larql-cli` pulls in `reqwest`→`hyper-rustls`→`rustls`→`ring` (an HTTP client's TLS backend), `tokenizers`→`onig`/`onig_sys` and `tokenizers`→`esaxx-rs` (an NLP tokenization library's C/C++ native dependencies), and `larql-inference`→`wasmtime` (a WASM runtime) — none of which have anything to do with whether `larql-boundary`/`larql-models`/`larql-vindex-spec`/`larql-compute` (the crates Stage A/B/B2/B3 actually mutate) are no_std/no_alloc-compatible. Confirmed directly against a real Stage C artifact for `aarch64-apple-darwin`: 14,905 error-level messages across 75 distinct crates, **zero** attributed to `larql-boundary`, `larql-models`, or even `larql-cli` itself — the build dies in third-party dependencies before the compiler frontier ever reaches the mutated crates. Swept across 23 real targets (Apple, Android, Windows-gnullvm, Linux-gnu/musl, sanitizers, netbsd, redox, uefi, bare-metal): the same three of four mutated crates are unreached on every one.

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (`secondary-stage-c-and-promotion`'s Stage C invocation; `secondary-layer-self-test`'s noise-floor invocation — aligning its flags with Stage C's real command, per finding I1, since it currently omits `-Z build-std=core,alloc` entirely and so certifies a different measurement than the one it licenses)

**Interfaces:**
- Consumes: `MUTATED_LIBRARY_CRATES` (now 5 entries including the golden fixture, per Task 2).
- Produces: nothing new — same `out/stage-c-$TARGET.json` file, now containing real signal about the actual mutated crates.

- [ ] **Step 1: Derive the `-p` flags from `MUTATED_LIBRARY_CRATES` and switch Stage C's invocation**

In `.github/workflows/target-analysis-pipeline.yml`, in `secondary-stage-c-and-promotion`'s "Stage C and promotion checks for every target in this batch" step, immediately before the `while IFS= read -r TARGET; do` line, add:
```bash
          CHECK_PACKAGES=$(python3 -c 'import sys; sys.path.insert(0, "."); from scripts.target_analysis_promotion import MUTATED_LIBRARY_CRATES; print(" ".join(f"-p {c}" for c in MUTATED_LIBRARY_CRATES))')
```
Then replace:
```bash
            cargo +nightly check -p larql-cli --target "$TARGET" \
              -Z build-std=core,alloc --keep-going --message-format=json \
              > "out/stage-c-$TARGET.json" || true
```
with:
```bash
            cargo +nightly check $CHECK_PACKAGES --target "$TARGET" \
              -Z build-std=core,alloc --keep-going --message-format=json \
              > "out/stage-c-$TARGET.json" || true
```

- [ ] **Step 2: Align the noise-floor self-test's flags with Stage C's real invocation (finding I1)**

In the same file, in `secondary-layer-self-test`'s "Noise floor" step, replace:
```bash
            cargo +nightly check -p larql-cli --target nvptx64-nvidia-cuda \
              --message-format=json --keep-going > noise-floor-run-$i.json || true
```
with:
```bash
            CHECK_PACKAGES=$(python3 -c 'import sys; sys.path.insert(0, "."); from scripts.target_analysis_promotion import MUTATED_LIBRARY_CRATES; print(" ".join(f"-p {c}" for c in MUTATED_LIBRARY_CRATES))')
            cargo +nightly check $CHECK_PACKAGES --target nvptx64-nvidia-cuda \
              -Z build-std=core,alloc --message-format=json --keep-going > noise-floor-run-$i.json || true
```
This step already loops `for i in 1 2; do ... done`, so `CHECK_PACKAGES` is recomputed each iteration — harmless (cheap, deterministic), left as-is rather than hoisted, since this self-test step is not part of the per-target/per-batch hot path this project has previously hoisted work out of.

- [ ] **Step 3: Validate YAML and heredoc syntax locally**

Run:
```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/target-analysis-pipeline.yml')); print('YAML OK')"
actionlint .github/workflows/target-analysis-pipeline.yml
```

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "fix: Stage C checks the mutated crates directly instead of larql-cli's whole irrelevant dependency closure"
```

- [ ] **Step 5: Push and re-run the golden fixture, record the new RED reason**

Push and confirm: the golden-fixture check (Task 2) still fails, but check whether the *reason* has changed (it should now be closer to a real signal about `larql-nostd-canary` and the other mutated crates specifically, not `larql-cli`'s irrelevant third-party closure). Also confirm real error counts for the four already-mutated crates drop sharply for most targets now that `ring`/`onig_sys`/`esaxx-rs`/`wasmtime` are no longer pulled in as check targets (note: they may still appear as *dependencies* of the checked crates if any of the four path-depend on something that path-depends on these — confirm via real data which, if any, still do). Record the real before/after error counts in the ledger.

---

### Task 5: Fix Stage B2 to patch the root workspace manifest, not just per-crate copies

**Real, confirmed defect (C5), with a working counterfactual proof.** Stage B2's mutation (`find crates -name Cargo.toml -exec sed ...`) only ever touches each crate's own `Cargo.toml`, patching `serde = { workspace = true, ... }` lines. It never touches the root `Cargo.toml`'s own `[workspace.dependencies] serde = { version = "1", features = ["derive"] }` entry (confirmed directly: `grep -n "^serde" Cargo.toml` shows no `default-features = false` anywhere in the root manifest, and the real captured `full-mutation.patch` from a real run has no root-`Cargo.toml` hunk touching this line at all). Cargo's real workspace-dependency-inheritance rule means a member's own `default-features = false` override has no effect when the root `[workspace.dependencies]` entry doesn't itself set it — proven via a minimal, isolated two-file repro (one workspace, one member, no unification pressure of any kind): with an unpatched root entry, the resolved unit graph shows `serde` and `serde_core` both carrying `std`; with `default-features = false` added to the root entry alone, `std` disappears from both. **Stage B2's patch has been a no-op on all 119 targets since it was written.**

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (`secondary-mutate`'s "Stage B2: patch std-defaulting dependency features" step)

**Interfaces:**
- Consumes: `STAGE_B2_SERDE_FEATURES` (already established, Task 21).
- Produces: nothing new — same mutation mechanism, now actually mutating the manifest that matters.

- [ ] **Step 1: Extend Stage B2 to also patch the root manifest's serde and serde_json entries**

In `.github/workflows/target-analysis-pipeline.yml`, in `secondary-mutate`'s "Stage B2: patch std-defaulting dependency features" step, after the existing per-crate `find crates -name Cargo.toml -exec sed ...` line, add:
```bash
          # Cargo's workspace-dependency-inheritance rule means a member's
          # own default-features override has no effect unless the root
          # [workspace.dependencies] entry itself sets it (confirmed via a
          # minimal isolated repro, 2026-08-21 review) -- patch it here too.
          sed -i -E "s/serde = \{ version = \"1\", features = \[\"derive\"\] \}/serde = { version = \"1\", default-features = false, features = [$FEATURES_TOML] }/" Cargo.toml
          sed -i -E 's/serde_json = "1"/serde_json = { version = "1", default-features = false }/' Cargo.toml
```
(`$FEATURES_TOML` is already computed by the existing step, immediately above these new lines — no new variable needed.)

- [ ] **Step 2: Validate YAML and heredoc syntax locally, and confirm the sed patterns match the real current root manifest**

Run:
```bash
grep -n "^serde" Cargo.toml
```
Expected output (before this fix runs, i.e. the real current committed state):
```
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```
Confirm these two lines match the sed patterns above exactly (they must, or the substitution silently no-ops). Then:
```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/target-analysis-pipeline.yml')); print('YAML OK')"
actionlint .github/workflows/target-analysis-pipeline.yml
```

- [ ] **Step 3: Local dry-run of the exact sed commands against a copy of the real root manifest**

```bash
cp Cargo.toml /tmp/cargo-toml-dryrun
FEATURES_TOML=$(python3 -c 'import sys; sys.path.insert(0, "."); from scripts.target_analysis_promotion import STAGE_B2_SERDE_FEATURES; print(", ".join(f"\"{f}\"" for f in STAGE_B2_SERDE_FEATURES))')
sed -i -E "s/serde = \{ version = \"1\", features = \[\"derive\"\] \}/serde = { version = \"1\", default-features = false, features = [$FEATURES_TOML] }/" /tmp/cargo-toml-dryrun
sed -i -E 's/serde_json = "1"/serde_json = { version = "1", default-features = false }/' /tmp/cargo-toml-dryrun
grep -n "^serde" /tmp/cargo-toml-dryrun
python3 -c "import tomllib; tomllib.load(open('/tmp/cargo-toml-dryrun', 'rb')); print('valid TOML after patch')"
rm /tmp/cargo-toml-dryrun
```
Expected: `serde = { version = "1", default-features = false, features = ["alloc", "derive"] }` and `serde_json = { version = "1", default-features = false }`, and `valid TOML after patch`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "fix: Stage B2 now patches the root workspace manifest's serde/serde_json entries -- the per-crate-only patch was a no-op under Cargo's workspace-dependency-inheritance rule"
```

- [ ] **Step 5: Push and re-run the golden fixture and Stage B2's own postcondition, record the new result**

Push and confirm: real Stage B2 mutation now touches the root `Cargo.toml` (check `full-mutation.patch` for a real root-manifest hunk); real `cargo metadata`/unit-graph output for the mutated tree shows `serde`/`serde_core` no longer carrying `std`, for at least the targets where the build gets far enough to resolve them. Note in the ledger whether `stage-b2`'s promotion verdict changes real-CI values yet (it may not, until Task 6 also fixes `serde_features_ok()`'s own unsatisfiable check).

---

### Task 6: Fix `serde_features_ok()`'s unsatisfiable exact-set check, and regenerate its test fixture from real cargo output

**Real, confirmed defect (C6).** `serde_features_ok()` requires `set(unit["features"]) == {"alloc", "derive"}` exactly. Real cargo always resolves the `derive` feature into `['alloc', 'derive', 'serde_derive']` — the implied `serde_derive` feature is always present alongside `derive`. Running the shipped function against a corrected, provably std-free real unit graph (post-Task-4 fix) still returns `False`, because of this exact-equality mismatch — meaning even a *perfectly successful* Stage B2 mutation could never satisfy this check as written. A second, independent defect in the same function: it only inspects the unit named `serde`, while the unit that actually carries `std` in modern serde (1.0.228+) is `serde_core`. A third: `tests/fixtures/target_analysis/unit_graph_serde_patched.json` is hand-invented (an obsolete `"pkg_id": "serde 1.0.210"` format; a `"features": ["alloc", "derive"]` list missing the implied feature real cargo always emits) rather than captured from real output — the same defect class already ruled on once in this plan's history (Task 18, `workspace_members_ok`'s obsolete-PackageId-format bug), recurring for the same root reason: a fixture invented by hand instead of captured from a real run.

**Files:**
- Modify: `scripts/target_analysis_promotion.py` (`serde_features_ok`)
- Modify: `tests/fixtures/target_analysis/unit_graph_serde_patched.json`, `tests/fixtures/target_analysis/unit_graph_serde_default.json` (regenerate from real cargo `--unit-graph` output)
- Modify: `tests/test_target_analysis_promotion.py` (update/add regression tests)

**Interfaces:**
- Consumes: `unit_graph_units_named()` (Task 1, unchanged signature).
- Produces: `serde_features_ok(unit_graph) -> bool` — same signature, corrected semantics: "does `std` appear in either `serde`'s or `serde_core`'s resolved features," not "does `serde`'s feature set equal an exact literal."

- [ ] **Step 1: Capture real cargo unit-graph output for both the unpatched and patched (post-Task-4) manifests**

In a scratch checkout with Task 5's fix applied and Stage A/B/B2 mutation run for real (or via a local dry-run of the same commands), capture:
```bash
cargo +nightly build -Z unstable-options --unit-graph -p larql-cli --target x86_64-unknown-linux-gnu > /tmp/real-unit-graph-unpatched.json
# ... apply Task 5's root-manifest patch locally ...
cargo +nightly build -Z unstable-options --unit-graph -p larql-cli --target x86_64-unknown-linux-gnu > /tmp/real-unit-graph-patched.json
python3 -c "
import json
for label, path in [('unpatched', '/tmp/real-unit-graph-unpatched.json'), ('patched', '/tmp/real-unit-graph-patched.json')]:
    data = json.load(open(path))
    for unit in data['units']:
        if unit.get('target', {}).get('name') in ('serde', 'serde_core'):
            print(label, unit['target']['name'], sorted(unit.get('features', [])))
"
```
Expected: `unpatched` shows `std` present for both `serde` and `serde_core`; `patched` shows `std` absent from both, with `serde_derive` present alongside `derive` for `serde` (confirming the real, implied-feature shape).

- [ ] **Step 2: Write the failing tests first (TDD)**

In `tests/test_target_analysis_promotion.py`, replace the two existing `serde_features_ok` tests with:
```python
def test_serde_features_ok_is_false_for_default_features():
    unit_graph = load_json(FIXTURES / "unit_graph_serde_default.json")
    assert serde_features_ok(unit_graph) is False


def test_serde_features_ok_is_true_for_patched_features():
    unit_graph = load_json(FIXTURES / "unit_graph_serde_patched.json")
    assert serde_features_ok(unit_graph) is True


def test_serde_features_ok_checks_serde_core_specifically():
    # Modern serde (1.0.228+) carries `std` on serde_core, not serde itself
    # -- a fixture where serde looks clean but serde_core still has std
    # must still read False.
    unit_graph = {
        "units": [
            {"target": {"name": "serde"}, "features": ["alloc", "derive", "serde_derive"]},
            {"target": {"name": "serde_core"}, "features": ["alloc", "std"]},
        ]
    }
    assert serde_features_ok(unit_graph) is False
```

- [ ] **Step 3: Run the tests, confirm they fail for the right reason (the exact-equality bug, not a missing fixture)**

Run: `python3 -m pytest tests/test_target_analysis_promotion.py -v -k serde_features`
Expected: `test_serde_features_ok_is_true_for_patched_features` FAILS (once the fixture is regenerated in Step 4 to real shape) because the current exact-equality check rejects the implied `serde_derive` feature; `test_serde_features_ok_checks_serde_core_specifically` FAILS because the current function never inspects `serde_core` at all.

- [ ] **Step 4: Regenerate the fixtures from real captured data (Step 1's output)**

Replace `tests/fixtures/target_analysis/unit_graph_serde_default.json` and `tests/fixtures/target_analysis/unit_graph_serde_patched.json` with the real JSON captured in Step 1 (trim to just the `units` array entries needed for `serde`/`serde_core`/one other real crate, matching this file's existing minimal-fixture style — do not hand-edit feature lists; use exactly what real cargo produced).

- [ ] **Step 5: Fix `serde_features_ok()`**

In `scripts/target_analysis_promotion.py`, replace:
```python
def serde_features_ok(unit_graph: dict[str, Any]) -> bool:
    units = unit_graph_units_named(unit_graph, "serde")
    if not units:
        return False
    expected = set(STAGE_B2_SERDE_FEATURES)
    return all(set(unit.get("features", [])) == expected for unit in units)
```
with:
```python
def serde_features_ok(unit_graph: dict[str, Any]) -> bool:
    for crate_name in ("serde", "serde_core"):
        units = unit_graph_units_named(unit_graph, crate_name)
        if not units:
            return False
        if any("std" in set(unit.get("features", [])) for unit in units):
            return False
    return True
```
This checks absence of `std` on both real carrier crates, not exact equality to a literal feature set — robust to any other implied feature cargo adds, and correctly scoped to the crate that actually carries `std` in modern serde.

- [ ] **Step 6: Run the tests, confirm GREEN**

Run: `python3 -m pytest tests/test_target_analysis_promotion.py -v`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/target_analysis_promotion.py tests/fixtures/target_analysis/unit_graph_serde_default.json tests/fixtures/target_analysis/unit_graph_serde_patched.json tests/test_target_analysis_promotion.py
git commit -m "fix: serde_features_ok checked serde_core and exact-set equality wrong -- both meant a correct Stage B2 mutation could never satisfy this check"
```

- [ ] **Step 8: Push and re-run, record whether stage-b2 now promotes for real**

Push and confirm: with Task 5's manifest fix and this task's postcondition fix both live, real `promotion-stage-b2-<target>.json` output shows `promotes: true` for at least some real targets where the build gets far enough to resolve `serde`/`serde_core` (this was 0/119 across all 7 audited runs prior to this task).

---

### Task 7: Add a genuine within-run unmutated-vs-mutated Stage C comparison

**Real, confirmed defect (C1 and C2), the largest structural gap.** `secondary-stage-c-and-promotion` checks out the repo, downloads the mutation patch, and applies it (`git apply mutation/full-mutation.patch`) *before* Stage C ever runs — there is no point in this job's lifecycle where a pristine, unmutated checkout is available to compare against. Instead, Stage C's "baseline" is `prior_round[...][target]["sibling_sites"]` — the *previous round's already-mutated* output. Since `secondary-mutate` never reads the prior round's baseline (`needs: [discovery, indexing]` only) and Stages A/B/B2/B3 are deterministic and unconditional, round N+1's mutated tree is byte-for-byte identical to round N's, forever — confirmed directly: real data across all 7 audited runs shows `baseline_site_count == sibling_site_count` exactly, set-identical, not merely numerically equal, for every one of 833 real target-measurements. **The comparison the pipeline actually computes is a mathematical fixed point by construction; it was never capable of detecting progress, independent of any toolchain issue.**

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (`secondary-stage-c-and-promotion` job: add a pre-mutation Stage C pass; change the per-target `depth_advanced` computation's baseline source)

**Interfaces:**
- Consumes: nothing new.
- Produces: `out/baseline-stage-c-$TARGET.json` (new — the real, unmutated-tree compiler output for this target, captured fresh every run); `depth-decision-$TARGET.json`'s `baseline_sites`/`baseline_site_count` now reflect a genuine within-run comparison instead of the prior round's mutated output.

- [ ] **Step 1: Add a pre-mutation Stage C baseline pass, before the patch is applied**

In `.github/workflows/target-analysis-pipeline.yml`, in `secondary-stage-c-and-promotion`, insert a new step between `- uses: actions/checkout@v4` and `- name: Download the Secondary layer's mutation artifact`:
```yaml
      - name: Install nightly Rust
        run: |
          rustup toolchain install nightly --profile minimal
          rustup component add rust-src --toolchain nightly
      - name: Stage C baseline -- check the UNMUTATED tree for every target in this batch
        env:
          BATCH_TARGETS: ${{ toJSON(fromJSON(needs.discovery.outputs.batches)[matrix.batch_index]) }}
        run: |
          mkdir -p baseline-out
          CHECK_PACKAGES=$(python3 -c 'import sys; sys.path.insert(0, "."); from scripts.target_analysis_promotion import MUTATED_LIBRARY_CRATES; print(" ".join(f"-p {c}" for c in MUTATED_LIBRARY_CRATES))')
          echo "$BATCH_TARGETS" | python3 -c 'import sys; sys.path.insert(0, "."); from scripts.target_analysis_discovery import expand_batch_json; print(expand_batch_json(sys.stdin.read()))' > targets-in-batch.txt
          while IFS= read -r TARGET; do
            echo "=== Stage C baseline: target=$TARGET ==="
            cargo +nightly check $CHECK_PACKAGES --target "$TARGET" \
              -Z build-std=core,alloc --keep-going --message-format=json \
              > "baseline-out/stage-c-$TARGET.json" || true
          done < targets-in-batch.txt
```
Note: no `rustup target add` here, deliberately -- `-Z build-std=core,alloc` builds `core`/`alloc` from source via the `rust-src` component (already added by this job, `rustup component add rust-src`), which is a completely different mechanism from the prebuilt-sysroot lookup `rustup target add` provides. Task 1's fix applies only to `build-attempt`/`dependency-graph`, whose commands run without any `-Z build-std` flag and so do need the prebuilt sysroot.

(This duplicates the "Install nightly Rust" step that already exists later in the job -- remove the later, now-redundant copy in Step 2 below, since this baseline pass needs the toolchain before the mutation-patch download too.)

- [ ] **Step 2: Remove the now-duplicate "Install nightly Rust" step later in the job**

In the same job, delete the second, now-redundant `- name: Install nightly Rust` step that currently appears after `Apply the mutation patch to this job's checkout` (the toolchain is already installed by Step 1 above, and installing it twice is wasted, though harmless, work).

- [ ] **Step 3: Change the per-target `depth_advanced` computation to use this run's own real baseline**

In the same file, in the final per-target Python heredoc (the one computing `depth_advanced`), replace:
```python
          sibling_messages = load_jsonl(Path(f"out/stage-c-{target}.json"))
          prior_round = load_json(Path("prior-round-baseline/round-baseline.json"))
          prior_record = (
              prior_round.get("promoted", {}).get(target)
              or prior_round.get("preserved_not_promoted", {}).get(target)
          )
          baseline_sites = (
              {tuple(s) for s in prior_record["sibling_sites"]} if prior_record is not None else set()
          )
          sibling_sites = error_sites(sibling_messages)
```
with:
```python
          sibling_messages = load_jsonl(Path(f"out/stage-c-{target}.json"))
          baseline_messages = load_jsonl(Path(f"baseline-out/stage-c-{target}.json"))
          baseline_sites = error_sites(baseline_messages)
          sibling_sites = error_sites(sibling_messages)
```
This is now a genuine, real, within-run unmutated-vs-mutated comparison — the primary, load-bearing progress signal. The cross-run `prior-round-baseline` mechanism is not deleted in this task (see Task 10, which explicitly re-scopes it as a separate, secondary, honestly-labeled longitudinal signal rather than removing working infrastructure).

- [ ] **Step 4: Validate YAML and heredoc syntax locally**

Run:
```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/target-analysis-pipeline.yml')); print('YAML OK')"
actionlint .github/workflows/target-analysis-pipeline.yml
```

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "fix: Stage C now compares against a real, freshly-captured unmutated-tree baseline every run, instead of the prior round's already-mutated output -- the comparison was a fixed point by construction"
```

- [ ] **Step 6: Push and verify on a real runner -- this roughly doubles this job's cargo-check work per batch**

Push and confirm: `baseline-out/stage-c-$TARGET.json` and `out/stage-c-$TARGET.json` are both real, distinct compiler-message captures (the unmutated and mutated tree respectively) for every target; job duration for `secondary-stage-c-and-promotion` roughly doubles compared to the pre-fix real baseline (~10 minutes/batch measured across 7 audited runs) -- this is an honest, necessary cost of computing a real comparison, not a regression to investigate away. Re-run the golden fixture (Task 2) and confirm its check now genuinely compares a fresh unmutated build of `larql-nostd-canary` against its mutated form within this same run.

---

### Task 8: Make spanless errors and build-script failures count as real signal instead of disappearing

**Real, confirmed defect (C4), plus an independently-found deeper case.** `error_sites()` only records a site when an error-level message has a primary span. Real data: `x86_64-unknown-linux-gnu`'s Stage C output has 51 real error-level messages, all spanless (`duplicate lang item in crate 'core' (which 'std' depends on): 'sized'`, the classic `-Z build-std` × prebuilt-sysroot collision) -- the recorded verdict is `sibling_site_count: 0`, byte-identical to what a genuinely clean, successful build would produce. Deeper still: a real build-script failure (confirmed directly, `larql-compute`'s own `csrc/q4_dot.c` native build failing on `aarch64-linux-android` for real: `error: failed to run custom build command for 'larql-compute v0.2.0' ... process didn't exit successfully ... exit status: 1`) doesn't appear in the compiler-message JSON stream *at all* -- it is raw, non-JSON stderr text, invisible to `error_sites()`/`count_errors_by_target()` no matter how they're extended, since Stage C's current invocation only redirects stdout (`> "out/stage-c-$TARGET.json"`, no `2>&1`).

**Files:**
- Modify: `scripts/target_analysis_common.py` (`error_sites`)
- Modify: `.github/workflows/target-analysis-pipeline.yml` (Stage C's invocation: also capture stderr to a separate file; the `depth_advanced` heredoc: fold build-script failures into the result)
- Modify: `tests/test_target_analysis_common.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `error_sites()` now returns a non-empty set for spanless errors (keyed `("<spanless>", -1, code)` instead of vanishing); `depth-decision-$TARGET.json` gains a new `build_script_failures: list[str]` field.

- [ ] **Step 1: Write the failing test for spanless-error handling (TDD)**

In `tests/test_target_analysis_common.py`, add:
```python
def test_error_sites_keeps_spanless_errors_instead_of_dropping_them():
    # Real case: a `duplicate lang item` error under -Z build-std has no
    # primary span at all -- dropping it silently reads as a clean build.
    messages = [
        {"reason": "compiler-message", "message": {"level": "error", "code": {"code": "E0152"}, "message": "duplicate lang item in crate 'core' (which 'std' depends on): 'sized'", "spans": []}},
    ]
    sites = error_sites(messages)
    assert len(sites) == 1
    assert sites == {("<spanless>", -1, "E0152")}
```

- [ ] **Step 2: Run the test, confirm it fails for the right reason**

Run: `python3 -m pytest tests/test_target_analysis_common.py -v -k spanless`
Expected: FAIL — current `error_sites()` returns an empty set (`len(sites) == 0`), not the asserted spanless entry.

- [ ] **Step 3: Fix `error_sites()`**

In `scripts/target_analysis_common.py`, replace:
```python
def error_sites(compiler_messages: list[dict[str, Any]]) -> set[tuple[str, int, str]]:
    sites: set[tuple[str, int, str]] = set()
    for _entry, message in error_level_messages(compiler_messages):
        code = (message.get("code") or {}).get("code") or message.get("message", "")[:60]
        for span in message.get("spans", []):
            if span.get("is_primary"):
                sites.add((span.get("file_name", ""), span.get("line_start", -1), code))
    return sites
```
with:
```python
def error_sites(compiler_messages: list[dict[str, Any]]) -> set[tuple[str, int, str]]:
    sites: set[tuple[str, int, str]] = set()
    for _entry, message in error_level_messages(compiler_messages):
        code = (message.get("code") or {}).get("code") or message.get("message", "")[:60]
        primary_spans = [s for s in message.get("spans", []) if s.get("is_primary")]
        if primary_spans:
            for span in primary_spans:
                sites.add((span.get("file_name", ""), span.get("line_start", -1), code))
        else:
            sites.add(("<spanless>", -1, code))
    return sites
```

- [ ] **Step 4: Run the test, confirm GREEN, and confirm the existing spanned-error test still passes unmodified**

Run: `python3 -m pytest tests/test_target_analysis_common.py -v`
Expected: all pass, including `test_error_sites_extracts_only_error_level_primary_spans` (unaffected, since it only exercises messages that already have primary spans).

- [ ] **Step 5: Capture stderr from both Stage C passes (baseline and sibling) into separate files**

In `.github/workflows/target-analysis-pipeline.yml`, in Task 7's new baseline-pass step, change:
```bash
            cargo +nightly check $CHECK_PACKAGES --target "$TARGET" \
              -Z build-std=core,alloc --keep-going --message-format=json \
              > "baseline-out/stage-c-$TARGET.json" || true
```
to:
```bash
            cargo +nightly check $CHECK_PACKAGES --target "$TARGET" \
              -Z build-std=core,alloc --keep-going --message-format=json \
              > "baseline-out/stage-c-$TARGET.json" 2> "baseline-out/stage-c-$TARGET.stderr.txt" || true
```
And in the main (post-mutation) Stage C invocation, change:
```bash
            cargo +nightly check $CHECK_PACKAGES --target "$TARGET" \
              -Z build-std=core,alloc --keep-going --message-format=json \
              > "out/stage-c-$TARGET.json" || true
```
to:
```bash
            cargo +nightly check $CHECK_PACKAGES --target "$TARGET" \
              -Z build-std=core,alloc --keep-going --message-format=json \
              > "out/stage-c-$TARGET.json" 2> "out/stage-c-$TARGET.stderr.txt" || true
```

- [ ] **Step 6: Fold build-script failures into the depth_advanced computation**

In the same file, in the final per-target Python heredoc, after computing `baseline_sites`/`sibling_sites` (per Task 7 Step 3), add:
```python
          def build_script_failures(stderr_path):
              text = Path(stderr_path).read_text(encoding="utf-8") if Path(stderr_path).exists() else ""
              return sorted({line.strip() for line in text.splitlines() if "failed to run custom build command for" in line})

          baseline_build_failures = build_script_failures(f"baseline-out/stage-c-{target}.stderr.txt")
          sibling_build_failures = build_script_failures(f"out/stage-c-{target}.stderr.txt")
```
Then extend the `result` dict (which currently ends with `"new_sites": sorted(...)`) to also include:
```python
              "baseline_build_script_failures": baseline_build_failures,
              "sibling_build_script_failures": sibling_build_failures,
```
This does not change `depth_advanced`'s own boolean semantics in this task (that would be a further design decision -- e.g. should a target with zero site-level errors but a real, unresolved build-script failure count as "not clean"? -- deliberately left to a follow-up, flagged in the Self-Review below, so this task stays focused on making the failure *visible* rather than also redefining the promotion rule's semantics in the same change).

- [ ] **Step 7: Validate YAML and heredoc syntax, and commit**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/target-analysis-pipeline.yml')); print('YAML OK')"
actionlint .github/workflows/target-analysis-pipeline.yml
git add scripts/target_analysis_common.py tests/test_target_analysis_common.py .github/workflows/target-analysis-pipeline.yml
git commit -m "fix: spanless compiler errors and build-script failures no longer vanish from the promotion/depth-advancement signal"
```

- [ ] **Step 8: Push and verify on a real runner**

Push and confirm: `x86_64-unknown-linux-gnu`'s real `sibling_site_count` is now non-zero (previously 0, masking 51 real errors); `larql-compute`'s real build-script failure on `aarch64-linux-android` (if Task 1's `rustup target add` doesn't already resolve it, since that's a `core`-availability fix, not a C-toolchain fix) now appears in `sibling_build_script_failures` instead of being silently absent from every field in the record.

---

### Task 9: Re-run the golden fixture on real CI and confirm it passes -- this plan's acceptance test

**Files:** none (verification-only task).

**Interfaces:** none new.

- [ ] **Step 1: Push the accumulated commits from Tasks 1-8 if not already pushed, and trigger a real run**

```bash
git push fork sdd/target-analysis-pipeline:experiment/target-analysis-pipeline
sleep 8
RUN_ID=$(gh run list --repo metavacua/larql-to-sparql --branch experiment/target-analysis-pipeline --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$RUN_ID" --repo metavacua/larql-to-sparql --exit-status
```
(If `gh run watch` exceeds a 10-minute tool timeout, poll instead: `gh run view "$RUN_ID" --repo metavacua/larql-to-sparql --json status,conclusion` in a loop until `status` is `completed` -- this is a known, already-encountered harness limitation in this project, not a real failure.)

- [ ] **Step 2: Download the real `secondary-layer-self-test` job's log and the golden-fixture check's real output**

```bash
JOB_ID=$(gh api "/repos/metavacua/larql-to-sparql/actions/runs/$RUN_ID/jobs" --paginate --jq '.jobs[] | select(.name | test("self-test")) | .id')
gh api "/repos/metavacua/larql-to-sparql/actions/jobs/$JOB_ID/logs" | grep -A5 "Golden fixture"
```

- [ ] **Step 3: Confirm the golden-fixture check now passes**

Expected real output: `Golden fixture passed: the known-good canary compiles cleanly under -Z build-std=core,alloc.` If it still fails, read the real captured `canary-check.json` from the job's artifact and diagnose against this plan's own tasks -- do not proceed to Task 10 until this genuinely passes on real CI. This is the plan's own acceptance criterion: a green golden fixture is required evidence that Tasks 1-8 collectively restored the measurement's capacity to report a correct result, not merely that each task's own narrower local check passed.

- [ ] **Step 4: Ledger the result**

Record in `.superpowers/sdd/2026-08-16-target-analysis-pipeline/progress.md`: the real run ID, the golden fixture's real pass/fail history across this plan's pushes (Task 2's RED, and the real reason it changed or didn't after each of Tasks 4-8), and the final confirmed GREEN.

---

### Task 10: Re-scope the cross-round baseline handoff as an explicit, separate longitudinal-drift signal

**Rationale.** Task 7 gives the pipeline a genuine within-run progress signal for the first time. The existing cross-run `fetch-prior-round-baseline`/`next-round-baseline` mechanism (Tasks 20-21 of the original plan) is real, working infrastructure -- it should not be deleted -- but per the spec's own Principle 4 ("no aggregation step collapses one probe's verdict into another's"), it must not be presented as equivalent to the new within-run comparison. Its actual, honest purpose going forward: detecting drift across calendar time (toolchain updates, registry version bumps) between otherwise-identical mutated-tree checks, a genuinely different question from "did this round's mutation improve on this round's own baseline."

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (`next-round-baseline`'s fold step: rename the folded field to make its scope explicit)
- Modify: `docs/superpowers/specs/2026-08-16-target-analysis-pipeline-design.md` (dated correction, matching this project's own established convention)

**Interfaces:**
- Consumes: nothing new.
- Produces: `round-baseline.json`'s per-target records gain an explicit `signal_type: "cross_round_drift"` field, distinguishing them from the within-run `depth-decision-<target>.json` records' implicit `signal_type: "within_run_mutation_effect"`.

- [ ] **Step 1: Label the cross-round fold's records explicitly**

In `.github/workflows/target-analysis-pipeline.yml`, in `next-round-baseline`'s fold step, change:
```python
                  record = {**depth_decision, "stage_promotions": stage_verdicts}
```
to:
```python
                  record = {**depth_decision, "stage_promotions": stage_verdicts, "signal_type": "cross_round_drift"}
```

- [ ] **Step 2: Add a dated correction to the spec, matching this project's own established convention**

In `docs/superpowers/specs/2026-08-16-target-analysis-pipeline-design.md`, immediately after the passage the 2026-08-21 review cited as already correctly anticipating this ("Every stage's effect is captured as a before/after diff... not inferred from Stage C's pass/fail alone"), add:
```
**Corrected 2026-08-21 (independent review + this plan):** this guardrail was
honored for Stage B/B2/B3 but not for Stage C, whose "baseline" was wired to
the *prior round's own mutated output* rather than a real unmutated
checkout -- a fixed point by construction, confirmed via real data across 7
runs and 833 target-measurements (0 ever showing progress). Stage C now
computes a genuine within-run unmutated-vs-mutated comparison (this plan's
Task 7); the pre-existing cross-run baseline handoff is retained as a
separate, explicitly-labeled longitudinal drift signal
(`signal_type: "cross_round_drift"`), never merged with the within-run
result, per Standing Principle 4.
```

- [ ] **Step 3: Validate and commit**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/target-analysis-pipeline.yml')); print('YAML OK')"
actionlint .github/workflows/target-analysis-pipeline.yml
git add .github/workflows/target-analysis-pipeline.yml docs/superpowers/specs/2026-08-16-target-analysis-pipeline-design.md
git commit -m "docs+fix: explicitly label the cross-round baseline as a longitudinal drift signal, distinct from the new within-run mutation-effect comparison"
```

- [ ] **Step 4: Push and verify on a real runner**

Push and confirm: `round-baseline.json`'s real records carry `signal_type: "cross_round_drift"`; `depth-decision-<target>.json` records (read directly, not via the cross-round fold) remain the primary, within-run signal.

---

### Task 11: Extend blast-radius containment to Stage A and Stage B2

**Real gap (finding I4).** The spec requires blast-radius containment "per stage." Only Stage B is checked (`secondary-layer-self-test`'s existing "Blast-radius containment" step). Stage A (`clippy --fix` across five crates, now including the golden fixture) and Stage B2 (a `sed -i -E` over every `crates/*/Cargo.toml`, plus, as of Task 5, the root `Cargo.toml`) are the two stages with genuinely unbounded blast radius, and neither is covered.

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (`secondary-layer-self-test`)

**Interfaces:**
- Consumes: `MUTATED_LIBRARY_CRATES`, `STAGE_B2_SERDE_FEATURES` (unchanged).
- Produces: two new self-test steps; no schema change.

- [ ] **Step 1: Add a Stage A blast-radius check**

In `secondary-layer-self-test`, add a new step before the existing "Blast-radius containment: assert Stage B only touches its declared scope" step:
```yaml
      - name: "Blast-radius containment: assert Stage A only touches its declared crates"
        run: |
          CRATES=$(python3 -c 'import sys; sys.path.insert(0, "."); from scripts.target_analysis_promotion import MUTATED_LIBRARY_CRATES; print(" ".join(MUTATED_LIBRARY_CRATES))')
          for CRATE in $CRATES; do
            cargo +nightly clippy --fix --allow-dirty --allow-staged \
              -p "$CRATE" --lib -- -W clippy::std_instead_of_core -W clippy::std_instead_of_alloc -W clippy::alloc_instead_of_core
          done
          CHANGED=$(git diff --name-only)
          EXPECTED_FILES=$(python3 -c 'import sys; sys.path.insert(0, "."); from scripts.target_analysis_promotion import MUTATED_LIBRARY_CRATES; print("\n".join(f"crates/{c}/src/lib.rs" for c in MUTATED_LIBRARY_CRATES))')
          for f in $CHANGED; do
            if ! echo "$EXPECTED_FILES" | grep -qx "$f"; then
              echo "::error::Stage A touched a file outside its declared scope: $f"
              exit 1
            fi
          done
          echo "Blast radius contained to the declared crates' lib.rs files."
```

- [ ] **Step 2: Add a Stage B2 blast-radius check**

Add another new step, before the same existing Stage B check:
```yaml
      - name: "Blast-radius containment: assert Stage B2 only touches its declared scope"
        run: |
          git checkout -- .
          FEATURES_TOML=$(python3 -c 'import sys; sys.path.insert(0, "."); from scripts.target_analysis_promotion import STAGE_B2_SERDE_FEATURES; print(", ".join(f"\"{f}\"" for f in STAGE_B2_SERDE_FEATURES))')
          find crates -name Cargo.toml -exec \
            sed -i -E "s/serde = \{ workspace = true[^}]*\}/serde = { workspace = true, default-features = false, features = [$FEATURES_TOML] }/" {} \;
          sed -i -E "s/serde = \{ version = \"1\", features = \[\"derive\"\] \}/serde = { version = \"1\", default-features = false, features = [$FEATURES_TOML] }/" Cargo.toml
          sed -i -E 's/serde_json = "1"/serde_json = { version = "1", default-features = false }/' Cargo.toml
          CHANGED=$(git diff --name-only)
          for f in $CHANGED; do
            if [[ "$f" != Cargo.toml && "$f" != crates/*/Cargo.toml ]]; then
              echo "::error::Stage B2 touched a file outside its declared scope: $f"
              exit 1
            fi
          done
          echo "Blast radius contained to Cargo.toml manifests only."
          git checkout -- .
```
The `git checkout -- .` calls bracket this step so it doesn't interfere with the Stage A/B/B3 checks that follow it in the same job (each blast-radius check must start from a clean tree).

- [ ] **Step 3: Validate and commit**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/target-analysis-pipeline.yml')); print('YAML OK')"
actionlint .github/workflows/target-analysis-pipeline.yml
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: extend blast-radius containment to Stage A and Stage B2 -- previously only Stage B was checked"
```

- [ ] **Step 4: Push and verify on a real runner, then deliberately break each new check once to confirm it fails loudly**

Push and confirm both new checks pass as currently written. Then, in a throwaway commit, deliberately widen Stage A's or Stage B2's real mutation to touch an extra file, confirm the corresponding check fails loudly, then revert the throwaway commit -- matching the precedent already established for Stage B's own blast-radius check.

---

### Task 12: Normalize site keys to strip absolute registry paths

**Real gap (finding I2).** 6,608 of 6,616 real sites recorded for `aarch64-apple-darwin` are absolute registry paths (`/home/runner/.cargo/registry/src/.../serde_core-1.0.228/src/...`). Any patch-version bump in any erroring crate rewrites every one of that crate's site keys, and `depth_advanced` would read `true` across the board -- attributing pure dependency-version drift to the mutation. This is a false-positive mode distinct from (and additional to) the false-negative fixed-point defect Task 7 already fixed.

**Files:**
- Modify: `scripts/target_analysis_common.py` (`error_sites`)
- Modify: `tests/test_target_analysis_common.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `error_sites()`'s file-name component is now a registry-relative path (`crate-name-version/src/path.rs` stripped to `src/path.rs`, or the crate's own workspace-relative path for first-party crates, unchanged) instead of an absolute, machine-specific path.

- [ ] **Step 1: Write the failing test (TDD)**

In `tests/test_target_analysis_common.py`, add:
```python
def test_error_sites_normalizes_absolute_registry_paths():
    messages = [
        {"reason": "compiler-message", "message": {"level": "error", "code": {"code": "E0463"}, "message": "x", "spans": [{"file_name": "/home/runner/.cargo/registry/src/index.crates.io-abc123/serde_core-1.0.228/src/lib.rs", "line_start": 5, "is_primary": True}]}},
    ]
    sites = error_sites(messages)
    assert sites == {("serde_core/src/lib.rs", 5, "E0463")}
```

- [ ] **Step 2: Run the test, confirm it fails for the right reason**

Run: `python3 -m pytest tests/test_target_analysis_common.py -v -k normalizes_absolute`
Expected: FAIL — current output includes the full absolute path, not the normalized `serde_core/src/lib.rs`.

- [ ] **Step 3: Add a path-normalization helper and wire it into `error_sites()`**

In `scripts/target_analysis_common.py`, add:
```python
import re


def _normalize_site_path(file_name: str) -> str:
    match = re.search(r"registry/src/[^/]+/([^/]+)-\d[^/]*/(.*)", file_name)
    if match:
        crate_name_no_version = match.group(1)
        rest = match.group(2)
        return f"{crate_name_no_version}/{rest}"
    return file_name
```
Then in `error_sites()`, change:
```python
                sites.add((span.get("file_name", ""), span.get("line_start", -1), code))
```
to:
```python
                sites.add((_normalize_site_path(span.get("file_name", "")), span.get("line_start", -1), code))
```

- [ ] **Step 4: Run the test, confirm GREEN, and confirm all existing `error_sites` tests still pass**

Run: `python3 -m pytest tests/test_target_analysis_common.py -v`
Expected: all pass, including the pre-existing `test_error_sites_extracts_only_error_level_primary_spans` (its fixture uses workspace-relative `crates/larql-boundary/src/lib.rs` paths, which the regex does not match, so `_normalize_site_path` returns them unchanged).

- [ ] **Step 5: Commit**

```bash
git add scripts/target_analysis_common.py tests/test_target_analysis_common.py
git commit -m "fix: normalize absolute registry paths out of site keys -- a dependency patch-version bump was rewriting every one of that crate's site keys, producing false-positive progress"
```

- [ ] **Step 6: Push and verify on a real runner**

Push and confirm: real site keys for third-party dependency errors (e.g. `serde_core`) now read as `serde_core/src/....rs` rather than an absolute, runner-specific path.

---

### Task 13: Add a real runner-capability probe, then wire the genuinely installable runner dependencies for the ~57%-bucket target families from what it discovers

**Real, exhaustively-researched finding (2026-08-21 audit).** Across all 119 real targets: ~57% (68 targets — linux-gnu, linux-musl, windows-gnu/gnullvm, android, most of wasm, freebsd, ohos, fuchsia, netbsd, redox, uefi) have a real, documented, standard installation method on `ubuntu-latest`; ~14% (17 — Apple family, windows-msvc) require a different runner OS or sit in commercial-toolchain/licensing-gray territory; ~29% (34 — bare-metal/no-std targets, nvptx, wasm32v1-none) are structurally blocked regardless of toolchain, since `std` has no implementation there at all. This task wires the cheapest, highest-value real fixes from the first bucket; it explicitly does not attempt the second or third.

**Design ruling (user-directed, two messages, 2026-08-21): discover the runner's real capabilities once, mechanically, and let downstream jobs consume that discovery — rather than each job independently assuming or blindly attempting the same fix.** The user asked directly whether install/misconfiguration issues "can be resolved by the runners themselves in the workflows," then followed up: "there seems like more than enough information for the runners to pull the information they need to discover runner-configuration issues and update the workflow dynamically for downstream jobs." This task's design answers both messages concretely, with one constraint carried over unchanged from the rest of this plan: **the capability manifest this task produces gates provisioning only — it must never decide which targets get probed or which measurements run.** Standing Principle 3 requires every relevant probe to run unconditionally, every time; an absent toolchain is a real, recorded outcome for that target (a `ToolNotFound` build-script failure, same as today), not a reason to skip or relabel the measurement. What changes is only whether the pipeline additionally *tries to fix* the gap before that measurement runs, and whether it tries just once (mechanically, with evidence) instead of guessing three times over.

Concretely: a new job, `runner-capability-probe`, runs once per workflow run, independent of the target matrix, and mechanically discovers two real facts about the actual runner image this run landed on — whether `ANDROID_NDK_HOME` really points at a real clang, and whether `gcc-mingw-w64-x86-64` has a real installable apt candidate — publishing both as job outputs and as an uploaded artifact (real L1 data: if a future `ubuntu-latest` image drops the NDK, this artifact's history shows exactly which run it happened on, which is the same kind of longitudinal signal Task 10's `cross_round_drift` relabeling already relies on). `build-attempt`, `dependency-graph`, and `secondary-stage-c-and-promotion` each consume that discovery instead of re-deriving it, and their own installs remain per-job (GitHub Actions gives every job, and every matrix instance within a job, its own fresh, unshared VM — there is no way to "install once, all jobs get it" without caching, which the Global Constraint above forbids; the value of discovering-once is not skipping installation, it's skipping the *redundant discovery* and the futile install attempts once the probe has already established a package isn't there).

**The probe itself must be structurally unable to fail** — every command it runs is tolerant (`|| true` / `2>/dev/null`), and it always writes both outputs on every path, using an empty string or `false` to mean "not found," never a job failure. If the probe job failed outright, every job that depends on it would be skipped by default (GitHub's ordinary `needs:` semantics), silently dropping all three downstream jobs' worth of real experimental data for reasons that have nothing to do with the experiment — the same "completeness check as involuntary kill switch" defect class as I1, reintroduced in a new job if this invariant is skipped.

**Files:**
- Modify: `.github/workflows/target-analysis-pipeline.yml` (add job `runner-capability-probe`; modify `build-attempt`, `dependency-graph`, and `secondary-stage-c-and-promotion`'s `needs:` lists and their "Install nightly Rust" steps)

**Interfaces:**
- Consumes: nothing new.
- Produces: `needs.runner-capability-probe.outputs.android-ndk-clang-dir` (string, empty if absent) and `needs.runner-capability-probe.outputs.mingw-w64-installable` (string `"true"`/`"false"`) — consumed by the three jobs below. Also produces an uploaded artifact `runner-capabilities` (`out/runner-capabilities.json`) for longitudinal record-keeping; nothing downstream reads the artifact itself, only the job outputs.

- [ ] **Step 1: Add the `runner-capability-probe` job**

Insert as a new top-level job (no `needs:` — it has no data dependency on `discovery` or anything else, so it runs from the start of the workflow in parallel with everything else):

```yaml
  runner-capability-probe:
    name: Probe this run's real runner toolchain capabilities
    runs-on: ubuntu-latest
    permissions:
      contents: read   # actions/checkout
    outputs:
      android-ndk-clang-dir: ${{ steps.probe.outputs.android-ndk-clang-dir }}
      mingw-w64-installable: ${{ steps.probe.outputs.mingw-w64-installable }}
    steps:
      - uses: actions/checkout@v4
      - name: Probe real runner capabilities -- structurally cannot fail; absence is data, never a job failure
        id: probe
        run: |
          mkdir -p out
          NDK_DIR=""
          if [ -n "$ANDROID_NDK_HOME" ] && [ -d "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin" ]; then
            NDK_DIR="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin"
          else
            echo "::warning::ANDROID_NDK_HOME is unset or its expected clang directory is missing on this real runner -- the audit's 'ubuntu-latest ships the NDK preinstalled' claim does not hold here."
          fi
          echo "android-ndk-clang-dir=$NDK_DIR" >> "$GITHUB_OUTPUT"

          sudo apt-get update -qq || true
          MINGW_OK="false"
          if apt-cache policy gcc-mingw-w64-x86-64 2>/dev/null | grep -q 'Candidate: [0-9]'; then
            MINGW_OK="true"
          else
            echo "::warning::gcc-mingw-w64-x86-64 has no installable apt candidate on this runner (or apt-get update failed) -- windows-gnu targets will keep their real ToolNotFound errors."
          fi
          echo "mingw-w64-installable=$MINGW_OK" >> "$GITHUB_OUTPUT"

          printf '{"android_ndk_clang_dir": "%s", "mingw_w64_installable": %s}\n' "$NDK_DIR" "$MINGW_OK" > out/runner-capabilities.json
      - name: Upload the real runner-capability manifest (L1 data: what this run's own runner image actually had)
        uses: actions/upload-artifact@v4
        with:
          name: runner-capabilities
          path: out/runner-capabilities.json
```

Note the tightened apt-cache check: `grep -q 'Candidate: [0-9]'`, not a bare `grep -q "Candidate:"` — `apt-cache policy` prints `Candidate: (none)` for a known-but-uninstallable package, and a bare match on the literal `Candidate:` label would misread that as "installable."

- [ ] **Step 2: Wire the three consumer jobs to depend on the probe and consume its outputs**

In `.github/workflows/target-analysis-pipeline.yml`, add `runner-capability-probe` to the `needs:` array of `build-attempt` (currently `[discovery, target-capability, target-independent-checks]`), `dependency-graph` (currently `[discovery, target-independent-checks]`), and `secondary-stage-c-and-promotion` (currently `[discovery, indexing, secondary-mutate, dependency-graph, fetch-prior-round-baseline]`).

In each of those three jobs, immediately after their existing "Install nightly Rust" step, add:

```yaml
      - name: Provision this job's runner from the probed capabilities (gates provisioning only -- never which targets get measured)
        env:
          ANDROID_NDK_CLANG_DIR: ${{ needs.runner-capability-probe.outputs.android-ndk-clang-dir }}
          MINGW_W64_INSTALLABLE: ${{ needs.runner-capability-probe.outputs.mingw-w64-installable }}
        run: |
          if [ -n "$ANDROID_NDK_CLANG_DIR" ] && [ -d "$ANDROID_NDK_CLANG_DIR" ]; then
            echo "PATH=$ANDROID_NDK_CLANG_DIR:$PATH" >> "$GITHUB_ENV"
            echo "Android NDK confirmed present at $ANDROID_NDK_CLANG_DIR (per this run's own probe); cc-rs now wired to its clang."
          else
            echo "::warning::this job's own runner does not have the NDK clang dir the probe reported (image skew between jobs in this same run, or the probe found none) -- android targets keep their real ToolNotFound errors for this job."
          fi

          if [ "$MINGW_W64_INSTALLABLE" = "true" ]; then
            sudo apt-get update -qq && sudo apt-get install -y gcc-mingw-w64-x86-64 gcc-mingw-w64-i686 \
              || echo "::warning::mingw-w64 install failed on this job's own runner despite the probe finding a candidate -- transient apt issue, continuing; windows-gnu targets keep their real ToolNotFound errors for this job"
          else
            echo "::warning::probe found no installable mingw-w64 candidate on this run's runner image -- skipping the install attempt for this job; windows-gnu targets keep their real ToolNotFound errors"
          fi
```

The re-check (`[ -d "$ANDROID_NDK_CLANG_DIR" ]`) before wiring PATH matters even though the probe already confirmed it: GitHub rolls runner-image updates progressively, so two jobs in the same workflow run can land on different image versions — the probe's output is a hint this step verifies locally, never a blind trust. Both branches use `$GITHUB_ENV`/`$GITHUB_OUTPUT` (not plain shell exports), matching this pipeline's own established pattern (e.g. the `CRATES`/`fmt-crates` derivation, Task 20/21's `/simplify` pass). Neither branch ever `exit 1`s — a loud diagnostic either way, never a reason to abort the batch and lose real data for every other target.

- [ ] **Step 3: Confirm `rustup target add` (Task 1) already covers the wasm family's rustup-installable members**

No new step needed here — `wasm32-unknown-unknown`, `wasm32-wasip1`, `wasm32-wasip1-threads`, `wasm32-wasip2` are plain `rustup target add`-installable (Task 1 already covers this); only `wasm32-unknown-emscripten` needs a separate SDK (deliberately out of scope for this task — flagged below).

- [ ] **Step 4: Validate and commit**

```bash
python3 -c "import yaml; yaml.safe_load(open('.github/workflows/target-analysis-pipeline.yml')); print('YAML OK')"
actionlint .github/workflows/target-analysis-pipeline.yml
git add .github/workflows/target-analysis-pipeline.yml
git commit -m "feat: add a real runner-capability probe job and wire cc-rs/mingw-w64 provisioning to its discovery -- removes two real, irrelevant classes of native-toolchain noise from the error signal without hardcoding an unverified assumption in three places"
```

- [ ] **Step 5: Push and verify on a real runner**

Push and confirm, from the real job logs (not assumed): first, download the `runner-capabilities` artifact and report both fields exactly as discovered on this real run — whether the NDK was found and whether mingw-w64 had an installable candidate. If the NDK was found and each consumer job's own local re-check also confirmed it: confirm real Stage C / build-attempt output for `aarch64-linux-android` no longer shows `ToolNotFound: aarch64-linux-android-clang`, and confirm `larql-compute`'s own native build (the `csrc/q4_dot.c` kernel) now succeeds on that target specifically (previously a real, confirmed build-script failure). If the NDK was not found, or a consumer job's local re-check disagreed with the probe: report this as a real, open finding for the task reviewer and controller to adjudicate — it means the audit's citation does not hold for this runner image (or there is real image skew across jobs within one run), and Task 13's Android-NDK fix needs a different approach (e.g. a dedicated NDK-install action) before it can be considered complete for that family. Independently of the NDK outcome, confirm real Stage C / build-attempt output for `x86_64-pc-windows-gnu` no longer shows `ToolNotFound: aarch64-w64-mingw32-clang` if the probe reported mingw-w64 as installable.

---

## Self-Review

**Spec coverage** — every spec section this plan touches maps to a task:
- Testing / golden fixtures ("a deliberately known, planted outcome... run through the full Stage A→C pipeline") → Task 2, previously entirely unbuilt.
- Error handling / "every stage's effect captured as a before/after diff... not inferred from Stage C's pass/fail alone" → Task 7 (the guardrail the spec already stated, honored for B/B2/B3, restored for C).
- Standing Principle 4 (no aggregation collapses distinct signals) → Task 10 (explicit `signal_type` labeling, not a merge).
- Testing / blast-radius "per stage" → Task 11 (previously only Stage B).
- Data flow / retention (cross-run artifact size, already flagged as open in the spec) → not addressed by this plan; the review's finding I3 (134MB `round-baseline.json`, ~330MB/run in `promotion-decision-batch-*` artifacts) is noted here as a real, separate follow-up, not folded into this plan's scope.

**Placeholder scan:** no "TBD"/"TODO" remain; every code block in every task is the actual real content to write, derived from a real captured artifact, a real local repro, or an existing proven function in this codebase — not a description of what to do.

**Type consistency:** `MUTATED_LIBRARY_CRATES` (Task 2) grows from 4 to 5 entries and is consumed identically by Task 4's `-p` derivation, Task 11's blast-radius checks, and the already-existing `secondary-mutate`/`target-independent-checks` sites (Task 20/21's `/simplify` pass) without any signature change — `GOLDEN_FIXTURE_CRATE` is additive, not a breaking change to the existing tuple's consumers. `error_sites()`'s return type (`set[tuple[str, int, str]]`) is unchanged across Tasks 8 and 12 — only the *values* inside the tuple change (a spanless sentinel, a normalized path), never the shape, so every existing consumer (`depth_advanced`, `next-round-baseline`'s fold) keeps working unmodified.

**Follow-ups explicitly flagged, not included in this plan** (real gaps, named so they aren't silently dropped):
- Redefining `depth_advanced`'s own boolean semantics to also account for `build_script_failures` (Task 8 makes the failure visible in the data; it does not change what counts as "progress" — a real, separate design decision).
- `wasm32-unknown-emscripten`'s SDK (`emscripten-core/setup-emsdk`), the Android NDK-shipped-but-unused-for-other-families question, FreeBSD/NetBSD/OpenHarmony/Fuchsia/Redox cross-toolchains (the audit's own ~57% bucket minus what Task 13 covers) — a real, larger body of work, deliberately scoped out of this plan to keep it finite; each is independently actionable using the audit's own family table.
- The user's own earlier "waterfall"/subgraph-branching architectural question (should mutation stages become round-parameterized so the cross-round loop can show *mechanical* advancement, not just drift detection) — Task 10 deliberately re-scopes rather than resolves this; it remains open, parked, and requires its own dedicated design conversation per the standing ruling already recorded in this project's ledger.
- Solaris/illumos (2 + 1 targets) — confirmed no standard, documented installation method exists; left as a permanently-open gap unless a future finding changes this.
