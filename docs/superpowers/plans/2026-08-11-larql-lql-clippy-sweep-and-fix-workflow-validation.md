# larql-lql Clippy Sweep + clippy-fix Workflow Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close `larql-lql`'s wasm32v1-none `cargo clippy -D warnings` gap (13 findings, both feature legs, fully diagnosed below) using the newly-authored `.github/workflows/clippy-fix.yml` for its first-ever real dispatch, cross-checking its output against an independently-derived expected fix, then confirm the crate stays green on both O_wasm and O_native before calling any of it done.

**Architecture:** No new source architecture — this is a diagnose-fix-verify cycle on existing files, plus the first real exercise of a CI mechanism authored but never dispatched. Two of the 13 findings need a manual `#[allow(dead_code)]` edit (clippy's `--fix` cannot auto-apply dead-code suppressions on structs/consts/type-aliases, only import/variable/mut-shaped MachineApplicable suggestions); the remaining 7 are plain unused-import removals `--fix` should handle unattended. `clippy-fix.yml`'s `fix` job proposes a patch via GitHub Actions artifact; `verify-native` gates it with a real native `cargo test`/`--all-targets` run before it's ever applied locally.

**Tech Stack:** Rust (stable toolchain, `wasm32v1-none` target), Cargo workspace, GitHub Actions (`gh` CLI for all dispatch/verification), `git apply` for patch application.

## Global Constraints

- Repo: `metavacua/larql-to-sparql`, branch `gating/larql-cli-wasm-and-safe`, working copy `/home/metavacua/larql-to-sparql-gating-2026-08-10`. `origin` = `metavacua/larql-to-sparql`, `upstream` = `chrishayuk/larql` — never let `gh` resolve to `upstream` (pass no `--repo` flag only after confirming `gh repo view --json nameWithOwner` still resolves to the fork; it does as of this plan's writing, `gh repo set-default` was already run for this clone).
- **No local `cargo build`/`check`/`test`, ever.** `cargo fmt`/`cargo tree` are metadata-only exceptions. `cargo clippy`/`cargo clippy --fix` may run locally one-at-a-time as a narrow exception, but this plan deliberately does NOT use that exception — its whole point is exercising the GitHub-hosted `clippy-fix.yml` mechanism instead.
- **Two independent oracles required before any fix is considered verified**: O_wasm (`cargo clippy --target wasm32v1-none -p larql-lql [--no-default-features] -- -D warnings`, via the existing `larql-cli-gating.yml`) and O_native (real `cargo test`/`--all-targets` via `gh workflow run larql-lql.yml --ref gating/larql-cli-wasm-and-safe`). Never trust one alone; never trust a background-task notification without an independent fresh `gh run view --json status,conclusion` / `gh api` call after it fires.
- **`--fix`'s output is incomplete until an immediate native verification confirms it** — this plan's whole structure (Tasks 1-2 dispatch-and-validate before Task 3 applies anything) exists to enforce that contract, not skip it because the expected fix is already known.
- The default action for any clippy "unused"/"dead code" finding is `#[cfg(not(target_arch = "wasm32"))]` or `#[allow(dead_code)]` (whichever the finding actually calls for — see per-task reasoning below), never deletion, unless deletion is justified in place with specific evidence. Every deletion in this plan (Task 3's 7 import removals) has that evidence already gathered in Task 0 below — don't re-derive it, don't skip verifying `--fix`'s actual patch matches it.
- Scope boundary: this plan closes `larql-lql`'s wasm32 clippy gap and validates `clippy-fix.yml` only. It does not touch task #36 (`larql-cli`'s own ~107 remaining wasm32v1-none files) or `larql-factory`'s clippy gap — both explicitly out of scope here.

### Task 0 (reference, already done — not a checkbox): the 13 findings, fully diagnosed

Both feature legs (`default`, `--no-default-features`) produce the **identical** 13 findings (verified via `gh api repos/metavacua/larql-to-sparql/actions/jobs/{93863169578,93863169698}/logs` against the most recent `larql-cli-gating.yml` run, `31516572184`, commit `bf289d114e1bc4bfc2d83df5d587077a7343fff4`). No genuine compile errors — every finding is `unused import`, `never used`, or `never constructed`.

**Group A (1 finding, `cargo clippy --fix`-applicable) — `crates/larql-lql/src/alloc_prelude.rs:29`:**
`use num_traits::Float;` is unused. Verified empirically: every `.sqrt()`/`.exp()`/`.tanh()`/`.round()`/`.abs()`-shaped call in the crate lives inside `crates/larql-lql/src/relations.rs` or `crates/larql-lql/src/executor/**` — both wholesale-gated `#[cfg(not(target_arch = "wasm32"))]` at their `pub mod` declaration in `lib.rs:21` and `lib.rs:25` respectively (confirmed by reading `lib.rs:19-27` directly — the `#[cfg]` sits on the line directly above the `pub mod`, not caught by a line-anchored grep). The one hit outside those two directories (`lexer.rs:737`) is inside a `#[test]` function, irrelevant to the `--lib` wasm32 build. Zero portable/wasm32-compiled code needs `Float`.

**Group B (6 findings, `cargo clippy --fix`-applicable) — six `parser/*.rs` files, each has an entirely-unused `#[cfg(target_arch = "wasm32")] use crate::alloc_prelude::*;`:**
- `crates/larql-lql/src/parser/introspection.rs:8`
- `crates/larql-lql/src/parser/lifecycle.rs:8`
- `crates/larql-lql/src/parser/mutation.rs:8`
- `crates/larql-lql/src/parser/patch.rs:8`
- `crates/larql-lql/src/parser/query.rs:8`
- `crates/larql-lql/src/parser/trace.rs:22`

Verified empirically (`grep -c 'Vec<\|Vec::\|Box<\|Box::\|\bString\b\|\.to_owned()\|\.to_string()\|ToString'` returns `0` for all six files): none of these six parser files reference `Vec`/`String`/`Box`/`ToString` anywhere, gated or not. The import is genuinely, currently dead.

**Group C (6 findings, NOT `cargo clippy --fix`-applicable — clippy's `--fix` has no machine-applicable suggestion for `dead_code` on a struct/const/type-alias) — all in `crates/larql-lql/src/collections.rs`:**
- Line 28: `unused imports: BTreeMap and BTreeSet` (the `#[cfg(target_arch = "wasm32")] pub use alloc::collections::{BTreeMap, BTreeSet};` re-export)
- Line 31: `type alias HashMap is never used`
- Line 33: `type alias HashSet is never used`
- Line 36: `constant FNV_OFFSET_BASIS is never used`
- Line 38: `constant FNV_PRIME is never used`
- Line 41: `struct FnvHasher is never constructed`

Root cause (read directly from `collections.rs`'s own doc comment, lines 13-18): every `HashMap`/`HashSet`/`BTreeMap`/`BTreeSet` use-site in this crate lives inside `relations.rs`/`executor/**`, both native-only, which reach for `std::collections::` directly rather than this crate-local alias. The wasm32-gated hasher machinery here is deliberately-kept scaffolding — "for parity with every other gated crate and for any future portable code that needs it" — not a mistake. The native side of this exact same file already carries the identical treatment one line above (`#[allow(unused_imports)]` on line 20's native `pub use std::collections::{...}`) for the same reason. The correct fix is `#[allow(dead_code)]` (or `#[allow(unused_imports)]` for the line-28 import specifically), matching that existing precedent — not deletion, and not further `#[cfg]` gating (these items are already correctly scoped to `wasm32`; the gate isn't wrong, the item is just currently unconsumed within that scope).

---

### Task 1: Dispatch `clippy-fix.yml` for `larql-lql` on both feature legs

**Files:** none (CI dispatch only)

**Interfaces:**
- Consumes: `.github/workflows/clippy-fix.yml` — originally `wasm32-clippy-fix.yml`, renamed and generalized (added a `target` input, default `wasm32v1-none`) after this plan was first written, once landing it on `main` (required for `workflow_dispatch` to register at all — a `workflow_dispatch`-only workflow present only on a feature branch is not dispatchable; confirmed via `gh api repos/.../actions/workflows/wasm32-clippy-fix.yml` returning 404 until merged) surfaced the reuse requirement. Also picked up 2 code-review fixes (`actions/upload-artifact`/`download-artifact` bumped v4→v7/v8; a `cargo fmt` step added before the diff is captured, so `--fix`'s output can't fail `verify-native`'s fmt gate for a purely cosmetic reason). All committed and merged before this task dispatches for real.
- Produces: two workflow runs (default, no-default-features), each with a `fix` job artifact `clippy-fix-larql-lql-wasm32v1-none-{default,no-default-features}` containing `clippy-fix.patch` + `clippy-remaining.log`, and a `verify-native` job conclusion, consumed by Task 2.

- [ ] **Step 1: Dispatch the default-features run**

```bash
gh workflow run clippy-fix.yml --ref gating/larql-cli-wasm-and-safe \
  -f crate=larql-lql -f target=wasm32v1-none -f features=default
```

- [ ] **Step 2: Dispatch the no-default-features run**

```bash
gh workflow run clippy-fix.yml --ref gating/larql-cli-wasm-and-safe \
  -f crate=larql-lql -f target=wasm32v1-none -f features=no-default-features
```

- [ ] **Step 3: Get both run IDs**

```bash
gh run list --workflow=clippy-fix.yml --branch gating/larql-cli-wasm-and-safe --limit 2 \
  --json databaseId,status,createdAt
```

Expected: two runs, both `status: queued` or `in_progress`, `createdAt` within seconds of Steps 1-2.

- [ ] **Step 4: Wait for both runs to complete, then independently re-verify — do not trust a notification alone**

```bash
gh run view <default-run-id> --json status,conclusion,jobs
gh run view <no-default-features-run-id> --json status,conclusion,jobs
```

Expected: both runs `status: completed`. Record each run's `fix` and `verify-native` job conclusions — this is the first real data this mechanism has ever produced, so don't assume a shape; read what actually comes back.

---

### Task 2: Validate the proposed patches against the known-expected fix (Group A + Group B only)

**Files:** none (read-only validation)

**Interfaces:**
- Consumes: the two artifacts from Task 1.
- Produces: a go/no-go decision for Task 3, and (if the patch doesn't match expectations) a concrete bug report against `clippy-fix.yml` for iteration.

- [ ] **Step 1: Download both artifacts**

```bash
gh run download <default-run-id> -n clippy-fix-larql-lql-wasm32v1-none-default -D /tmp/lql-fix-default
gh run download <no-default-features-run-id> -n clippy-fix-larql-lql-wasm32v1-none-no-default-features -D /tmp/lql-fix-nodefault
```

- [ ] **Step 2: Confirm both patches touch exactly the 7 Group A/B locations and nothing else**

```bash
grep -c '^diff --git' /tmp/lql-fix-default/clippy-fix.patch
grep -c '^diff --git' /tmp/lql-fix-nodefault/clippy-fix.patch
```

Expected: `7` for both (one diff hunk per file: `alloc_prelude.rs` + the 6 `parser/*.rs` files). If either count is different — higher (touched something unexpected), lower (missed one of the 7), or the patch is empty — **stop, do not proceed to Task 3**. That's a real discrepancy between `--fix`'s actual behavior and this plan's independently-derived expectation (Task 0). Read the patch in full, determine whether the plan's diagnosis or the workflow's behavior is wrong, and fix whichever is actually broken before continuing (if it's the workflow — e.g., a shell-quoting bug, a wrong `cargo clippy` flag — edit `.github/workflows/clippy-fix.yml`, commit, push, and re-run Task 1 for the affected leg).

- [ ] **Step 3: Confirm the two patches are identical to each other**

```bash
diff /tmp/lql-fix-default/clippy-fix.patch /tmp/lql-fix-nodefault/clippy-fix.patch
```

Expected: no output (byte-identical). Group A/B's 7 findings don't depend on feature flags, so both legs' `--fix` runs should have produced the same diff. If they differ, read both patches and understand why before proceeding — this crate may have a feature-gated code path this plan's diagnosis didn't account for.

- [ ] **Step 4: Confirm `clippy-remaining.log` shows exactly the 6 Group C findings, nothing else**

```bash
grep -E 'error:|error\[' /tmp/lql-fix-default/clippy-remaining.log
```

Expected: exactly 6 lines, matching Task 0's Group C list (`BTreeMap`/`BTreeSet` unused import, `HashMap`/`HashSet` type aliases never used, `FNV_OFFSET_BASIS`/`FNV_PRIME` never used, `FnvHasher` never constructed) plus the `could not compile ... due to 6 previous errors` summary line. If Group A/B items still appear here, `--fix` didn't actually remove them from the patch it applied before re-checking — investigate.

- [ ] **Step 5: Check `verify-native`'s conclusion for both runs**

```bash
gh run view <default-run-id> --json jobs --jq '.jobs[] | select(.name | startswith("verify-native")) | {name, conclusion}'
gh run view <no-default-features-run-id> --json jobs --jq '.jobs[] | select(.name | startswith("verify-native")) | {name, conclusion}'
```

Expected: `success` for both — the 7 import removals are genuinely dead code by Task 0's own empirical check, so applying them and running `cargo check --all-targets`/`clippy --no-deps -D warnings`/`cargo test` natively should pass cleanly. If `verify-native` failed, **stop, do not proceed to Task 3** — that means either Task 0's diagnosis was wrong (something in Group A/B is actually load-bearing on native after all) or `clippy-fix.yml` has a bug in its verify step (e.g. the patch didn't apply cleanly, a missing dependency install). Download `verify-native`'s log, find the actual failure, and resolve it before continuing.

---

### Task 3: Apply the validated patch locally and commit

**Files:**
- Modify: `crates/larql-lql/src/alloc_prelude.rs`
- Modify: `crates/larql-lql/src/parser/introspection.rs`
- Modify: `crates/larql-lql/src/parser/lifecycle.rs`
- Modify: `crates/larql-lql/src/parser/mutation.rs`
- Modify: `crates/larql-lql/src/parser/patch.rs`
- Modify: `crates/larql-lql/src/parser/query.rs`
- Modify: `crates/larql-lql/src/parser/trace.rs`

**Interfaces:**
- Consumes: `/tmp/lql-fix-default/clippy-fix.patch` (validated identical to the no-default-features patch in Task 2 Step 3).
- Produces: a commit ready for Task 5's push.

- [ ] **Step 1: Apply the patch**

```bash
cd /home/metavacua/larql-to-sparql-gating-2026-08-10
git apply --check /tmp/lql-fix-default/clippy-fix.patch
git apply /tmp/lql-fix-default/clippy-fix.patch
```

- [ ] **Step 2: Confirm it touched exactly the 7 expected files**

```bash
git status --porcelain crates/larql-lql
```

Expected: exactly 7 lines, matching the Files list above.

- [ ] **Step 3: `cargo fmt --check` sanity (pre-approved local exception, metadata-only)**

```bash
cargo fmt -p larql-lql -- --check
```

Expected: clean (no output). `verify-native` in Task 2 already ran a fmt-check step against this same patch and passed, so this is a redundant final confirmation, not new information.

- [ ] **Step 4: Commit**

```bash
git add crates/larql-lql/src/alloc_prelude.rs crates/larql-lql/src/parser/introspection.rs \
  crates/larql-lql/src/parser/lifecycle.rs crates/larql-lql/src/parser/mutation.rs \
  crates/larql-lql/src/parser/patch.rs crates/larql-lql/src/parser/query.rs \
  crates/larql-lql/src/parser/trace.rs
git commit -m "$(cat <<'EOF'
fix(lql): remove 7 dead wasm32-only imports (via clippy-fix.yml)

alloc_prelude.rs's num_traits::Float and six parser/*.rs files'
alloc_prelude::* glob are all genuinely unused on wasm32v1-none: every
real Float-trait-method call site lives inside relations.rs/executor/**
(both wholesale native-gated), and none of the six parser files
reference Vec/String/Box/ToString at all.

Produced and verified via clippy-fix.yml's first real dispatch
(fix job's patch cross-checked against an independently-derived
expected diff; verify-native job -- real native cargo check/clippy/test
-- passed on both feature legs before this patch was applied).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Hand-fix the 6 non-auto-fixable `collections.rs` findings

**Files:**
- Modify: `crates/larql-lql/src/collections.rs`

**Interfaces:**
- Consumes: nothing beyond the current file content (Task 0's Group C diagnosis).
- Produces: a commit ready for Task 5's push, alongside Task 3's.

- [ ] **Step 1: Apply the following edit** (full replacement of lines 27-41; every other line in the file is unchanged)

Before:
```rust
#[cfg(target_arch = "wasm32")]
pub use alloc::collections::{BTreeMap, BTreeSet};

#[cfg(target_arch = "wasm32")]
pub type HashMap<K, V> = hashbrown::HashMap<K, V, ::core::hash::BuildHasherDefault<FnvHasher>>;
#[cfg(target_arch = "wasm32")]
pub type HashSet<K> = hashbrown::HashSet<K, ::core::hash::BuildHasherDefault<FnvHasher>>;

#[cfg(target_arch = "wasm32")]
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
#[cfg(target_arch = "wasm32")]
const FNV_PRIME: u64 = 0x100000001b3;

#[cfg(target_arch = "wasm32")]
pub struct FnvHasher(u64);
```

After:
```rust
// Currently unused on wasm32 for the same reason the native re-export
// above needs #[allow(unused_imports)]: every HashMap/HashSet/BTreeMap/
// BTreeSet site in larql-lql lives inside relations.rs/executor/ (both
// wholesale-gated #[cfg(not(target_arch = "wasm32"))]) -- kept for
// parity with every other gated crate and for any future portable code
// that needs it, per this file's own top-of-file rationale.
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
pub use alloc::collections::{BTreeMap, BTreeSet};

#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
pub type HashMap<K, V> = hashbrown::HashMap<K, V, ::core::hash::BuildHasherDefault<FnvHasher>>;
#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
pub type HashSet<K> = hashbrown::HashSet<K, ::core::hash::BuildHasherDefault<FnvHasher>>;

#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
const FNV_PRIME: u64 = 0x100000001b3;

#[cfg(target_arch = "wasm32")]
#[allow(dead_code)]
pub struct FnvHasher(u64);
```

- [ ] **Step 2: `cargo fmt --check` sanity**

```bash
cargo fmt -p larql-lql -- --check
```

Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add crates/larql-lql/src/collections.rs
git commit -m "$(cat <<'EOF'
fix(lql): allow(dead_code) on collections.rs's wasm32-only hasher scaffolding

HashMap/HashSet/BTreeMap/BTreeSet/FNV_OFFSET_BASIS/FNV_PRIME/FnvHasher
are all correctly #[cfg(target_arch = "wasm32")]-gated already -- the
gate isn't wrong, the items are just currently unconsumed within that
scope, since every real HashMap/HashSet use-site in this crate lives
inside relations.rs/executor/ (native-only). This file's own doc
comment already documents the scaffolding as deliberately kept for
parity with every other gated crate; matches the #[allow(unused_imports)]
already present one line above on the native-side re-export. Not
auto-fixable by `cargo clippy --fix` (no machine-applicable suggestion
exists for dead_code on a struct/const/type-alias) -- confirmed via
clippy-fix.yml's clippy-remaining.log, which listed
exactly these 6 findings as unresolved after --fix.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Push and re-run O_wasm to confirm `larql-lql` is clean on both feature legs

**Files:** none (CI dispatch + verification)

**Interfaces:**
- Consumes: Task 3 + Task 4's commits.
- Produces: independent confirmation that all 13 findings are resolved on the real gating oracle (not just `clippy-fix.yml`'s narrower per-crate check).

- [ ] **Step 1: Push**

```bash
git push origin gating/larql-cli-wasm-and-safe
```

- [ ] **Step 2: Get the freshly-triggered `larql-cli-gating.yml` run ID**

```bash
gh run list --workflow=larql-cli-gating.yml --branch gating/larql-cli-wasm-and-safe --limit 1 \
  --json databaseId,status
```

- [ ] **Step 3: Wait for completion, then independently re-verify `larql-lql`'s two jobs specifically**

```bash
gh api "repos/metavacua/larql-to-sparql/actions/runs/<run-id>/jobs?per_page=100" \
  --jq '[.jobs[] | select(.name | contains("larql-lql")) | {name, conclusion}]'
```

Expected: `wasm32v1-none / larql-lql / ubuntu-latest / default` and `.../no-default-features` both `success`. (Use the fully-paginated `gh api .../jobs?per_page=100` call, not `gh run view --json jobs`, which silently truncates on large runs — see this branch's own SDD ledger for why that distinction matters.)

---

### Task 6: Dispatch and verify O_native for `larql-lql` — native build and test verification before completion

**Files:** none (CI dispatch + verification)

**Interfaces:**
- Consumes: Task 5's confirmed-clean O_wasm state.
- Produces: the second, independent oracle's confirmation — this task is what makes the work "complete" per this plan's Global Constraints, not Task 5 alone.

- [ ] **Step 1: Dispatch**

```bash
gh workflow run larql-lql.yml --ref gating/larql-cli-wasm-and-safe
```

- [ ] **Step 2: Get the run ID**

```bash
gh run list --workflow=larql-lql.yml --branch gating/larql-cli-wasm-and-safe --limit 1 \
  --json databaseId,status
```

- [ ] **Step 3: Wait for completion, then independently re-verify via a fresh API call**

```bash
gh run view <run-id> --json status,conclusion,jobs
```

Expected: `status: completed`, `conclusion: success`, every job (`test - ubuntu-latest`, `test - windows-latest`, `test - macos-14`, `coverage - ubuntu`) green. If anything fails, read the actual failure — per this whole campaign's standing discipline, a red O_native here after a green O_wasm is real, actionable signal, not something to wave off.

---

### Task 7: Record the outcome

**Files:**
- Modify: `.superpowers/sdd/2026-08-10-larql-cli-wasm-and-safe-gating/progress.md` (gitignored, local-only — not part of this branch's diff, but the durable record for future sessions on this campaign)

**Interfaces:**
- Consumes: Task 1-6's verified results.
- Produces: an updated ledger entry and closed task-tracker items.

- [ ] **Step 1: Append a ledger entry** covering: the 13 findings and their 3-group root-cause breakdown (Task 0), the `clippy-fix.yml` first-dispatch validation result (did the patch match expectations on the first try, or did it need iteration — record whichever actually happened), and both oracles' final green confirmation (Task 5 + Task 6's run IDs and conclusions).

- [ ] **Step 2: Mark relevant tracked tasks completed** (e.g., a `larql-lql` clippy-sweep entry, if one exists in the task tracker at execution time — check via `TaskList` and update whatever's live rather than assuming a specific ID here, since IDs may have shifted since this plan was written).
