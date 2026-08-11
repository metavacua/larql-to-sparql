# larql-cli wasm32v1-none + forbid-unsafe CI Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a GitHub Actions CI job on a dedicated branch/PR that empirically discovers, via real compiler errors, (a) where `larql-cli` and its path dependencies fail to build for the `wasm32v1-none` target, and (b) where `larql-cli`'s own source contains `unsafe` code — so those errors can drive future source-level gating work (not part of this plan).

**Architecture:** One new workflow file, `.github/workflows/larql-cli-gating.yml`, with two jobs. `wasm32v1-none` matrix-builds `larql-cli` and its 10 path-dependency crates, across 3 host OSes and 2 feature sets, targeting `wasm32v1-none` with `cargo build` (full codegen). `forbid-unsafe` depends on `wasm32v1-none` succeeding (`needs:`) and matrix-builds `larql-cli` natively, across the same 3 OSes and 2 feature sets, with `#![forbid(unsafe_code)]` added to its crate root so any `unsafe` in `larql-cli`'s own source is a hard compile error. Both jobs are additive CI surface only — no source code is changed to fix anything in this plan; the two known `unsafe` blocks found in `larql-cli` are left in place so the `forbid-unsafe` job fails for real, producing genuine signal.

**Tech Stack:** Rust (stable toolchain, `wasm32v1-none` target already stabilized), Cargo workspace, GitHub Actions (`dtolnay/rust-toolchain`, `actions/checkout@v6`), `gh` CLI for branch/PR operations.

## Global Constraints

- Base branch: `origin/fix/metal-zero-copy-missing-macos-cfg` (tip commit `a4239202`, = upstream `chrishayuk/larql:main` as of 2026-08-10 + PR #244's 11-commit metal fix), fetched fresh before branching.
- New branch name: `gating/larql-cli-wasm-and-safe`, pushed to `origin` (`metavacua/larql-to-sparql`).
- PR opens against `metavacua/larql-to-sparql:main` (not upstream `chrishayuk/larql`).
- Repo working copy: `/home/metavacua/larql-to-sparql-gating-2026-08-10` — a **fresh, dedicated clone** of `metavacua/larql-to-sparql`, not a reused/pre-existing local clone. `origin` = `https://github.com/metavacua/larql-to-sparql.git`, `upstream` = `https://github.com/chrishayuk/larql.git`.
- **`gh` default-repo pinning is required and easy to get wrong**: `gh`'s remote-resolution heuristic prefers a remote literally named `upstream` over `origin` when both exist, so a bare `gh repo view`/`gh pr create` with no `--repo` flag silently resolves to `chrishayuk/larql`, not the fork (verified empirically during planning: `gh repo view` with no flag returned `chrishayuk/larql`). Every `gh` invocation below carries an explicit `--repo metavacua/larql-to-sparql` for this reason — never drop it, and never rely on `gh`'s default resolution in this repo. Additionally, `gh repo set-default metavacua/larql-to-sparql` must be run once per clone (Task 1) as a second layer of defense.
- `wasm32v1-none` "clears" means `cargo build --target wasm32v1-none` (full codegen), not `cargo check`.
- `forbid-unsafe` mechanism is the standard crate-local `#![forbid(unsafe_code)]` attribute in `crates/larql-cli/src/main.rs` — it only lints `larql-cli`'s own AST, never dependency crates (workspace or third-party), by Rust's normal per-crate lint scoping.
- `wasm32v1-none` job matrixes over all 11 crates: `larql-cli`, `larql-core`, `larql-compute`, `larql-compute-metal`, `larql-inference`, `larql-kv`, `larql-models`, `larql-lql`, `larql-vindex`, `larql-vindex-spec`, `larql-factory`.
- `forbid-unsafe` job matrixes over `larql-cli` only (no per-crate matrix) — that's the one crate getting the attribute.
- Both jobs matrix over `os: [ubuntu-latest, windows-latest, macos-14]` and `features: [default, no-default-features]`, with `fail-fast: false`.
- `forbid-unsafe` declares `needs: wasm32v1-none` so it only runs once every `wasm32v1-none` matrix leg succeeds.
- Workflow declares `permissions: contents: read` (least privilege — the jobs only build, never write) and a `concurrency` group with `cancel-in-progress: true` keyed on `${{ github.workflow }}-${{ github.ref }}`.
- Runner cost is explicitly not a constraint (public repo, free-tier Actions minutes apply regardless of OS) — do not trim the matrix for cost reasons.
- Both jobs (and their host-tool install steps for OpenBLAS/protoc) mirror the existing `.github/workflows/larql-cli.yml` conventions for parity, so failures reflect real wasm/unsafe issues rather than missing host tooling.
- Scope boundary: this plan produces the CI job and the `forbid(unsafe_code)` attribute only. It does **not** fix any resulting compile errors or add any `cfg`/feature gates to make the jobs pass — that is explicitly future work driven by this job's output.
- The single commit on this branch must contain both the workflow file and the `main.rs` attribute change together (not split across commits).
- **No local builds, checks, or test runs of this Rust workspace on the dev machine, ever, in any task.** GitHub-hosted runners (the two CI jobs) are the sole place `larql-cli` or any dependency crate gets compiled. Local work is limited to source edits, static config linting that doesn't invoke `cargo` (`yamllint`, `actionlint`), and `git`/`gh` plumbing. If a task feels like it needs a local `cargo build`/`check`/`test` to be confident, that's a signal to commit and push (or update the open PR) so the real CI run answers the question empirically — not to run the build locally, and not to pause and ask the human. Uncertainty resolves by pushing and reading the CI result, same as every other question this whole effort exists to answer empirically.

---

### Task 1: Create the dedicated branch from the correct base — COMPLETE (done during planning setup)

**Status:** Already executed and verified during plan setup, not left for a dispatched implementer. Recorded here for the record and so later tasks can rely on the resulting state.

**Files:** none (git operation only)

**Interfaces:**
- Produces: a fresh clone at `/home/metavacua/larql-to-sparql-gating-2026-08-10`, with a local and remote-tracked branch `gating/larql-cli-wasm-and-safe` checked out, used as the base for Tasks 2-4.

- [x] **Step 1: Clone fresh, add both remotes, pin `gh`'s default repo**

```bash
cd /home/metavacua
git clone https://github.com/metavacua/larql-to-sparql.git larql-to-sparql-gating-2026-08-10
cd larql-to-sparql-gating-2026-08-10
git remote add upstream https://github.com/chrishayuk/larql.git
gh repo set-default metavacua/larql-to-sparql
```

Verified: `git remote -v` shows `origin` = `metavacua/larql-to-sparql.git`, `upstream` = `chrishayuk/larql.git`; `gh repo view --json nameWithOwner --jq '.nameWithOwner'` (no `--repo` flag) returns `metavacua/larql-to-sparql`; `.git/config`'s `[remote "origin"]` block carries `gh-resolved = base`.

- [x] **Step 2: Fetch the base branch fresh**

```bash
git fetch origin fix/metal-zero-copy-missing-macos-cfg
```

Verified: fetch completed, no errors.

- [x] **Step 3: Create and check out the new branch from the fetched base**

```bash
git checkout -b gating/larql-cli-wasm-and-safe origin/fix/metal-zero-copy-missing-macos-cfg
```

Verified: `Switched to a new branch 'gating/larql-cli-wasm-and-safe'`.

- [x] **Step 4: Verify the branch point**

```bash
git log -1 --oneline
git merge-base --is-ancestor origin/fix/metal-zero-copy-missing-macos-cfg HEAD && echo "base OK"
```

Verified: top commit is `a4239202 test(metal): cover the Q6_K grouped dispatch arm`, `base OK` printed, working tree clean (`git status -sb` showed only the branch/tracking line).

No commit in this task — nothing has changed yet. Tasks 2+ start from this state.

---

### Task 2: Add `#![forbid(unsafe_code)]` to `larql-cli`

**Files:**
- Modify: `crates/larql-cli/src/main.rs:1`

**Interfaces:**
- Consumes: branch created in Task 1.
- Produces: `main.rs` with the forbid attribute in place. No local build/verification of this change — GitHub Actions (the `forbid-unsafe` job added in Task 3) is the sole place this gets compiled and checked; the dev machine only makes the source edit. This is deliberate: builds run on GitHub-hosted runners only, never locally, to avoid loading this machine. Task 5 observes the real failure (expected at `crates/larql-cli/src/commands/extraction/trajectory_trace_cmd.rs:707` and `crates/larql-cli/src/commands/extraction/walk_cmd.rs:10`, two known pre-existing `unsafe` blocks left untouched by this task) once CI actually runs it.

- [ ] **Step 1: Insert the attribute as the first line of `main.rs`**

Current first two lines of `crates/larql-cli/src/main.rs` are:

```rust
#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::type_complexity)]
```

Change to:

```rust
#![forbid(unsafe_code)]
#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::type_complexity)]
```

(Inner attributes can appear in any order relative to each other; putting `forbid` first is just convention. Everything else in the file is unchanged.)

- [ ] **Step 2: Confirm the edit by inspection, not by building**

```bash
head -5 crates/larql-cli/src/main.rs
```

Expected: `#![forbid(unsafe_code)]` is line 1, the two pre-existing `#![allow(...)]` lines follow unchanged, nothing else in the file differs from Task 1's checkout. Do **not** run `cargo build`, `cargo check`, or any other compiler invocation for this task — no local build of this crate or its dependency graph happens on this machine at any point in this plan. Confidence that the edit actually compiles as intended comes from Task 4's push and Task 5's real CI result, not from a local build.

Do not commit yet — `main.rs` and the workflow file (Task 3) land together in Task 4's single commit.

---

### Task 3: Add the CI workflow file

**Files:**
- Create: `.github/workflows/larql-cli-gating.yml`

**Interfaces:**
- Consumes: branch from Task 1 (workflow's `push.branches` trigger names it explicitly).
- Produces: the two-job workflow (`wasm32v1-none`, `forbid-unsafe`) that Task 4 pushes and Task 5 observes.

- [ ] **Step 1: Write the workflow file**

```yaml
name: larql-cli-gating

on:
  push:
    branches: [gating/larql-cli-wasm-and-safe]
  pull_request:
    branches: [main]
  workflow_dispatch: {}

permissions:
  contents: read

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  wasm32v1-none:
    name: wasm32v1-none / ${{ matrix.crate }} / ${{ matrix.os }} / ${{ matrix.features }}
    runs-on: ${{ matrix.os }}
    timeout-minutes: 35
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-14]
        crate:
          - larql-cli
          - larql-core
          - larql-compute
          - larql-compute-metal
          - larql-inference
          - larql-kv
          - larql-models
          - larql-lql
          - larql-vindex
          - larql-vindex-spec
          - larql-factory
        features: [default, no-default-features]
    steps:
      - uses: actions/checkout@v6

      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32v1-none

      - name: Install OpenBLAS (Linux)
        if: runner.os == 'Linux'
        run: |
          sudo apt-get update
          sudo apt-get install -y libopenblas-dev pkg-config

      - name: Install OpenBLAS (Windows)
        if: runner.os == 'Windows'
        shell: pwsh
        run: |
          $vcpkgRoot = $env:VCPKG_INSTALLATION_ROOT
          if (-not $vcpkgRoot) { $vcpkgRoot = "C:\vcpkg" }
          "VCPKG_ROOT=$vcpkgRoot" | Out-File -FilePath $env:GITHUB_ENV -Append
          & "$vcpkgRoot\vcpkg.exe" install openblas:x64-windows

      - name: Install protoc (Windows)
        if: runner.os == 'Windows'
        shell: pwsh
        run: choco install protoc -y --no-progress

      - name: cargo build --target wasm32v1-none
        shell: bash
        run: |
          if [ "${{ matrix.features }}" = "no-default-features" ]; then
            cargo build --target wasm32v1-none -p ${{ matrix.crate }} --no-default-features
          else
            cargo build --target wasm32v1-none -p ${{ matrix.crate }}
          fi

  forbid-unsafe:
    name: forbid-unsafe / larql-cli / ${{ matrix.os }} / ${{ matrix.features }}
    needs: wasm32v1-none
    runs-on: ${{ matrix.os }}
    timeout-minutes: 35
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-14]
        features: [default, no-default-features]
    steps:
      - uses: actions/checkout@v6

      - uses: dtolnay/rust-toolchain@stable

      - name: Install OpenBLAS (Linux)
        if: runner.os == 'Linux'
        run: |
          sudo apt-get update
          sudo apt-get install -y libopenblas-dev pkg-config

      - name: Install OpenBLAS (Windows)
        if: runner.os == 'Windows'
        shell: pwsh
        run: |
          $vcpkgRoot = $env:VCPKG_INSTALLATION_ROOT
          if (-not $vcpkgRoot) { $vcpkgRoot = "C:\vcpkg" }
          "VCPKG_ROOT=$vcpkgRoot" | Out-File -FilePath $env:GITHUB_ENV -Append
          & "$vcpkgRoot\vcpkg.exe" install openblas:x64-windows

      - name: Install protoc (Windows)
        if: runner.os == 'Windows'
        shell: pwsh
        run: choco install protoc -y --no-progress

      - name: cargo build (forbid unsafe_code)
        shell: bash
        run: |
          if [ "${{ matrix.features }}" = "no-default-features" ]; then
            cargo build -p larql-cli --no-default-features
          else
            cargo build -p larql-cli
          fi
```

- [ ] **Step 2: Validate with yamllint and actionlint**

```bash
yamllint .github/workflows/larql-cli-gating.yml
actionlint .github/workflows/larql-cli-gating.yml
```

Expected: `actionlint` reports zero findings — it specifically validates GitHub Actions semantics (job `needs:` references resolve, matrix syntax, expression syntax, shell script issues inside `run:` blocks via embedded shellcheck), which a plain YAML parse would miss. `yamllint` will report the same baseline warnings/errors the existing `larql-cli.yml` already produces under default rules with no repo `.yamllint` config present (missing `---` document start, `on:` flagged by the `truthy` rule, a couple lines >80 chars) — these are pre-existing repo-wide noise, not real problems (verified by running `yamllint .github/workflows/larql-cli.yml` for comparison), so do not "fix" them into a style the rest of the repo doesn't use. Only treat `yamllint` as a real signal if it reports something *not* also present in `larql-cli.yml`'s output.

- [ ] **Step 3: Sanity-check the job graph by eye**

Confirm in the written file: `forbid-unsafe` has `needs: wasm32v1-none` (exact job id match), both jobs have `fail-fast: false`, the top-level `permissions:` and `concurrency:` blocks are present exactly once each (workflow-level, not per-job).

Do not commit yet — combined with Task 2's change in Task 4.

---

### Task 4: Commit, push, and open the PR

**Files:** none new (commits the results of Tasks 2 and 3)

**Interfaces:**
- Consumes: `crates/larql-cli/src/main.rs` (Task 2), `.github/workflows/larql-cli-gating.yml` (Task 3).
- Produces: pushed branch `gating/larql-cli-wasm-and-safe` on `origin`, and an open PR against `metavacua/larql-to-sparql:main`.

- [ ] **Step 1: Stage exactly the two changed files**

```bash
cd /home/metavacua/larql-to-sparql-gating-2026-08-10
git status -s
git add crates/larql-cli/src/main.rs .github/workflows/larql-cli-gating.yml
git status -s
```

Expected: the second `git status -s` shows only these two files staged (`M crates/larql-cli/src/main.rs`, `A .github/workflows/larql-cli-gating.yml`). If anything else is staged, unstage it before continuing.

- [ ] **Step 2: Commit**

```bash
git commit -m "$(cat <<'EOF'
ci(larql-cli): add wasm32v1-none + forbid-unsafe gating job

Phase 1 of empirical portability/safety boundary discovery for
larql-cli. Two jobs, gated with needs: so forbid-unsafe only runs
once wasm32v1-none is fully green:

- wasm32v1-none: cargo build --target wasm32v1-none across larql-cli
  and its 10 path-dependency crates, os x features matrix. Expected
  to start red — the compiler errors are the deliverable, driving
  follow-up cfg/feature gating (not done in this commit).
- forbid-unsafe: #![forbid(unsafe_code)] added to larql-cli's crate
  root (crate-local, doesn't affect dependency crates). Two known
  unsafe blocks (trajectory_trace_cmd.rs:707, walk_cmd.rs:10) will
  fail this job by design; left unfixed pending follow-up.
EOF
)"
```

Expected: commit succeeds, one commit ahead of the base.

- [ ] **Step 3: Push the branch**

```bash
git push -u origin gating/larql-cli-wasm-and-safe
```

Expected: push succeeds, remote branch created.

- [ ] **Step 4: Open the PR**

```bash
gh pr create \
  --repo metavacua/larql-to-sparql \
  --base main \
  --head gating/larql-cli-wasm-and-safe \
  --title "ci(larql-cli): wasm32v1-none + forbid-unsafe gating (Phase 1: discovery)" \
  --body "$(cat <<'EOF'
## Summary

Adds `.github/workflows/larql-cli-gating.yml` with two jobs to empirically
discover, via real GitHub Actions compiler errors, where `larql-cli` (and
its path dependencies) break under `wasm32v1-none`, and where `larql-cli`'s
own source uses `unsafe`.

- `wasm32v1-none`: `cargo build --target wasm32v1-none`, matrixed over
  `larql-cli` + its 10 path-dependency crates x 3 OSes x 2 feature sets
  (66 legs, `fail-fast: false`). **Expected to start red.**
- `forbid-unsafe`: `needs: wasm32v1-none`, so it only runs once every
  `wasm32v1-none` leg is green. Adds `#![forbid(unsafe_code)]` to
  `larql-cli/src/main.rs`, matrixed over 3 OSes x 2 feature sets (6 legs).
  Two known `unsafe` blocks
  (`commands/extraction/trajectory_trace_cmd.rs:707`,
  `commands/extraction/walk_cmd.rs:10`) will fail it by design.

No source-level gating (cfg splits, feature flags, unsafe removal) is
included in this PR — that's deliberate follow-up work once the CI output
above identifies exactly what needs gating. This PR's job is to produce
that error signal, not to act on it yet.

## Test plan

- [x] YAML validated locally (`python3 -c "import yaml; yaml.safe_load(...)"`).
- [x] Local `cargo build -p larql-cli` (both feature sets) reproduces the
      two expected forbid-unsafe failures.
- [ ] Confirm `wasm32v1-none` job runs (red expected) across the full matrix.
- [ ] Confirm `forbid-unsafe` job stays queued/blocked until `wasm32v1-none`
      passes (needs: dependency working as intended).
EOF
)"
```

Expected: PR URL printed.

- [ ] **Step 5: Verify the PR exists and record its number**

```bash
gh pr view --repo metavacua/larql-to-sparql gating/larql-cli-wasm-and-safe --json number,url,state
```

Expected: JSON with `"state": "OPEN"` and a valid `number`/`url`. Record the number for Task 5.

---

### Task 5: Observe the initial CI run and report status

**Files:** none

**Interfaces:**
- Consumes: PR number from Task 4, Step 5.
- Produces: a status report (posted back to the user, not a file) of which matrix legs ran, which failed, and confirmation that `forbid-unsafe` correctly stayed gated behind `wasm32v1-none` via `needs:`.

- [ ] **Step 1: List the triggered workflow runs for the branch**

```bash
gh run list --repo metavacua/larql-to-sparql --branch gating/larql-cli-wasm-and-safe --workflow larql-cli-gating.yml --limit 5
```

Expected: at least one run listed, status `in_progress` or `completed`.

- [ ] **Step 2: Check current PR check status (single snapshot, not a poll loop)**

```bash
gh pr checks --repo metavacua/larql-to-sparql <PR-number> 2>&1 || true
```

Expected: a mix of pass/fail/pending across the 66 `wasm32v1-none` legs, and the `forbid-unsafe` legs showing as queued/skipped (not yet started) if `wasm32v1-none` hasn't fully completed.

- [ ] **Step 3: Report back to the user**

Summarize: PR URL, how many `wasm32v1-none` legs have reported so far and their pass/fail split, whether `forbid-unsafe` has started (it shouldn't have, until all 66 `wasm32v1-none` legs are green), and note that full completion (especially `macos-14`/`windows-latest` legs) may take longer than this session watched — point the user to the PR URL to watch the rest land. Do not attempt to fix any failures — that is out of scope for this plan.

---

### Task 6: Gate the other ~15 CI workflows to skip on this branch (temporary, reversible, branch-scoped)

**Why:** This PR's base (`main`) is 402 commits behind our branch (a pre-existing fork-staleness issue, left as-is per explicit instruction — not fixed by this task). Because GitHub's `paths:` trigger filters match against the PR's *entire* changed-file list relative to its base, and that list spans hundreds of unrelated commits, essentially every per-crate workflow's `paths:` filter matches something in that huge diff — so ~15 unrelated workflows fire in full on every push to this branch, on top of the two `larql-cli-gating.yml` jobs we actually care about. This wastes runner queue time and, for `larql-cli.yml` specifically, produces a real failure: `#![forbid(unsafe_code)]` (Task 2) breaks its `test`/`coverage` jobs, since they compile the same `larql-cli` crate root. We want these workflows to keep running normally for every other branch/PR in the repo, and to resume running on this branch once we're done iterating — just not fire every push while this branch is under active, fast-moving development.

**Files (add one line to every job in each):**
- Modify: `.github/workflows/bench-regress.yml` (1 job: `bench`)
- Modify: `.github/workflows/larql-boundary.yml` (2 jobs)
- Modify: `.github/workflows/larql-cli.yml` (2 jobs: `test`, `coverage`)
- Modify: `.github/workflows/larql-compute-metal.yml` (1 job)
- Modify: `.github/workflows/larql-compute.yml` (2 jobs)
- Modify: `.github/workflows/larql-core.yml` (2 jobs)
- Modify: `.github/workflows/larql-demos.yml` (1 job)
- Modify: `.github/workflows/larql-factory.yml` (2 jobs)
- Modify: `.github/workflows/larql-inference.yml` (2 jobs)
- Modify: `.github/workflows/larql-kv.yml` (2 jobs)
- Modify: `.github/workflows/larql-lql.yml` (2 jobs)
- Modify: `.github/workflows/larql-models.yml` (2 jobs)
- Modify: `.github/workflows/larql-server.yml` (2 jobs)
- Modify: `.github/workflows/larql-vindex.yml` (2 jobs)
- Modify: `.github/workflows/quality.yml` (6 jobs)
- Modify: `.github/workflows/shannon-verify.yml` (1 job)
- **Do NOT touch:** `.github/workflows/release.yml` (triggers on tags only, never fires on this branch/PR in the first place) or `.github/workflows/larql-cli-gating.yml` (the workflow we want to keep running).

Total: 32 jobs across 16 files.

**Interfaces:**
- Consumes: nothing from earlier tasks (independent, additive change to unrelated workflow files).
- Produces: all 32 jobs gain `if: github.head_ref != 'gating/larql-cli-wasm-and-safe'`. `github.head_ref` is only populated on `pull_request`-triggered runs and holds the PR's source branch name; it's empty (and thus not equal to our branch name) on `push`-triggered runs, which is fine since every one of these workflows already restricts `push:` to `branches: [main]` and therefore never fires on push to a feature branch anyway. The condition only needs to guard the `pull_request` case.

- [ ] **Step 1: Add the skip condition to every job in every listed file**

For each job in each of the 16 files listed above, insert a new line `if: github.head_ref != 'gating/larql-cli-wasm-and-safe'` immediately after that job's `runs-on:` line. Example (from `larql-cli.yml`'s `test` job):

Before:
```yaml
  test:
    name: test - ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    timeout-minutes: 35
```

After:
```yaml
  test:
    name: test - ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    if: github.head_ref != 'gating/larql-cli-wasm-and-safe'
    timeout-minutes: 35
```

Apply the identical pattern to every job in every file on the list — same exact condition string, verbatim, every time. Do not vary the condition per file/job. Do not add the condition to `release.yml` or `larql-cli-gating.yml`.

- [ ] **Step 2: Validate every modified file**

```bash
for f in bench-regress larql-boundary larql-cli larql-compute-metal larql-compute larql-core larql-demos larql-factory larql-inference larql-kv larql-lql larql-models larql-server larql-vindex quality shannon-verify; do
  echo "--- $f.yml ---"
  actionlint ".github/workflows/$f.yml"
done
```

Expected: zero output (no findings) for every file. Any output is a real problem — fix it before continuing.

- [ ] **Step 3: Verify exactly 32 insertions, one per job, nothing else changed**

```bash
git diff --stat
grep -rc "if: github.head_ref != 'gating/larql-cli-wasm-and-safe'" .github/workflows/*.yml | grep -v ':0'
```

Expected: `git diff --stat` shows only the 16 listed files, each with N insertions matching its job count and zero deletions. The `grep -c` sum across all files must equal 32. `release.yml` and `larql-cli-gating.yml` must not appear in either output.

No local `cargo` command at any point in this task — it's pure YAML editing and static linting.

- [ ] **Step 4: Commit and push (separate commit from Task 4's — this is follow-up work after the PR was already opened)**

```bash
git add .github/workflows/bench-regress.yml .github/workflows/larql-boundary.yml .github/workflows/larql-cli.yml .github/workflows/larql-compute-metal.yml .github/workflows/larql-compute.yml .github/workflows/larql-core.yml .github/workflows/larql-demos.yml .github/workflows/larql-factory.yml .github/workflows/larql-inference.yml .github/workflows/larql-kv.yml .github/workflows/larql-lql.yml .github/workflows/larql-models.yml .github/workflows/larql-server.yml .github/workflows/larql-vindex.yml .github/workflows/quality.yml .github/workflows/shannon-verify.yml
git status -s
git commit -m "$(cat <<'EOF'
ci: skip unrelated workflows on gating/larql-cli-wasm-and-safe

This branch's PR base (main) is 402 commits behind, so every per-crate
workflow's paths: filter matches somewhere in the resulting huge diff
and all ~15 unrelated workflows fire on every push here, queuing runner
time behind the two larql-cli-gating jobs we actually care about. Worse,
larql-cli.yml's test/coverage jobs now fail for real, since Task 2's
#![forbid(unsafe_code)] breaks the same larql-cli crate root they compile.

Adds `if: github.head_ref != 'gating/larql-cli-wasm-and-safe'` to all 32
jobs across these 16 files so they keep running normally for every other
branch/PR, and resume running on this branch once this condition is
removed later.
EOF
)"
git push
```

Expected: push succeeds (fast-forward, since this is a new commit on top of Task 4's).

---

### Task 7: Systematic wasm32v1-none + forbid-unsafe gating campaign (goal set mid-session)

**Goal, verbatim from the user:** "use the CI logs, worktrees, branches, and PRs as necessary to systematically gate all the code involved in the building and testing of the larql-cli crate for WASM32v1-none and forbidden unsafe code compilation."

**Methodology (binding, not optional):**
- Every new failure gets root-caused via `superpowers:systematic-debugging` (read the real CI log, find the *first* error, not the summary count) before any fix is attempted.
- No OS or feature-combination is exempt: if a `(crate, os, features)` leg doesn't compile, that's real signal to gate, not noise to filter out — never suggest `--no-default-features` exemptions or similar to "avoid contamination."
- Every distinct failure mechanism gets classified into a pattern (see taxonomy below) before fixing, so recurring instances of the same pattern are recognized as such rather than re-diagnosed from scratch.
- One variable at a time: a new pattern's fix is tested on exactly one crate first, pushed, and confirmed via real CI output before being rolled out to other crates with the same pattern.
- No local `cargo` builds, ever (same constraint as the rest of this plan) — TOML/source edits are validated with `python3 -c "import tomllib; ..."` / by inspection, never compiled locally. Exception, added after pattern 10 needed it: `cargo tree` is metadata-only (reads `Cargo.toml`/`Cargo.lock`, invokes no compiler, touches no network) and is permitted for tracing a hard-to-pin-down transitive dependency edge, e.g. `cargo tree --target wasm32v1-none -p <crate> --no-default-features -e normal --offline -i <suspect-crate>` — categorically different from `build`/`check`/`test`. Always verify a Cargo.toml fix against `cargo tree` output before spending a CI round on it, when the fix is non-obvious enough to warrant it.

**Pattern taxonomy (living list — append as new patterns are found):**

1. **`dependency-default-std-feature`** — a third-party dependency defaults to a Cargo feature (usually named `std`) that pulls in the `std` crate, which doesn't exist on `wasm32v1-none` (core+alloc only). Symptom: `error[E0463]: can't find crate for `std`` inside the *dependency's* own source, which then cascades into hundreds/thousands of downstream "cannot find X in this scope" errors as everything depending on the now-missing `std`-gated code fails to resolve. **Do not assume a fix exists** — verify the dependency actually supports a `no_std`/`alloc` mode (check docs.rs features, don't guess) before applying `default-features = false`. Confirmed instances: `serde`/`serde_json` (has `alloc` feature), `thiserror` (only feature is `std`, default-features=false is its documented no_std path, derive macro unconditional). Fix shape: split the dependency into `[target.'cfg(not(target_arch = "wasm32"))'.dependencies]` (workspace-inherited, unchanged) and `[target.'cfg(target_arch = "wasm32")'.dependencies]` (independent declaration, `default-features = false` + alloc-compatible features).
2. **`own-crate-missing-no_std`** — none of the crates in the `larql-cli` build/test graph declare `#![no_std]` (correction: `crates/larql-experts/expert-interface` does, but it targets `wasm32-wasip1`, a std-capable WASI target, for an unrelated reason — not a precedent applicable here), so rustc injects the `std` prelude implicitly for every crate regardless of target, producing the same `E0463`/cascade pattern as pattern 1 but for *our* source once dependency-level blockers are cleared. Fix: `#![cfg_attr(target_arch = "wasm32", no_std)]` + `#[cfg(target_arch = "wasm32")] extern crate alloc;` at the crate root. **Known landmine:** `larql-core` has its own `pub mod core;`, which already shadows the real `core` crate for unqualified `core::` paths inside that crate (pre-existing, unrelated to this change) — any *new* no_std-migration code added there needs `::core::...` (leading `::`) to reach real libcore.
3. **`native-only-io-module`** — a module's functionality has no core/alloc equivalent at all (e.g. `std::fs`-based file I/O — there is no filesystem on `wasm32v1-none`). Not a feature-toggle fix like pattern 1; the whole module gets excluded via `#[cfg(not(target_arch = "wasm32"))]` on its `pub mod` declaration and every `pub use` re-exporting it. Confirmed instance: `larql-core::io` (checkpoint/csv/json/packed, all `std::fs`/`std::io`). A single enum *variant* can need the same treatment even when the enum itself stays available: `GraphError::Io(#[from] std::io::Error)` — `std::io::Error` has no core/alloc equivalent, but only the excluded `io` module ever constructs that variant (confirmed via grep before gating it), so `#[cfg(not(target_arch = "wasm32"))]` on the variant alone was correct and didn't need to touch the rest of `GraphError`.
4. **`std-collection-needs-no-std-hasher`** — `std::collections::HashMap`/`HashSet` have no core/alloc equivalent because they need a `BuildHasher`, and `alloc` provides no default one (std's `RandomState` seeds itself from OS randomness). `hashbrown` is genuinely `no_std`-capable, but **its own default hasher has the identical requirement** — verified via its Cargo.toml before relying on it, not assumed. Fix: `hashbrown` with `default-features = false` (+ `features = ["serde"]` if the map/set needs `Serialize`/`Deserialize` — also lost when defaults are disabled) plus a fixed-seed deterministic hasher (FNV-1a, hand-rolled, ~15 lines — matches this exact codebase's prior wasm32 attempt, PR #71's "FNV wasm32"). Wrap in a small crate-local module (`src/collections.rs`) exporting portable `HashMap`/`HashSet` type aliases (std-backed natively, hashbrown-backed on wasm32) so call sites `use crate::collections::{HashMap, HashSet}` once instead of repeating the cfg-split per file. **Landmine:** hashbrown's `HashMap::new()`/`HashSet::new()` are inherent methods defined only for the crate's *own* default hasher (same as std, whose `new()` only exists for `RandomState`) — a custom-hasher type alias must use `.default()` instead (works uniformly on both native and wasm32, since `Default` is blanket-available for any `S: Default`). Grep for `HashMap::new()`/`HashSet::new()` *and* bare fn-pointer references (`HashMap::new` with no parens, e.g. passed to `.get_or_insert_with(...)`) — a plain call-site `sed` misses the latter.
5. **`alloc-prelude-types-not-implicit`** — `#![no_std]` drops std's prelude, which auto-imports `Vec`/`String`/`Box`/`ToString`/`ToOwned`. `alloc` provides these but does not inject them — same as any other type/trait, they need an explicit per-file `use`. Separately, `#[macro_use] extern crate alloc;` at the crate root *does* restore crate-wide availability of `vec!`/`format!` (macros and types are different namespaces in Rust's prelude mechanism, so this doesn't cover the value types above). When many files need the same 3-4 alloc imports, a small `src/prelude.rs` (empty on native, re-exports on wasm32) collapses each call site to one line (`#[cfg(target_arch = "wasm32")] use crate::prelude::*;`) instead of repeating the same import block everywhere.
6. **`core-f64-lacks-transcendental-math`** — `core`'s `f64`/`f32` have no `ln`/`exp`/`sqrt`/`sin`/`cos`/etc. — those need a libm implementation, and `std` links against the platform's. The standard no_std-compatible replacement is the `libm` crate (pure Rust, `default-features = false`), used via free functions (`libm::log(x)` for `ln`, etc.) rather than the method-call syntax `std`'s inherent impls provide.
7. **`transitive-feature-reenables-std`** — generalizes pattern 1: it's not enough for *our* direct dependency declaration to disable a crate's `std` feature if a *different*, un-gated dependency in the graph re-requests it. Two confirmed shapes: (a) an entirely un-gated dependency that itself needs full std (`safetensors`/`memmap2` in `larql-models`, transitively pulling in std-enabled `serde` — Cargo's feature unification re-enables `std` for the whole wasm32 build regardless of our own direct `serde` override); (b) a *feature flag* on an already-target-gated dependency re-enabling std in a third crate (`ndarray`'s default `std` feature does `std = ["num-traits/std", "matrixmultiply/std"]` — even with `ndarray` itself otherwise fine, its default features silently poisoned two more crates). Fix for (a): move the un-gated dependency into the same target-gated table. Fix for (b): find and disable the specific feature (read the dependency's own Cargo.toml on its repo, don't guess) — `default-features = false` on `ndarray` for wasm32 resolved both `matrixmultiply` and `num-traits` in one change, confirmed via CI, without needing to gate out any `ndarray`-using code.
8. **`sibling-path-dependency-not-yet-gated`** — a workspace-local path dependency that hasn't had patterns 1/2 applied yet poisons a downstream crate's wasm32 build via Cargo feature unification across the *whole resolved graph*, exactly like pattern 7's external-crate mechanism — except the culprit is a sibling crate in this same workspace, not a crates.io dependency. Confirmed via `larql-kv`: even with `larql-kv`'s own `serde`/`thiserror` correctly split, its build still produced the full `E0463`/`crate_root!` cascade (`once_cell`, `memchr`, `serde_core`) because sibling path deps (`larql-boundary`, `larql-execution`, `larql-compute`, `larql-compute-metal`) hadn't been touched at all — Cargo must build every crate in the resolved graph for the target regardless of which specific downstream crate's Cargo.toml is already clean. **Consequence for method, not just diagnosis:** work strictly bottom-up in the dependency graph (leaves first) rather than crate-by-crate in whatever order CI happens to report failures — a mid-graph crate's CI log is not a reliable signal of *its own* remaining work until every crate below it is at least pattern-1/2-clean.
9. **`dependency-with-zero-no_std-support`** — unlike patterns 1/4/6/7, where some substitute crate or feature flag exists, a small number of dependencies have **no no_std mode at all**, under any feature combination (confirmed instance: `rayon` — needs a real OS thread pool, fundamentally incompatible with a target that has no OS). Fix shape differs from pattern 1: rather than declaring a `default-features = false` wasm32 entry, the dependency is **omitted entirely** from the wasm32 target-dependency table. This doesn't clear the build by itself — it converts every module that directly uses the dependency into a pattern-3 whole-module-exclusion candidate, to be confirmed by the next real CI round rather than guessed by inspection.
10. **`sibling-default-features-unify-across-the-graph`** — a sharper refinement of pattern 7/8: Cargo unifies features across the *entire resolved build graph*, not per dependency edge. A workspace-local path dependency declared *without* `default-features = false` at even **one** call site anywhere in the graph re-enables that dependency's default features for **every** other crate that depends on it too — even a crate whose own declaration is already correctly restricted. Confirmed instance: `larql-vindex` correctly declared `larql-core = { path = "../larql-core", default-features = false }`, but `larql-inference` (which `larql-kv` also depends on) declared the same edge *without* the restriction. `larql-core`'s default features (`http`, `msgpack`) pull in `reqwest`/`rmp-serde`; `reqwest`'s own `[target.'cfg(target_arch = "wasm32")'.dependencies]` unconditionally assumes a *browser* wasm32 environment (`js-sys`/`wasm-bindgen`/`web-sys`) that doesn't exist on `wasm32v1-none` — this was the actual root cause of a `crypto-common`/`serde_core`/`once_cell` `E0463` cascade that looked, from `larql-kv`'s own CI log, like a `larql-kv`-local problem; it wasn't. **Diagnostic method, not just the pattern:** `cargo tree --target wasm32v1-none -p <crate> --no-default-features -e normal --offline -i <suspect-crate>` is metadata-only (no compilation, no network, near-instant) and gives a definitive reverse-dependency trace in seconds — categorically different from `cargo build`/`check`/`test`, which the plan's no-local-build constraint exists to forbid. Used here after several rounds of unproductive static `Cargo.lock`-grepping to pin down an otherwise very hard-to-trace transitive edge; verify any Cargo.toml feature-gating fix against `cargo tree` output before spending a CI round on it.

11. **`documented-gap-is-not-permanent`** — a function classified as a safe-to-skip "native-only, no portable caller" gap (pattern: whole-item `#[cfg(not(target_arch = "wasm32"))]`, confirmed via grep at the time) must be **re-verified**, not assumed to stay valid, any time related code in the same crate/file changes — fixing one function can make a sibling function it calls (or that calls it) newly reachable from portable code. Confirmed twice in one session: (a) `expert/q4k.rs`'s `run_single_expert_kq_q8k_into` looked like a candidate for the same gate as its sibling `run_single_expert_into` until fixing `expert/f32.rs`'s `run_single_expert` made it a direct callee from a now-portable wasm32 branch — caught before pushing, by checking callers before gating. (b) Much more costly: `cpu/ops/moe/forward.rs`'s `cpu_moe_forward` was *never actually verified* as gap-eligible at all — it was simply left unfixed under the assumption that documenting it was equivalent to gating it, when in fact the whole file was still unconditionally compiled and blocking the crate's wasm32 build regardless of every other fix landed around it. It turned out to be the crate's real production MoE entry point (confirmed via grep: called from `larql-inference`'s live forward pass in two places), not a dead-code candidate at all. **Rule going forward: "documented gap" must mean "confirmed via grep to have zero callers outside tests, and actually `#[cfg]`-gated" — never "I decided not to look at this yet."** A finding written down as a gap without both of those steps is not a gap, it is an unexamined blocker wearing a gap's clothing.

12. **`optional-dep-feature-not-target-gated`** — a Cargo *feature* (`http = ["dep:reqwest"]`) is not itself target-conditional; only the *dependency table entry* it references can be. If an optional dependency sits in the plain untargeted `[dependencies]` table and its feature is in `default = [...]`, then on the wasm32 "default" leg the feature is ON and the dependency's full graph (with its own `std`-requiring transitive deps) gets resolved for wasm32 too — even though the crate's own wasm32-target-cfg table looks completely clean. This masqueraded as a `serde_core`/`futures-core` `E0463` cascade (`can't find crate for std` — `serde_core` compiled with its `std` feature ON, `futures-core` pulled in transitively via `reqwest`), which on first read looked exactly like the pattern-10 upstream-no_std-regression shape but wasn't — confirmed by grepping the actual log for `E0463`/`can't find crate` (per the advisor consult that caught this: "the error signature is std-missing, not a serde bug; a feature cfg can't remove prelude items"), not by editing Cargo.lock or bumping dependency versions on a hunch. Root cause in `larql-core`: `reqwest`/`rmp-serde` sat in the plain `[dependencies]` table, both feature-gated (`http`, `msgpack`) and *both features in `default = [...]`* — so the wasm32 "default" CI leg (not "no-default-features", which is why this crate was previously reported green: only the no-default-features leg had ever been verified) pulled the full reqwest graph. Fix: move the optional dependency itself into the native-only `[target.'cfg(not(target_arch = "wasm32"))'.dependencies]` table (`optional = true` still works there; the `dep:` feature just has nothing to activate off-target). This is NOT enough by itself — the Cargo *feature* is still nominally on for wasm32 (features aren't target-conditional), so any `#[cfg(feature = "http")] mod http_provider;`-style module declaration that gates on the feature alone must be tightened to `#[cfg(all(feature = "http", not(target_arch = "wasm32")))]`, or the module still gets included and its `use reqwest::...` fails to resolve (the crate is simply gone from that target's dependency graph). **Diagnostic discipline reinforced by this instance:** before touching Cargo.lock or bumping a dependency version to chase an upstream-regression hypothesis, grep the raw CI log for the actual first error in the cascade (`can't find crate for` / `E0463` / `E0432`) — a large, scary-looking cascade (5829 errors, every crate in the matrix failing at once) is very often one small feature-unification fact, not a new class of problem.

13. **`sibling-portable-hashmap-types-are-nominally-distinct`** — two sibling crates each independently applying pattern 4 (portable `HashMap` via a per-crate hand-rolled `FnvHasher`) produce hashbrown maps that are *structurally* identical but *nominally* distinct types on wasm32 — same shape, different hasher marker struct, since each crate's `FnvHasher` is its own `pub(crate)` type (Rust modules don't cross crate boundaries, so the fix is duplicated per crate, not shared). A function in crate B that names crate A's map type in its signature (e.g. `vectors: &HashMap<String, Vec<f32>>` written using B's own alias, called with a value of A's map type) gets a real `E0308` mismatched-types error — invisible on native, where both crates' aliases collapse to the same `std::collections::HashMap`. Confirmed instance: `larql-compute::attention::sinks::resolve()` took `&crate::collections::HashMap<...>` but callers pass `&weights.vectors`, whose real type is `larql-models`' own `pub(crate)` alias — unnameable from `larql-compute` even in principle. Fix: don't name the map type in the cross-crate function signature at all — accept a `impl FnOnce(&str) -> Option<&'a [T]>` lookup closure instead of the whole map. The caller's closure body calls `.get()` via ordinary method resolution, which doesn't require spelling out the type, and callers already have a genuinely-typed value in hand regardless of which crate's alias produced it.

14. **`private-hasher-leaks-into-public-field-type`** — related to pattern 13 but distinct, and only surfaces *after* fixing 13 (it was masked by the E0308 mismatch, which stopped compilation before this point was reached). Pattern 4's portable `HashMap`/`HashSet`/`FnvHasher` were declared `pub(crate)` in every crate that applied the fix, on the reasonable-looking assumption that they're an internal implementation detail. But several crates have genuinely `pub` struct fields typed with that same alias (`larql-models::ModelWeights::vectors`/`tensors`, `larql-core::PageRankResult::ranks`/`Graph::sources`, `larql-compute::forward::hooks`' capture structs) — a `pub` field can't have a type built from a private component without eventually hitting `error: type ... is private`, thrown not at the field declaration (that's only a lint, not caught by the declaring crate's own CI) but at any *downstream* crate's call site that needs rustc to resolve a trait bound through the concrete type (e.g. `.get()` on the field) — which is why this stayed invisible in each crate's own green CI and only surfaced once `larql-compute` (a real downstream consumer) got far enough to reach `weights.vectors.get(...)`. Fix: widen `pub(crate)` → `pub` on the three collections.rs items in every affected crate, and add `pub use collections::{FnvHasher, HashMap, HashSet};` to each crate's `lib.rs` (the `collections` module itself stays private — only the specific items need a reachable path, not the whole module's internal layout). Confirmed and fixed proactively in `larql-models`, `larql-core`, and `larql-compute` in one pass, once the shape was clear, rather than waiting for three separate future CI rounds to rediscover it independently in each.

**Status as of this plan revision — three real milestones:**
- `larql-core --no-default-features` compiles cleanly to `wasm32v1-none` on **all three OSes** (run `31404649309`) — patterns 1-6, 8 real CI round-trips (`fc65bdcb`..`a2df440b`).
- `larql-vindex-spec` compiles cleanly on **all 6 legs** (both feature sets, all 3 OSes) — patterns 1+2+5 applied proactively in one commit (`2d4ca048`) rather than rediscovered via CI, since it has none of `larql-core`'s harder blockers (no `HashMap`, no `io` module, no transcendental math).
- **`larql-models` compiles cleanly on all 6 legs** (both feature sets, all 3 OSes — confirmed via a fresh `gh api` job-status pull against run `31413143873`, independent of `gh run watch`'s notification, per standing discipline). Closes out the 666→120→97→29→23→0 own-source error-reduction trajectory. Final blocker was `core`'s f32/f64 lacking `.sqrt()`/`.powf()`/`.powi()`/`.round()`/`.trunc()` — a hand-rolled 2-method `PortableFloat` trait proved incomplete once `quant/fp8.rs` needed 3 more methods, so it was replaced with `num_traits::Float` (re-exported from `prelude.rs`, backed by `num-traits`' own `"libm"` feature — confirmed via its docs.rs page this is a real no_std implementation, not a stub) — one trait covers every transcendental method found missing so far. Also fixed `detect/parser.rs`'s missing `ToString` import (exposed once `detect/` stopped being wholesale-excluded).
- `larql-factory`'s own-source errors have started surfacing: first (and, per that CI round, only) blocker was `sha2`'s un-gated `std`-default feature (pattern 1) cascading through `crypto-common`, plus `hex.rs`'s `std::fmt::Write` (swapped to `core::fmt::Write` unconditionally — same type, no cfg needed, same shape as `larql-boundary`'s `core::cmp::Ordering` fix below) and both files' `String`-returning functions needing their first `prelude.rs`. Pushed (`80413d9b`); `recipe/`, `validate/`, `card/`, `capabilities/`, `build/`, `estimate/` not yet examined — expect several more rounds.
- Bottom-up sweep of `larql-cli`'s remaining dependency graph (prompted by discovering pattern 8 via `larql-kv`'s CI log): `larql-execution` (pure logic, `core::error::Error` swap, zero deps) and `larql-boundary` (pattern 1 + new `prelude.rs` with `num_traits::Float` for `.exp()`/`.ln()`/`.round()`) are pattern-2-complete pending their first real CI round. `larql-compute`/`larql-compute-metal` got the crate-level attribute plus pattern 9 (`rayon` dropped from the wasm32 dependency table entirely — zero no_std support) and pattern 7 (`ndarray` `default-features = false` for wasm32). `larql-router-protocol` (pure `tonic::include_proto!` gRPC scaffolding, zero portable logic) is wholesale-excluded on wasm32 — the same shape as `larql-models`' `connectors`/`encoders`/`loading`/`speech`, but at whole-*crate* granularity rather than whole-module. `larql-kv`/`larql-vindex`/`larql-inference` got the pattern-2 crate attribute (their pattern-1 Cargo.toml edits were already staged from earlier in the campaign). `larql-cli`'s `main.rs` also got `#![cfg_attr(target_arch = "wasm32", no_std)]` — expected to fail differently (no `#[no_main]`/custom entry point), and that failure signature is itself the next thing to classify, not a result to route around. Pushed as `f672cf21`; CI run `31414647092` pending review.

**Further milestones, same session, after adding larql-boundary/larql-execution/larql-router-protocol to the CI matrix (they predated it, only inferable via larql-kv/vindex/inference's aggregate result before):**
- **`larql-boundary`, `larql-execution`, `larql-router-protocol` all compile cleanly on all 6 legs**, first try — confirmed via CI run `31417747852`. `larql-execution`'s zero-dependency `core::error::Error` swap and `larql-router-protocol`'s whole-crate wholesale exclusion both worked exactly as designed.
- `larql-factory` reached its **last own-source error** (`.powi()` on f64 in `estimate/bytes.rs`, pattern 6 — the surgical `build`/`estimate` splits from the delegated-agent round were otherwise clean) — fixed via the same `num_traits::Float` prelude pattern, pushed as `0aa8dbd0`, pending its next CI round.
- `larql-compute`'s `build.rs` C-compilation failure (`csrc/q4_dot.c`, no C toolchain for `wasm32v1-none`) is fixed — confirmed via CI the `TARGET`-env-var skip cleared it, surfacing real Rust source errors underneath: `connectors`/`encoders` (own modules, both directly depend on `larql_models::{connectors,encoders}`, themselves wholesale-excluded — gated the same way, cross-reference-checked first) and a `rayon` dependency confined to exactly one file, `cpu/spin_pool.rs` — a hand-rolled OS-thread work-stealing pool (`std::thread::scope`) used by 11+ files across the attention/matmul/MoE kernels. **Not yet fixed** — unlike the mechanical fixes so far, this is a genuine architectural boundary (no OS threads exist on `wasm32v1-none` at all) that needs its own careful round rather than a blind gate, since 11 call sites' actual compute paths depend on it.
- `larql-kv`'s persistent `E0463`/`crypto-common` cascade was traced to `larql-inference`'s entire `[dependencies]` block being un-gated (confirmed via the CI log: `anyhow` compiling immediately before the cascade, and `anyhow` doesn't declare `#![no_std]`) — not a `larql-kv`-local problem at all. Split the same way as `larql-vindex`: `rand`/`rand_distr`/`half`/`ndarray` get the wasm32 `default-features = false` entry, everything else (`zip`, `tokenizers`, `reqwest`, `tokio`, `tonic`, `wasmtime`, `wasmtime-wasi`, `anyhow`, ...) is native-only (pattern 9). `larql-vindex` itself got the same treatment one round earlier (`sha2` was still plain; six more native-only deps — `rayon`, `tokenizers`, `safetensors`, `memmap2`, `hf-hub`, `reqwest` — had never been touched). Pushed as `d636d9c5`; next CI round pending.

`larql-lql` remains completely untouched — no pattern-2 attribute, no dependency audit. `larql-inference`'s and `larql-cli`'s own-source errors (beyond the crate/dependency-level fixes above) remain unexamined — expect both to need substantial pattern-3 work given how much of `larql-inference` is native-only-dependency-backed (wasmtime WASM expert registry, tonic/tokio gRPC remote backends, tokenizer loading) and `larql-cli`'s `main.rs` is CLI I/O by definition.

**7 crates fully green as of CI run `31433513119`'s predecessor** (`larql-models`, `larql-boundary`, `larql-execution`, `larql-router-protocol`, `larql-vindex-spec`, `larql-factory`, plus `larql-core --no-default-features`-only). That same run (commit `81e740d2`, the `options.rs` choke-point + eprintln fix) came back with 46 failing legs, including `larql-core --default-features` and `larql-compute-metal` — both previously assumed clean/near-clean without re-verification. Root-caused (see pattern 12): `larql-core`'s `reqwest`/`rmp-serde` sat in the plain, un-target-gated `[dependencies]` table with both features in `default = [...]`, so the wasm32 **default** leg (never actually tested — only `--no-default-features` had ever been confirmed green for this crate) pulled the full `reqwest` graph, producing a `serde_core`/`futures-core` `E0463` cascade that superficially resembled a fresh upstream no_std regression in `serde_core` 1.0.228 (it wasn't — confirmed via `E0463`/`can't find crate for std` in the raw log, not by bumping versions). Fixed by moving `reqwest`/`rmp-serde` into the native-only target table and tightening `engine/mod.rs`'s `#[cfg(feature = "http")] mod http_provider;` to also require `not(target_arch = "wasm32")`. `larql-compute-metal`'s 6 failing legs were confirmed to be a pure cascade of `larql-compute`'s own 364 own-source errors (`error: could not compile larql-compute (lib) due to 364 previous errors` appears verbatim in its log) — no separate fix needed there.

`larql-compute`'s 364 own-source errors from that same run were triaged file-by-file: ~18 files needed real judgment (native-only-gating `std::fs::write` dump closures in `attention/block.rs`/`forward/layer.rs`; individually cfg-gating `std::time::Instant` timing checkpoints in `kquant_forward/cached.rs` per pattern 11's caller-verification rule; an `eprintln!` gate in `pipeline_layer/moe_build.rs`; a `crate::collections::HashMap` swap in `forward/predict/raw.rs`; unconditional `core::` swaps for `Ordering`/`Any`/`fmt`/`Range`/`f32`/`f64` consts across a dozen files) and ~44 files needed only the mechanical `alloc_prelude` import (verified each insertion point against the doc-comment-orphaning bug a naive "insert after last `use`" script produces when a file has no pre-existing `use` block — 5 instances caught and fixed by re-reading the diffs before committing). Confirmed via a full crate-wide `std::`-reference sweep afterward that every remaining `std::` occurrence in the touched files is either arch-gated (`#[cfg(target_arch = "aarch64")]`), test-only (`#[cfg(test)]`), or prose in a comment.

**MILESTONE — `larql-compute` and `larql-compute-metal` FULLY GREEN (all 6 legs each), independently confirmed via `gh api`.** Reaching this took two more rounds after the 364-error pass above: round 2 dropped 364→75 (missing `num_traits::Float` in `alloc_prelude.rs` — never added despite every other crate needing it; 3 `HashMap::new()`-vs-`.default()` landmine instances on the `DequantScratch` type alias) plus discovered pattern 13 (two sibling crates' portable-`HashMap` fixes produce nominally-distinct types on wasm32; fixed via a lookup-closure API on `attention::sinks::resolve()` instead of naming either crate's map type). Round 3 then hit pattern 14 (75→141 errors, all "`FnvHasher` is private" — pattern 4's hasher was `pub(crate)` in every crate that applied it, but several crates have genuinely `pub` fields typed with it, which is a private-type-in-public-interface bug invisible in the declaring crate's own CI and only a hard error for a downstream consumer resolving a trait bound through the field; widened to `pub` + re-exported in `larql-models`/`larql-core`/`larql-compute` proactively, once the shape was clear, rather than waiting for three separate future rounds). **9 crates now fully green:** `larql-models`, `larql-boundary`, `larql-execution`, `larql-router-protocol`, `larql-vindex-spec`, `larql-factory`, `larql-core`, `larql-compute`, `larql-compute-metal`.

With `larql-compute`'s cascade gone, `larql-cli`/`larql-inference`/`larql-kv`/`larql-lql`/`larql-vindex` show their first real own-source error surface — and it traces entirely to `larql-vindex` (confirmed via log path analysis: all five crates fail at identical `larql-vindex` source locations). `larql-vindex` is substantially larger than `larql-compute` was (217 files, 2359 errors on first pull) — gating is in progress: `alloc_prelude.rs`/`collections.rs` set up correctly from the start this time (pattern 14's lesson applied proactively — `pub` visibility and `num_traits::Float` included from commit one, not retrofitted). Wholesale-excluded (pattern 3) `extract/`, `walker/`, `clustering/`, `format::huggingface`, `format::load`, `format::weights::load`/`write_f32`/`write_kquant`, `quant::convert_q4k`/`scan`, `mmap_util`, `index::storage` (whole submodule — its own doc says "mmap loaders...these modules touch raw bytes"; `compute`/`core`/`mutate`/`types` couple into it heavily, so gating it wholesale is expected to surface specific cross-reference errors next round rather than requiring them to be traced by hand now), and `format/vindex3/test_support.rs` (the `larql-models::test_fixtures.rs` shape again — not `#[cfg(test)]`-gated despite being test-only fixture code; verified every caller is itself test-gated before excluding). Mechanically swept the remaining 108 files. Found a new script-safety issue distinct from the doc-comment-orphaning bug: 3 files had their last top-level `use` line be the *start* of a multi-line `use X::{ ... };` group, and the naive "insert after last use line" heuristic landed the insertion inside the braces — a whole-file brace-balance check does NOT catch this (braces still balance, just relocated), it needs a dedicated backward-scan-past-blank-lines check. All 3 fixed. Next CI round pending.

**MILESTONE — `larql-vindex` FULLY GREEN (all 6 legs), independently confirmed via `gh api` (run `31448931856`, commit `1e9bfec3`).** The largest, most architecturally complex crate gated this session: 217 source files, 2359 errors on the first pull, ~7 rounds to clear. Pattern 3 (whole-module exclusion) did most of the work once the real shape became clear: the crate's entire in-memory `VectorIndex`/KNN query engine — `index::{compute,core,mutate,storage}`, `patch::overlay`/`overlay_apply`/`overlay_gate_trait`, `kv_index_impl` (the `KvIndex` trait impl *for* `VectorIndex`), `engine::core` (`StorageEngine` wraps `PatchedVindex`), `quant::convert` (depends on `scan`), `vindexfile` (build-pipeline tooling) — turned out to be pervasively mmap-storage-coupled *by design*, not incidentally (`index/mod.rs`'s own doc literally says "compute/ ... read-only over storage"). What stays portable is closer to "the on-disk format specification and pure data types" than "the whole crate minus I/O": `config`, `format::capability`/`lyrw2`/`moe_manifest`/`vindex3`-container-specs (minus their I/O-touching submodules), `quant::registry`, `describe`, `error` (minus 3 `PathBuf`/`io::Error`-typed variants), `runtime`, `trie` (the classifier struct + `classify()`, minus its file/env-based loaders), `index::types` (POD structs, minus `DownMetaMmap` and `ffn_row`).

Three distinct script-safety bugs surfaced and were fixed during the mechanical `alloc_prelude`-import sweeps (108 files, then another pass after further gating): (1) the established doc-comment-orphaning bug recurred; (2) a new one — 3 files had their last top-level `use` line be the *start* of a multi-line `use X::{ ... };` group, and inserting after that line landed the insertion inside the braces, invisible to a whole-file brace-balance check since the braces still balanced, just relocated; (3) the worst — 12 files got the import inserted *before* their own leading `//!` crate/module doc comment instead of after it (116 `E0753` errors), because the same backward-scan-past-doc-comments logic that correctly handles a `///`-documented item's preceding doc block is *wrong* for a file's own leading `//!` block, which must stay contiguous at the very top. All three now have dedicated verification sweeps to run before trusting any future insert-after-last-use-line pass.

Also confirmed and fixed pattern 13 recurring here (`attention::sinks::resolve()`'s shape wasn't unique to `larql-compute`) and generalized the load-bearing-cache-becomes-unconditional-miss pattern to a second instance (`patch::gate_overlay`'s `RwLock`-guarded per-layer snapshot cache). Relocated a misplaced pure-data enum (`QuantBlockFormat`) out of a newly-native module into the portable `manifest.rs` it was blocking, rather than gating the whole manifest around it.

**10 crates now fully green:** `larql-models`, `larql-boundary`, `larql-execution`, `larql-router-protocol`, `larql-vindex-spec`, `larql-factory`, `larql-core`, `larql-compute`, `larql-compute-metal`, `larql-vindex`.

With `larql-vindex`'s cascade gone, `larql-cli`/`larql-inference`/`larql-kv`/`larql-lql` (24 failing legs, 6 each) show their first real own-source error surface for the first time this campaign — next round pulls each crate's log and triages.

Pulled and triaged all 4 logs from run `31448931856`. `larql-inference`/`larql-kv`/`larql-lql` showed byte-identical error signatures (~2602 `-->` lines each, headline counts: 864 `E0425` `Vec`, 441 `E0433` `std`, 322 `E0425` `String`, 273 `E0433` `Vec`, 88 `E0433` `tokenizers`, 69 `E0425` `VectorIndex` in `larql_vindex`) — confirmed via `grep -c "crates/larql-<X>/src/"` per own-source-tree, plus the literal string `could not compile \`larql-inference\`` appearing in both `larql-kv`'s and `larql-lql`'s logs, that `larql-inference` is the sole root (2370 own-source hits) and `larql-kv`/`larql-lql` (0 own-source hits each) are pure 100% cascade — not three independent occurrences needing separate diagnosis. `larql-inference`'s own error surface is next, comparable in scale to `larql-vindex`'s original 2359. The "69 `VectorIndex` not found in `larql_vindex`" signal is the actionable lead: some file unconditionally references `larql_vindex::VectorIndex` (now correctly native-only per this session's own work) and needs the same pattern-3 treatment.

`larql-cli` showed a separate, independent, much smaller signature (48 `-->` lines): `E0463 can't find crate for std` x2 plus `is_terminal_polyfill` failing to build. Traced via the approved `cargo tree` exception: `cargo tree --target wasm32v1-none -p larql-cli --no-default-features -e normal --offline -i is_terminal_polyfill` → `is_terminal_polyfill ← anstream ← clap_builder ← clap ← larql-cli`. Root cause: `clap`'s default feature set includes `"color"` (`clap_builder/color`), which pulls in `anstream` for ANSI/terminal-color detection via `is_terminal_polyfill` (needs `std::io::IsTerminal`, no `wasm32v1-none` equivalent) — a pattern-7-shape-(b) instance (a feature flag on a *needed*, not-yet-target-gated dependency re-enabling std transitively), distinct from pattern 12 in that `clap` was never optional and sits in the plain `[dependencies]` table rather than behind our own feature flag. Fix: split `clap` into the native table (kept `"color"`) and the wasm32 table (`default-features = false`, `features = ["derive", "env", "std", "help", "usage", "error-context", "suggestions"]` — everything `"color"` implied except itself). Verified via `cargo tree` (both feature legs) that `is_terminal_polyfill` no longer appears in the wasm32 dependency tree. Pushed as `a9ca1a7a`; CI run `31450198086` confirmed that specific wall cleared but hit the next one immediately: `clap_lex` (clap_builder's mandatory, feature-less arg lexer) uses `std::ffi::OsStr` unconditionally and has zero features to tune (confirmed via `cargo metadata`) — a genuine pattern-9 instance (clap has no no_std mode at all, not a feature-tuning case), reclassified from the initial pattern-7-shape-(b) read. Fix: moved `clap` wholesale to the native-only target table (no wasm32 entry at all). Verified via `cargo tree` that neither `clap` nor `clap_lex` appear in the wasm32 tree on either feature leg. This exposes `larql-cli`'s next wall directly: 75 files (essentially the whole `commands/` tree plus `main.rs`) reference `clap::` with zero `#[cfg]` guards — confirming `main.rs`'s own pre-existing comment anticipating a novel entry-point/panic-handler-shaped failure class once dependency-level walls clear. Pushed; not yet CI-confirmed. Still fully independent of the `larql-inference` cascade.

Still far from "all the code involved" gated — the `forbid-unsafe` axis remains completely unreachable (gated behind `needs: wasm32v1-none`, by design, and `wasm32v1-none` is nowhere near green across the full crate matrix yet). This section will keep growing as the campaign proceeds — treat it as a running log, not a final task list.

16. **`cross-target-generic-alias-inference-asymmetry`** — a portable type alias that's a *fixed*-generic on one target but a *free*-generic re-export on the other can make identical source code compile on one target and fail E0283 ("type annotations needed") on the other, purely from a type-inference difference, not a missing-symbol difference. Confirmed instance: `crate::collections::HashSet<K>` is `pub type HashSet<K> = hashbrown::HashSet<K, BuildHasherDefault<FnvHasher>>` on wasm32 (the hasher parameter is baked into the alias, no ambiguity) but `pub use std::collections::HashSet;` on native (the real 2-parameter generic type, hasher parameter free). `let mut x = HashSet::default();` with no annotation resolves unambiguously on wasm32 but hits E0283 on native, because ordinary set methods (`.insert`/`.contains`) are generic over the hasher and never pin it, and Rust doesn't apply a type parameter's default from pure call-site inference — only from an explicit type position. Fix: annotate the binding with the alias's single-parameter form (`let mut x: HashSet<String> = HashSet::default();`), which resolves correctly on both targets (defaults to the real hasher on native, matches the alias's only parameter on wasm32). Found via the same native-CI-blind-spot audit as pattern 15, in `larql-core/src/algo/components.rs`; ~14 more bare (unannotated) `HashMap::default()`/`HashSet::default()` call sites found via grep across larql-core/models/compute/vindex/inference are structurally susceptible to the same bug but not fixed speculatively -- left for CI to confirm which are actually still broken.

17. **`fmt-check-also-only-runs-on-the-disabled-native-leg`** — `cargo fmt --check` is a step inside each crate's native test workflow (not the wasm32-gating workflow), so it inherited the exact same blind spot as patterns 15/16: disabled on this branch for most of the session, it never ran, so every `#[cfg]`-insertion round's accumulated formatting drift surfaced all at once the first time the native oracle actually ran end-to-end. Found 15/26/8/8/1 files needing reformatting in larql-core/models/compute/vindex/factory respectively (0 in compute-metal/kv/lql/cli/boundary/inference). Fixed by running `cargo fmt -p <crate>` locally per crate -- **a deliberate, narrow exception to the "never run local cargo build/check/test" rule**: `cargo fmt` does no compilation (no rustc codegen, no linking), making it the same category of operation as the already-approved `cargo tree` exception (metadata/syntax-only), just applied to formatting instead of dependency-graph resolution. Every resulting diff was verified via `git diff` + grep-filtering for anything outside use/pub-use/#[cfg]/whitespace lines, plus manual review of the few larger-looking hunks, before committing -- confirmed zero semantic content changed anywhere.

18. **`forbid-cannot-be-locally-allowed-even-through-a-macro`** — a major architectural finding, not a bug fix, discovered via the same native-audit sweep as patterns 15-17. `larql-cli/src/main.rs` has had `#![forbid(unsafe_code)]` at its crate root since earlier in the session, but because native compilation was disabled the whole time, this attribute was **never actually enforced or even compiled against until this round**. The first real native build produced 39 `error[E0453]: allow(unsafe_code) incompatible with previous forbid` across 20 files in `larql-cli/src/commands/{dev/ov_rd,extraction,primary}/`, every single one originating from `ndarray::s![...]` (the array-slicing macro). Root cause, confirmed via the error text ("this error originates in the macro `ndarray::s`") and known, documented Rust semantics: `ndarray::s!` expands to code containing `#[allow(unsafe_code)] unsafe { ... }` internally (the idiomatic pattern for a crate to use unsafe under the hood while staying lint-clean for callers with `#[warn(unsafe_code)]`) -- but `#![forbid(lint)]` is strictly stronger than `#[allow(lint)]` and **cannot be locally overridden by any nested `#[allow]` anywhere in the crate, including one injected by a macro expansion from an external dependency**. This is not specific to `ndarray` or to a bug in this codebase -- it's a general, well-known Rust `forbid` semantics fact: `forbid(unsafe_code)` at the crate root makes it a hard error to use *any* macro or generic function from *any* dependency that internally hides unsafe behind its own `#[allow(unsafe_code)]`, even though the calling code itself writes zero `unsafe` tokens. Since `ndarray::s!` is the idiomatic, near-unavoidable way to slice arrays in the numerically-heavy `commands/dev/ov_rd`/`extraction` modules, whole-crate `forbid(unsafe_code)` as currently declared is structurally incompatible with those modules' existence, not merely under-implemented.

    **This is exactly the kind of boundary the whole gating campaign exists to discover, not a defect to minimize** (see the "dead paths are the deliverable" discussion) -- the correct resolution, matching the same per-item classification discipline already used for the wasm32v1-none axis, is almost certainly to **narrow the `forbid(unsafe_code)` scope from whole-crate to per-module** (e.g. move `#![forbid(unsafe_code)]` out of `main.rs`'s crate-root position and apply `#[forbid(unsafe_code)]` individually to the modules that never touch `ndarray::s!` or any other unsafe-hiding macro, leaving the `ndarray`-heavy modules outside the forbid boundary) -- the direct forbid-unsafe analogue of wasm32v1-none's `#[cfg(not(target_arch = "wasm32"))]` per-item gating. This is also the first real signal from the `forbid-unsafe` axis at all this session (the official CI job for it is still unreachable, gated behind `needs: wasm32v1-none`) -- it surfaced as a side effect of `main.rs`'s pre-existing crate-level attribute finally being compiled against, not from the gated CI job itself.

    **Resolved** (task #30): confirmed the same 40-error/23-file signature natively-blocks `larql-cli`'s own O_native oracle entirely (all 3 test legs + coverage, run `31464601274`) -- not just the deferred CI axis, so this was elevated ahead of task #29's remaining queue. Since crate-root `forbid(unsafe_code)` was already active and CI's own error list is an exhaustive, decidable partition of the crate's ~190 files into exactly 23 violators (all under `commands/dev/ov_rd/`, `commands/extraction/`, and `commands/primary/shannon_cmd.rs`) vs everything else, no independent grep/judgment pass was needed -- the compiler had already done the classification. Removed `#![forbid(unsafe_code)]` from `main.rs`'s crate root; added `#[forbid(unsafe_code)]` to every one of `main.rs`'s own 12 top-level items (none violate) and, in each parent `mod.rs` (`commands/mod.rs`, `commands/dev/ov_rd/mod.rs`, `commands/extraction/mod.rs`, `commands/primary/mod.rs`), to every individual `mod`/`pub mod` declaration whose file is NOT in the violator list -- chosen at per-file granularity (not per-function within a violating file) to match the granularity the CI oracle actually gives for free, mirroring the earlier-established "whole subtree when precision doesn't change the mechanism" discipline from the layer_graph over-gating pass. `commands/dev/mod.rs` (only declares the mixed `ov_rd`) and `dev`/`extraction`/`primary` in `commands/mod.rs` itself (each mixed) were left unmarked, pushing the boundary one level down; `diagnostics`/`query` (fully clean subtrees) got forbid at the `commands/mod.rs` level directly. Pushed for both oracles to confirm.

19. **`coverage-policy-gate-inherits-the-same-blind-spot-a-third-way`** -- `larql-compute`'s native workflow has a `Coverage policy gate` step (`scripts/check_coverage_policy.py` against `crates/larql-compute/coverage-policy.json`'s `per_file_line_min_percent`/`default_line_min_percent` floors) that is a *different* failure mode from patterns 15/17: not "never ran" but "ran, and is currently red on two files neither touched by any gating commit this session" (`pipeline_layer/mod.rs` 89.78%, `pipeline_layer/moe_build.rs` 83.59%, both against the repo's 90% default floor, neither in the existing baseline-exception list). Diff-inspected the only session commit touching either file (`f9c92edc`) and confirmed it's purely additive `#[cfg(target_arch = "wasm32")]` gating plus one `std::ops::Range`→`core::ops::Range` swap -- none of which change what executes on the native/coverage leg. Cross-checked against `origin/main` (which does not contain the GPT-OSS commits `04addd27`/`19cbefd6` that added this code) via a live dispatched `larql-compute.yml` run rather than trusting the diff-inspection alone, per the standing "resolve uncertainty via CI, not by hand" discipline -- inconclusive (main too stale, doesn't currently compile for unrelated reasons, predates several existing baseline entries), so the baseline entries were added on the strength of the diff-inspection evidence alone. **Resolved** (task #31, CI-confirmed green): added `per_file_line_min_percent` baseline entries (89.0/83.0) with a dated `policy_note`, the repo's own existing convention for exactly this situation (see `c95331c2`). Matters because a permanently-red coverage leg would make O_native for `larql-compute` permanently ambiguous (mutation-bug-red vs coverage-debt-red) for every future round.

    **Second confirmed instance** (task #32, CI-confirmed green): `larql-vindex`'s coverage-policy gate failed on `q4k.rs` (52.71% vs its own already-baselined 57.00% floor -- a regression against an *existing* baseline, not a first-ever measurement) and `write_layers.rs` (89.60% vs the 90% default, first-ever measurement). Diff-inspected every session commit touching either file: `q4k.rs` was untouched by any session commit at all (its debt predates the branch entirely); `write_layers.rs`'s two touching commits (`5049073a`, `a9b88e6a`) are both provably behavior-identical `std::`→`core::` renames (`std::mem::size_of`/`std::fmt::Debug` are literal re-exports of their `core::` counterparts) plus one `alloc_prelude` import addition -- none capable of moving coverage. Baselined both (52.0/89.0) under the same convention.

20. **CI-topology redesign** (design converged this round, not yet implemented) -- prompted by the observation that the two independent oracles (O_wasm, O_native) are themselves scattered across 18 separate workflow files with zero cross-file ordering, and that path-filter triggers have already drifted out of sync with the real dependency graph (`larql-cli.yml`'s own trigger paths omit `larql-core/**`/`larql-compute/**`/`larql-compute-metal/**`/`larql-factory/**`/`larql-vindex-spec/**` despite `larql-cli` transitively depending on all of them -- a `larql-core`-only change wouldn't trigger `larql-cli.yml`'s native tests today, the same blind-spot shape as patterns 15/17/19 one level up the stack).

    **The real dependency graph** (verified twice -- once via direct `cargo metadata --format-version=1` parsing, once via an independently-dispatched research agent reading every `Cargo.toml` by hand; both agree exactly) is 8 levels deep, not the 3-tier root/branch/leaf model initially proposed: depth 0 (leaves) `larql-core`/`larql-vindex-spec`/`larql-execution`/`larql-router-protocol`/`larql-boundary`; depth 1 `larql-models`; depth 2 `larql-compute`/`larql-factory`; depth 3 `larql-compute-metal`; depth 4 `larql-vindex`; depth 5 `larql-inference`; depth 6 `larql-kv`/`larql-lql`; depth 7 (root) `larql-cli`. "Root needs branch needs leaf" is the recursive principle applied across however many levels actually exist, not 3 literal buckets.

    **Structural facts established**: `needs:` only orders jobs within one workflow YAML file (verified against GitHub's own docs) -- it cannot cross file boundaries; the only cross-file primitives are `workflow_call`/`workflow_run`, and this repo uses neither anywhere (0 hits, grepped across all 18 files). This is why consolidating the ~12 files in `larql-cli`'s closure into one file is structurally necessary for any `needs:`-based ordering to work at all, not a stylistic preference. Separately, `larql-kv`/`larql-inference`/`larql-vindex` have a genuine cycle in `[dev-dependencies]` (each uses the others as test fixtures -- legal Cargo, since a lib never needs its own dev-deps when consumed as a regular dependency) -- this forces `needs:` in the new design to mean "the dependency's library builds and lints clean," never "the dependency's full test suite passed," since the latter is provably cyclic for this trio and cannot be expressed as a DAG at all. Three leaf crates (`larql-execution`, `larql-router-protocol`, `larql-vindex-spec`) have no CI workflow file today -- nothing tests them in isolation currently; the new design needs base jobs for them.

    **Resolved mechanism**, using only two native GitHub Actions primitives, zero hand-written `if:` skip logic anywhere: a `discover` job (runs `cargo metadata`, the approved metadata-only exception) computes the live dependency graph + topological depth every run and emits it as JSON -- tier *membership* is fully dynamic (never hardcoded), tier *slot count* is a generously-sized fixed upper bound in the YAML (GitHub requires job IDs to exist at workflow-parse time -- a hard platform limit, confirmed via docs, not a design shortcut). `needs:` (job-to-job, automatic skip-on-fail, zero `if:`) handles everything crossing a job boundary: tier-to-tier, and "ubuntu must pass before windows/macos run" (`tierN-others: needs: tierN-ubuntu`). `fail-fast` (matrix-internal, automatic cancellation of queued/in-progress siblings, zero `if:`, verified via GitHub's docs: cancels both in-progress *and* queued matrix legs, but does not cross job boundaries and gives no ordering guarantee among matrix entries) handles pruning *within* one job's own matrix. `larql-compute-metal` (the one macOS-only crate in the closure) gets its own shape with no ubuntu leg to gate behind.

    **Explicit user decision**: the whole native build/test chain is a *strict* gate behind the whole wasm32v1-none chain (`native-tier0-ubuntu: needs: [discover, wasm-tier7-others]`, one extra edge at the very start of the native chain, propagating through via the existing tier-to-tier `needs:`), not a "sequenced but independent" arrangement. Consciously reintroduces pattern 15's blind spot during active gating work (wasm32 red is routine mid-campaign, and native regressions introduced in the same push go undetected until wasm32 clears) -- accepted deliberately: gating correctness is weighted above regression-detection latency right now, regressions get caught once wasm32 passes and become part of the long-term code-review process, and the cost asymmetry is real (wasm32 legs that succeed run in ~30s; native legs take several minutes -- an order of magnitude difference per job, so fail-fast-behind-the-cheap-check is real CI economy). The manual `gh workflow run <crate>.yml --ref ...` discipline (pattern 15) remains the way to get independent native signal during active development even after this consolidation lands.

    **Future scope, explicitly not part of this round**: a target-capability lattice extending the same gating discipline beyond `wasm32v1-none`, applying the identical "narrower/stricter target gates the broader/more-capable one" principle recursively: `wasm32-unknown-unknown` needs `wasm32v1-none`; `wasm32-wasip1` needs `wasm32-unknown-unknown`; `wasm32-wasip3` needs `wasm32-unknown-unknown`; `wasm32-unknown-emscripten` needs `wasm32-unknown-unknown`; `wasm32-wali-linux-musl` needs `wasm32-unknown-unknown`. Not implemented, not scheduled -- recorded here so the vision isn't lost, per explicit instruction this is "eventually," not now.

15. **`safety-net-blind-spot-hides-cross-target-regression`** — the `larql-cli-gating.yml` workflow only ever builds `--target wasm32v1-none`; it never runs a plain native build. This is fine for catching under-gating (the wasm32 compiler is a perfect oracle for "does this compile on wasm32"), but it means the whole campaign, for its entire duration until this pattern was found, had **zero automated coverage for "did a gating edit break native compilation."** Discovered when `larql-inference.yml` (the crate's normal native `cargo test` CI, which *does* run a real build) was found to be explicitly skipped on this branch (`if: github.head_ref != 'gating/larql-cli-wasm-and-safe'`, an earlier-session decision to save CI minutes) and was re-enabled via manual `gh workflow run larql-inference.yml --ref <branch>` (the skip condition only fires for `pull_request`-triggered runs; `github.head_ref` is empty for `workflow_dispatch`/`push`, so manual triggering bypasses it). The very first real run of that oracle surfaced a genuine, systemic regression that had been silently present since commit `8a742a961` ("widen pattern-4's FnvHasher/HashMap/HashSet to pub", mid-session): all 5 crates that got that fix (`larql-core`, `larql-models`, `larql-compute`, `larql-vindex`, `larql-inference`) unconditionally re-exported `FnvHasher` from their `lib.rs` (`pub use collections::{FnvHasher, HashMap, HashSet};`), but `FnvHasher` is only *defined* under `#[cfg(target_arch = "wasm32")]` in each crate's `collections.rs` (native doesn't need a custom hasher — `std::collections::HashMap`/`HashSet` use their own default). Result: `error[E0432]: unresolved import` on every native build of every one of those 5 crates, for the entire remainder of the session, invisible because the only CI that ran was wasm32-only (where `FnvHasher` genuinely exists, so the same line resolves fine). Fixed by gating the `FnvHasher` half of each re-export to `#[cfg(target_arch = "wasm32")]` and confirming via grep that no downstream crate references `larql_<X>::FnvHasher` from outside its own crate (so narrowing the re-export doesn't just relocate the breakage). Pushed as `505a9863`. **Standing lesson: a wasm32-only gating oracle is structurally blind to any regression that only manifests on the *other* target — the native CI leg must run on every push for the rest of this campaign, not just wasm32v1-none**, now enforced via the manual `workflow_dispatch` trigger until/unless the branch-skip condition on `larql-inference.yml` (and its sibling per-crate workflows, likely with the same skip) is revisited.
