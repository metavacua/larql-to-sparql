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
- No local `cargo` builds, ever (same constraint as the rest of this plan) — TOML/source edits are validated with `python3 -c "import tomllib; ..."` / by inspection, never compiled locally.

**Pattern taxonomy (living list — append as new patterns are found):**

1. **`dependency-default-std-feature`** — a third-party dependency defaults to a Cargo feature (usually named `std`) that pulls in the `std` crate, which doesn't exist on `wasm32v1-none` (core+alloc only). Symptom: `error[E0463]: can't find crate for `std`` inside the *dependency's* own source, which then cascades into hundreds/thousands of downstream "cannot find X in this scope" errors as everything depending on the now-missing `std`-gated code fails to resolve. **Do not assume a fix exists** — verify the dependency actually supports a `no_std`/`alloc` mode (check docs.rs features, don't guess) before applying `default-features = false`. Confirmed instances: `serde`/`serde_json` (has `alloc` feature), `thiserror` (only feature is `std`, default-features=false is its documented no_std path, derive macro unconditional). Fix shape: split the dependency into `[target.'cfg(not(target_arch = "wasm32"))'.dependencies]` (workspace-inherited, unchanged) and `[target.'cfg(target_arch = "wasm32")'.dependencies]` (independent declaration, `default-features = false` + alloc-compatible features).
2. **`own-crate-missing-no_std`** — none of the crates in the `larql-cli` build/test graph declare `#![no_std]` (correction: `crates/larql-experts/expert-interface` does, but it targets `wasm32-wasip1`, a std-capable WASI target, for an unrelated reason — not a precedent applicable here), so rustc injects the `std` prelude implicitly for every crate regardless of target, producing the same `E0463`/cascade pattern as pattern 1 but for *our* source once dependency-level blockers are cleared. Fix: `#![cfg_attr(target_arch = "wasm32", no_std)]` + `#[cfg(target_arch = "wasm32")] extern crate alloc;` at the crate root. **Known landmine:** `larql-core` has its own `pub mod core;`, which already shadows the real `core` crate for unqualified `core::` paths inside that crate (pre-existing, unrelated to this change) — any *new* no_std-migration code added there needs `::core::...` (leading `::`) to reach real libcore.
3. **`native-only-io-module`** — a module's functionality has no core/alloc equivalent at all (e.g. `std::fs`-based file I/O — there is no filesystem on `wasm32v1-none`). Not a feature-toggle fix like pattern 1; the whole module gets excluded via `#[cfg(not(target_arch = "wasm32"))]` on its `pub mod` declaration and every `pub use` re-exporting it. Confirmed instance: `larql-core::io` (checkpoint/csv/json/packed, all `std::fs`/`std::io`). A single enum *variant* can need the same treatment even when the enum itself stays available: `GraphError::Io(#[from] std::io::Error)` — `std::io::Error` has no core/alloc equivalent, but only the excluded `io` module ever constructs that variant (confirmed via grep before gating it), so `#[cfg(not(target_arch = "wasm32"))]` on the variant alone was correct and didn't need to touch the rest of `GraphError`.
4. **`std-collection-needs-no-std-hasher`** — `std::collections::HashMap`/`HashSet` have no core/alloc equivalent because they need a `BuildHasher`, and `alloc` provides no default one (std's `RandomState` seeds itself from OS randomness). `hashbrown` is genuinely `no_std`-capable, but **its own default hasher has the identical requirement** — verified via its Cargo.toml before relying on it, not assumed. Fix: `hashbrown` with `default-features = false` (+ `features = ["serde"]` if the map/set needs `Serialize`/`Deserialize` — also lost when defaults are disabled) plus a fixed-seed deterministic hasher (FNV-1a, hand-rolled, ~15 lines — matches this exact codebase's prior wasm32 attempt, PR #71's "FNV wasm32"). Wrap in a small crate-local module (`src/collections.rs`) exporting portable `HashMap`/`HashSet` type aliases (std-backed natively, hashbrown-backed on wasm32) so call sites `use crate::collections::{HashMap, HashSet}` once instead of repeating the cfg-split per file. **Landmine:** hashbrown's `HashMap::new()`/`HashSet::new()` are inherent methods defined only for the crate's *own* default hasher (same as std, whose `new()` only exists for `RandomState`) — a custom-hasher type alias must use `.default()` instead (works uniformly on both native and wasm32, since `Default` is blanket-available for any `S: Default`). Grep for `HashMap::new()`/`HashSet::new()` *and* bare fn-pointer references (`HashMap::new` with no parens, e.g. passed to `.get_or_insert_with(...)`) — a plain call-site `sed` misses the latter.
5. **`alloc-prelude-types-not-implicit`** — `#![no_std]` drops std's prelude, which auto-imports `Vec`/`String`/`Box`/`ToString`/`ToOwned`. `alloc` provides these but does not inject them — same as any other type/trait, they need an explicit per-file `use`. Separately, `#[macro_use] extern crate alloc;` at the crate root *does* restore crate-wide availability of `vec!`/`format!` (macros and types are different namespaces in Rust's prelude mechanism, so this doesn't cover the value types above). When many files need the same 3-4 alloc imports, a small `src/prelude.rs` (empty on native, re-exports on wasm32) collapses each call site to one line (`#[cfg(target_arch = "wasm32")] use crate::prelude::*;`) instead of repeating the same import block everywhere.
6. **`core-f64-lacks-transcendental-math`** — `core`'s `f64`/`f32` have no `ln`/`exp`/`sqrt`/`sin`/`cos`/etc. — those need a libm implementation, and `std` links against the platform's. The standard no_std-compatible replacement is the `libm` crate (pure Rust, `default-features = false`), used via free functions (`libm::log(x)` for `ln`, etc.) rather than the method-call syntax `std`'s inherent impls provide.
7. **`transitive-feature-reenables-std`** — generalizes pattern 1: it's not enough for *our* direct dependency declaration to disable a crate's `std` feature if a *different*, un-gated dependency in the graph re-requests it. Two confirmed shapes: (a) an entirely un-gated dependency that itself needs full std (`safetensors`/`memmap2` in `larql-models`, transitively pulling in std-enabled `serde` — Cargo's feature unification re-enables `std` for the whole wasm32 build regardless of our own direct `serde` override); (b) a *feature flag* on an already-target-gated dependency re-enabling std in a third crate (`ndarray`'s default `std` feature does `std = ["num-traits/std", "matrixmultiply/std"]` — even with `ndarray` itself otherwise fine, its default features silently poisoned two more crates). Fix for (a): move the un-gated dependency into the same target-gated table. Fix for (b): find and disable the specific feature (read the dependency's own Cargo.toml on its repo, don't guess) — `default-features = false` on `ndarray` for wasm32 resolved both `matrixmultiply` and `num-traits` in one change, confirmed via CI, without needing to gate out any `ndarray`-using code.

**Status as of this plan revision — two real milestones:**
- `larql-core --no-default-features` compiles cleanly to `wasm32v1-none` on **all three OSes** (run `31404649309`) — patterns 1-6, 8 real CI round-trips (`fc65bdcb`..`a2df440b`).
- `larql-vindex-spec` compiles cleanly on **all 6 legs** (both feature sets, all 3 OSes) — patterns 1+2+5 applied proactively in one commit (`2d4ca048`) rather than rediscovered via CI, since it has none of `larql-core`'s harder blockers (no `HashMap`, no `io` module, no transcendental math).
- `larql-models`'s entire external-dependency graph is now clear (patterns 1, 2, 7 — commits `965146b2`..`6a234922`); compilation reaches its own 666 errors next, the same "own-source" milestone `larql-core` hit before patterns 2/5/6 got it to green. Not yet fixed — this crate is filesystem/mmap-heavy throughout (`loading/`, `detect/`, `connectors/`, `encoders/`, `speech/`, `weights/`), very likely needs several pattern-3-style whole-module exclusions, deliberately not guessed at without compiler feedback (see the commit message for the cross-module-reference risk).
- `larql-factory` similarly has its dependency-level patterns applied (1, 2, plus `reqwest`/`serde_yaml` target-gated per pattern 7's lesson, commit `3a7ad09b`) but not yet its own-source errors.
- `larql-core --default-features` (pulls in `reqwest`/`rmp-serde`) still fails on all three OSes — not yet investigated.

Pattern 1 remains staged (uncommitted) for `larql-kv`, `larql-vindex`, `larql-inference`, `larql-cli` — not yet re-verified against patterns 2-7. `larql-cli` itself is a `[[bin]]` target whose `main.rs` has no `#![no_std]` and won't get one in this pass — it is expected to keep failing on its own implicit std usage even after every dependency/crate-level pattern is cleared; that's the real, final portability boundary this whole exercise exists to find, not a bug to route around.

Still far from "all the code involved" gated — `larql-compute`, `larql-compute-metal`, `larql-inference`'s many native-only deps (`tokio`, `wasmtime`, `reqwest`, `rayon`, `tokenizers`, GPU/Metal), `larql-lql`, and the `forbid-unsafe` axis (currently unreachable behind `needs: wasm32v1-none`, by design) are all still ahead. This section will keep growing as the campaign proceeds — treat it as a running log, not a final task list.
