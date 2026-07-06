# Plan: Integration — chrishayuk base ← ianblenke head (additive, sync-preserving merge)

## Context

Converge the two forks onto **one codebase that keeps chrishayuk's architecture** — the Metal
mega-split (`larql-compute` → `larql-compute-metal`), `larql-vindex-spec` extraction, GGUF
modularization — **while preserving ianblenke's genuine fixes and features** (DeepSeek V4,
Qwen 3.5, speculative decoding, CUDA backend, KV accuracy harness). This is the **PR #129
direction** (base `integration/chrishayuk-main` ← head `integration/ianblenke-main`).

Mechanism: a **single `git merge`** (preserves both forks' commits pristine → bidirectional
`git merge upstream/main` / `ianblenke/main` keep working; brings *all* commits so no cascade), with
**all reconciliation as additive atomic conventional commits on top**. (Cherry-pick was rejected —
consumer/provider cascade, 15/123 applied, build broke; commit-rewriting was rejected — severs
upstream sync.)

## Why cherry-pick was rejected (full evidence, consolidated from the 2026-06-08 candidate review)

Candidate branch `integration/three-way-candidate` (local only, never pushed): ianblenke/main @
`e5e2a905` as base, chrishayuk's unique commits cherry-picked with conflicts deferred.

**Result:** 15/123 (12%) of chrishayuk's commits applied cleanly; 107/123 (87%) deferred on
conflict; 1 already present. **All of chrishayuk's actual substance — the Metal split
(`larql-compute-metal`), KV unification, performance work, Granite vision, GGUF modularization —
is in the 107 deferred**, not the 15 applied. The 15 that landed are trivial (license pin, CI
config, a 32-bit overflow guard, GGUF-input acceptance, remote-FFN norm wiring) — cosmetically
clean, substantively empty.

**The headline finding, stated precisely:** composing only non-conflicting commits cannot
integrate two forks that both rewrote the same code. This is inherent to the "non-conflicting
first" rule, not an execution bug — it is the structural mirror of the complaint that agents keep
dropping ianblenke's work; applied to chrishayuk-onto-ianblenke, the same rule instead drops
chrishayuk's substance.

**Concrete proof the 15 "clean" cherry-picks don't even yield a buildable tree:**
`cargo build --workspace -j2` failed (exit 101) on `error[E0425]: cannot find function
write_model_weights_kquant_with_opts in crate larql_vindex` — `larql-cli` fails to compile. Root
cause: cherry-pick `62ce88f8` ("fix(extract): accept GGUF input") is a `larql-cli` consumer commit
that applied cleanly, but the `larql-vindex` provider commit defining the function it calls
(chrishayuk's `write_kquant/mod.rs`) was deferred as a conflict. Consumer applied, provider
deferred → dangling reference → break. This is a real, composition-induced failure, not an
environment or pre-existing issue (the pure ianblenke base is unaffected).

**Caveat on the 15/107 split (don't over-read it):** the cherry-pick loop aborts each conflicting
commit and tries the next against a HEAD still missing it, so deferring commit N cascades to N's
dependents. 107 is a floor on composability under this specific procedure, not a clean measure of
the true simultaneous conflict surface — a single `git merge` (the approach actually adopted,
per this plan) shows that without the cascade artifact.

**Still-open decision this review flagged, not yet resolved anywhere:** scratch candidates present
only in the ianblenke base — `RESUME_PROMPT.md`, `RESUME_PROMPT_SESSION_2026-05-14.md`, agent-config
dirs (`.opencode/`, `.kilocode/`, `.gemini/`, `.codex/`, `.claude/`), and `openspec/` (84 files —
flagged as possibly a real spec system, not scratch, verify before discarding) — keep or remove is
explicitly "for your call," not decided by this pass or the merge pass since. Resolve this when
resuming reconciliation, not before.

## Status — Phase 0–1 DONE (this session)

- **Merge stood up** in isolated worktree `/home/metavacua/larql-integration-wt` (branch
  `integration/chrishayuk-base`, off `upstream/main` @ cac5c8e0, **not committed**). Upstream
  commits pristine.
- **96 conflicts**, matching prediction: **56 content (UU) + 35 modify/delete (21 UD + 14 DU) +
  5 add-type (1 AA + 4 UA)**.
- **Features came in free:** 138/138 ianblenke model files (DSv4, Qwen3.5, speculative, CUDA)
  present in the merged tree; **only 1 conflicts** (`larql-models/src/quant/ggml/q5_k.rs`).
- **Both baselines green:** `cargo check --workspace` is clean (0 errors / 0 warnings) on
  chrishayuk *and* ianblenke independently → **any merged failure is integration-caused**, and we
  have two green targets to converge toward.
- **Retarget map + dependency-ordered resolution queue produced** →
  `~/larql-integration-resolution-queue.md`.
- Reference worktrees up for per-file version comparison: `larql-chris-wt`, `larql-ian-wt`.

## Dependency-graph delta — the structural finding

The two forks' internal crate graphs are the **same shape** except for **one node**:

- **ADD `larql-rotorquant`** — an ianblenke leaf crate (no internal deps). `larql-inference` and
  `larql-server` depend on it with **normal** edges; `larql-compute` optionally. Proven buildable
  (it's in the green ianblenke baseline). chrishayuk lacks it entirely → the crate's files merge
  additively; only its **3 dependency edges** need wiring.
- **KEEP** chrishayuk-only nodes: `larql-compute-metal`, `larql-vindex-spec`, `larql-boundary`.
- **`larql-kv → larql-inference` is identical in both forks** (normal dep; `inference→kv` is
  dev-only) → no reconciliation; no cycle.
- **DROP** `kv-cache-benchmark` (chrishayuk promoted its accuracy suite into `larql-kv`).

So the graph integration = *add rotorquant + wire 3 edges*; everything else is in-file content, not
graph restructuring. **The resolution order is exactly the topological layering** (confirmed):
`models → rotorquant → compute → {compute-metal, vindex-spec, vindex} → inference → {kv, lql,
server} → {cli, python}`, then drop kv-cache-benchmark / fold accuracy into kv.

## Reconciliation surface (verified)

The 35 modify/delete have **0 git-trackable renames** → all need semantic retargeting. Retarget
map (the 14 DU — chrishayuk deleted, ianblenke modified):

| ianblenke file(s) | → chrishayuk destination | mechanism |
|---|---|---|
| `kv-cache-benchmark/src/accuracy_suite/*` (9) | `larql-kv/src/accuracy_suite/*` (+ `examples/`) | content merge |
| `larql-inference/src/vindex/q4k_forward/*` (3) | `larql-compute/src/kquant_forward/` | substrate move (ADR-0022) |
| `larql-models/src/loading/gguf.rs` (1) | `larql-models/src/loading/gguf/{loader,parser,reader,writer,…}` | GGUF modularization |
| `larql-server/src/routes/walk_ffn.rs` (1) | `larql-compute/src/kquant_forward/walk_ffn.rs` | substrate move |

The 21 UD (chrishayuk modified, ianblenke deleted) = **keep chrishayuk's version**; ianblenke's
work in those areas arrives via the UU conflicts / additive files.

Conflict concentration (both forks reworked the same files = competing):

| crate | conflicts | nature |
|---|---|---|
| larql-vindex | 27 | competing — real reconciliation |
| larql-inference | 27 | competing (most ian commits are additive DSv4/spec; 27 are the true overlap) |
| larql-models | 7 | competing — gates features |
| larql-compute | 7 | Metal-split battleground |
| kv-cache-benchmark | 9 | leaf — fold accuracy into larql-kv, drop crate |
| server/cli/lql/router-protocol | ~6 + adds | downstream consumers |

## Remaining work — staged reconciliation (resume here)

All in the existing worktree. Each conflict → **one additive atomic conventional resolution
commit**. **`cargo check` after each stage** to converge toward the green baselines.

1. **larql-models + workspace wiring (foundation).** 7 UU + the `gguf.rs`→`gguf/*` modular retarget
   + the `q5_k.rs` feature conflict + **root `Cargo.toml`** (add `larql-rotorquant` member, drop
   `kv-cache-benchmark`) + **`Cargo.lock`** union. Gates the additive features.
2. **larql-rotorquant.** Confirm the additive crate is present; no conflict expected.
3. **larql-compute (7 UU).** Honor the Metal split; add the `rotorquant` (opt) edge; receive the
   `q4k_forward` + `walk_ffn` retargets into `kquant_forward/`.
4. **larql-vindex (27 UU + 14 UD-keep-chrishayuk).**
5. **larql-inference (27 UU + gpu UD-keep + q4k DU).** Add the `rotorquant` (normal) edge.
6. **Downstream consumers:** larql-server (add `rotorquant` edge), larql-cli, larql-lql,
   larql-router-protocol (~6 + add-type).
7. **larql-kv accuracy reconciliation (last).** Fold ianblenke's `kv-cache-benchmark/src/
   accuracy_suite` (+770 lines) into chrishayuk's existing `larql-kv/src/accuracy_suite/`; optionally
   re-home example harnesses to `larql-kv/examples/`; the crate itself stays dropped.

On breakage: retarget / defer (surface the decision) or the permitted reconciliation surgery —
never invent unrelated code.

## Critical files / areas

- Root `Cargo.toml` + `Cargo.lock`, and `crates/{larql-compute,larql-inference,larql-server}/Cargo.toml`
  — the rotorquant-edge wiring.
- `crates/larql-vindex/src/**` and `crates/larql-inference/src/**` — the two 27-conflict competing cores.
- `crates/larql-compute` ↔ `crates/larql-compute-metal`; `crates/larql-compute/src/kquant_forward/`
  — Metal-split honor + the q4k_forward/walk_ffn retarget home.
- `crates/larql-models/src/{config.rs, lib.rs, detect/*, loading/gguf/*, quant/ggml/q5_k.rs}` — the
  feature-gating conflicts.
- `crates/larql-kv/src/accuracy_suite/*` — the accuracy reconciliation home.

## Verification

- **Per stage:** `cargo check --workspace` in the worktree trends from 96-conflict → green.
- **Final build:** `cargo build --workspace` clean; both GPU backends (Metal + CUDA) compile;
  DSv4/Qwen3.5/speculative crates build.
- **Upstream sync intact (load-bearing):** `git merge-base --is-ancestor upstream/main HEAD` and
  `… ianblenke/main HEAD` both true; a dry `git merge --no-commit upstream/main` finds the real base.
- **No upstream commit rewritten:** the merge's two parents (`upstream/main`, `ianblenke/main`)
  keep their original SHAs.
- **Changelog:** every resolution commit is conventional-commit; `git-cliff` over `<merge>..HEAD`.

## Non-actions / safety

- Isolated worktree; integration branch only. Do **not** replace or force-push `main`; do **not**
  rewrite any upstream commit; do **not** modify PRs beyond (optionally, when ready) updating #129.

## Resolved decisions

- **Base = chrishayuk, head = ianblenke** (PR #129 direction).
- **`kv-cache-benchmark`:** drop the crate; reconcile ianblenke's accuracy improvements into
  chrishayuk's existing `larql-kv/src/accuracy_suite/` modules (already the chrishayuk structure).
- **`larql-rotorquant`:** add as a workspace crate + wire its inference/server (normal) and compute
  (opt) edges.
- **CUDA:** keep ianblenke's CUDA as a **dormant, fully-gated feature inside larql-compute** this
  pass (cuda-off build is green/verified here; no local CUDA toolkit). The `larql-compute-cuda`
  sibling-crate extraction (mirroring larql-compute-metal) is **deferred to a CUDA/CI branch** where
  `nvcc` exists — `cargo check --features cuda` needs nvcc (rotorquant's build.rs compiles a `.cu`
  kernel), which this machine lacks. Tracked follow-up; do NOT extract blind here. So in the 4 crate
  Cargo.toml conflicts: keep chrishayuk's metal-as-sibling structure, ADD ianblenke's dormant
  `cuda`/`cuda-oxide` features + the rotorquant dep, SKIP ianblenke's in-`larql-compute` `metal`
  feature, and preserve chrishayuk's Android-safe choices (e.g. plain `ndarray`).

## Artifacts

- Worktree (in progress): `/home/metavacua/larql-integration-wt` — branch `integration/chrishayuk-base`.
- Resolution queue + retarget map: `~/larql-integration-resolution-queue.md`.
- Green reference checkouts: `/home/metavacua/larql-chris-wt`, `/home/metavacua/larql-ian-wt`.
