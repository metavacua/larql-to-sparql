# WASM32V1-NONE campaign — project context & handoff

> Working context for the `wasm32v1-none` portability effort. Living scratch doc
> (currently **untracked** — not committed, not in any PR). Last updated during
> the MVV + CI-coverage session.

## 1. North star

Decompose the larql-to-sparql workspace's OS/IO assumptions from its portable
compute, so a **portable compute kernel** runs on `wasm32v1-none` (WASM MVP 1.0
+ mutable-globals; `no_std`; no-OS; no imports; no atomics/SIMD). Gate **only**
what genuinely cannot compile there; verify we did not over-gate. Native must
keep building, byte-identical in behavior to the reference originals.

The work lives in a **nested Cargo workspace** `crates/larql-wasm/` (separate
from the root workspace). Each of 16 original crates was copied into a
`CRATE-wasm32v1-none` (portable kernel) and a `CRATE-interface` (native I/O
adapter) variant, joined by `larql-bridge` traits (`WeightProvider`, `KvStore`,
`ExpertDispatch`, `HttpFetch`). Originals under `crates/<name>/` are invariant
references — never edited.

## 2. Conceptual framework (developed this session)

"wasm-safe" is **not a per-crate boolean** — it's a *lattice* of WASM-subset
dialects, and the unit is the *portable subset within* each native crate.

Three independent, **statically checkable** axes:
1. **Capability** (imports): does the reachable call graph touch a host import?
   — sandbox escape. Measured on the *compiled* module (wasmparser).
2. **Computational power / termination** (`T ⇔ L∧M`): unbounded iteration `L`
   (`loop`/back-`br`/recursion/`call_indirect`) **and** unbounded memory `M`
   (`memory.grow`) together ⇒ Turing/intractable. Safe ⇔ `¬(L∧M)`; the two
   safe fragments (loop-free; bounded-memory) are dual, incomparable, and
   **don't compose** (linking a loop-fragment to a memory-fragment can re-create
   `L∧M`). `call_indirect` is the key escape (arbitrary code execution + breaks
   static call-graph), hence the focus on it.
3. **Arithmetic definability** (`Q_fin`): a syntactic, alias-closed scan for the
   7 generators of bounded **Robinson Q** (Sx≠0, S injective, predecessor, +/·
   base+recursion). Missing any one ⇒ the module is in a decidable safe fragment
   (`S_mul`≈Presburger, `S_add`≈Skolem, `S_loop`=loop-free, …). Detection is
   purely static on the un-executed blob; **safety-critical generators are
   uniquely generated** (`call_indirect`←ACE; imports←host; `memory.grow`←
   growth), so single-construct absence is a complete guarantee without modeling
   dynamic behavior. (Multiply-realizable arithmetic generators need the
   alias-closure; that's the definability classification, not the safety hinge.)

The certified target = **WASM-valid ∩ wasm-safe ∩ Q-classified ∩ total**. The
~"5–7 DSLs" are points in feature × environment × std-availability space:
`wasm32v1-none` (the floor) ⊂ unknown-unknown(+wasm-bindgen) ⊂ {+simd128,
+atomics, wasip1, emscripten}. **Effectiveness/correctness first; efficiency
(optimized kernels) is a later, dialect-stratified layer over an invariant
format.**

## 3. Branches & PRs (origin = metavacua/larql-to-sparql; gh needs `--repo metavacua/larql-to-sparql`)

- **`wasm/vindex-mvv` → PR #73 (PRIMARY, OPEN).** Current branch. Copy of
  `wasm/wasm32v1-none-gating`, so it carries the whole gating delta + the MVV +
  CI work. Targets `main`. **#71 is the fallback.**
- `wasm/wasm32v1-none-gating` → PR #71 (fallback): the gating campaign (all 15
  kernels compile both targets, over-gate audit clean). Superseded by #73.
- **Abandoned R&D** (do not build on; mine for cherry-picks only): #64
  (`wasm/blocker4-on-main`, cfg-gates), #66 (`refactor/wasm-lql-core-split`,
  W_pure larql-lql-core extraction — the parser-is-pure blueprint), #68
  (`wasm/vindex-wasm32uu`, `Arc<Mmap>→Bytes`/`from_bytes` blueprint), #69
  (`wasm/wasmi-host`, extern-C ABI + wasmi-host), #60, #59. **The `xtask`
  certifier (`crates/xtask/`, wasmparser+ascent) lives on the #64/#68/#69
  lineage** — its `wasm_facts.rs`+`rules.rs` are the source for the certifier.
- #70 (`ci/consolidated-matrix`): CI matrix + CLAUDE.md.

## 4. State of the kernel crates (on this branch)

All 15 `-wasm32v1-none` kernels compile on **both** `cargo check --target
wasm32v1-none --lib` and native `--lib`. Over-gate audit (`check_overgate.py
prove`) clean. Dependency chain: `models → compute → vindex → core`,
`inference → lql → {server, cli, python}`; `model-compute`, `kv-cache-benchmark`
(and likely `router-protocol`) are standalone (not in the roots' closure).

Key facts learned (also in `~/.claude/.../memory/`):
- **Module-gating masks native breaks**: a crate that gates whole modules off
  wasm can compile on wasm while broken on native (the wasm cell never parses
  the file). `larql-server` shipped a native break (a `use` jammed into a
  `use X::{…}` brace group in `announce.rs`) this way — invisible because CI
  never built these crates natively. → fixed, and the CI native column added.
- **Verify BOTH targets** for every crate.
- **De-gating module-gated crates** = restore the native-only module files from
  the reference original (server went 338→23 gates this way).
- `larql-python-wasm32v1-none` is the only **cdylib** kernel → needs a
  `#[global_allocator]` (dlmalloc) + `#[panic_handler]`; rlibs inherit them.
- **ndarray + BLAS (`gemv`/`gate_matmul`) are native-only** and fully gated off
  wasm — the numeric compute is NOT portable; a portable numeric layer
  (scalar/`matrixmultiply`/`simd128`/faer per dialect) is future work.
- `serde_json`, `matrixmultiply`, `faer` (default-features=false), `ndarray`
  (default-features=false), `nalgebra` (+`libm`) are all **no_std-capable** —
  verified via web search. The MVV floor still uses hand-rolled scalar (minimal
  + certifiable); libraries are L1+.

## 5. Minimal-Viable Vindex (the MVV) — DELIVERED on this branch

`crates/larql-wasm/larql-vindex-wasm32v1-none/`:
- **Spec**: `docs/minimal-vindex-spec.md` v0.1 (minimal subset of the canonical
  `crates/larql-vindex/docs/format-spec.md` v0.4). Format = headerless f32/f16
  gate blob + `index.json` v2 (source of truth) + optional self-describing
  header *constrained to agree with* index.json. Query core (the *intersection*):
  `gate_knn`/`num_features`/`feature_meta`; ext: `gate_knn_expert`/`walk`.
- **Reference**: `src/index/mvv/` (`error.rs` `MvvError`; `descriptor.rs`
  parse+validate; `query.rs` scalar checked matvec + `total_cmp` top-k;
  `mod.rs`). **Total** (no unsafe/unwrap/panic/recursion/locks; checked
  `chunks_exact`+`from_le_bytes`; abs via sign-bit clear). Compiles un-gated on
  both targets; census shows it adds **zero gated stdlib** (it's in SET B).
- **Conformance**: `tests/mvv_conformance.rs` — integration test (isolated from
  the crate's native-only inline tests), **16/16 green**, 11 adversarial/
  totality cases each → typed `MvvError`, no panic.
- **Certifier + harness** (NOT pure-kernel; verification scaffolding):
  `crates/larql-wasm/larql-vindex-mvv-cdylib/` (extern-C exports the kernel →
  standalone wasm32v1-none module; dlmalloc + panic_handler) and
  `crates/larql-wasm/larql-wasm-certify/` (native, wasmparser 0.248). The MVV
  cdylib certifies **0 imports, 0 call_indirect = WASM-SAFE**.

Deferred (per decisions): the L1–L3 efficiency-ladder kernels; certifying the
reader's `serde_json` layer separately; REUSE/CHANGELOG (pre-merge only).

## 6. Tooling (`crates/larql-wasm/tools/`)

- `census.py` — on-demand boundary census. `--sites` (gated-stdlib worklist),
  `--inventory` (regenerates `.github/wasm-inventory.md`), `--paths`,
  `--crates-only`. Asserts the invariant **no active `use std::` in
  wasm-reachable code**. Excludes cfg(test) code from the `--lib` boundary.
  Shared helpers in `wasm_gate_common.py`. cfg-aware (target_arch/os/feature
  axes), reachability walk, core-vs-local-`core`-module collision handled.
- `check_overgate.py` (lint + prove), `ungate.py`, `autogate_fn.py`,
  `gate_cascade.py`, `normalize_prelude.py`, `dedupe_gates.py`, `split_dep.py`.
- `larql-wasm-certify` (Rust bin): wasm-safe whole-module check (import-free +
  call_indirect-free). The deeper Q-free motif detector is a documented follow-on.

## 7. CI/CD — purpose, state, and the in-flight work

**Purpose** (per the user): CI is the **coding agent's feedback oracle over
environments it can't manipulate locally** (real browsers, wasmi, pyodide, the
REUSE/cocogitto/git-cliff tooling). **Coverage + correctness ≫ speed**; nightly
is irrelevant (the agent only sees PR checks). Keep per-cell granularity. The
runner **queue (concurrency cap), not per-cell latency**, dominates wall-clock —
so do NOT batch/defer/parallelize for speed; that sacrifices coverage.

Workflows: `ci.yml` (the `wasm-check` matrix from `discover-crates.py`, +
`native-test` for original crates, + gates), `validate.yml` (provenance/REUSE,
first-party-licenses, commits/cocogitto, changelog/git-cliff), `quality.yml`,
`bench-regress.yml`, `kv-cache-benchmark.yml`, `larql-python.yml`. Only
`native ✓` is branch-required; the wasm matrix is informational
(`wasm boundary ✓` always passes). `discover-crates.py`: `native_matrix`
**excludes** `crates/larql-wasm/*`; `wasm_matrix` enumerates the `-wasm32v1-none`
(+`-interface`) members.

**Delivered & pushed this session (commits `ab946378`, `9b358fcb`):**
- **native `--lib` compile column** for the `-wasm32v1-none` crates (closes the
  gap that hid server's native break). `discover-crates.py` KERNEL_RUNTIMES now
  `["wasm32v1-none","native","wasmi","node","firefox"]`; ci.yml step
  `cargo check -p CRATE --lib` (no `--target`).
- **`census ✓` gate** (fails on native-leak invariant violation) and
  **`MVV wasm-safe ✓` gate** (builds the MVV cdylib + runs `larql-wasm-certify`).
  Both FAIL on violation (real signal); both dry-run green.

**In flight / blocked / decided:**
- **Native-test layer** (the user's "build+test the roots" idea): probing showed
  **roots `larql-server`/`larql-python` compile native `--lib` tests with 0
  errors** (cli needs only `tempfile`). So the roots-native-test is **achievable
  now**. BUT mid-tier crates (e.g. `vindex`) have a **pre-existing defect**:
  `normalize_prelude` stripped `use std::collections::HashMap` from inline
  `#[cfg(test)]` modules (which don't inherit the file-scope private prelude),
  uncaught because the campaign only ran `cargo check --lib`. So mid-tier native
  `cargo test --lib` fails (13× HashMap in vindex); copied **integration tests**
  also reference the old crate name `larql_vindex`. This is **error-correction**
  (user-authorized) but a cross-crate sub-project, NOT a quick wire-up.
  → Path: deliver roots+standalones native-test now (server/python/cli +
  model-compute/kv-cache-benchmark); fix mid-tier test-module imports as a
  follow-up.
- **Dev-deps**: probe added `tempfile`/`mockito`/`serial_test` to
  `larql-vindex-wasm32v1-none/Cargo.toml` (UNCOMMITTED — see §9). Per-crate
  test-needed dev-deps (native-only, exclude `criterion`): inference
  (`assert_approx_eq`,`tempfile`,`larql-kv`→wasm32v1-none), vindex
  (`tempfile`,`mockito`,`serial_test`), model-compute (`wat`,`wasmi`), lql
  (`mockito`), cli (`tempfile`), compute (`serde_json`,`memmap2`), core
  (`assert_approx_eq`), kv (`tempfile`), models (`tempfile`); server/python/
  router-protocol/boundary/router/kv-cache-benchmark need none for `--lib` tests.
  Use `cargo test --lib` (or `--tests` once integration crate-names fixed);
  copies have auto-`benches/` so avoid `criterion` by not building benches.
- **Dialect cells (simd128/wasip1): SKIPPED** — these crates force `no_std` on
  all `wasm32`, so simd128 (additive) and wasip1 cells just duplicate the
  v1-none compile cell; dialect *execution* is already the wasmi/node/firefox
  cells.
- **Clippy**: low value now (pre-existing unused-import warnings → noise); pair
  with a warnings-cleanup pass before making it a gate.

## 8. Open items / recommended next steps

1. **Finish native-test**: add a `native-test` runtime in `discover-crates.py`
   for roots (server/cli/python) + standalones; ci.yml `cargo test -p CRATE
   --lib` gated on native compile. Add cli's `tempfile` dev-dep. Roots are green
   now; ship it.
2. **Error-correct mid-tier test modules** (HashMap/prelude over-strip;
   integration-test crate names) so their native-test cells go green — then
   extend native-test to all crates. This verifies native parity of the copies.
3. **L1–L3 numeric kernels** behind the MVV format (matrixmultiply / simd128 /
   ndarray-BLAS) when efficiency matters.
4. **Interface phase** (next major plan): clone finished kernels → `-interface`
   adapters implementing `larql-bridge`; rewrite the wasm runtime tests (the
   deferred wasmi/node/firefox columns).
5. Pre-merge only: REUSE annotations + CHANGELOG regen for PR #73.

## 9. Uncommitted working-tree state (as of this writing)

```
 M crates/larql-wasm/Cargo.lock                              # from MVV-cdylib/certify deps + vindex dev-deps probe
 M crates/larql-wasm/larql-vindex-wasm32v1-none/Cargo.toml   # vindex native-only dev-deps (tempfile/mockito/serial_test) — probe; keep if finishing native-test, else revert
```
Branch `wasm/vindex-mvv` is otherwise pushed (`origin/wasm/vindex-mvv` ==
`9b358fcb`). PR #73 open against `main`.

## 10. Verification commands

```bash
WASM=crates/larql-wasm/Cargo.toml
cargo check --manifest-path $WASM -p CRATE-wasm32v1-none --target wasm32v1-none --lib   # portable
cargo check --manifest-path $WASM -p CRATE-wasm32v1-none --lib                          # native parity
cargo test  --manifest-path $WASM -p larql-vindex-wasm32v1-none --test mvv_conformance  # MVV totality (16/16)
python3 crates/larql-wasm/tools/census.py            # invariant: no active `use std::`
python3 crates/larql-wasm/tools/census.py --inventory > .github/wasm-inventory.md       # regen boundary doc
python3 crates/larql-wasm/tools/check_overgate.py prove CRATE-wasm32v1-none             # over-gate audit
# MVV wasm-safe certification:
cargo build --manifest-path $WASM -p larql-vindex-mvv-cdylib --target wasm32v1-none --release
cargo run   --manifest-path $WASM -p larql-wasm-certify --quiet -- \
  crates/larql-wasm/target/wasm32v1-none/release/larql_vindex_mvv_cdylib.wasm
```
