# Vindex + WalkFFN review — 2026-07-30

> **Remediation status (2026-08-01): PROGRAMME CLOSED — all 24 items.**
> Tiers 0
> and 1 landed in full on 2026-07-30 (commits `6907a209`, `e55b3edf`,
> `6b5067f6`, `e21475f2`), including all four high-severity findings.
> Item 13 resolved 2026-07-31 with the finding INVERTED (`c9ec5e1f`):
> `gate_walk` was never wired on `VectorIndex`, so `enable_hnsw()` had
> been leaking approximate selection INTO walk numerics — the exact
> opposite of §5's claim; the intended exact-first chain is now wired
> and pinned. Tier 2 is complete: base+delta patched execution
> (`b813c018`), the forward/forward_observed split (`4e592c68` +
> `69ade08b`), runtime trace emission (`c6f31ad8`), and the execution
> planner (item 18, §5 item 3): `FfnPlan`/`PlanReason` in
> `walk_ffn/plan.rs` + `planner.rs`, base+delta as a plan variant,
> declines re-planned with an exclusion set and recorded in the
> reason, `WalkFfn::plan_for` pure inspection, routing pinned
> decision-identical by the unchanged dispatch/routing suites.
> The walk-vs-dense parity suite (item 20) landed with the per-file
> 90%-coverage pass (`e21475f2`); every walk_ffn file is ≥90% line
> coverage. Two-stage shortlist-then-rerank selection (item 19, §5
> item 5) landed 2026-07-31: opt-in `WalkFfnConfig::shortlist_m`,
> stage 1 = top-M through the production gate chain, stage 2 = the
> `FeatureSelector` criterion for only those M via per-row primitives
> (O(M·d), formulas single-sourced with `joint_gate_knn`), declines
> observable (`shortlist:declined`), cost pinned by a mock that panics
> on any full-projection call. KnnStore unification (item 21, §5
> item 6) closed 2026-07-31 at the retrieval-kernel level: the
> parallel `key_matrices`/`dirty` scoring machinery is deleted and
> both `PatchedVindex::gate_knn` and every `KnnStore` query score
> through the shared `patch/gate_overlay.rs::GateOverlay` (H3 guard,
> mixed-width fallback, auto-invalidating per-layer snapshots);
> API + `.vlp`/`knn_store.bin` formats unchanged; full arch-B
> retirement is explicitly gated in the rewritten
> `FFN_VINDEX_UNIFICATION_SPEC.md` (FR1/FR2 routers ship on the
> post-logits override; α calibration unvalidated). The v1
> conformance contract (22) closed 2026-08-01:
> `crates/larql-vindex/docs/conformance-v1.md` + the
> `tests/conformance_v1_*.rs` suite (38 tests) pin
> error-not-panic-not-garbage per artifact × corruption class, with
> byte-level LE golden vectors as the cross-platform guard; the two
> §3 LOW conformance violations (the legacy `down_meta::read_binary`
> allocation bomb, the Vindexfile bare-comma/`unwrap_or(0)` items)
> are fixed with regression tests; the perf benchmark protocol is a
> documented follow-up in that doc (§4). Doc drift (23, §5 item 8)
> closed 2026-08-01: every cited number traced to its source first —
> the README/operations-spec `0.008 ms/layer` + `0.3 ms` 34-layer
> walk headline was the pre-2026-04-05 `vindex_bench` example at its
> reduced 1024×256 synthetic shape, replaced with the current
> criterion `vindex_ops` numbers at both shapes (22.7 µs synthetic /
> 2.64 ms Gemma 10240×2560) plus the exact-brute-gemv scope note
> from the item-13 inversion (`enable_hnsw` = gate-KNN consumers
> only); the extract-level default contradiction resolved in favour
> of the code (CLI defaults to inference, bare LQL `EXTRACT` to
> browse — stated per surface); walk.md's "K=8092" kept rather than
> "corrected" to 8192 — the 2026-04-03 boundary sweep and sparse.md
> literally ran the 8092 harness constant (79% of 10240, genuinely
> sparse; K≥8192 hits the 80% full-K dense rewrite), now documented
> as such — and the "production" framing rebalanced (WalkFfn =
> instrumentable/editable execution layer + CPU sparse path; Q4K GPU
> decode ~88 tok/s is the perf centre), with historical results kept
> and date-qualified; campaign-sweep updates for runtime trace
> emission (17), base+delta-first (16) and GateOverlay-backed
> KnnStore (21) folded into the same docs. Hygiene (24) closed
> 2026-08-01, triaged per its own licence: the §4 generic-engine
> violations fixed (English word lists + Wikidata categories out of
> `clustering/` into `data/entity_patterns.json` /
> `data/stop_words.json` / `data/wikidata_categories.json`, loaded via
> a `LARQL_DATA_DIR`-then-workspace search chain — never cwd — with
> minimal LOUD built-in fallbacks; the bare 0.25 floor and the
> 50%-majority pattern threshold named and pinned); the two deferred
> item-23 16384→10240 fixes landed (Gemma 3 4B intermediate verified
> = 10240); the GeluTanh/SiLU dispatch — §4 counted 8 copies, 27
> existed across walk/kquant/weight/expert paths by close — routes
> through one wildcard-free exhaustive
> `Activation::uses_gelu_tanh_gate_up()` (new variants are compile
> errors; two drifted GeluTanh-only copies made consistent); FFN
> component indices unified on pub `FFN_GATE`/`FFN_UP`/`FFN_DOWN`
> constants with cross-crate compile-time pins; and the three
> untested §4 files got 41 colocated tests (`hnsw.rs` 97.3%,
> `index/mutate/mod.rs` 96.0%, `write_f32.rs` 92.6% line coverage).
> That test pass surfaced a NEW finding, pinned not papered over:
> HNSW's level-0 graph fragments beyond ~64 nodes (naive
> `add_connection` eviction orphans nodes; recall@10 = 0.16 at n=200
> uniform even with ef=n). Standing follow-ups carried out of the
> closed programme: server/lql `try_apply_patch` migration, remote
> http/sharded transport coverage, logit-contribution trace field,
> walk-FFN thresholds surfaced into `WalkFfnConfig`, the HNSW
> level-0 fragmentation above, and the remaining >250-line file
> splits (`huggingface/download/mod.rs` 1329, `patch/overlay.rs`
> 1071, `quant/convert.rs` 653 — the item-24 documented remainder).
> Per-item detail:
> [`ROADMAP.md`](../../ROADMAP.md) § "Vindex + WalkFFN review
> (2026-07-30)".

Subsystem review of **`larql-vindex`** (~51K LOC, + `larql-vindex-spec`) and
the **walk-FFN engine** (`larql-inference/src/vindex/walk_ffn/`, ~3.3K LOC
across 11 modules) on branch `feat/dec-funnel-v0-4`. Two parallel deep
readers (one per subsystem), with the four high-severity findings and two
load-bearing structural claims re-verified by hand against the source
(marked ✅). Medium/low findings carry `file:line` evidence but were not
independently re-checked.

This review was subsequently **merged with two further inputs** supplied the
same day: an external strategic review of the vindex/WalkFfn architecture
(written against a pre-refactor snapshot — see §5 for what it got right and
where it was stale) and a prior kernel deep-dive through the routing ladder,
selection chain, and sparse kernel. The consolidated programme in §6 is the
canonical action list, tracked in [`ROADMAP.md`](../../ROADMAP.md)
§"Codebase hardening" under "Vindex + WalkFFN review (2026-07-30)".

## Verdict

Both subsystems are structurally healthy. `larql-vindex`'s storage layer
(`mmap_storage.rs` bounds-checks every manifest range via `checked_view`;
`down_meta::mmap_binary` is model parsing code) and the zero-dep spec crate
are defensive engineering done right; the walk-FFN routing ladder is well
documented and its trait-dispatch refactor (`ffn_row_*`, quant registry)
has genuinely paid off — FP4 support cost zero kernel code.

Against that baseline the review found **four high-severity runtime bugs**
(one silent-garbage data-layout bug on non-256-aligned models, one panic,
two malformed-input panics/corruptions), a **silent-wrong-numerics cluster**
in the quantized walk paths (the same "produces a number, the number is a
lie" theme as the 2026-07-22 DEC review), and one systemic test gap:
**no walk-vs-dense numerical parity test exists anywhere in the tree.**

---

## 1. High-severity findings (all hand-verified ✅)

### H1 ✅ Q4K dequant cache mis-strides row-padded slabs → silent garbage

The kquant writer pads each row's column dim to the 256-element block
(`larql-vindex/src/format/weights/write_kquant/ffn.rs:70`, shape
`[rows, padded_cols]`), but both cache decoders assume **unpadded** layout:
`kquant_ffn_layer` (`larql-vindex/src/index/storage/ffn_store/kquant_cache.rs:138-161`)
and `kquant_ffn_layer_once` (`:236-256`) decode
`ceil(intermediate*hidden/256)*256` elements then index rows at stride
`intermediate`/`hidden`. `check_block_input` only rejects *short* buffers,
so when `hidden % 256 != 0` (gate/up) or `intermediate % 256 != 0` (down
transpose) the decode **succeeds** and every row after the first is shifted
— silent garbage, no diagnostic.

- Trigger models: GPT-OSS-20B (hidden=2880 — the K3 ladder's rung 1),
  Gemma3-1B (hidden=1152). Any Q4K vindex without the feature-major down
  sidecar.
- The tell: the non-cache path in
  `larql-inference/src/vindex/kquant_forward/walk_ffn.rs:63-70` explicitly
  handles `inter_padded` — someone hit this and fixed one of three copies.
- Downstream victims inside the walk engine: `sparse:parallel_q4k_down`
  (`sparse.rs:291`), the per-feature down accumulate via
  `kquant_ffn_row_scaled_add_via_cache` (reached from `sparse.rs:428`), and
  `down_row_norms`/`up_row_norms` (`selector.rs:53,117` — joint selectors
  rank on garbage norms).
- Every in-tree Q4K fixture is 256-aligned (`Q4KTestFixtures` is 256×256),
  so the suite **structurally cannot catch this class**.

### H2 ✅ Q4_0 interleaved path panics on a CPU backend (wrong-format gate)

Ladder step 5 (`walk_ffn/mod.rs:405-414`) admits **Q4_0** data if the
backend `supports_quant(QuantFormat::Q4_K)`. `CpuBackend` advertises Q4_K
(`larql-compute/src/cpu/mod.rs:151-160`) but does not override
`q4_matvec_pair_batch` (trait default returns `None`). Inside
`walk_ffn_q4_interleaved` the *same predicate* selects the "Metal" branch
(`interleaved_q4.rs:52-54`), which `.unwrap()`s the batch call (`:58-62`,
`:81-82`) → **panic on the first FFN layer** for
`WalkFfn::new_with_backend(…, &CpuBackend)` over a Q4_0 vindex. The CPU
kernel `else` branch (`:89-115`) is unreachable through the ladder because
entry and branch use the same predicate.

### H3 ✅ Patch-overlay gate cache poisoned by zero-width gate vectors

`patch/overlay.rs:176-191`: the mixed-width guard in `layer_gate_cache`
only trips once `d != 0`, so a `len == 0` gate vector coexists with real
ones. `feature_ids` includes the empty entry but the flattened matrix skips
its 0 floats, so row slicing (`gate_matrix[i*d..(i+1)*d]`, `overlay.rs:446`,
`:458`) is misaligned: rows read the wrong feature's data and the last index
**panics** (slice out of range). Panic vs wrong-scores vs safe-fallback
depends on HashMap iteration order — a nondeterministic failure. The trigger
is in-tree: `vindexfile/mod.rs:125` inserts `vec![]` as the gate vector for
every Vindexfile `INSERT`; any real gate override at the same layer
completes the setup.

### H4 ✅ Loader panics on malformed `index.json`

`format/load.rs:81` — `gate_slices[info.layer] = …` where the vec is sized
`config.num_layers` and `info.layer` comes straight from parsed JSON. An
out-of-range entry (truncated/hand-edited/corrupt manifest) panics instead
of returning `VindexError::Parse`. Same pattern at `format/load.rs:293`
(`synthesize_gate_from_q4k`). Contradicts the crate's own stated standard
(`quant/convert.rs:643-654` test: "library code must surface this as an
error").

---

## 2. Silent-wrong-numerics tier (medium)

- **M1 — Sparse K ≥ 80% silently rewritten to dense.** ✅
  `helpers.rs:24`: `hits_len_ge_intermediate` fires the full-K gemv at
  `k >= (intermediate * 8) / 10` while its doc (and the module doc) say
  "K ≥ feature count". A configured K in [0.8·N, N) runs *all* features,
  dense math, different numerics — traced as `sparse:gemv_full_k`. For
  fidelity-vs-K curves (the speed/accuracy-scissors work) points above 0.8
  density are secretly dense unless `force_walk` was set. Unnamed magic
  ratio besides.
- **M2 — Activation output silently zero on two paths.** The parallel
  Q4K-down path (hits ≥ 512, `sparse.rs:283-371`) never writes
  `full_activation`; the L1 cache hit fabricates `Array2::zeros`
  (`mod.rs:367`). Activation-consuming instrumentation (server
  `full_output`, probes, traces) silently records zeros depending on which
  path fired.
- **M3 — `config.activation_floor` is dead.** Documented ("skip features
  whose |activation| falls below this"), settable from `predict_cmd.rs:241`,
  read by nothing. The real skip threshold is a hardcoded `1e-10` at
  `sparse.rs:338`, `:411`, `:549`.
- **M4 — Unaligned `&[u8] → &[f32]` transmutes (UB) + native-endian
  on-disk floats.** `patch/format.rs:202`, `quant/convert.rs:565`,
  `config/dtype.rs:60` cast byte buffers to `*const f32` with no alignment
  guarantee; `quant/scan.rs:463`'s SAFETY comment is wrong as written.
  Encode side writes native-endian (`.vlp`/`.bin` non-portable), unlike
  `down_meta.rs` which does explicit LE. Fix: `f32::from_le_bytes` chunks
  or `bytemuck::try_cast_slice`.
- **M5 — `apply_patch` swallows decode failures.**
  `overlay_apply.rs:86,122`: `if let Ok(vec) = decode_gate_vector(b64)` —
  a corrupt vector is dropped while the op's metadata still applies:
  half-applied patch, no error. Compounded by the hand-rolled base64
  decoder truncating trailing 1–3 chars silently (`patch/format.rs:247-249`).
- **M6 — Tombstone inconsistency on Delete→Update.** `PatchOp::Update`
  (`overlay_apply.rs:102-138`) and `update_feature_meta` (`overlay.rs:241-244`)
  never clear `deleted`, unlike Insert. After Delete→Update,
  `feature_meta()` says the feature exists while `gate_knn()` filters it
  out (`overlay.rs:496`) — two query paths disagree.
- **M7 — Override layers can silently fall through to override-blind
  paths.** `mod.rs:333-339`: if `walk_ffn_sparse` returns `None` on an
  overridden layer (a single failed `ffn_row_dot` aborts via `?`,
  `sparse.rs:387`), the ladder continues into whole-layer paths that ignore
  overrides — exactly the failure the module doc warns about.
- **M8 — `kquant_ffn_row_dot` accepts `component == 2` with the wrong
  stride.** The scaled-add twin rejects down explicitly ("W2 footgun",
  pinned test, `kquant_dispatch.rs:135-143`); the dot twin
  (`kquant_dispatch.rs:97-122`) strides the down slab wrongly and silently
  returns meaningless values. No current caller; loaded API.
- **M9 — L1 cache correctness envelope.** `residual_key` quantises at
  fixed ×256 into i16 with clamping (`l1_cache.rs:58-65`) — late-layer
  residuals in the ~270× post-FFN-gain regime saturate many dims, raising
  collision probability; a u64-hash collision serves a wrong FFN output
  with no stored-key verification; hits zero the activation matrix (M2).
  Fill-and-freeze, no eviction.
- **M10 — Selector silent degradation.** `joint_gate_knn`'s fallbacks
  return production GateOnly hits when norms/batched scores are
  unavailable — an A/B sweep labelled `GateXUpDownNorm` can silently be
  GateOnly. Needs a `selector:fallback` trace suffix.
- **M11 — Deletion oversampling can under-fill top-k.** `overlay.rs:426`
  oversamples 2× then retains; >top_k tombstoned hits in the top 2·top_k
  silently under-fills the result.

## 3. Low tier

- `take_trace` (`mod.rs:281-306`) re-runs plain `gate_knn`, ignoring
  selector/pools/cell-router — the recorded trace can be a different
  feature set than the walk executed; also gate scores, not contributions.
- NaN contract split: `top_k_by_abs` deliberately panics on NaN;
  `selector.rs:267-269`, `:320-322` use `partial_cmp().unwrap_or(Equal)`
  and let NaN scramble top-K silently. Pick one contract.
- `exact.rs:41`, `:67` unwrap safetensors tensors (walk-only mode with gate
  dropped panics uninformatively); `sparse.rs:550` discards `down_sa`'s
  Result (`let _ =`) — a corrupt sidecar row drops a feature silently.
- `vindexfile/mod.rs:118`: `find_free_feature(layer).unwrap_or(0)` — no
  free slot silently overwrites feature 0. `vindexfile/mod.rs:117`:
  `num_layers / 2` insertion-placement policy as an inline literal.
- `format/down_meta.rs:141-156` (legacy non-mmap path):
  `Vec::with_capacity` sized from attacker-controlled u32 header fields —
  multi-GB allocation abort before any read fails. (The mmap path is
  exemplary by contrast.)
- `vindexfile/parser.rs:187-205`: `extract_triple` splits on bare commas
  (`INSERT ("Acme, Inc", …)` mis-parses); DELETE condition form silently
  accepts missing keys.
- `format/load.rs:224-225`: tied-embedding f16 detection heuristic
  (`len >= expected && len < expected*2`) is undocumented magic-band logic.
- `kquant_matmul_transb` doc claims backend dispatch; the parameter is
  discarded (`kquant_dispatch.rs:61`). Doc rot.
- Perf nits: per-position `x_row.to_owned()` then `.to_vec()` double copy
  in the sparse preamble; serial nested reduce over `partials`;
  `pool_restricted_gate_knn` full-projection fallback showing up in traces
  means the pool route isn't getting its Q4K bytes.

## 4. Standing-rules + test-posture audit

**File size (≤~250-line rule):** 88 of 186 `larql-vindex` src files exceed
250 lines (worst: `format/huggingface/download/mod.rs` 1329,
`index/storage/vindex_storage/mmap_storage.rs` 1187 (~600 test),
`patch/overlay.rs` 959, `extract/build/mod.rs` 862,
`format/weights/write_f32.rs` 777). Walk side: `mod.rs` 926 (split:
timings / routing ladder / builders), `sparse.rs` 842 (split along the
existing specialisations: gemv / route / parallel / gather),
`walk_config.rs` 421 (`CellRouter` out). The crate predates the rule and
shows active decomposition; direction-of-travel item.

**Magic values:** vindex proper is mostly good (named consts with spec
citations). Offenders: the `8/10` density threshold (`helpers.rs:24`);
gather minimum `256` (`sparse.rs:189`); parallel threshold `512`
(`sparse.rs:288`); the `1e-10` epsilons ×3 (should *be* `activation_floor`);
L1 scale `256.0` (`l1_cache.rs:61`); bare component indices `0/1/2`
throughout walk code despite `FFN_DOWN` existing; `* 3` components
(`interleaved_q4.rs:32`, `FFN_COMPONENTS_PER_LAYER` exists);
`clustering/labeling.rs:161` bare `0.25` similarity floor; HNSW LCG
constants (`index/compute/hnsw.rs:123-124,253-255`); inline `* 4`
bytes-per-f32 (`ffn_store/down.rs:47-49,72`).

**Generic-engine rule:** `clustering/labeling.rs:192+` embeds static
English word lists (countries/languages/months) and fixed pattern classes
in engine code; `clustering/categories.rs:44+` hardcodes a Wikidata
category vocabulary and probes **cwd-relative** data paths. The
GeluTanh/SiLU activation dispatch is copy-pasted **eight times** across
walk backends (`sparse.rs:68`, `exact.rs:35`, `full_mmap.rs:24`,
`interleaved.rs:27`, `interleaved_q4.rs:44`,
`interleaved_kquant_native.rs:71`, `interleaved_kquant_dequant.rs:43`,
`selector.rs:235`) — a new activation silently lands in the SiLU arm.
`interleaved_q4.rs:29-39` assumes uniform intermediate width across layers.

**Dead code:** ✅ `larql-vindex/src/walk/` is a fully **orphaned module** —
no `mod walk;` declares it anywhere, it never compiles, and it holds a
stale duplicate of `WalkFfnConfig` (missing `force_walk`/selector/pool/
router fields; left behind by commit `3944359b`). Delete, don't migrate.
Plus `pub use engine as storage` (self-flagged), the dead `n` in
`quant/convert.rs:503-504`, 4 justified `#[allow(dead_code)]` sites.

**Test posture:** vindex proper is strong (~1157 colocated + 223
integration tests; storage substores, patch format/overlay, spec crate all
covered incl. corrupt-input cases). Gaps: `index/compute/hnsw.rs` (455
lines, zero tests — the standout), `index/mutate/mod.rs` (409),
`format/weights/write_f32.rs` (777), `format/weights/load/{f32,q4k}.rs`,
`format/down_meta.rs`. Walk side is worse in kind:

- `routing_tests.rs` tests a **hand-copied replica of the routing ladder**
  (two copies, `:182-212`, `:302-325`), not the real one — the file admits
  the extraction is an unfulfilled follow-up; the copy can drift without
  any test failing. `dispatch_trace_is_opt_in` asserts nothing.
- The only cross-path parity test is native↔dequant
  (`interleaved_kquant_native.rs:161-187` — same-source bytes through an
  independent kernel, exactly the right bar). Nothing for: sparse serial
  vs exact; gemv full-K vs serial; parallel_q4k_down vs serial (would have
  caught M2); gather vs serial (delegated to an example, not CI);
  `interleaved_q4.rs` (**zero tests** — would have caught H2);
  `interleaved.rs`/`full_mmap.rs`/`exact.rs`.
- The larql-server walk-ffn tests (`test_walk_ffn_coverage.rs`,
  `test_walk_ffn_q8k_coverage.rs`, `test_walk_ffn_dec_replay.rs`) are all
  HTTP-contract/wire-shape tests — valuable, none numerical. **No test
  anywhere compares a walk-path FFN output against dense ground truth on a
  served vindex.**

The four tests that would have caught the four worst bugs: (1) a
non-256-aligned Q4K fixture (e.g. hidden=320) through `kquant_ffn_layer` +
the sparse serial path vs the dequant baseline; (2) a
`WalkFfn::new_with_backend(&CpuBackend)` + Q4_0 fixture forward; (3) a
serial-vs-parallel sparse parity test at hits ≥ 512 asserting output *and*
activation; (4) dispatch-trace assertions against the real ladder using
the existing `MockIndex`.

---

## 5. Strategic merge (external review + kernel deep-dive)

An external strategic review of the architecture and a prior kernel
deep-dive were merged into this record. Where the three inputs overlap
they agree (dead `activation_floor`, the 80% threshold, L1 zeroing,
selector fallback were found independently twice). What each added:

**The strategic review's keepers** (its code-level critique was written
against a pre-refactor snapshot — it cited `walk_ffn.rs` as a single file
and misread the file-move as staleness — but the architecture direction
stands):

1. **Base-plus-delta patched FFN execution** — the standout. The FFN is
   feature-separable, so
   `y_patched = y_base + Σ_{i∈P} [aᵢ_new(x)·dᵢ_new − aᵢ_old(x)·dᵢ_old]`
   is *exact*, turning override cost from O(N) forced-sparse into O(|P|)
   extra dots on the fast dense path. Inserts are pure additions, deletes
   pure subtractions — the whole Vindexfile edit vocabulary maps on.
   **Exactness conditions:** the old-contribution subtraction must go
   through the **same quantised `row_dot` bytes** the dense base used
   (subtracting an f32-recomputed old term from a Q4K-native base injects
   quantisation mismatch into every patched slot), and the old-down
   subtraction needs the feature-major down sidecar (already exists from
   the gather work). Also removes the M7 hazard class by construction.
2. **Execution/observation split** — `forward` vs
   `forward_observed(observer)`; sparse paths emit
   `Vec<(FeatureId, f32)>` instead of a dense `seq_len × intermediate`
   zero-fill. Subsumes M2 (makes it type-system-impossible) and removes
   the dense-activation allocation from ordinary generation.
3. **Explicit per-layer execution plan** — a `VindexFfnPlan` enum with
   structured *reasons* and config-surfaced thresholds. The ladder already
   emits structured path names the tests assert on; what's missing is the
   reason field and de-magicked thresholds. Subsumes the H2 class. Only
   worth freezing once base+delta exists as a plan variant.
4. **Traces as causal contributions** — runtime emission into a sink from
   the executed path (replacing `take_trace`'s re-run), recording gate
   score, up score, activation, ‖down‖, residual-delta norm, logit
   contribution, rank, path. Align with the chuk-introspect schema rather
   than inventing a second trace format.
5. **Two-stage sparse selection** (gate shortlist top-M → exact rerank by
   `|φ(g·x)(u·x)|·‖d‖`) — the rerank criterion already exists
   character-for-character as `FeatureSelector::ActXUpScoreXDownNorm`;
   what's missing is the *shortlist* structure (selectors currently
   compute full projections). This is the production-cost shape of the
   existing experiment harness, not a new mechanism.
6. **KnnStore unification unfinished** — confirmed: `KnnStore` still
   exported and live in `patch/knn_store_io.rs`, `overlay.rs`,
   `overlay_apply.rs`; the unification spec still describes it.
   *(Resolved 2026-07-31 at the retrieval-kernel level — shared
   `GateOverlay` scoring structure, parallel query path deleted,
   formats/API kept; full arch-B retirement gated in the rewritten
   spec. See ROADMAP item 21.)*
7. **Vindex v1 conformance contract** — capability manifest, exact tensor
   layouts, quant tags, corruption tests, cross-platform round trips,
   benchmark protocols. Mostly assembling what exists (spec crate,
   per-shard digests, provenance); missing the corruption-test and
   round-trip suite.
8. **Doc drift is real**: README `0.008 ms/layer` headline vs the crate's
   2.65 ms full-projection figure (needs the HNSW-vs-brute / which-N
   qualifier); README extract-level default contradiction; walk.md
   "Lossless at K=8092" (typo for 8192) and "production" framing. The
   review itself being misled by stale docs is the strongest argument for
   fixing them.
   *(Resolved 2026-08-01 — see the header note and ROADMAP item 23.
   One finding adjusted on verification: the boundary sweep, sparse.md
   and the remote-codec tests all literally ran K=8092 — the typo is
   baked into the 2026-04 harness constant, not walk.md — so the doc
   now documents 8092 as the harness's literal K instead of silently
   rewriting history to 8192.)*

**Rejected/adjusted from the strategic review:** the WalkFfn→VindexFfn
rename undersells the load-bearing identity (the full-K gemv path proves
in code that the walk and the matmul are the same operation on the same
bytes); rename the larql-core knowledge-graph walk instead. "Patches force
sparse execution is the largest hidden performance cliff" assumes edited
vindexes on hot serving paths — for current usage the cliff is real but
rarely stood on; base+delta's value is *unlocking editing as
production-viable*, not fixing a live regression. Its quoted path
heuristic (`top_k < intermediate/2`) was stale — current code is ≥80% via
`hits_len_ge_intermediate`.

**Deep-dive claims verified this session ✅:**

- **HNSW is unreachable on the sparse hot path** — the walk tries
  `gate_walk` first (`sparse.rs:231`, `:268-273`) and HNSW lives only
  inside the `gate_knn` fallback, so `enable_hnsw()` changes nothing
  whenever `gate_walk` succeeds (any f32/warmed vindex). If brute gemv is
  intentionally preferred at these N, document it at `enable_hnsw()`;
  otherwise it's a wiring gap.
- **The gather kernel carries a live "experimental — not yet correct for
  production down" caveat** (`sparse.rs:450`) yet is production-reachable
  via route-pool + feature-major sidecar (guarded only by ≥256 features,
  gated arch, no overrides, no native up). Either the caveat is stale
  (predates the sidecar) and should be deleted, or the routing gate needs
  an opt-in flag.

---

## 6. Consolidated programme

Sequencing is driven by three interactions: **H1 gates base+delta** (the
delta path leans on the same row-dot/sidecar machinery; building it on a
mis-strided cache bakes the bug into the "exact" feature — and GPT-OSS-20B
hidden=2880 is precisely K3 rung 1); **the observation split *is* the fix
for M2** (don't patch it twice); **the planner subsumes the H2 class** but
H2 panics today so it gets the two-line fix now.

**Tier 0 — correctness, small independent diffs:** H1 + non-aligned
fixture; H2 gating fix; H3 zero-width guard + stop `vindexfile` inserting
`vec![]` gates; H4 loader bounds checks; M7 hard-error on override
fallthrough (stopgap until base+delta); M4 transmutes → `from_le_bytes`;
M5 surface patch-decode failures.

**Tier 1 — kernel-semantics campaign (one PR neighbourhood):** wire or
delete `activation_floor` (M3); name the 80% threshold and align doc/code
(M1); `selector:fallback` trace suffix (M10); resolve the gather caveat;
unify the NaN contract; delete `larql-vindex/src/walk/`; decide the HNSW
hot-path question explicitly; M6 tombstone semantics + pinning test.

**Tier 2 — capability:** `forward`/`forward_observed` split (fixes M2, L1
zeroing, dense-activation allocation); base+delta patched execution (after
H1; same-bytes subtraction; as a routing-ladder branch, not a rewrite);
runtime trace emission replacing `take_trace`; then the planner enum with
structured reasons; two-stage shortlist-then-rerank selection.

**Tier 3 — productization:** KnnStore unification; v1 conformance contract
(corruption tests + cross-platform round trips); standing walk-vs-dense
parity suite (the four tests in §4); doc drift (README numbers +
extract-level default, walk.md); file splits (`mod.rs`, `sparse.rs`,
`overlay.rs`, `download/mod.rs`, `convert.rs`); clustering word-list
extraction to data files; colocated tests for `hnsw.rs`,
`index/mutate/mod.rs`, `write_f32.rs`.
