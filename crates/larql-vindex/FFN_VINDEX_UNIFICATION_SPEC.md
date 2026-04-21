# FFN-Vindex Unification Spec

**Version:** 0.1 (2026-04-15)
**Scope:** `larql-vindex`, `larql-lql`, `larql-inference`, `larql-python`
**Goal:** Collapse arch-B's parallel `KnnStore` into the FFN vindex itself. One data structure, one INSERT path, one INFER path.

---

## 1. Motivation

Arch-B's `KnnStore` (added on branch `architecture-b`) stores fact keys and target tokens in a side-structure keyed on residual cosine at install layer. INFER queries both the FFN *and* the KnnStore, overriding the model's prediction when `cos > 0.75`.

This is logically redundant. The FFN is already a KNN store:

- **gate matrix** = L2-normalizable keys (one row per feature)
- **down matrix** = value vectors (one column per feature)
- forward pass = cosine match + activation + value retrieval

A compiled fact edge (arch-A) does exactly what a `KnnStore` entry does — it just uses the FFN's own machinery instead of a side map. The two paths differ only in (1) the *shape* of the retrieval (hard top-1 override vs dense activation sum) and (2) the *storage* location (separate HashMap vs appended row in gate_vectors).

Unifying to a single "FFN = KNN index = vindex" abstraction:

- Deletes a parallel data structure (~500 lines).
- Deletes an override check in the INFER loop.
- Makes `INSERT` semantically just "grow the FFN by one feature".
- Folds `.vlp` patch format to one `Insert` variant (drop `InsertKnn`).
- Gives composition and chaining for free — inserted facts participate in the residual stream naturally, can be used by downstream layers.

## 2. Current State

### Storage (what exists now in `PatchedVindex`)

```rust
pub struct PatchedVindex {
    pub base: VectorIndex,                         // immutable mmap'd base
    pub patches: Vec<VindexPatch>,                 // applied .vlp patches
    overrides_meta: HashMap<(L,F), FeatureMeta>,   // feature meta overlay
    overrides_gate: HashMap<(L,F), Vec<f32>>,      // gate row overlay
    deleted: HashSet<(L,F)>,                       // tombstones
    pub knn_store: KnnStore,                       // ← arch-B, SEPARATE
}
```

`knn_store` is the anomaly. Every other field is scoped to `(layer, feature)` addressable slots in the FFN; `knn_store` invents its own keyspace.

### Install paths

- **arch-A (`exec_compile_from_vector` / `insert_feature`)**: picks a free feature slot, writes `gate_row` into `overrides_gate[(L, slot)]`, `down_col` via `base.set_down_vector`, meta via `overrides_meta`. Slot is within the base's FFN width (e.g., 0..2048).
- **arch-B (`exec_insert` on branch `architecture-b`)**: captures residual via forward pass, L2-normalizes, `knn_store.add(layer, residual_key, target_id, ...)`. No slot allocation.

### Retrieval paths

- **Dense FFN (`walk_ffn_full_mmap`, `forward_walk`)**: normal forward pass. Sees overrides through `overrides_gate_at(L,F)` and `down_overrides(L,F)`. Compiled arch-A facts fire here.
- **arch-B override check** (`larql_lql::executor::query::infer`): explicit cosine match against `patched.knn_store.query_top1(layer, residual)` at `cos > 0.75`, result presented as KNN override in INFER output. Runs in parallel with the dense FFN pass.

## 3. Target State

### Storage (unified)

```rust
pub struct PatchedVindex {
    pub base: VectorIndex,
    pub patches: Vec<VindexPatch>,
    overrides_meta: HashMap<(L,F), FeatureMeta>,   // unchanged
    overrides_gate: HashMap<(L,F), Vec<f32>>,      // unchanged; now also covers appended slots
    overrides_up:   HashMap<(L,F), Vec<f32>>,      // NEW: up row per appended feature
    appended_count: HashMap<L, usize>,             // NEW: # of appended features per layer
    deleted: HashSet<(L,F)>,                       // unchanged
    // knn_store: REMOVED
}
```

### Slot allocation

Every layer's FFN has a **base feature count** `base_ffn_dim` (e.g., 2048 for v11). Appended features live at slots `[base_ffn_dim, base_ffn_dim + appended_count[L])`. Features at appended slots:

- have no entry in `base.gate_vectors` / `base.down_weights` (the mmap'd matrices)
- have their gate row in `overrides_gate[(L, slot)]`
- have their up row in `overrides_up[(L, slot)]`
- have their down column in `base.down_overrides[(L, slot)]` (existing mechanism)
- have meta in `overrides_meta[(L, slot)]`

All retrieval paths (dense, top-k walk, gate_knn) enumerate `[0, base_ffn_dim + appended_count[L])` and consult the overlays for any slot ≥ `base_ffn_dim`.

### Install path (one)

```rust
impl PatchedVindex {
    pub fn append_feature(
        &mut self,
        layer: usize,
        gate_row: Vec<f32>,
        up_row: Vec<f32>,
        down_col: Vec<f32>,
        meta: FeatureMeta,
    ) -> usize /* new feature index */;
}
```

`exec_insert` (the LQL executor) now:

1. Capture residual at install layer via forward pass (unchanged).
2. Read target token embedding from the embedding matrix.
3. Scale down_col = `α * embed(target)` where α is the confidence-modulated magnitude.
4. Set gate_row = L2-normalized residual (for override semantics) or computed via FactCompiler-style QR ortho (for composition semantics) based on `WITH mode = override | compose`.
5. `patched.append_feature(layer, gate_row, up_row, down_col, meta)`.
6. Record `PatchOp::AppendFeature { layer, feature, ... }` for persistence.

### Retrieval path (one)

Normal forward pass. That's it. No override branch in `exec_infer`. If the gate matches strongly, the feature fires; the down column writes the target direction into the residual; logits at the final layer project onto the target token.

The `cos > 0.75` threshold from arch-B becomes a property of the install — features installed with `mode:override` have `down_col` scaled large enough that any gate activation > some threshold dominates logits. Install-time scaling decides run-time override behavior.

## 4. Patch Format (.vlp)

### Retire
```
PatchOp::InsertKnn { layer, entity, relation, target, target_id, confidence, key_vector_b64 }
PatchOp::DeleteKnn { entity }
```

### Replace with
```
PatchOp::AppendFeature {
    layer: usize,
    feature: usize,                    // absolute slot index (= base_ffn_dim + n)
    entity: String,
    relation: String,
    target: String,
    confidence: Option<f32>,
    mode: AppendMode,                  // Override | Compose
    gate_vector_b64: String,           // L2-normalized residual (Override) or engineered gate (Compose)
    up_vector_b64: String,             // usually a copy of gate, or unit vector
    down_vector_b64: String,           // α * embed(target)
    alpha: f32,                        // down-scaling factor (records effective magnitude)
}
PatchOp::DeleteFeature { layer, feature, reason: Option<String> }
```

### Backward compatibility

Existing `.vlp` files with `InsertKnn`/`DeleteKnn` ops must still load and apply. A migration path:

- Reader: accept both `insert_knn` and `append_feature` tags on deserialize.
- `InsertKnn` on load → convert to `AppendFeature` at slot `base_ffn_dim + next_free(L)`, synthesize `up_row` as a copy of the gate (cheap default), synthesize `down_col` as `α * embed(target_id)` scaled so that run-time logits on the target token exceed the model's baseline prediction by at least the margin implied by the old `cos > 0.75` threshold. Record `alpha` for reproducibility.
- Writer: always emit the new format. No dual-write.

The existing `PatchOp::Insert` (arch-A compile path into free slots < `base_ffn_dim`) stays as-is — it's still valid for ones that want to replace existing FFN features rather than append.

## 5. Per-Crate Migration

### `larql-vindex`

**Add:**
- `PatchedVindex::append_feature(layer, gate, up, down, meta) -> usize`
- `PatchedVindex::appended_count(layer) -> usize`
- `PatchedVindex::feature_count(layer) -> usize` returns `base_ffn_dim + appended_count(layer)`
- `overrides_up: HashMap<(L,F), Vec<f32>>`
- `PatchOp::AppendFeature` / `PatchOp::DeleteFeature` variants
- Migration: `PatchOp::InsertKnn` → `AppendFeature` on load (inside `apply_patch`)

**Modify:**
- `gate_knn(layer, query, k)` to enumerate `0..feature_count(layer)` (not just `0..base_ffn_dim`).
- Any iteration over FFN features must use the extended range.
- `walk_ffn_full_mmap` to include appended features in the dense matmul. Two options:
  - (a) materialize a per-inference extended matrix (base slice + appended rows concatenated) — simple, small allocation if appended_count is small.
  - (b) run base matmul + separate appended matmul, add outputs. More code, avoids allocation.
  
  Pick (a) for simplicity; (b) if benchmark shows the allocation is hot.

**Delete:**
- `patch/knn_store.rs` (whole file, ~500 lines) — retired.
- `patch/mod.rs`: drop `pub use knn_store::...`.
- `KnnStore` field on `PatchedVindex`.

### `larql-lql`

**`executor/mutation.rs` — `exec_insert`:**
- Keep the residual-capture forward pass (unchanged).
- Keep the target token resolution.
- Replace `patched.knn_store.add(...)` with `patched.append_feature(layer, gate_row, up_row, down_col, meta)` where:
  - `gate_row` = L2-normalized residual (override mode, default) or engineered (compose mode, if `WITH mode = compose`).
  - `up_row` = copy of gate_row (or the identity-projecting variant if we later separate them).
  - `down_col` = `alpha * embed_row_of_target_id` scaled to produce an override-strength target bias.
  - `meta` = FeatureMeta { relation, entity, target, confidence }.
- Record `PatchOp::AppendFeature`.
- Output message changes from `"... at L{layer} (KNN store)"` to `"... at L{layer} F{feature} (appended)"`.

**`executor/query.rs` — `exec_infer`:**
- Delete the KNN override branch (lines around 197–260 on the `architecture-b` branch).
- Keep the normal walk/predict flow. The appended features participate in the dense matmul naturally; if they fire hard, they dominate logits for their target token — which is the override.

**Existing tests:**
- LQL executor tests that exercise `INSERT INTO EDGES ... AT LAYER N` (mutation.rs tests, around line 140+). Update expected output strings and assertions about KNN store size → assert against `feature_count(layer)` increase instead.

### `larql-inference`

- No changes expected to public API.
- Walk FFN implementations (`WalkFfn`, `walk_ffn_full_mmap`, sparse/top-k variants) must respect `patched.feature_count(layer)` rather than hardcoding `base_ffn_dim`. Most already take a matrix parameter; check that PatchedVindex provides a view that includes appended rows.

### `larql-python`

- `PyVindex.insert(entity, relation, target, layer, confidence) -> (layer, feature)` already returns `(layer, feature)` — the unified path returns an appended slot index rather than a free base slot. API signature unchanged.
- `exec_insert` output format changes slightly; update any Python test that parses "KNN store" from the output.

## 6. Semantic Equivalence (correctness argument)

The old arch-B path:
1. Compute residual `r` at install layer.
2. L2-normalize `r` → `r̂`.
3. Store `(r̂, target_id)` in KnnStore.
4. At inference, compute live residual `r_live`, normalize, compute `cos(r̂, r̂_live)`.
5. If `cos > 0.75`, emit `target_id` as override.

The unified path:
1. Same.
2. Same.
3. Append `gate_row = r̂`, `up_row = r̂` (copy), `down_col = α * embed(target_id)`.
4. At inference, FFN computes `gate_score = gate @ r̂_live ≈ cos(r̂, r̂_live)` for this slot (modulo magnitude; both are unit norm).
5. `feature_activation = silu(gate_score) * (up @ r̂_live) ≈ silu(gate_score) * gate_score`.
6. FFN output includes `feature_activation * down_col = silu(c) * c * α * embed(target)`.
7. Logits at position of this token pick up `α' * embed(target) · embed_rows` — strongly biased toward `target_id`.

For `cos > 0.75`, `silu(0.75) * 0.75 ≈ 0.4`. If `α` is chosen so that `0.4 * α` exceeds the baseline logit margin by the desired amount, the override fires. Calibration of `α` reproduces the cos=0.75 threshold exactly.

The one subtle difference: unified path injects into the **residual stream** (via down column), not directly into logits. Downstream layers (L_install+1 onward) see the target direction and can either reinforce it or modulate it. Arch-B's override short-circuited this. **This is the feature, not a bug** — composition becomes available.

Unified path also responds to cosine below 0.75 gracefully (small contributions rather than binary override). Consistent with how the rest of the FFN operates.

## 7. Testing

**Unit tests (`larql-vindex`):**
- `append_feature` allocates at `base_ffn_dim + n`, increments count, is visible in `feature_count`.
- `gate_knn` returns appended features when their gate is near the query.
- Loading a `.vlp` with `InsertKnn` migrates to `AppendFeature` correctly.

**Integration tests (`larql-lql`):**
- `INSERT INTO EDGES ... AT LAYER N` appends and INFER on the canonical prompt retrieves the target in top-1.
- Parity test: run the arch-B WASM arithmetic benchmark (189 facts) on the unified path. Expect 189/189 at 100% with similar wall time (~200ms per install).

**Regression suite:**
- Existing 309 tests in `larql-lql` and larql-vindex must pass after the refactor (allowing for output format string updates).

## 8. Plan of Work

1. **Vindex core** (half day): `PatchedVindex::append_feature`, `overrides_up`, `appended_count`, `feature_count`. Add the `PatchOp::AppendFeature` variant.
2. **Migration on patch load** (2 hours): `InsertKnn` → `AppendFeature` conversion at load time.
3. **Walk FFN extension** (2 hours): ensure dense and top-k walks see appended features. Verify via a unit test that appends a single feature and runs a forward pass.
4. **Executor `exec_insert` rewrite** (1 hour): replace `knn_store.add` with `append_feature` plus the embedding-lookup-for-down-column step.
5. **Executor `exec_infer` cleanup** (1 hour): delete KNN override branch; verify INFER still emits overrides for appended features via natural FFN pass.
6. **Delete `patch/knn_store.rs`** (30 min): remove file, update `patch/mod.rs`.
7. **Test pass + parity benchmark** (half day): run existing tests; run the 189-fact arch-B WASM benchmark on the unified path; compare accuracy and latency.
8. **Doc update** (30 min): `arch_b_RESULTS.md` addendum noting the unification.

Estimated total: **1.5 days of focused work**.

## 9. Open Questions

**Q1: up_row policy.** The simplest choice is `up_row = gate_row`. That gives `silu(gate·x) * (gate·x)` — quadratic-ish in the cosine. For compositional compile (arch-A), the up row sometimes differs from gate to allow conjunction/conditional logic. Keep the option for different up_row in `append_feature`, default to copy-of-gate.

**Q2: α calibration.** What value of α in `down_col = α * embed(target)` reproduces the cos=0.75 override behavior? Needs empirical tuning. First pass: pick α so that `silu(0.75) * 0.75 * α * ||embed(target)|| = ceil(max_logit_baseline)`. Calibrate via one test install, then use as default.

**Q3: appended features in `.vlp` portability.** The `gate_vector_b64` in the new op is base-relative (L2-norm'd residual). Applying the patch on a different vindex/model will produce different residuals for the same prompt — patch portability requires recomputing the gate from the canonical prompt rather than re-using the stored bytes. Solution: store **the install prompt** alongside the gate, and on apply, recompute gate from prompt if the target model's checksum differs.

**Q4: dense FFN slot budget.** Appending hundreds of features grows the per-layer matmul size by `appended_count[L] × dim`. For v11 (dim=512), 1000 appends at one layer = 512K extra floats per forward pass — negligible. For Gemma-3-4B (dim=2560), 10K appends = 25M floats, still cheap. Scale monitoring via `feature_count` stats.

**Q5: removal semantics.** `DeleteFeature` tombstones an appended slot — next append can reuse the index? Or permanently skip? First pass: skip (append-only + tombstone); revisit if fragmentation becomes an issue.

---

## References

- `patch/core.rs` — PatchedVindex, PatchOp, VindexPatch (will be modified)
- `patch/knn_store.rs` — KnnStore (will be deleted)
- `larql-lql/src/executor/mutation.rs` `exec_insert` (will be rewritten)
- `larql-lql/src/executor/query.rs` `exec_infer` (KNN override branch deleted)
- `experiments/15_v11_model/TWO_LEVEL_ARCHITECTURE_SPEC.md` — the architectural context that motivates this unification
- `arch_b_RESULTS.md` — the 189/189 WASM arithmetic result that the unified path must match
