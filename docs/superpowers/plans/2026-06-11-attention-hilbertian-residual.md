# Attention Hilbertian Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `larql hilbertian <vindex>` — a weights-only command that scores, per attention head, how close that head's query/key coupling is to being genuinely complex-linear (i.e. the real part of a complex Hermitian form), and writes the per-head scores to `hilbertian_meta.json`.

**Architecture:** A new pure-linear-algebra module `canonical/hilbertian.rs` builds the split-half complex structure `J` (`J²=−I`) and computes the relative commutator residual `‖[C,J]‖_F / ‖C‖_F` of a head's coupling matrix `C`. A manifest-driven loader `format/attn_load.rs` reads per-layer `W_Q`/`W_K` from `attn_weights.bin`. A new `larql-cli` subcommand assembles per-head couplings, scores them, and writes a JSON sidecar. Everything is intrinsic (a pure function of the weights), requires no corpus and no forward pass, and leaves all existing vindex files untouched.

**Tech Stack:** Rust 2021, ndarray 0.16 (no BLAS feature — keep matrices small), serde / serde_json, clap, existing `larql-vindex` format helpers (`decode_floats`, `load_vindex_config`, filename constants).

---

## Domain background (read once)

For an attention head, the query/key projections are `W_Q^{(h)}` and `W_K^{(g)}` (shape `[d_head, hidden]`, PyTorch `[out, in]` orientation). Their **coupling matrix** is

```
C_h = W_Q^{(h)} · (W_K^{(g)})ᵀ        # shape [d_head, d_head]
```

A complex structure on `ℝ^{d_head}` is a linear map `J` with `J² = −I`. The **split-half** convention (the one RoPE uses in this codebase — `crates/larql-compute/src/attention/rope.rs` pairs coordinate `i` with `i + d_head/2`) is:

```
J e_i        =  e_{i+half}      for i in [0, half)
J e_{i+half} = −e_i             where half = d_head/2
```

`C_h` is the **real part of a complex Hermitian form** (it is "complex-linear" w.r.t. `J`) **iff** `C_h` commutes with `J`: `C_h J = J C_h`. We therefore score each head by the **relative commutator residual**

```
r_h = ‖C_h J − J C_h‖_F / ‖C_h‖_F        ∈ [0, 2]
```

`r_h = 0` ⟺ the head's coupling is genuinely complex-linear w.r.t. the split-half `J` (the realification of a complex `(d_head/2)×(d_head/2)` matrix). Larger `r_h` ⟺ more irreducibly real. Because `J` is orthogonal, `‖C_h J‖_F = ‖J C_h‖_F = ‖C_h‖_F`, so `r_h ≤ 2`.

**Scope of this v1 (documented choices, refinements deferred to a follow-up issue in Task 6):**
- Uses the **fixed** split-half `J`, not the *optimal* `J` minimised over all complex structures. The fixed-`J` residual is an **upper bound** on the optimal-`J` residual (the fixed `J` is one admissible choice, so the minimum can only be smaller). A small `r_h` therefore *proves* the head is near-complex-linear; a large `r_h` is inconclusive until the optimal-`J` fit (deferred) is done.
- Analyses the `d_head×d_head` coupling `C_h` (cheap, RoPE-faithful `J`), not the `hidden×hidden` residual-space QK form (expensive without BLAS; deferred).
- Computed in the model's native basis, not the canonical whitened basis (deferred — makes `r_h` a function of the canonical form).

## Verified codebase facts (do not re-derive)

- `attn_weights.bin` has **no header**: a raw concatenation of tensors. Per-tensor `{key, shape:[rows,cols], offset, length, file}` lives in `weight_manifest.json`, which is a **top-level JSON array**.
- Keys: `"layers.{L}.self_attn.q_proj.weight"` (`[hidden, hidden]`) and `"layers.{L}.self_attn.k_proj.weight"` (`[num_kv_heads*head_dim, hidden]`). Orientation `[out, in]`, row-major.
- dtype is per-tensor and detectable: `bytes_per_elem = length / (rows*cols)` → `2 = f16`, `4 = f32`. Decode with `larql_vindex::config::dtype::decode_floats(&[u8], StorageDtype) -> Vec<f32>`. `StorageDtype` is re-exported at `larql_vindex::StorageDtype`. (Confirm the variant names — likely `F16`/`F32` — by reading `crates/larql-vindex/src/config/dtype.rs` lines 12–42 in Task 4 Step 1.)
- Head config: `VindexConfig.model_config: Option<VindexModelConfig>` with fields `num_q_heads: usize`, `num_kv_heads: usize`, `head_dim: usize` (`crates/larql-vindex/src/config/model.rs`). `hidden_size = num_q_heads * head_dim`.
- GQA: query head `h` uses KV head `h / (num_q_heads / num_kv_heads)`. SmolLM2-360M: 15 Q-heads, 5 KV-heads (group size 3), `head_dim = 64`, `hidden_size = 960`, 32 layers, dtype f16, `quant: none`.
- `VindexError::Io(#[from] std::io::Error)` exists (so `?` on `std::fs` works) and `VindexError::Parse(String)` exists.
- Filename constants live in `crates/larql-vindex/src/format/filenames.rs`; `ATTN_WEIGHTS_BIN` and `WEIGHT_MANIFEST_JSON` already exist there.
- CLI command pattern: see the existing `Canonicalize` variant (`crates/larql-cli/src/main.rs`) and `canonicalize_cmd.rs` — runners return `Result<(), Box<dyn std::error::Error>>`; the dispatch `match` arm returns the `Result` directly (no `?`).

## File structure

**New files:**
- `crates/larql-vindex/src/canonical/hilbertian.rs` — `complex_structure_split_half`, `commutator_residual`, `head_block`, `head_coupling`, `kv_head_for_query`, `head_hilbertian_residual`
- `crates/larql-vindex/src/format/attn_load.rs` — `load_attention_qk`
- `crates/larql-cli/src/commands/extraction/hilbertian_cmd.rs` — `HilbertianArgs`, `run`

**Modified files:**
- `crates/larql-vindex/src/canonical/types.rs` — add `HeadHilbertianInfo`, `HilbertianMeta`
- `crates/larql-vindex/src/canonical/mod.rs` — wire `hilbertian` module + re-exports
- `crates/larql-vindex/src/format/filenames.rs` — add `HILBERTIAN_META_JSON`
- `crates/larql-vindex/src/format/mod.rs` — add `pub mod attn_load;`
- `crates/larql-cli/src/commands/extraction/mod.rs` — add `pub mod hilbertian_cmd;`
- `crates/larql-cli/src/main.rs` — add `Hilbertian` variant + dispatch arm

---

## Task 1: Complex structure `J` and commutator residual

**Files:**
- Create: `crates/larql-vindex/src/canonical/hilbertian.rs`
- Modify: `crates/larql-vindex/src/canonical/mod.rs`

- [ ] **Step 1: Create the file with implementation + failing tests**

Create `crates/larql-vindex/src/canonical/hilbertian.rs`:

```rust
//! Per-head "Hilbertian" residual: how close an attention head's query/key
//! coupling is to being complex-linear w.r.t. the split-half complex
//! structure J (J² = −I) that RoPE uses. See the plan doc for the math.

use ndarray::{s, Array2};

/// Build the split-half complex structure J on R^n (n must be even):
///   J e_i        =  e_{i+half}     for i in [0, half)
///   J e_{i+half} = −e_i
/// so that J·J = −I. Panics if n is odd.
pub fn complex_structure_split_half(n: usize) -> Array2<f64> {
    assert!(n % 2 == 0, "complex structure requires even dimension, got {n}");
    let half = n / 2;
    let mut j = Array2::<f64>::zeros((n, n));
    for i in 0..half {
        j[[half + i, i]] = 1.0; // J e_i = e_{i+half}
        j[[i, half + i]] = -1.0; // J e_{i+half} = -e_i
    }
    j
}

fn frob_norm(a: &Array2<f64>) -> f64 {
    a.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Relative commutator residual ‖M J − J M‖_F / ‖M‖_F ∈ [0, 2].
/// 0 ⟺ M commutes with J ⟺ M is complex-linear w.r.t. J.
/// Returns 0.0 for the zero matrix (no division by zero).
pub fn commutator_residual(m: &Array2<f64>, j: &Array2<f64>) -> f64 {
    let comm = &m.dot(j) - &j.dot(m);
    let den = frob_norm(m);
    if den == 0.0 {
        0.0
    } else {
        frob_norm(&comm) / den
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn j_squares_to_negative_identity() {
        let j = complex_structure_split_half(4);
        let jj = j.dot(&j);
        let neg_i = -Array2::<f64>::eye(4);
        for i in 0..4 {
            for k in 0..4 {
                assert!((jj[[i, k]] - neg_i[[i, k]]).abs() < 1e-12,
                    "J^2 != -I at ({i},{k})");
            }
        }
    }

    #[test]
    fn realified_complex_matrix_has_zero_residual() {
        // M = [[A, -B], [B, A]] (2x2 blocks) commutes with split-half J on R^4.
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let mut m = Array2::<f64>::zeros((4, 4));
        m.slice_mut(s![0..2, 0..2]).assign(&a);
        m.slice_mut(s![0..2, 2..4]).assign(&(-&b));
        m.slice_mut(s![2..4, 0..2]).assign(&b);
        m.slice_mut(s![2..4, 2..4]).assign(&a);
        let j = complex_structure_split_half(4);
        assert!(commutator_residual(&m, &j) < 1e-12);
    }

    #[test]
    fn diagonal_matrix_has_positive_residual() {
        // diag(1,2,3,4) does not commute with J (it mixes paired coords).
        let m = Array2::from_diag(&array![1.0, 2.0, 3.0, 4.0]);
        let j = complex_structure_split_half(4);
        assert!(commutator_residual(&m, &j) > 0.1);
    }

    #[test]
    fn identity_has_zero_residual() {
        let m = Array2::<f64>::eye(4);
        let j = complex_structure_split_half(4);
        assert!(commutator_residual(&m, &j) < 1e-12);
    }

    #[test]
    fn zero_matrix_has_zero_residual_not_nan() {
        let m = Array2::<f64>::zeros((4, 4));
        let j = complex_structure_split_half(4);
        let r = commutator_residual(&m, &j);
        assert_eq!(r, 0.0);
    }

    #[test]
    #[should_panic]
    fn odd_dimension_panics() {
        let _ = complex_structure_split_half(3);
    }
}
```

- [ ] **Step 2: Wire the module into `canonical/mod.rs`**

Append to the end of `crates/larql-vindex/src/canonical/mod.rs`:

```rust
pub mod hilbertian;
pub use hilbertian::{commutator_residual, complex_structure_split_half};
```

- [ ] **Step 3: Run the tests**

Run: `cd /home/metavacua/larql && cargo test -p larql-vindex canonical::hilbertian 2>&1 | tail -10`
Expected: `test result: ok. 6 passed`

- [ ] **Step 4: Commit**

```bash
cd /home/metavacua/larql
git add crates/larql-vindex/src/canonical/hilbertian.rs crates/larql-vindex/src/canonical/mod.rs
git commit -m "feat(canonical): add split-half complex structure J + commutator residual"
```

---

## Task 2: Per-head coupling, head slicing, and GQA mapping

**Files:**
- Modify: `crates/larql-vindex/src/canonical/hilbertian.rs`

- [ ] **Step 1: Add the failing tests**

In `crates/larql-vindex/src/canonical/hilbertian.rs`, inside `mod tests`, add:

```rust
    #[test]
    fn kv_head_for_query_maps_gqa_groups() {
        // 15 query heads, 5 kv heads -> group size 3.
        assert_eq!(kv_head_for_query(0, 15, 5), 0);
        assert_eq!(kv_head_for_query(2, 15, 5), 0);
        assert_eq!(kv_head_for_query(3, 15, 5), 1);
        assert_eq!(kv_head_for_query(14, 15, 5), 4);
        // No GQA (num_kv == num_q): identity.
        assert_eq!(kv_head_for_query(2, 4, 4), 2);
        // Single kv head: everything maps to 0.
        assert_eq!(kv_head_for_query(7, 8, 1), 0);
    }

    #[test]
    fn head_block_extracts_rows() {
        // proj is [4, 3] = 2 heads of head_dim 2.
        let proj = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let h0 = head_block(&proj, 0, 2);
        let h1 = head_block(&proj, 1, 2);
        assert_eq!(h0, array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_eq!(h1, array![[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]);
    }

    #[test]
    fn head_coupling_is_wq_times_wk_transpose() {
        // wq, wk are [d_head=2, hidden=3]; C = wq · wkᵀ is [2,2].
        let wq = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let wk = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let c = head_coupling(&wq, &wk);
        // row0·wkᵀ = [1,4]; row1·wkᵀ = [2,5]
        assert_eq!(c, array![[1.0, 4.0], [2.0, 5.0]]);
    }

    #[test]
    fn head_hilbertian_residual_matches_manual_composition() {
        let wq = array![[1.0, 2.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]];
        let wk = array![[0.0, 1.0, 2.0, 0.0], [1.0, 0.0, 0.0, 3.0]];
        let j = complex_structure_split_half(2); // d_head = 2
        let direct = head_hilbertian_residual(&wq, &wk, &j);
        let manual = commutator_residual(&head_coupling(&wq, &wk), &j);
        assert!((direct - manual).abs() < 1e-15);
    }
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `cd /home/metavacua/larql && cargo test -p larql-vindex canonical::hilbertian 2>&1 | tail -10`
Expected: compile error — `kv_head_for_query`, `head_block`, `head_coupling`, `head_hilbertian_residual` not defined.

- [ ] **Step 3: Add the implementations**

In `crates/larql-vindex/src/canonical/hilbertian.rs`, add before the `#[cfg(test)]` block:

```rust
/// Map a query-head index to its KV-head index under grouped-query attention.
/// `num_q_heads` must be a multiple of `num_kv_heads`.
pub fn kv_head_for_query(query_head: usize, num_q_heads: usize, num_kv_heads: usize) -> usize {
    let group = num_q_heads / num_kv_heads;
    query_head / group.max(1)
}

/// Extract head `head`'s `[head_dim, hidden]` block from a stacked projection
/// matrix `[n*head_dim, hidden]` (PyTorch `[out, in]` orientation).
pub fn head_block(proj: &Array2<f64>, head: usize, head_dim: usize) -> Array2<f64> {
    proj.slice(s![head * head_dim..(head + 1) * head_dim, ..]).to_owned()
}

/// Per-head query/key coupling C = W_Q · W_Kᵀ, shape `[head_dim, head_dim]`.
/// `wq_head` and `wk_head` are both `[head_dim, hidden]`.
pub fn head_coupling(wq_head: &Array2<f64>, wk_head: &Array2<f64>) -> Array2<f64> {
    wq_head.dot(&wk_head.t())
}

/// Hilbertian residual for one head: ‖[C, J]‖_F / ‖C‖_F where C = W_Q W_Kᵀ.
/// `j` must be the split-half complex structure of dimension `head_dim`.
pub fn head_hilbertian_residual(
    wq_head: &Array2<f64>,
    wk_head: &Array2<f64>,
    j: &Array2<f64>,
) -> f64 {
    let c = head_coupling(wq_head, wk_head);
    commutator_residual(&c, j)
}
```

- [ ] **Step 4: Run tests to confirm they pass**

Run: `cd /home/metavacua/larql && cargo test -p larql-vindex canonical::hilbertian 2>&1 | tail -10`
Expected: `test result: ok. 10 passed`

- [ ] **Step 5: Update re-exports in `canonical/mod.rs`**

In `crates/larql-vindex/src/canonical/mod.rs`, change the hilbertian re-export line to:

```rust
pub use hilbertian::{
    commutator_residual, complex_structure_split_half, head_block, head_coupling,
    head_hilbertian_residual, kv_head_for_query,
};
```

- [ ] **Step 6: Commit**

```bash
cd /home/metavacua/larql
git add crates/larql-vindex/src/canonical/hilbertian.rs crates/larql-vindex/src/canonical/mod.rs
git commit -m "feat(canonical): add per-head coupling, head slicing, GQA mapping, head residual"
```

---

## Task 3: Result types and the `hilbertian_meta.json` filename constant

**Files:**
- Modify: `crates/larql-vindex/src/canonical/types.rs`
- Modify: `crates/larql-vindex/src/canonical/mod.rs`
- Modify: `crates/larql-vindex/src/format/filenames.rs`

- [ ] **Step 1: Add the failing tests for the types**

In `crates/larql-vindex/src/canonical/types.rs`, inside `mod tests`, add:

```rust
    #[test]
    fn hilbertian_meta_round_trips_through_json() {
        let meta = HilbertianMeta {
            version: 1,
            model: "test/model".into(),
            hidden_size: 4,
            head_dim: 2,
            num_q_heads: 2,
            num_kv_heads: 1,
            complex_structure: "split_half".into(),
            heads: vec![
                HeadHilbertianInfo { layer: 0, query_head: 0, kv_head: 0, residual: 0.0 },
                HeadHilbertianInfo { layer: 0, query_head: 1, kv_head: 0, residual: 1.25 },
            ],
        };
        let json = serde_json::to_string_pretty(&meta).unwrap();
        let back: HilbertianMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, "test/model");
        assert_eq!(back.heads.len(), 2);
        assert_eq!(back.heads[1].query_head, 1);
        assert!((back.heads[1].residual - 1.25).abs() < 1e-15);
        assert_eq!(back.complex_structure, "split_half");
    }
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `cd /home/metavacua/larql && cargo test -p larql-vindex canonical::types::tests::hilbertian_meta_round_trips_through_json 2>&1 | tail -10`
Expected: compile error — `HilbertianMeta` / `HeadHilbertianInfo` not defined.

- [ ] **Step 3: Add the types**

In `crates/larql-vindex/src/canonical/types.rs`, after the `CanonicalMeta` impl block (before `#[cfg(test)]`), add:

```rust
/// Per-head Hilbertian residual: how close head `query_head`'s query/key
/// coupling is to complex-linear w.r.t. the split-half complex structure.
/// `residual` ∈ [0, 2]; 0 = exactly complex-linear (an upper bound on the
/// optimal-J residual).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadHilbertianInfo {
    pub layer: usize,
    pub query_head: usize,
    pub kv_head: usize,
    pub residual: f64,
}

/// Root metadata written to `hilbertian_meta.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HilbertianMeta {
    pub version: u32,
    pub model: String,
    pub hidden_size: usize,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    /// The fixed complex-structure convention used (currently always "split_half").
    pub complex_structure: String,
    pub heads: Vec<HeadHilbertianInfo>,
}
```

- [ ] **Step 4: Run test to confirm it passes**

Run: `cd /home/metavacua/larql && cargo test -p larql-vindex canonical::types::tests::hilbertian_meta_round_trips_through_json 2>&1 | tail -5`
Expected: `test result: ok. 1 passed`

- [ ] **Step 5: Re-export the types in `canonical/mod.rs`**

In `crates/larql-vindex/src/canonical/mod.rs`, change the `types` re-export line so it also exports the new types. The current line is:

```rust
pub use types::{CanonicalMeta, LayerCanonicalInfo, Regime};
```

Replace it with:

```rust
pub use types::{
    CanonicalMeta, HeadHilbertianInfo, HilbertianMeta, LayerCanonicalInfo, Regime,
};
```

- [ ] **Step 6: Add the filename constant + its uniqueness test**

In `crates/larql-vindex/src/format/filenames.rs`, in the "Canonical form sidecar" section (right after `CANONICAL_META_JSON`), add:

```rust
pub const HILBERTIAN_META_JSON: &str = "hilbertian_meta.json";
```

Then add `HILBERTIAN_META_JSON` to the `names` array in the `all_filenames_unique` test (so future additions can't collide with it), and add this test inside `mod tests`:

```rust
    #[test]
    fn hilbertian_meta_json_is_distinct_from_canonical() {
        assert_ne!(HILBERTIAN_META_JSON, CANONICAL_META_JSON);
        assert_eq!(HILBERTIAN_META_JSON, "hilbertian_meta.json");
    }
```

- [ ] **Step 7: Run tests**

Run: `cd /home/metavacua/larql && cargo test -p larql-vindex format::filenames canonical::types 2>&1 | tail -10`
Expected: all pass (including `all_filenames_unique` and the two new tests).

- [ ] **Step 8: Commit**

```bash
cd /home/metavacua/larql
git add crates/larql-vindex/src/canonical/types.rs \
        crates/larql-vindex/src/canonical/mod.rs \
        crates/larql-vindex/src/format/filenames.rs
git commit -m "feat(canonical): add HilbertianMeta types + hilbertian_meta.json constant"
```

---

## Task 4: Manifest-driven attention Q/K loader

**Files:**
- Create: `crates/larql-vindex/src/format/attn_load.rs`
- Modify: `crates/larql-vindex/src/format/mod.rs`

- [ ] **Step 1: Confirm the `StorageDtype` variant names**

Run: `cd /home/metavacua/larql && sed -n '12,42p' crates/larql-vindex/src/config/dtype.rs`
Note the exact variant names for f32 and f16 (the code below assumes `StorageDtype::F32` and `StorageDtype::F16`). If they differ (e.g. `Float32`), use the actual names in Step 3.

- [ ] **Step 2: Create the file with implementation + a hermetic test**

Create `crates/larql-vindex/src/format/attn_load.rs`:

```rust
//! Manifest-driven reader for per-layer attention Q/K weight matrices.
//!
//! `attn_weights.bin` is a header-less concatenation of tensors; offsets and
//! shapes come from `weight_manifest.json` (a top-level JSON array). We read
//! only the `q_proj` / `k_proj` tensors — no tokenizer, no forward pass, and
//! no full-model load.

use std::collections::HashMap;
use std::path::Path;

use ndarray::Array2;
use serde::Deserialize;

use crate::config::dtype::{decode_floats, StorageDtype};
use crate::error::VindexError;
use crate::format::filenames::{ATTN_WEIGHTS_BIN, WEIGHT_MANIFEST_JSON};

#[derive(Deserialize)]
struct ManifestEntry {
    key: String,
    shape: Vec<usize>,
    offset: usize,
    length: usize,
    file: String,
}

/// Load per-layer `(W_Q, W_K)` from `attn_weights.bin`.
/// `W_Q` is `[num_q_heads*head_dim, hidden]`, `W_K` is `[num_kv_heads*head_dim, hidden]`,
/// both as `f32` (decoded from on-disk f16/f32). Index in the returned Vec is the layer.
pub fn load_attention_qk(
    dir: &Path,
    num_layers: usize,
) -> Result<Vec<(Array2<f32>, Array2<f32>)>, VindexError> {
    let manifest_bytes = std::fs::read(dir.join(WEIGHT_MANIFEST_JSON))?;
    let entries: Vec<ManifestEntry> = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| VindexError::Parse(format!("weight_manifest.json: {e}")))?;
    let by_key: HashMap<&str, &ManifestEntry> =
        entries.iter().map(|e| (e.key.as_str(), e)).collect();

    let bin = std::fs::read(dir.join(ATTN_WEIGHTS_BIN))?;

    let mut out = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let q_key = format!("layers.{layer}.self_attn.q_proj.weight");
        let k_key = format!("layers.{layer}.self_attn.k_proj.weight");
        let wq = read_entry(&bin, lookup(&by_key, &q_key)?)?;
        let wk = read_entry(&bin, lookup(&by_key, &k_key)?)?;
        out.push((wq, wk));
    }
    Ok(out)
}

fn lookup<'a>(
    by_key: &HashMap<&str, &'a ManifestEntry>,
    key: &str,
) -> Result<&'a ManifestEntry, VindexError> {
    by_key
        .get(key)
        .copied()
        .ok_or_else(|| VindexError::Parse(format!("weight_manifest.json missing key {key}")))
}

fn read_entry(bin: &[u8], e: &ManifestEntry) -> Result<Array2<f32>, VindexError> {
    if e.file != ATTN_WEIGHTS_BIN {
        return Err(VindexError::Parse(format!(
            "entry {} is in {}, expected {ATTN_WEIGHTS_BIN}",
            e.key, e.file
        )));
    }
    if e.shape.len() != 2 {
        return Err(VindexError::Parse(format!(
            "entry {} has non-2D shape {:?}",
            e.key, e.shape
        )));
    }
    let (rows, cols) = (e.shape[0], e.shape[1]);
    let n = rows * cols;
    if n == 0 {
        return Err(VindexError::Parse(format!("entry {} is empty", e.key)));
    }
    let dtype = match e.length / n {
        4 => StorageDtype::F32,
        2 => StorageDtype::F16,
        other => {
            return Err(VindexError::Parse(format!(
                "entry {} has {other} bytes/elem (expected 2 or 4)",
                e.key
            )))
        }
    };
    let end = e.offset + e.length;
    if end > bin.len() {
        return Err(VindexError::Parse(format!(
            "entry {} range {}..{end} exceeds {ATTN_WEIGHTS_BIN} ({} bytes)",
            e.key,
            e.offset,
            bin.len()
        )));
    }
    let floats = decode_floats(&bin[e.offset..end], dtype);
    Array2::from_shape_vec((rows, cols), floats)
        .map_err(|e| VindexError::Parse(format!("reshape attention weight: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_qk_from_a_synthetic_vindex() {
        let dir = tempfile::tempdir().unwrap();
        // Two f32 tensors: q_proj [2,2] then k_proj [2,2], concatenated.
        let q: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let k: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
        let mut bin = Vec::new();
        for v in q.iter().chain(k.iter()) {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(dir.path().join(ATTN_WEIGHTS_BIN), &bin).unwrap();

        let manifest = serde_json::json!([
            {"key": "layers.0.self_attn.q_proj.weight", "shape": [2, 2],
             "offset": 0, "length": 16, "file": ATTN_WEIGHTS_BIN},
            {"key": "layers.0.self_attn.k_proj.weight", "shape": [2, 2],
             "offset": 16, "length": 16, "file": ATTN_WEIGHTS_BIN}
        ]);
        std::fs::write(
            dir.path().join(WEIGHT_MANIFEST_JSON),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let qk = load_attention_qk(dir.path(), 1).unwrap();
        assert_eq!(qk.len(), 1);
        let (wq, wk) = &qk[0];
        assert_eq!(wq.shape(), [2, 2]);
        assert_eq!(wk.shape(), [2, 2]);
        assert_eq!(wq[[0, 0]], 1.0);
        assert_eq!(wq[[1, 1]], 4.0);
        assert_eq!(wk[[0, 0]], 5.0);
        assert_eq!(wk[[1, 1]], 8.0);
    }

    #[test]
    fn missing_key_errors() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(ATTN_WEIGHTS_BIN), [0u8; 16]).unwrap();
        std::fs::write(
            dir.path().join(WEIGHT_MANIFEST_JSON),
            serde_json::to_vec(&serde_json::json!([])).unwrap(),
        )
        .unwrap();
        assert!(load_attention_qk(dir.path(), 1).is_err());
    }
}
```

- [ ] **Step 3: Adjust dtype variant names if needed**

If Step 1 showed the variants are not `F32` / `F16`, edit the `match e.length / n` arms in `read_entry` accordingly.

- [ ] **Step 4: Wire the module into `format/mod.rs`**

In `crates/larql-vindex/src/format/mod.rs`, add (next to the other `pub mod` lines, alphabetically near `attn_*` or at the top of the module list):

```rust
pub mod attn_load;
```

- [ ] **Step 5: Run tests**

Run: `cd /home/metavacua/larql && cargo test -p larql-vindex format::attn_load 2>&1 | tail -10`
Expected: `test result: ok. 2 passed`

- [ ] **Step 6: Confirm `tempfile` is available for the test**

`tempfile` is already used by `format::filenames` tests, so it is a dev-dependency. If the build complains it is missing, run `cargo add --dev tempfile -p larql-vindex` and re-run Step 5.

- [ ] **Step 7: Commit**

```bash
cd /home/metavacua/larql
git add crates/larql-vindex/src/format/attn_load.rs crates/larql-vindex/src/format/mod.rs
git commit -m "feat(format): add manifest-driven attention Q/K loader"
```

---

## Task 5: `larql hilbertian` CLI command

**Files:**
- Create: `crates/larql-cli/src/commands/extraction/hilbertian_cmd.rs`
- Modify: `crates/larql-cli/src/commands/extraction/mod.rs`
- Modify: `crates/larql-cli/src/main.rs`

- [ ] **Step 1: Create the command file**

Create `crates/larql-cli/src/commands/extraction/hilbertian_cmd.rs`:

```rust
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_vindex::{
    canonical::{
        complex_structure_split_half, head_block, head_hilbertian_residual, kv_head_for_query,
        HeadHilbertianInfo, HilbertianMeta,
    },
    format::{attn_load::load_attention_qk, filenames::HILBERTIAN_META_JSON, load::load_vindex_config},
};

#[derive(Args)]
pub struct HilbertianArgs {
    /// Path to the .vindex directory to analyse.
    vindex: PathBuf,
}

pub fn run(args: HilbertianArgs) -> Result<(), Box<dyn std::error::Error>> {
    let dir = &args.vindex;
    let t0 = Instant::now();
    println!("Hilbertian residual for vindex at {}", dir.display());

    let config = load_vindex_config(dir)?;
    let mc = config
        .model_config
        .as_ref()
        .ok_or("hilbertian: index.json has no model_config (need head_dim / head counts)")?;
    let (num_q, num_kv, head_dim, hidden) =
        (mc.num_q_heads, mc.num_kv_heads, mc.head_dim, config.hidden_size);
    println!(
        "  model: {} ({}), {} layers, hidden={hidden}, q_heads={num_q}, kv_heads={num_kv}, head_dim={head_dim}",
        config.model, config.family, config.num_layers
    );

    let j = complex_structure_split_half(head_dim);

    print!("  loading attention Q/K ... ");
    let qk = load_attention_qk(dir, config.num_layers)?;
    println!("{} layers ({:.1}ms)", qk.len(), t0.elapsed().as_secs_f64() * 1000.0);

    let mut heads: Vec<HeadHilbertianInfo> = Vec::with_capacity(config.num_layers * num_q);
    for (layer, (wq, wk)) in qk.iter().enumerate() {
        let wq64 = wq.mapv(|v| v as f64);
        let wk64 = wk.mapv(|v| v as f64);
        for h in 0..num_q {
            let g = kv_head_for_query(h, num_q, num_kv);
            let wq_h = head_block(&wq64, h, head_dim);
            let wk_g = head_block(&wk64, g, head_dim);
            let residual = head_hilbertian_residual(&wq_h, &wk_g, &j);
            heads.push(HeadHilbertianInfo { layer, query_head: h, kv_head: g, residual });
        }
    }

    let meta = HilbertianMeta {
        version: 1,
        model: config.model.clone(),
        hidden_size: hidden,
        head_dim,
        num_q_heads: num_q,
        num_kv_heads: num_kv,
        complex_structure: "split_half".into(),
        heads,
    };

    let out = dir.join(HILBERTIAN_META_JSON);
    std::fs::write(&out, serde_json::to_string_pretty(&meta)?)?;

    let rs: Vec<f64> = meta.heads.iter().map(|h| h.residual).collect();
    let mean = rs.iter().sum::<f64>() / rs.len() as f64;
    let min = rs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = rs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "  residual over {} heads: mean {:.4}, min {:.4}, max {:.4}",
        rs.len(),
        mean,
        min,
        max
    );
    println!("  wrote {}", out.display());
    println!("  total: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}
```

- [ ] **Step 2: Register the module**

In `crates/larql-cli/src/commands/extraction/mod.rs`, add (alphabetically, after `pub mod hf_cmd;`):

```rust
pub mod hilbertian_cmd;
```

- [ ] **Step 3: Add the `Hilbertian` variant to the `Commands` enum**

In `crates/larql-cli/src/main.rs`, in the "Build / extract" section of `enum Commands` (after the `Canonicalize` variant), add:

```rust
    #[command(next_help_heading = "Build")]
    /// Score per-head complex-linearity (Hilbertian residual); writes hilbertian_meta.json.
    Hilbertian(hilbertian_cmd::HilbertianArgs),
```

- [ ] **Step 4: Add the dispatch arm**

In `crates/larql-cli/src/main.rs`, in the `let result = match cli.command {` block, in the `// ── Build / extract ──` section (right after the `Commands::Canonicalize(args) => ...` arm), add:

```rust
        Commands::Hilbertian(args) => hilbertian_cmd::run(args),
```

- [ ] **Step 5: Build**

Run: `cd /home/metavacua/larql && cargo build -p larql-cli 2>&1 | tail -20`
Expected: compiles without errors. If `model_config` field access fails, confirm field names with `sed -n '10,30p' crates/larql-vindex/src/config/model.rs` and adjust.

- [ ] **Step 6: Commit**

```bash
cd /home/metavacua/larql
git add crates/larql-cli/src/commands/extraction/hilbertian_cmd.rs \
        crates/larql-cli/src/commands/extraction/mod.rs \
        crates/larql-cli/src/main.rs
git commit -m "feat(cli): add larql hilbertian command"
```

---

## Task 6: Integration smoke test, discrimination check, full suite

**Files:** none modified unless a fixup is needed.

- [ ] **Step 1: Run the full workspace tests for the touched crates**

Run: `cd /home/metavacua/larql && cargo test -p larql-vindex --lib 2>&1 | tail -8`
Expected: all pass. (Note: the pre-existing integration test `vector_extractor_ffn_down_byte_identical` in `tests/test_walker_accuracy.rs` fails on this environment independent of these changes — it is a 64↔32-bit golden-drift failure that also fails on the base commit. Use `--lib` to scope to unit tests and avoid it.)

Run: `cd /home/metavacua/larql && cargo test -p larql-cli 2>&1 | tail -5`
Expected: all pass.

- [ ] **Step 2: Build release and run the smoke test on SmolLM2-360M**

```
cd /home/metavacua/larql && cargo build --release -p larql-cli 2>&1 | tail -3
./target/release/larql hilbertian /home/metavacua/larql-vindexes/smollm2-360m.vindex 2>&1
```
Expected: prints `model: ... (llama), 32 layers, hidden=960, q_heads=15, kv_heads=5, head_dim=64`, then a `residual over 480 heads: mean ... min ... max ...` line, then `wrote .../hilbertian_meta.json`. Should complete in well under a minute (each commutator is 64×64).

- [ ] **Step 3: Verify the JSON structure and that the metric is discriminating**

```
python3 -c "
import json
m = json.load(open('/home/metavacua/larql-vindexes/smollm2-360m.vindex/hilbertian_meta.json'))
assert m['version'] == 1
assert m['head_dim'] == 64 and m['num_q_heads'] == 15 and m['num_kv_heads'] == 5
rs = [h['residual'] for h in m['heads']]
assert len(rs) == 32 * 15, f'expected 480 heads, got {len(rs)}'
lo, hi = min(rs), max(rs)
print(f'heads={len(rs)} min={lo:.4f} max={hi:.4f} mean={sum(rs)/len(rs):.4f}')
assert all(0.0 <= r <= 2.0 + 1e-9 for r in rs), 'residual out of [0,2]'
# Discrimination: the metric must not be degenerate (all ~equal). If it IS
# degenerate, that is itself a finding to record (see Step 5), not a pass.
assert hi - lo > 1e-3, 'DEGENERATE: residuals do not vary across heads'
print('OK — metric is well-formed and discriminating')
"
```
Expected: prints the stats and `OK — metric is well-formed and discriminating`.

- [ ] **Step 4: Confirm existing vindex files were not modified**

```
cd /home/metavacua/larql-vindexes/smollm2-360m.vindex && ls -la canonical_meta.json hilbertian_meta.json attn_weights.bin index.json
```
Expected: `hilbertian_meta.json` exists; the command only ever wrote that one file (verify by inspection that `hilbertian_cmd.rs` has exactly one `std::fs::write`, to the `HILBERTIAN_META_JSON` path).

- [ ] **Step 5: If Step 3's discrimination assertion FAILS (degenerate metric)**

Do not silently pass. Record the finding the way #134 was recorded: post an issue **only to `metavacua/larql-to-sparql`** (the hard remote constraint — verify with `git remote -v` that you push to `fork`, never to `origin`/chrishayuk), titled e.g. "hilbertian: split-half-J residual is degenerate on SmolLM2", describing the observed distribution, and note that the optimal-J fit (below) is the likely fix. Then continue — the implementation is still correct; the metric choice is the open question.

- [ ] **Step 6: File the deferred-refinement issue**

Post an issue to `metavacua/larql-to-sparql` (verify remote first) titled "hilbertian: optimal-J fit, residual-space QK form, whitened-basis" capturing the three v1 simplifications documented in this plan's "Scope" note: (a) optimal `J` minimised over complex structures instead of fixed split-half, (b) the `hidden×hidden` residual-space QK form `W_Qᵀ W_K` instead of the `d_head×d_head` coupling, (c) computing in the canonical whitened basis so the residual is a function of the canonical form. Reference this plan and the `larql hilbertian` command.

- [ ] **Step 7: Commit any fixups**

If Steps 1–4 required code changes, commit them:

```bash
cd /home/metavacua/larql
git add -A
git commit -m "fix(hilbertian): resolve integration issues found in smoke test"
```

Otherwise nothing to commit for this task.

---

## Self-review checklist

**Spec coverage** (the "spec" is the metric design in the Domain background section):
- [x] Split-half complex structure `J` with `J²=−I` — Task 1 (`complex_structure_split_half`)
- [x] Relative commutator residual `‖[C,J]‖/‖C‖` — Task 1 (`commutator_residual`)
- [x] Per-head coupling `C = W_Q W_Kᵀ` + GQA head pairing + head slicing — Task 2
- [x] Result types + sidecar filename — Task 3
- [x] Weights-only attention Q/K loader (no corpus, no forward pass) — Task 4
- [x] `larql hilbertian` command writing `hilbertian_meta.json` — Task 5
- [x] Integration + discrimination check + untouched-files check — Task 6
- [ ] **Deferred (Task 6 Step 6 issue):** optimal-J fit, residual-space QK form, whitened-basis computation

**Type consistency:**
- `complex_structure_split_half(n) -> Array2<f64>` defined Task 1, called Task 5 with `head_dim`.
- `head_block(proj, head, head_dim)`, `head_coupling`, `head_hilbertian_residual`, `kv_head_for_query` defined Task 2, all consumed in Task 5 with matching signatures.
- `HeadHilbertianInfo { layer, query_head, kv_head, residual }` and `HilbertianMeta { version, model, hidden_size, head_dim, num_q_heads, num_kv_heads, complex_structure, heads }` defined Task 3, constructed identically in Task 5.
- `load_attention_qk(dir, num_layers) -> Result<Vec<(Array2<f32>, Array2<f32>)>, VindexError>` defined Task 4, called Task 5.
- `HILBERTIAN_META_JSON` defined Task 3, used Task 5.
- CLI runner returns `Result<(), Box<dyn std::error::Error>>` and the dispatch arm returns it directly (no `?`) — matches the `Canonicalize` precedent.

**No placeholders:** every code step contains complete code; every run step has an exact command and expected output. The two investigation steps (Task 4 Step 1 dtype variant names, Task 5 Step 5 field-name fallback) point at exact file:line ranges to read, not "figure it out."
