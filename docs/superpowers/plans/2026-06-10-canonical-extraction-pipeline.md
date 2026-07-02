# Canonical Extraction Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `larql canonicalize <vindex-path>` — a command that computes the flat-canonical (single G constant across all layers, semi-intrinsic W_E calibration) form of a vindex and writes `canonical_meta.json` into the vindex directory.

**Architecture:** A new `canonical` module in `larql-vindex` implements four pure algorithms: covariance estimation from the token embedding matrix, Cholesky whitening factor computation, on-shell feature scoring via c_score percentile, and regime classification per layer from the existing layer bands and activation density. The `larql-cli` gets a `Canonicalize` subcommand that opens a vindex directory, runs the pipeline, and writes `canonical_meta.json`. The existing vindex files are left untouched — canonicalize is purely additive.

**Tech Stack:** Rust 2021, ndarray 0.16, existing `larql_compute::cpu::ops::linalg` (Cholesky, cholesky_inverse), `larql-vindex` format loaders (embeddings.bin, down_meta.bin, gate_vectors.bin, index.json), serde_json, clap.

---

## Scope note: three separate plans

This synthesis document covers three independent subsystems. This plan covers **Plan 1 only**:

- **Plan 1 (this):** Canonical extraction pipeline — `larql canonicalize`. The load-bearing missing piece. No prerequisites.
- **Plan 2 (separate):** Knowledge graph integration — INSERT ops importer + entity→QID alignment. Depends on Plan 1 canonical_meta.json for Class A label addressing.
- **Plan 3 (separate):** Cross-model Procrustes alignment CLI — `larql align`. Depends on Plan 1 whitening factor.

---

## Key domain concepts

- **G (activation covariance):** `G = (1/N) Σ_v (embed_scale · W_E[v])^T (embed_scale · W_E[v])` where W_E is the token embedding matrix. Shape [hidden_size, hidden_size]. The flat-canonical choice uses a single G for all layers (semi-intrinsic, computed from model weights alone, no external corpus needed).
- **Cholesky whitening:** `G = L L^T`. The whitened gate vector is `g̃_f = L^{-T} g_f` (back-solve `L^T g̃_f = g_f`). Two whitened vectors have Mahalanobis inner product = ordinary dot product.
- **c_score:** the logit of the feature's top down-projected token (already in `down_meta.bin`). High c_score = feature strongly predicts a specific token = factual/on-shell.
- **On-shell filter:** the top 15% features by c_score within each layer. These are the "detail wavelets" (factual subspace). Features below the 85th percentile are structural ("dark space").
- **Regime per layer:** Wave (mean_density > 0.5), Particle (mean_density < 0.05), Wavelet (otherwise). `mean_density` = fraction of layer features with c_score above a low floor (c_score > 0.1 indicates the feature fires for at least some inputs).

## File structure

**New files:**
- `crates/larql-compute/src/cpu/ops/linalg.rs` — add `back_solve_lt`, `compute_l_inv_t` functions
- `crates/larql-vindex/src/canonical/mod.rs`
- `crates/larql-vindex/src/canonical/types.rs` — `Regime`, `LayerCanonicalInfo`, `CanonicalMeta`
- `crates/larql-vindex/src/canonical/covariance.rs` — `estimate_covariance`
- `crates/larql-vindex/src/canonical/whitening.rs` — `WhiteningData`, `compute_whitening`
- `crates/larql-vindex/src/canonical/onshell.rs` — `compute_onshell_mask`
- `crates/larql-vindex/src/canonical/regime.rs` — `classify_layer_regime`
- `crates/larql-cli/src/commands/extraction/canonicalize_cmd.rs`

**Modified files:**
- `crates/larql-compute/src/cpu/ops/linalg.rs` — add two functions
- `crates/larql-vindex/src/format/down_meta.rs` — add `read_cscores_binary` (no tokenizer needed)
- `crates/larql-vindex/src/format/filenames.rs` — add `CANONICAL_META_JSON` constant
- `crates/larql-vindex/src/lib.rs` — add `pub mod canonical;` and re-export types
- `crates/larql-cli/src/commands/extraction/mod.rs` — add `pub mod canonicalize_cmd;`
- `crates/larql-cli/src/main.rs` — add `Canonicalize` variant to `Commands`

---

## Task 1: `back_solve_lt` and `compute_l_inv_t` in linalg.rs

These are the only new numerical primitives needed. Everything else uses existing Cholesky.

**Files:**
- Modify: `crates/larql-compute/src/cpu/ops/linalg.rs`

- [ ] **Step 1: Write the failing tests**

Add to the bottom of `crates/larql-compute/src/cpu/ops/linalg.rs`:

```rust
#[cfg(test)]
mod canonical_linalg_tests {
    use super::*;
    use ndarray::Array2;

    fn lower_3x3() -> Array2<f64> {
        // L = [[2,0,0],[1,3,0],[4,5,6]]
        let mut l = Array2::<f64>::zeros((3, 3));
        l[[0,0]] = 2.0; l[[1,0]] = 1.0; l[[1,1]] = 3.0;
        l[[2,0]] = 4.0; l[[2,1]] = 5.0; l[[2,2]] = 6.0;
        l
    }

    #[test]
    fn back_solve_lt_recovers_rhs() {
        // L^T z = b, check L^T (back_solve_lt(L, b)) == b
        let l = lower_3x3();
        let b = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let z = back_solve_lt(&l, &b);
        // Verify L^T z == b
        let lt = l.t().to_owned();
        for col in 0..2 {
            for row in 0..3 {
                let dot: f64 = (0..3).map(|k| lt[[row, k]] * z[[k, col]]).sum();
                assert!((dot - b[[row, col]]).abs() < 1e-10, "row={row} col={col} dot={dot} expected={}", b[[row,col]]);
            }
        }
    }

    #[test]
    fn compute_l_inv_t_times_l_t_is_identity() {
        let l = lower_3x3();
        let l_inv_t = compute_l_inv_t(&l);
        // l_inv_t @ l^T should == I
        let lt = l.t().to_owned();
        let product = l_inv_t.dot(&lt);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((product[[i,j]] - expected).abs() < 1e-10,
                    "product[{i},{j}]={} expected={expected}", product[[i,j]]);
            }
        }
    }
}
```

- [ ] **Step 2: Run tests to confirm they fail**

```
cargo test -p larql-compute canonical_linalg_tests 2>&1 | tail -5
```
Expected: FAIL — `back_solve_lt` and `compute_l_inv_t` not defined.

- [ ] **Step 3: Implement `back_solve_lt` and `compute_l_inv_t`**

Add to `crates/larql-compute/src/cpu/ops/linalg.rs` before the `#[cfg(test)]` block:

```rust
/// Back-substitution: solve L^T X = B where L is lower-triangular.
/// L^T is upper-triangular, so we iterate from bottom row to top.
/// Returns X of the same shape as B.
pub fn back_solve_lt(l: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let n = l.shape()[0];
    let m = b.shape()[1];
    let mut x = Array2::<f64>::zeros((n, m));
    // L^T[i,j] = L[j,i]. Upper-triangular back-substitution:
    for i in (0..n).rev() {
        for col in 0..m {
            let mut sum = b[[i, col]];
            for k in (i + 1)..n {
                sum -= l[[k, i]] * x[[k, col]]; // L^T[i,k] = L[k,i]
            }
            x[[i, col]] = sum / l[[i, i]];
        }
    }
    x
}

/// Compute L^{-T} explicitly: the d×d matrix such that L^{-T} @ L^T = I.
/// Equivalent to: for each unit column e_j, solve L^T x = e_j via back_solve_lt.
/// Returns a d×d array — the whitening matrix for gate vectors.
pub fn compute_l_inv_t(l: &Array2<f64>) -> Array2<f64> {
    let n = l.shape()[0];
    let identity = Array2::<f64>::eye(n);
    back_solve_lt(l, &identity)
}
```

- [ ] **Step 4: Run tests to confirm they pass**

```
cargo test -p larql-compute canonical_linalg_tests 2>&1 | tail -5
```
Expected: `test result: ok. 2 passed`

- [ ] **Step 5: Run full larql-compute tests to verify no regression**

```
cargo test -p larql-compute 2>&1 | tail -5
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/larql-compute/src/cpu/ops/linalg.rs
git commit -m "feat(linalg): add back_solve_lt and compute_l_inv_t for canonical whitening"
```

---

## Task 2: `CANONICAL_META_JSON` filename constant

**Files:**
- Modify: `crates/larql-vindex/src/format/filenames.rs`

- [ ] **Step 1: Write the failing test**

In `crates/larql-vindex/src/format/filenames.rs`, inside `mod tests`, add:

```rust
#[test]
fn canonical_meta_json_is_unique() {
    // CANONICAL_META_JSON must not collide with any existing constant.
    let existing = [
        INDEX_JSON, TOKENIZER_JSON, TOKENIZER_CONFIG_JSON, GENERATION_CONFIG_JSON,
        WEIGHT_MANIFEST_JSON, EMBEDDINGS_BIN, NORMS_BIN, GATE_VECTORS_BIN, DOWN_META_BIN,
    ];
    for name in existing {
        assert_ne!(CANONICAL_META_JSON, name,
            "CANONICAL_META_JSON collides with {name}");
    }
    assert_eq!(CANONICAL_META_JSON, "canonical_meta.json");
}
```

- [ ] **Step 2: Run test to confirm it fails**

```
cargo test -p larql-vindex canonical_meta_json_is_unique 2>&1 | tail -5
```
Expected: FAIL — constant not defined.

- [ ] **Step 3: Add the constant**

In `crates/larql-vindex/src/format/filenames.rs`, after the `INDEX_JSON` group (around line 22), add:

```rust
// ── Canonical form sidecar ──────────────────────────────────────────────
pub const CANONICAL_META_JSON: &str = "canonical_meta.json";
```

- [ ] **Step 4: Run test to confirm it passes**

```
cargo test -p larql-vindex canonical_meta_json_is_unique 2>&1 | tail -5
```
Expected: `test result: ok. 1 passed`

- [ ] **Step 5: Commit**

```bash
git add crates/larql-vindex/src/format/filenames.rs
git commit -m "feat(filenames): add CANONICAL_META_JSON constant"
```

---

## Task 3: Canonical types

**Files:**
- Create: `crates/larql-vindex/src/canonical/types.rs`
- Create: `crates/larql-vindex/src/canonical/mod.rs`
- Modify: `crates/larql-vindex/src/lib.rs`

- [ ] **Step 1: Write failing tests (in types.rs, will fail until the file exists)**

Create `crates/larql-vindex/src/canonical/types.rs`:

```rust
use serde::{Deserialize, Serialize};

/// Wave/Particle/Wavelet activation regime for a transformer layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Regime {
    /// Dense activations — many weak gates fire (e.g., early syntax layers).
    Wave,
    /// Sparse activations — few strong gates fire (e.g., MoE knowledge layers).
    Particle,
    /// Mixed — multi-resolution, some wave structure, some particle selectivity.
    Wavelet,
}

/// Per-layer canonical metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerCanonicalInfo {
    pub layer: usize,
    pub regime: Regime,
    /// Number of features that pass the on-shell filter (top 15% by c_score).
    pub on_shell_count: usize,
    /// Total features at this layer.
    pub total_features: usize,
    /// Fraction of features with c_score > 0.1 (activation density proxy).
    pub mean_density: f32,
}

/// Root canonical metadata written to `canonical_meta.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalMeta {
    /// Format version; increment when the schema changes.
    pub version: u32,
    pub model: String,
    pub family: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    /// Number of embedding rows sampled to estimate G.
    pub covariance_sample_size: usize,
    /// embed_scale used when computing G.
    pub embed_scale: f32,
    /// Cholesky factor L packed row-major lower-triangle:
    /// indices (i,j) with j<=i stored as L[i*(i+1)/2 + j].
    /// Length = hidden_size * (hidden_size + 1) / 2. Values are f64.
    pub cholesky_l_packed: Vec<f64>,
    /// Per-layer info.
    pub layers: Vec<LayerCanonicalInfo>,
}

impl CanonicalMeta {
    /// Unpack the lower-triangular Cholesky factor into a dense d×d matrix.
    pub fn unpack_cholesky_l(&self) -> ndarray::Array2<f64> {
        let d = self.hidden_size;
        let mut l = ndarray::Array2::<f64>::zeros((d, d));
        for i in 0..d {
            for j in 0..=i {
                l[[i, j]] = self.cholesky_l_packed[i * (i + 1) / 2 + j];
            }
        }
        l
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regime_serialises_as_snake_case() {
        assert_eq!(serde_json::to_string(&Regime::Wave).unwrap(), "\"wave\"");
        assert_eq!(serde_json::to_string(&Regime::Particle).unwrap(), "\"particle\"");
        assert_eq!(serde_json::to_string(&Regime::Wavelet).unwrap(), "\"wavelet\"");
    }

    #[test]
    fn canonical_meta_round_trips_through_json() {
        let meta = CanonicalMeta {
            version: 1,
            model: "test/model".into(),
            family: "llama".into(),
            num_layers: 2,
            hidden_size: 4,
            covariance_sample_size: 32,
            embed_scale: 1.0,
            // 4×4 lower triangle: 10 values (indices 0..9)
            cholesky_l_packed: vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0],
            layers: vec![
                LayerCanonicalInfo {
                    layer: 0, regime: Regime::Wave,
                    on_shell_count: 1, total_features: 4, mean_density: 0.75,
                },
                LayerCanonicalInfo {
                    layer: 1, regime: Regime::Particle,
                    on_shell_count: 1, total_features: 4, mean_density: 0.02,
                },
            ],
        };
        let json = serde_json::to_string_pretty(&meta).unwrap();
        let back: CanonicalMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(back.family, "llama");
        assert_eq!(back.layers[0].regime, Regime::Wave);
        assert_eq!(back.layers[1].regime, Regime::Particle);
        assert_eq!(back.cholesky_l_packed.len(), 10);
    }

    #[test]
    fn unpack_cholesky_l_recovers_diagonal() {
        // Packed lower-triangle of 3×3: [L00, L10, L11, L20, L21, L22]
        let meta = CanonicalMeta {
            version: 1, model: "x".into(), family: "y".into(),
            num_layers: 1, hidden_size: 3,
            covariance_sample_size: 8, embed_scale: 1.0,
            cholesky_l_packed: vec![2.0, 1.0, 3.0, 4.0, 5.0, 6.0],
            layers: vec![],
        };
        let l = meta.unpack_cholesky_l();
        assert_eq!(l[[0, 0]], 2.0);
        assert_eq!(l[[1, 0]], 1.0);
        assert_eq!(l[[1, 1]], 3.0);
        assert_eq!(l[[2, 0]], 4.0);
        assert_eq!(l[[2, 1]], 5.0);
        assert_eq!(l[[2, 2]], 6.0);
        // Upper triangle must be zero
        assert_eq!(l[[0, 1]], 0.0);
        assert_eq!(l[[0, 2]], 0.0);
    }
}
```

- [ ] **Step 2: Create `crates/larql-vindex/src/canonical/mod.rs`**

```rust
pub mod types;
pub use types::{CanonicalMeta, LayerCanonicalInfo, Regime};
```

- [ ] **Step 3: Add `pub mod canonical;` to `crates/larql-vindex/src/lib.rs`**

In `crates/larql-vindex/src/lib.rs`, after `pub mod config;`, add:

```rust
pub mod canonical;
```

Also add re-exports after the existing `// Re-export essentials at crate root` section:

```rust
// Canonical
pub use canonical::{CanonicalMeta, LayerCanonicalInfo, Regime};
```

- [ ] **Step 4: Run tests**

```
cargo test -p larql-vindex canonical::types 2>&1 | tail -5
```
Expected: `test result: ok. 3 passed`

- [ ] **Step 5: Commit**

```bash
git add crates/larql-vindex/src/canonical/ crates/larql-vindex/src/lib.rs
git commit -m "feat(canonical): add CanonicalMeta, LayerCanonicalInfo, Regime types"
```

---

## Task 4: C-score-only reader for down_meta

We need to read `down_meta.bin` without constructing token strings (no tokenizer dependency).

**Files:**
- Modify: `crates/larql-vindex/src/format/down_meta.rs`

- [ ] **Step 1: Write the failing test**

In `crates/larql-vindex/src/format/down_meta.rs`, inside the existing `#[cfg(test)]` block, add:

```rust
#[test]
fn read_cscores_binary_matches_full_read() {
    // Build a tiny down_meta.bin with known c_scores, then verify
    // read_cscores_binary extracts them correctly.
    let dir = tempfile::tempdir().unwrap();
    let meta: Vec<Option<Vec<Option<FeatureMeta>>>> = vec![
        Some(vec![
            Some(FeatureMeta {
                top_token: "hello".into(),
                top_token_id: 1,
                c_score: 3.5,
                top_k: vec![larql_models::TopKEntry {
                    token: "hello".into(), token_id: 1, logit: 3.5,
                }],
            }),
            Some(FeatureMeta {
                top_token: "world".into(),
                top_token_id: 2,
                c_score: 1.2,
                top_k: vec![larql_models::TopKEntry {
                    token: "world".into(), token_id: 2, logit: 1.2,
                }],
            }),
        ]),
        None,
    ];
    write_binary(dir.path(), &meta, 1).unwrap();
    let cscores = read_cscores_binary(dir.path()).unwrap();
    assert_eq!(cscores.len(), 2);
    assert_eq!(cscores[0], vec![3.5f32, 1.2f32]);
    assert!(cscores[1].is_empty(), "None layer should give empty vec");
}
```

- [ ] **Step 2: Run test to confirm it fails**

```
cargo test -p larql-vindex read_cscores_binary_matches_full_read 2>&1 | tail -5
```
Expected: FAIL — `read_cscores_binary` not defined.

- [ ] **Step 3: Implement `read_cscores_binary`**

Add to `crates/larql-vindex/src/format/down_meta.rs` after the `read_binary` function:

```rust
/// Read only c_scores from down_meta.bin — no tokenizer needed.
/// Returns a Vec (per layer) of Vec<f32> (per feature).
/// Layers with no features (None in the full format) yield an empty Vec.
pub fn read_cscores_binary(dir: &Path) -> Result<Vec<Vec<f32>>, VindexError> {
    let path = dir.join(DOWN_META_BIN);
    let file = std::fs::File::open(&path)?;
    let mut r = BufReader::new(file);

    let magic = read_u32(&mut r)?;
    if magic != MAGIC && magic != LEGACY_LITERAL_MAGIC {
        return Err(VindexError::Parse(format!(
            "invalid down_meta.bin magic: 0x{magic:08X}"
        )));
    }
    let version = read_u32(&mut r)?;
    if version != FORMAT_VERSION {
        return Err(VindexError::Parse(format!(
            "unsupported down_meta.bin version: {version}"
        )));
    }
    let num_layers = read_u32(&mut r)? as usize;
    let top_k_count = read_u32(&mut r)? as usize;
    // Bytes to skip per feature after reading c_score:
    // top_k_count × (u32 token_id + f32 logit) = top_k_count × 8 bytes
    let skip_per_feature = top_k_count * (U32_BYTES + F32_BYTES);

    let mut result = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let num_features = read_u32(&mut r)? as usize;
        if num_features == 0 {
            result.push(Vec::new());
            continue;
        }
        let mut cscores = Vec::with_capacity(num_features);
        for _ in 0..num_features {
            let _top_token_id = read_u32(&mut r)?;
            let c_score = read_f32(&mut r)?;
            cscores.push(c_score);
            // Skip top_k entries
            let mut skip_buf = vec![0u8; skip_per_feature];
            r.read_exact(&mut skip_buf)?;
        }
        result.push(cscores);
    }
    Ok(result)
}
```

- [ ] **Step 4: Run test to confirm it passes**

```
cargo test -p larql-vindex read_cscores_binary_matches_full_read 2>&1 | tail -5
```
Expected: `test result: ok. 1 passed`

- [ ] **Step 5: Run full larql-vindex tests**

```
cargo test -p larql-vindex 2>&1 | tail -8
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/larql-vindex/src/format/down_meta.rs
git commit -m "feat(down_meta): add read_cscores_binary (no tokenizer dependency)"
```

---

## Task 5: Covariance estimation

**Files:**
- Create: `crates/larql-vindex/src/canonical/covariance.rs`
- Modify: `crates/larql-vindex/src/canonical/mod.rs`

- [ ] **Step 1: Write the failing tests**

Create `crates/larql-vindex/src/canonical/covariance.rs`:

```rust
use ndarray::{Array2, s};

/// Estimate the activation covariance G = (1/N) Σ (s·W_E[v])^T (s·W_E[v])
/// using at most `max_samples` rows from the embedding matrix.
/// Rows are subsampled deterministically (every stride-th row).
/// Returns a [hidden_size, hidden_size] f64 matrix.
pub fn estimate_covariance(
    embed: &Array2<f32>,
    embed_scale: f32,
    max_samples: usize,
) -> Array2<f64> {
    let (vocab, d) = (embed.shape()[0], embed.shape()[1]);
    let stride = (vocab / max_samples).max(1);
    let indices: Vec<usize> = (0..vocab).step_by(stride).collect();
    let n = indices.len();

    let mut g = Array2::<f64>::zeros((d, d));
    let scale = embed_scale as f64;

    for &v in &indices {
        let row = embed.slice(s![v, ..]);
        for i in 0..d {
            let xi = row[i] as f64 * scale;
            for j in 0..=i {
                let xj = row[j] as f64 * scale;
                g[[i, j]] += xi * xj;
                if i != j {
                    g[[j, i]] += xi * xj;
                }
            }
        }
    }

    let norm = n as f64;
    g.mapv_inplace(|v| v / norm);
    g
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn identity_embed(d: usize) -> Array2<f32> {
        // Each row is a basis vector — G should be I/d * d = I (after normalization).
        // Actually G = (1/d) I·I^T = I (diagonal 1.0) when embed = I_d.
        Array2::<f32>::eye(d)
    }

    #[test]
    fn covariance_of_identity_is_identity_scaled() {
        // embed = 4×4 identity, embed_scale = 1.0, max_samples = 4
        let embed = identity_embed(4);
        let g = estimate_covariance(&embed, 1.0, 4);
        // G = (1/4) Σ_v e_v e_v^T = (1/4) I. Diagonal = 0.25, off-diag = 0.
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 0.25 } else { 0.0 };
                assert!(
                    (g[[i, j]] - expected).abs() < 1e-10,
                    "G[{i},{j}]={} expected={expected}", g[[i, j]]
                );
            }
        }
    }

    #[test]
    fn covariance_is_positive_semidefinite() {
        // A PSD matrix has non-negative diagonal and the off-diagonal satisfies
        // G[i,j]^2 <= G[i,i] * G[j,j] (Cauchy-Schwarz).
        let embed = Array2::from_shape_fn((8, 4), |(v, d)| (v as f32 + 1.0) * (d as f32 + 1.0));
        let g = estimate_covariance(&embed, 0.5, 8);
        for i in 0..4 {
            assert!(g[[i, i]] >= 0.0, "diagonal must be non-negative");
            for j in 0..4 {
                assert!(
                    g[[i, j]] * g[[i, j]] <= g[[i, i]] * g[[j, j]] + 1e-10,
                    "Cauchy-Schwarz violated at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn covariance_is_symmetric() {
        let embed = Array2::from_shape_fn((16, 4), |(v, d)| (v * d) as f32 * 0.1 + 0.01);
        let g = estimate_covariance(&embed, 1.0, 16);
        for i in 0..4 {
            for j in 0..4 {
                assert!((g[[i, j]] - g[[j, i]]).abs() < 1e-10,
                    "G not symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn embed_scale_squares_into_covariance() {
        let embed = identity_embed(4);
        let g1 = estimate_covariance(&embed, 1.0, 4);
        let g2 = estimate_covariance(&embed, 2.0, 4);
        // Scaling by s multiplies G by s^2
        for i in 0..4 {
            for j in 0..4 {
                assert!((g2[[i, j]] - 4.0 * g1[[i, j]]).abs() < 1e-10,
                    "scale^2 law violated at ({i},{j})");
            }
        }
    }

    #[test]
    fn subsampling_reduces_sample_count_not_shape() {
        let embed = Array2::from_shape_fn((100, 4), |(v, _)| v as f32);
        let g = estimate_covariance(&embed, 1.0, 10);
        assert_eq!(g.shape(), [4, 4]);
    }
}
```

- [ ] **Step 2: Add to `mod.rs`**

In `crates/larql-vindex/src/canonical/mod.rs`, add:

```rust
pub mod covariance;
pub use covariance::estimate_covariance;
```

- [ ] **Step 3: Run tests to confirm they fail first**

```
cargo test -p larql-vindex canonical::covariance 2>&1 | tail -5
```
Expected: compile error — module doesn't exist yet (you just created the file, so this should compile but tests should pass; if the impl is in the file you wrote, they should pass immediately).

- [ ] **Step 4: Run tests to confirm they pass**

```
cargo test -p larql-vindex canonical::covariance 2>&1 | tail -5
```
Expected: `test result: ok. 5 passed`

- [ ] **Step 5: Commit**

```bash
git add crates/larql-vindex/src/canonical/covariance.rs crates/larql-vindex/src/canonical/mod.rs
git commit -m "feat(canonical): add estimate_covariance from token embeddings"
```

---

## Task 6: Cholesky whitening

**Files:**
- Create: `crates/larql-vindex/src/canonical/whitening.rs`
- Modify: `crates/larql-vindex/src/canonical/mod.rs`

- [ ] **Step 1: Write the failing tests and implementation together**

Create `crates/larql-vindex/src/canonical/whitening.rs`:

```rust
use ndarray::Array2;

use larql_compute::cholesky;
// back_solve_lt and compute_l_inv_t are re-exported from larql_compute
pub use larql_compute::{back_solve_lt, compute_l_inv_t};

/// Ridge regularisation added to G diagonal before Cholesky decomposition.
/// Prevents failure on near-singular covariance (small hidden dims in tests).
const DEFAULT_RIDGE: f64 = 1e-5;

/// The Cholesky whitening data computed from a covariance matrix G.
pub struct WhiteningData {
    /// Lower triangular Cholesky factor L such that G = L L^T.
    pub l: Array2<f64>,
    /// Packed lower triangle of L for storage in canonical_meta.json.
    /// Entry (i,j) with j<=i is at index i*(i+1)/2 + j.
    pub l_packed: Vec<f64>,
}

/// Compute the Cholesky factor of the covariance matrix G.
/// Returns `WhiteningData` containing L and its packed form.
pub fn compute_whitening(g: &Array2<f64>) -> Result<WhiteningData, String> {
    let l = cholesky(g, DEFAULT_RIDGE)?;
    let d = l.shape()[0];
    let mut l_packed = Vec::with_capacity(d * (d + 1) / 2);
    for i in 0..d {
        for j in 0..=i {
            l_packed.push(l[[i, j]]);
        }
    }
    Ok(WhiteningData { l, l_packed })
}

/// Unpack a lower-triangle-packed Cholesky factor back to a dense d×d matrix.
pub fn unpack_l(packed: &[f64], d: usize) -> Array2<f64> {
    let mut l = Array2::<f64>::zeros((d, d));
    for i in 0..d {
        for j in 0..=i {
            l[[i, j]] = packed[i * (i + 1) / 2 + j];
        }
    }
    l
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn spd_3x3() -> Array2<f64> {
        // G = [[4,2,1],[2,5,3],[1,3,6]] (symmetric positive definite)
        let data = vec![4.0_f64, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0];
        Array2::from_shape_vec((3, 3), data).unwrap()
    }

    #[test]
    fn cholesky_recovers_g() {
        let g = spd_3x3();
        let wd = compute_whitening(&g).unwrap();
        let reconstructed = wd.l.dot(&wd.l.t());
        // With ridge=1e-5, G_reconstructed ≈ G + 1e-5 * I
        for i in 0..3 {
            for j in 0..3 {
                let expected = g[[i, j]] + if i == j { 1e-5 } else { 0.0 };
                assert!((reconstructed[[i, j]] - expected).abs() < 1e-8,
                    "L L^T differs from G+ridge at ({i},{j})");
            }
        }
    }

    #[test]
    fn l_is_lower_triangular() {
        let g = spd_3x3();
        let wd = compute_whitening(&g).unwrap();
        for i in 0..3 {
            for j in (i + 1)..3 {
                assert_eq!(wd.l[[i, j]], 0.0, "upper triangle not zero at ({i},{j})");
            }
        }
    }

    #[test]
    fn l_packed_length_is_triangular_number() {
        let g = spd_3x3();
        let wd = compute_whitening(&g).unwrap();
        assert_eq!(wd.l_packed.len(), 3 * 4 / 2); // 6
    }

    #[test]
    fn unpack_roundtrips_packed() {
        let g = spd_3x3();
        let wd = compute_whitening(&g).unwrap();
        let l2 = unpack_l(&wd.l_packed, 3);
        for i in 0..3 {
            for j in 0..3 {
                assert!((l2[[i, j]] - wd.l[[i, j]]).abs() < 1e-14,
                    "unpack mismatch at ({i},{j})");
            }
        }
    }

    #[test]
    fn whitening_makes_mahalanobis_a_dot_product() {
        // After whitening g̃ = L^{-T} g, the Mahalanobis score g^T G^{-1} h
        // equals the ordinary dot product g̃ · h̃.
        let g = spd_3x3();
        let wd = compute_whitening(&g).unwrap();
        let l_inv_t = compute_l_inv_t(&wd.l);

        // Two test vectors
        let g_vec = Array2::from_shape_vec((3, 1), vec![1.0f64, 2.0, 3.0]).unwrap();
        let h_vec = Array2::from_shape_vec((3, 1), vec![4.0f64, 5.0, 6.0]).unwrap();

        // Whitened vectors
        let g_tilde = l_inv_t.dot(&g_vec); // L^{-T} g
        let h_tilde = l_inv_t.dot(&h_vec); // L^{-T} h

        // Mahalanobis: g^T G^{-1} h = g^T L^{-T} L^{-1} h = g̃^T h̃
        let dot_whitened: f64 = (0..3).map(|i| g_tilde[[i, 0]] * h_tilde[[i, 0]]).sum();

        // Direct Mahalanobis using cholesky_inverse
        let l_inv = larql_compute::cholesky_inverse(&wd.l);
        let g_inv = l_inv.t().dot(&l_inv);
        let mahal: f64 = {
            let tmp = g_inv.dot(&h_vec);
            (0..3).map(|i| g_vec[[i, 0]] * tmp[[i, 0]]).sum()
        };

        assert!((dot_whitened - mahal).abs() < 1e-8,
            "whitened dot={dot_whitened} mahal={mahal}");
    }
}
```

- [ ] **Step 2: Expose `back_solve_lt` and `compute_l_inv_t` from `larql-compute`**

In `crates/larql-compute/src/lib.rs`, add to the existing public exports:

```rust
pub use cpu::ops::linalg::{back_solve_lt, cholesky, cholesky_inverse, cholesky_solve,
    compute_l_inv_t, ridge_decomposition_solve};
```

(Find the existing `pub use cpu::ops::linalg::{cholesky, ...}` line and extend it.)

- [ ] **Step 3: Add `larql-compute` dependency to `larql-vindex/Cargo.toml` if not already present**

Check:
```
grep "larql-compute" crates/larql-vindex/Cargo.toml
```
If absent, add under `[dependencies]`:
```toml
larql-compute = { path = "../larql-compute" }
```

- [ ] **Step 4: Add to `canonical/mod.rs`**

```rust
pub mod whitening;
pub use whitening::{compute_whitening, unpack_l, WhiteningData};
```

- [ ] **Step 5: Run tests**

```
cargo test -p larql-vindex canonical::whitening 2>&1 | tail -5
```
Expected: `test result: ok. 5 passed`

- [ ] **Step 6: Commit**

```bash
git add crates/larql-vindex/src/canonical/whitening.rs \
        crates/larql-vindex/src/canonical/mod.rs \
        crates/larql-compute/src/lib.rs \
        crates/larql-vindex/Cargo.toml
git commit -m "feat(canonical): add Cholesky whitening (compute_whitening, unpack_l)"
```

---

## Task 7: On-shell filter

**Files:**
- Create: `crates/larql-vindex/src/canonical/onshell.rs`
- Modify: `crates/larql-vindex/src/canonical/mod.rs`

- [ ] **Step 1: Write the failing tests and implementation**

Create `crates/larql-vindex/src/canonical/onshell.rs`:

```rust
/// Compute a boolean on-shell mask for features in one layer.
/// Features whose c_score ranks in the top `top_fraction` are on-shell.
/// `top_fraction = 0.15` reproduces the "15% factual subspace" from the synthesis.
///
/// Returns a Vec<bool> of length equal to `c_scores`.
/// Empty input returns an empty Vec.
pub fn compute_onshell_mask(c_scores: &[f32], top_fraction: f32) -> Vec<bool> {
    if c_scores.is_empty() {
        return Vec::new();
    }
    let n = c_scores.len();
    // Number of on-shell features (at least 1 if any features exist)
    let k = ((n as f32 * top_fraction).ceil() as usize).max(1).min(n);
    // Find the k-th largest c_score (threshold)
    let mut sorted: Vec<f32> = c_scores.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = sorted[k - 1];
    // Mark as on-shell all features with c_score >= threshold.
    // Ties are included (may yield slightly more than k on-shell features).
    c_scores.iter().map(|&s| s >= threshold).collect()
}

/// Count on-shell features across all layers and return (on_shell_count, total).
pub fn onshell_stats(c_scores_per_layer: &[Vec<f32>], top_fraction: f32) -> Vec<(usize, usize)> {
    c_scores_per_layer
        .iter()
        .map(|cscores| {
            let mask = compute_onshell_mask(cscores, top_fraction);
            let on_shell = mask.iter().filter(|&&b| b).count();
            (on_shell, cscores.len())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_15pct_of_10_is_2() {
        // 15% of 10 = 1.5 → ceil = 2
        let scores: Vec<f32> = (0..10).map(|i| i as f32).collect(); // 0..9
        let mask = compute_onshell_mask(&scores, 0.15);
        // Top 2 = scores 8 and 9
        let on_shell_indices: Vec<usize> =
            mask.iter().enumerate().filter(|(_, &b)| b).map(|(i, _)| i).collect();
        assert_eq!(on_shell_indices.len(), 2);
        assert!(on_shell_indices.contains(&8));
        assert!(on_shell_indices.contains(&9));
    }

    #[test]
    fn empty_scores_returns_empty_mask() {
        assert!(compute_onshell_mask(&[], 0.15).is_empty());
    }

    #[test]
    fn single_feature_always_on_shell() {
        let mask = compute_onshell_mask(&[0.5], 0.15);
        assert_eq!(mask, vec![true]);
    }

    #[test]
    fn mask_length_matches_input() {
        let scores: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let mask = compute_onshell_mask(&scores, 0.15);
        assert_eq!(mask.len(), 100);
    }

    #[test]
    fn on_shell_count_is_at_least_top_fraction() {
        let scores: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let mask = compute_onshell_mask(&scores, 0.15);
        let count = mask.iter().filter(|&&b| b).count();
        // 15% of 20 = 3 → expect exactly 3
        assert_eq!(count, 3);
    }

    #[test]
    fn all_same_scores_all_on_shell() {
        // When all c_scores are equal, all features tie at the threshold.
        let scores = vec![1.0f32; 10];
        let mask = compute_onshell_mask(&scores, 0.15);
        // All values >= threshold (which is 1.0), so all are on-shell.
        assert!(mask.iter().all(|&b| b));
    }

    #[test]
    fn onshell_stats_counts_per_layer() {
        let layers = vec![
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![],
        ];
        let stats = onshell_stats(&layers, 0.15);
        assert_eq!(stats.len(), 2);
        let (on, total) = stats[0];
        assert_eq!(total, 10);
        assert_eq!(on, 2); // top 15% of 10 = 2
        let (on2, total2) = stats[1];
        assert_eq!(total2, 0);
        assert_eq!(on2, 0);
    }
}
```

- [ ] **Step 2: Add to `canonical/mod.rs`**

```rust
pub mod onshell;
pub use onshell::{compute_onshell_mask, onshell_stats};
```

- [ ] **Step 3: Run tests**

```
cargo test -p larql-vindex canonical::onshell 2>&1 | tail -5
```
Expected: `test result: ok. 7 passed`

- [ ] **Step 4: Commit**

```bash
git add crates/larql-vindex/src/canonical/onshell.rs crates/larql-vindex/src/canonical/mod.rs
git commit -m "feat(canonical): add on-shell feature filter (top 15% by c_score)"
```

---

## Task 8: Regime classifier

**Files:**
- Create: `crates/larql-vindex/src/canonical/regime.rs`
- Modify: `crates/larql-vindex/src/canonical/mod.rs`

- [ ] **Step 1: Write the failing tests and implementation**

Create `crates/larql-vindex/src/canonical/regime.rs`:

```rust
use crate::canonical::types::Regime;

/// Activation density floor: features with c_score above this are "active".
const ACTIVE_FLOOR: f32 = 0.1;

/// Classify the regime for a single layer.
/// `c_scores`: c_score for each feature in this layer.
/// `mean_density` = fraction of features with c_score > ACTIVE_FLOOR.
///   - mean_density > 0.5  → Wave
///   - mean_density < 0.05 → Particle
///   - otherwise           → Wavelet
pub fn classify_layer_regime(c_scores: &[f32]) -> (Regime, f32) {
    if c_scores.is_empty() {
        return (Regime::Wavelet, 0.0);
    }
    let active = c_scores.iter().filter(|&&s| s > ACTIVE_FLOOR).count();
    let density = active as f32 / c_scores.len() as f32;
    let regime = if density > 0.5 {
        Regime::Wave
    } else if density < 0.05 {
        Regime::Particle
    } else {
        Regime::Wavelet
    };
    (regime, density)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_layer_is_wave() {
        // 80% of features active
        let scores: Vec<f32> = (0..10).map(|i| if i < 8 { 0.5 } else { 0.0 }).collect();
        let (regime, density) = classify_layer_regime(&scores);
        assert_eq!(regime, Regime::Wave);
        assert!((density - 0.8).abs() < 1e-5);
    }

    #[test]
    fn sparse_layer_is_particle() {
        // 2% of features active
        let mut scores = vec![0.0f32; 100];
        scores[0] = 0.5;
        scores[1] = 0.5;
        let (regime, density) = classify_layer_regime(&scores);
        assert_eq!(regime, Regime::Particle);
        assert!((density - 0.02).abs() < 1e-5);
    }

    #[test]
    fn mid_density_is_wavelet() {
        // 20% of features active
        let mut scores = vec![0.0f32; 10];
        for i in 0..2 {
            scores[i] = 0.5;
        }
        let (regime, density) = classify_layer_regime(&scores);
        assert_eq!(regime, Regime::Wavelet);
        assert!((density - 0.2).abs() < 1e-5);
    }

    #[test]
    fn empty_layer_is_wavelet_density_zero() {
        let (regime, density) = classify_layer_regime(&[]);
        assert_eq!(regime, Regime::Wavelet);
        assert_eq!(density, 0.0);
    }

    #[test]
    fn boundary_at_0_5_is_wave() {
        // Exactly 50% active (density == 0.5, > 0.5 is false, so Wavelet)
        let scores: Vec<f32> = (0..10).map(|i| if i < 5 { 0.5 } else { 0.0 }).collect();
        let (regime, density) = classify_layer_regime(&scores);
        assert_eq!(regime, Regime::Wavelet, "density=0.5 is not > 0.5, should be Wavelet");
        assert!((density - 0.5).abs() < 1e-5);
    }

    #[test]
    fn boundary_at_0_05_is_wavelet() {
        // density = 0.05, which is NOT < 0.05, so Wavelet
        let mut scores = vec![0.0f32; 20];
        scores[0] = 0.5; // 1/20 = 0.05
        let (regime, density) = classify_layer_regime(&scores);
        assert_eq!(regime, Regime::Wavelet, "density=0.05 is not < 0.05, should be Wavelet");
        assert!((density - 0.05).abs() < 1e-5);
    }
}
```

- [ ] **Step 2: Add to `canonical/mod.rs`**

```rust
pub mod regime;
pub use regime::classify_layer_regime;
```

- [ ] **Step 3: Run tests**

```
cargo test -p larql-vindex canonical::regime 2>&1 | tail -5
```
Expected: `test result: ok. 6 passed`

- [ ] **Step 4: Commit**

```bash
git add crates/larql-vindex/src/canonical/regime.rs crates/larql-vindex/src/canonical/mod.rs
git commit -m "feat(canonical): add per-layer regime classifier (Wave/Particle/Wavelet)"
```

---

## Task 9: `larql canonicalize` command

This wires all the pieces together into a usable CLI command.

**Files:**
- Create: `crates/larql-cli/src/commands/extraction/canonicalize_cmd.rs`
- Modify: `crates/larql-cli/src/commands/extraction/mod.rs`
- Modify: `crates/larql-cli/src/main.rs`

- [ ] **Step 1: Implement `canonicalize_cmd.rs`**

Create `crates/larql-cli/src/commands/extraction/canonicalize_cmd.rs`:

```rust
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_vindex::{
    canonical::{
        classify_layer_regime, compute_onshell_mask, compute_whitening, estimate_covariance,
        CanonicalMeta, LayerCanonicalInfo,
    },
    format::{
        down_meta::read_cscores_binary,
        filenames::CANONICAL_META_JSON,
        load::{load_vindex_config, load_vindex_embeddings},
    },
};

/// Number of embedding rows to subsample for covariance estimation.
const COVARIANCE_SAMPLES: usize = 4096;
/// Fraction of features per layer that are considered "on-shell" (factual subspace).
const ONSHELL_FRACTION: f32 = 0.15;

#[derive(Args)]
pub struct CanonicalizeArgs {
    /// Path to the .vindex directory to canonicalize.
    vindex: PathBuf,

    /// Override the on-shell fraction (default 0.15 = top 15% by c_score).
    #[arg(long, default_value = "0.15")]
    onshell_fraction: f32,

    /// Override the covariance sample size (default 4096 embedding rows).
    #[arg(long, default_value = "4096")]
    covariance_samples: usize,
}

pub fn run(args: CanonicalizeArgs) -> anyhow::Result<()> {
    let vindex_dir = &args.vindex;
    let onshell_fraction = args.onshell_fraction;
    let covariance_samples = args.covariance_samples;

    println!("Canonicalizing vindex at {}", vindex_dir.display());

    // ── 1. Load index.json ──────────────────────────────────────────────
    let t0 = Instant::now();
    let config = load_vindex_config(vindex_dir)?;
    println!(
        "  model: {} ({}), {} layers, hidden={}, vocab={}",
        config.model, config.family, config.num_layers, config.hidden_size, config.vocab_size
    );

    // ── 2. Load embeddings.bin ──────────────────────────────────────────
    print!("  loading embeddings.bin ... ");
    let (embed, embed_scale) = load_vindex_embeddings(vindex_dir)?;
    println!(
        "{}×{} ({:.1}ms)",
        embed.shape()[0], embed.shape()[1],
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // ── 3. Estimate covariance G ────────────────────────────────────────
    let t1 = Instant::now();
    print!("  estimating G ({covariance_samples} samples) ... ");
    let g = estimate_covariance(&embed, embed_scale, covariance_samples);
    println!("{:.1}ms", t1.elapsed().as_secs_f64() * 1000.0);

    // ── 4. Cholesky whitening ───────────────────────────────────────────
    let t2 = Instant::now();
    print!("  Cholesky decomposition ... ");
    let whitening = compute_whitening(&g).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("{:.1}ms", t2.elapsed().as_secs_f64() * 1000.0);

    // ── 5. Read c_scores from down_meta.bin ────────────────────────────
    let t3 = Instant::now();
    print!("  reading c_scores from down_meta.bin ... ");
    let cscores_per_layer = read_cscores_binary(vindex_dir)?;
    println!(
        "{} layers, {:.1}ms",
        cscores_per_layer.len(),
        t3.elapsed().as_secs_f64() * 1000.0
    );

    // ── 6. Per-layer: regime + on-shell ────────────────────────────────
    let mut layers_info: Vec<LayerCanonicalInfo> = Vec::with_capacity(config.num_layers);
    for layer in 0..config.num_layers {
        let cscores = cscores_per_layer
            .get(layer)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let (regime, mean_density) = classify_layer_regime(cscores);
        let mask = compute_onshell_mask(cscores, onshell_fraction);
        let on_shell_count = mask.iter().filter(|&&b| b).count();
        layers_info.push(LayerCanonicalInfo {
            layer,
            regime,
            on_shell_count,
            total_features: cscores.len(),
            mean_density,
        });
    }

    // ── 7. Build and write canonical_meta.json ─────────────────────────
    let meta = CanonicalMeta {
        version: 1,
        model: config.model.clone(),
        family: config.family.clone(),
        num_layers: config.num_layers,
        hidden_size: config.hidden_size,
        covariance_sample_size: covariance_samples,
        embed_scale,
        cholesky_l_packed: whitening.l_packed,
        layers: layers_info,
    };

    let out_path = vindex_dir.join(CANONICAL_META_JSON);
    let json = serde_json::to_string_pretty(&meta)?;
    std::fs::write(&out_path, &json)?;

    let total_on_shell: usize = meta.layers.iter().map(|l| l.on_shell_count).sum();
    let total_features: usize = meta.layers.iter().map(|l| l.total_features).sum();
    let pct = if total_features > 0 {
        100.0 * total_on_shell as f32 / total_features as f32
    } else {
        0.0
    };

    println!("  on-shell features: {total_on_shell}/{total_features} ({pct:.1}%)");
    println!("  wrote {}", out_path.display());
    println!("  total: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

    Ok(())
}
```

- [ ] **Step 2: Register module in `mod.rs`**

In `crates/larql-cli/src/commands/extraction/mod.rs`, add:

```rust
pub mod canonicalize_cmd;
```

- [ ] **Step 3: Add `Canonicalize` variant to `Commands` in `main.rs`**

Note: `load_vindex_config` and `load_vindex_embeddings` are already `pub` in `larql_vindex::format::load` — no visibility changes needed.

In `crates/larql-cli/src/main.rs`, inside the `Commands` enum, after the `Compile` variant, add:

```rust
    #[command(next_help_heading = "Build")]
    /// Compute canonical form metadata for a vindex (writes canonical_meta.json).
    Canonicalize(canonicalize_cmd::CanonicalizeArgs),
```

Also add `use commands::extraction::canonicalize_cmd;` to the imports if needed (the existing wildcard `use commands::extraction::*;` may not cover it — check and add explicitly if so).

In the `match` arm that dispatches commands (look for the large `match cli.command {` block), add:

```rust
    Commands::Canonicalize(args) => canonicalize_cmd::run(args)?,
```

- [ ] **Step 4: Compile check**

```
cargo build -p larql-cli 2>&1 | tail -20
```
Expected: compiles without errors.

- [ ] **Step 5: Smoke test against the SmolLM2-360M vindex**

```
cargo run -p larql-cli -- canonicalize /home/metavacua/larql-vindexes/smollm2-360m.vindex 2>&1
```
Expected output: something like:
```
Canonicalizing vindex at /home/metavacua/larql-vindexes/smollm2-360m.vindex
  model: output/smollm2-360m-src (llama), 32 layers, hidden=960, vocab=49152
  loading embeddings.bin ... 49152×960 (...)ms
  estimating G (4096 samples) ... ...ms
  Cholesky decomposition ... ...ms
  reading c_scores from down_meta.bin ... 32 layers, ...ms
  on-shell features: .../... (~15%)
  wrote /home/metavacua/larql-vindexes/smollm2-360m.vindex/canonical_meta.json
  total: ...ms
```

- [ ] **Step 6: Verify canonical_meta.json structure**

```
python3 -c "
import json
with open('/home/metavacua/larql-vindexes/smollm2-360m.vindex/canonical_meta.json') as f:
    m = json.load(f)
print('version:', m['version'])
print('hidden_size:', m['hidden_size'])
print('cholesky_l_packed length:', len(m['cholesky_l_packed']))
expected_packed = m['hidden_size'] * (m['hidden_size'] + 1) // 2
print('expected packed length:', expected_packed)
assert len(m['cholesky_l_packed']) == expected_packed
print('layers:', len(m['layers']))
for l in m['layers'][:3]:
    print(f\"  layer {l['layer']}: regime={l['regime']}, on_shell={l['on_shell_count']}/{l['total_features']}, density={l['mean_density']:.3f}\")
print('OK')
"
```
Expected: no assertion error, cholesky_l_packed length = 960*961/2 = 461280.

- [ ] **Step 7: Commit**

```bash
git add crates/larql-cli/src/commands/extraction/canonicalize_cmd.rs \
        crates/larql-cli/src/commands/extraction/mod.rs \
        crates/larql-cli/src/main.rs
git commit -m "feat(cli): add larql canonicalize command"
```

---

## Task 10: Full test suite pass

- [ ] **Step 1: Run all tests**

```
cargo test --workspace 2>&1 | tail -20
```
Expected: all tests pass, no regressions.

- [ ] **Step 2: If any tests fail, diagnose and fix**

Read the failing test output carefully. Common failure modes:
- Visibility errors on `load_config`/`load_embeddings` — fix by promoting to `pub`
- Missing `use` in `whitening.rs` — add `use larql_compute::{back_solve_lt, cholesky, cholesky_inverse, compute_l_inv_t};`
- `cholesky_l_packed` assertion on SmolLM2 — verify `960 * 961 / 2 = 461280`

- [ ] **Step 3: Final commit if any fixup patches were needed**

```bash
git add -p  # stage only the fixup changes
git commit -m "fix(canonical): resolve visibility and import issues after integration"
```

---

## Self-review checklist

**Spec coverage:**
- [x] G_l covariance computation — Task 5 (`estimate_covariance`)
- [x] Gate vector whitening — Task 6 (`compute_whitening`, `back_solve_lt`, `compute_l_inv_t`)
- [x] On-shell projection filter — Task 7 (`compute_onshell_mask` via c_score percentile)
- [x] Regime classifier — Task 8 (`classify_layer_regime`)
- [x] `larql canonicalize` command — Task 9
- [ ] **Not in this plan:** Writing whitened gate vectors to `gate_vectors_whitened.bin` — the canonical form stores the Cholesky factor; applying it to gate vectors is Plan 3 prerequisite
- [ ] **Not in this plan:** Knowledge graph integration (Plan 2)
- [ ] **Not in this plan:** Cross-model Procrustes alignment (Plan 3)

**Type consistency:**
- `back_solve_lt` defined in Task 1, used in Task 6 (`whitening.rs`) and re-exported via `larql_compute`
- `compute_l_inv_t` defined in Task 1, used in whitening test (`Task 6 Step 1`)
- `CanonicalMeta.cholesky_l_packed` field name matches in types.rs (Task 3), whitening.rs (Task 6), and canonicalize_cmd.rs (Task 9)
- `read_cscores_binary` defined in Task 4, called in Task 9
- `LayerCanonicalInfo` fields in types.rs (Task 3) match construction in canonicalize_cmd.rs (Task 9)

**No placeholders confirmed.**
