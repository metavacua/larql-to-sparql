# Vindex Entanglement / Compressibility Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `larql entanglement <vindex>` — a CLI that loads a vindex's attention weights, computes per-head the Hilbertian residual **and** the entanglement entropy of the QK coupling, and writes `entanglement_meta.json` — turning the classical-vs-quantum compressibility conjecture into measured numbers on a real model.

**Architecture:** A new `larql-cli` subcommand parallel to `larql hilbertian`. It reuses the existing attention loader and per-head helpers from `larql-vindex` (`load_attention_qk`, `head_block`, `head_coupling`, `commutator_residual`, `complex_structure_split_half`, `kv_head_for_query`) and the new `entanglement_entropy` from `larql-hilbert`. For each head it forms the coupling `C = W_Q W_Kᵀ`, then reports its residual (complex-linearity) and its entanglement entropy (Schmidt/tensor-network compressibility, in ebits). Purely additive — no existing vindex file is modified.

**Tech Stack:** Rust 2021, `ndarray` 0.16, `serde`/`serde_json` (all already in `larql-cli`), `larql-vindex` (loaders + head helpers), `larql-hilbert` (`entanglement_entropy`, new dependency).

---

## Scope note

This is **Phase 7** of the QLM roadmap (`docs/QLM-ROADMAP.md`): the *analysis* that runs the Phase-4 meter on real weights. It builds on the entanglement-entropy meter (PR #141) and the Hilbertian command (PR #137). Deferred to a later pass: the **canonical-vs-raw** and **on-shell-vs-full** comparisons (need the canonical whitening applied to the weights and the on-shell mask joined in) and the **Zipfian/HTSR** power-law fit — this plan delivers the raw per-head residual+entropy numbers first.

## Domain background (read once)

For attention head `h` (query head `h`, its GQA key head `g = h / (num_q_heads/num_kv_heads)`), the **QK coupling** is `C_h = W_Q^{(h)} (W_K^{(g)})ᵀ` (shape `[head_dim, head_dim]`). Two per-head compressibility quantities, on the *same* matrix:

- **Hilbertian residual** `‖[C_h, J]‖_F / ‖C_h‖_F ∈ [0,2]` — complex-linearity; `0` ⇒ the head is complex-compressible (the superdense factor-2). Already computed by `larql hilbertian`.
- **Entanglement entropy** `S(C_h) ∈ [0, log₂(head_dim)]` ebits — the spectral entropy of the squared singular values; `0` ⇒ rank-1 (fully tensor-network-compressible), `log₂(head_dim)` ⇒ flat (incompressible).

Reused APIs (all `pub`, verified):
- `larql_vindex::format::attn_load::load_attention_qk(dir, num_layers) -> Result<Vec<(Array2<f32>, Array2<f32>)>, VindexError>` — per-layer `(W_Q [n_q·head_dim, hidden], W_K [n_kv·head_dim, hidden])`.
- `larql_vindex::format::load::load_vindex_config(dir) -> Result<VindexConfig, VindexError>`; `config.model_config: Option<VindexModelConfig>` with `num_q_heads`, `num_kv_heads`, `head_dim`.
- `larql_vindex::canonical::{complex_structure_split_half, head_block, head_coupling, commutator_residual, kv_head_for_query}`.
- `larql_hilbert::entanglement_entropy(&Array2<f64>) -> f64`.

## File structure

- Modify: `crates/larql-cli/Cargo.toml` — add `larql-hilbert` dependency.
- Create: `crates/larql-cli/src/commands/extraction/entanglement_cmd.rs` — types, the pure `coupling_metrics` helper, and `run`.
- Modify: `crates/larql-cli/src/commands/extraction/mod.rs` — `pub mod entanglement_cmd;`.
- Modify: `crates/larql-cli/src/main.rs` — `Entanglement` variant + dispatch.

---

## Task 1: Dependency + types + the per-head metric helper (TDD)

**Files:**
- Modify: `crates/larql-cli/Cargo.toml`
- Create: `crates/larql-cli/src/commands/extraction/entanglement_cmd.rs`

- [ ] **Step 1: Add `larql-hilbert` to `crates/larql-cli/Cargo.toml`**

In the `[dependencies]` section, after the `larql-vindex = { path = "../larql-vindex" }` line, add:
```toml
larql-hilbert = { path = "../larql-hilbert" }
```

- [ ] **Step 2: Create `crates/larql-cli/src/commands/extraction/entanglement_cmd.rs` with the failing test**

Write this file (types + the pure helper + its test; the `run` function is added in Task 2):

```rust
use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use larql_hilbert::entanglement_entropy;
use larql_vindex::canonical::{
    commutator_residual, complex_structure_split_half, head_block, head_coupling, kv_head_for_query,
};
use larql_vindex::format::{attn_load::load_attention_qk, load::load_vindex_config};

/// Per-head compressibility metrics on the QK coupling `C = W_Q W_Kᵀ`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadEntanglementInfo {
    pub layer: usize,
    pub query_head: usize,
    pub kv_head: usize,
    /// Hilbertian residual ‖[C,J]‖/‖C‖ ∈ [0,2] — complex-linearity.
    pub residual: f64,
    /// Entanglement entropy of C, in ebits ∈ [0, log2(head_dim)].
    pub entropy: f64,
}

/// Root metadata written to `entanglement_meta.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntanglementMeta {
    pub version: u32,
    pub model: String,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub heads: Vec<HeadEntanglementInfo>,
}

#[derive(Args)]
pub struct EntanglementArgs {
    /// Path to the .vindex directory to analyse.
    vindex: PathBuf,
}

/// The two compressibility metrics of a coupling matrix `C` against the
/// split-half complex structure `J`: (Hilbertian residual, entanglement entropy).
fn coupling_metrics(coupling: &Array2<f64>, j: &Array2<f64>) -> (f64, f64) {
    let residual = commutator_residual(coupling, j);
    let entropy = entanglement_entropy(coupling);
    (residual, entropy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn identity_coupling_is_complex_linear_and_flat() {
        // I₄ commutes with J → residual 0; flat spectrum → log2(4) = 2 ebits.
        let c = Array2::<f64>::eye(4);
        let j = complex_structure_split_half(4);
        let (r, s) = coupling_metrics(&c, &j);
        assert!(r.abs() < 1e-9, "identity should commute with J → residual 0, got {r}");
        assert!((s - 2.0).abs() < 1e-9, "I₄ flat spectrum → 2 ebits, got {s}");
    }

    #[test]
    fn rank_one_coupling_has_zero_entanglement() {
        // Rank-1 4×4 (a 2×2 rank-1 block, rest zero) → 0 ebits.
        let c = array![
            [1.0, 2.0, 0.0, 0.0],
            [2.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ];
        let j = complex_structure_split_half(4);
        let (_r, s) = coupling_metrics(&c, &j);
        assert!(s.abs() < 1e-9, "rank-1 coupling → 0 ebits, got {s}");
    }
}
```

- [ ] **Step 3: Run the test to confirm it passes (it exercises the new dependency + reused helpers)**

```
cd /home/metavacua/larql && cargo test -p larql-cli entanglement 2>&1 | tail -10
```
Expected: `test result: ok. 2 passed`. If it fails to compile because `larql_vindex::canonical::commutator_residual` (or another helper) is not at that path, check `crates/larql-vindex/src/canonical/mod.rs` (the `pub use hilbertian::{...}` block re-exports them) and fix the import path; do NOT modify `larql-vindex`.

NOTE: the `EntanglementArgs`, `HeadEntanglementInfo`, `EntanglementMeta`, and the imports of `head_block`, `kv_head_for_query`, `head_coupling`, `load_attention_qk`, `load_vindex_config`, `Instant`, `PathBuf` are unused until Task 2 — the compiler will warn about unused imports/types. That is expected at this checkpoint; do NOT delete them (Task 2 uses them). If `cargo test` treats warnings as errors (it does not by default), add a temporary `#![allow(unused)]`-free workaround is unnecessary — warnings are fine. Proceed once the 2 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /home/metavacua/larql
git add crates/larql-cli/Cargo.toml crates/larql-cli/src/commands/extraction/entanglement_cmd.rs
git commit -m "feat(cli): entanglement command scaffold — coupling_metrics (residual + entropy)"
```

## Report
Status DONE/BLOCKED, test output last 5 lines, commit SHA.

---

## Task 2: The `run` function (load, per-head loop, write sidecar)

**Files:**
- Modify: `crates/larql-cli/src/commands/extraction/entanglement_cmd.rs`

- [ ] **Step 1: Add the `run` function**

In `crates/larql-cli/src/commands/extraction/entanglement_cmd.rs`, add (after `coupling_metrics`, before the `#[cfg(test)]` block):

```rust
pub fn run(args: EntanglementArgs) -> Result<(), Box<dyn std::error::Error>> {
    let dir = &args.vindex;
    let t0 = Instant::now();
    println!("Entanglement / compressibility analysis for vindex at {}", dir.display());

    let config = load_vindex_config(dir)?;
    let mc = config
        .model_config
        .as_ref()
        .ok_or("entanglement: index.json has no model_config (need head_dim / head counts)")?;
    let (num_q, num_kv, head_dim) = (mc.num_q_heads, mc.num_kv_heads, mc.head_dim);
    println!(
        "  model: {} ({}), {} layers, q_heads={num_q}, kv_heads={num_kv}, head_dim={head_dim}",
        config.model, config.family, config.num_layers
    );

    let j = complex_structure_split_half(head_dim);

    print!("  loading attention Q/K ... ");
    let qk = load_attention_qk(dir, config.num_layers)?;
    println!("{} layers ({:.1}ms)", qk.len(), t0.elapsed().as_secs_f64() * 1000.0);

    let mut heads: Vec<HeadEntanglementInfo> = Vec::with_capacity(config.num_layers * num_q);
    for (layer, (wq, wk)) in qk.iter().enumerate() {
        if wq.nrows() < num_q * head_dim || wk.nrows() < num_kv * head_dim {
            return Err(format!(
                "layer {layer}: attention weights too small (wq {} rows, wk {} rows; need {} and {})",
                wq.nrows(),
                wk.nrows(),
                num_q * head_dim,
                num_kv * head_dim
            )
            .into());
        }
        let wq64 = wq.mapv(|v| v as f64);
        let wk64 = wk.mapv(|v| v as f64);
        for h in 0..num_q {
            let g = kv_head_for_query(h, num_q, num_kv);
            let wq_h = head_block(&wq64, h, head_dim);
            let wk_g = head_block(&wk64, g, head_dim);
            let coupling = head_coupling(&wq_h, &wk_g);
            let (residual, entropy) = coupling_metrics(&coupling, &j);
            heads.push(HeadEntanglementInfo {
                layer,
                query_head: h,
                kv_head: g,
                residual,
                entropy,
            });
        }
    }

    let meta = EntanglementMeta {
        version: 1,
        model: config.model.clone(),
        head_dim,
        num_q_heads: num_q,
        num_kv_heads: num_kv,
        heads,
    };

    let out = dir.join("entanglement_meta.json");
    std::fs::write(&out, serde_json::to_string_pretty(&meta)?)?;

    let entropies: Vec<f64> = meta.heads.iter().map(|h| h.entropy).collect();
    let residuals: Vec<f64> = meta.heads.iter().map(|h| h.residual).collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let min = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = |v: &[f64]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "  entropy (ebits)  over {} heads: mean {:.3}, min {:.3}, max {:.3}  (max possible {:.1})",
        entropies.len(),
        mean(&entropies),
        min(&entropies),
        max(&entropies),
        (head_dim as f64).log2()
    );
    println!(
        "  residual         over {} heads: mean {:.3}, min {:.3}, max {:.3}",
        residuals.len(),
        mean(&residuals),
        min(&residuals),
        max(&residuals)
    );
    println!("  wrote {}", out.display());
    println!("  total: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}
```

- [ ] **Step 2: Build (resolves the previously-unused imports)**

```
cd /home/metavacua/larql && cargo build -p larql-cli 2>&1 | tail -15
```
Expected: compiles cleanly. If a `model_config` field name differs, verify against `crates/larql-vindex/src/config/model.rs` and adjust only this file.

- [ ] **Step 3: Run the unit tests + clippy**

```
cd /home/metavacua/larql && cargo test -p larql-cli entanglement 2>&1 | tail -5
cd /home/metavacua/larql && cargo clippy -p larql-cli 2>&1 | grep -iE "entanglement|warning: " | head || echo clean
```
Expected: 2 tests pass; clippy clean (fix any genuine warnings in this file).

- [ ] **Step 4: Commit**

```bash
cd /home/metavacua/larql
git add crates/larql-cli/src/commands/extraction/entanglement_cmd.rs
git commit -m "feat(cli): entanglement command run — per-head residual + entropy, writes entanglement_meta.json"
```

## Report
Status DONE/BLOCKED, build result, test output, commit SHA.

---

## Task 3: Wire the subcommand into the CLI

**Files:**
- Modify: `crates/larql-cli/src/commands/extraction/mod.rs`
- Modify: `crates/larql-cli/src/main.rs`

- [ ] **Step 1: Register the module**

In `crates/larql-cli/src/commands/extraction/mod.rs`, add (alphabetically near the other commands, e.g. after `pub mod embedding_jump_cmd;`):
```rust
pub mod entanglement_cmd;
```

- [ ] **Step 2: Add the `Entanglement` variant**

In `crates/larql-cli/src/main.rs`, in the "Build / extract" section of `enum Commands` (right after the `Hilbertian(hilbertian_cmd::HilbertianArgs)` variant), add:
```rust
    #[command(next_help_heading = "Build")]
    /// Per-head entanglement entropy + Hilbertian residual (compressibility); writes entanglement_meta.json.
    Entanglement(entanglement_cmd::EntanglementArgs),
```

- [ ] **Step 3: Add the dispatch arm**

In `crates/larql-cli/src/main.rs`, in the `let result = match cli.command {` block, right after the `Commands::Hilbertian(args) => hilbertian_cmd::run(args),` arm, add:
```rust
        Commands::Entanglement(args) => entanglement_cmd::run(args),
```

- [ ] **Step 4: Build + clippy**

```
cd /home/metavacua/larql && cargo build -p larql-cli 2>&1 | tail -5
cd /home/metavacua/larql && cargo clippy -p larql-cli 2>&1 | grep -iE "entanglement|warning: " | head || echo clean
```
Expected: compiles; the help shows the command (`./target/debug/larql entanglement --help` prints the usage with `<VINDEX>`).

- [ ] **Step 5: Confirm the command is registered (no model load yet)**

```
cd /home/metavacua/larql && ./target/debug/larql entanglement --help 2>&1 | head -5
```
Expected: usage text mentioning `<VINDEX>` and the description.

- [ ] **Step 6: Commit**

```bash
cd /home/metavacua/larql
git add crates/larql-cli/src/commands/extraction/mod.rs crates/larql-cli/src/main.rs
git commit -m "feat(cli): wire larql entanglement subcommand"
```

## Report
Status DONE/BLOCKED, build result, commit SHA. (The end-to-end run on the real vindex is done as a separate verification step after this plan, not as a task.)

---

## Self-review checklist

**Spec coverage:**
- [x] Per-head coupling metrics (residual + entanglement entropy) — Task 1 (`coupling_metrics`)
- [x] Result types + `entanglement_meta.json` — Task 1 (types) + Task 2 (write)
- [x] `run`: load vindex, per-head loop, write sidecar, summary — Task 2
- [x] `larql entanglement` subcommand wired — Task 3
- [ ] **Deferred (later pass):** canonical-vs-raw, on-shell-vs-full, Zipfian/HTSR fit.

**Type consistency:**
- `coupling_metrics(&Array2<f64>, &Array2<f64>) -> (f64, f64)` defined Task 1, called in `run` (Task 2).
- `HeadEntanglementInfo { layer, query_head, kv_head, residual, entropy }` and `EntanglementMeta { version, model, head_dim, num_q_heads, num_kv_heads, heads }` defined Task 1, constructed in Task 2.
- `EntanglementArgs { vindex }` defined Task 1, used in `run` (Task 2) and the `Commands::Entanglement` variant (Task 3).
- Reused `larql_vindex::canonical::{commutator_residual, complex_structure_split_half, head_block, head_coupling, kv_head_for_query}` and `larql_hilbert::entanglement_entropy` — signatures verified against the codebase.
- Runner returns `Result<(), Box<dyn std::error::Error>>` and the dispatch arm returns it directly (matches the `Hilbertian`/`Canonicalize` precedent).

**No placeholders:** every code step contains complete code; every run step has an exact command + expected output. Task 1's unused-until-Task-2 imports are called out explicitly rather than left as a surprise.
