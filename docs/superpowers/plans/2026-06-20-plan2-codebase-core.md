# larql-codebase-core: Weight Derivation Math Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill in `larql-codebase-core` with the pure Tier-0 math that converts a `larql_core::Graph` (unit-asserted knowledge graph) into weight matrices suitable for BitNet (I2_S) export — adjacency normalisation, god-node detection, superposition encoding, architecture sizing, and a `BasisTransform` trait.

**Architecture:** `larql-codebase-core` is wasm32v1-none safe (no filesystem, no network). All functions take `&Graph` or owned data structs and return computed results. The single external dependency on `larql-core` is already in `Cargo.toml` (added in Plan 1). A `BasisTransform` trait with a `BitNetBasis` implementation is the entry point; other basis types (`F16Basis`) are stubs for future plans.

**Tech Stack:** Rust stable, `larql-core::Graph`, no external math crates (all linear algebra is hand-rolled to keep wasm32v1-none safe), `serde` for serialising `WeightRepr`.

## Global Constraints

- Must compile for `wasm32v1-none` — no `std::fs`, `std::net`, `std::thread`.
- No floating-point non-determinism: use `f64` throughout; results must match across platforms.
- All public types must implement `serde::Serialize + serde::Deserialize`.
- Run `cargo test -p larql-codebase-core` before every commit.
- `cargo build --target wasm32v1-none -p larql-codebase-core` must pass before every commit.

---

### Task 1: Node index and sparse adjacency representation

**Files:**
- Create: `crates/larql-codebase-core/src/adj.rs`
- Modify: `crates/larql-codebase-core/src/lib.rs`

**Interfaces:**
- Consumes: `larql_core::core::graph::Graph` (has `.nodes()`, `.edges()`, `.node_names()`)
- Produces:
  - `pub struct NodeIndex { pub names: Vec<String>, pub index: HashMap<String, usize> }`
  - `pub struct SparseAdj { pub n: usize, pub entries: Vec<(usize, usize, f64)> }`
  - `pub fn build_node_index(graph: &Graph) -> NodeIndex`
  - `pub fn build_adjacency(graph: &Graph, idx: &NodeIndex) -> SparseAdj`

- [ ] **Step 1: Write the failing test**

In `crates/larql-codebase-core/src/adj.rs` (new file):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use larql_core::core::{edge::Edge, graph::Graph};

    fn triangle() -> Graph {
        let mut g = Graph::new();
        g.add_edge(Edge::new("A", "calls", "B"));
        g.add_edge(Edge::new("B", "calls", "C"));
        g.add_edge(Edge::new("C", "calls", "A"));
        g
    }

    #[test]
    fn node_index_covers_all_nodes() {
        let g = triangle();
        let idx = build_node_index(&g);
        assert_eq!(idx.names.len(), 3);
        assert!(idx.index.contains_key("A"));
        assert!(idx.index.contains_key("B"));
        assert!(idx.index.contains_key("C"));
    }

    #[test]
    fn adjacency_has_correct_entry_count() {
        let g = triangle();
        let idx = build_node_index(&g);
        let adj = build_adjacency(&g, &idx);
        assert_eq!(adj.n, 3);
        assert_eq!(adj.entries.len(), 3);
    }
}
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cargo test -p larql-codebase-core node_index_covers_all_nodes
```

Expected: `error[E0425]: cannot find function 'build_node_index'`

- [ ] **Step 3: Implement `NodeIndex` and `SparseAdj`**

Write `crates/larql-codebase-core/src/adj.rs`:

```rust
use std::collections::HashMap;
use larql_core::core::graph::Graph;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeIndex {
    pub names: Vec<String>,
    pub index: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseAdj {
    pub n: usize,
    pub entries: Vec<(usize, usize, f64)>,
}

pub fn build_node_index(graph: &Graph) -> NodeIndex {
    let mut names: Vec<String> = graph.node_names().into_iter().collect();
    names.sort(); // deterministic ordering
    let index: HashMap<String, usize> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();
    NodeIndex { names, index }
}

pub fn build_adjacency(graph: &Graph, idx: &NodeIndex) -> SparseAdj {
    let mut entries: Vec<(usize, usize, f64)> = Vec::new();
    for edge in graph.edges() {
        if let (Some(&i), Some(&j)) = (
            idx.index.get(&edge.subject),
            idx.index.get(&edge.object),
        ) {
            entries.push((i, j, edge.confidence));
        }
    }
    SparseAdj { n: idx.names.len(), entries }
}
```

- [ ] **Step 4: Expose from `lib.rs`**

```rust
pub mod adj;
pub use adj::{build_adjacency, build_node_index, NodeIndex, SparseAdj};
```

- [ ] **Step 5: Run tests**

```bash
cargo test -p larql-codebase-core
```

Expected: `node_index_covers_all_nodes ... ok` + `adjacency_has_correct_entry_count ... ok`

- [ ] **Step 6: wasm32v1-none build**

```bash
cargo build --target wasm32v1-none -p larql-codebase-core
```

Expected: `Finished`.

- [ ] **Step 7: Commit**

```bash
git add crates/larql-codebase-core/src/adj.rs crates/larql-codebase-core/src/lib.rs
git commit -m "feat(codebase-core): NodeIndex + SparseAdj from larql_core::Graph"
```

---

### Task 2: Symmetric normalisation D^{-½}AD^{-½}

**Files:**
- Create: `crates/larql-codebase-core/src/norm.rs`
- Modify: `crates/larql-codebase-core/src/lib.rs`

**Interfaces:**
- Consumes: `SparseAdj` from Task 1
- Produces:
  - `pub struct NormAdj { pub n: usize, pub entries: Vec<(usize, usize, f64)> }`
  - `pub fn symmetric_normalise(adj: &SparseAdj) -> NormAdj`
  - Entry value = `1.0 / (deg[i].sqrt() * deg[j].sqrt())` for each (i,j) in adj.

- [ ] **Step 1: Write the failing test**

In `crates/larql-codebase-core/src/norm.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::adj::SparseAdj;

    fn star_adj() -> SparseAdj {
        // Hub node 0 connects to 3 leaves: 1, 2, 3. Degrees: hub=3, leaves=1.
        SparseAdj {
            n: 4,
            entries: vec![(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)],
        }
    }

    #[test]
    fn normalised_hub_leaf_edge_value() {
        let adj = star_adj();
        let norm = symmetric_normalise(&adj);
        // deg[0]=3, deg[1]=1 → weight = 1/sqrt(3*1) ≈ 0.5774
        let (_, _, w) = norm.entries[0];
        let expected = 1.0_f64 / (3.0_f64 * 1.0_f64).sqrt();
        assert!((w - expected).abs() < 1e-10, "got {w}, expected {expected}");
    }

    #[test]
    fn normalised_entry_count_matches_input() {
        let adj = star_adj();
        let norm = symmetric_normalise(&adj);
        assert_eq!(norm.entries.len(), adj.entries.len());
    }
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
cargo test -p larql-codebase-core normalised_hub_leaf_edge_value
```

Expected: `error[E0425]: cannot find function 'symmetric_normalise'`

- [ ] **Step 3: Implement `symmetric_normalise`**

```rust
use crate::adj::SparseAdj;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormAdj {
    pub n: usize,
    pub entries: Vec<(usize, usize, f64)>,
}

pub fn symmetric_normalise(adj: &SparseAdj) -> NormAdj {
    let mut deg = vec![0.0_f64; adj.n];
    for &(i, j, w) in &adj.entries {
        deg[i] += w;
        deg[j] += w; // treat as undirected for normalisation
    }
    let inv_sqrt: Vec<f64> = deg
        .iter()
        .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();
    let entries = adj
        .entries
        .iter()
        .map(|&(i, j, _)| (i, j, inv_sqrt[i] * inv_sqrt[j]))
        .collect();
    NormAdj { n: adj.n, entries }
}
```

- [ ] **Step 4: Expose from `lib.rs`**

```rust
pub mod norm;
pub use norm::{symmetric_normalise, NormAdj};
```

- [ ] **Step 5: Run tests + wasm build**

```bash
cargo test -p larql-codebase-core
cargo build --target wasm32v1-none -p larql-codebase-core
```

Expected: all tests pass, wasm build `Finished`.

- [ ] **Step 6: Commit**

```bash
git add crates/larql-codebase-core/src/norm.rs crates/larql-codebase-core/src/lib.rs
git commit -m "feat(codebase-core): symmetric normalisation D^(-1/2) A D^(-1/2)"
```

---

### Task 3: God-node detection

**Files:**
- Create: `crates/larql-codebase-core/src/god_node.rs`
- Modify: `crates/larql-codebase-core/src/lib.rs`

**Interfaces:**
- Consumes: `NodeIndex` (Task 1), degree information derived from `SparseAdj`
- Produces:
  - `pub struct DegreeStats { pub mean: f64, pub std: f64, pub threshold: f64 }`
  - `pub fn degree_stats(adj: &SparseAdj) -> DegreeStats`
  - `pub fn god_nodes(adj: &SparseAdj, idx: &NodeIndex, sigma: f64) -> Vec<String>` — returns names of nodes whose total degree exceeds `mean + sigma * std`.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::adj::{NodeIndex, SparseAdj};
    use std::collections::HashMap;

    fn hub_graph() -> (SparseAdj, NodeIndex) {
        // Hub "H" connects to 4 leaves. Leaves have degree 1, hub degree 4.
        let names = vec!["H".into(), "L1".into(), "L2".into(), "L3".into(), "L4".into()];
        let index: HashMap<String, usize> =
            names.iter().enumerate().map(|(i, n)| (n.clone(), i)).collect();
        let idx = NodeIndex { names, index };
        let adj = SparseAdj {
            n: 5,
            entries: vec![(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0)],
        };
        (adj, idx)
    }

    #[test]
    fn hub_is_a_god_node() {
        let (adj, idx) = hub_graph();
        let gods = god_nodes(&adj, &idx, 1.0);
        assert!(gods.contains(&"H".to_string()), "Hub must be a god node");
    }

    #[test]
    fn leaves_are_not_god_nodes() {
        let (adj, idx) = hub_graph();
        let gods = god_nodes(&adj, &idx, 1.0);
        for leaf in ["L1", "L2", "L3", "L4"] {
            assert!(!gods.contains(&leaf.to_string()), "{leaf} should not be a god node");
        }
    }
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
cargo test -p larql-codebase-core hub_is_a_god_node
```

Expected: `error[E0425]: cannot find function 'god_nodes'`

- [ ] **Step 3: Implement**

```rust
use crate::adj::{NodeIndex, SparseAdj};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegreeStats {
    pub mean: f64,
    pub std: f64,
    pub threshold: f64,
}

pub fn degree_stats(adj: &SparseAdj) -> DegreeStats {
    let mut deg = vec![0.0_f64; adj.n];
    for &(i, j, _) in &adj.entries {
        deg[i] += 1.0;
        deg[j] += 1.0;
    }
    let n = deg.len() as f64;
    let mean = deg.iter().sum::<f64>() / n;
    let variance = deg.iter().map(|&d| (d - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    DegreeStats { mean, std, threshold: mean + 3.0 * std }
}

pub fn god_nodes(adj: &SparseAdj, idx: &NodeIndex, sigma: f64) -> Vec<String> {
    let mut deg = vec![0.0_f64; adj.n];
    for &(i, j, _) in &adj.entries {
        deg[i] += 1.0;
        deg[j] += 1.0;
    }
    let stats = degree_stats(adj);
    let threshold = stats.mean + sigma * stats.std;
    idx.names
        .iter()
        .enumerate()
        .filter(|&(i, _)| deg[i] > threshold)
        .map(|(_, name)| name.clone())
        .collect()
}
```

- [ ] **Step 4: Expose from `lib.rs`**

```rust
pub mod god_node;
pub use god_node::{degree_stats, god_nodes, DegreeStats};
```

- [ ] **Step 5: Run tests + wasm build**

```bash
cargo test -p larql-codebase-core
cargo build --target wasm32v1-none -p larql-codebase-core
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add crates/larql-codebase-core/src/god_node.rs crates/larql-codebase-core/src/lib.rs
git commit -m "feat(codebase-core): god-node detection (degree > μ + σ·std)"
```

---

### Task 4: Trit quantisation and I2_S block encoding

**Files:**
- Create: `crates/larql-codebase-core/src/trit.rs`
- Modify: `crates/larql-codebase-core/src/lib.rs`

**Interfaces:**
- Consumes: `NormAdj` (Task 2), god-node list (Task 3)
- Produces:
  - `pub fn quantise_to_trits(values: &[f64], scale: f64) -> Vec<i8>` — clamps each value to {-1, 0, +1} via `sign(v / scale)`.
  - `pub fn pack_i2s_block(trits: &[i8; 128]) -> [u8; 32]` — encodes one 128-element block in Microsoft I2_S strided layout ({-1→0, 0→1, +1→2} mapped to {0x00, 0x55, 0xAA} patterns per byte).

Note on I2_S layout: a block of 128 trits is stored in 32 bytes. Byte `p` packs elements at positions `{p, p+32, p+64, p+96}` in the 128-element block using 2 bits each: `(trit+1) & 0x3` → bits. Element at offset `k` goes into bits `(k/32)*2` of byte `k%32`.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantise_positive() {
        let values = [0.8_f64, 0.2, -0.9, 0.0];
        let trits = quantise_to_trits(&values, 0.5);
        assert_eq!(trits, vec![1i8, 0, -1, 0]);
    }

    #[test]
    fn i2s_round_trips_all_plus_one() {
        let block = [1i8; 128];
        let packed = pack_i2s_block(&block);
        let unpacked = unpack_i2s_block(&packed);
        assert_eq!(unpacked, block);
    }

    #[test]
    fn i2s_round_trips_all_minus_one() {
        let block = [-1i8; 128];
        let packed = pack_i2s_block(&block);
        let unpacked = unpack_i2s_block(&packed);
        assert_eq!(unpacked, block);
    }

    #[test]
    fn i2s_round_trips_all_zero() {
        let block = [0i8; 128];
        let packed = pack_i2s_block(&block);
        let unpacked = unpack_i2s_block(&packed);
        assert_eq!(unpacked, block);
    }
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
cargo test -p larql-codebase-core quantise_positive
```

Expected: `error[E0425]: cannot find function 'quantise_to_trits'`

- [ ] **Step 3: Implement trit encoding and I2_S block packing**

```rust
/// Map a continuous value to {-1, 0, +1} trit by comparing to scale.
/// Values in (-scale/2, +scale/2) → 0; above → +1; below → -1.
pub fn quantise_to_trits(values: &[f64], scale: f64) -> Vec<i8> {
    let half = scale / 2.0;
    values
        .iter()
        .map(|&v| {
            if v > half { 1 }
            else if v < -half { -1 }
            else { 0 }
        })
        .collect()
}

/// Encode 128 trits as a 32-byte I2_S block (Microsoft strided layout).
/// Trit code: -1 → 0b00, 0 → 0b01, +1 → 0b10
/// Byte p packs elements {p, p+32, p+64, p+96}, two bits each in low→high order.
pub fn pack_i2s_block(trits: &[i8; 128]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for p in 0..32usize {
        let mut byte = 0u8;
        for k in 0..4usize {
            let elem = trits[p + k * 32];
            let code = (elem + 1) as u8 & 0x3; // -1→0, 0→1, +1→2
            byte |= code << (k * 2);
        }
        out[p] = byte;
    }
    out
}

/// Decode a 32-byte I2_S block back to 128 trits.
pub fn unpack_i2s_block(bytes: &[u8; 32]) -> [i8; 128] {
    let mut out = [0i8; 128];
    for p in 0..32usize {
        let byte = bytes[p];
        for k in 0..4usize {
            let code = (byte >> (k * 2)) & 0x3;
            out[p + k * 32] = code as i8 - 1;
        }
    }
    out
}
```

- [ ] **Step 4: Expose from `lib.rs`**

```rust
pub mod trit;
pub use trit::{pack_i2s_block, quantise_to_trits, unpack_i2s_block};
```

- [ ] **Step 5: Run tests + wasm build**

```bash
cargo test -p larql-codebase-core
cargo build --target wasm32v1-none -p larql-codebase-core
```

Expected: all 4 new tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/larql-codebase-core/src/trit.rs crates/larql-codebase-core/src/lib.rs
git commit -m "feat(codebase-core): I2_S trit encoding — quantise_to_trits + pack/unpack_i2s_block"
```

---

### Task 5: Architecture sizing from graph statistics

**Files:**
- Create: `crates/larql-codebase-core/src/arch.rs`
- Modify: `crates/larql-codebase-core/src/lib.rs`

**Interfaces:**
- Consumes: `DegreeStats` (Task 3), node count, edge count
- Produces:
  - `pub struct ArchConfig { pub hidden_size: usize, pub n_layers: usize, pub n_heads: usize, pub head_dim: usize, pub ffn_dim: usize }`
  - `pub fn size_architecture(n_nodes: usize, n_edges: usize, god_node_count: usize) -> ArchConfig`

Sizing rules (from spec §4.2):
- `hidden_size` = next power of 2 ≥ sqrt(n_nodes), minimum 64
- `n_layers` = max(2, log2(n_edges / n_nodes).ceil() as usize)
- `n_heads` = max(4, hidden_size / 64)
- `head_dim` = 64 (BitNet standard)
- `ffn_dim` = hidden_size * 4

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_size_is_power_of_two() {
        let cfg = size_architecture(1000, 5000, 10);
        assert!(cfg.hidden_size.is_power_of_two());
        assert!(cfg.hidden_size >= 64);
    }

    #[test]
    fn head_dim_always_64() {
        let cfg = size_architecture(500, 2000, 5);
        assert_eq!(cfg.head_dim, 64);
    }

    #[test]
    fn ffn_is_4x_hidden() {
        let cfg = size_architecture(500, 2000, 5);
        assert_eq!(cfg.ffn_dim, cfg.hidden_size * 4);
    }
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
cargo test -p larql-codebase-core hidden_size_is_power_of_two
```

Expected: `error[E0425]: cannot find function 'size_architecture'`

- [ ] **Step 3: Implement `size_architecture`**

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchConfig {
    pub hidden_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
}

fn next_power_of_two_min(n: usize, min: usize) -> usize {
    let mut p = min;
    while p < n {
        p *= 2;
    }
    p
}

pub fn size_architecture(n_nodes: usize, n_edges: usize, _god_node_count: usize) -> ArchConfig {
    let sqrt_n = (n_nodes as f64).sqrt().ceil() as usize;
    let hidden_size = next_power_of_two_min(sqrt_n, 64);
    let avg_degree = if n_nodes > 0 { n_edges / n_nodes } else { 1 };
    let n_layers = (avg_degree.max(2) as f64).log2().ceil() as usize;
    let n_layers = n_layers.max(2);
    let n_heads = (hidden_size / 64).max(4);
    let head_dim = 64;
    let ffn_dim = hidden_size * 4;
    ArchConfig { hidden_size, n_layers, n_heads, head_dim, ffn_dim }
}
```

- [ ] **Step 4: Expose from `lib.rs`**

```rust
pub mod arch;
pub use arch::{size_architecture, ArchConfig};
```

- [ ] **Step 5: Run tests + wasm build**

```bash
cargo test -p larql-codebase-core
cargo build --target wasm32v1-none -p larql-codebase-core
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/larql-codebase-core/src/arch.rs crates/larql-codebase-core/src/lib.rs
git commit -m "feat(codebase-core): graph-driven architecture sizing (hidden_size, n_layers, n_heads)"
```

---

### Task 6: BasisTransform trait + BitNetBasis implementation

**Files:**
- Create: `crates/larql-codebase-core/src/basis.rs`
- Modify: `crates/larql-codebase-core/src/lib.rs`

**Interfaces:**
- Consumes: `NormAdj` (Task 2), `ArchConfig` (Task 5), `pack_i2s_block` (Task 4)
- Produces:
  - `pub trait BasisTransform { fn name(&self) -> &'static str; fn transform(&self, adj: &NormAdj, arch: &ArchConfig) -> WeightRepr; }`
  - `pub struct WeightRepr { pub tensors: Vec<NamedTensor>, pub arch: ArchConfig }`
  - `pub struct NamedTensor { pub name: String, pub dims: Vec<u64>, pub ggml_type: u32, pub data: Vec<u8> }`
  - `pub struct BitNetBasis;` implementing `BasisTransform`

`ggml_type` for I2_S = 36 (from `larql-models/src/loading/gguf/loader.rs` TYPE_I2_S).

`BitNetBasis::transform` slices the normalised adjacency into `hidden_size × hidden_size` blocks (zero-padded), quantises to trits, packs with `pack_i2s_block`, and names each tensor `"blk.{layer}.attn_qkv.weight"`.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        adj::{build_adjacency, build_node_index, SparseAdj},
        arch::size_architecture,
        norm::symmetric_normalise,
    };
    use larql_core::core::{edge::Edge, graph::Graph};

    fn small_graph() -> Graph {
        let mut g = Graph::new();
        for i in 0..10 {
            g.add_edge(Edge::new(
                format!("node_{i}"),
                "refs",
                format!("node_{}", (i + 1) % 10),
            ));
        }
        g
    }

    #[test]
    fn bitnet_basis_produces_named_tensors() {
        let g = small_graph();
        let idx = build_node_index(&g);
        let adj = build_adjacency(&g, &idx);
        let norm = symmetric_normalise(&adj);
        let arch = size_architecture(g.node_count(), g.edge_count(), 0);
        let basis = BitNetBasis;
        let repr = basis.transform(&norm, &arch);
        assert!(!repr.tensors.is_empty());
        assert!(repr.tensors[0].name.starts_with("blk."));
    }

    #[test]
    fn tensor_ggml_type_is_i2s() {
        let g = small_graph();
        let idx = build_node_index(&g);
        let adj = build_adjacency(&g, &idx);
        let norm = symmetric_normalise(&adj);
        let arch = size_architecture(g.node_count(), g.edge_count(), 0);
        let repr = BitNetBasis.transform(&norm, &arch);
        for t in &repr.tensors {
            assert_eq!(t.ggml_type, 36u32, "I2_S = 36, got {}", t.ggml_type);
        }
    }
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
cargo test -p larql-codebase-core bitnet_basis_produces_named_tensors
```

Expected: `error[E0412]: cannot find type 'BitNetBasis'`

- [ ] **Step 3: Implement**

```rust
use crate::{arch::ArchConfig, norm::NormAdj, trit::{pack_i2s_block, quantise_to_trits}};
use serde::{Deserialize, Serialize};

const GGML_TYPE_I2_S: u32 = 36;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightRepr {
    pub tensors: Vec<NamedTensor>,
    pub arch: ArchConfig,
}

pub trait BasisTransform {
    fn name(&self) -> &'static str;
    fn transform(&self, adj: &NormAdj, arch: &ArchConfig) -> WeightRepr;
}

pub struct BitNetBasis;

impl BasisTransform for BitNetBasis {
    fn name(&self) -> &'static str { "bitnet_i2s" }

    fn transform(&self, adj: &NormAdj, arch: &ArchConfig) -> WeightRepr {
        let h = arch.hidden_size;
        // Build a dense h×h matrix per layer by cycling through adj entries.
        let mut tensors = Vec::new();
        for layer in 0..arch.n_layers {
            // Fill a flat h*h buffer with adjacency values (row-major, zero-padded).
            let mut dense = vec![0.0_f64; h * h];
            for &(i, j, w) in adj.entries.iter() {
                let row = i % h;
                let col = j % h;
                // Accumulate (multiple edges may map to same cell across layers)
                if (i / h) % arch.n_layers == layer {
                    dense[row * h + col] += w;
                }
            }
            // Find scale as max absolute value (avoid div by zero)
            let scale = dense.iter().cloned().fold(0.0_f64, f64::max).max(1e-6);
            // Quantise entire matrix to trits
            let trits = quantise_to_trits(&dense, scale);
            // Pack in 128-element I2_S blocks (h*h must be divisible by 128 — zero-pad)
            let total = trits.len();
            let n_blocks = (total + 127) / 128;
            let mut data = Vec::with_capacity(n_blocks * 32);
            for b in 0..n_blocks {
                let mut block = [0i8; 128];
                for k in 0..128 {
                    let idx = b * 128 + k;
                    block[k] = if idx < total { trits[idx] } else { 0 };
                }
                data.extend_from_slice(&pack_i2s_block(&block));
            }
            tensors.push(NamedTensor {
                name: format!("blk.{layer}.attn_qkv.weight"),
                dims: vec![h as u64, h as u64],
                ggml_type: GGML_TYPE_I2_S,
                data,
            });
        }
        WeightRepr { tensors, arch: arch.clone() }
    }
}
```

- [ ] **Step 4: Expose from `lib.rs`**

```rust
pub mod basis;
pub use basis::{BasisTransform, BitNetBasis, NamedTensor, WeightRepr};
```

- [ ] **Step 5: Run tests + wasm build**

```bash
cargo test -p larql-codebase-core
cargo build --target wasm32v1-none -p larql-codebase-core
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/larql-codebase-core/src/basis.rs crates/larql-codebase-core/src/lib.rs
git commit -m "feat(codebase-core): BasisTransform trait + BitNetBasis I2_S weight synthesis"
```

---

### Task 7: End-to-end `graph_to_weight_repr` helper + integration test

**Files:**
- Modify: `crates/larql-codebase-core/src/lib.rs`

**Interfaces:**
- Consumes: all types from Tasks 1–6
- Produces: `pub fn graph_to_weight_repr(graph: &Graph, basis: &dyn BasisTransform) -> WeightRepr` — the single public entrypoint used by Plan 3's `extract-codebase` command.

- [ ] **Step 1: Write the failing integration test**

In `crates/larql-codebase-core/src/lib.rs` (new test module):

```rust
#[cfg(test)]
mod integration {
    use larql_core::core::{edge::Edge, graph::Graph};
    use crate::{basis::BitNetBasis, graph_to_weight_repr};

    #[test]
    fn end_to_end_graph_to_weight_repr() {
        let mut g = Graph::new();
        for i in 0..50 {
            g.add_edge(Edge::new(
                format!("mod_{}", i % 10),
                "calls",
                format!("fn_{}", i % 20),
            ));
        }
        let repr = graph_to_weight_repr(&g, &BitNetBasis);
        assert!(!repr.tensors.is_empty());
        // Every tensor must have I2_S data (32 bytes per 128-trit block)
        for t in &repr.tensors {
            assert!(t.data.len() % 32 == 0, "I2_S data must be 32-byte aligned");
        }
    }
}
```

- [ ] **Step 2: Run to confirm failure**

```bash
cargo test -p larql-codebase-core end_to_end_graph_to_weight_repr
```

Expected: `error[E0425]: cannot find function 'graph_to_weight_repr'`

- [ ] **Step 3: Implement in `lib.rs`**

```rust
use larql_core::core::graph::Graph;
use crate::{
    adj::{build_adjacency, build_node_index},
    arch::size_architecture,
    basis::{BasisTransform, WeightRepr},
    god_node::god_nodes,
    norm::symmetric_normalise,
};

pub fn graph_to_weight_repr(graph: &Graph, basis: &dyn BasisTransform) -> WeightRepr {
    let idx = build_node_index(graph);
    let adj = build_adjacency(graph, &idx);
    let norm = symmetric_normalise(&adj);
    let god = god_nodes(&adj, &idx, 3.0);
    let arch = size_architecture(graph.node_count(), graph.edge_count(), god.len());
    basis.transform(&norm, &arch)
}
```

- [ ] **Step 4: Run all tests**

```bash
cargo test -p larql-codebase-core
```

Expected: all pass, including the new `end_to_end_graph_to_weight_repr` test.

- [ ] **Step 5: Final wasm32v1-none build**

```bash
cargo build --target wasm32v1-none -p larql-codebase-core
```

Expected: `Finished`.

- [ ] **Step 6: Commit**

```bash
git add crates/larql-codebase-core/src/lib.rs
git commit -m "feat(codebase-core): graph_to_weight_repr end-to-end entrypoint"
```
