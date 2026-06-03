//! `MetadataStore` — owns down-meta heap/mmap state and per-feature
//! overrides (INSERT/DELETE-side mutations).
//!
//! Carved out of `VectorIndex` in the 2026-04-25 reorg.
#[allow(unused_imports)]
use alloc::{boxed::Box, string::{String, ToString}, vec::Vec, vec, format, borrow::{Cow, ToOwned}, rc::Rc, sync::Arc, collections::{BTreeMap, BTreeSet, VecDeque, BinaryHeap}};
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use hashbrown::{HashMap, HashSet};
#[cfg(not(target_arch = "wasm32"))]
#[allow(unused_imports)]
use std::collections::{HashMap, HashSet};
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use larql_wasm_math::FloatExt as _;

#[cfg(not(target_arch = "wasm32"))]
use crate::index::types::{DownMetaMmap, FeatureMeta};
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone)]
pub struct MetadataStore {
    /// Per-layer, per-feature output token metadata (heap mode).
    pub down_meta: Vec<Option<Vec<Option<FeatureMeta>>>>,
    /// Mmap'd down_meta.bin (zero-copy mode).
    pub down_meta_mmap: Option<Arc<DownMetaMmap>>,
    /// Down vector overrides — `(layer, feature) → hidden_size f32`.
    pub down_overrides: HashMap<(usize, usize), Vec<f32>>,
    /// Up vector overrides — same shape; written by INSERT.
    pub up_overrides: HashMap<(usize, usize), Vec<f32>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl MetadataStore {
    pub fn empty(num_layers: usize) -> Self {
        Self {
            down_meta: vec![None; num_layers],
            down_meta_mmap: None,
            down_overrides: HashMap::new(),
            up_overrides: HashMap::new(),
        }
    }
}
