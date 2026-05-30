use serde::{Deserialize, Serialize};
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
#[derive(Debug, Clone, Deserialize)]
pub(super) struct PromptRecord {
    pub(super) id: Option<String>,
    pub(super) stratum: Option<String>,
    pub(super) prompt: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(super) struct HeadId {
    pub(super) layer: usize,
    pub(super) head: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub(super) struct PqConfig {
    pub(super) k: usize,
    pub(super) groups: usize,
    pub(super) bits_per_group: usize,
}
