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
/// An entity node — always derived from edges, never stored directly.
/// Node type is an optional free-form string, inferred from schema rules at runtime.
#[derive(Debug, Clone)]
pub struct Node {
    pub name: String,
    pub node_type: Option<String>,
    pub degree: usize,
    pub out_degree: usize,
    pub in_degree: usize,
}
