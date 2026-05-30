use larql_inference::ModelWeights;
use larql_vindex::VectorIndex;
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
pub(super) fn insert_q4k_layer_tensors(
    weights: &mut ModelWeights,
    index: &VectorIndex,
    layer: usize,
) -> Result<Vec<String>, Box<dyn core::error::Error>> {
    larql_inference::vindex::insert_q4k_layer_tensors(weights, index, layer).map_err(|err| {
        Box::<dyn core::error::Error>::from(std::io::Error::new(std::io::ErrorKind::Other, err))
    })
}

pub(super) fn remove_layer_tensors(weights: &mut ModelWeights, keys: Vec<String>) {
    larql_inference::vindex::remove_layer_tensors(weights, keys);
}
