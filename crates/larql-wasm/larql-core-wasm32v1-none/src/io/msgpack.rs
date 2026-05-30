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
#[cfg(feature = "msgpack")]
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

#[cfg(feature = "msgpack")]
use crate::core::graph::{Graph, GraphError};
/// Serialize a graph to MessagePack bytes.
#[cfg(feature = "msgpack")]
pub fn to_msgpack_bytes(graph: &Graph) -> Result<Vec<u8>, GraphError> {
    let value = graph.to_json_value();
    rmp_serde::to_vec(&value).map_err(|e| GraphError::Deserialize(e.to_string()))
}

/// Deserialize a graph from MessagePack bytes.
#[cfg(feature = "msgpack")]
pub fn from_msgpack_bytes(bytes: &[u8]) -> Result<Graph, GraphError> {
    let value: serde_json::Value =
        rmp_serde::from_slice(bytes).map_err(|e| GraphError::Deserialize(e.to_string()))?;
    Graph::from_json_value(&value)
}

#[cfg(not(target_arch = "wasm32"))]
#[cfg(not(target_arch = "wasm32"))]
/// Load a graph from a MessagePack file.
#[cfg(feature = "msgpack")]
pub fn load_msgpack(path: impl AsRef<Path>) -> Result<Graph, GraphError> {
    let bytes = std::fs::read(path)?;
    from_msgpack_bytes(&bytes)
}

#[cfg(not(target_arch = "wasm32"))]
#[cfg(not(target_arch = "wasm32"))]
/// Save a graph to a MessagePack file.
#[cfg(feature = "msgpack")]
pub fn save_msgpack(graph: &Graph, path: impl AsRef<Path>) -> Result<(), GraphError> {
    let bytes = to_msgpack_bytes(graph)?;
    std::fs::write(path, bytes)?;
    Ok(())
}
