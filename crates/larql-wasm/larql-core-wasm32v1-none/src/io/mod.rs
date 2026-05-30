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
// csv / checkpoint / packed are std::io-coupled throughout (Cursor/Write/File);
// json & format expose portable in-memory codecs (to_json_bytes etc.).
#[cfg(not(target_arch = "wasm32"))]
pub mod checkpoint;
#[cfg(not(target_arch = "wasm32"))]
pub mod csv;
pub mod format;
pub mod json;
pub mod msgpack;
#[cfg(not(target_arch = "wasm32"))]
pub mod packed;

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use crate::core::graph::{Graph, GraphError};
pub use format::Format;
/// Load a graph from disk, auto-detecting format from the file extension.
#[cfg(not(target_arch = "wasm32"))]
pub fn load(path: impl AsRef<Path>) -> Result<Graph, GraphError> {
    let path = path.as_ref();
    let fmt = Format::from_path(path).ok_or_else(|| {
        GraphError::Deserialize(format!("unrecognised file extension: {}", path.display()))
    })?;
    load_with_format(path, fmt)
}

/// Load a graph using an explicit format.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_with_format(path: impl AsRef<Path>, fmt: Format) -> Result<Graph, GraphError> {
    match fmt {
        Format::Json => json::load_json(path),
        #[cfg(feature = "msgpack")]
        Format::MessagePack => msgpack::load_msgpack(path),
        Format::Packed => packed::load_packed(path),
    }
}

/// Save a graph to disk, auto-detecting format from the file extension.
#[cfg(not(target_arch = "wasm32"))]
pub fn save(graph: &Graph, path: impl AsRef<Path>) -> Result<(), GraphError> {
    let path = path.as_ref();
    let fmt = Format::from_path(path).ok_or_else(|| {
        GraphError::Deserialize(format!("unrecognised file extension: {}", path.display()))
    })?;
    save_with_format(graph, path, fmt)
}

/// Save a graph using an explicit format.
#[cfg(not(target_arch = "wasm32"))]
pub fn save_with_format(
    graph: &Graph,
    path: impl AsRef<Path>,
    fmt: Format,
) -> Result<(), GraphError> {
    match fmt {
        Format::Json => json::save_json(graph, path),
        #[cfg(feature = "msgpack")]
        Format::MessagePack => msgpack::save_msgpack(graph, path),
        Format::Packed => packed::save_packed(graph, path),
    }
}

/// Serialize a graph to bytes in the given format.
///
/// JSON is portable (serde_json + alloc). The Packed codec uses `std::io`
/// byte cursors, so it is native-only; on wasm32 that arm returns an error.
pub fn to_bytes(graph: &Graph, fmt: Format) -> Result<Vec<u8>, GraphError> {
    match fmt {
        Format::Json => json::to_json_bytes(graph),
        #[cfg(feature = "msgpack")]
        Format::MessagePack => msgpack::to_msgpack_bytes(graph),
        #[cfg(not(target_arch = "wasm32"))]
        Format::Packed => packed::to_packed_bytes(graph),
        #[cfg(target_arch = "wasm32")]
        Format::Packed => Err(GraphError::Deserialize(
            "packed format is native-only".into(),
        )),
    }
}

/// Deserialize a graph from bytes in the given format.
pub fn from_bytes(bytes: &[u8], fmt: Format) -> Result<Graph, GraphError> {
    match fmt {
        Format::Json => json::from_json_bytes(bytes),
        #[cfg(feature = "msgpack")]
        Format::MessagePack => msgpack::from_msgpack_bytes(bytes),
        #[cfg(not(target_arch = "wasm32"))]
        Format::Packed => packed::from_packed_bytes(bytes),
        #[cfg(target_arch = "wasm32")]
        Format::Packed => Err(GraphError::Deserialize(
            "packed format is native-only".into(),
        )),
    }
}
