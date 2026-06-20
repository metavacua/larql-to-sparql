pub mod checkpoint;
pub mod csv;
pub mod format;
pub mod json;
pub mod msgpack;
pub mod packed;

use std::path::Path;

use crate::core::graph::{Graph, GraphError};
pub use format::Format;

/// Load a graph from a JSON file.
///
/// This is a convenience function that delegates to `json::load_json`.
/// Unlike `load()`, it does not require the file to have a recognized extension.
pub fn load_graph(path: &Path) -> Result<Graph, GraphError> {
    json::load_json(path)
}

/// Save a graph to a JSON file.
///
/// This is a convenience function that delegates to `json::save_json`.
/// Unlike `save()`, it does not require the file to have a recognized extension.
pub fn save_graph(graph: &Graph, path: &Path) -> Result<(), GraphError> {
    json::save_json(graph, path)
}

/// Load a graph from disk, auto-detecting format from the file extension.
pub fn load(path: impl AsRef<Path>) -> Result<Graph, GraphError> {
    let path = path.as_ref();
    let fmt = Format::from_path(path).ok_or_else(|| {
        GraphError::Deserialize(format!("unrecognised file extension: {}", path.display()))
    })?;
    load_with_format(path, fmt)
}

/// Load a graph using an explicit format.
pub fn load_with_format(path: impl AsRef<Path>, fmt: Format) -> Result<Graph, GraphError> {
    match fmt {
        Format::Json => json::load_json(path),
        #[cfg(feature = "msgpack")]
        Format::MessagePack => msgpack::load_msgpack(path),
        Format::Packed => packed::load_packed(path),
    }
}

/// Save a graph to disk, auto-detecting format from the file extension.
pub fn save(graph: &Graph, path: impl AsRef<Path>) -> Result<(), GraphError> {
    let path = path.as_ref();
    let fmt = Format::from_path(path).ok_or_else(|| {
        GraphError::Deserialize(format!("unrecognised file extension: {}", path.display()))
    })?;
    save_with_format(graph, path, fmt)
}

/// Save a graph using an explicit format.
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
pub fn to_bytes(graph: &Graph, fmt: Format) -> Result<Vec<u8>, GraphError> {
    match fmt {
        Format::Json => json::to_json_bytes(graph),
        #[cfg(feature = "msgpack")]
        Format::MessagePack => msgpack::to_msgpack_bytes(graph),
        Format::Packed => packed::to_packed_bytes(graph),
    }
}

/// Deserialize a graph from bytes in the given format.
pub fn from_bytes(bytes: &[u8], fmt: Format) -> Result<Graph, GraphError> {
    match fmt {
        Format::Json => json::from_json_bytes(bytes),
        #[cfg(feature = "msgpack")]
        Format::MessagePack => msgpack::from_msgpack_bytes(bytes),
        Format::Packed => packed::from_packed_bytes(bytes),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::edge::Edge;
    use tempfile::NamedTempFile;

    #[test]
    fn round_trips_graph_through_file() {
        let mut g = Graph::new();
        g.add_edge(Edge::new("A", "calls", "B"));
        let f = NamedTempFile::new().unwrap();
        save_graph(&g, f.path()).unwrap();
        let g2 = load_graph(f.path()).unwrap();
        assert_eq!(g2.edge_count(), 1);
        assert!(g2.exists("A", "calls", "B"));
    }
}
