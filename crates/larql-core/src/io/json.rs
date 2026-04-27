use std::path::Path;

use crate::core::graph::{Graph, GraphError};

/// Load a .larql.json graph from disk.
pub fn load_json(path: impl AsRef<Path>) -> Result<Graph, GraphError> {
    let contents = std::fs::read_to_string(path)?;
    let value: serde_json::Value =
        serde_json::from_str(&contents).map_err(|e| GraphError::Deserialize(e.to_string()))?;
    Graph::from_json_value(&value)
}

/// Save a graph to .larql.json format (pretty-printed).
pub fn save_json(graph: &Graph, path: impl AsRef<Path>) -> Result<(), GraphError> {
    let json = graph.to_json_value();
    let formatted =
        serde_json::to_string_pretty(&json).map_err(|e| GraphError::Deserialize(e.to_string()))?;
    std::fs::write(path, formatted)?;
    Ok(())
}

/// Serialize a graph to a JSON byte vec.
pub fn to_json_bytes(graph: &Graph) -> Result<Vec<u8>, GraphError> {
    let json = graph.to_json_value();
    serde_json::to_vec_pretty(&json).map_err(|e| GraphError::Deserialize(e.to_string()))
}

/// Deserialize a graph from JSON bytes.
pub fn from_json_bytes(bytes: &[u8]) -> Result<Graph, GraphError> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| GraphError::Deserialize(e.to_string()))?;
    Graph::from_json_value(&value)
}
