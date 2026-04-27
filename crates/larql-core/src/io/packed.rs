//! Packed binary edge format (.larql.pak).
//!
//! Compact, string-interned format optimized for fast loading of runtime graphs.
//!
//! Layout:
//!   Header (40 bytes)
//!   Edge records (28 bytes each, fixed-width)
//!   Metadata section (variable-length JSON per edge)
//!   String table (length-prefixed UTF-8 strings)

use std::collections::HashMap;
use std::io::{self, Cursor, Read, Write};
use std::path::Path;

use crate::core::edge::Edge;
use crate::core::enums::SourceType;
use crate::core::graph::{Graph, GraphError};

const MAGIC: [u8; 4] = *b"LARQ";
const FORMAT_VERSION: u16 = 1;
const HEADER_SIZE: usize = 32;
const EDGE_RECORD_SIZE: usize = 28;

// ── String table ──

struct StringTable {
    strings: Vec<String>,
    index: HashMap<String, u32>,
}

impl StringTable {
    fn new() -> Self {
        Self {
            strings: Vec::new(),
            index: HashMap::new(),
        }
    }

    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&idx) = self.index.get(s) {
            return idx;
        }
        let idx = self.strings.len() as u32;
        self.index.insert(s.to_string(), idx);
        self.strings.push(s.to_string());
        idx
    }

    fn resolve(&self, idx: u32) -> &str {
        &self.strings[idx as usize]
    }

    fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        for s in &self.strings {
            let bytes = s.as_bytes();
            w.write_all(&(bytes.len() as u32).to_le_bytes())?;
            w.write_all(bytes)?;
        }
        Ok(())
    }

    fn read_from(data: &[u8], count: u64) -> Result<Self, GraphError> {
        let mut table = StringTable::new();
        let mut cursor = Cursor::new(data);

        for _ in 0..count {
            let mut len_buf = [0u8; 4];
            cursor
                .read_exact(&mut len_buf)
                .map_err(|e| GraphError::Deserialize(format!("string table: {e}")))?;
            let len = u32::from_le_bytes(len_buf) as usize;

            let mut str_buf = vec![0u8; len];
            cursor
                .read_exact(&mut str_buf)
                .map_err(|e| GraphError::Deserialize(format!("string table: {e}")))?;
            let s = String::from_utf8(str_buf)
                .map_err(|e| GraphError::Deserialize(format!("invalid UTF-8: {e}")))?;

            let idx = table.strings.len() as u32;
            table.index.insert(s.clone(), idx);
            table.strings.push(s);
        }

        Ok(table)
    }
}

// ── Source type encoding ──

fn source_to_u8(s: &SourceType) -> u8 {
    match s {
        SourceType::Unknown => 0,
        SourceType::Parametric => 1,
        SourceType::Document => 2,
        SourceType::Installed => 3,
        SourceType::Wikidata => 4,
        SourceType::Manual => 5,
    }
}

fn u8_to_source(v: u8) -> SourceType {
    match v {
        1 => SourceType::Parametric,
        2 => SourceType::Document,
        3 => SourceType::Installed,
        4 => SourceType::Wikidata,
        5 => SourceType::Manual,
        _ => SourceType::Unknown,
    }
}

// ── Edge record ──
//
// 28 bytes:
//   subject_idx:  u32 (4)
//   relation_idx: u32 (4)
//   object_idx:   u32 (4)
//   confidence:   f32 (4)
//   source:       u8  (1)
//   has_metadata:  u8  (1) — 1 if this edge has metadata in the metadata section
//   has_injection: u8  (1) — 1 if this edge has injection data
//   _padding:      u8  (1)
//   meta_offset:   u32 (4) — offset into metadata section (0 if none)
//   meta_len:      u32 (4) — byte length in metadata section

struct PackedEdgeRecord {
    subj: u32,
    rel: u32,
    obj: u32,
    conf: f32,
    source: u8,
    has_meta: bool,
    has_inj: bool,
    meta_offset: u32,
    meta_len: u32,
}

impl PackedEdgeRecord {
    fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(&self.subj.to_le_bytes())?;
        w.write_all(&self.rel.to_le_bytes())?;
        w.write_all(&self.obj.to_le_bytes())?;
        w.write_all(&self.conf.to_le_bytes())?;
        w.write_all(&[self.source, self.has_meta as u8, self.has_inj as u8, 0])?;
        w.write_all(&self.meta_offset.to_le_bytes())?;
        w.write_all(&self.meta_len.to_le_bytes())?;
        Ok(())
    }
}

// ── Public API ──

/// Serialize a graph to packed binary bytes.
pub fn to_packed_bytes(graph: &Graph) -> Result<Vec<u8>, GraphError> {
    let mut strings = StringTable::new();
    let edges = graph.edges();

    // First pass: intern all strings and build metadata blobs
    struct EdgeRecord {
        subj: u32,
        rel: u32,
        obj: u32,
        conf: f32,
        source: u8,
        meta_blob: Option<Vec<u8>>,
        inj_blob: Option<Vec<u8>>,
    }

    let mut records: Vec<EdgeRecord> = Vec::with_capacity(edges.len());
    for edge in edges {
        let subj = strings.intern(&edge.subject);
        let rel = strings.intern(&edge.relation);
        let obj = strings.intern(&edge.object);

        let meta_blob = edge.metadata.as_ref().map(|m| {
            serde_json::to_vec(m).unwrap_or_default()
        });

        let inj_blob = edge.injection.map(|(layer, score)| {
            let mut buf = Vec::with_capacity(12);
            buf.extend_from_slice(&(layer as u32).to_le_bytes());
            buf.extend_from_slice(&(score as f32).to_le_bytes());
            buf
        });

        records.push(EdgeRecord {
            subj,
            rel,
            obj,
            conf: edge.confidence as f32,
            source: source_to_u8(&edge.source),
            meta_blob,
            inj_blob,
        });
    }

    // Calculate metadata section
    let mut meta_section = Vec::new();
    let mut meta_offsets: Vec<(u32, u32)> = Vec::with_capacity(records.len());
    for rec in &records {
        // Combine metadata + injection into one blob per edge
        let has_meta = rec.meta_blob.is_some();
        let has_inj = rec.inj_blob.is_some();
        if has_meta || has_inj {
            let offset = meta_section.len() as u32;
            if let Some(ref blob) = rec.meta_blob {
                meta_section.extend_from_slice(blob);
            }
            if let Some(ref blob) = rec.inj_blob {
                meta_section.extend_from_slice(blob);
            }
            let len = meta_section.len() as u32 - offset;
            meta_offsets.push((offset, len));
        } else {
            meta_offsets.push((0, 0));
        }
    }

    // Calculate offsets
    let edge_section_size = records.len() * EDGE_RECORD_SIZE;
    let meta_section_offset = HEADER_SIZE + edge_section_size;
    let string_table_offset = meta_section_offset + meta_section.len();

    // Write header
    let total_size = string_table_offset + estimate_string_table_size(&strings);
    let mut buf = Vec::with_capacity(total_size);

    buf.extend_from_slice(&MAGIC);
    buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    buf.extend_from_slice(&[0u8; 2]); // flags
    buf.extend_from_slice(&(records.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(strings.strings.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(string_table_offset as u64).to_le_bytes());

    // Write edge records
    for (i, rec) in records.iter().enumerate() {
        let (meta_off, meta_len) = meta_offsets[i];
        let packed = PackedEdgeRecord {
            subj: rec.subj,
            rel: rec.rel,
            obj: rec.obj,
            conf: rec.conf,
            source: rec.source,
            has_meta: rec.meta_blob.is_some(),
            has_inj: rec.inj_blob.is_some(),
            meta_offset: meta_off,
            meta_len,
        };
        packed.write_to(&mut buf).map_err(GraphError::Io)?;
    }

    // Write metadata section
    buf.extend_from_slice(&meta_section);

    // Write string table
    strings
        .write_to(&mut buf)
        .map_err(GraphError::Io)?;

    Ok(buf)
}

/// Deserialize a graph from packed binary bytes.
pub fn from_packed_bytes(bytes: &[u8]) -> Result<Graph, GraphError> {
    if bytes.len() < HEADER_SIZE {
        return Err(GraphError::Deserialize("file too small for header".into()));
    }

    // Read header
    if bytes[0..4] != MAGIC {
        return Err(GraphError::Deserialize("invalid magic bytes".into()));
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != FORMAT_VERSION {
        return Err(GraphError::Deserialize(format!(
            "unsupported format version: {version}"
        )));
    }
    let num_edges = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let num_strings = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    let string_table_offset = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;

    // Read string table
    let string_data = &bytes[string_table_offset..];
    let strings = StringTable::read_from(string_data, num_strings)?;

    // Metadata section is between edge records and string table
    let edge_section_end = HEADER_SIZE + num_edges * EDGE_RECORD_SIZE;
    let meta_section = &bytes[edge_section_end..string_table_offset];

    // Read edge records
    let mut graph = Graph::new();
    for i in 0..num_edges {
        let offset = HEADER_SIZE + i * EDGE_RECORD_SIZE;
        let rec = &bytes[offset..offset + EDGE_RECORD_SIZE];

        let subj_idx = u32::from_le_bytes(rec[0..4].try_into().unwrap());
        let rel_idx = u32::from_le_bytes(rec[4..8].try_into().unwrap());
        let obj_idx = u32::from_le_bytes(rec[8..12].try_into().unwrap());
        let conf = f32::from_le_bytes(rec[12..16].try_into().unwrap());
        let source = u8_to_source(rec[16]);
        let has_meta = rec[17] != 0;
        let has_inj = rec[18] != 0;
        let meta_offset = u32::from_le_bytes(rec[20..24].try_into().unwrap()) as usize;
        let meta_len = u32::from_le_bytes(rec[24..28].try_into().unwrap()) as usize;

        let subject = strings.resolve(subj_idx).to_string();
        let relation = strings.resolve(rel_idx).to_string();
        let object = strings.resolve(obj_idx).to_string();

        let mut edge = Edge::new(subject, relation, object)
            .with_confidence(conf as f64)
            .with_source(source);

        // Decode metadata + injection from blob
        if meta_len > 0 && meta_offset + meta_len <= meta_section.len() {
            let blob = &meta_section[meta_offset..meta_offset + meta_len];

            if has_meta && has_inj && blob.len() >= 8 {
                // Last 8 bytes are injection (u32 layer + f32 score)
                let meta_json_end = blob.len() - 8;
                if let Ok(meta) = serde_json::from_slice::<HashMap<String, serde_json::Value>>(
                    &blob[..meta_json_end],
                ) {
                    edge.metadata = Some(meta);
                }
                let inj_layer =
                    u32::from_le_bytes(blob[meta_json_end..meta_json_end + 4].try_into().unwrap())
                        as usize;
                let inj_score = f32::from_le_bytes(
                    blob[meta_json_end + 4..meta_json_end + 8].try_into().unwrap(),
                ) as f64;
                edge.injection = Some((inj_layer, inj_score));
            } else if has_meta {
                if let Ok(meta) =
                    serde_json::from_slice::<HashMap<String, serde_json::Value>>(blob)
                {
                    edge.metadata = Some(meta);
                }
            } else if has_inj && blob.len() >= 8 {
                let inj_layer = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;
                let inj_score = f32::from_le_bytes(blob[4..8].try_into().unwrap()) as f64;
                edge.injection = Some((inj_layer, inj_score));
            }
        }

        graph.add_edge(edge);
    }

    Ok(graph)
}

/// Save a graph to a packed binary file.
pub fn save_packed(graph: &Graph, path: impl AsRef<Path>) -> Result<(), GraphError> {
    let bytes = to_packed_bytes(graph)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

/// Load a graph from a packed binary file.
pub fn load_packed(path: impl AsRef<Path>) -> Result<Graph, GraphError> {
    let bytes = std::fs::read(path)?;
    from_packed_bytes(&bytes)
}

fn estimate_string_table_size(strings: &StringTable) -> usize {
    strings
        .strings
        .iter()
        .map(|s| 4 + s.len())
        .sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::enums::SourceType;

    #[test]
    fn test_roundtrip_basic() {
        let mut graph = Graph::new();
        graph.add_edge(Edge::new("France", "capital-of", "Paris").with_confidence(0.9));
        graph.add_edge(
            Edge::new("Germany", "capital-of", "Berlin")
                .with_confidence(0.85)
                .with_source(SourceType::Parametric),
        );
        graph.add_edge(Edge::new("Japan", "capital-of", "Tokyo").with_confidence(0.95));

        let bytes = to_packed_bytes(&graph).unwrap();
        let loaded = from_packed_bytes(&bytes).unwrap();

        assert_eq!(loaded.edge_count(), 3);
        assert!(loaded.exists("France", "capital-of", "Paris"));
        assert!(loaded.exists("Germany", "capital-of", "Berlin"));
        assert!(loaded.exists("Japan", "capital-of", "Tokyo"));
    }

    #[test]
    fn test_roundtrip_with_metadata() {
        let mut graph = Graph::new();
        graph.add_edge(
            Edge::new("France", "capital-of", "Paris")
                .with_confidence(0.9)
                .with_metadata("layer", serde_json::json!(26))
                .with_metadata("feature", serde_json::json!(9515)),
        );

        let bytes = to_packed_bytes(&graph).unwrap();
        let loaded = from_packed_bytes(&bytes).unwrap();

        assert_eq!(loaded.edge_count(), 1);
        let edge = &loaded.edges()[0];
        let meta = edge.metadata.as_ref().unwrap();
        assert_eq!(meta.get("layer").unwrap(), &serde_json::json!(26));
        assert_eq!(meta.get("feature").unwrap(), &serde_json::json!(9515));
    }

    #[test]
    fn test_roundtrip_with_injection() {
        let mut graph = Graph::new();
        let mut edge = Edge::new("A", "rel", "B").with_confidence(0.5);
        edge.injection = Some((10, 0.75));
        graph.add_edge(edge);

        let bytes = to_packed_bytes(&graph).unwrap();
        let loaded = from_packed_bytes(&bytes).unwrap();

        assert_eq!(loaded.edge_count(), 1);
        let loaded_edge = &loaded.edges()[0];
        let (layer, score) = loaded_edge.injection.unwrap();
        assert_eq!(layer, 10);
        assert!((score - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_roundtrip_with_metadata_and_injection() {
        let mut graph = Graph::new();
        let mut edge = Edge::new("X", "rel", "Y")
            .with_confidence(0.8)
            .with_metadata("key", serde_json::json!("value"));
        edge.injection = Some((5, 0.5));
        graph.add_edge(edge);

        let bytes = to_packed_bytes(&graph).unwrap();
        let loaded = from_packed_bytes(&bytes).unwrap();

        assert_eq!(loaded.edge_count(), 1);
        let e = &loaded.edges()[0];
        assert!(e.metadata.is_some());
        assert_eq!(
            e.metadata.as_ref().unwrap().get("key").unwrap(),
            &serde_json::json!("value")
        );
        let (layer, score) = e.injection.unwrap();
        assert_eq!(layer, 5);
        assert!((score - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_roundtrip_empty_graph() {
        let graph = Graph::new();
        let bytes = to_packed_bytes(&graph).unwrap();
        let loaded = from_packed_bytes(&bytes).unwrap();
        assert_eq!(loaded.edge_count(), 0);
    }

    #[test]
    fn test_roundtrip_source_types() {
        let mut graph = Graph::new();
        graph.add_edge(Edge::new("A", "r", "B").with_source(SourceType::Parametric));
        graph.add_edge(Edge::new("C", "r", "D").with_source(SourceType::Document));
        graph.add_edge(Edge::new("E", "r", "F").with_source(SourceType::Wikidata));
        graph.add_edge(Edge::new("G", "r", "H").with_source(SourceType::Manual));
        graph.add_edge(Edge::new("I", "r", "J").with_source(SourceType::Installed));
        graph.add_edge(Edge::new("K", "r", "L").with_source(SourceType::Unknown));

        let bytes = to_packed_bytes(&graph).unwrap();
        let loaded = from_packed_bytes(&bytes).unwrap();

        assert_eq!(loaded.edge_count(), 6);
        // Check that sources roundtrip
        let sources: Vec<_> = loaded.edges().iter().map(|e| e.source.clone()).collect();
        assert!(sources.contains(&SourceType::Parametric));
        assert!(sources.contains(&SourceType::Document));
        assert!(sources.contains(&SourceType::Wikidata));
        assert!(sources.contains(&SourceType::Manual));
        assert!(sources.contains(&SourceType::Installed));
        assert!(sources.contains(&SourceType::Unknown));
    }

    #[test]
    fn test_string_interning() {
        let mut graph = Graph::new();
        // "capital-of" appears 3 times but should be interned once
        graph.add_edge(Edge::new("France", "capital-of", "Paris"));
        graph.add_edge(Edge::new("Germany", "capital-of", "Berlin"));
        graph.add_edge(Edge::new("Japan", "capital-of", "Tokyo"));

        let bytes = to_packed_bytes(&graph).unwrap();
        // Should be much smaller than JSON
        let json_bytes = serde_json::to_vec(&graph.to_json_value()).unwrap();
        assert!(bytes.len() < json_bytes.len());
    }

    #[test]
    fn test_confidence_precision() {
        let mut graph = Graph::new();
        graph.add_edge(Edge::new("A", "rel", "B").with_confidence(0.123456789));

        let bytes = to_packed_bytes(&graph).unwrap();
        let loaded = from_packed_bytes(&bytes).unwrap();

        // f32 precision means we lose some digits
        let conf = loaded.edges()[0].confidence;
        assert!((conf - 0.123456789).abs() < 0.001);
    }

    #[test]
    fn test_invalid_magic() {
        let bytes = vec![0u8; 40];
        let result = from_packed_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_file_roundtrip() {
        let mut graph = Graph::new();
        graph.add_edge(Edge::new("A", "rel", "B").with_confidence(0.9));

        let dir = std::env::temp_dir().join("larql_packed_test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.larql.pak");

        save_packed(&graph, &path).unwrap();
        let loaded = load_packed(&path).unwrap();

        assert_eq!(loaded.edge_count(), 1);
        assert!(loaded.exists("A", "rel", "B"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
