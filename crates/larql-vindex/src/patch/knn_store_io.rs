//! Binary `.lknn` save / load for `KnnStore`.
//!
//! Format (little-endian):
//!   magic   = `b"LKNN"`        (4 bytes)
//!   version = 1                (u32)
//!   dim     = key dimension    (u32)
//!   n_layers                   (u32)
//!   per layer:
//!     layer_id                 (u32)
//!     n_entries                (u32)
//!     keys                     (n_entries × dim × f16)
//!     target_ids               (n_entries × u32)
//!     per entry: meta_len (u32) + meta_bytes (JSON)
//!
//! Keys are quantised to f16 — KNN cosine retrieval doesn't need f32
//! precision. Reconstruction goes through `KnnStore::from_entries`.

use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::path::Path;

use super::knn_store::{KnnEntry, KnnStore};

const MAGIC: &[u8; 4] = b"LKNN";
const VERSION: u32 = 1;

impl KnnStore {
    /// Save to binary format with f16 keys.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());

        // Infer dim from first entry
        let entries = self.entries();
        let dim = entries
            .values()
            .flat_map(|v| v.first())
            .map(|e| e.key.len())
            .next()
            .unwrap_or(0) as u32;
        buf.extend_from_slice(&dim.to_le_bytes());

        let num_layers = entries.len() as u32;
        buf.extend_from_slice(&num_layers.to_le_bytes());

        // Per layer
        for (&layer, entries) in entries {
            buf.extend_from_slice(&(layer as u32).to_le_bytes());
            buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());

            // Keys as f16 (flat: num_entries * dim)
            for entry in entries {
                for &v in &entry.key {
                    let bits = larql_models::quant::half::f32_to_f16(v);
                    buf.extend_from_slice(&bits.to_le_bytes());
                }
            }

            // Target IDs
            for entry in entries {
                buf.extend_from_slice(&entry.target_id.to_le_bytes());
            }

            // Metadata as JSON blob per entry
            for entry in entries {
                let meta = serde_json::json!({
                    "target_token": entry.target_token,
                    "entity": entry.entity,
                    "relation": entry.relation,
                    "confidence": entry.confidence,
                });
                let meta_bytes = serde_json::to_vec(&meta)
                    .map_err(|e| format!("json encode: {e}"))?;
                buf.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
                buf.extend_from_slice(&meta_bytes);
            }
        }

        std::fs::write(path, &buf).map_err(|e| format!("write knn_store: {e}"))
    }

    /// Load from binary format.
    pub fn load(path: &Path) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("read knn_store: {e}"))?;
        let mut cursor = Cursor::new(data.as_slice());

        let mut magic = [0u8; 4];
        cursor
            .read_exact(&mut magic)
            .map_err(|e| format!("read magic: {e}"))?;
        if &magic != MAGIC {
            return Err(format!("bad magic: expected LKNN, got {:?}", magic));
        }

        let version = read_u32(&mut cursor)?;
        if version != VERSION {
            return Err(format!("unsupported knn_store version: {version}"));
        }

        let dim = read_u32(&mut cursor)? as usize;
        let num_layers = read_u32(&mut cursor)? as usize;

        let mut entries = HashMap::new();
        for _ in 0..num_layers {
            let layer = read_u32(&mut cursor)? as usize;
            let num_entries = read_u32(&mut cursor)? as usize;

            // Keys (f16 → f32)
            let mut keys = Vec::with_capacity(num_entries);
            for _ in 0..num_entries {
                let mut key = Vec::with_capacity(dim);
                for _ in 0..dim {
                    let bits = read_u16(&mut cursor)?;
                    key.push(larql_models::quant::half::f16_to_f32(bits));
                }
                keys.push(key);
            }

            // Target IDs
            let mut target_ids = Vec::with_capacity(num_entries);
            for _ in 0..num_entries {
                target_ids.push(read_u32(&mut cursor)?);
            }

            // Metadata JSON blobs
            let mut layer_entries = Vec::with_capacity(num_entries);
            for i in 0..num_entries {
                let meta_len = read_u32(&mut cursor)? as usize;
                let mut meta_bytes = vec![0u8; meta_len];
                cursor
                    .read_exact(&mut meta_bytes)
                    .map_err(|e| format!("read meta: {e}"))?;
                let meta: serde_json::Value = serde_json::from_slice(&meta_bytes)
                    .map_err(|e| format!("json decode: {e}"))?;

                layer_entries.push(KnnEntry {
                    key: keys[i].clone(),
                    target_id: target_ids[i],
                    target_token: meta["target_token"].as_str().unwrap_or("").to_string(),
                    entity: meta["entity"].as_str().unwrap_or("").to_string(),
                    relation: meta["relation"].as_str().unwrap_or("").to_string(),
                    confidence: meta["confidence"].as_f64().unwrap_or(1.0) as f32,
                });
            }

            entries.insert(layer, layer_entries);
        }

        Ok(KnnStore::from_entries(entries))
    }
}

fn read_u32(cursor: &mut Cursor<&[u8]>) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| format!("read u32: {e}"))?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u16(cursor: &mut Cursor<&[u8]>) -> Result<u16, String> {
    let mut buf = [0u8; 2];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| format!("read u16: {e}"))?;
    Ok(u16::from_le_bytes(buf))
}
