//! Safetensors header scanning.

use std::io::Write;
use std::path::Path;

use crate::inventory::tensors::scan_tensors;

/// Write a safetensors file whose header describes `entries`; tensor data
/// bytes are irrelevant to the scan and omitted.
fn write_shard(dir: &Path, name: &str, header: &serde_json::Value) {
    let header_bytes = serde_json::to_vec(header).unwrap();
    let mut file = std::fs::File::create(dir.join(name)).unwrap();
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header_bytes).unwrap();
}

fn shard_header(entries: &[(&str, &str, &[usize], u64)]) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    let mut offset = 0u64;
    for (name, dtype, shape, bytes) in entries {
        map.insert(
            (*name).to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + bytes]
            }),
        );
        offset += bytes;
    }
    serde_json::Value::Object(map)
}

#[test]
fn scans_shards_via_the_index_file() {
    let dir = tempfile::tempdir().unwrap();
    write_shard(
        dir.path(),
        "model-00001-of-00002.safetensors",
        &shard_header(&[
            ("model.layers.0.mlp.gate_proj.weight", "BF16", &[8, 4], 64),
            ("model.embed_tokens.weight", "BF16", &[16, 4], 128),
        ]),
    );
    write_shard(
        dir.path(),
        "model-00002-of-00002.safetensors",
        &shard_header(&[("lm_head.weight", "BF16", &[16, 4], 128)]),
    );
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        serde_json::json!({
            "weight_map": {
                "model.layers.0.mlp.gate_proj.weight": "model-00001-of-00002.safetensors",
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors"
            }
        })
        .to_string(),
    )
    .unwrap();

    let inv = scan_tensors(dir.path()).unwrap();
    assert_eq!(inv.total_tensors, 3);
    assert_eq!(inv.total_bytes, 320);
    assert_eq!(inv.files.len(), 2);

    // Tensors are name-sorted.
    let names: Vec<&str> = inv.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "lm_head.weight",
            "model.embed_tokens.weight",
            "model.layers.0.mlp.gate_proj.weight"
        ]
    );

    // Shapes and shard attribution flow through.
    let head = &inv.tensors[0];
    assert_eq!(head.shape, vec![16, 4]);
    assert_eq!(head.dtype, "BF16");
    assert_eq!(head.file, "model-00002-of-00002.safetensors");
}

#[test]
fn groups_cut_at_the_first_numeric_segment() {
    let dir = tempfile::tempdir().unwrap();
    write_shard(
        dir.path(),
        "model.safetensors",
        &shard_header(&[
            ("model.layers.0.mlp.gate_proj.weight", "BF16", &[8, 4], 64),
            ("model.layers.1.mlp.gate_proj.weight", "BF16", &[8, 4], 64),
            ("lm_head.weight", "BF16", &[16, 4], 128),
        ]),
    );
    let inv = scan_tensors(dir.path()).unwrap();
    let prefixes: Vec<(&str, usize, u64)> = inv
        .groups
        .iter()
        .map(|g| (g.prefix.as_str(), g.tensors, g.bytes))
        .collect();
    assert_eq!(
        prefixes,
        vec![("lm_head", 1, 128), ("model.layers", 2, 128)]
    );
}

/// Without an index file, `*.safetensors` in the directory are scanned.
#[test]
fn falls_back_to_directory_scan_without_an_index() {
    let dir = tempfile::tempdir().unwrap();
    write_shard(
        dir.path(),
        "model.safetensors",
        &shard_header(&[("w", "F32", &[2, 2], 16)]),
    );
    let inv = scan_tensors(dir.path()).unwrap();
    assert_eq!(inv.total_tensors, 1);
    assert_eq!(inv.files, vec!["model.safetensors"]);
}

/// A config-only directory yields an empty inventory, not an error.
#[test]
fn empty_directory_is_an_empty_inventory() {
    let dir = tempfile::tempdir().unwrap();
    let inv = scan_tensors(dir.path()).unwrap();
    assert_eq!(inv.total_tensors, 0);
    assert_eq!(inv.total_bytes, 0);
    assert!(inv.files.is_empty());
}

/// The `__metadata__` header entry is not a tensor.
#[test]
fn metadata_header_entry_is_skipped() {
    let dir = tempfile::tempdir().unwrap();
    let mut header = shard_header(&[("w", "F32", &[2], 8)]);
    header.as_object_mut().unwrap().insert(
        "__metadata__".to_string(),
        serde_json::json!({"format": "pt"}),
    );
    write_shard(dir.path(), "model.safetensors", &header);
    let inv = scan_tensors(dir.path()).unwrap();
    assert_eq!(inv.total_tensors, 1);
    assert_eq!(inv.tensors[0].name, "w");
}

/// An absurd header length is a corrupt file, not an allocation.
#[test]
fn oversized_header_length_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let mut file = std::fs::File::create(dir.path().join("model.safetensors")).unwrap();
    file.write_all(&u64::MAX.to_le_bytes()).unwrap();
    let err = scan_tensors(dir.path()).unwrap_err();
    assert!(err
        .to_string()
        .contains("corrupt or not a safetensors file"));
}
