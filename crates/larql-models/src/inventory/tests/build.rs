//! End-to-end inventory build over an on-disk fixture directory.

use std::io::Write;

use crate::inventory::{build_inventory, KeyStatus, INVENTORY_SCHEMA};

/// Minimal Glimmer-shaped checkpoint: config + one shard.
fn write_fixture(dir: &std::path::Path) {
    let config = serde_json::json!({
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "dtype": "bfloat16",
        "model_type": "muse_glimmer",
        "text_config": {
            "model_type": "muse_glimmer_text",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "intermediate_size": 256,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "sliding_window": 16,
            "vocab_size": 128,
            "qk_scale_factor": 3.87,
            "layer_types": [
                "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention"
            ]
        }
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap();

    let header = serde_json::json!({
        "model.layers.0.mlp.gate_proj.weight": {
            "dtype": "BF16", "shape": [256, 64], "data_offsets": [0, 32768]
        }
    });
    let header_bytes = serde_json::to_vec(&header).unwrap();
    let mut file = std::fs::File::create(dir.join("model.safetensors")).unwrap();
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header_bytes).unwrap();
}

#[test]
fn builds_a_complete_inventory() {
    let dir = tempfile::tempdir().unwrap();
    write_fixture(dir.path());

    let inv = build_inventory(dir.path()).unwrap();
    assert_eq!(inv.schema, INVENTORY_SCHEMA);
    assert_eq!(inv.identity.model_type, "muse_glimmer_text");
    // Glimmer is a registered family since its semantics were judged.
    assert!(!inv.detection.generic_fallback);
    assert_eq!(inv.detection.family, "muse_glimmer");
    assert_eq!(inv.resolved.num_layers, 4);
    assert_eq!(inv.resolved.attention.sliding_layers, 3);
    assert_eq!(inv.tensors.total_tensors, 1);
    assert_eq!(inv.tensors.total_bytes, 32768);

    // The once-dangerous scalar is consumed since G2b; the instrument's
    // negative case is covered by `an_unjudged_key_still_reads_unconsumed`.
    let qk = inv
        .config_keys
        .iter()
        .find(|f| f.path == "text_config.qk_scale_factor")
        .unwrap();
    assert_eq!(qk.status, KeyStatus::Consumed);
}

/// The report round-trips through JSON — it is the CLI contract.
#[test]
fn inventory_round_trips_through_json() {
    let dir = tempfile::tempdir().unwrap();
    write_fixture(dir.path());
    let inv = build_inventory(dir.path()).unwrap();
    let json = serde_json::to_string_pretty(&inv).unwrap();
    let back: crate::inventory::ArchitectureInventory = serde_json::from_str(&json).unwrap();
    assert_eq!(back.schema, inv.schema);
    assert_eq!(back.config_keys.len(), inv.config_keys.len());
    assert_eq!(back.tensors.total_bytes, inv.tensors.total_bytes);
}

#[test]
fn missing_config_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let err = build_inventory(dir.path()).unwrap_err();
    assert!(err.to_string().contains("config.json"));
}

#[test]
fn non_directory_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("not_a_dir");
    std::fs::write(&file, "x").unwrap();
    assert!(build_inventory(&file).is_err());
}
