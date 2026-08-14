//! E0 — the generation-boundary rows of the VINDEX2 preservation matrix.
//!
//! # Scope, stated honestly
//!
//! The full E0 matrix (`docs/vindex3-experiments.md`) covers loading,
//! verification, WALK, slicing, full decode, layer and expert sharding, and
//! publish/pull round-trips against real indexes. Those rows need multi-GB
//! checkpoints and cannot run in CI; they run locally against the committed
//! goldens in `tests/goldens/e0/` via `scripts/e0-capture-goldens.sh`.
//!
//! What runs *here* is the subset that needs no weights at all: **generation
//! detection and the loader boundary**. That is deliberately the subset most
//! at risk from VINDEX3 work, because every commit touching `format/load.rs`,
//! `format/generation.rs` or the LYRW parsers can break it, and nothing else
//! in CI would notice.
//!
//! The matrix row this discharges:
//!
//! ```text
//! generation dispatch → index.json.version 2→3 routing correct;
//!                       unknown version fails naming both sides
//! ```

use std::path::Path;

use larql_vindex::format::filenames::INDEX_JSON;
use larql_vindex::format::generation::{
    detect_generation, ContainerGeneration, V2_CURRENT_SCHEMA, V3_CURRENT_SCHEMA,
};
use larql_vindex::format::load::load_vindex_config;
use larql_vindex::VindexError;
use tempfile::tempdir;

/// A VINDEX2 `index.json` with the fields the shipped loader requires.
fn write_v2_index(dir: &Path) {
    std::fs::write(
        dir.join(INDEX_JSON),
        serde_json::json!({
            "version": V2_CURRENT_SCHEMA,
            "model": "e0-fixture",
            "family": "llama",
            "num_layers": 2,
            "hidden_size": 8,
            "intermediate_size": 8,
            "vocab_size": 4,
            "embed_scale": 1.0,
            "layers": [],
            "down_top_k": 0
        })
        .to_string(),
    )
    .unwrap();
}

/// A VINDEX3 `index.json`, carrying keys the v1 config struct has never seen.
fn write_v3_index(dir: &Path) {
    std::fs::write(
        dir.join(INDEX_JSON),
        serde_json::json!({
            "version": V3_CURRENT_SCHEMA,
            "moe_manifest": "moe_manifest.json",
            "profiles": ["exact", "browse"],
            "segments": { "routed/layer_000": 2 }
        })
        .to_string(),
    )
    .unwrap();
}

#[test]
fn a_shipped_generation_index_is_detected_as_vindex2() {
    let dir = tempdir().unwrap();
    write_v2_index(dir.path());
    assert_eq!(
        detect_generation(dir.path()).unwrap(),
        ContainerGeneration::V2
    );
}

#[test]
fn a_successor_index_is_detected_as_vindex3() {
    let dir = tempdir().unwrap();
    write_v3_index(dir.path());
    assert_eq!(
        detect_generation(dir.path()).unwrap(),
        ContainerGeneration::V3
    );
}

#[test]
fn the_shipped_loader_still_loads_a_shipped_index() {
    // The row that matters most: VINDEX3 work must not have broken VINDEX2.
    let dir = tempdir().unwrap();
    write_v2_index(dir.path());
    let config = load_vindex_config(dir.path()).expect("VINDEX2 must still load");
    assert_eq!(config.version, V2_CURRENT_SCHEMA);
    assert_eq!(config.num_layers, 2);
}

#[test]
fn the_shipped_loader_refuses_a_successor_index_by_generation() {
    // Without this check the v2 config struct deserialises a v3 index on the
    // strength of shared field names and proceeds against a layout whose
    // weights live somewhere else — a served model with wrong weights.
    let dir = tempdir().unwrap();
    write_v3_index(dir.path());
    let Err(err) = load_vindex_config(dir.path()) else {
        panic!("must refuse a VINDEX3 directory");
    };
    let text = err.to_string();
    assert!(
        matches!(err, VindexError::WrongContainerGeneration { .. }),
        "{text}"
    );
    assert!(text.contains("VINDEX3"), "{text}");
    assert!(text.contains("VINDEX2"), "{text}");
}

#[test]
fn an_unknown_generation_names_the_version_found_and_both_supported() {
    let dir = tempdir().unwrap();
    std::fs::write(
        dir.path().join(INDEX_JSON),
        serde_json::json!({ "version": 99 }).to_string(),
    )
    .unwrap();
    let err = detect_generation(dir.path()).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("99"), "{text}");
    assert!(text.contains("VINDEX2"), "{text}");
    assert!(text.contains("VINDEX3"), "{text}");
}

#[test]
fn an_index_without_a_version_field_is_refused_rather_than_guessed() {
    let dir = tempdir().unwrap();
    std::fs::write(
        dir.path().join(INDEX_JSON),
        serde_json::json!({ "model": "no-version" }).to_string(),
    )
    .unwrap();
    assert!(detect_generation(dir.path()).is_err());
}

#[test]
fn a_legacy_schema_index_still_loads_as_the_shipped_generation() {
    // The regression E0 caught. index.json version 1 predates several fields
    // and loads with defaults; it is the same container generation, and
    // refusing it broke exactly the compatibility this matrix protects.
    let dir = tempdir().unwrap();
    std::fs::write(
        dir.path().join(INDEX_JSON),
        serde_json::json!({
            "version": 1,
            "model": "legacy",
            "family": "test",
            "num_layers": 1,
            "hidden_size": 4,
            "intermediate_size": 4,
            "vocab_size": 4,
            "embed_scale": 1.0,
            "layers": [],
            "down_top_k": 0
        })
        .to_string(),
    )
    .unwrap();
    assert_eq!(
        detect_generation(dir.path()).unwrap(),
        ContainerGeneration::V2
    );
    let config = load_vindex_config(dir.path()).expect("legacy schema must still load");
    assert_eq!(config.version, 1);
}

#[test]
fn a_missing_index_json_is_an_io_error_not_a_generation_verdict() {
    let dir = tempdir().unwrap();
    assert!(matches!(
        detect_generation(dir.path()),
        Err(VindexError::Io(_))
    ));
}

#[test]
fn a_lyrw_v2_layer_file_is_refused_by_the_shipped_parser() {
    // The other half of the boundary. A VINDEX3 layer file parsed at the v1
    // stride would not bounds-fail — it would yield offsets still inside the
    // file and hand back bytes from the wrong expert.
    use larql_vindex::format::weights::write_layers::parse_layer_weights_header;

    let mut header = Vec::new();
    header.extend_from_slice(&u32::from_le_bytes(*b"LYRW").to_le_bytes());
    header.extend_from_slice(&2u32.to_le_bytes()); // format_version = 2
    header.extend_from_slice(&[0u8; 64]);

    assert!(
        parse_layer_weights_header(&header).is_none(),
        "the shipped parser must refuse a LYRW v2 file"
    );
}

#[test]
fn a_lyrw_v1_layer_file_is_still_accepted_by_the_shipped_parser() {
    // Refusing the successor must not have broken the incumbent.
    use larql_vindex::format::weights::write_layers::parse_layer_weights_header;

    let mut header = Vec::new();
    header.extend_from_slice(&u32::from_le_bytes(*b"LYRW").to_le_bytes());
    header.extend_from_slice(&1u32.to_le_bytes()); // format_version = 1
    header.extend_from_slice(&4u32.to_le_bytes()); // quant_format = Q4_K
    header.extend_from_slice(&0u32.to_le_bytes()); // num_entries
    header.extend_from_slice(&8u32.to_le_bytes()); // intermediate
    header.extend_from_slice(&8u32.to_le_bytes()); // hidden

    assert!(
        parse_layer_weights_header(&header).is_some(),
        "the shipped parser must still read LYRW v1"
    );
}
