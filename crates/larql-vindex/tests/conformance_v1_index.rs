//! Vindex v1 conformance — `index.json`.
//!
//! Contract: a corrupt or inconsistent `index.json` fails
//! `load_vindex` / `load_vindex_config` with `VindexError` — never a
//! panic, never a silently mis-shaped index. Companion doc:
//! `docs/conformance-v1.md` §index.json.

mod common;

use common::{load_dir, save_minimal_vindex, FIXTURE_LAYERS};
use larql_vindex::load_vindex_config;

const INDEX_JSON: &str = "index.json";

/// Replace one artifact in an otherwise-valid vindex directory.
fn vindex_with_index_json(json: &str) -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    save_minimal_vindex(tmp.path());
    std::fs::write(tmp.path().join(INDEX_JSON), json).unwrap();
    tmp
}

#[test]
fn valid_fixture_loads() {
    let tmp = tempfile::tempdir().unwrap();
    save_minimal_vindex(tmp.path());
    let index = load_dir(tmp.path()).expect("canonical fixture loads");
    assert_eq!(index.num_layers, FIXTURE_LAYERS);
}

#[test]
fn malformed_json_is_error() {
    let tmp = vindex_with_index_json("{not valid json");
    assert!(load_dir(tmp.path()).is_err());
    assert!(load_vindex_config(tmp.path()).is_err());
}

/// Required structural fields must be present — serde has no default
/// for `num_layers`, `hidden_size`, `layers`, etc.
#[test]
fn missing_required_field_is_error() {
    // Valid JSON, but no `num_layers`.
    let json = r#"{
        "version": 2, "model": "m", "family": "synthetic",
        "hidden_size": 8, "intermediate_size": 4, "vocab_size": 16,
        "embed_scale": 1.0, "layers": [], "down_top_k": 1
    }"#;
    let tmp = vindex_with_index_json(json);
    let err = load_dir(tmp.path())
        .err()
        .expect("missing num_layers errors");
    assert!(err.to_string().contains("num_layers"), "{err}");
}

/// An enum field with an unknown value is rejected, not defaulted.
#[test]
fn unknown_dtype_is_error() {
    let json = r#"{
        "version": 2, "model": "m", "family": "synthetic",
        "num_layers": 2, "hidden_size": 8, "intermediate_size": 4,
        "vocab_size": 16, "embed_scale": 1.0, "layers": [],
        "down_top_k": 1, "dtype": "f64"
    }"#;
    let tmp = vindex_with_index_json(json);
    let err = load_dir(tmp.path()).err().expect("unknown dtype errors");
    assert!(err.to_string().contains("f64"), "{err}");
}

#[test]
fn unknown_quant_tag_is_error() {
    let json = r#"{
        "version": 2, "model": "m", "family": "synthetic",
        "num_layers": 2, "hidden_size": 8, "intermediate_size": 4,
        "vocab_size": 16, "embed_scale": 1.0, "layers": [],
        "down_top_k": 1, "quant": "q9z"
    }"#;
    let tmp = vindex_with_index_json(json);
    assert!(load_dir(tmp.path()).is_err());
}

/// Contract pin for the 2026-07-30 H4 fix: a layer entry with
/// `layer >= num_layers` (truncated or hand-edited manifest) errors
/// instead of panicking on the `gate_slices[layer]` write.
#[test]
fn out_of_range_layer_entry_is_error_not_panic() {
    let json = r#"{
        "version": 2, "model": "m", "family": "synthetic",
        "num_layers": 2, "hidden_size": 8, "intermediate_size": 4,
        "vocab_size": 16, "embed_scale": 1.0, "down_top_k": 1,
        "layers": [
            {"layer": 5, "num_features": 4, "offset": 0, "length": 128}
        ]
    }"#;
    let tmp = vindex_with_index_json(json);
    let err = load_dir(tmp.path())
        .err()
        .expect("out-of-range layer errors");
    let msg = err.to_string();
    assert!(msg.contains("out of range"), "{msg}");
    assert!(msg.contains('5'), "{msg}");
}

/// Wrong *types* on structural fields are rejected (string where a
/// number belongs).
#[test]
fn wrong_field_type_is_error() {
    let json = r#"{
        "version": 2, "model": "m", "family": "synthetic",
        "num_layers": "two", "hidden_size": 8, "intermediate_size": 4,
        "vocab_size": 16, "embed_scale": 1.0, "layers": [],
        "down_top_k": 1
    }"#;
    let tmp = vindex_with_index_json(json);
    assert!(load_dir(tmp.path()).is_err());
}

#[test]
fn missing_index_json_is_error() {
    let tmp = tempfile::tempdir().unwrap();
    save_minimal_vindex(tmp.path());
    std::fs::remove_file(tmp.path().join(INDEX_JSON)).unwrap();
    assert!(load_dir(tmp.path()).is_err());
}
