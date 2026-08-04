//! Vindex v1 conformance — `.vlp` patch files and `knn_store.bin` /
//! `.lknn`.
//!
//! Contract: corrupt patch bytes fail `VindexPatch::load` or
//! `try_apply_patch` with an error and leave the base overlay
//! untouched (no half-applied patches — the M5 class); corrupt KNN
//! stores fail `KnnStore::load` with an error and never panic or
//! pre-allocate from an attacker-controlled count. Companion doc:
//! `docs/conformance-v1.md` §.vlp / §knn_store.

mod common;

use common::{load_dir, save_minimal_vindex, FIXTURE_HIDDEN};
use larql_vindex::patch::format::encode_gate_vector;
use larql_vindex::{KnnStore, PatchedVindex, VindexPatch};

const VLP_NAME: &str = "edit.vlp";

fn write_vlp(dir: &std::path::Path, json: &str) -> std::path::PathBuf {
    let path = dir.join(VLP_NAME);
    std::fs::write(&path, json).unwrap();
    path
}

fn patch_json(ops: &str) -> String {
    format!(
        r#"{{"version":1,"base_model":"conformance-v1","created_at":"2026-07-31","operations":[{ops}]}}"#
    )
}

fn insert_op(feature: usize, gate_b64: &str) -> String {
    format!(
        r#"{{"op":"insert","layer":0,"feature":{feature},"entity":"Acme","target":"London","gate_vector_b64":"{gate_b64}"}}"#
    )
}

#[test]
fn corrupt_json_vlp_fails_load() {
    let tmp = tempfile::tempdir().unwrap();
    let path = write_vlp(tmp.path(), "{this is not json");
    assert!(VindexPatch::load(&path).is_err());
}

#[test]
fn unknown_op_tag_fails_load() {
    let tmp = tempfile::tempdir().unwrap();
    let path = write_vlp(
        tmp.path(),
        &patch_json(r#"{"op":"sabotage","layer":0,"feature":0}"#),
    );
    let err = VindexPatch::load(&path).expect_err("unknown op errors");
    assert!(err.to_string().contains("sabotage"), "{err}");
}

/// The M5 contract: a patch containing one corrupt embedded vector is
/// rejected wholesale — the valid ops before it must NOT have applied.
#[test]
fn corrupt_base64_vector_rejects_patch_without_half_applying() {
    let tmp = tempfile::tempdir().unwrap();
    save_minimal_vindex(tmp.path());
    let base = load_dir(tmp.path()).unwrap();
    let good = encode_gate_vector(&[0.5f32; FIXTURE_HIDDEN]);
    let ops = format!("{},{}", insert_op(0, &good), insert_op(1, "!!!!"));
    let path = write_vlp(tmp.path(), &patch_json(&ops));

    let patch = VindexPatch::load(&path).expect("JSON itself is valid");
    let mut patched = PatchedVindex::new(base);
    assert!(patched.try_apply_patch(patch).is_err());
    assert_eq!(patched.num_overrides(), 0, "no half-applied ops");
}

/// Truncated base64 (the silently-droppable 4k-chunk prefix) is an
/// error, not a shorter vector.
#[test]
fn truncated_base64_vector_rejects_patch() {
    let tmp = tempfile::tempdir().unwrap();
    save_minimal_vindex(tmp.path());
    let base = load_dir(tmp.path()).unwrap();
    let full = encode_gate_vector(&[1.0f32; FIXTURE_HIDDEN]);
    let truncated = &full[..full.len() - 2];
    let path = write_vlp(tmp.path(), &patch_json(&insert_op(0, truncated)));

    let patch = VindexPatch::load(&path).unwrap();
    let mut patched = PatchedVindex::new(base);
    assert!(patched.try_apply_patch(patch).is_err());
    assert_eq!(patched.num_overrides(), 0);
}

/// Positive control: a well-formed patch loads and applies.
#[test]
fn valid_patch_applies() {
    let tmp = tempfile::tempdir().unwrap();
    save_minimal_vindex(tmp.path());
    let base = load_dir(tmp.path()).unwrap();
    let good = encode_gate_vector(&[0.25f32; FIXTURE_HIDDEN]);
    let path = write_vlp(tmp.path(), &patch_json(&insert_op(0, &good)));

    let patch = VindexPatch::load(&path).unwrap();
    let mut patched = PatchedVindex::new(base);
    patched.try_apply_patch(patch).expect("valid patch applies");
    assert!(patched.num_overrides() > 0);
}

// ── knn_store.bin / .lknn ───────────────────────────────────────────
// Byte-level corruption grid lives in `src/patch/knn_store_io.rs`
// tests; these pin the public-API contract.

fn lknn(dir: &std::path::Path, bytes: &[u8]) -> std::path::PathBuf {
    let path = dir.join("knn_store.bin");
    std::fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn knn_store_bad_magic_is_error() {
    let tmp = tempfile::tempdir().unwrap();
    let path = lknn(tmp.path(), b"XXXX\x01\x00\x00\x00");
    let err = KnnStore::load(&path).unwrap_err();
    assert!(err.contains("bad magic"), "{err}");
}

#[test]
fn knn_store_truncated_is_error() {
    let tmp = tempfile::tempdir().unwrap();
    let path = lknn(tmp.path(), b"LKNN\x01\x00");
    assert!(KnnStore::load(&path).is_err());
}

/// A header declaring u32::MAX entries fails with a read error —
/// quickly, without attempting the multi-GB allocation the count
/// implies (the .lknn allocation-bomb pin).
#[test]
fn knn_store_absurd_count_is_error_not_allocation() {
    let tmp = tempfile::tempdir().unwrap();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"LKNN");
    bytes.extend_from_slice(&1u32.to_le_bytes()); // version
    bytes.extend_from_slice(&4u32.to_le_bytes()); // dim
    bytes.extend_from_slice(&1u32.to_le_bytes()); // n_layers
    bytes.extend_from_slice(&0u32.to_le_bytes()); // layer id
    bytes.extend_from_slice(&u32::MAX.to_le_bytes()); // n_entries
    let path = lknn(tmp.path(), &bytes);
    assert!(KnnStore::load(&path).is_err());
}
