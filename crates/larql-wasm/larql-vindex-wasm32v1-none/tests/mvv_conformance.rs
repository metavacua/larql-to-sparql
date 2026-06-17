//! Conformance + adversarial suite for the Minimal-Viable Vindex (MVV) kernel.
//!
//! An integration test against the public API (`index::mvv`), so it compiles
//! the library normally and is isolated from the crate's native-only inline
//! test modules. Beyond the happy path, every adversarial case asserts a
//! *typed* `MvvError` and — by completing without panicking — that the kernel
//! is total (no panic, no UB) on malformed input.
use larql_vindex_wasm32v1_none::index::mvv::{MvvError, MvvIndex};

// ── helpers ──────────────────────────────────────────────────────────────────

fn f32_blob(rows: &[&[f32]]) -> Vec<u8> {
    let mut b = Vec::new();
    for r in rows {
        for &x in r.iter() {
            b.extend_from_slice(&x.to_le_bytes());
        }
    }
    b
}

/// index.json for a single f32 layer; `length` lets a test inject a
/// deliberately-wrong declared length.
fn index_json_f32(num_features: usize, hidden: usize, length: usize) -> String {
    format!(
        r#"{{"version":2,"num_layers":1,"hidden_size":{hidden},"dtype":"f32",
            "layers":[{{"layer":0,"num_features":{num_features},"offset":0,"length":{length}}}]}}"#
    )
}

// ── happy path ───────────────────────────────────────────────────────────────

#[test]
fn gate_knn_ranks_by_absolute_dot_product() {
    let blob = f32_blob(&[&[1.0, 0.0, 0.0], &[0.0, 2.0, 0.0]]);
    let json = index_json_f32(2, 3, blob.len());
    let idx = MvvIndex::from_index_json(json.as_bytes(), blob).unwrap();

    let out = idx.gate_knn(0, &[0.5, 1.0, 0.0], 2).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].0, 1);
    assert!((out[0].1 - 2.0).abs() < 1e-6);
    assert_eq!(out[1].0, 0);
    assert!((out[1].1 - 0.5).abs() < 1e-6);
}

#[test]
fn gate_knn_respects_top_k_truncation() {
    let blob = f32_blob(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]]);
    let json = index_json_f32(3, 2, blob.len());
    let idx = MvvIndex::from_index_json(json.as_bytes(), blob).unwrap();
    let out = idx.gate_knn(0, &[1.0, 1.0], 1).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].0, 2);
}

#[test]
fn negative_dot_ranks_by_magnitude() {
    let blob = f32_blob(&[&[0.1, 0.0], &[-5.0, 0.0]]);
    let json = index_json_f32(2, 2, blob.len());
    let idx = MvvIndex::from_index_json(json.as_bytes(), blob).unwrap();
    let out = idx.gate_knn(0, &[1.0, 0.0], 2).unwrap();
    assert_eq!(out[0].0, 1);
    assert!((out[0].1 - (-5.0)).abs() < 1e-6);
}

#[test]
fn expert_range_scopes_features() {
    let blob = f32_blob(&[&[1.0, 0.0], &[2.0, 0.0], &[3.0, 0.0]]);
    let json = index_json_f32(3, 2, blob.len());
    let idx = MvvIndex::from_index_json(json.as_bytes(), blob).unwrap();
    let out = idx.gate_knn_expert(0, &[1.0, 0.0], 1, 3, 5).unwrap();
    assert_eq!(out.len(), 2);
    assert!(out.iter().all(|(f, _)| *f == 1 || *f == 2));
}

#[test]
fn num_features_total_and_metadata_absent() {
    let blob = f32_blob(&[&[1.0, 0.0]]);
    let json = index_json_f32(1, 2, blob.len());
    let idx = MvvIndex::from_index_json(json.as_bytes(), blob).unwrap();
    assert_eq!(idx.num_features(0), 1);
    assert_eq!(idx.num_features(99), 0); // total, no panic
    assert!(idx.feature_meta(0, 0).is_none());
    assert!(idx.feature_meta(99, 99).is_none());
}

// ── adversarial / totality ───────────────────────────────────────────────────

#[test]
fn malformed_index_json_is_typed_error() {
    assert_eq!(
        MvvIndex::from_index_json(b"not json {", vec![]).err(),
        Some(MvvError::BadIndexJson)
    );
}

#[test]
fn unsupported_version_is_typed_error() {
    let json = r#"{"version":99,"num_layers":0,"hidden_size":2,"dtype":"f32","layers":[]}"#;
    assert!(matches!(
        MvvIndex::from_index_json(json.as_bytes(), vec![]),
        Err(MvvError::UnsupportedVersion(99))
    ));
}

#[test]
fn layer_count_mismatch_is_typed_error() {
    let json = r#"{"version":2,"num_layers":2,"hidden_size":1,"dtype":"f32",
        "layers":[{"layer":0,"num_features":1,"offset":0,"length":4}]}"#;
    assert!(matches!(
        MvvIndex::from_index_json(json.as_bytes(), vec![0; 4]),
        Err(MvvError::LayerCountMismatch { declared: 2, actual: 1 })
    ));
}

#[test]
fn declared_length_not_matching_dims_is_typed_error() {
    let json = index_json_f32(1, 2, 12); // expected 8
    assert!(matches!(
        MvvIndex::from_index_json(json.as_bytes(), vec![0; 12]),
        Err(MvvError::LayerLengthMismatch { layer: 0, expected: 8, declared: 12 })
    ));
}

#[test]
fn truncated_blob_is_typed_error() {
    let json = index_json_f32(1, 2, 8);
    assert!(matches!(
        MvvIndex::from_index_json(json.as_bytes(), vec![0; 4]),
        Err(MvvError::BlobLengthMismatch { .. })
    ));
}

#[test]
fn trailing_slack_blob_is_typed_error() {
    let json = index_json_f32(1, 2, 8);
    assert!(matches!(
        MvvIndex::from_index_json(json.as_bytes(), vec![0; 12]),
        Err(MvvError::BlobLengthMismatch { expected: 8, actual: 12 })
    ));
}

#[test]
fn dim_mismatch_residual_is_typed_error() {
    let blob = f32_blob(&[&[1.0, 0.0]]);
    let json = index_json_f32(1, 2, blob.len());
    let idx = MvvIndex::from_index_json(json.as_bytes(), blob).unwrap();
    assert!(matches!(
        idx.gate_knn(0, &[1.0, 0.0, 0.0], 1),
        Err(MvvError::DimMismatch { expected: 2, actual: 3 })
    ));
}

#[test]
fn layer_out_of_range_query_is_typed_error() {
    let blob = f32_blob(&[&[1.0, 0.0]]);
    let json = index_json_f32(1, 2, blob.len());
    let idx = MvvIndex::from_index_json(json.as_bytes(), blob).unwrap();
    assert!(matches!(
        idx.gate_knn(5, &[1.0, 0.0], 1),
        Err(MvvError::LayerOutOfRange { layer: 5, num_layers: 1 })
    ));
}

#[test]
fn feature_range_out_of_bounds_is_typed_error() {
    let blob = f32_blob(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let json = index_json_f32(2, 2, blob.len());
    let idx = MvvIndex::from_index_json(json.as_bytes(), blob).unwrap();
    assert!(matches!(
        idx.gate_knn_expert(0, &[1.0, 0.0], 1, 9, 5),
        Err(MvvError::FeatureRange { start: 1, end: 9, num_features: 2 })
    ));
}

#[test]
fn nan_residual_does_not_panic() {
    let blob = f32_blob(&[&[1.0, 0.0], &[0.0, 1.0]]);
    let json = index_json_f32(2, 2, blob.len());
    let idx = MvvIndex::from_index_json(json.as_bytes(), blob).unwrap();
    let out = idx.gate_knn(0, &[f32::NAN, 1.0], 2).unwrap();
    assert_eq!(out.len(), 2); // defined output, no panic
}

#[test]
fn f16_layer_round_trips() {
    // f16 little-endian: 3.0 = 0x4200 → [00,42]; 4.0 = 0x4400 → [00,44].
    let blob = vec![0x00u8, 0x42, 0x00, 0x44]; // 1 feature, hidden 2, 4 bytes
    let json = r#"{"version":2,"num_layers":1,"hidden_size":2,"dtype":"f16",
        "layers":[{"layer":0,"num_features":1,"offset":0,"length":4}]}"#;
    let idx = MvvIndex::from_index_json(json.as_bytes(), blob).unwrap();
    let out = idx.gate_knn(0, &[1.0, 1.0], 1).unwrap();
    assert_eq!(out[0].0, 0);
    assert!((out[0].1 - 7.0).abs() < 0.05); // 3+4 within f16 tolerance
}
