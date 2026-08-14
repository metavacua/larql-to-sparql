//! Encode → inspect round trip: the G3 gate, on the fixture system.

use std::io::{Read, Seek, SeekFrom};

use crate::format::vindex3::encode::segment::read_segment_header;
use crate::format::vindex3::encode::{encode_system, SEGMENTS_DIR, SYSTEM_GRAPH_JSON};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::plan::tests_support::{
    drafter_shaped, glimmer_shaped_target, known_dense, payload_pattern,
};

fn glimmer_system() -> (
    tempfile::TempDir,
    tempfile::TempDir,
    Vec<(String, larql_models::inventory::ArchitectureInventory)>,
) {
    let target_dir = tempfile::tempdir().unwrap();
    let drafter_dir = tempfile::tempdir().unwrap();
    let named = vec![
        (
            "target-artifact".to_string(),
            glimmer_shaped_target(target_dir.path()),
        ),
        (
            "drafter-artifact".to_string(),
            drafter_shaped(drafter_dir.path()),
        ),
    ];
    (target_dir, drafter_dir, named)
}

/// THE G3 gate: encode the pair, then reconstruct the system solely from
/// the container — components, topology (incl. NoPE), the edge — with a
/// coherent directory and no source access.
#[test]
fn encode_then_inspect_reconstructs_the_system_without_the_source() {
    let (_a, _b, named) = glimmer_system();
    let out = tempfile::tempdir().unwrap();
    let outcome = encode_system(&named, out.path()).unwrap();
    assert!(outcome.representations >= 7);
    assert!(outcome.total_payload_bytes > 0);

    // Sources gone: move nothing, read nothing — inspect uses only `out`.
    let inspection = inspect_container(out.path(), true).unwrap();
    assert!(
        inspection.is_coherent(),
        "defects: {:?}",
        inspection.defects
    );

    let target = inspection
        .components
        .iter()
        .find(|c| c.id == "target")
        .unwrap();
    assert_eq!(target.num_layers, 8);
    assert_eq!(target.hidden_size, 64);
    assert_eq!(target.sliding_layers, Some(6));
    assert_eq!(target.full_layers, Some(2));
    assert_eq!(target.nope_layers, Some(2));
    assert_eq!(target.window, Some(16));

    let draft = inspection
        .components
        .iter()
        .find(|c| c.id == "draft")
        .unwrap();
    assert_eq!(draft.num_layers, 2);

    assert_eq!(inspection.graph.edges.len(), 1);
    let edge = &inspection.graph.edges[0];
    assert_eq!(edge.producer_layers, vec![1, 3, 5]);
    assert_eq!(edge.block_size, Some(4));
    assert_eq!(edge.consumer_object, "draft.feature_projector");
}

/// Payload bytes survive the trip exactly: read a tensor back out of its
/// segment via the header table and compare with the deterministic source
/// pattern.
#[test]
fn payload_bytes_round_trip_exactly() {
    let (_a, _b, named) = glimmer_system();
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();

    // The drafter's projector: encoder.fc.weight is 192*64 BF16 = 24576
    // bytes at source offsets [8320, 32896).
    let segment_path = out
        .path()
        .join(SEGMENTS_DIR)
        .join("draft.feature_projector.bin");
    let (header, payload_start) = read_segment_header(&segment_path).unwrap();
    let entry = header
        .tensors
        .iter()
        .find(|t| t.name == "fc.weight")
        .expect("object-relative name for the fusion tensor");
    assert_eq!(entry.shape, vec![192, 64]);

    let mut file = std::fs::File::open(&segment_path).unwrap();
    file.seek(SeekFrom::Start(payload_start + entry.offset))
        .unwrap();
    let mut encoded = vec![0u8; entry.len as usize];
    file.read_exact(&mut encoded).unwrap();

    // Source pattern over the whole shard payload region, sliced at the
    // tensor's declared source offsets.
    let source_slice = &payload_pattern(32896)[8320..32896];
    assert_eq!(encoded, source_slice, "payload bytes differ from source");
}

/// Object-relative names carry no artifact-global prefixes.
#[test]
fn segment_tensor_names_are_object_relative() {
    let (_a, _b, named) = glimmer_system();
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();
    let (header, _) = read_segment_header(
        &out.path()
            .join(SEGMENTS_DIR)
            .join("target.decoder_stack.bin"),
    )
    .unwrap();
    for tensor in &header.tensors {
        assert!(
            !tensor.name.starts_with("model."),
            "artifact-global name leaked: {}",
            tensor.name
        );
    }
    // Multi-binding object: names stay unique after prefix stripping.
    let (header, _) = read_segment_header(
        &out.path()
            .join(SEGMENTS_DIR)
            .join("draft.feature_projector.bin"),
    )
    .unwrap();
    let names: Vec<&str> = header.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, vec!["fc.weight", "output_norm_enc.weight"]);
}

/// Encoding is deterministic: same inputs, same hashes.
#[test]
fn encode_is_deterministic() {
    let (_a, _b, named) = glimmer_system();
    let out1 = tempfile::tempdir().unwrap();
    let out2 = tempfile::tempdir().unwrap();
    encode_system(&named, out1.path()).unwrap();
    encode_system(&named, out2.path()).unwrap();
    let read = |p: &std::path::Path| {
        std::fs::read_to_string(p.join(crate::format::filenames::INDEX_JSON)).unwrap()
    };
    assert_eq!(read(out1.path()), read(out2.path()));
}

/// An inadmissible plan is refused before a single byte is written.
#[test]
fn inadmissible_plan_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut inventory = known_dense(dir.path());
    inventory
        .config_keys
        .push(larql_models::inventory::ConfigKeyFact {
            path: "some_future_field_nobody_reviewed".to_string(),
            value: serde_json::json!(42),
            status: larql_models::inventory::KeyStatus::Unconsumed,
        });
    let named = vec![("llama-artifact".to_string(), inventory)];
    let out = tempfile::tempdir().unwrap();
    let err = encode_system(&named, out.path()).unwrap_err();
    assert!(err.to_string().contains("inadmissible"), "{err}");
    assert!(
        !out.path()
            .join(crate::format::filenames::INDEX_JSON)
            .exists(),
        "a refused encode must not leave an index behind"
    );
}

/// A corrupted segment byte is caught by `inspect --verify` — with no
/// source access.
#[test]
fn verify_catches_a_flipped_payload_byte() {
    let (_a, _b, named) = glimmer_system();
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();

    let victim = out.path().join(SEGMENTS_DIR).join("target.embedding.bin");
    let mut bytes = std::fs::read(&victim).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&victim, bytes).unwrap();

    let clean = inspect_container(out.path(), false).unwrap();
    assert!(
        clean.is_coherent(),
        "structure still coherent without verify"
    );
    let verified = inspect_container(out.path(), true).unwrap();
    assert!(!verified.is_coherent());
    assert!(verified
        .defects
        .iter()
        .any(|d| format!("{d:?}").contains("target.embedding")));
}

/// The graph manifest is written verbatim and reloads as the same graph.
#[test]
fn graph_manifest_round_trips() {
    let (_a, _b, named) = glimmer_system();
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();
    let graph: crate::format::vindex3::graph::SystemGraph =
        serde_json::from_str(&std::fs::read_to_string(out.path().join(SYSTEM_GRAPH_JSON)).unwrap())
            .unwrap();
    assert!(graph.validate().is_empty());
    assert_eq!(graph.edges.len(), 1);
}

/// A known dense model encodes and reconstructs the same way — the path is
/// generic, not Glimmer-shaped.
#[test]
fn known_dense_encodes_and_inspects() {
    let dir = tempfile::tempdir().unwrap();
    let named = vec![("llama-artifact".to_string(), known_dense(dir.path()))];
    let out = tempfile::tempdir().unwrap();
    encode_system(&named, out.path()).unwrap();
    let inspection = inspect_container(out.path(), true).unwrap();
    assert!(inspection.is_coherent(), "{:?}", inspection.defects);
    assert!(inspection.graph.edges.is_empty());
    assert!(inspection
        .index
        .representations
        .keys()
        .any(|k| k.starts_with("target.embedding@")));
}
