//! Byte-equivalence gates: both ends re-hashed, both failure stories
//! distinguished, and the relabel case hashes cannot see.

use std::io::{Read, Seek, SeekFrom, Write};

use super::encoded_glimmer_system;
use crate::format::vindex3::encode::segment::read_segment_header;
use crate::format::vindex3::encode::SEGMENTS_DIR;
use crate::format::vindex3::verify_system::{verify_system, Authority};

/// THE G4 gate, byte half: every representation's three hashes agree on an
/// untouched encode.
#[test]
fn every_representation_is_byte_equal_on_an_untouched_encode() {
    let system = encoded_glimmer_system();
    let verification = verify_system(&system.named, system.container.path()).unwrap();
    assert!(verification.payloads.len() >= 7);
    for check in &verification.payloads {
        assert!(check.pass, "{}: {}", check.representation, check.detail);
        assert_eq!(check.source_sha256, check.recorded_sha256);
        assert_eq!(check.encoded_sha256, check.recorded_sha256);
        assert!(check.payload_bytes > 0);
    }
}

/// Container corruption: a flipped payload byte fails exactly the
/// container side, and the detail says so.
#[test]
fn a_flipped_container_byte_names_the_container() {
    let system = encoded_glimmer_system();
    let victim = system
        .container
        .path()
        .join(SEGMENTS_DIR)
        .join("target.embedding.bin");
    let mut bytes = std::fs::read(&victim).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    std::fs::write(&victim, bytes).unwrap();

    let verification = verify_system(&system.named, system.container.path()).unwrap();
    assert!(!verification.verified);
    let failure = verification
        .payload_failures()
        .find(|p| p.representation.starts_with("target.embedding@"))
        .expect("the corrupted representation must be named");
    assert_eq!(failure.source_sha256, failure.recorded_sha256);
    assert_ne!(failure.encoded_sha256, failure.recorded_sha256);
    assert!(
        failure.detail.contains("container changed"),
        "{}",
        failure.detail
    );
}

/// Source drift: a flipped checkpoint byte fails exactly the source side.
/// `inspect --verify` can never catch this — it has no source access.
#[test]
fn a_flipped_source_byte_names_the_checkpoint() {
    let system = encoded_glimmer_system();
    let shard = system._target_dir.path().join("model.safetensors");
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&shard)
        .unwrap();
    file.seek(SeekFrom::End(-1)).unwrap();
    let mut byte = [0u8; 1];
    file.read_exact(&mut byte).unwrap();
    byte[0] ^= 0x01;
    file.seek(SeekFrom::End(-1)).unwrap();
    file.write_all(&byte).unwrap();

    let verification = verify_system(&system.named, system.container.path()).unwrap();
    assert!(!verification.verified);
    let failure = verification
        .payload_failures()
        .next()
        .expect("some representation must fail");
    assert_ne!(failure.source_sha256, failure.recorded_sha256);
    assert_eq!(failure.encoded_sha256, failure.recorded_sha256);
    assert!(
        failure.detail.contains("checkpoint changed"),
        "{}",
        failure.detail
    );
}

/// A deleted segment is a failing row, not a panic, and the sweep
/// continues to the remaining representations.
#[test]
fn a_missing_segment_is_a_finding_not_an_abort() {
    let system = encoded_glimmer_system();
    std::fs::remove_file(
        system
            .container
            .path()
            .join(SEGMENTS_DIR)
            .join("target.embedding.bin"),
    )
    .unwrap();

    let verification = verify_system(&system.named, system.container.path()).unwrap();
    assert!(!verification.verified);
    assert!(verification
        .payload_failures()
        .any(|p| p.representation.starts_with("target.embedding@")
            && p.detail.contains("unreadable")));
    // The rest of the sweep still ran.
    assert!(verification.payloads.iter().filter(|p| p.pass).count() >= 6);
}

/// The pre-hash failure rows, exercised directly against a mutated graph:
/// an object with no representation, one with no directory entry, and one
/// whose bindings no longer plan against the verify set.
#[test]
fn pre_hash_failures_are_rows_not_errors() {
    let system = encoded_glimmer_system();
    let plan = crate::format::vindex3::plan::plan_system(&system.named);
    let inspection =
        crate::format::vindex3::inspect::inspect_container(system.container.path(), false).unwrap();

    let mut graph = plan.graph.clone();
    graph.objects[0].representations.clear();
    graph.objects[1].id = "target.ghost_object".to_string();
    for binding in &mut graph.objects[2].source_bindings {
        binding.artifact = "ghost-artifact".to_string();
    }
    let (payloads, _) = crate::format::vindex3::verify_system::payload::verify_payloads(
        &system.named,
        &graph,
        &inspection.index,
        system.container.path(),
    )
    .unwrap();
    let detail_of = |needle: &str| {
        payloads
            .iter()
            .find(|p| p.detail.contains(needle))
            .unwrap_or_else(|| panic!("no row with detail `{needle}`: {payloads:?}"))
    };
    assert!(!detail_of("no representation").pass);
    assert!(!detail_of("no directory entry").pass);
    assert!(!detail_of("source no longer plans").pass);
    // The untouched objects still verified in the same sweep.
    assert!(payloads.iter().filter(|p| p.pass).count() >= 4);
}

/// The relabel no hash can see: rewrite one stored shape to a different
/// factorisation of the same element count, byte length preserved. Every
/// hash stays equal; only the tensor-table link may catch it.
#[test]
fn a_shape_relabel_over_identical_bytes_is_caught_by_the_table() {
    let system = encoded_glimmer_system();
    let victim = system
        .container
        .path()
        .join(SEGMENTS_DIR)
        .join("target.embedding.bin");
    // Serialized forms `[128,64]` and `[64,128]` are byte-length equal, so
    // the header length prefix — and every recorded hash input except the
    // header bytes themselves — is unchanged. The first occurrence is in
    // the header: the payload region is binary pattern bytes.
    let mut bytes = std::fs::read(&victim).unwrap();
    let (header, _) = read_segment_header(&victim).unwrap();
    assert!(header.tensors.iter().any(|t| t.shape == vec![128, 64]));
    let needle = b"[128,64]";
    let replacement = b"[64,128]";
    let at = bytes
        .windows(needle.len())
        .position(|w| w == needle)
        .expect("fixture shape not found in header");
    bytes[at..at + replacement.len()].copy_from_slice(replacement);
    std::fs::write(&victim, bytes).unwrap();

    let verification = verify_system(&system.named, system.container.path()).unwrap();
    assert!(!verification.verified);
    // Payload hashes all still pass — the bytes are the bytes.
    assert!(
        verification
            .payloads
            .iter()
            .find(|p| p.representation.starts_with("target.embedding@"))
            .unwrap()
            .pass
    );
    let failure = verification
        .semantic_failures()
        .find(|c| c.subject.starts_with("target.embedding@") && c.subject.ends_with("tensor_table"))
        .expect("the relabelled table must fail its Declared ≡ Encoded link");
    assert_eq!(failure.left, Authority::Declared);
    assert_eq!(failure.right, Authority::Encoded);
}
