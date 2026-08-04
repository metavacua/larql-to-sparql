//! Colocated tests for `generation` — schema-to-generation dispatch.
//!
//! The mapping is **many-to-one**, and an earlier version that assumed a
//! bijection refused every legacy-schema index in existence. These tests pin
//! the table rather than the arithmetic, so a future "simplification" back to
//! `version == generation` fails loudly.

use super::generation::{
    detect_generation, generation_for_schema, supported_schema_summary, ContainerGeneration,
    IndexSchemaVersion, ALL_GENERATIONS, V2_CURRENT_SCHEMA, V2_MIN_SCHEMA, V3_CURRENT_SCHEMA,
};
use crate::format::filenames::INDEX_JSON;
use crate::VindexError;

fn schema(v: u32) -> IndexSchemaVersion {
    IndexSchemaVersion(v)
}

// ── The pinned table ───────────────────────────────────────────────────────

#[test]
fn schema_one_is_the_shipped_generation_with_defaults() {
    assert_eq!(
        generation_for_schema(schema(1)).unwrap(),
        ContainerGeneration::V2
    );
}

#[test]
fn schema_two_is_the_shipped_generation() {
    assert_eq!(
        generation_for_schema(schema(2)).unwrap(),
        ContainerGeneration::V2
    );
}

#[test]
fn schema_three_is_the_successor() {
    assert_eq!(
        generation_for_schema(schema(3)).unwrap(),
        ContainerGeneration::V3
    );
}

#[test]
fn schema_zero_is_unsupported_and_names_the_supported_sets() {
    let err = generation_for_schema(schema(0)).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("1-2 (VINDEX2)"), "{text}");
    assert!(text.contains("3 (VINDEX3)"), "{text}");
}

#[test]
fn schema_four_is_unsupported_and_names_the_supported_sets() {
    let err = generation_for_schema(schema(4)).unwrap_err();
    let text = err.to_string();
    assert!(text.contains('4'), "{text}");
    assert!(text.contains("VINDEX2"), "{text}");
    assert!(text.contains("VINDEX3"), "{text}");
}

// ── Schema and generation are different things ─────────────────────────────

#[test]
fn a_generation_spans_more_schemas_than_it_writes() {
    // The model error the E0 regression exposed, as an assertion.
    let v2 = ContainerGeneration::V2;
    assert_eq!(v2.current_schema_version(), schema(V2_CURRENT_SCHEMA));
    assert!(v2.reads_schema(schema(V2_MIN_SCHEMA)));
    assert!(v2.reads_schema(schema(V2_CURRENT_SCHEMA)));
    assert_ne!(
        v2.supported_schema_versions().count(),
        1,
        "VINDEX2 reads more than one schema; a bijection would be wrong"
    );
}

#[test]
fn the_successor_currently_reads_exactly_one_schema() {
    let v3 = ContainerGeneration::V3;
    assert_eq!(v3.current_schema_version(), schema(V3_CURRENT_SCHEMA));
    assert_eq!(v3.supported_schema_versions().count(), 1);
}

#[test]
fn no_two_generations_claim_the_same_schema() {
    // Overlap would make dispatch ambiguous and the "sole discriminator"
    // property false.
    for a in ALL_GENERATIONS {
        for b in ALL_GENERATIONS {
            if a == b {
                continue;
            }
            let overlap = a
                .supported_schema_versions()
                .any(|v| b.reads_schema(schema(v)));
            assert!(!overlap, "{} and {} overlap", a.name(), b.name());
        }
    }
}

#[test]
fn every_generation_reads_the_schema_it_writes() {
    for g in ALL_GENERATIONS {
        assert!(g.reads_schema(g.current_schema_version()), "{}", g.name());
    }
}

#[test]
fn the_lyrw_version_trails_the_current_schema_by_one() {
    for g in ALL_GENERATIONS {
        assert_eq!(
            g.lyrw_format_version() + 1,
            g.current_schema_version().get(),
            "{}",
            g.name()
        );
    }
}

#[test]
fn lyrw_versions_map_back_to_their_generation() {
    for g in ALL_GENERATIONS {
        assert_eq!(
            ContainerGeneration::from_lyrw_format_version(g.lyrw_format_version()).unwrap(),
            g
        );
    }
    assert!(ContainerGeneration::from_lyrw_format_version(9).is_err());
}

// ── Unified dispatch versus direct-loader refusal ──────────────────────────

#[test]
fn unified_dispatch_routes_a_legacy_schema_to_the_shipped_generation() {
    assert_eq!(
        generation_for_schema(schema(1)).unwrap(),
        ContainerGeneration::V2
    );
}

#[test]
fn the_successor_loader_refuses_a_legacy_schema_by_name() {
    // Dispatch accepts it; the wrong loader must not.
    let found = generation_for_schema(schema(1)).unwrap();
    let err = found.require(ContainerGeneration::V3).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("VINDEX2"), "{text}");
    assert!(text.contains("VINDEX3"), "{text}");
}

#[test]
fn a_matching_loader_accepts() {
    assert!(ContainerGeneration::V2
        .require(ContainerGeneration::V2)
        .is_ok());
}

#[test]
fn the_summary_lists_ranges_and_singletons_differently() {
    let s = supported_schema_summary();
    assert!(s.contains("1-2 (VINDEX2)"), "{s}");
    assert!(s.contains("3 (VINDEX3)"), "{s}");
}

// ── Detection from disk ────────────────────────────────────────────────────

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir()
        .join("vindex-generation-tests")
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn detection_reads_only_the_version_field() {
    // A VINDEX3 index.json carries keys the shipped config struct has never
    // seen; detection must report the generation rather than fail on shape.
    let dir = temp_dir("rich-v3");
    std::fs::write(
        dir.join(INDEX_JSON),
        r#"{"version": 3, "profiles": ["exact"], "segments": {"routed/layer_0": 2}}"#,
    )
    .unwrap();
    assert_eq!(detect_generation(&dir).unwrap(), ContainerGeneration::V3);
}

#[test]
fn a_missing_version_field_is_refused_naming_what_is_supported() {
    let dir = temp_dir("no-version");
    std::fs::write(dir.join(INDEX_JSON), r#"{"model": "x"}"#).unwrap();
    let err = detect_generation(&dir).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("no version field"), "{text}");
    assert!(text.contains("VINDEX2"), "{text}");
}

#[test]
fn a_missing_index_json_is_io_not_a_generation_verdict() {
    let dir = temp_dir("absent");
    assert!(matches!(detect_generation(&dir), Err(VindexError::Io(_))));
}

#[test]
fn malformed_json_is_a_parse_error() {
    let dir = temp_dir("malformed");
    std::fs::write(dir.join(INDEX_JSON), "{not json").unwrap();
    assert!(matches!(
        detect_generation(&dir),
        Err(VindexError::Parse(_))
    ));
}

#[test]
fn a_schema_version_displays_as_its_number() {
    assert_eq!(schema(3).to_string(), "3");
    assert_eq!(schema(3).get(), 3);
}
