//! Colocated tests for [`super::check`] — the filesystem-reading
//! registry validator `larql registry check <PATH>` (R3B) and CI both
//! call. Deliberately independent of the real embedded `registry/`
//! contents (see `embedded_tests` for those): a temp directory is
//! written fresh per test, proving the function against synthetic data
//! the same way `embedded::assemble_registry`'s own tests do.

use super::check::load_registry_from_dir;
use super::error::RegistryError;

const VALID_MODEL_JSON: &str = r#"{
    "default_variant": "bf16",
    "variants": {
        "bf16": {
            "artifact": {"repo": "larql/example", "revision": "abc123"},
            "abi": 1,
            "source": {
                "repo": "example/example",
                "revision": "def456",
                "attestation": {"kind": "mechanical"}
            }
        }
    }
}"#;

fn write(dir: &std::path::Path, rel: &str, contents: &str) {
    let path = dir.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, contents).unwrap();
}

#[test]
fn a_well_formed_registry_directory_loads_and_validates() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "index.json",
        r#"{"schema_version": 1, "models": ["example"]}"#,
    );
    write(dir.path(), "models/example.json", VALID_MODEL_JSON);

    let manifest = load_registry_from_dir(dir.path()).unwrap();
    assert!(manifest.models.contains_key("example"));
}

#[test]
fn a_missing_index_file_reports_which_path_it_looked_at() {
    let dir = tempfile::tempdir().unwrap();
    let err = load_registry_from_dir(dir.path()).unwrap_err();
    let RegistryError::MalformedManifest { reason } = err else {
        panic!("expected MalformedManifest, got {err:?}");
    };
    assert!(reason.contains("index.json"), "{reason}");
}

#[test]
fn malformed_index_json_refuses() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "index.json", "not json");

    let err = load_registry_from_dir(dir.path()).unwrap_err();
    assert!(matches!(err, RegistryError::MalformedManifest { .. }));
}

#[test]
fn a_model_the_index_names_with_no_file_reports_the_expected_path() {
    // The filename is derived from the index-listed name — never read
    // from a separate field inside the model's own JSON — so a missing
    // file names exactly the path this function expected, not a bare
    // "file not found."
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "index.json",
        r#"{"schema_version": 1, "models": ["ghost"]}"#,
    );

    let err = load_registry_from_dir(dir.path()).unwrap_err();
    let RegistryError::MalformedManifest { reason } = err else {
        panic!("expected MalformedManifest, got {err:?}");
    };
    assert!(reason.contains("ghost"), "{reason}");
    assert!(reason.contains("models"), "{reason}");
}

#[test]
fn malformed_model_json_refuses_naming_the_file() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "index.json",
        r#"{"schema_version": 1, "models": ["example"]}"#,
    );
    write(dir.path(), "models/example.json", "not json");

    let err = load_registry_from_dir(dir.path()).unwrap_err();
    let RegistryError::MalformedManifest { reason } = err else {
        panic!("expected MalformedManifest, got {err:?}");
    };
    assert!(reason.contains("example.json"), "{reason}");
}

#[test]
fn a_manifest_that_parses_but_fails_validation_refuses() {
    // An unpinned artifact revision — parses fine, but validate() must
    // still catch it. Proves this path actually validates, not just
    // parses.
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "index.json",
        r#"{"schema_version": 1, "models": ["example"]}"#,
    );
    let floating = VALID_MODEL_JSON.replace("\"abc123\"", "\"main\"");
    write(dir.path(), "models/example.json", &floating);

    let err = load_registry_from_dir(dir.path()).unwrap_err();
    assert!(matches!(err, RegistryError::UnpinnedRevision { .. }));
}

#[test]
fn a_registry_with_multiple_models_loads_them_all() {
    let dir = tempfile::tempdir().unwrap();
    write(
        dir.path(),
        "index.json",
        r#"{"schema_version": 1, "models": ["alpha", "beta"]}"#,
    );
    write(dir.path(), "models/alpha.json", VALID_MODEL_JSON);
    write(dir.path(), "models/beta.json", VALID_MODEL_JSON);

    let manifest = load_registry_from_dir(dir.path()).unwrap();
    assert_eq!(manifest.models.len(), 2);
    assert!(manifest.models.contains_key("alpha"));
    assert!(manifest.models.contains_key("beta"));
}

#[test]
fn the_real_checked_in_registry_directory_also_loads_through_this_path() {
    // The same real registry/ files embedded_tests proves against
    // production_registry(), read from disk instead — the two entry
    // points must agree on the one registry that actually exists.
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../registry")
        .canonicalize()
        .expect("crates/larql-vindex/../../registry must resolve to the repo-root registry/ dir");
    let manifest = load_registry_from_dir(&repo_root).unwrap();
    assert!(manifest.models.contains_key("granite-4.1-3b"));
}
