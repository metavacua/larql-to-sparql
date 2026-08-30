//! `larql registry <subcommand>` — validate the VINDEX3 production
//! registry (R3B).
//!
//! `check` is the only verb today. It reuses the exact assembly/
//! validation code `production_registry()` panics behind
//! (`larql_vindex::registry::{load_production_registry, load_registry_from_dir}`)
//! rather than a second, parallel checker — CI's definition of "valid
//! registry" can never drift from the runtime's own definition, which a
//! Python JSON schema check run alongside it could.
//!
//! A future `larql registry promote` (R3D) is the reason this is a
//! subcommand group already, not a bare top-level `larql registry-check`
//! — the promotion workflow ends with exactly this same check:
//! `write model JSON -> larql registry check -> git diff -> PR`.

use std::path::PathBuf;

use clap::{Args, Subcommand};

#[derive(Subcommand)]
pub enum RegistryCommand {
    /// Validate `index.json` + `models/*.json` — schema, pinned
    /// revisions, a valid attestation, every referenced model file
    /// present and parseable, `default_variant` resolvable, ABI
    /// supported.
    Check(CheckArgs),
}

#[derive(Args)]
pub struct CheckArgs {
    /// Registry directory to validate (`index.json` + `models/*.json`).
    /// Omit to validate the registry embedded in this binary — for a
    /// CI build, that's exactly the checked-out PR's own `registry/`
    /// files, since `include_str!` embeds whatever was present at
    /// compile time.
    pub path: Option<PathBuf>,
}

pub fn run(cmd: RegistryCommand) -> Result<(), Box<dyn std::error::Error>> {
    match cmd {
        RegistryCommand::Check(args) => check(args),
    }
}

fn check(args: CheckArgs) -> Result<(), Box<dyn std::error::Error>> {
    let manifest = match &args.path {
        Some(path) => larql_vindex::registry::load_registry_from_dir(path)?,
        None => larql_vindex::registry::load_production_registry()?,
    };

    let source = args
        .path
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "(embedded in this binary)".to_string());

    println!("Registry OK: {source}");
    println!(
        "  schema_version {}, {} model(s)",
        manifest.schema_version,
        manifest.models.len()
    );
    for (name, model) in &manifest.models {
        println!(
            "  {name} — default `{}`, {} variant(s)",
            model.default_variant,
            model.variants.len()
        );
        for (variant_name, variant) in &model.variants {
            println!(
                "    {variant_name}: {}@{} (ABI {})",
                variant.artifact.repo, variant.artifact.revision, variant.abi
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // The no-path (embedded) form is exercised at the larql-vindex layer
    // (`registry::embedded_tests`, `registry::check_tests`) against the
    // real checked-in registry/ — duplicating that here would just be
    // the same assertion against the same data through an extra layer.
    // These tests cover what's actually specific to the CLI wrapper: the
    // explicit-path dispatch, and that a validation failure propagates
    // as an `Err` (the caller's `Error: {e}` + exit 1 convention) rather
    // than a panic or a silently-swallowed exit 0.

    #[test]
    fn check_with_an_explicit_path_to_a_valid_registry_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "index.json",
            r#"{"schema_version": 1, "models": ["example"]}"#,
        );
        write(dir.path(), "models/example.json", VALID_MODEL_JSON);

        check(CheckArgs {
            path: Some(dir.path().to_path_buf()),
        })
        .expect("a well-formed registry directory must validate");
    }

    #[test]
    fn check_with_an_explicit_path_to_an_invalid_registry_errors() {
        let dir = tempfile::tempdir().unwrap();
        write(
            dir.path(),
            "index.json",
            r#"{"schema_version": 1, "models": ["example"]}"#,
        );
        let floating = VALID_MODEL_JSON.replace("\"abc123\"", "\"main\"");
        write(dir.path(), "models/example.json", &floating);

        let err = check(CheckArgs {
            path: Some(dir.path().to_path_buf()),
        })
        .expect_err("a floating revision must refuse, not print success");
        assert!(err.to_string().contains("main"));
    }

    #[test]
    fn check_with_a_missing_path_errors_naming_the_directory() {
        let missing = std::path::PathBuf::from("/does/not/exist/at/all");
        let err = check(CheckArgs {
            path: Some(missing),
        })
        .expect_err("a nonexistent registry directory must error, not panic");
        assert!(err.to_string().contains("index.json"));
    }
}
