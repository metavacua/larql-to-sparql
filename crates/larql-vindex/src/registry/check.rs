//! Validate a `registry/` directory on disk — the filesystem-reading
//! counterpart to [`super::embedded`]'s compile-time `include_str!` path.
//!
//! # Why this exists (R3B)
//!
//! `larql registry check <PATH>` needs to validate a checked-out PR's
//! `registry/index.json` + `registry/models/*.json` — real files on disk,
//! not yet baked into any binary. The load-bearing choice: it reuses
//! [`assemble_registry`], the exact core `production_registry()` panics
//! behind for the embedded case, rather than a second parallel checker.
//! A CI gate whose definition of "valid registry" could drift from the
//! runtime's own definition is worse than no gate — it can pass while
//! the real resolver would refuse, or refuse while the real resolver
//! would accept.
//!
//! # No path argument
//!
//! `larql registry check` with no path validates the *embedded*
//! registry instead — see [`super::embedded::load_production_registry`].
//! For a CI build, that's equivalent to checking the same files this
//! module reads, since `include_str!` embeds whatever was checked out
//! at compile time; the two entry points exist because R3D's future
//! promotion workflow needs to check a *candidate* directory that may
//! not be `registry/` itself yet, before it's moved into place.

use std::collections::BTreeMap;
use std::path::Path;

use super::embedded::{assemble_registry, parse_index};
use super::error::RegistryError;
use super::manifest::RegistryManifest;

/// Filename of the table-of-contents file within a registry directory.
const INDEX_FILE: &str = "index.json";
/// Subdirectory holding one JSON file per model.
const MODELS_DIR: &str = "models";

/// Read and validate a registry directory laid out as
/// `<dir>/index.json` + `<dir>/models/<name>.json`.
///
/// The model filename is derived from the index-listed name
/// (`models/<name>.json`), never read from a separate field inside the
/// model's own JSON — there is no independent "manifest's own name" to
/// drift out of sync with the index, the same single-source-of-truth
/// choice the schema already makes for every other fact ([`super::manifest`]).
/// A name the index lists with no matching file is therefore reported
/// as exactly that: a missing file at the path this function expected,
/// not a generic IO error.
pub fn load_registry_from_dir(dir: &Path) -> Result<RegistryManifest, RegistryError> {
    let index_path = dir.join(INDEX_FILE);
    let index_json =
        std::fs::read_to_string(&index_path).map_err(|e| RegistryError::MalformedManifest {
            reason: format!("reading {}: {e}", index_path.display()),
        })?;
    let index = parse_index(&index_json)?;

    // Every model file is read eagerly into an owned map before
    // `assemble_registry` runs: its `lookup` closure hands back a
    // borrow, and a closure can't return a borrow of bytes it would
    // otherwise have to allocate fresh on each call.
    let mut files = BTreeMap::new();
    for name in &index.models {
        let model_path = dir.join(MODELS_DIR).join(format!("{name}.json"));
        let text =
            std::fs::read_to_string(&model_path).map_err(|e| RegistryError::MalformedManifest {
                reason: format!(
                    "registry/index.json names model '{name}', expected at {}: {e}",
                    model_path.display()
                ),
            })?;
        files.insert(name.clone(), text);
    }

    assemble_registry(&index_json, |name| files.get(name).map(String::as_str))
}
