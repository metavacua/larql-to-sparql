//! G4 gate tests. The instrument rule applies: every claim the verifier can
//! make is exercised against a known-different input before its pass is
//! trusted (see `feedback_controls_before_parity`).

mod payload;
mod semantic;

use larql_models::inventory::ArchitectureInventory;

use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::plan::tests_support::{drafter_shaped, glimmer_shaped_target};

/// The fixture pair, encoded: source dirs, container dir, and the named
/// inventory set — everything a verify run needs.
pub(super) struct EncodedSystem {
    /// Held for lifetime only: dropping them deletes the source checkpoints.
    pub _target_dir: tempfile::TempDir,
    pub _drafter_dir: tempfile::TempDir,
    pub container: tempfile::TempDir,
    pub named: Vec<(String, ArchitectureInventory)>,
}

pub(super) fn encoded_glimmer_system() -> EncodedSystem {
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
    let container = tempfile::tempdir().unwrap();
    encode_system(&named, container.path()).unwrap();
    EncodedSystem {
        _target_dir: target_dir,
        _drafter_dir: drafter_dir,
        container,
        named,
    }
}
