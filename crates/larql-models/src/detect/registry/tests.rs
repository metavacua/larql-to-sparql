//! Cross-checks the registry against real `detect_from_json` dispatch.
//! Per-type unit tests for the pieces that don't need dispatch at all
//! live next to those types (`pattern.rs`, `entry.rs`, `table.rs`).

use super::*;
use crate::detect::detect_from_json;

const GENERIC_FAMILY: &str = "generic";

fn probe_for(pattern: ModelTypeMatch) -> String {
    match pattern {
        ModelTypeMatch::Exact(s) => s.to_string(),
        // Exercises the prefix behaviour, not just the bare prefix
        // string, in case a future match arm accidentally narrows to
        // an exact match.
        ModelTypeMatch::Prefix(p) => format!("{p}-probe-suffix"),
    }
}

#[test]
fn find_architecture_returns_none_for_unknown_model_types() {
    assert!(find_architecture("totally-unknown-arch").is_none());
    assert!(find_architecture("").is_none());
}

#[test]
fn find_architecture_returns_some_for_every_registered_pattern() {
    for entry in ARCHITECTURE_REGISTRY {
        for &pattern in entry.patterns {
            let probe = probe_for(pattern);
            let found = find_architecture(&probe);
            assert!(
                found.is_some(),
                "probe {probe:?} (from {}) should resolve via find_architecture",
                entry.model_type
            );
        }
    }
}

/// The load-bearing test: every pattern this registry declares must
/// actually be honored by the real `detect_from_json` dispatch — i.e.
/// must NOT silently fall back to `GenericArch`. This is what catches
/// a registry entry going stale if a `match` arm in `detect_from_json`
/// is reordered, narrowed, or removed.
#[test]
fn registry_entries_are_honored_by_detect_from_json() {
    for entry in ARCHITECTURE_REGISTRY {
        for &pattern in entry.patterns {
            let probe = probe_for(pattern);
            let config = serde_json::json!({ "model_type": probe });
            let arch = detect_from_json(&config);
            assert_ne!(
                arch.family(),
                GENERIC_FAMILY,
                "registry entry {} claims to match {probe:?}, but detect_from_json fell back to GenericArch",
                entry.model_type
            );
        }
    }
}

#[test]
fn unregistered_model_type_falls_back_to_generic_in_both_registry_and_dispatch() {
    let probe = "totally-unknown-arch";
    assert!(find_architecture(probe).is_none());

    let config = serde_json::json!({ "model_type": probe });
    let arch = detect_from_json(&config);
    assert_eq!(arch.family(), GENERIC_FAMILY);
}
