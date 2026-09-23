//! `index.json` for a VINDEX3 container — the sole root authority (spec §12).
//!
//! Deliberately a **different type** from [`VindexConfig`](crate::config::index::VindexConfig),
//! not a superset of it. The shipped generation's index describes a dense
//! Gemma-shaped extraction: `layers`, `down_top_k`, `intermediate_size`. A
//! VINDEX3 index describes a *catalogue* — which segments exist, which
//! manifest interprets them, which profiles may be selected. Growing one
//! struct to cover both would give every field an "unless version 3" caveat,
//! and the loader would sniff fields to decide which half it is looking at,
//! which is exactly the heuristic dispatch §12.1 forbids.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::format::generation::V3_CURRENT_SCHEMA;

/// Profile name every container carries: full-fidelity execution.
pub const PROFILE_EXACT: &str = "exact";

/// `index.json` as a VINDEX3 container writes it.
///
/// `segments` maps a segment *key* to the number of physical files under it.
/// A key is a path stem relative to the container root (`routed/layer_000`),
/// so the loader composes a filename rather than globbing a directory —
/// filename sniffing is what §12.1 rules out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Vindex3Index {
    /// Always [`V3_CURRENT_SCHEMA`]. The sole generation discriminator.
    pub version: u32,
    /// Identity, carried so a container names itself without a sidecar.
    pub model: String,
    pub family: String,
    /// Residual width the model operates at.
    pub hidden_size: usize,
    pub num_layers: usize,
    /// Filename of the MoE programme manifest, relative to the root.
    pub moe_manifest: String,
    /// Profiles this container declares. Never empty: a container with no
    /// selectable profile cannot be served, and discovering that at bind time
    /// rather than at load time is the failure mode profiles exist to prevent.
    pub profiles: Vec<String>,
    /// Segment key → physical file count.
    pub segments: BTreeMap<String, u32>,
}

impl Vindex3Index {
    /// A single-profile container over the given segments.
    pub fn new(
        model: impl Into<String>,
        family: impl Into<String>,
        hidden_size: usize,
        num_layers: usize,
        moe_manifest: impl Into<String>,
        segments: BTreeMap<String, u32>,
    ) -> Self {
        Self {
            version: V3_CURRENT_SCHEMA,
            model: model.into(),
            family: family.into(),
            hidden_size,
            num_layers,
            moe_manifest: moe_manifest.into(),
            profiles: vec![PROFILE_EXACT.to_string()],
            segments,
        }
    }

    /// Whether `profile` is one this container physically carries.
    ///
    /// The check a caller must pass *before* binding: §9.1 requires that a
    /// profile can only select variants that were extracted, and answering
    /// that at load time is what stops a silent conversion later.
    pub fn declares_profile(&self, profile: &str) -> bool {
        self.profiles.iter().any(|p| p == profile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segments() -> BTreeMap<String, u32> {
        BTreeMap::from([("routed/layer_000".to_string(), 1)])
    }

    fn index() -> Vindex3Index {
        Vindex3Index::new(
            "fixture-a",
            "direct-moe",
            256,
            1,
            "moe_manifest.json",
            segments(),
        )
    }

    #[test]
    fn a_fresh_index_declares_the_successor_schema() {
        // The whole dispatch turns on this one number being 3 and nothing else.
        assert_eq!(index().version, V3_CURRENT_SCHEMA);
    }

    #[test]
    fn a_fresh_index_is_servable_because_it_declares_a_profile() {
        let i = index();
        assert!(i.declares_profile(PROFILE_EXACT));
        assert!(
            !i.profiles.is_empty(),
            "a profile-less container cannot serve"
        );
    }

    #[test]
    fn an_undeclared_profile_is_refused_rather_than_assumed() {
        assert!(!index().declares_profile("browse"));
    }

    #[test]
    fn the_index_round_trips_through_json() {
        // It is the root authority; if it cannot survive its own serialisation
        // nothing below it is reachable.
        let before = index();
        let json = serde_json::to_string(&before).unwrap();
        let after: Vindex3Index = serde_json::from_str(&json).unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn the_serialised_form_carries_the_version_a_detector_reads() {
        // `detect_generation` parses the raw JSON, not this struct, so the
        // wire key matters independently of the field name.
        let json: serde_json::Value = serde_json::to_value(index()).unwrap();
        assert_eq!(json["version"], serde_json::json!(V3_CURRENT_SCHEMA));
    }
}
