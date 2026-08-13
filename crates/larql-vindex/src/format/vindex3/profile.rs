//! A profile — the thing that selects variants (spec §9.1).
//!
//! # Why a name was not enough
//!
//! A profile used to be a string in a list, and `declares_profile` answered
//! "is that string one of ours". That check cannot fail the way §9.1 needs it
//! to, because a name carries no claim about bytes: a container could declare
//! `"routed-mxfp4"`, carry no MXFP4 pack, and pass. The absence surfaced at
//! bind time as a missing region, one layer below where anyone would look for
//! it, with the profile name nowhere in the message.
//!
//! So a profile names, per region set, which variant it selects. A region set
//! the profile says nothing about takes that set's baseline — which keeps the
//! common profile (everything canonical) empty rather than a restatement of
//! the whole catalogue, and keeps a new region set from silently becoming
//! unselectable the moment it is added.
//!
//! # Resolution is total, not first-failure
//!
//! [`Profile::resolve`] returns every defect it finds. A profile that names
//! two absent packs should take one repair pass; being shown the first, fixing
//! it, and being shown the second is how a load-time check teaches people to
//! bypass it.

use crate::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::variants::{StoredVariant, VariantCatalogue, VariantDefect};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Profile name every container carries: full-fidelity execution.
pub const PROFILE_EXACT: &str = "exact";

/// A named selection over the variant catalogue.
///
/// # Reads a bare name as well as an object
///
/// A profile used to serialise as just its name (`"profiles": ["exact"]`).
/// Containers written that way still load, deserialising to a profile that
/// selects nothing — which is exactly what a bare name meant.
///
/// This matters because the schema version cannot catch it: both spellings are
/// schema 3, so §12.1's version dispatch sees no disagreement and the failure
/// surfaces as `invalid type: map, expected a string` from deep inside serde,
/// naming neither the field nor the container. The generation discriminator
/// guards *between* generations; nothing guards within one, and the VINDEX3
/// ABI is explicitly not frozen yet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Profile {
    pub name: String,
    /// Region set → variant name. Absent means "take the baseline", so this is
    /// empty for a profile that wants everything canonical.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub selects: BTreeMap<String, String>,
}

impl<'de> Deserialize<'de> for Profile {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        /// The object form, with the same field defaults the derive gave it.
        #[derive(Deserialize)]
        struct AsObject {
            name: String,
            #[serde(default)]
            selects: BTreeMap<String, String>,
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Either {
            /// Pre-selection spelling: the profile was only ever a name.
            Name(String),
            Object(AsObject),
        }

        Ok(match Either::deserialize(d)? {
            Either::Name(name) => Profile {
                name,
                selects: BTreeMap::new(),
            },
            Either::Object(o) => Profile {
                name: o.name,
                selects: o.selects,
            },
        })
    }
}

impl Profile {
    /// A profile that takes every region set's baseline.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            selects: BTreeMap::new(),
        }
    }

    /// The canonical profile: every baseline, nothing overridden.
    pub fn exact() -> Self {
        Self::new(PROFILE_EXACT)
    }

    /// Override one region set's variant.
    pub fn selecting(mut self, region_set: impl Into<String>, variant: impl Into<String>) -> Self {
        self.selects.insert(region_set.into(), variant.into());
        self
    }

    /// Resolve this profile against what the container physically carries.
    ///
    /// Every catalogued region set appears in the result — the ones this
    /// profile names at its chosen variant, the rest at their baseline — so a
    /// caller holds a complete map of the bytes it is about to execute, and
    /// never has to fall back to "look it up again later", which is where a
    /// silent conversion would get its opportunity.
    pub fn resolve<'a>(
        &self,
        catalogue: &'a VariantCatalogue,
    ) -> Result<ResolvedProfile<'a>, Vec<VariantDefect>> {
        let mut defects = Vec::new();

        // Selections first: a name this profile got wrong is the failure we
        // are here to report, and reporting it before the catalogue's own
        // structural defects puts the profile's error at the top.
        for (region_set, variant) in &self.selects {
            if let Err(defect) = catalogue.select(region_set, variant) {
                defects.push(defect);
            }
        }
        defects.extend(catalogue.defects());

        if !defects.is_empty() {
            return Err(defects);
        }

        let mut chosen = BTreeMap::new();
        for region_set in catalogue.region_sets() {
            let stored = match self.selects.get(&region_set) {
                Some(variant) => catalogue.select(&region_set, variant),
                None => catalogue.select_baseline(&region_set),
            }
            .expect("defects were collected above, so every lookup resolves");
            chosen.insert(region_set, stored);
        }
        Ok(ResolvedProfile {
            name: self.name.clone(),
            chosen,
        })
    }
}

/// A profile that resolved — every region set bound to a present variant.
///
/// Holding borrows of the catalogue rather than copies is deliberate: this
/// cannot outlive the index it was resolved against, so it cannot be carried
/// past a reload and quietly describe bytes that were replaced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedProfile<'a> {
    name: String,
    chosen: BTreeMap<String, &'a StoredVariant>,
}

impl<'a> ResolvedProfile<'a> {
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The variant this profile executes for `region_set`.
    pub fn variant_for(&self, region_set: &str) -> Option<&'a StoredVariant> {
        self.chosen.get(region_set).copied()
    }

    /// Every (region set, variant) pair, in region-set order.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &'a StoredVariant)> + '_ {
        self.chosen.iter().map(|(k, v)| (k.as_str(), *v))
    }

    pub fn len(&self) -> usize {
        self.chosen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chosen.is_empty()
    }

    /// Segment keys this profile will read, deduplicated and sorted.
    ///
    /// What a loader needs in order to touch only the packs in play — the
    /// reason a routed-MXFP4 profile does not page in the Q6_K baseline.
    pub fn storage_keys(&self) -> Vec<&'a str> {
        let mut keys: Vec<&str> = self.chosen.values().map(|v| v.storage.as_str()).collect();
        keys.sort_unstable();
        keys.dedup();
        keys
    }
}

/// Why a container cannot serve the profile it was asked for.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProfileSelectionError {
    #[error("no profile '{requested}' in this container; declared: [{}]", declared.join(", "))]
    UndeclaredProfile {
        requested: String,
        declared: Vec<String>,
    },

    /// Declared, but its selections do not match the bytes on disk — the
    /// failure §9.1 exists to catch. Carries every defect, so one repair pass
    /// clears them all.
    #[error(
        "profile '{profile}' cannot be served: {}",
        defects.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("; ")
    )]
    Unsatisfiable {
        profile: String,
        defects: Vec<VariantDefect>,
    },
}

/// Resolve a declared profile of `index` against the variants it carries.
///
/// Lives here rather than on the index because every rule it enforces is a
/// profile rule: the index only holds the two halves side by side.
pub fn select<'a>(
    index: &'a super::index::Vindex3Index,
    name: &str,
) -> Result<ResolvedProfile<'a>, ProfileSelectionError> {
    let profile = index
        .profile(name)
        .ok_or_else(|| ProfileSelectionError::UndeclaredProfile {
            requested: name.to_string(),
            declared: index
                .profile_names()
                .into_iter()
                .map(String::from)
                .collect(),
        })?;
    profile
        .resolve(&index.variants)
        .map_err(|defects| ProfileSelectionError::Unsatisfiable {
            profile: name.to_string(),
            defects,
        })
}

/// Resolve every profile `index` declares, for a caller that must know the
/// whole container is servable — `Vindex3Container::open` does this before
/// reading a segment byte.
pub fn resolve_all(index: &super::index::Vindex3Index) -> Result<(), ProfileSelectionError> {
    for name in index.profile_names() {
        select(index, name)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "profile_tests.rs"]
mod tests;
