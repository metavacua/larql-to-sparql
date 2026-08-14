//! Tests for profile resolution (§9.1).

use super::*;
use crate::format::capability::authority::Fidelity;
use crate::format::vindex3::variants::{RegionSetVariants, StoredVariant};

const Q6K: &str = "exact-q6k";
const MXFP4: &str = "native-mxfp4";
const GATE_UP: &str = "layer.12.routed.gate_up";
const DOWN: &str = "layer.12.routed.down";

fn q6k() -> StoredVariant {
    StoredVariant::new("routed/layer_012.q6k", Fidelity::SourceEquivalent)
}

fn mxfp4() -> StoredVariant {
    StoredVariant::new("routed/layer_012.mxfp4", Fidelity::SourceExact)
}

/// Two region sets; only `gate_up` has an MXFP4 pack beside its baseline.
fn catalogue() -> VariantCatalogue {
    VariantCatalogue::new()
        .with_set(
            GATE_UP,
            RegionSetVariants::single(Q6K, q6k()).with_variant(MXFP4, mxfp4()),
        )
        .with_set(DOWN, RegionSetVariants::single(Q6K, q6k()))
}

// ── The canonical profile ────────────────────────────────────────────────

#[test]
fn the_exact_profile_takes_every_baseline_without_naming_any() {
    let c = catalogue();
    let resolved = Profile::exact().resolve(&c).unwrap();
    assert_eq!(resolved.name(), PROFILE_EXACT);
    assert_eq!(resolved.len(), 2);
    assert_eq!(resolved.variant_for(GATE_UP), Some(&q6k()));
    assert_eq!(resolved.variant_for(DOWN), Some(&q6k()));
}

#[test]
fn the_canonical_profile_carries_no_selections_at_all() {
    // If "everything canonical" had to restate the catalogue, every new region
    // set would need every profile edited, and the one that was forgotten
    // would be the one that broke.
    assert!(Profile::exact().selects.is_empty());
}

#[test]
fn a_region_set_the_profile_says_nothing_about_still_resolves() {
    // `down` has no MXFP4 pack and the profile does not mention it; it must
    // still appear, at its baseline, rather than dropping out of the result.
    let profile = Profile::new("routed-mxfp4").selecting(GATE_UP, MXFP4);
    let c = catalogue();
    let resolved = profile.resolve(&c).unwrap();
    assert_eq!(resolved.variant_for(GATE_UP), Some(&mxfp4()));
    assert_eq!(resolved.variant_for(DOWN), Some(&q6k()));
}

// ── Selecting an absent variant — the refusal §9.1 asks for ─────────────

#[test]
fn selecting_an_absent_variant_fails_naming_set_request_and_present() {
    let profile = Profile::new("routed-mxfp4").selecting(DOWN, MXFP4);
    let defects = profile.resolve(&catalogue()).unwrap_err();
    match defects.as_slice() {
        [VariantDefect::AbsentVariant {
            region_set,
            requested,
            present,
        }] => {
            assert_eq!(region_set, DOWN);
            assert_eq!(requested, MXFP4);
            assert_eq!(present, &vec![Q6K.to_string()]);
        }
        other => panic!("wrong defects: {other:?}"),
    }
}

#[test]
fn selecting_an_uncatalogued_region_set_fails_naming_what_exists() {
    let profile = Profile::new("typo").selecting("layer.12.routed.gate-up", MXFP4);
    let defects = profile.resolve(&catalogue()).unwrap_err();
    assert!(matches!(
        defects.as_slice(),
        [VariantDefect::UnknownRegionSet { .. }]
    ));
    assert!(defects[0].to_string().contains(GATE_UP));
}

#[test]
fn every_bad_selection_is_reported_in_one_pass() {
    let profile = Profile::new("two-wrongs")
        .selecting(DOWN, MXFP4)
        .selecting("nope", Q6K);
    let defects = profile.resolve(&catalogue()).unwrap_err();
    assert_eq!(defects.len(), 2, "{defects:?}");
}

#[test]
fn a_failed_resolution_yields_no_partial_selection() {
    // Fail closed: the caller gets defects or a complete map, never a map with
    // the resolvable half filled in. A partial map is what would let a load
    // continue far enough to read bytes for the region sets that did resolve.
    let profile = Profile::new("routed-mxfp4").selecting(DOWN, MXFP4);
    assert!(profile.resolve(&catalogue()).is_err());
}

#[test]
fn a_structurally_broken_catalogue_fails_even_a_profile_that_names_nothing() {
    // The baseline is what an unnamed region set resolves to, so a missing
    // baseline breaks the canonical profile too — it is not only a problem for
    // profiles that select.
    let broken = VariantCatalogue::new().with_set(
        GATE_UP,
        RegionSetVariants {
            baseline: "gone".into(),
            variants: std::collections::BTreeMap::from([(Q6K.to_string(), q6k())]),
        },
    );
    let defects = Profile::exact().resolve(&broken).unwrap_err();
    assert!(matches!(
        defects.as_slice(),
        [VariantDefect::BaselineAbsent { .. }]
    ));
}

// ── What a resolved profile is for ──────────────────────────────────────

#[test]
fn a_resolved_profile_lists_only_the_storage_it_will_touch() {
    // The incremental-pack payoff: a profile on the MXFP4 pack must not name
    // the Q6_K file for that region set, or nothing was saved by packing.
    let profile = Profile::new("routed-mxfp4").selecting(GATE_UP, MXFP4);
    let c = catalogue();
    let resolved = profile.resolve(&c).unwrap();
    assert_eq!(
        resolved.storage_keys(),
        vec!["routed/layer_012.mxfp4", "routed/layer_012.q6k"]
    );
}

#[test]
fn storage_keys_are_deduplicated() {
    // Two region sets can live in one segment file; a loader asked to read it
    // twice would double the resident bytes.
    let shared = VariantCatalogue::new()
        .with_set(GATE_UP, RegionSetVariants::single(Q6K, q6k()))
        .with_set(DOWN, RegionSetVariants::single(Q6K, q6k()));
    let resolved = Profile::exact().resolve(&shared).unwrap();
    assert_eq!(resolved.storage_keys(), vec!["routed/layer_012.q6k"]);
}

#[test]
fn entries_come_out_in_region_set_order() {
    let c = catalogue();
    let resolved = Profile::exact().resolve(&c).unwrap();
    let names: Vec<&str> = resolved.entries().map(|(k, _)| k).collect();
    assert_eq!(names, vec![DOWN, GATE_UP]);
}

#[test]
fn an_empty_catalogue_resolves_to_an_empty_profile() {
    let c = VariantCatalogue::new();
    let resolved = Profile::exact().resolve(&c).unwrap();
    assert!(resolved.is_empty());
    assert!(resolved.storage_keys().is_empty());
    assert_eq!(resolved.variant_for(GATE_UP), None);
}

// ── Wire form ────────────────────────────────────────────────────────────

#[test]
fn a_profile_round_trips_through_json() {
    let before = Profile::new("routed-mxfp4").selecting(GATE_UP, MXFP4);
    let json = serde_json::to_string(&before).unwrap();
    let after: Profile = serde_json::from_str(&json).unwrap();
    assert_eq!(before, after);
}

#[test]
fn a_selectionless_profile_omits_the_empty_map_on_the_wire() {
    // Every container carries the canonical profile, so this key would
    // otherwise appear empty in every index.json ever written.
    let json = serde_json::to_value(Profile::exact()).unwrap();
    assert_eq!(json["name"], serde_json::json!(PROFILE_EXACT));
    assert!(json.get("selects").is_none(), "{json}");
}

#[test]
fn a_profile_without_selects_still_deserialises() {
    let p: Profile = serde_json::from_str(r#"{"name":"exact"}"#).unwrap();
    assert_eq!(p, Profile::exact());
}

// ── Reading the pre-selection spelling ──────────────────────────────────

#[test]
fn a_bare_profile_name_still_deserialises() {
    // `"profiles": ["exact"]` is how every container written before profiles
    // could select was spelled. Both are schema 3, so version dispatch cannot
    // tell them apart — this is the only thing that keeps them readable.
    let p: Profile = serde_json::from_str(r#""exact""#).unwrap();
    assert_eq!(p, Profile::exact());
    assert!(p.selects.is_empty(), "a bare name selects nothing");
}

#[test]
fn a_list_mixing_both_spellings_loads() {
    let list: Vec<Profile> = serde_json::from_str(
        r#"["exact", {"name":"routed-mxfp4","selects":{"layer.12.routed.gate_up":"native-mxfp4"}}]"#,
    )
    .unwrap();
    assert_eq!(list.len(), 2);
    assert_eq!(list[0], Profile::exact());
    assert_eq!(list[1].name, "routed-mxfp4");
    assert_eq!(list[1].selects.len(), 1);
}

#[test]
fn the_object_form_still_round_trips() {
    // The compatibility path must not cost the real one: writing stays the
    // object form, and reading it back must be unchanged.
    let before = Profile::new("routed-mxfp4").selecting(GATE_UP, MXFP4);
    let after: Profile = serde_json::from_str(&serde_json::to_string(&before).unwrap()).unwrap();
    assert_eq!(before, after);
}
