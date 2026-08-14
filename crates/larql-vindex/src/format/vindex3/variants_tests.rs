//! Tests for the §9.1 variant catalogue.
//!
//! The theme throughout: a refusal is only useful if it names the remedy, so
//! most of these assert on the *contents* of the error rather than that one
//! occurred.

use super::*;

const Q6K: &str = "exact-q6k";
const MXFP4: &str = "native-mxfp4";
const GATE_UP: &str = "layer.12.routed.gate_up";

fn q6k() -> StoredVariant {
    StoredVariant::new("routed/layer_012.q6k", Fidelity::SourceEquivalent)
}

fn mxfp4() -> StoredVariant {
    StoredVariant::new("routed/layer_012.mxfp4", Fidelity::SourceExact)
}

/// The spec's own worked example (§9.1).
fn both() -> RegionSetVariants {
    RegionSetVariants::single(Q6K, q6k()).with_variant(MXFP4, mxfp4())
}

fn catalogue() -> VariantCatalogue {
    VariantCatalogue::new().with_set(GATE_UP, both())
}

// ── Selecting what is there ──────────────────────────────────────────────

#[test]
fn a_present_variant_resolves_to_its_own_bytes() {
    let c = catalogue();
    assert_eq!(c.select(GATE_UP, MXFP4).unwrap(), &mxfp4());
    assert_eq!(c.select(GATE_UP, Q6K).unwrap(), &q6k());
}

#[test]
fn the_baseline_is_what_a_profile_naming_nothing_gets() {
    assert_eq!(catalogue().select_baseline(GATE_UP).unwrap(), &q6k());
}

#[test]
fn adding_a_variant_leaves_the_baseline_alone() {
    // §9.1's incremental-pack rule: the multi-terabyte baseline is never
    // rewritten, so adding a pack must not move which variant is canonical.
    let before = RegionSetVariants::single(Q6K, q6k());
    let after = before.clone().with_variant(MXFP4, mxfp4());
    assert_eq!(after.baseline, before.baseline);
    assert_eq!(after.variants.len(), 2);
}

#[test]
fn a_variant_is_removable_without_touching_the_baseline() {
    // The other half of "individually removable": deleting a pack leaves a
    // servable region set behind.
    let mut set = both();
    set.variants.remove(MXFP4);
    assert!(set.select_baseline(GATE_UP).is_ok());
    assert_eq!(set.present(), vec![Q6K.to_string()]);
}

// ── Selecting what is not there — the point of the module ───────────────

#[test]
fn an_absent_variant_names_the_set_the_request_and_what_is_present() {
    let err = catalogue().select(GATE_UP, "mxfp4").unwrap_err();
    match &err {
        VariantDefect::AbsentVariant {
            region_set,
            requested,
            present,
        } => {
            assert_eq!(region_set, GATE_UP);
            assert_eq!(requested, "mxfp4");
            assert_eq!(present, &vec![Q6K.to_string(), MXFP4.to_string()]);
        }
        other => panic!("wrong defect: {other:?}"),
    }
    // The near-miss this is really for: `mxfp4` vs `native-mxfp4`. The message
    // has to carry the spelling that would have worked, or the reader
    // re-extracts instead of re-typing.
    let msg = err.to_string();
    assert!(msg.contains(GATE_UP), "{msg}");
    assert!(msg.contains("mxfp4"), "{msg}");
    assert!(msg.contains(MXFP4), "{msg}");
}

#[test]
fn an_unknown_region_set_names_what_is_catalogued() {
    let err = catalogue()
        .select("layer.99.routed.down", MXFP4)
        .unwrap_err();
    match &err {
        VariantDefect::UnknownRegionSet {
            region_set,
            catalogued,
        } => {
            assert_eq!(region_set, "layer.99.routed.down");
            assert_eq!(catalogued, &vec![GATE_UP.to_string()]);
        }
        other => panic!("wrong defect: {other:?}"),
    }
    assert!(err.to_string().contains(GATE_UP));
}

#[test]
fn an_absent_variant_is_distinguished_from_an_absent_region_set() {
    // Same remedy family, different remedy: one is a profile edit, the other
    // means the container never catalogued the component at all. Collapsing
    // them into one error is what makes a load failure unactionable.
    let c = catalogue();
    assert!(matches!(
        c.select(GATE_UP, "nope"),
        Err(VariantDefect::AbsentVariant { .. })
    ));
    assert!(matches!(
        c.select("nope", Q6K),
        Err(VariantDefect::UnknownRegionSet { .. })
    ));
}

#[test]
fn the_present_list_is_sorted_so_a_diagnostic_is_stable() {
    let set = RegionSetVariants::single("zzz", q6k())
        .with_variant("aaa", mxfp4())
        .with_variant("mmm", q6k());
    assert_eq!(set.present(), vec!["aaa", "mmm", "zzz"]);
}

// ── Structural defects, found without any profile ───────────────────────

#[test]
fn a_clean_catalogue_reports_no_defects() {
    assert!(catalogue().defects().is_empty());
}

#[test]
fn a_baseline_that_is_not_present_is_a_defect() {
    // The failure that makes a region set unservable while every individual
    // pack looks fine — nothing else in the file disagrees with itself.
    let set = RegionSetVariants {
        baseline: "gone".into(),
        variants: BTreeMap::from([(Q6K.to_string(), q6k())]),
    };
    let defects = VariantCatalogue::new().with_set(GATE_UP, set).defects();
    match defects.as_slice() {
        [VariantDefect::BaselineAbsent {
            region_set,
            baseline,
            present,
        }] => {
            assert_eq!(region_set, GATE_UP);
            assert_eq!(baseline, "gone");
            assert_eq!(present, &vec![Q6K.to_string()]);
        }
        other => panic!("wrong defects: {other:?}"),
    }
}

#[test]
fn a_region_set_with_no_variants_is_a_defect() {
    let set = RegionSetVariants {
        baseline: Q6K.into(),
        variants: BTreeMap::new(),
    };
    let defects = VariantCatalogue::new().with_set(GATE_UP, set).defects();
    assert_eq!(
        defects,
        vec![VariantDefect::NoVariants {
            region_set: GATE_UP.to_string()
        }]
    );
}

#[test]
fn an_empty_region_set_reports_only_that_not_a_missing_baseline_too() {
    // One cause, one defect. Reporting both would send someone to fix the
    // baseline name when the region set has nothing to name.
    let set = RegionSetVariants {
        baseline: "anything".into(),
        variants: BTreeMap::new(),
    };
    assert_eq!(
        VariantCatalogue::new()
            .with_set(GATE_UP, set)
            .defects()
            .len(),
        1
    );
}

#[test]
fn every_broken_region_set_is_reported_in_one_pass() {
    // Two repairs should take one load, not two.
    let broken = |name: &str| RegionSetVariants {
        baseline: name.to_string(),
        variants: BTreeMap::from([(Q6K.to_string(), q6k())]),
    };
    let c = VariantCatalogue::new()
        .with_set("a", broken("missing-a"))
        .with_set("b", broken("missing-b"));
    assert_eq!(c.defects().len(), 2);
}

// ── Wire form ────────────────────────────────────────────────────────────

#[test]
fn the_catalogue_round_trips_through_json() {
    let before = catalogue();
    let json = serde_json::to_string(&before).unwrap();
    let after: VariantCatalogue = serde_json::from_str(&json).unwrap();
    assert_eq!(before, after);
}

#[test]
fn the_wire_form_is_the_shape_the_spec_prints() {
    // §9.1 shows `variants` as an object keyed by variant name, each carrying
    // `storage` and `fidelity`, beside a `baseline`. A reader written against
    // the spec has to find those keys.
    let json = serde_json::to_value(catalogue()).unwrap();
    let set = &json[GATE_UP];
    assert_eq!(set["baseline"], serde_json::json!(Q6K));
    assert_eq!(
        set["variants"][MXFP4]["storage"],
        serde_json::json!("routed/layer_012.mxfp4")
    );
    assert_eq!(
        set["variants"][MXFP4]["fidelity"],
        serde_json::json!("source-exact"),
        "fidelity must serialise in the spec's kebab-case spelling"
    );
}

#[test]
fn an_empty_catalogue_is_the_default_so_a_pack_less_container_still_loads() {
    let c = VariantCatalogue::default();
    assert!(c.is_empty());
    assert_eq!(c.len(), 0);
    assert!(c.defects().is_empty());
    assert!(c.region_sets().is_empty());
    assert!(c.get(GATE_UP).is_none());
}
