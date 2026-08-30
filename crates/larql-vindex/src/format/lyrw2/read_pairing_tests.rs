//! Pair-bound region resolution.
//!
//! Before split-scale banks every role in a bank was unique, so a role was
//! a sufficient address. It is not any more: a bank can declare two
//! `Scales` regions, one partnering the fused gate/up payload and one
//! partnering `down`. `pair_id` is what tells them apart, and these tests
//! require the reader to *use* it rather than carry it as metadata.
//!
//! ```text
//! GateUpFused ── pair_id 1 ── Scales
//! Down        ── pair_id 2 ── Scales
//! ```
//!
//! The malformed cases here cannot be produced by the importer, which
//! always writes consistent pairs — they are built by hand because a
//! reader must refuse a container it did not write, and a hand-edited or
//! foreign file is exactly where a broken pairing arrives from.

use super::bank::{BankDescriptor, BankKind};
use super::browse_mode::BrowseMode;
use super::error::Lyrw2Error;
use super::plan::Lyrw2Plan;
use super::read::Lyrw2Reader;
use super::region_format::{Packing, RegionFormat};
use super::region_layout::RegionLayout;
use super::region_role::RegionRole;
use super::region_schema::RegionSchema;
use super::test_fixtures::{temp_path, BANK_ID};
use super::write::Lyrw2Writer;

const HIDDEN: u32 = 64;
const INTERMEDIATE: u32 = 64;
const ENTRIES: u32 = 2;
const LOGICAL_LAYER: u32 = 0;

const PAIR_GATE_UP: u16 = 1;
const PAIR_DOWN: u16 = 2;

/// Distinguishable per (entry, schema) so a crossed resolution cannot pass.
fn pattern(entry: u32, schema: u16, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (entry as usize * 97 + schema as usize * 13 + i) as u8)
        .collect()
}

const PAYLOAD_LEN: usize = 32;
const SCALE_LEN: usize = 8;

fn schema_len(schema: u16) -> usize {
    // Payload schemas are 0 and 1; scale schemas are 2 and 3.
    if schema < 2 {
        PAYLOAD_LEN
    } else {
        SCALE_LEN
    }
}

fn bank(schema_count: u16) -> BankDescriptor {
    BankDescriptor {
        bank_id: BANK_ID,
        kind: BankKind::Routed,
        num_entries: ENTRIES,
        input_dim: HIDDEN,
        intermediate_dim: INTERMEDIATE,
        output_dim: HIDDEN,
        region_schema_count: schema_count,
        browse: BrowseMode::None,
    }
}

fn payload(schema_index: u16, role: RegionRole, pair_id: u16) -> RegionSchema {
    RegionSchema {
        schema_index,
        role,
        format: RegionFormat::Mxfp4,
        packing: Packing::BlocksValues,
        pair_id,
        // Only the fused operand has an arrangement to declare; `Down`
        // claiming one is what `LayoutOnNonFusedRegion` refuses.
        layout: if matches!(role, RegionRole::GateUpFused) {
            RegionLayout::Interleaved
        } else {
            RegionLayout::Unspecified
        },
        rows: INTERMEDIATE,
        cols: HIDDEN,
    }
}

fn scales(schema_index: u16, pair_id: u16) -> RegionSchema {
    RegionSchema {
        schema_index,
        role: RegionRole::Scales,
        format: RegionFormat::Mxfp4,
        packing: Packing::BlocksScales,
        pair_id,
        layout: RegionLayout::Unspecified,
        rows: INTERMEDIATE,
        cols: HIDDEN,
    }
}

/// The well-formed split-scale bank: two payloads, two scale partners.
fn well_formed() -> Vec<RegionSchema> {
    vec![
        payload(0, RegionRole::GateUpFused, PAIR_GATE_UP),
        payload(1, RegionRole::Down, PAIR_DOWN),
        scales(2, PAIR_GATE_UP),
        scales(3, PAIR_DOWN),
    ]
}

fn write(name: &str, schemas: Vec<RegionSchema>) -> Vec<u8> {
    let count = schemas.len() as u16;
    let plan = Lyrw2Plan::single_segment(LOGICAL_LAYER, bank(count), schemas);
    let path = temp_path(name);
    let mut w = Lyrw2Writer::create(&path, plan).unwrap();
    for entry in 0..ENTRIES {
        for schema in 0..count {
            w.write_region(&pattern(entry, schema, schema_len(schema)))
                .unwrap();
        }
    }
    w.finish().unwrap();
    std::fs::read(&path).unwrap()
}

// ── the positive cases ───────────────────────────────────────────────────

/// Each payload resolves *its own* partner. The crossing check is the
/// point: a reader that returned the first `Scales` for both would pass
/// every assertion except the inequality.
#[test]
fn each_payload_resolves_its_own_scale_partner() {
    let bytes = write("pairing-ok.weights", well_formed());
    let r = Lyrw2Reader::parse(&bytes).unwrap();

    for entry in 0..ENTRIES {
        let gu = r
            .paired_region_bytes(BANK_ID, entry, RegionRole::GateUpFused, RegionRole::Scales)
            .unwrap()
            .expect("gate_up scales");
        let dn = r
            .paired_region_bytes(BANK_ID, entry, RegionRole::Down, RegionRole::Scales)
            .unwrap()
            .expect("down scales");

        assert_eq!(
            gu,
            &pattern(entry, 2, SCALE_LEN)[..],
            "entry {entry} gate_up"
        );
        assert_eq!(dn, &pattern(entry, 3, SCALE_LEN)[..], "entry {entry} down");
        assert_ne!(gu, dn, "entry {entry}: both pairings hit one region");
    }
}

/// Pairing is by `pair_id`, not by ordinal position. Declaring the scale
/// regions in the opposite order must not change which payload each
/// belongs to — a positional reader would silently cross them.
#[test]
fn pairing_survives_the_scale_regions_being_declared_in_reverse() {
    let reversed = vec![
        payload(0, RegionRole::GateUpFused, PAIR_GATE_UP),
        payload(1, RegionRole::Down, PAIR_DOWN),
        scales(2, PAIR_DOWN),
        scales(3, PAIR_GATE_UP),
    ];
    let bytes = write("pairing-reversed.weights", reversed);
    let r = Lyrw2Reader::parse(&bytes).unwrap();

    // schema 2 now carries DOWN's scales and schema 3 gate_up's.
    let gu = r
        .paired_region_bytes(BANK_ID, 0, RegionRole::GateUpFused, RegionRole::Scales)
        .unwrap()
        .expect("gate_up scales");
    let dn = r
        .paired_region_bytes(BANK_ID, 0, RegionRole::Down, RegionRole::Scales)
        .unwrap()
        .expect("down scales");
    assert_eq!(gu, &pattern(0, 3, SCALE_LEN)[..], "followed pair_id");
    assert_eq!(dn, &pattern(0, 2, SCALE_LEN)[..], "followed pair_id");
}

/// Unique roles still resolve by role alone. The ambiguity refusal must
/// not become a blanket refusal on any bank that happens to be paired.
#[test]
fn unique_roles_still_resolve_by_role_alone() {
    let bytes = write("pairing-unique-roles.weights", well_formed());
    let r = Lyrw2Reader::parse(&bytes).unwrap();
    for entry in 0..ENTRIES {
        assert_eq!(
            r.region_bytes(BANK_ID, entry, RegionRole::GateUpFused)
                .unwrap()
                .expect("gate_up"),
            &pattern(entry, 0, PAYLOAD_LEN)[..]
        );
    }
}

// ── the refusals ─────────────────────────────────────────────────────────

/// The defect this closed: a role-only lookup for a duplicated role used
/// to return the first match, handing back gate/up's exponents when
/// down's were meant.
#[test]
fn a_duplicated_role_refuses_role_only_lookup() {
    let bytes = write("pairing-ambiguous-role.weights", well_formed());
    let r = Lyrw2Reader::parse(&bytes).unwrap();
    let err = r
        .region_bytes(BANK_ID, 0, RegionRole::Scales)
        .expect_err("two Scales regions cannot resolve by role");
    match err {
        Lyrw2Error::AmbiguousRole { count, .. } => assert_eq!(count, 2),
        other => panic!("expected AmbiguousRole, got {other}"),
    }
}

/// A payload naming a `pair_id` no scale region carries is malformed, and
/// must be named as a missing partner rather than resolving to nothing.
#[test]
fn a_pair_id_with_no_partner_is_refused() {
    let orphaned = vec![
        payload(0, RegionRole::GateUpFused, PAIR_GATE_UP),
        payload(1, RegionRole::Down, PAIR_DOWN),
        // Both scale regions claim gate_up's pairing; nothing partners DOWN.
        scales(2, PAIR_GATE_UP),
        scales(3, PAIR_GATE_UP),
    ];
    let bytes = write("pairing-orphan.weights", orphaned);
    let r = Lyrw2Reader::parse(&bytes).unwrap();
    let err = r
        .paired_region_bytes(BANK_ID, 0, RegionRole::Down, RegionRole::Scales)
        .expect_err("down names a pair_id nothing carries");
    match err {
        Lyrw2Error::MissingPartner { pair_id, .. } => assert_eq!(pair_id, PAIR_DOWN),
        other => panic!("expected MissingPartner, got {other}"),
    }
}

/// Two partners with the same role *and* the same `pair_id` make the
/// pairing itself ambiguous — refuse rather than take the first.
#[test]
fn two_partners_sharing_a_pair_id_are_refused() {
    let doubled = vec![
        payload(0, RegionRole::GateUpFused, PAIR_GATE_UP),
        payload(1, RegionRole::Down, PAIR_DOWN),
        scales(2, PAIR_GATE_UP),
        scales(3, PAIR_GATE_UP),
    ];
    let bytes = write("pairing-doubled.weights", doubled);
    let r = Lyrw2Reader::parse(&bytes).unwrap();
    let err = r
        .paired_region_bytes(BANK_ID, 0, RegionRole::GateUpFused, RegionRole::Scales)
        .expect_err("two partners share gate_up's pair_id");
    match err {
        Lyrw2Error::AmbiguousPartner { count, pair_id, .. } => {
            assert_eq!(count, 2);
            assert_eq!(pair_id, PAIR_GATE_UP);
        }
        other => panic!("expected AmbiguousPartner, got {other}"),
    }
}

/// Asking an unpaired region for a partner is a caller mistake, not an
/// absence. `None` means "this bank has no such region"; an unpaired
/// payload asked for its scales must say so distinctly.
#[test]
fn an_unpaired_payload_refuses_a_partner_lookup() {
    // A genuine inline bank. `BlocksValues` cannot be declared unpaired —
    // `Lyrw2Plan::validate` already refuses that as `InconsistentPairing`,
    // so the packing has to say "my scales are inline" for the schema to
    // be well-formed at all.
    let unpaired = vec![
        // Unpaired *scales*, but still a fused operand — so it declares an
        // arrangement. The two axes are independent, and this fixture
        // exercising one must not accidentally violate the other.
        RegionSchema {
            layout: RegionLayout::ContiguousHalves,
            ..RegionSchema::unpaired(
                0,
                RegionRole::GateUpFused,
                RegionFormat::Q6K,
                Packing::BlocksWithScalesInline,
                INTERMEDIATE,
                HIDDEN,
            )
        },
        RegionSchema::unpaired(
            1,
            RegionRole::Down,
            RegionFormat::Q6K,
            Packing::BlocksWithScalesInline,
            HIDDEN,
            INTERMEDIATE,
        ),
    ];
    let bytes = write("pairing-unpaired.weights", unpaired);
    let r = Lyrw2Reader::parse(&bytes).unwrap();
    let err = r
        .paired_region_bytes(BANK_ID, 0, RegionRole::Down, RegionRole::Scales)
        .expect_err("an unpaired region has no partner");
    assert!(
        matches!(err, Lyrw2Error::NotPaired { .. }),
        "expected NotPaired, got {err}"
    );
}

/// A role the bank does not declare at all is absent, not an error — the
/// distinction between "no such region" and "malformed pairing" is what
/// lets a caller tell an optional region from a broken one.
#[test]
fn an_absent_source_role_is_none_not_an_error() {
    let bytes = write("pairing-absent.weights", well_formed());
    let r = Lyrw2Reader::parse(&bytes).unwrap();
    assert!(r
        .paired_region_bytes(BANK_ID, 0, RegionRole::LatentIn, RegionRole::Scales)
        .unwrap()
        .is_none());
}

// ── the layout declaration is load-bearing ───────────────────────────────

/// The negative control: **identical bytes, different declared layout.**
///
/// Both containers hold the same payload, so no byte-level check can tell
/// them apart — yet they mean different matrices, because the declaration
/// is the only thing saying which rows are gate and which are up. That is
/// precisely why the fact has to be stored rather than inferred, and why a
/// reader ignoring it produces plausible output instead of an error.
#[test]
fn identical_bytes_under_different_declared_layouts_differ_only_in_the_declaration() {
    let mut interleaved = well_formed();
    interleaved[0].layout = RegionLayout::Interleaved;
    let mut contiguous = well_formed();
    contiguous[0].layout = RegionLayout::ContiguousHalves;

    let a = write("layout-interleaved.weights", interleaved);
    let b = write("layout-contiguous.weights", contiguous);

    let ra = Lyrw2Reader::parse(&a).unwrap();
    let rb = Lyrw2Reader::parse(&b).unwrap();

    // Same payload, byte for byte.
    for entry in 0..ENTRIES {
        assert_eq!(
            ra.region_bytes(BANK_ID, entry, RegionRole::GateUpFused)
                .unwrap()
                .unwrap(),
            rb.region_bytes(BANK_ID, entry, RegionRole::GateUpFused)
                .unwrap()
                .unwrap(),
            "entry {entry}: the fixture must differ ONLY in the declaration"
        );
    }

    // Different meaning, and the container is where that lives.
    let la = ra.schemas_for(BANK_ID).unwrap()[0].layout;
    let lb = rb.schemas_for(BANK_ID).unwrap()[0].layout;
    assert_ne!(la, lb);
    assert!(la.is_declared() && lb.is_declared());
}

/// And the two arrangements really do address different rows, so the
/// declaration above is not a distinction without a difference.
///
/// Uses the model-side indexing rule directly rather than re-deriving it:
/// two implementations of one addressing rule can agree with each other
/// and disagree with the model.
#[test]
fn the_two_arrangements_address_different_rows() {
    use larql_models::config::experts::{GateUpBranch, GateUpLayout};
    const HALF: usize = 4;

    let contiguous: Vec<usize> = (0..HALF)
        .map(|i| GateUpLayout::ContiguousHalves.row(GateUpBranch::Up, i, HALF))
        .collect();
    let interleaved: Vec<usize> = (0..HALF)
        .map(|i| GateUpLayout::Interleaved.row(GateUpBranch::Up, i, HALF))
        .collect();

    assert_eq!(contiguous, vec![4, 5, 6, 7]);
    assert_eq!(interleaved, vec![1, 3, 5, 7]);
    assert_ne!(
        contiguous, interleaved,
        "if these agreed, the declaration would carry no information"
    );
}

/// A region that cannot have an arrangement may not claim one — enforced
/// by the format, not by whichever writer happens to be careful.
#[test]
fn a_non_fused_region_declaring_a_layout_is_refused_by_the_plan() {
    let mut schemas = well_formed();
    schemas[1].layout = RegionLayout::Interleaved; // `Down`
    let plan = Lyrw2Plan::single_segment(LOGICAL_LAYER, bank(schemas.len() as u16), schemas);
    match plan.validate() {
        Err(Lyrw2Error::LayoutOnNonFusedRegion { role, .. }) => {
            assert!(role.contains("down"), "should name the role, got {role}");
        }
        other => panic!("expected LayoutOnNonFusedRegion, got {other:?}"),
    }
}

/// A new container must declare its fused arrangement, while a legacy one
/// that does not stays readable — the two checks are deliberately split.
#[test]
fn an_undeclared_fused_region_is_readable_but_not_writable() {
    let mut schemas = well_formed();
    schemas[0].layout = RegionLayout::Unspecified;
    let plan = Lyrw2Plan::single_segment(LOGICAL_LAYER, bank(schemas.len() as u16), schemas);

    // Readable: a schema-3 container looks exactly like this.
    assert!(plan.validate().is_ok());
    // Not writable at this schema.
    assert!(matches!(
        plan.validate_for_write(),
        Err(Lyrw2Error::UndeclaredFusedLayout { .. })
    ));
}

/// Every declared region is enumerable by schema index, including both
/// members of a duplicated role. This is what the verifier walks, and it
/// must reach the second `Scales` region that role lookup refuses.
#[test]
fn every_declared_region_is_reachable_by_schema_index() {
    let bytes = write("pairing-by-index.weights", well_formed());
    let r = Lyrw2Reader::parse(&bytes).unwrap();
    for entry in 0..ENTRIES {
        for schema in 0..well_formed().len() as u16 {
            let region = r
                .resolve_by_schema_index(BANK_ID, entry, schema)
                .unwrap()
                .unwrap_or_else(|| panic!("entry {entry} schema {schema} did not resolve"));
            assert_eq!(region.length as usize, schema_len(schema));
        }
    }
}
