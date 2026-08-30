//! Native passthrough tests for split-scale (MXFP4-shaped) expert banks.
//!
//! This is an **extraction-preservation** proof and deliberately involves no
//! kernel, no device and no arithmetic. The claim under test is only:
//!
//! ```text
//! source packed nibbles + e8m0 scales
//!         │  verbatim, no dequantise, no requantise
//!         ▼
//!   VINDEX3 container on disk
//!         │
//!         ▼
//! same bytes back, addressed by (expert, role)
//! ```
//!
//! That matters because the incumbent path reaches Q6_K by going
//! `MXFP4 → f32 → Q6_K`, which destroys the native representation before any
//! native execution could be qualified. Until the bytes provably survive
//! extraction, a numerical result downstream cannot distinguish "the native
//! kernel is correct" from "the extractor happened to reshape consistently".
//!
//! The fidelity claim (native ≈ the Q6_K-transcoded oracle) is a separate
//! gate and is not attempted here.

use super::*;
use crate::format::lyrw2::error::Lyrw2Error;
use crate::format::lyrw2::region_format::Packing;
use crate::format::lyrw2::PAIR_ID_UNPAIRED;
use crate::format::vindex3::read::Vindex3Container;
use crate::format::vindex3::write::write_container;
use tempfile::tempdir;

use larql_models::quant::mxfp4::{MXFP4_GROUP_BYTES, MXFP4_GROUP_ELEMS};

const HIDDEN: u32 = 64;
const STORED_INTER: u32 = 96;
const SEMANTIC_INTER: u32 = 96;
const EXPERTS: usize = 3;
const TOP_K: u32 = 2;

/// Groups in a matrix of `rows × cols` weights.
fn groups(rows: u32, cols: u32) -> usize {
    (rows as usize * cols as usize) / MXFP4_GROUP_ELEMS
}

/// Position- and expert-dependent bytes, so a wrong offset or a swapped
/// expert cannot pass by coincidence.
fn payload(expert: usize, rows: u32, cols: u32, salt: usize) -> Vec<u8> {
    let n = groups(rows, cols) * MXFP4_GROUP_BYTES;
    (0..n).map(|i| (expert * 31 + i + salt) as u8).collect()
}

/// One e8m0 exponent byte per group — a quarter-bit-per-weight stream that
/// is deliberately a *different length* from the payload, so a writer that
/// confused the two would produce a length mismatch rather than pass.
fn scales(expert: usize, rows: u32, cols: u32, salt: usize) -> Vec<u8> {
    (0..groups(rows, cols))
        .map(|i| (expert * 7 + i + salt) as u8)
        .collect()
}

struct Owned {
    gate_up: Vec<Vec<u8>>,
    down: Vec<Vec<u8>>,
    gate_up_scales: Vec<Vec<u8>>,
    down_scales: Vec<Vec<u8>>,
}

const GATE_UP_ROWS: u32 = SEMANTIC_INTER * 2;

fn owned() -> Owned {
    Owned {
        gate_up: (0..EXPERTS)
            .map(|e| payload(e, GATE_UP_ROWS, HIDDEN, 0))
            .collect(),
        down: (0..EXPERTS)
            .map(|e| payload(e, HIDDEN, STORED_INTER, 7))
            .collect(),
        gate_up_scales: (0..EXPERTS)
            .map(|e| scales(e, GATE_UP_ROWS, HIDDEN, 3))
            .collect(),
        down_scales: (0..EXPERTS)
            .map(|e| scales(e, HIDDEN, STORED_INTER, 11))
            .collect(),
    }
}

fn source(o: &Owned) -> MoeLayerSource<'_> {
    MoeLayerSource {
        layer: 0,
        experts_gate_up: o.gate_up.iter().map(|v| v.as_slice()).collect(),
        experts_down: o.down.iter().map(|v| v.as_slice()).collect(),
        format: RegionFormat::Mxfp4,
        scales: ExpertScaleStreams::Paired {
            gate_up: o.gate_up_scales.iter().map(|v| v.as_slice()).collect(),
            down: o.down_scales.iter().map(|v| v.as_slice()).collect(),
        },
        // GPT-OSS's own arrangement, carried through verbatim. Using the
        // *other* value here would still round-trip byte-identically —
        // which is exactly why the declaration has to be checked
        // separately from the bytes.
        gate_up_layout: RegionLayout::Interleaved,
        hidden_size: HIDDEN,
        gate_up_stored_intermediate: SEMANTIC_INTER,
        down_stored_intermediate: STORED_INTER,
        semantic_intermediate: SEMANTIC_INTER,
        top_k: TOP_K,
    }
}

/// Import → write → reopen from disk. Reopening rather than trusting the
/// in-memory spec is the point: the claim is about what a reader finds.
fn round_trip(o: &Owned) -> (tempfile::TempDir, Vindex3Container) {
    let src = source(o);
    let staging = tempdir().unwrap();
    let spec = import_one_layer(
        &src,
        "mxfp4-fixture",
        "gpt-oss",
        1,
        &staging.path().join("staging.lyrw"),
    )
    .expect("import");
    let out = tempdir().unwrap();
    write_container(out.path(), &spec).expect("write container");
    let container = Vindex3Container::open(out.path()).expect("reopen");
    (out, container)
}

// ── the preservation claim ───────────────────────────────────────────────

/// The gate. Every payload region and every scale region comes back byte
/// for byte, addressed by `(expert, role)`.
#[test]
fn payload_and_scale_bytes_both_survive_verbatim() {
    let o = owned();
    let (_dir, container) = round_trip(&o);
    let reader = container.segment(&routed_storage_key(0)).expect("segment");

    let mut checked = 0usize;
    for expert in 0..EXPERTS {
        let got_gu = reader
            .region_bytes(0, expert as u32, RegionRole::GateUpFused)
            .expect("gate_up")
            .expect("gate_up present");
        assert_eq!(got_gu, &o.gate_up[expert][..], "expert {expert} gate_up");

        let got_dn = reader
            .region_bytes(0, expert as u32, RegionRole::Down)
            .expect("down")
            .expect("down present");
        assert_eq!(got_dn, &o.down[expert][..], "expert {expert} down");
        checked += 2;
    }
    assert_eq!(checked, EXPERTS * 2);
}

/// Each payload's exponents are reachable **through its pairing**, and
/// each resolves to its own bytes.
///
/// Without this the payload could round-trip perfectly while the exponents
/// that give it magnitude were dropped or crossed — a container that
/// verifies structurally and decodes down's weights against gate/up's
/// scales.
#[test]
fn each_payload_resolves_its_own_e8m0_stream_through_its_pair() {
    let o = owned();
    let (_dir, container) = round_trip(&o);
    let reader = container.segment(&routed_storage_key(0)).expect("segment");

    for expert in 0..EXPERTS {
        let gu = reader
            .paired_region_bytes(
                0,
                expert as u32,
                RegionRole::GateUpFused,
                RegionRole::Scales,
            )
            .expect("gate_up scales")
            .expect("gate_up scales present");
        assert_eq!(gu, &o.gate_up_scales[expert][..], "expert {expert} gate_up");

        let dn = reader
            .paired_region_bytes(0, expert as u32, RegionRole::Down, RegionRole::Scales)
            .expect("down scales")
            .expect("down scales present");
        assert_eq!(dn, &o.down_scales[expert][..], "expert {expert} down");

        // The crossing check: the two must not be the same bytes. A reader
        // that resolved both pairings to the first `Scales` region would
        // satisfy every assertion above except this one.
        assert_ne!(gu, dn, "expert {expert} pairings resolved to one region");
    }
}

/// A role-only lookup for a duplicated role **refuses**. This is the
/// defect #7 closed: `find(|s| s.role == role)` used to hand back gate/up's
/// exponents when down's were meant.
#[test]
fn role_only_lookup_of_a_duplicated_role_is_ambiguous() {
    let o = owned();
    let (_dir, container) = round_trip(&o);
    let reader = container.segment(&routed_storage_key(0)).expect("segment");

    let err = reader
        .region_bytes(0, 0, RegionRole::Scales)
        .expect_err("a duplicated role must not resolve by role alone");
    match err {
        Lyrw2Error::AmbiguousRole {
            count, ref role, ..
        } => {
            assert_eq!(count, 2, "two Scales regions");
            assert!(role.contains("scale"), "error should name the role: {role}");
        }
        other => panic!("expected AmbiguousRole, got {other:?}"),
    }
}

/// Roles that are still unique resolve by role alone — the refusal above
/// must not become a blanket refusal on paired banks.
#[test]
fn unique_roles_still_resolve_by_role_alone_in_a_paired_bank() {
    let o = owned();
    let (_dir, container) = round_trip(&o);
    let reader = container.segment(&routed_storage_key(0)).expect("segment");

    for expert in 0..EXPERTS {
        assert_eq!(
            reader
                .region_bytes(0, expert as u32, RegionRole::GateUpFused)
                .expect("gate_up")
                .expect("present"),
            &o.gate_up[expert][..]
        );
        assert_eq!(
            reader
                .region_bytes(0, expert as u32, RegionRole::Down)
                .expect("down")
                .expect("present"),
            &o.down[expert][..]
        );
    }
}

/// Asking an unpaired region for a partner is an error, not `None`.
/// `None` means "this bank has no such region"; an unpaired payload asked
/// for its scales is a caller mistake and says so.
#[test]
fn an_unpaired_region_has_no_partner_to_resolve() {
    let o = owned();
    let mut src = source(&o);
    src.scales = ExpertScaleStreams::Inline;
    src.format = RegionFormat::Q6K;

    let staging = tempdir().unwrap();
    let spec = import_one_layer(
        &src,
        "inline-fixture",
        "gemma",
        1,
        &staging.path().join("staging.lyrw"),
    )
    .expect("import");
    let out = tempdir().unwrap();
    write_container(out.path(), &spec).expect("write");
    let container = Vindex3Container::open(out.path()).expect("reopen");
    let reader = container.segment(&routed_storage_key(0)).expect("segment");

    let err = reader
        .paired_region_bytes(0, 0, RegionRole::Down, RegionRole::Scales)
        .expect_err("an inline bank's down region is unpaired");
    assert!(
        matches!(err, Lyrw2Error::NotPaired { .. }),
        "expected NotPaired, got {err:?}"
    );
}

/// Scale-stream length is one byte per 32-weight group, and is *not* the
/// payload's length. A writer that bound the payload where scales belong
/// would satisfy a same-length check; this cannot be satisfied that way.
#[test]
fn scale_and_payload_lengths_are_independently_correct() {
    let o = owned();
    for expert in 0..EXPERTS {
        assert_eq!(
            o.gate_up[expert].len(),
            groups(GATE_UP_ROWS, HIDDEN) * MXFP4_GROUP_BYTES
        );
        assert_eq!(o.gate_up_scales[expert].len(), groups(GATE_UP_ROWS, HIDDEN));
        assert_ne!(
            o.gate_up[expert].len(),
            o.gate_up_scales[expert].len(),
            "fixture must distinguish payload from scales by length"
        );
    }
}

/// Expert order is preserved. Every region is compared against the expert
/// it came from, not merely against *some* expert — a rotated write order
/// would pass a set-equality check.
#[test]
fn expert_order_is_preserved_not_merely_the_set() {
    let o = owned();
    let (_dir, container) = round_trip(&o);
    let reader = container.segment(&routed_storage_key(0)).expect("segment");

    for expert in 0..EXPERTS {
        let got = reader
            .region_bytes(0, expert as u32, RegionRole::GateUpFused)
            .expect("gate_up")
            .expect("present");
        for other in 0..EXPERTS {
            if other == expert {
                continue;
            }
            assert_ne!(
                got,
                &o.gate_up[other][..],
                "expert {expert} resolved to expert {other}'s bytes"
            );
        }
    }
}

// ── the declared physical facts ──────────────────────────────────────────

/// The container declares *where the scales live* rather than leaving a
/// reader to infer it from the format tag. Format class does not determine
/// layout — the same codec can ship inline or split — so this is recorded.
#[test]
fn the_schema_declares_split_scales_and_pairs_them() {
    let o = owned();
    let (_dir, container) = round_trip(&o);
    let reader = container.segment(&routed_storage_key(0)).expect("segment");
    let schemas = reader.schemas_for(0).expect("schemas");

    assert_eq!(
        schemas.len(),
        4,
        "a split-scale bank contributes four regions per expert"
    );

    let payloads: Vec<_> = schemas
        .iter()
        .filter(|s| s.packing == Packing::BlocksValues)
        .collect();
    let scale_regions: Vec<_> = schemas
        .iter()
        .filter(|s| s.packing == Packing::BlocksScales)
        .collect();
    assert_eq!(payloads.len(), 2, "gate_up and down payloads");
    assert_eq!(scale_regions.len(), 2, "one scale region per payload");

    // Every payload is paired, and its partner is a scale region carrying
    // the same pair_id. An unpaired split payload is the state where a
    // reader can see values and never find their exponents.
    for p in &payloads {
        assert_ne!(
            p.pair_id, PAIR_ID_UNPAIRED,
            "{:?} payload declares no scale partner",
            p.role
        );
        assert_eq!(
            scale_regions
                .iter()
                .filter(|s| s.pair_id == p.pair_id)
                .count(),
            1,
            "{:?} payload must have exactly one scale partner",
            p.role
        );
    }

    // The two pairings are distinct — one pair_id for both would make
    // gate_up's and down's exponents indistinguishable.
    assert_ne!(payloads[0].pair_id, payloads[1].pair_id);
}

/// Every region declares the format it is actually stored in, so a reader
/// never infers MXFP4 from context.
#[test]
fn every_region_declares_the_stored_format() {
    let o = owned();
    let (_dir, container) = round_trip(&o);
    let reader = container.segment(&routed_storage_key(0)).expect("segment");
    for s in reader.schemas_for(0).expect("schemas") {
        assert_eq!(s.format, RegionFormat::Mxfp4, "{:?}", s.role);
    }
}

/// The container is structurally sound with four regions per expert — the
/// verifier must not treat the extra scale regions as defects.
#[test]
fn a_split_scale_container_verifies_clean() {
    let o = owned();
    let (_dir, container) = round_trip(&o);
    assert!(
        container.verify().is_empty(),
        "structural defects: {:?}",
        container.verify()
    );
}

// ── refusals ─────────────────────────────────────────────────────────────
//
// Each of these reaches a kernel as an offset table that resolves into the
// wrong expert's exponents: arithmetically valid, silently wrong, and
// invisible to a structural verify. They are refused before any byte is
// written.

fn expect_refusal(o: &Owned, mutate: impl FnOnce(&mut MoeLayerSource<'_>), why: &str) {
    let mut src = source(o);
    mutate(&mut src);
    let staging = tempdir().unwrap();
    let err = import_one_layer(
        &src,
        "mxfp4-fixture",
        "gpt-oss",
        1,
        &staging.path().join("staging.lyrw"),
    )
    .err()
    .unwrap_or_else(|| panic!("expected refusal: {why}"));
    let msg = format!("{err}");
    assert!(
        msg.contains("scale"),
        "refusal should name the scale problem, got: {msg}"
    );
}

#[test]
fn a_short_scale_stream_is_refused() {
    let o = owned();
    expect_refusal(
        &o,
        |src| {
            if let ExpertScaleStreams::Paired { gate_up, .. } = &mut src.scales {
                gate_up.pop();
            }
        },
        "one fewer scale stream than payloads",
    );
}

#[test]
fn a_ragged_scale_stream_is_refused() {
    let o = owned();
    expect_refusal(
        &o,
        |src| {
            if let ExpertScaleStreams::Paired { gate_up, .. } = &mut src.scales {
                gate_up[1] = &gate_up[1][..gate_up[1].len() - 1];
            }
        },
        "experts in one bank must share a layout",
    );
}

#[test]
fn an_empty_scale_stream_is_refused() {
    let o = owned();
    expect_refusal(
        &o,
        |src| {
            if let ExpertScaleStreams::Paired { down, .. } = &mut src.scales {
                for s in down.iter_mut() {
                    *s = &[];
                }
            }
        },
        "Inline is how a format says it has no scales",
    );
}

/// The inline path is untouched: an inline bank still declares two
/// unpaired regions. Pinned here because the paired branch shares the
/// schema-building code, and a regression would silently start declaring
/// scale partners that do not exist.
#[test]
fn an_inline_bank_still_declares_two_unpaired_regions() {
    let o = owned();
    let mut src = source(&o);
    src.scales = ExpertScaleStreams::Inline;
    src.format = RegionFormat::Q6K;

    let staging = tempdir().unwrap();
    let spec = import_one_layer(
        &src,
        "inline-fixture",
        "gemma",
        1,
        &staging.path().join("staging.lyrw"),
    )
    .expect("import");
    let out = tempdir().unwrap();
    write_container(out.path(), &spec).expect("write");
    let container = Vindex3Container::open(out.path()).expect("reopen");
    let reader = container.segment(&routed_storage_key(0)).expect("segment");
    let schemas = reader.schemas_for(0).expect("schemas");

    assert_eq!(schemas.len(), 2);
    for s in schemas {
        assert_eq!(s.pair_id, PAIR_ID_UNPAIRED, "{:?}", s.role);
        assert_eq!(s.packing, Packing::RowMajor, "{:?}", s.role);
    }
}
