//! Tests for the c8 importer.
//!
//! The gate that matters is the last one: a container the importer built must
//! hand back **the same bytes** the source held, per expert, per role. That is
//! what makes a downstream execution comparison meaningful — if the importer
//! were free to reshape on the way in, bit-identity downstream would only
//! prove the reshape was self-consistent.

use super::*;
use crate::format::lyrw2::read::Lyrw2Reader;
use crate::format::vindex3::read::Vindex3Container;
use crate::format::vindex3::write::write_container;
use tempfile::tempdir;

const HIDDEN: u32 = 8;
const STORED_INTER: u32 = 12;
const SEMANTIC_INTER: u32 = 10;
const EXPERTS: usize = 3;
const TOP_K: u32 = 2;

/// Distinct, position-dependent bytes so a wrong offset cannot pass.
///
/// Sized at the **semantic** width: gate_up is never padded (only down is), so
/// a fixture padding both equally could not distinguish a correct importer
/// from one that describes gate_up using down's stored width.
fn gate_up_bytes(expert: usize) -> Vec<u8> {
    let n = (SEMANTIC_INTER * 2 * HIDDEN) as usize * 4;
    (0..n).map(|i| (expert * 31 + i) as u8).collect()
}

fn down_bytes(expert: usize) -> Vec<u8> {
    let n = (HIDDEN * STORED_INTER) as usize * 4;
    (0..n).map(|i| (expert * 17 + i + 7) as u8).collect()
}

struct Owned {
    gate_up: Vec<Vec<u8>>,
    down: Vec<Vec<u8>>,
}

fn owned() -> Owned {
    Owned {
        gate_up: (0..EXPERTS).map(gate_up_bytes).collect(),
        down: (0..EXPERTS).map(down_bytes).collect(),
    }
}

fn source(o: &Owned) -> MoeLayerSource<'_> {
    MoeLayerSource {
        layer: 0,
        experts_gate_up: o.gate_up.iter().map(|v| v.as_slice()).collect(),
        experts_down: o.down.iter().map(|v| v.as_slice()).collect(),
        format: RegionFormat::F32,
        scales: ExpertScaleStreams::Inline,
        gate_up_layout: RegionLayout::ContiguousHalves,
        hidden_size: HIDDEN,
        gate_up_stored_intermediate: SEMANTIC_INTER,
        down_stored_intermediate: STORED_INTER,
        semantic_intermediate: SEMANTIC_INTER,
        top_k: TOP_K,
    }
}

// ── The verbatim guarantee ───────────────────────────────────────────────

#[test]
fn every_region_round_trips_byte_for_byte_through_a_written_container() {
    // The c8 gate. Import → write → open → read back, comparing each expert's
    // gate_up and down against the source slice it came from.
    let o = owned();
    let src = source(&o);
    let staging = tempdir().unwrap();
    let spec = import_one_layer(
        &src,
        "gemma-fixture",
        "gemma",
        1,
        &staging.path().join("seg.lyrw"),
    )
    .expect("import");

    let root = tempdir().unwrap();
    write_container(root.path(), &spec).expect("write container");
    let container = Vindex3Container::open(root.path()).expect("open");

    let key = routed_storage_key(0);
    let reader = container.segment(&key).expect("segment parses");
    for expert in 0..EXPERTS as u32 {
        let gu = reader
            .region_bytes(0, expert, RegionRole::GateUpFused)
            .expect("resolve gate_up")
            .expect("gate_up present");
        assert_eq!(
            gu,
            o.gate_up[expert as usize].as_slice(),
            "expert {expert}: gate_up came back changed"
        );
        let dn = reader
            .region_bytes(0, expert, RegionRole::Down)
            .expect("resolve down")
            .expect("down present");
        assert_eq!(
            dn,
            o.down[expert as usize].as_slice(),
            "expert {expert}: down came back changed"
        );
    }
}

#[test]
fn experts_are_not_transposed_with_each_other() {
    // A single off-by-one in entry ordering would still round-trip *some*
    // bytes; this pins that expert e gets expert e's, which the distinct
    // per-expert seeds above make detectable.
    let o = owned();
    let staging = tempdir().unwrap();
    let bytes =
        segment_bytes_for_layer(&source(&o), &staging.path().join("s.lyrw")).expect("segment");
    let reader = Lyrw2Reader::parse(&bytes).expect("parse");
    for expert in 0..EXPERTS as u32 {
        let gu = reader
            .region_bytes(0, expert, RegionRole::GateUpFused)
            .unwrap()
            .unwrap();
        assert_eq!(gu[0], (expert as usize * 31) as u8, "expert {expert} seed");
    }
}

// ── What the container declares ──────────────────────────────────────────

#[test]
fn the_manifest_records_the_semantic_width_not_the_stored_one() {
    // Gemma pads the intermediate axis. The bank descriptor carries the stored
    // extent because that is what the bytes contain; `expert_dims` carries the
    // semantic width because that is what the operation means. Collapsing the
    // two is the `physical shape != semantic operand shape` bug the Gemma
    // routing-parity work already had to fix once.
    let o = owned();
    let staging = tempdir().unwrap();
    let spec = import_one_layer(&source(&o), "m", "gemma", 1, &staging.path().join("s.lyrw"))
        .expect("import");
    let dims = spec.manifest.layers[0]
        .routed_bank
        .expert_dims
        .expect("expert dims");
    assert_eq!(dims.intermediate, SEMANTIC_INTER);

    let reader = Lyrw2Reader::parse(&spec.segments[0].bytes).expect("parse");
    assert_eq!(
        reader.banks()[0].intermediate_dim,
        STORED_INTER,
        "the bank must carry the stored extent"
    );
}

#[test]
fn the_imported_container_declares_one_routed_bank_under_gated_mlp() {
    let o = owned();
    let staging = tempdir().unwrap();
    let spec = import_one_layer(&source(&o), "m", "gemma", 1, &staging.path().join("s.lyrw"))
        .expect("import");
    let layer = &spec.manifest.layers[0];
    assert_eq!(layer.routed_bank.programme, GATED_MLP_PROGRAMME);
    assert_eq!(layer.routed_bank.experts, EXPERTS as u32);
    assert!(
        layer.routed_bank.resolve_programme().is_some(),
        "the declared programme must resolve, or nothing can bind it"
    );
    assert_eq!(spec.segments.len(), 1);
    assert_eq!(spec.segments[0].key, routed_storage_key(0));
}

#[test]
fn an_imported_container_is_bindable() {
    // The whole point: `verify` must find no structural defect in a container
    // this module produced, or the import is describing something unusable.
    let o = owned();
    let staging = tempdir().unwrap();
    let spec = import_one_layer(&source(&o), "m", "gemma", 1, &staging.path().join("s.lyrw"))
        .expect("import");
    let root = tempdir().unwrap();
    write_container(root.path(), &spec).expect("write");
    let container = Vindex3Container::open(root.path()).expect("open");
    assert_eq!(container.verify(), Vec::new(), "imported container defects");
}

// ── Refusals, before any byte is written ─────────────────────────────────

#[test]
fn a_source_with_no_experts_is_refused() {
    let o = Owned {
        gate_up: vec![],
        down: vec![],
    };
    let staging = tempdir().unwrap();
    let err = segment_bytes_for_layer(&source(&o), &staging.path().join("s.lyrw"))
        .expect_err("an empty bank must not import");
    assert!(format!("{err}").contains("no experts"), "{err}");
}

#[test]
fn mismatched_gate_up_and_down_counts_are_refused() {
    let mut o = owned();
    o.down.pop();
    let staging = tempdir().unwrap();
    let err = segment_bytes_for_layer(&source(&o), &staging.path().join("s.lyrw"))
        .expect_err("every expert needs both regions");
    assert!(
        format!("{err}").contains("every expert needs both"),
        "{err}"
    );
}

#[test]
fn a_ragged_expert_layout_is_refused_naming_the_expert() {
    // A source whose experts disagree on size would otherwise be baked into a
    // container that fails at bind with nothing pointing back here.
    let mut o = owned();
    o.gate_up[1].truncate(8);
    let staging = tempdir().unwrap();
    let err = segment_bytes_for_layer(&source(&o), &staging.path().join("s.lyrw"))
        .expect_err("ragged experts must not import");
    let msg = format!("{err}");
    assert!(msg.contains("expert 1"), "must name the odd expert: {msg}");
}

#[test]
fn a_top_k_larger_than_the_bank_is_refused() {
    let o = owned();
    let mut src = source(&o);
    src.top_k = EXPERTS as u32 + 1;
    let staging = tempdir().unwrap();
    let err = segment_bytes_for_layer(&src, &staging.path().join("s.lyrw"))
        .expect_err("top_k must be selectable");
    assert!(format!("{err}").contains("not selectable"), "{err}");
}

#[test]
fn a_semantic_width_wider_than_the_stored_one_is_refused() {
    // A view narrows the stored extent; it cannot invent columns. Accepting
    // this would produce a container whose kernel reads past its own bytes.
    let o = owned();
    let mut src = source(&o);
    src.semantic_intermediate = STORED_INTER + 1;
    let staging = tempdir().unwrap();
    let err = segment_bytes_for_layer(&src, &staging.path().join("s.lyrw"))
        .expect_err("a widening view must not import");
    let msg = format!("{err}");
    // The refusal names which region it is too wide for, because the two
    // regions carry different stored widths and "the stored one" is ambiguous.
    assert!(msg.contains("exceeds the"), "{msg}");
    assert!(
        msg.contains("gate_up") || msg.contains("down"),
        "the refusal must name the region whose stored width was exceeded: {msg}"
    );
}

#[test]
fn each_region_is_declared_at_its_own_stored_width() {
    // The defect this pins: `gate_up` and `down` do not share a stored width.
    // gate_up is `[2 x inter, hidden]` and never padded; down is
    // `[hidden, inter]` with inter padded to the next super-block. Describing
    // both with down's width over-declares gate_up by the padding — and
    // nothing downstream catches it, because regions are copied verbatim (so
    // the declared shape never enters the byte length) and `verify` checks
    // structure rather than shape-against-length.
    let o = owned();
    let src = source(&o);
    assert_ne!(
        src.gate_up_stored_intermediate, src.down_stored_intermediate,
        "the fixture must exercise the asymmetry or this test proves nothing"
    );

    let staging = tempdir().unwrap();
    let path = staging.path().join("s.lyrw");
    let bytes = segment_bytes_for_layer(&src, &path).unwrap();
    let reader = Lyrw2Reader::parse(&bytes).unwrap();

    let schemas = reader.schemas_for(0).expect("bank 0 has schemas");
    let gate_up = schemas
        .iter()
        .find(|s| s.role == RegionRole::GateUpFused)
        .expect("a gate_up schema");
    let down = schemas
        .iter()
        .find(|s| s.role == RegionRole::Down)
        .expect("a down schema");

    assert_eq!(
        gate_up.rows,
        SEMANTIC_INTER * 2,
        "gate_up must be declared at its own (unpadded) width"
    );
    assert_eq!(gate_up.cols, HIDDEN);
    assert_eq!(down.rows, HIDDEN);
    assert_eq!(
        down.cols, STORED_INTER,
        "down must be declared at its own (padded) width"
    );

    // And the declaration must match what is actually there.
    let region = reader
        .region_bytes(0, 0, RegionRole::GateUpFused)
        .unwrap()
        .unwrap();
    assert_eq!(
        region.len(),
        (gate_up.rows * gate_up.cols) as usize * 4,
        "declared gate_up shape does not account for the region's bytes"
    );
}

// ── Format naming ────────────────────────────────────────────────────────

#[test]
fn a_representable_expert_format_maps_to_its_region_format() {
    use larql_compute::QuantFormat;
    assert_eq!(
        region_format_for(QuantFormat::Q4_K).unwrap(),
        RegionFormat::Q4K
    );
    assert_eq!(
        region_format_for(QuantFormat::F32).unwrap(),
        RegionFormat::F32
    );
    // The native route's reason for existing: MXFP4 reaches a container
    // as MXFP4 rather than being transcoded on the way in. The mapping
    // only says the encoding is *describable* — its scales still have to
    // be supplied as partner regions.
    assert_eq!(
        region_format_for(QuantFormat::MXFP4).unwrap(),
        RegionFormat::Mxfp4
    );
}

#[test]
fn an_unrepresentable_expert_format_is_refused_not_relabelled() {
    // The whole importer exists to avoid silent conversion, so a format the
    // container cannot name must stop the import rather than be written under
    // a neighbouring tag. Mislabelling here would surface as wrong numbers at
    // execution, with nothing pointing back to the import.
    use larql_compute::QuantFormat;
    for unnameable in [QuantFormat::Q6_K, QuantFormat::BF16, QuantFormat::Q8_0] {
        let err = region_format_for(unnameable)
            .expect_err("a format with no VINDEX3 region format must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("no VINDEX3 region format") && msg.contains("transcode"),
            "the refusal must say why it cannot proceed, got: {msg}"
        );
    }
}

// ── Per-region width refusals ────────────────────────────────────────────

#[test]
fn a_semantic_width_wider_than_the_down_region_is_refused_naming_down() {
    // The gate_up arm is checked first, so a source that only violates the
    // down width is the one that proves the second arm is reachable — and
    // that the message names the region rather than saying "the stored one".
    let o = owned();
    let mut src = source(&o);
    src.gate_up_stored_intermediate = STORED_INTER * 2;
    src.down_stored_intermediate = SEMANTIC_INTER - 1;
    let staging = tempdir().unwrap();
    let err = segment_bytes_for_layer(&src, &staging.path().join("s.lyrw"))
        .expect_err("a view wider than the down region must not import");
    let msg = format!("{err}");
    assert!(msg.contains("down"), "must name the down region: {msg}");
}

// ── Segment placement ────────────────────────────────────────────────────

#[test]
fn writing_a_segment_creates_the_directories_its_key_implies() {
    // A storage key is a path (`routed/layer_000`), composed rather than
    // globbed, so its directory is part of the key's meaning — the c9 builder
    // relies on that instead of pre-creating a tree it would have to keep in
    // step with the key format.
    let o = owned();
    let src = source(&o);
    let root = tempdir().unwrap();
    let nested = root.path().join("routed").join("deeper").join("s.lyrw");
    assert!(!nested.parent().unwrap().exists());

    write_segment_file(&src, &nested).unwrap();
    assert!(nested.exists(), "segment file was not placed at its key");
}
