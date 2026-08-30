//! Byte-level proof for the native-MXFP4 VINDEX2 layer store.
//!
//! The numbers come later; these pin the bytes. Every assertion here is
//! about what a writer put on disk and what a reader takes back off it —
//! payload ranges, *independent* e8m0 ranges, entry ordering, and the two
//! arrangement facts — because a store that round-trips its geometry
//! incorrectly still decodes to finite, plausible activations. `docs/
//! k3-funnel.md` §4.7 is the record of what that costs.

use larql_models::config::experts::GateUpLayout;
use larql_vindex::format::filenames::layer_weights_filename;
use larql_vindex::format::weights::layer_store_layout::LayerScaleBinding;
use larql_vindex::format::weights::write_layers::{
    parse_layer_weights_header, write_layer_weights, write_layer_weights_with_layout, LayerEntry,
    LayerEntryScales, LayerWeightFormat,
};
use tempfile::tempdir;

const LAYER: usize = 3;
const INTER: usize = 8;
const HIDDEN: usize = 16;

/// Distinct, position-encoding payloads: every stream gets its own byte
/// value so a swapped range shows up as a wrong value, not a wrong length.
/// Two experts with *different* sizes, because equal sizes let an
/// off-by-one-entry bug read the neighbour's range and still pass.
fn split_scale_entries() -> Vec<LayerEntry> {
    vec![
        LayerEntry {
            gate_up: vec![0x11; 96],
            down: vec![0x22; 48],
            scales: Some(LayerEntryScales {
                gate_up: vec![0x33; 6],
                down: vec![0x44; 3],
            }),
        },
        LayerEntry {
            gate_up: vec![0x55; 64],
            down: vec![0x66; 32],
            scales: Some(LayerEntryScales {
                gate_up: vec![0x77; 4],
                down: vec![0x88; 2],
            }),
        },
    ]
}

fn write_split_scale(dir: &std::path::Path, layout: GateUpLayout) -> Vec<u8> {
    write_layer_weights_with_layout(
        dir,
        LAYER,
        LayerWeightFormat::MXFP4,
        &split_scale_entries(),
        INTER,
        HIDDEN,
        LayerScaleBinding::SplitE8M0,
        layout,
    )
    .unwrap();
    std::fs::read(dir.join(layer_weights_filename(LAYER))).unwrap()
}

#[test]
fn every_stream_round_trips_to_its_own_bytes() {
    let dir = tempdir().unwrap();
    let bytes = write_split_scale(dir.path(), GateUpLayout::Interleaved);
    let h = parse_layer_weights_header(&bytes).unwrap();
    let entries = split_scale_entries();

    assert_eq!(h.format, LayerWeightFormat::MXFP4);
    assert_eq!(h.num_entries, entries.len());
    assert_eq!(h.entries.len(), entries.len());

    for (e, (parsed, written)) in h.entries.iter().zip(entries.iter()).enumerate() {
        let take = |r: larql_vindex::format::weights::write_layers::StoredRange| {
            bytes[r.offset..r.offset + r.len].to_vec()
        };
        assert_eq!(take(parsed.gate_up), written.gate_up, "expert {e} gate_up");
        assert_eq!(take(parsed.down), written.down, "expert {e} down");
        let w = written.scales.as_ref().unwrap();
        assert_eq!(
            take(parsed.gate_up_scales.unwrap()),
            w.gate_up,
            "expert {e} gate_up e8m0"
        );
        assert_eq!(
            take(parsed.down_scales.unwrap()),
            w.down,
            "expert {e} down e8m0"
        );
    }
}

#[test]
fn exponent_ranges_are_stored_not_derived_from_the_payload_offset() {
    // `payload_offset / 16` is true of two parallel banks and of nothing
    // else. This writer interleaves payload and exponents per expert, so if
    // any consumer ever re-derives the exponent offset it lands in the
    // middle of a payload — and MXFP4 decodes whatever it finds.
    let dir = tempdir().unwrap();
    let bytes = write_split_scale(dir.path(), GateUpLayout::Interleaved);
    let h = parse_layer_weights_header(&bytes).unwrap();

    for (e, entry) in h.entries.iter().enumerate() {
        let derived = entry.gate_up.offset / 16;
        let actual = entry.gate_up_scales.unwrap().offset;
        assert_ne!(
            derived, actual,
            "expert {e}: the derived offset coincides with the stored one, so this \
             test cannot tell a deriving reader from a reading one — change the \
             fixture's sizes"
        );
    }
}

#[test]
fn regions_stay_inside_the_file_and_do_not_overlap() {
    let dir = tempdir().unwrap();
    let bytes = write_split_scale(dir.path(), GateUpLayout::Interleaved);
    let h = parse_layer_weights_header(&bytes).unwrap();

    let mut spans: Vec<(usize, usize, String)> = Vec::new();
    for (e, entry) in h.entries.iter().enumerate() {
        for (name, r) in [
            ("gate_up", Some(entry.gate_up)),
            ("down", Some(entry.down)),
            ("gate_up_scales", entry.gate_up_scales),
            ("down_scales", entry.down_scales),
        ] {
            let r = r.unwrap();
            assert!(
                r.offset + r.len <= bytes.len(),
                "expert {e} {name} overruns the file"
            );
            spans.push((r.offset, r.offset + r.len, format!("expert {e} {name}")));
        }
    }
    spans.sort();
    for w in spans.windows(2) {
        assert!(
            w[0].1 <= w[1].0,
            "{} overlaps {} — one expert's exponents would decode another's payload",
            w[0].2,
            w[1].2
        );
    }
    // The table must also not overlap the first payload.
    assert!(
        spans[0].0 >= 32 + h.num_entries * 64,
        "data runs into the table"
    );
}

#[test]
fn the_layout_field_is_carried_not_implied_by_the_format() {
    // The whole point of the layout block: the SAME format and the SAME
    // binding must be able to say either arrangement. If MXFP4 implied
    // Interleaved these two would be indistinguishable on disk.
    let dir_i = tempdir().unwrap();
    let dir_c = tempdir().unwrap();
    let bytes_i = write_split_scale(dir_i.path(), GateUpLayout::Interleaved);
    let bytes_c = write_split_scale(dir_c.path(), GateUpLayout::ContiguousHalves);

    let hi = parse_layer_weights_header(&bytes_i).unwrap();
    let hc = parse_layer_weights_header(&bytes_c).unwrap();
    assert_eq!(hi.fused_row_layout, GateUpLayout::Interleaved);
    assert_eq!(hc.fused_row_layout, GateUpLayout::ContiguousHalves);
    assert_eq!(hi.format, hc.format);
    assert_eq!(hi.scale_binding, hc.scale_binding);
    // Identical payloads: the files differ in exactly one word.
    assert_ne!(bytes_i, bytes_c);
    assert_eq!(bytes_i.len(), bytes_c.len());
    let diffs: Vec<usize> = (0..bytes_i.len())
        .filter(|&i| bytes_i[i] != bytes_c[i])
        .collect();
    assert!(
        diffs.iter().all(|&i| (28..32).contains(&i)),
        "the arrangement must live in the layout block alone, differing bytes: {diffs:?}"
    );
}

#[test]
fn an_inline_store_reports_the_arrangement_its_writer_produced() {
    // A pre-MXFP4 store carries no layout block. The reader must still
    // return a definite arrangement — the one those writers actually emit —
    // rather than an "unknown" that a consumer then guesses about.
    let dir = tempdir().unwrap();
    let entries = vec![LayerEntry {
        gate_up: vec![1, 2, 3, 4],
        down: vec![5, 6],
        scales: None,
    }];
    write_layer_weights(
        dir.path(),
        LAYER,
        LayerWeightFormat::Q6_K,
        &entries,
        INTER,
        HIDDEN,
    )
    .unwrap();
    let bytes = std::fs::read(dir.path().join(layer_weights_filename(LAYER))).unwrap();
    let h = parse_layer_weights_header(&bytes).unwrap();

    assert_eq!(h.scale_binding, LayerScaleBinding::Inline);
    assert_eq!(h.fused_row_layout, GateUpLayout::ContiguousHalves);
    assert!(h.entries[0].gate_up_scales.is_none());
    assert!(h.entries[0].down_scales.is_none());
    // Narrow stride: first payload sits directly after a 4-field table.
    assert_eq!(h.entries[0].gate_up.offset, 24 + 32);
}

#[test]
fn a_pre_mxfp4_reader_refuses_the_store_instead_of_misparsing_it() {
    // The compatibility argument for NOT bumping `format_version`, pinned.
    // An older build's parser is this one minus the `8 => MXFP4` arm; it
    // rejects on the quant code before reading `num_entries` or the table,
    // so it can never apply the narrow stride to a wide table.
    let dir = tempdir().unwrap();
    let bytes = write_split_scale(dir.path(), GateUpLayout::Interleaved);

    assert_eq!(&bytes[0..4], b"LYRW", "magic must stay put");
    assert_eq!(
        u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        1,
        "format_version must stay 1 — bumping it would make every EXISTING \
         store unreadable by the same older build this is protecting"
    );
    assert_eq!(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        LayerWeightFormat::MXFP4.as_u32(),
    );

    // Simulate the older parser: same bytes, no MXFP4 arm.
    let quant = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let older_reader_accepts = matches!(quant, 0..=7);
    assert!(
        !older_reader_accepts,
        "an older reader must refuse this file at the quant code"
    );
}

#[test]
fn a_split_binding_is_refused_on_a_format_with_nowhere_to_record_it() {
    // Q6_K carries no layout block, so a split binding could not be read
    // back — the reader would derive Inline and parse the wide table at the
    // narrow stride, yielding in-bounds offsets from the wrong places.
    let dir = tempdir().unwrap();
    let err = write_layer_weights_with_layout(
        dir.path(),
        LAYER,
        LayerWeightFormat::Q6_K,
        &split_scale_entries(),
        INTER,
        HIDDEN,
        LayerScaleBinding::SplitE8M0,
        GateUpLayout::ContiguousHalves,
    )
    .unwrap_err();
    assert!(
        format!("{err}").contains("layout block"),
        "refusal must name the cause: {err}"
    );
}

#[test]
fn an_interleaved_layout_is_refused_on_a_format_with_nowhere_to_record_it() {
    let dir = tempdir().unwrap();
    let entries = vec![LayerEntry {
        gate_up: vec![1, 2],
        down: vec![3, 4],
        scales: None,
    }];
    let err = write_layer_weights_with_layout(
        dir.path(),
        LAYER,
        LayerWeightFormat::Q6_K,
        &entries,
        INTER,
        HIDDEN,
        LayerScaleBinding::Inline,
        GateUpLayout::Interleaved,
    )
    .unwrap_err();
    assert!(format!("{err}").contains("Interleaved"), "{err}");
}

#[test]
fn scale_streams_and_the_declared_binding_must_agree() {
    let dir = tempdir().unwrap();

    // Declared split, entry carries none.
    let bare = vec![LayerEntry {
        gate_up: vec![1, 2],
        down: vec![3, 4],
        scales: None,
    }];
    let err = write_layer_weights_with_layout(
        dir.path(),
        LAYER,
        LayerWeightFormat::MXFP4,
        &bare,
        INTER,
        HIDDEN,
        LayerScaleBinding::SplitE8M0,
        GateUpLayout::Interleaved,
    )
    .unwrap_err();
    assert!(format!("{err}").contains("absent"), "{err}");

    // Declared inline, entry carries streams anyway — they would be written
    // nowhere and silently dropped.
    let err = write_layer_weights_with_layout(
        dir.path(),
        LAYER,
        LayerWeightFormat::MXFP4,
        &split_scale_entries(),
        INTER,
        HIDDEN,
        LayerScaleBinding::Inline,
        GateUpLayout::Interleaved,
    )
    .unwrap_err();
    assert!(format!("{err}").contains("present"), "{err}");
}

#[test]
fn an_unknown_arrangement_code_refuses_the_file() {
    // A store naming an arrangement this build cannot express must not be
    // served under a different one.
    let dir = tempdir().unwrap();
    let good = write_split_scale(dir.path(), GateUpLayout::Interleaved);

    let mut bad_binding = good.clone();
    bad_binding[24..28].copy_from_slice(&7u32.to_le_bytes());
    assert!(parse_layer_weights_header(&bad_binding).is_none());

    let mut bad_layout = good.clone();
    bad_layout[28..32].copy_from_slice(&7u32.to_le_bytes());
    assert!(parse_layer_weights_header(&bad_layout).is_none());

    // Control: the instrument passes on the unmodified file, so the two
    // refusals above are about the fields and not about the fixture.
    assert!(parse_layer_weights_header(&good).is_some());
}

#[test]
fn mxfp4_cannot_be_quantised_from_f32() {
    // Native MXFP4 exists to carry checkpoint bytes through unchanged. A
    // quantiser here would produce a bank that is native in name only and
    // would silently become what the arm is measured against.
    let err = larql_vindex::format::weights::write_layers::quantize_f32(
        &[0.0f32; 32],
        LayerWeightFormat::MXFP4,
    )
    .unwrap_err();
    assert!(format!("{err}").contains("checkpoint"), "{err}");
}
