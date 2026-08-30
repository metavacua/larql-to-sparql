//! MXFP4 format-contract tests.
//!
//! MXFP4 is the first format here that is block-packed *and* carries
//! external scales, and the first whose external scales are bytes rather
//! than f32. Both properties are load-bearing rather than incidental:
//!
//! - folding the e8m0 stream into `packed_block_layout` would over-size
//!   every derived buffer by one byte per 32 weights (the Q4_KF 160-vs-144
//!   defect class, capability audit F15);
//! - widening the e8m0 stream to f32 would take the format from 4.25 bpw
//!   to 5.0, forfeiting a third of the bandwidth win that is the entire
//!   reason to serve MXFP4 natively instead of transcoding it to Q6_K.
//!
//! So the tests pin the *widths*, not just the routing.

use larql_models::quant::mxfp4;

use crate::pipeline::quant_format::*;

/// Codeword width of an OCP MXFP4 weight. Named rather than spelled `4`
/// inline, because every bpw assertion below is derived from it.
const FP4_PAYLOAD_BITS: usize = 4;

/// One e8m0 exponent byte per group.
const E8M0_SCALE_BITS: usize = 8;

/// Build an MXFP4 weight of `rows` rows at `stored_cols` columns each,
/// with a correctly-sized e8m0 stream. Contents are irrelevant to every
/// property under test here — only the widths are.
fn mxfp4_rows(rows: usize, stored_cols: usize) -> (Vec<u8>, Vec<u8>) {
    let groups = rows * (stored_cols / mxfp4::MXFP4_GROUP_ELEMS);
    (
        vec![0u8; groups * mxfp4::MXFP4_GROUP_BYTES],
        vec![0u8; groups],
    )
}

/// The all-in cost is 4.25 bpw: 4 bits of payload plus one e8m0 byte
/// amortised over 32 weights. This is the number the whole native-MXFP4
/// case rests on, so it is asserted from the constants rather than
/// assumed.
#[test]
fn all_in_cost_is_four_and_a_quarter_bits_per_weight() {
    let payload_bits = mxfp4::MXFP4_GROUP_BYTES * 8;
    assert_eq!(payload_bits, mxfp4::MXFP4_GROUP_ELEMS * FP4_PAYLOAD_BITS);

    let all_in = (payload_bits + E8M0_SCALE_BITS) as f64 / mxfp4::MXFP4_GROUP_ELEMS as f64;
    assert!(
        (all_in - 4.25).abs() < f64::EPSILON,
        "all-in bpw was {all_in}, expected 4.25"
    );
}

/// `packed_matrix_bytes` answers payload bytes only. A caller that needs
/// the scale stream must size it from `scale_storage`, not by assuming
/// this number covers everything.
#[test]
fn packed_matrix_bytes_excludes_the_scale_stream() {
    let (rows, cols) = (4usize, 2 * mxfp4::MXFP4_GROUP_ELEMS);
    let groups = rows * (cols / mxfp4::MXFP4_GROUP_ELEMS);
    assert_eq!(
        QuantFormat::MXFP4.packed_matrix_bytes(rows, cols),
        Some(groups * mxfp4::MXFP4_GROUP_BYTES),
    );
}

/// The registry tag round-trips, so a natively stored expert bank keeps
/// its format across the writer → loader boundary. A tag that failed to
/// round-trip is what decoded a Q6_K expert store as Q4_K garbage.
#[test]
fn registry_tag_round_trips() {
    assert_eq!(QuantFormat::MXFP4.registry_tag(), "MXFP4");
    assert_eq!(
        QuantFormat::from_registry_tag(QuantFormat::MXFP4.registry_tag()),
        Some(QuantFormat::MXFP4),
    );
}

/// The e8m0 stream is reachable at its stored width, and the f32
/// accessor refuses to answer for it rather than silently converting.
#[test]
fn e8m0_stream_is_exposed_at_its_stored_width() {
    let (packed, scales) = mxfp4_rows(2, mxfp4::MXFP4_GROUP_ELEMS);
    let w = QuantWeight::new(QuantFormat::MXFP4, &packed, QuantAux::ExternalE8M0(&scales));
    assert_eq!(w.external_e8m0(), Some(&scales[..]));
    assert!(
        w.external_scales().is_none(),
        "an MXFP4 weight must not answer the f32 scale accessor"
    );
}

/// The inverse: an f32-scaled format does not answer the e8m0 accessor.
#[test]
fn f32_scaled_formats_do_not_answer_the_e8m0_accessor() {
    let s = [1.0f32, 2.0];
    let w = QuantWeight::new(QuantFormat::Q8_0, &[0u8; 4], QuantAux::ExternalScales(&s));
    assert!(w.external_e8m0().is_none());
}

/// Padded rows answer their stored width, exactly as the k-quant formats
/// do. GPT-OSS's hidden 2880 pads to 3072; the derivation must survive
/// that, or the row stride desynchronises from row 1 onward.
#[test]
fn padded_rows_answer_the_stored_width() {
    let logical = 2880usize;
    let stored = logical.div_ceil(mxfp4::MXFP4_GROUP_ELEMS) * mxfp4::MXFP4_GROUP_ELEMS;
    let rows = 3usize;
    let (packed, scales) = mxfp4_rows(rows, stored);

    let w = QuantWeight::new(QuantFormat::MXFP4, &packed, QuantAux::ExternalE8M0(&scales));
    assert_eq!(w.stored_cols(rows, logical), stored);
}

// ── the aux-width contract ──────────────────────────────────────────
//
// Before MXFP4 every external format was f32-scaled, so `External(_)`
// paired with any array was sound. These pin that it no longer is, in
// both directions.

#[test]
#[should_panic(expected = "stores scales externally")]
fn mxfp4_cannot_exist_without_its_scale_stream() {
    let _ = QuantWeight::new(QuantFormat::MXFP4, &[0u8; 16], QuantAux::None);
}

#[test]
#[should_panic(expected = "different width")]
fn mxfp4_refuses_an_f32_scale_array() {
    let s = [1.0f32];
    let _ = QuantWeight::new(QuantFormat::MXFP4, &[0u8; 16], QuantAux::ExternalScales(&s));
}

#[test]
#[should_panic(expected = "different width")]
fn f32_scaled_formats_refuse_an_e8m0_stream() {
    let e = [0u8; 2];
    let _ = QuantWeight::new(QuantFormat::Q8_0, &[0u8; 4], QuantAux::ExternalE8M0(&e));
}

#[test]
#[should_panic(expected = "packs its scales inline")]
fn inline_formats_refuse_an_e8m0_stream() {
    let e = [0u8; 2];
    let _ = QuantWeight::new(QuantFormat::Q6_K, &[0u8; 4], QuantAux::ExternalE8M0(&e));
}

#[test]
#[should_panic(expected = "unquantised")]
fn unquantised_formats_refuse_an_e8m0_stream() {
    let e = [0u8; 2];
    let _ = QuantWeight::new(QuantFormat::F32, &[0u8; 4], QuantAux::ExternalE8M0(&e));
}

/// Retagging may not cross the aux *width* boundary either — MXFP4 and
/// Q8_0 are both "external", but their streams are not interchangeable.
#[test]
#[should_panic(expected = "different width")]
fn with_format_refuses_e8m0_to_f32_retag() {
    let e = [0u8; 2];
    let w = QuantWeight::new(QuantFormat::MXFP4, &[0u8; 16], QuantAux::ExternalE8M0(&e));
    let _ = w.with_format(QuantFormat::Q8_0);
}

/// MXFP4 is not a GGUF 256-superblock k-quant and must not join the
/// family that gates the Q4_K / Q6_K matvec dispatchers.
#[test]
fn is_not_a_kquant_and_is_self_identifying() {
    assert!(!QuantFormat::MXFP4.is_kquant_family());
    assert!(!QuantFormat::MXFP4.is_legacy_q8());
    assert!(!QuantFormat::MXFP4.is_ternary());
    assert!(QuantFormat::MXFP4.is_mxfp4());
    assert!(!QuantFormat::Q6_K.is_mxfp4());
}
