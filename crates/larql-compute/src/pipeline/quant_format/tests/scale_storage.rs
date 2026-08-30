//! Scale-storage contract tests: the (format, aux) pair.
//!
//! Every format must answer `scale_storage()`, and every auxiliary
//! payload must match the width that format actually stores. The
//! panicking cases are the states this repository actually reached —
//! a fabricated scale buffer on an inline format, and a Q8_0 weight
//! with no scale source at all.

use crate::pipeline::quant_format::*;

/// Every format answers, and the answer matches the enum's own
/// documentation of how it stores scales.
#[test]
fn scale_storage_is_exhaustive_and_matches_the_documented_layout() {
    use ExternalScaleKind::*;
    use ScaleStorage::*;
    let table = [
        (QuantFormat::Q4_0, Inline),
        (QuantFormat::Q4_K, Inline),
        (QuantFormat::Q4_KF, Inline),
        (QuantFormat::Q6_K, Inline),
        (QuantFormat::Q8_0, External(PerBlockF32)),
        (QuantFormat::I2S, External(PerChannelF32)),
        (QuantFormat::MXFP4, External(PerGroupE8M0)),
        (QuantFormat::BF16, None),
        (QuantFormat::F16, None),
        (QuantFormat::F32, None),
    ];
    for (f, expected) in table {
        assert_eq!(f.scale_storage(), expected, "{f:?}");
    }
}

/// Q4_KF's packed layout is the standard 144-byte GGUF Q4_K block —
/// the tag selects the llama.cpp-exact kernels, not a storage
/// format. The 160-byte pre-baked layout (`Q4_KF_BLOCK_BYTES`) has
/// no kernel consumer; answering it here mis-sized every derived
/// buffer by 16 bytes per super-block (capability audit F15).
#[test]
fn q4_kf_packed_layout_is_gguf_q4_k() {
    use larql_models::quant::ggml;
    assert_eq!(
        QuantFormat::Q4_KF.packed_block_layout(),
        Some((ggml::Q4_K_BLOCK_ELEMS, ggml::Q4_K_BLOCK_BYTES)),
    );
    assert_eq!(
        QuantFormat::Q4_KF.packed_block_layout(),
        QuantFormat::Q4_K.packed_block_layout(),
    );
}

/// Block-packed no longer implies inline scales, and MXFP4 is the
/// format that broke it.
///
/// The previous version of this test asserted the two coincided, with
/// a note that "a future packed format with external scales must break
/// this test rather than silently misbehave". MXFP4 is that format: a
/// regular 32-elem/16-byte packed stream whose e8m0 scales live
/// outside it. The assertion is inverted rather than deleted, so the
/// property stays pinned in the direction that is now true.
#[test]
fn block_packed_does_not_imply_inline_scales() {
    for f in [
        QuantFormat::Q4_0,
        QuantFormat::Q4_K,
        QuantFormat::Q4_KF,
        QuantFormat::Q6_K,
    ] {
        assert!(f.packed_block_layout().is_some());
        assert_eq!(f.scale_storage(), ScaleStorage::Inline);
    }
    // The counterexample, named.
    assert!(QuantFormat::MXFP4.packed_block_layout().is_some());
    assert_eq!(
        QuantFormat::MXFP4.scale_storage(),
        ScaleStorage::External(ExternalScaleKind::PerGroupE8M0),
    );
}

// The rest of MXFP4's format contract — packed geometry, the e8m0
// stream's width, and the aux-width rules this file's matrix below does
// not cover — lives in `tests/mxfp4.rs`.

// ── the Phase A specification, as a matrix ──────────────────────
//
//                     no aux      external scales
//   Q4_0 / Q4_K         ok             panic
//   Q4_KF / Q6_K        ok             panic
//   Q8_0 / I2S        panic              ok
//   F16 / F32 / BF16    ok             panic

#[test]
fn inline_formats_accept_no_aux() {
    for f in [
        QuantFormat::Q4_0,
        QuantFormat::Q4_K,
        QuantFormat::Q4_KF,
        QuantFormat::Q6_K,
    ] {
        let w = QuantWeight::new(f, &[0u8; 4], QuantAux::None);
        assert!(w.external_scales().is_none(), "{f:?}");
    }
}

#[test]
fn unquantised_formats_accept_no_aux() {
    for f in [QuantFormat::BF16, QuantFormat::F16, QuantFormat::F32] {
        let w = QuantWeight::new(f, &[0u8; 4], QuantAux::None);
        assert!(w.external_scales().is_none(), "{f:?}");
    }
}

#[test]
fn external_formats_accept_and_expose_their_scales() {
    let s = [1.0f32, 2.0];
    for f in [QuantFormat::Q8_0, QuantFormat::I2S] {
        let w = QuantWeight::new(f, &[0u8; 4], QuantAux::ExternalScales(&s));
        assert_eq!(w.external_scales(), Some(&s[..]), "{f:?}");
    }
}

/// The state that produced the dead O-projection fixture and the 24
/// fabricated buffers: an inline format carrying external scales.
#[test]
#[should_panic(expected = "packs its scales inline")]
fn q4k_cannot_carry_external_scales() {
    let s = [1.0f32];
    let _ = QuantWeight::new(QuantFormat::Q4_K, &[0u8; 4], QuantAux::ExternalScales(&s));
}

/// Q4_0's 18-byte block *is* an f16 scale plus 16 bytes of nibbles.
/// A test fixture in this repository supplied external scales for it
/// and passed; that is now unrepresentable.
#[test]
#[should_panic(expected = "packs its scales inline")]
fn q4_0_cannot_carry_external_scales() {
    let s = [1.0f32];
    let _ = QuantWeight::new(QuantFormat::Q4_0, &[0u8; 4], QuantAux::ExternalScales(&s));
}

#[test]
#[should_panic(expected = "unquantised")]
fn unquantised_formats_cannot_carry_scales() {
    let s = [1.0f32];
    let _ = QuantWeight::new(QuantFormat::F32, &[0u8; 4], QuantAux::ExternalScales(&s));
}

/// The other half: a format that genuinely needs scales cannot exist
/// without them. Previously `scales: None` on a Q8_0 weight was a
/// silently-constructible state.
#[test]
#[should_panic(expected = "stores scales externally")]
fn q8_0_cannot_exist_without_scales() {
    let _ = QuantWeight::new(QuantFormat::Q8_0, &[0u8; 4], QuantAux::None);
}

#[test]
#[should_panic(expected = "stores scales externally")]
fn i2s_cannot_exist_without_scales() {
    let _ = QuantWeight::new(QuantFormat::I2S, &[0u8; 4], QuantAux::None);
}

/// The default is a valid state, not merely a zeroed one.
#[test]
fn default_weight_is_internally_consistent() {
    let w = QuantWeight::default();
    assert_eq!(w.format().scale_storage(), ScaleStorage::Inline);
    assert!(w.external_scales().is_none());
}

// ── with_format: retagging within an aux class is fine; crossing
//    classes is the desynchronisation hole and must panic ─────────

#[test]
fn with_format_allows_retagging_within_the_inline_class() {
    let w = QuantWeight::new(QuantFormat::Q4_K, &[0u8; 4], QuantAux::None);
    let w = w.with_format(QuantFormat::Q4_KF);
    assert_eq!(w.format(), QuantFormat::Q4_KF);
    assert!(w.external_scales().is_none());
}

#[test]
fn with_format_allows_retagging_within_the_external_class() {
    let s = [1.0f32, 2.0];
    let w = QuantWeight::new(QuantFormat::Q8_0, &[0u8; 4], QuantAux::ExternalScales(&s));
    let w = w.with_format(QuantFormat::I2S);
    assert_eq!(w.format(), QuantFormat::I2S);
    assert_eq!(w.external_scales(), Some(&s[..]));
}

/// The exact shape of the hole this closes: a weight built inline
/// (no scales) retagged to a format whose kernels read an external
/// scale buffer that does not exist.
#[test]
#[should_panic(expected = "stores scales externally")]
fn with_format_refuses_inline_to_external_retag() {
    let w = QuantWeight::new(QuantFormat::Q4_K, &[0u8; 4], QuantAux::None);
    let _ = w.with_format(QuantFormat::Q8_0);
}

#[test]
#[should_panic(expected = "packs its scales inline")]
fn with_format_refuses_external_to_inline_retag() {
    let s = [1.0f32];
    let w = QuantWeight::new(QuantFormat::Q8_0, &[0u8; 4], QuantAux::ExternalScales(&s));
    let _ = w.with_format(QuantFormat::Q4_K);
}
