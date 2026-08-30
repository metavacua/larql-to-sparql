//! Storage-encoded arrangement facts for a per-layer expert store
//! (`layers/layer_{L:02}.weights`, magic `LYRW`).
//!
//! Three facts travel together in that file's header and were, until MXFP4,
//! all implied by one of them:
//!
//! | fact | question it answers |
//! |---|---|
//! | [`LayerWeightFormat`] | how is a *block* decoded? |
//! | [`LayerScaleBinding`]  | where do the *scales* live? |
//! | [`GateUpLayout`]  | which *rows* are gate and which are up? |
//!
//! Every store larql had written before MXFP4 answered "inside the block"
//! and "all gate rows, then all up rows", so a reader could derive both from
//! the quant code and be right. A checkpoint-native MXFP4 bank answers
//! neither the same way, and `docs/k3-funnel.md` §4.7 records what deriving
//! the layout costs: read as [`GateUpLayout::ContiguousHalves`], an
//! interleaved bank yields two 50/50 mixtures of the real gate and up rows,
//! with matching summary statistics and coherent-looking output — a served
//! model was wrong that way once already.
//!
//! So they are encoded as three independent fields. `MXFP4` does not mean
//! `Interleaved`, and the offset-table stride keys off [`LayerScaleBinding`]
//! rather than off the format: a split-scale k-quant bank, or an MXFP4 bank
//! that a writer chose to de-interleave, are both expressible.

use larql_models::config::experts::GateUpLayout;

/// Where a per-layer expert store keeps its quantisation scales.
///
/// This is a statement about *this file*, not about the format in the
/// abstract: MXFP4 admits both bindings, and which one a given store uses is
/// the writer's choice, recorded here so the reader never has to guess.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerScaleBinding {
    /// Scales ride inside the payload blocks. No partner stream exists —
    /// not "an empty one". Every k-quant store larql has ever written.
    Inline = 0,
    /// Per-entry e8m0 exponent streams, one per payload stream, carried in
    /// their own byte ranges. What a verbatim MXFP4 passthrough preserves.
    SplitE8M0 = 1,
}

/// Offset-table fields per entry under [`LayerScaleBinding::Inline`]:
/// `(gate_up_off, gate_up_bytes, down_off, down_bytes)`.
pub const INLINE_OFFSET_FIELDS_PER_ENTRY: usize = 4;

/// Offset-table fields per entry under [`LayerScaleBinding::SplitE8M0`]:
/// the inline four, then the two exponent ranges. The exponent ranges are
/// stored rather than derived — `payload_offset / 16` happens to be true of
/// two parallel banks and is not a property of the format, and deriving it
/// is exactly the physical-placement invariant that
/// `mxfp4g_split_lut16` had to stop assuming.
pub const SPLIT_SCALE_OFFSET_FIELDS_PER_ENTRY: usize = 8;

const U64_FIELD_BYTES: usize = std::mem::size_of::<u64>();

impl LayerScaleBinding {
    pub fn as_u32(self) -> u32 {
        self as u32
    }

    /// `None` for an unrecognised code — the caller refuses the file rather
    /// than picking a binding, because guessing wrong here reads exponent
    /// bytes as payload.
    pub fn from_u32(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::Inline),
            1 => Some(Self::SplitE8M0),
            _ => None,
        }
    }

    /// Whether entries carry their own exponent byte ranges.
    pub fn is_split(self) -> bool {
        matches!(self, Self::SplitE8M0)
    }

    /// Offset-table fields per entry under this binding. **The
    /// offset-table stride is a function of the binding, not of the quant
    /// format** — that is what keeps the two facts separable.
    pub fn offset_fields_per_entry(self) -> usize {
        match self {
            Self::Inline => INLINE_OFFSET_FIELDS_PER_ENTRY,
            Self::SplitE8M0 => SPLIT_SCALE_OFFSET_FIELDS_PER_ENTRY,
        }
    }

    /// Offset-table bytes per entry under this binding.
    pub fn offset_entry_bytes(self) -> usize {
        self.offset_fields_per_entry() * U64_FIELD_BYTES
    }
}

/// Stored code for [`GateUpLayout::ContiguousHalves`].
pub const FUSED_ROW_LAYOUT_CONTIGUOUS_HALVES: u32 = 0;
/// Stored code for [`GateUpLayout::Interleaved`].
pub const FUSED_ROW_LAYOUT_INTERLEAVED: u32 = 1;

/// Encode a fused-row layout for the header.
///
/// The codec lives here rather than on `GateUpLayout` because the
/// numbering is a property of *this container*, and compute should not grow
/// a storage concern to satisfy one writer.
pub fn fused_row_layout_code(layout: GateUpLayout) -> u32 {
    match layout {
        GateUpLayout::ContiguousHalves => FUSED_ROW_LAYOUT_CONTIGUOUS_HALVES,
        GateUpLayout::Interleaved => FUSED_ROW_LAYOUT_INTERLEAVED,
    }
}

/// Decode a fused-row layout from the header. `None` for an unrecognised
/// code: a store that declares a layout this build cannot name is refused,
/// not defaulted. Defaulting is how an interleaved bank gets read as
/// contiguous halves.
pub fn fused_row_layout_from_code(code: u32) -> Option<GateUpLayout> {
    match code {
        FUSED_ROW_LAYOUT_CONTIGUOUS_HALVES => Some(GateUpLayout::ContiguousHalves),
        FUSED_ROW_LAYOUT_INTERLEAVED => Some(GateUpLayout::Interleaved),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_binding_codes_round_trip() {
        for b in [LayerScaleBinding::Inline, LayerScaleBinding::SplitE8M0] {
            assert_eq!(LayerScaleBinding::from_u32(b.as_u32()), Some(b));
        }
    }

    #[test]
    fn scale_binding_refuses_unknown_code() {
        assert_eq!(LayerScaleBinding::from_u32(2), None);
        assert_eq!(LayerScaleBinding::from_u32(u32::MAX), None);
    }

    #[test]
    fn inline_binding_keeps_the_historic_four_field_stride() {
        // Every pre-MXFP4 store on disk was written with this stride. If it
        // ever changes, those files become unreadable silently — the parse
        // stays in bounds and returns offsets from the wrong place.
        let b = LayerScaleBinding::Inline;
        assert_eq!(b.offset_fields_per_entry(), 4);
        assert_eq!(b.offset_entry_bytes(), 32);
        assert!(!b.is_split());
    }

    #[test]
    fn split_binding_widens_the_stride_to_carry_two_exponent_ranges() {
        let b = LayerScaleBinding::SplitE8M0;
        assert_eq!(b.offset_fields_per_entry(), 8);
        assert_eq!(b.offset_entry_bytes(), 64);
        assert!(b.is_split());
    }

    #[test]
    fn fused_row_layout_codes_round_trip() {
        for l in [GateUpLayout::ContiguousHalves, GateUpLayout::Interleaved] {
            assert_eq!(
                fused_row_layout_from_code(fused_row_layout_code(l)),
                Some(l)
            );
        }
    }

    #[test]
    fn fused_row_layout_refuses_unknown_code() {
        assert_eq!(fused_row_layout_from_code(2), None);
        assert_eq!(fused_row_layout_from_code(u32::MAX), None);
    }

    #[test]
    fn contiguous_halves_is_code_zero_so_a_zeroed_block_is_the_historic_layout() {
        // Not cosmetic: the layout block is absent from every store written
        // before it existed, and any future reader that zero-fills a missing
        // block must land on the arrangement those files actually use.
        assert_eq!(fused_row_layout_code(GateUpLayout::ContiguousHalves), 0);
        assert_eq!(LayerScaleBinding::Inline.as_u32(), 0);
    }
}
