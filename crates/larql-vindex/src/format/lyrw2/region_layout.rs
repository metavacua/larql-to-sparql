//! How a region's rows are arranged *as stored*.
//!
//! Distinct from the model-level `GateUpLayout` in `larql-models`, and the
//! distinction is the point. That type describes the **checkpoint's**
//! operand; this one describes **these bytes**. They agree under a verbatim
//! passthrough and diverge the moment an extractor canonicalises:
//!
//! ```text
//! GPT-OSS checkpoint          GPT-OSS checkpoint
//!   Interleaved                 Interleaved
//!       │ verbatim                  │ transform
//!       ▼                           ▼
//!   VINDEX3                     VINDEX3
//!   Interleaved                 ContiguousHalves
//! ```
//!
//! An execution path must consume *this* declaration, never reach back to
//! `arch.gate_up_layout()` and assume the store still resembles the
//! checkpoint. `GateUpLayout`'s own documentation records why: reading one
//! arrangement as the other does not fail, it silently mixes the two
//! branches and produces plausible garbage.
//!
//! This is an independent axis from the others a schema carries:
//!
//! ```text
//! role     what mathematical operand this is
//! format   how values are encoded
//! packing  how bytes are grouped
//! pair_id  which regions belong together
//! layout   how rows are arranged   ← here
//! ```

/// Row arrangement of a stored region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RegionLayout {
    /// Not declared.
    ///
    /// **Not a synonym for [`Self::ContiguousHalves`].** Every container
    /// written before this field existed says `Unspecified`, and reading
    /// that as "contiguous" would be exactly the inference this field was
    /// added to remove — today's compatibility concession becoming
    /// tomorrow's hardcoding. A consumer that needs the arrangement must
    /// treat `Unspecified` as "this container does not say", and resolve it
    /// by a specifically justified legacy rule or refuse.
    #[default]
    Unspecified,
    /// `[all gate rows | all up rows]` — the first half is gate.
    ContiguousHalves,
    /// `gate = rows[0], rows[2], …`, `up = rows[1], rows[3], …`.
    Interleaved,
    /// An arrangement this binary does not recognise. Round-trips unchanged
    /// so a newer container survives a read/write cycle intact.
    Unknown(u16),
}

impl RegionLayout {
    pub fn from_u16(tag: u16) -> Self {
        match tag {
            0 => Self::Unspecified,
            1 => Self::ContiguousHalves,
            2 => Self::Interleaved,
            other => Self::Unknown(other),
        }
    }

    pub fn as_u16(self) -> u16 {
        match self {
            Self::Unspecified => 0,
            Self::ContiguousHalves => 1,
            Self::Interleaved => 2,
            Self::Unknown(tag) => tag,
        }
    }

    pub fn name(self) -> String {
        match self {
            Self::Unspecified => "unspecified".into(),
            Self::ContiguousHalves => "contiguous_halves".into(),
            Self::Interleaved => "interleaved".into(),
            Self::Unknown(tag) => format!("layout_{tag}"),
        }
    }

    /// Whether this names an actual arrangement a consumer can act on.
    ///
    /// `Unknown` counts: the container *did* declare something, this
    /// binary simply cannot interpret it — which is a refusal, not the
    /// absence of a declaration.
    pub fn is_declared(self) -> bool {
        !matches!(self, Self::Unspecified)
    }

    /// The row walk an execution path may perform on these bytes, or `None`
    /// when this binary must not act on the declaration.
    ///
    /// The two `None` cases are different failures and both are real:
    /// [`Self::Unspecified`] is a container that never said, and
    /// [`Self::Unknown`] is one that said something this binary post-dates.
    /// Neither may resolve to a concrete arrangement — mapping either to
    /// [`larql_compute::MoeFusedRowLayout::ContiguousHalves`] because it is
    /// the common case would reintroduce exactly the inference this type
    /// was added to remove, and it would do it silently, since a wrong
    /// arrangement computes a plausible answer rather than failing.
    ///
    /// The caller decides what to do with `None`. There is no default here
    /// to fall back to, deliberately: a default would be indistinguishable
    /// from a real declaration by the time it reached a kernel.
    pub fn fused_row_layout(self) -> Option<larql_compute::MoeFusedRowLayout> {
        match self {
            Self::ContiguousHalves => Some(larql_compute::MoeFusedRowLayout::ContiguousHalves),
            Self::Interleaved => Some(larql_compute::MoeFusedRowLayout::Interleaved),
            Self::Unspecified | Self::Unknown(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_round_trip() {
        for layout in [
            RegionLayout::Unspecified,
            RegionLayout::ContiguousHalves,
            RegionLayout::Interleaved,
            RegionLayout::Unknown(900),
        ] {
            assert_eq!(RegionLayout::from_u16(layout.as_u16()), layout);
        }
    }

    /// Zero is what every pre-existing container's reserved field holds, so
    /// it must decode to "not declared" and nothing else.
    #[test]
    fn zero_is_unspecified_and_is_the_default() {
        assert_eq!(RegionLayout::from_u16(0), RegionLayout::Unspecified);
        assert_eq!(RegionLayout::default(), RegionLayout::Unspecified);
    }

    /// The trap this type exists to prevent: `Unspecified` must never be
    /// treated as a concrete arrangement.
    #[test]
    fn unspecified_is_not_a_declaration_but_unknown_is() {
        assert!(!RegionLayout::Unspecified.is_declared());
        assert!(RegionLayout::ContiguousHalves.is_declared());
        assert!(RegionLayout::Interleaved.is_declared());
        // A newer container declared *something*; refusing to interpret it
        // is different from it having said nothing.
        assert!(RegionLayout::Unknown(7).is_declared());
    }

    #[test]
    fn unknown_tags_name_themselves() {
        assert_eq!(RegionLayout::Unknown(42).name(), "layout_42");
    }

    /// Both undeclared cases must refuse, and for different reasons — see
    /// `fused_row_layout`. Pinning them together because they share an
    /// answer today would hide it if one of them ever stopped refusing.
    #[test]
    fn neither_undeclared_case_resolves_to_an_arrangement() {
        assert!(RegionLayout::Unspecified.fused_row_layout().is_none());
        assert!(RegionLayout::Unknown(9).fused_row_layout().is_none());
    }

    #[test]
    fn declared_layouts_map_to_their_own_row_walk() {
        use larql_compute::MoeFusedRowLayout;
        assert_eq!(
            RegionLayout::ContiguousHalves.fused_row_layout(),
            Some(MoeFusedRowLayout::ContiguousHalves)
        );
        assert_eq!(
            RegionLayout::Interleaved.fused_row_layout(),
            Some(MoeFusedRowLayout::Interleaved)
        );
        // The mapping must not collapse the two — that collapse IS the bug.
        assert_ne!(
            RegionLayout::ContiguousHalves.fused_row_layout(),
            RegionLayout::Interleaved.fused_row_layout()
        );
    }
}
