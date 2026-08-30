//! Which container representation an extraction is asked to produce.
//!
//! # Why this is a type and not an `if`
//!
//! The obvious way to make GPT-OSS extract natively is one branch on the
//! model family. That branch would be wrong twice over: it makes GPT-OSS
//! the *reason* the native path exists rather than its first witness, and
//! it puts a model name where a capability belongs — the same shape the
//! expert-route work removed from the Metal dispatch, where
//! `_ => q4k_grouped_experts` silently served a format it could not read.
//!
//! The decision is instead:
//!
//! ```text
//! source expert format keeps split scales
//! + a region format exists for its encoding
//! + the stored fused layout is declared
//! + native extraction was admitted
//!         ↓
//!    NativeV3
//! ```
//!
//! Any one of those missing is an explicit refusal naming what was absent,
//! never a quiet fall back to the transcoding route. A caller that asked
//! for native bytes and silently received Q6_K ones would go on to compare
//! them against the very transcode it meant to avoid.
//!
//! # Ownership: who decides what
//!
//! ```text
//! orchestrator        chooses the target
//! writers             execute one target each, and are leaves
//! checkpoint adapter  exposes a source's bytes and facts
//! container           records what it was told
//! ```
//!
//! **Writers do not choose container generations.** The rule exists
//! because the alternative is concrete and close: teaching
//! `write_model_weights_kquant_with_opts` to notice a native-capable
//! source and branch would put VINDEX2/VINDEX3 policy inside the very
//! writer that only knows how to transcode, and every later
//! representation question would accumulate in the same place. A target
//! is chosen once, above; each writer is then handed a decision it does
//! not re-litigate.
//!
//! The three stages are separate types on purpose:
//!
//! ```text
//! caller intent      → ExtractionRequest
//! facts about source → SourceCapabilities
//! request + facts    → ExtractionTarget
//! ```
//!
//! # The legacy route is not deprecated by this
//!
//! `LegacyKQuant` stays a first-class target. It is the control the native
//! path is qualified against: the same checkpoint reaching two containers,
//! one transcoded and one verbatim, is what separates "extraction is
//! equivalent" from "execution is equivalent" from "this is faster".

use larql_models::ExpertFormat;

use crate::format::lyrw2::region_format::RegionFormat;
use crate::format::lyrw2::region_layout::RegionLayout;
use crate::VindexError;

/// What a caller **asked for**.
///
/// Deliberately a different type from [`ExtractionTarget`], which is what
/// admission *decided*. Collapsing the two would make "native was
/// requested and refused" and "native was never requested" the same value,
/// and those must never be confused: the first is an error the caller has
/// to see, the second is ordinary operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractionRequest {
    /// The shipped transcoding route, explicitly.
    Legacy,
    /// Native, explicitly. Refuses if the source cannot supply it —
    /// never downgrades.
    Native,
    /// No preference; admission applies policy.
    Auto,
}

/// What an extraction produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractionTarget {
    /// The shipped route: expert weights are transcoded into an
    /// inline-scale k-quant and written as VINDEX2. Lossless for MXFP4 —
    /// Q6_K's 64 levels hold all 15 fp4 codepoints — but it discards the
    /// native representation on the way.
    LegacyKQuant,
    /// Native: the source's own bytes into a VINDEX3 container, with
    /// partner scale regions where the format needs them.
    NativeV3,
}

impl ExtractionTarget {
    pub fn name(self) -> &'static str {
        match self {
            Self::LegacyKQuant => "legacy-kquant",
            Self::NativeV3 => "native-v3",
        }
    }
}

/// What a source can actually offer, as facts rather than as a family name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceCapabilities {
    /// The source's expert encoding, or `None` if no `RegionFormat`
    /// describes it. A container may not transcode, so an unrepresentable
    /// encoding is a hard stop for the native route rather than a prompt
    /// to convert.
    pub region_format: Option<RegionFormat>,
    /// Whether the expert format keeps scales in a separate stream.
    pub split_scale_streams: bool,
    /// Whether those streams are actually reachable from this source. A
    /// format that *should* have them and a source that *can hand them
    /// over* are different facts, and the second is where the raw-access
    /// gap lives.
    pub scale_streams_available: bool,
    /// The stored fused gate/up arrangement. `Unspecified` means the
    /// source did not say, which a schema-4 container may not record.
    pub gate_up_layout: RegionLayout,
}

impl SourceCapabilities {
    /// Derive what is knowable from the expert format alone.
    ///
    /// Deliberately leaves `scale_streams_available` and `gate_up_layout`
    /// to the caller: the first depends on whether *this* source exposes
    /// raw bytes, and the second is a property of the stored artifact, not
    /// of the format class. Defaulting either here would let the writer
    /// answer a question only the extraction path can.
    pub fn from_expert_format(format: ExpertFormat, region_format: Option<RegionFormat>) -> Self {
        Self {
            region_format,
            split_scale_streams: format.has_split_scale_streams(),
            scale_streams_available: false,
            gate_up_layout: RegionLayout::Unspecified,
        }
    }

    pub fn with_scale_streams(mut self, available: bool) -> Self {
        self.scale_streams_available = available;
        self
    }

    pub fn with_gate_up_layout(mut self, layout: RegionLayout) -> Self {
        self.gate_up_layout = layout;
        self
    }
}

/// Resolve a request against what the source can offer.
///
/// Returns the target to execute, or an error naming precisely what is
/// missing. There is no third outcome — in particular there is no
/// "asked for native, quietly got legacy".
///
/// **`Auto` is policy, not capability.** While the native route is being
/// qualified it resolves to [`ExtractionTarget::LegacyKQuant`] even from a
/// fully capable source, so existing extraction behaviour is unchanged for
/// every caller that did not opt in. That is deliberately *not* the same
/// decision as a refused `Native` request, and the two must stay
/// distinguishable: one is a caller being told no, the other is a caller
/// who never asked.
pub fn admit(
    request: ExtractionRequest,
    caps: &SourceCapabilities,
) -> Result<ExtractionTarget, VindexError> {
    let requested = match request {
        ExtractionRequest::Legacy => ExtractionTarget::LegacyKQuant,
        ExtractionRequest::Native => ExtractionTarget::NativeV3,
        // Policy: native stays opt-in until it is qualified. Revisit when
        // #4's parity gate passes, not before.
        ExtractionRequest::Auto => return Ok(ExtractionTarget::LegacyKQuant),
    };
    let refuse = |why: String| {
        Err(VindexError::Parse(format!(
            "native extraction was requested but {why}; the legacy k-quant \
             route remains available and must be selected explicitly rather \
             than fallen back to"
        )))
    };

    match requested {
        // Always available: it is the shipped path and transcodes whatever
        // it is given.
        ExtractionTarget::LegacyKQuant => Ok(ExtractionTarget::LegacyKQuant),
        ExtractionTarget::NativeV3 => {
            if caps.region_format.is_none() {
                return refuse(
                    "no VINDEX3 region format describes this expert encoding, and a \
                     container may not transcode on the way in"
                        .into(),
                );
            }
            if caps.split_scale_streams && !caps.scale_streams_available {
                return refuse(
                    "this expert format keeps its scales in a separate stream and the \
                     source cannot hand those bytes over — writing the payload alone \
                     would produce a bank whose every group decodes at 2^0"
                        .into(),
                );
            }
            if !caps.gate_up_layout.is_declared() {
                return refuse(
                    "the stored fused gate/up arrangement is undeclared; a container \
                     written at this schema must say whether its rows are contiguous \
                     halves or interleaved, because reading one as the other mixes \
                     the two branches without failing"
                        .into(),
                );
            }
            Ok(ExtractionTarget::NativeV3)
        }
    }
}

#[cfg(test)]
#[path = "target_tests.rs"]
mod tests;
