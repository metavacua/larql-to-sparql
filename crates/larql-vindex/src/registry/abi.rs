//! VINDEX3 runtime ABI — a genuinely new axis, not a restatement of
//! `index.json`'s schema version.
//!
//! [`ContainerGeneration`](crate::format::generation::ContainerGeneration)'s
//! schema dispatch (spec §12.1) answers "can this binary parse this
//! container's on-disk bytes." This answers a different question: "does
//! this binary implement the *runtime* capabilities a registry variant was
//! built assuming." Nothing in the codebase conflated the two before this
//! module — `format::vindex3::profile` itself notes the VINDEX3 ABI is
//! "explicitly not frozen yet" (design doc §3). This type is the seam that
//! statement will grow into, kept deliberately minimal: one supported
//! value, exact match, no compatibility range invented ahead of a second
//! value actually existing to reason about.

/// The runtime capability a VINDEX3 registry entry was built against.
///
/// A newtype over `u32` for the same reason
/// [`IndexSchemaVersion`](crate::format::generation::IndexSchemaVersion) is
/// one: no API should accept a bare integer and call it an ABI.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct Vindex3Abi(pub u32);

impl Vindex3Abi {
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Whether this binary implements the given ABI.
    ///
    /// Exact match only — see module docs for why a range isn't offered
    /// yet.
    pub fn is_supported(self) -> bool {
        self == CURRENT_VINDEX3_ABI
    }
}

impl std::fmt::Display for Vindex3Abi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The only ABI this binary implements today.
///
/// Bump this — and add the second arm a real compatibility range would
/// need — the day a registry variant is built against runtime capabilities
/// an older binary genuinely cannot execute. Until then, "one value, exact
/// match" is the honest claim: nothing has needed a range yet.
pub const CURRENT_VINDEX3_ABI: Vindex3Abi = Vindex3Abi(1);
