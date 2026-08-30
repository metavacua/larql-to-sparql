//! What happened when one bound operation was compared against the incumbent.
//!
//! A sweep over thirty layers reports one line each, and the temptation is a
//! pass/fail column. That column is a lie in both directions: it calls a
//! non-resident expert a failure when the route was correct and the operand
//! simply lives elsewhere, and it calls a within-tolerance agreement a pass
//! when nothing was asserted. Both readings send the next person to the wrong
//! place.
//!
//! So the vocabulary lives here, in the library, rather than in whichever
//! harness happens to need it — the locked-input sweep, the free-propagation
//! residual comparison and the decode-parity run must classify the same
//! situation the same way, and three private enums would drift within a week.
//!
//! # The distinction that matters most
//!
//! ```text
//! Exact           every asserted checkpoint was bit-identical
//! ObservedExact   the whole comparison came out bit-identical, uncontracted
//! ```
//!
//! The second is a *measurement*, not a promise. A reduction whose order
//! depends on the parallel schedule can come out bit-identical on one
//! configuration and not on another, and recording that as `Exact` would
//! manufacture a contract nobody can keep. Anything downstream that needs a
//! guarantee must read [`Verdict::is_contracted`], not "it agreed last time".

use super::error::ExecutionError;

/// The response a refusal requires — re-exported from `larql-execution`.
///
/// It moved there when `FfnBackend`'s error channel needed to name it:
/// `FfnBackend` lives in `larql-compute`, which sits *below* this crate, so the
/// vocabulary could not stay here and still cross that boundary. Re-exported so
/// existing `runtime::RefusalKind` paths keep working.
pub use larql_execution::RefusalKind;

impl ExecutionError {
    /// Which kind of refusal this is, for a sweep that must not report a
    /// missing operand as a wrong answer.
    ///
    /// Exhaustive on purpose: a new variant will not compile until someone has
    /// decided which of the three it is, which is the point. The default for an
    /// unclassified failure must never be `Residency` — that is the one kind
    /// that reads as "nothing is wrong".
    pub const fn refusal(&self) -> RefusalKind {
        match self {
            // The route was right; the operand lives elsewhere.
            Self::SelectedExpertNotResident { .. } => RefusalKind::Residency,

            // Well-formed, but nothing bound serves it.
            Self::UnsupportedFormat { .. }
            | Self::UnsupportedView { .. }
            | Self::KernelOperandUnsuitable { .. }
            | Self::KernelActivationUnsupported { .. } => RefusalKind::Unsupported,

            // The binding is wrong. `NonFinite*` land here rather than in a
            // category of their own: a NaN reaching a routing decision is a
            // statement about the bytes that were bound, not about arithmetic
            // this runtime performed.
            Self::DimensionMismatch { .. }
            | Self::ShortRegion { .. }
            | Self::RowOutOfRange { .. }
            | Self::NotAMatrix { .. }
            | Self::NonFiniteRouterScore { .. }
            | Self::NonFiniteExpertScale { .. }
            | Self::ExpertOutOfRange { .. } => RefusalKind::BindingDefect,
        }
    }
}

/// The outcome of comparing one bound execution against the incumbent.
///
/// Ordered by how much it claims, weakest last, so a sweep can report its
/// worst outcome by taking the maximum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Verdict {
    /// Every asserted checkpoint was bit-identical. The strongest claim, and
    /// the only one that is a contract.
    Exact,
    /// The comparison came out bit-identical without being asserted — a
    /// reduction whose order the parallel schedule chose, say. True of this
    /// run; not promised of the next.
    ObservedExact,
    /// Within a named, justified tolerance. The tolerance has to be stated
    /// wherever this is produced; a bare "equivalent" is the vague bucket this
    /// enum exists to prevent.
    Equivalent,
    /// Execution completed and the answers disagree. The genuine failure.
    NumericMismatch,
    /// Execution did not complete. See [`RefusalKind`] for who acts on it.
    Refused(RefusalKind),
}

impl Verdict {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::ObservedExact => "observed_exact",
            Self::Equivalent => "equivalent",
            Self::NumericMismatch => "numeric_mismatch",
            Self::Refused(kind) => kind.name(),
        }
    }

    /// Whether this outcome is a guarantee rather than a measurement.
    ///
    /// Only [`Self::Exact`] is. `ObservedExact` deliberately is not: promoting
    /// an observation to a contract is how a scheduling-dependent result
    /// becomes a promise nobody can keep.
    pub const fn is_contracted(self) -> bool {
        matches!(self, Self::Exact)
    }

    /// Whether the two paths agreed, to whatever strength.
    pub const fn agreed(self) -> bool {
        matches!(self, Self::Exact | Self::ObservedExact | Self::Equivalent)
    }

    /// Whether this outcome indicts the execution.
    ///
    /// A residency refusal does not: the route was correct and the operand was
    /// absent. Neither does an unsupported one — it says a kernel is missing,
    /// which is a gap rather than a defect.
    pub const fn indicts_execution(self) -> bool {
        matches!(
            self,
            Self::NumericMismatch | Self::Refused(RefusalKind::BindingDefect)
        )
    }

    /// Classify a completed comparison.
    ///
    /// `asserted` says whether the checkpoints that came out identical were
    /// ones the harness contracts, or ones it merely watched. Passing `true`
    /// for a schedule-dependent comparison is the mistake this parameter
    /// exists to make visible at the call site.
    pub const fn from_comparison(identical: bool, within_tolerance: bool, asserted: bool) -> Self {
        match (identical, asserted) {
            (true, true) => Self::Exact,
            (true, false) => Self::ObservedExact,
            (false, _) if within_tolerance => Self::Equivalent,
            (false, _) => Self::NumericMismatch,
        }
    }
}

impl From<&ExecutionError> for Verdict {
    fn from(error: &ExecutionError) -> Self {
        Self::Refused(error.refusal())
    }
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}
