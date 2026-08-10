//! Why an execution refused, classified by the response it requires.
//!
//! This crate exists because a dependency cycle pointed at a missing layer.
//! `FfnBackend` lives in `larql-compute`; the refusal vocabulary grew up in
//! `larql-vindex`, which depends on `larql-compute`. Neither could name the
//! other's type, so the classification could not cross the boundary that most
//! needed it — the one between a route that refused and an engine deciding what
//! to do about it.
//!
//! # The axis is the response, not the cause
//!
//! [`RefusalKind`] has three variants because there are three distinct
//! responses, and each has a different person or system on the other end:
//!
//! ```text
//! Residency       same operation, operand becomes available  → may succeed
//! Unsupported     same operands, a different capable executor → may succeed
//! BindingDefect   the binding or artifact itself must change
//! ```
//!
//! That partition is what makes the enum actionable. Finer reasons — an
//! unsupported activation, a decomposed projection where a fused one was
//! needed, a misaligned base — are *causes of* `Unsupported`, not separate
//! responses, and they belong in the concrete error's payload where they can
//! carry their own detail. Promoting them here would put two abstraction levels
//! in one enum and leave a caller unable to switch on the thing it must act on.
//!
//! # `BindingDefect` is load-bearing
//!
//! It is the only kind that means *reject the artifact*. An expert id outside
//! the router's population, a region shorter than its declared shape, an
//! internally inconsistent plan — none of those is fixed by fetching an operand
//! or by choosing another kernel, and classifying them as either would send
//! someone to repair the wrong thing.
//!
//! The invariant to hold when adding a variant or classifying a new error:
//!
//! > `BindingDefect` means that repeating the operation with more residency, or
//! > through any other capable executor, cannot make *this* bound plan valid.
//!
//! # Transport failures are not a fourth kind
//!
//! A network timeout, a refused connection, an expired lease — those are
//! execution-*attempt* failures, often retryable, and they belong to whatever
//! transport owns them. One may eventually *conclude* in `Residency`, when the
//! semantic finding is that the operand is unavailable to this plan. Admitting
//! them directly would turn this enum from execution semantics into a catalogue
//! of everything that can go wrong operationally.

// See crates/larql-core/src/lib.rs for the pattern-2 rationale.
// core::error::Error was stabilized in 1.81 (workspace MSRV 1.88), so
// unlike patterns 1/4/6 there is no substitute crate needed here -- just
// a source swap. core::fmt is the same API as std::fmt (fmt is not a
// std-only concept), and Box needs alloc explicitly since no_std has no
// prelude Box.
#![cfg_attr(target_arch = "wasm32", no_std)]
#[cfg(target_arch = "wasm32")]
extern crate alloc;

#[cfg(target_arch = "wasm32")]
use alloc::boxed::Box;
#[cfg(target_arch = "wasm32")]
use core::error::Error;
#[cfg(target_arch = "wasm32")]
use core::fmt;
#[cfg(not(target_arch = "wasm32"))]
use std::error::Error;
#[cfg(not(target_arch = "wasm32"))]
use std::fmt;

/// The response a refusal requires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RefusalKind {
    /// A valid operand exists, and is not available here.
    ///
    /// **Not a defect.** A shard that does not hold an expert is behaving as
    /// designed; the response is to fetch, load or reroute. Counting these as
    /// failures makes a sweep's headline number a measurement of slice coverage
    /// rather than of execution.
    Residency,
    /// A valid operation this route cannot execute.
    ///
    /// The representation is well-formed and the operands are present; no bound
    /// kernel serves them. The response is another executor, or another stored
    /// variant of the same component — never a quiet reinterpretation of the
    /// bytes.
    Unsupported,
    /// The bound artifact or plan violates its own contract.
    ///
    /// The only kind that means reject the index and repair whatever produced
    /// it.
    BindingDefect,
}

impl RefusalKind {
    /// Every kind, so a consumer sweeping them cannot silently narrow.
    pub const ALL: [Self; 3] = [Self::Residency, Self::Unsupported, Self::BindingDefect];

    /// Snake case, matching the sibling outcome names it is printed beside
    /// (`ok`, `declined`) — one table, one convention.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Residency => "residency",
            Self::Unsupported => "unsupported",
            Self::BindingDefect => "binding_defect",
        }
    }

    /// Whether this indicts the artifact rather than the environment.
    ///
    /// The question an operator actually asks: is something broken, or is
    /// something merely elsewhere?
    pub const fn indicts_the_artifact(self) -> bool {
        matches!(self, Self::BindingDefect)
    }

    /// Whether the same bound plan could succeed given a different environment
    /// — more residency, or another capable executor.
    pub const fn is_recoverable_without_rebinding(self) -> bool {
        matches!(self, Self::Residency | Self::Unsupported)
    }
}

impl fmt::Display for RefusalKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// An error that carries its response category across a crate boundary.
///
/// Concrete errors stay in the crates that own them, with all their detail —
/// which expert, which bank, which axis. This trait carries only the one thing
/// a caller two layers up must switch on.
pub trait ExecutionRefusal: Error + Send + Sync + 'static {
    fn kind(&self) -> RefusalKind;
}

/// A refusal crossing a boundary that cannot name its concrete type.
pub type BoxRefusal = Box<dyn ExecutionRefusal>;

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Refused(RefusalKind);

    impl fmt::Display for Refused {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "refused: {}", self.0)
        }
    }
    impl Error for Refused {}
    impl ExecutionRefusal for Refused {
        fn kind(&self) -> RefusalKind {
            self.0
        }
    }

    #[test]
    fn every_kind_has_a_distinct_name() {
        // The names reach operator-facing diagnostics, so two kinds sharing one
        // would make a report unactionable exactly where it matters.
        let mut names: Vec<&str> = RefusalKind::ALL.iter().map(|k| k.name()).collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count);
    }

    #[test]
    fn only_a_binding_defect_indicts_the_artifact() {
        // The partition that decides who is sent to fix something.
        assert!(RefusalKind::BindingDefect.indicts_the_artifact());
        assert!(!RefusalKind::Residency.indicts_the_artifact());
        assert!(!RefusalKind::Unsupported.indicts_the_artifact());
    }

    #[test]
    fn residency_and_unsupported_can_succeed_without_rebinding() {
        // The stated invariant: more residency or another executor may rescue
        // those two, and cannot rescue a binding defect.
        assert!(RefusalKind::Residency.is_recoverable_without_rebinding());
        assert!(RefusalKind::Unsupported.is_recoverable_without_rebinding());
        assert!(!RefusalKind::BindingDefect.is_recoverable_without_rebinding());
    }

    #[test]
    fn the_two_predicates_partition_the_vocabulary() {
        // Guards against a future variant that is neither or both, which would
        // leave a caller with no branch to take.
        for kind in RefusalKind::ALL {
            assert_ne!(
                kind.indicts_the_artifact(),
                kind.is_recoverable_without_rebinding(),
                "{kind} is not on exactly one side of the partition"
            );
        }
    }

    #[test]
    fn a_kind_displays_as_its_name() {
        assert_eq!(RefusalKind::Residency.to_string(), "residency");
        assert_eq!(RefusalKind::BindingDefect.to_string(), "binding_defect");
    }

    #[test]
    fn a_boxed_refusal_keeps_its_kind_and_its_message() {
        // The point of the trait: the classification survives a boundary that
        // cannot name the concrete type, and the detail survives with it.
        let boxed: BoxRefusal = Box::new(Refused(RefusalKind::Residency));
        assert_eq!(boxed.kind(), RefusalKind::Residency);
        assert!(boxed.to_string().contains("residency"));
        let source: &dyn Error = boxed.as_ref();
        assert!(source.source().is_none());
    }
}
