//! Operation admission (spec §9, §11, §15).
//!
//! Three questions, kept apart:
//!
//! ```text
//! Can this operation run?          → admission (here)
//! How faithfully would it run?     → authority, folded over the regions THIS
//!                                    operation consumes
//! Which implementation runs it?    → kernel binding, elsewhere
//! ```
//!
//! Admission is inferred first and authority attached afterwards, because the
//! two answer different questions and one of them has no answer when the
//! selection is contradictory.
//!
//! # The decisive property: non-interference
//!
//! Changing a component an operation does not use must not change that
//! operation's capability. WALK reads gate rows; making every `down` region
//! unreadable must leave the WALK report byte-for-byte identical. That is what
//! proves traversal facts are *projected per operation* rather than globally
//! summarised, and it is why each operation declares a narrow dependency
//! surface rather than consulting a shared "is the index healthy" verdict.

use super::authority::Fidelity;
use super::coordinate::RegionCoordinate;
use super::plan::QualifiedOperationRoute;

/// Why an operation cannot run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationFailure {
    /// A region this operation needs is unusable, with its exact coordinate.
    RequiredRegionUnusable {
        coordinate: RegionCoordinate,
        cause: String,
    },
    /// The selection is self-contradictory. Fails closed — never downgraded
    /// into a weak authority.
    InvalidSelection { detail: String },
    /// A document-level input this operation needs is absent.
    MissingDocumentInput { what: &'static str },
    /// The operation needs a policy declaration the caller did not supply.
    /// Notably: remote execution cannot be inferred from local absence.
    MissingContract { what: &'static str },
    /// No layer offers an executable route for this operation.
    NoExecutableRoute { layer: u32 },
}

impl OperationFailure {
    pub fn describe(&self) -> String {
        match self {
            Self::RequiredRegionUnusable { coordinate, cause } => {
                format!("{}: {cause}", coordinate.describe())
            }
            Self::InvalidSelection { detail } => format!("invalid selection: {detail}"),
            Self::MissingDocumentInput { what } => format!("missing document input: {what}"),
            Self::MissingContract { what } => {
                format!("no declared contract for {what}; it cannot be inferred")
            }
            Self::NoExecutableRoute { layer } => {
                format!("layer {layer} has no executable route")
            }
        }
    }
}

/// Something absent that reduces richness without preventing the operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Degradation {
    /// Optional query metadata is absent. Reduces label richness; never
    /// affects correctness (§15.3).
    MissingQueryMetadata { what: &'static str },
    /// A bank cannot be browsed, so its features are outside query reach while
    /// other banks remain reachable.
    BankNotBrowsable { bank_id: u16, reason: String },
}

impl Degradation {
    pub fn describe(&self) -> String {
        match self {
            Self::MissingQueryMetadata { what } => {
                format!("{what} absent — reduced richness, unchanged correctness")
            }
            Self::BankNotBrowsable { bank_id, reason } => {
                format!("bank {bank_id} not browsable: {reason}")
            }
        }
    }
}

/// Whether an operation can run, and how completely.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperationAdmission {
    Available,
    Degraded { reasons: Vec<Degradation> },
    Unavailable { reasons: Vec<OperationFailure> },
}

impl OperationAdmission {
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available | Self::Degraded { .. })
    }

    /// Whether admission failed because the selection is contradictory, as
    /// opposed to incomplete. Contradictions must fail closed.
    pub fn is_invalid_selection(&self) -> bool {
        match self {
            Self::Unavailable { reasons } => reasons
                .iter()
                .any(|r| matches!(r, OperationFailure::InvalidSelection { .. })),
            _ => false,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Available => "available".into(),
            Self::Degraded { reasons, .. } => format!(
                "available, degraded: {}",
                reasons
                    .iter()
                    .map(|r| r.describe())
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
            Self::Unavailable { reasons } => format!(
                "unavailable: {}",
                reasons
                    .iter()
                    .map(|r| r.describe())
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
        }
    }
}

/// Admission plus every route that admits it.
///
/// Authority is deliberately **not** a field here. An operation can run more
/// than one way, the ways can differ in fidelity, and no route has been bound
/// yet — so there is no single number that is both meaningful and honest. A
/// caller wanting one must ask for a *ceiling* and be told it is achievable
/// rather than achieved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationCapability {
    pub admission: OperationAdmission,
    /// Empty exactly when unavailable. That equivalence is the fail-closed
    /// invariant: no routes means no authority, and a contradictory selection
    /// therefore cannot acquire a weak-but-valid fidelity by default.
    pub routes: Vec<QualifiedOperationRoute>,
}

impl OperationCapability {
    pub fn available(routes: Vec<QualifiedOperationRoute>) -> Self {
        Self {
            admission: OperationAdmission::Available,
            routes,
        }
    }

    pub fn degraded(routes: Vec<QualifiedOperationRoute>, reasons: Vec<Degradation>) -> Self {
        Self {
            admission: OperationAdmission::Degraded { reasons },
            routes,
        }
    }

    pub fn unavailable(reasons: Vec<OperationFailure>) -> Self {
        Self {
            admission: OperationAdmission::Unavailable { reasons },
            routes: Vec::new(),
        }
    }

    pub fn is_available(&self) -> bool {
        self.admission.is_available()
    }

    /// The strongest fidelity any admitted route could achieve.
    ///
    /// Named *achievable* on purpose. Binding has not chosen a route, so this
    /// is a ceiling; quoting it as the fidelity of an execution would describe
    /// a decision nobody has made.
    pub fn best_achievable_authority(&self) -> Option<Fidelity> {
        self.routes
            .iter()
            .map(|r| r.best_achievable_authority().level)
            .max()
    }

    /// The weakest fidelity binding could land on across admitted routes.
    pub fn worst_achievable_authority(&self) -> Option<Fidelity> {
        self.routes
            .iter()
            .map(|r| r.worst_achievable_authority().level)
            .min()
    }

    /// Whether every admitted route carries the same settled fidelity, so the
    /// ceiling is also the answer.
    pub fn authority_is_settled(&self) -> bool {
        !self.routes.is_empty()
            && self.best_achievable_authority() == self.worst_achievable_authority()
    }

    /// Internal consistency: routes exist exactly when the operation admits.
    pub fn is_well_formed(&self) -> bool {
        // Reads oddly but is the honest form: admitted iff routes exist.
        self.is_available() != self.routes.is_empty()
    }
}
