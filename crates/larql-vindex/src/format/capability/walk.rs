//! WALK inference (spec §15.1, §15.2, §15.4).
//!
//! The acceptance test for this whole module:
//!
//! > A WALK report must be derivable without ever asking whether the model can
//! > perform an expert forward pass.
//!
//! So this function never consults `up`, `down`, expert-output codecs, router
//! semantics, decode traversal, kernel maturity or `query/` metadata. If
//! unreadable `down` regions, broken routing or absent shared experts can
//! perturb a WALK report, the projection is not real.
//!
//! # The declared surface is a declaration
//!
//! Membership comes from an explicit document-level list, never from filtering
//! banks by health. Defining the surface as "banks whose browse mode is not
//! none" would make a broken bank *disappear* from the request and recreate
//! working-subset semantics through the back door. A bank declared searchable
//! that carries `browse: none`, an unreadable gate codec, or a missing latent
//! transform stays in the report — as a failure.
//!
//! # Coverage and fidelity are different axes
//!
//! A partial result set does not lower authority. If bank A's gate rows are
//! source-exact and bank B is unavailable, a partial walk over A returns
//! source-exact scores over an incomplete population. The numbers did not
//! become approximate; the result set became smaller. Completeness therefore
//! sits *beside* authority, never inside the fold — the same distinction
//! already drawn for absent query metadata.

use crate::format::lyrw2::browse_mode::BrowseMode;
use crate::format::lyrw2::region_format::Packing;

use super::coordinate::{BankCoordinate, RegionCoordinate};
use super::operation::{OperationCapability, OperationFailure};
use super::plan::{
    OperationPlan, PlanChoice, PlannedRegion, QualifiedAlternative, QualifiedOperationRoute,
};
use super::role::ReferenceSupport;
use super::walk_request::{validate_query_vector, PartialPolicy, WalkRequest};

/// One way to reach a bank's gate rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateAccess {
    /// `Direct` for an own-region gate; `Strided` for the gate half of a fused
    /// region. `None` never appears here — it is a bank-level refusal.
    pub mode: BrowseMode,
    pub region: PlannedRegion,
    pub packing: Packing,
    pub support: ReferenceSupport,
}

impl GateAccess {
    /// Whether this access can actually deliver gate rows.
    ///
    /// Strided access additionally requires a packing that permits striding
    /// into the gate half without decoding `up` — which is the whole cost
    /// browse was avoiding.
    pub fn is_usable(&self) -> bool {
        self.support.is_supported() && self.mode.is_satisfiable_by(self.packing)
    }

    pub fn reason_unusable(&self) -> Option<String> {
        if self.is_usable() {
            return None;
        }
        if !self.support.is_supported() {
            return Some(self.support.describe());
        }
        Some(format!(
            "{} access needs a packing that permits striding; '{}' does not",
            self.mode.name(),
            self.packing.name()
        ))
    }
}

/// Everything WALK needs to know about one bank. Produced by resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalkableBank {
    pub coordinate: BankCoordinate,
    /// Declared browse eligibility. `None` refuses the bank outright.
    pub browse: BrowseMode,
    /// Whether gate rows live in a latent space, requiring the query to be
    /// projected through `routed_input` first (§15.4).
    pub is_latent: bool,
    pub gate_accesses: Vec<GateAccess>,
    /// residual → latent projection. Required exactly when `is_latent`.
    pub routed_input: Option<PlannedRegion>,
}

/// Query-construction inputs, needed only for `TextQuery`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextQueryInputs {
    pub has_tokenizer: bool,
    /// Embedding weights. A numerical component, so it enters route authority
    /// — unlike the tokenizer, which has availability but no fidelity.
    pub embeddings: Option<PlannedRegion>,
}

/// Document-level facts WALK depends on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalkEnvironment {
    pub residual_dimension: u32,
    /// The declared searchable surface. An explicit list, never a filter.
    pub declared_surface: Vec<BankCoordinate>,
    pub banks: Vec<WalkableBank>,
    pub text_query: TextQueryInputs,
}

impl WalkEnvironment {
    fn bank(&self, at: BankCoordinate) -> Option<&WalkableBank> {
        self.banks.iter().find(|b| b.coordinate == at)
    }
}

/// Why one targeted bank could not be walked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetFailure {
    pub bank: BankCoordinate,
    pub reason: String,
}

/// Whether the result set covers everything requested.
///
/// Deliberately separate from authority: losing coverage and losing numerical
/// faithfulness are different axes, and folding one into the other would make
/// an exact-but-incomplete answer indistinguishable from an approximate one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResultSetCompleteness {
    Complete,
    Partial {
        included: Vec<BankCoordinate>,
        omitted: Vec<TargetFailure>,
    },
}

impl ResultSetCompleteness {
    pub fn is_complete(&self) -> bool {
        matches!(self, Self::Complete)
    }
}

/// A WALK capability with its coverage stated alongside.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalkCapability {
    pub capability: OperationCapability,
    pub completeness: ResultSetCompleteness,
}

/// Infer WALK capability for one request.
pub fn infer_walk(env: &WalkEnvironment, request: &WalkRequest) -> WalkCapability {
    if let Err(fault) = validate_query_vector(request.input, env.residual_dimension) {
        return refuse(vec![OperationFailure::MissingDocumentInput {
            what: "query vector of the model's residual width",
        }
        .with_detail(fault.describe())]);
    }

    let mut fixed_regions = Vec::new();
    if request.input.needs_text_construction() {
        if !env.text_query.has_tokenizer {
            return refuse(vec![OperationFailure::MissingDocumentInput {
                what: "tokenizer",
            }]);
        }
        match &env.text_query.embeddings {
            Some(e) => fixed_regions.push(e.clone()),
            None => {
                return refuse(vec![OperationFailure::MissingDocumentInput {
                    what: "embeddings",
                }])
            }
        }
    }

    let targets = request.target.resolve(&env.declared_surface);
    if targets.is_empty() {
        return refuse(vec![OperationFailure::MissingDocumentInput {
            what: "a non-empty searchable surface",
        }]);
    }

    let mut choices = Vec::new();
    let mut included = Vec::new();
    let mut omitted = Vec::new();

    for target in targets {
        match walk_routes_for(env, target) {
            Ok(alternatives) => {
                included.push(target);
                choices.push(PlanChoice {
                    bank: target,
                    alternatives,
                });
            }
            Err(reason) => omitted.push(TargetFailure {
                bank: target,
                reason,
            }),
        }
    }

    if !omitted.is_empty() && request.partial == PartialPolicy::RequireAll {
        return refuse(
            omitted
                .into_iter()
                .map(|f| OperationFailure::RequiredRegionUnusable {
                    coordinate: RegionCoordinate::new(
                        f.bank.layer,
                        f.bank.bank_id,
                        None,
                        crate::format::lyrw2::region_role::RegionRole::Gate,
                    ),
                    cause: f.reason,
                })
                .collect(),
        );
    }

    if choices.is_empty() {
        return refuse(vec![OperationFailure::NoExecutableRoute {
            layer: included.first().map(|b| b.layer).unwrap_or_default(),
        }]);
    }

    let plan = OperationPlan {
        fixed_components: fixed_regions
            .iter()
            .map(PlannedRegion::as_component)
            .collect(),
        choices,
    };
    let completeness = if omitted.is_empty() {
        ResultSetCompleteness::Complete
    } else {
        ResultSetCompleteness::Partial { included, omitted }
    };

    WalkCapability {
        capability: OperationCapability::available(vec![QualifiedOperationRoute::new(plan)]),
        completeness,
    }
}

/// Every usable gate route for one bank, or why there are none.
fn walk_routes_for(
    env: &WalkEnvironment,
    target: BankCoordinate,
) -> Result<Vec<QualifiedAlternative>, String> {
    let Some(bank) = env.bank(target) else {
        return Err("bank is declared searchable but absent from the index".into());
    };
    if !bank.browse.is_browsable() {
        return Err("bank declares browse mode 'none'".into());
    }

    // A latent bank's query must pass through routed_input, so the transform
    // is part of every route rather than an afterthought on authority.
    let transform = if bank.is_latent {
        match &bank.routed_input {
            Some(t) => Some(t.clone()),
            None => {
                return Err(
                    "bank is latent-space but declares no routed_input transform to project \
                     the query through"
                        .into(),
                )
            }
        }
    } else {
        None
    };

    let mut alternatives = Vec::new();
    for access in &bank.gate_accesses {
        if !access.is_usable() {
            continue;
        }
        let mut regions = vec![access.region.clone()];
        if let Some(t) = &transform {
            regions.push(t.clone());
        }
        alternatives.push(QualifiedAlternative {
            alternative: gate_layout_for(access.mode),
            regions,
            components: Vec::new(),
        });
    }

    if alternatives.is_empty() {
        let why = bank
            .gate_accesses
            .iter()
            .filter_map(|a| a.reason_unusable())
            .collect::<Vec<_>>()
            .join("; ");
        return Err(if why.is_empty() {
            "bank exposes no gate region".into()
        } else {
            why
        });
    }
    Ok(alternatives)
}

/// The role layout a gate access reads. WALK's alternatives are gate-shaped,
/// not programme-shaped — it never needs `up` or `down`.
fn gate_layout_for(mode: BrowseMode) -> &'static [crate::format::lyrw2::region_role::RegionRole] {
    use crate::format::lyrw2::region_role::RegionRole;
    const DIRECT: &[RegionRole] = &[RegionRole::Gate];
    const STRIDED: &[RegionRole] = &[RegionRole::GateUpFused];
    match mode {
        BrowseMode::Strided => STRIDED,
        _ => DIRECT,
    }
}

fn refuse(reasons: Vec<OperationFailure>) -> WalkCapability {
    WalkCapability {
        capability: OperationCapability::unavailable(reasons),
        completeness: ResultSetCompleteness::Complete,
    }
}

impl OperationFailure {
    /// Attach detail to a document-input failure without a new variant.
    fn with_detail(self, detail: String) -> Self {
        match self {
            Self::MissingDocumentInput { what } => Self::RequiredRegionUnusable {
                coordinate: RegionCoordinate::new(
                    0,
                    0,
                    None,
                    crate::format::lyrw2::region_role::RegionRole::Gate,
                ),
                cause: format!("{what}: {detail}"),
            },
            other => other,
        }
    }
}
