//! What a WALK is actually asking for (spec §15.1, §15.4).
//!
//! WALK capability is **request-scoped**. "Is WALK available?" has no answer
//! without a target: a document can hold one bank with a direct gate route and
//! another that is serving-only with browse mode `none`, and both "yes" and
//! "no" are defensible until someone says which bank they meant.
//!
//! # Missing banks change the answer, not the richness
//!
//! Absent `query/` metadata reduces label richness and leaves rankings correct
//! (§15.3), so it degrades. A missing *searchable gate population* is not like
//! that: it changes the result set. Silently dropping an unwalkable bank and
//! returning a global ranking over what remained would be a wrong answer
//! wearing a successful one's clothes. So an unwalkable requested bank fails
//! the request unless partial results were explicitly asked for.
//!
//! # A residual vector is always residual
//!
//! `ResidualVector` means the *model's* residual space, never a bank's latent
//! space. The distinction is load-bearing: a latent bank's gate rows live in
//! its own width, and a caller handing over a vector that happens to match
//! that width would otherwise bypass the `routed_input` projection entirely
//! and silently change WALK semantics. Width is therefore checked against the
//! model's residual dimension, not against the bank being searched.

use super::coordinate::BankCoordinate;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// How a query vector reaches WALK.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalkInput {
    /// A pre-built vector in the model's residual space.
    ///
    /// Needs no tokenizer and no embedding table — a low-level caller must not
    /// be refused for lacking text-query infrastructure it never uses.
    ResidualVector { dimension: u32 },
    /// Text, to be tokenised and embedded before the walk.
    TextQuery,
}

impl WalkInput {
    /// Whether this input needs the document's text-query machinery.
    pub fn needs_text_construction(&self) -> bool {
        matches!(self, Self::TextQuery)
    }
}

/// Which banks the request covers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WalkTarget {
    Bank(BankCoordinate),
    Banks(Vec<BankCoordinate>),
    /// Every bank declared part of the searchable surface.
    ///
    /// Deliberately *not* "whichever banks happen to work". The declared
    /// surface is the request; banks in it that cannot be walked are failures,
    /// not omissions.
    AllWalkableBanks,
}

impl WalkTarget {
    /// The banks this target names, given the declared searchable surface.
    pub fn resolve(&self, declared_surface: &[BankCoordinate]) -> Vec<BankCoordinate> {
        match self {
            Self::Bank(b) => vec![*b],
            Self::Banks(bs) => bs.clone(),
            Self::AllWalkableBanks => declared_surface.to_vec(),
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Bank(b) => b.describe(),
            Self::Banks(bs) => bs
                .iter()
                .map(|b| b.describe())
                .collect::<Vec<_>>()
                .join(", "),
            Self::AllWalkableBanks => "the declared searchable surface".into(),
        }
    }
}

/// Whether an incomplete result is acceptable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PartialPolicy {
    /// Any requested bank that cannot be walked fails the whole request.
    ///
    /// The default, because a ranking computed over a subset of the requested
    /// population is a different answer, not a degraded one.
    #[default]
    RequireAll,
    /// Explicitly accept a result over whatever subset is walkable. The dropped
    /// banks are reported so the caller knows the ranking's true scope.
    AllowPartial,
}

/// A complete WALK request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalkRequest {
    pub input: WalkInput,
    pub target: WalkTarget,
    pub partial: PartialPolicy,
}

impl WalkRequest {
    /// A single-bank request with the safe default.
    pub fn bank(bank: BankCoordinate, input: WalkInput) -> Self {
        Self {
            input,
            target: WalkTarget::Bank(bank),
            partial: PartialPolicy::RequireAll,
        }
    }

    /// A request over the whole declared searchable surface.
    pub fn surface(input: WalkInput) -> Self {
        Self {
            input,
            target: WalkTarget::AllWalkableBanks,
            partial: PartialPolicy::RequireAll,
        }
    }

    pub fn allowing_partial(mut self) -> Self {
        self.partial = PartialPolicy::AllowPartial;
        self
    }

    pub fn tolerates_partial(&self) -> bool {
        self.partial == PartialPolicy::AllowPartial
    }
}

/// Why a query vector cannot be used as given.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryVectorFault {
    /// The vector's width is not the model's residual width.
    ///
    /// Checked against the *residual* dimension deliberately. A vector matching
    /// some bank's latent width is still wrong — accepting it would skip the
    /// residual→latent projection and answer a different question.
    WrongWidth { supplied: u32, residual: u32 },
}

impl QueryVectorFault {
    pub fn describe(&self) -> String {
        match self {
            Self::WrongWidth { supplied, residual } => format!(
                "query vector is {supplied}-wide; the model's residual space is {residual}-wide \
                 (a vector matching a bank's latent width is still wrong — it would bypass the \
                 residual-to-latent projection)"
            ),
        }
    }
}

/// Check a supplied residual vector against the model's residual width.
pub fn validate_query_vector(
    input: WalkInput,
    residual_dimension: u32,
) -> Result<(), QueryVectorFault> {
    match input {
        WalkInput::ResidualVector { dimension } if dimension != residual_dimension => {
            Err(QueryVectorFault::WrongWidth {
                supplied: dimension,
                residual: residual_dimension,
            })
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RESIDUAL: u32 = 7_168;
    const LATENT: u32 = 3_584;

    fn bank(layer: u32, id: u16) -> BankCoordinate {
        BankCoordinate::new(layer, id)
    }

    #[test]
    fn a_residual_vector_needs_no_text_machinery() {
        assert!(!WalkInput::ResidualVector {
            dimension: RESIDUAL
        }
        .needs_text_construction());
    }

    #[test]
    fn a_text_query_needs_text_machinery() {
        assert!(WalkInput::TextQuery.needs_text_construction());
    }

    #[test]
    fn a_correct_width_residual_vector_validates() {
        assert_eq!(
            validate_query_vector(
                WalkInput::ResidualVector {
                    dimension: RESIDUAL
                },
                RESIDUAL
            ),
            Ok(())
        );
    }

    #[test]
    fn a_latent_width_vector_is_refused_even_though_a_bank_would_accept_it() {
        // The trap: K3's latent banks are 3584 wide. A 3584-wide query would
        // dot-product against latent gate rows "successfully" while skipping
        // routed_input entirely, answering a different question.
        let err = validate_query_vector(WalkInput::ResidualVector { dimension: LATENT }, RESIDUAL)
            .unwrap_err();
        assert_eq!(
            err,
            QueryVectorFault::WrongWidth {
                supplied: LATENT,
                residual: RESIDUAL,
            }
        );
        assert!(err.describe().contains("bypass"), "{}", err.describe());
    }

    #[test]
    fn the_width_diagnosis_names_both_numbers() {
        let s = QueryVectorFault::WrongWidth {
            supplied: 256,
            residual: RESIDUAL,
        }
        .describe();
        assert!(s.contains("256-wide"), "{s}");
        assert!(s.contains("7168-wide"), "{s}");
    }

    #[test]
    fn a_text_query_is_not_width_checked() {
        // Its vector does not exist yet.
        assert_eq!(
            validate_query_vector(WalkInput::TextQuery, RESIDUAL),
            Ok(())
        );
    }

    #[test]
    fn a_single_bank_target_resolves_to_itself() {
        let surface = vec![bank(0, 0), bank(1, 0)];
        assert_eq!(
            WalkTarget::Bank(bank(1, 0)).resolve(&surface),
            vec![bank(1, 0)]
        );
    }

    #[test]
    fn an_explicit_list_resolves_verbatim() {
        let surface = vec![bank(0, 0), bank(1, 0), bank(2, 0)];
        let target = WalkTarget::Banks(vec![bank(0, 0), bank(2, 0)]);
        assert_eq!(target.resolve(&surface), vec![bank(0, 0), bank(2, 0)]);
    }

    #[test]
    fn all_walkable_means_the_declared_surface_not_the_working_subset() {
        // The whole point of the naming: a bank in the surface that cannot be
        // walked is a failure to report, not a member to quietly drop.
        let surface = vec![bank(0, 0), bank(1, 0)];
        assert_eq!(WalkTarget::AllWalkableBanks.resolve(&surface), surface);
    }

    #[test]
    fn an_empty_surface_resolves_to_nothing() {
        assert!(WalkTarget::AllWalkableBanks.resolve(&[]).is_empty());
    }

    #[test]
    fn requests_default_to_requiring_every_targeted_bank() {
        // A ranking over a subset of the requested population is a different
        // answer, not a degraded one, so partial must be opt-in.
        assert!(!WalkRequest::bank(bank(0, 0), WalkInput::TextQuery).tolerates_partial());
        assert!(!WalkRequest::surface(WalkInput::TextQuery).tolerates_partial());
        assert_eq!(PartialPolicy::default(), PartialPolicy::RequireAll);
    }

    #[test]
    fn partial_results_are_opt_in() {
        let r = WalkRequest::surface(WalkInput::TextQuery).allowing_partial();
        assert!(r.tolerates_partial());
    }

    #[test]
    fn targets_describe_themselves_for_diagnostics() {
        assert_eq!(WalkTarget::Bank(bank(3, 1)).describe(), "layer 3 bank 1");
        assert!(WalkTarget::Banks(vec![bank(0, 0), bank(1, 0)])
            .describe()
            .contains(", "));
        assert!(WalkTarget::AllWalkableBanks
            .describe()
            .contains("searchable surface"));
    }

    #[test]
    fn a_surface_request_keeps_its_input_mode() {
        let r = WalkRequest::surface(WalkInput::ResidualVector {
            dimension: RESIDUAL,
        });
        assert_eq!(
            r.input,
            WalkInput::ResidualVector {
                dimension: RESIDUAL
            }
        );
        assert_eq!(r.target, WalkTarget::AllWalkableBanks);
    }
}
