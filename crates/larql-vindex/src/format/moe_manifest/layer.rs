//! Per-layer MoE description (format spec §8.1).
//!
//! Per-layer, deliberately. A global `first_k_dense_replace` field would have
//! handled Kimi-Linear's leading dense layer and then failed on Inkling-Small,
//! whose dense MLP sits at layer index 2 — mid-stack. Per-layer manifests
//! handle arbitrary dense/MoE schedules for free, which is the point.

use serde::{Deserialize, Serialize};

use super::bank_ref::BankRef;
use super::router::Router;

/// The space a layer's experts consume and produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputSpace {
    /// Experts read and write the residual stream directly.
    Residual,
    /// Experts operate in a projected latent space; the layer's transforms
    /// carry the residual↔latent hop.
    Latent,
}

/// Residual↔latent projections for a latent-space layer (§8.1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transforms {
    /// residual → latent. Also what WALK projects a query through before
    /// dot-producting against latent gate rows (§15.4).
    pub routed_input: String,
    /// latent → residual.
    pub routed_output: String,
}

/// How selected expert outputs are combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reduction {
    GateWeightedSum,
}

/// How the reduced expert output rejoins the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Combine {
    ResidualAdd,
}

/// One layer's complete MoE programme.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoeLayer {
    pub layer: u32,
    pub input_space: InputSpace,
    pub router: Router,
    /// Null for a conventional residual-space MoE.
    #[serde(default)]
    pub transforms: Option<Transforms>,
    pub routed_bank: BankRef,
    /// Absent for models with no always-active experts (GPT-OSS, Gemma).
    #[serde(default)]
    pub shared_bank: Option<BankRef>,
    pub reduction: Reduction,
    #[serde(default)]
    pub routed_output_norm: Option<String>,
    pub combine: Combine,
}

/// Why a layer cannot be executed as declared.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LayerDefect {
    #[error("layer {layer}: bank '{storage}' names unknown programme '{programme}'")]
    UnknownProgramme {
        layer: u32,
        storage: String,
        programme: String,
    },

    #[error(
        "layer {layer}: {which} bank runs {programme}, which operates in a latent space, \
         but the layer declares no routed_input/routed_output transforms"
    )]
    LatentBankWithoutTransforms {
        layer: u32,
        which: &'static str,
        programme: String,
    },

    #[error(
        "layer {layer}: input_space is '{declared}' but the routed bank's programme \
         '{programme}' operates in the other space"
    )]
    InputSpaceContradictsProgramme {
        layer: u32,
        declared: String,
        programme: String,
    },

    #[error("layer {layer}: router scores tensor name is empty")]
    RouterHasNoScores { layer: u32 },

    #[error(
        "layer {layer}: router selects {selected} experts but the routed bank declares only \
         {available}"
    )]
    SelectionExceedsBank {
        layer: u32,
        selected: usize,
        available: u32,
    },

    #[error(
        "layer {layer}: router declares shared_expert_sink but the layer has no shared bank \
         for the router to score"
    )]
    SinkWithoutSharedBank { layer: u32 },

    #[error("layer {layer}: {which} bank has an empty storage reference")]
    BankHasNoStorage { layer: u32, which: &'static str },

    #[error("layer {layer}: routed and shared banks both name storage '{storage}'")]
    DuplicateBankStorage { layer: u32, storage: String },

    #[error("layer {layer}: {which} bank declares zero experts")]
    BankHasNoExperts { layer: u32, which: &'static str },

    #[error(
        "layer {layer}: {which} bank declares a zero dimension \
         (input {input}, intermediate {intermediate}, output {output})"
    )]
    BankHasZeroDimension {
        layer: u32,
        which: &'static str,
        input: u32,
        intermediate: u32,
        output: u32,
    },
}

impl MoeLayer {
    /// Every way this layer is internally inconsistent.
    ///
    /// These are checks the *manifest* can fail on its own, before any weight
    /// byte is read — distinct from operand-absence, which needs the LYRW
    /// files and lives in the capability check (§11).
    ///
    /// **Suppression is dependency-scoped, not blanket.** An unresolvable
    /// programme suppresses only the checks whose *interpretation* depends on
    /// knowing the programme — currently the input-space compatibility test.
    /// Storage-shape defects are independent of programme identity and always
    /// run, so a typo'd programme name can never conceal a duplicate storage
    /// reference or a zero-width bank sitting behind it.
    pub fn defects(&self) -> Vec<LayerDefect> {
        let mut out = Vec::new();
        self.check_programmes_resolve(&mut out);
        self.check_latent_consistency(&mut out);
        self.check_router(&mut out);
        self.check_bank_storage(&mut out);
        out
    }

    pub fn is_well_formed(&self) -> bool {
        self.defects().is_empty()
    }

    fn check_programmes_resolve(&self, out: &mut Vec<LayerDefect>) {
        for bank in self.banks() {
            if bank.resolve_programme().is_none() {
                out.push(LayerDefect::UnknownProgramme {
                    layer: self.layer,
                    storage: bank.storage.clone(),
                    programme: bank.programme.clone(),
                });
            }
        }
    }

    fn check_latent_consistency(&self, out: &mut Vec<LayerDefect>) {
        if self.transforms.is_none() {
            for (which, bank) in self.labelled_banks() {
                if bank.needs_latent_transforms() {
                    out.push(LayerDefect::LatentBankWithoutTransforms {
                        layer: self.layer,
                        which,
                        programme: bank.programme.clone(),
                    });
                }
            }
        }

        let routed_is_latent = self.routed_bank.needs_latent_transforms();
        let declared_latent = self.input_space == InputSpace::Latent;
        // Only contradicts when the programme resolved; an unknown programme
        // is already reported and must not produce a second, misleading defect.
        if self.routed_bank.resolve_programme().is_some() && routed_is_latent != declared_latent {
            out.push(LayerDefect::InputSpaceContradictsProgramme {
                layer: self.layer,
                declared: format!("{:?}", self.input_space).to_lowercase(),
                programme: self.routed_bank.programme.clone(),
            });
        }
    }

    fn check_router(&self, out: &mut Vec<LayerDefect>) {
        if self.router.scores.trim().is_empty() {
            out.push(LayerDefect::RouterHasNoScores { layer: self.layer });
        }

        let selected = self.router.selection.experts_per_token();
        if selected > self.routed_bank.experts as usize {
            out.push(LayerDefect::SelectionExceedsBank {
                layer: self.layer,
                selected,
                available: self.routed_bank.experts,
            });
        }

        if self.router.shared_expert_sink && self.shared_bank.is_none() {
            out.push(LayerDefect::SinkWithoutSharedBank { layer: self.layer });
        }
    }

    /// Storage-shape checks. Independent of programme identity by design —
    /// see the suppression note on [`MoeLayer::defects`].
    fn check_bank_storage(&self, out: &mut Vec<LayerDefect>) {
        for (which, bank) in self.labelled_banks() {
            if bank.storage.trim().is_empty() {
                out.push(LayerDefect::BankHasNoStorage {
                    layer: self.layer,
                    which,
                });
            }
            if bank.experts == 0 {
                out.push(LayerDefect::BankHasNoExperts {
                    layer: self.layer,
                    which,
                });
            }
            if let Some(d) = bank.expert_dims {
                if d.input == 0 || d.intermediate == 0 || d.output == 0 {
                    out.push(LayerDefect::BankHasZeroDimension {
                        layer: self.layer,
                        which,
                        input: d.input,
                        intermediate: d.intermediate,
                        output: d.output,
                    });
                }
            }
        }

        if let Some(shared) = &self.shared_bank {
            if !shared.storage.trim().is_empty() && shared.storage == self.routed_bank.storage {
                out.push(LayerDefect::DuplicateBankStorage {
                    layer: self.layer,
                    storage: shared.storage.clone(),
                });
            }
        }
    }

    /// Every bank in the layer, routed first.
    pub fn banks(&self) -> Vec<&BankRef> {
        self.labelled_banks().into_iter().map(|(_, b)| b).collect()
    }

    fn labelled_banks(&self) -> Vec<(&'static str, &BankRef)> {
        let mut v = vec![("routed", &self.routed_bank)];
        if let Some(shared) = &self.shared_bank {
            v.push(("shared", shared));
        }
        v
    }

    /// Total walkable features this layer contributes (§15.1), or `None` if
    /// any bank left its dimensions undeclared.
    pub fn walkable_feature_count(&self) -> Option<u64> {
        self.banks()
            .iter()
            .map(|b| b.walkable_feature_count())
            .try_fold(0u64, |acc, n| Some(acc + n?))
    }
}
