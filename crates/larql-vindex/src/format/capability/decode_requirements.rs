//! What an architecture needs for a complete forward path (spec §4, §8.2).
//!
//! Adapter-owned, deliberately. VINDEX3 is explicitly not a general
//! neural-graph container (§2), so a model's dense spine — KDA recurrence
//! parameters, MLA latent KV, AttnRes, output gates — is *declared* here
//! rather than described in the MoE programme vocabulary.
//!
//! # Requirements do not restate the programme
//!
//! An adapter names the banks a layer uses and the non-bank tensors that
//! surround them. It never restates `gate/up/down` alternatives: those belong
//! to the programme registry, and duplicating them would mean a programme
//! change required synchronised edits to every adapter.
//!
//! ```text
//! adapter            which banks participate, and what surrounds them
//! programme registry which bank-region arrangements execute
//! local decode       joins the two
//! ```
//!
//! # A complete bank is not a decodable layer
//!
//! Programme traversal answers whether the *bank computation* runs. These
//! requirements answer whether the *whole layer path* does. Mini-K3 separates
//! them immediately: `gate/up/down` can be perfect while `routed_input` is
//! absent, and then WALK and decode fail for different reasons through
//! different dependency projections.
//!
//! # The dense schedule is a list, not a field
//!
//! No `first_k_dense_replace`, no `dense_mlp_idx`. An adapter emits per-layer
//! requirements, so Kimi-Linear's leading dense layer and Inkling-Small's
//! mid-stack one need no special cases — which is why the format chose
//! per-layer manifests in the first place.

use super::component::{ComponentContract, ComponentCoordinate};
use super::coordinate::BankCoordinate;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// What a required component is *for*, independent of where it is stored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentRequirement {
    /// Human-readable purpose, used in diagnostics: "final norm", "lm_head".
    pub purpose: &'static str,
    pub coordinate: ComponentCoordinate,
    /// Expected shape and kind, when the adapter pins one.
    pub contract: Option<ComponentContract>,
}

impl ComponentRequirement {
    pub fn new(purpose: &'static str, coordinate: ComponentCoordinate) -> Self {
        Self {
            purpose,
            coordinate,
            contract: None,
        }
    }

    pub fn with_contract(mut self, contract: ComponentContract) -> Self {
        self.contract = Some(contract);
        self
    }
}

/// One requirement, possibly satisfiable more than one way.
///
/// The fused/decomposed problem recurs outside expert regions. A tied LM head
/// is the obvious case: some models store a dedicated `lm_head`, others reuse
/// the embedding tensor. A flat list of mandatory tensors would reject a
/// perfectly valid tied-weight model, so requirements carry alternatives for
/// the same reason programmes do — and with the same rule: every satisfied
/// alternative is preserved, and declaration order decides only which failure
/// is reported when none succeed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Requirement {
    One(ComponentRequirement),
    AnyOf(Vec<ComponentRequirement>),
}

impl Requirement {
    pub fn one(purpose: &'static str, coordinate: ComponentCoordinate) -> Self {
        Self::One(ComponentRequirement::new(purpose, coordinate))
    }

    /// Candidates in declaration order.
    pub fn candidates(&self) -> &[ComponentRequirement] {
        match self {
            Self::One(r) => std::slice::from_ref(r),
            Self::AnyOf(rs) => rs,
        }
    }

    /// Purpose of the first candidate — what the requirement is *called*, even
    /// when several tensors could satisfy it.
    pub fn purpose(&self) -> &'static str {
        self.candidates()
            .first()
            .map(|r| r.purpose)
            .unwrap_or("unnamed requirement")
    }
}

/// The non-bank components surrounding one MoE layer.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MoeLayerRequirements {
    /// Router score weights, bias, and anything else the manifest names for
    /// selection. A complete expert bank without these is not a decodable
    /// layer.
    pub router: Vec<Requirement>,
    /// Residual↔latent projections, routed output norm, shared pre/post
    /// transforms.
    pub transforms: Vec<Requirement>,
    /// Attention / norms / recurrence for this layer.
    pub fixed_spine: Vec<Requirement>,
    /// Banks whose executable arrangements come from programme traversal.
    pub banks: Vec<BankCoordinate>,
}

/// What one layer needs, by kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerDecodeRequirements {
    Dense { components: Vec<Requirement> },
    Moe(MoeLayerRequirements),
}

impl LayerDecodeRequirements {
    /// Every non-bank requirement this layer imposes, in a stable order.
    pub fn fixed(&self) -> Vec<&Requirement> {
        match self {
            Self::Dense { components } => components.iter().collect(),
            Self::Moe(m) => m
                .fixed_spine
                .iter()
                .chain(&m.router)
                .chain(&m.transforms)
                .collect(),
        }
    }

    pub fn banks(&self) -> &[BankCoordinate] {
        match self {
            Self::Dense { .. } => &[],
            Self::Moe(m) => &m.banks,
        }
    }
}

/// The execution boundary being requested.
///
/// Deliberately one variant. Other boundaries — residual-to-logits, a layer
/// range — change which fixed components are required, so each deserves its
/// own explicit variant rather than a general request with optional fields
/// whose combinations become ambiguous.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalDecodeRequest {
    /// Token ids in, logits out.
    ///
    /// Needs embeddings, every active layer, the final norm, and an LM head or
    /// its tied equivalent. Does **not** need a tokenizer: the ids are already
    /// supplied, and requiring one would refuse a caller that never needed it.
    TokenIdsToLogits,
}

/// Everything an architecture needs for one decode boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeRequirements {
    /// Model-global components — embeddings, final norm, LM head.
    pub fixed_components: Vec<Requirement>,
    /// One entry per active layer, in order. Dense and MoE layers interleave
    /// freely; no global schedule field exists to get wrong.
    pub layers: Vec<LayerDecodeRequirements>,
}

impl DecodeRequirements {
    /// Every bank any layer requires.
    pub fn all_banks(&self) -> Vec<BankCoordinate> {
        let mut out: Vec<BankCoordinate> = self
            .layers
            .iter()
            .flat_map(|l| l.banks())
            .copied()
            .collect();
        out.sort();
        out.dedup();
        out
    }

    /// Layers that run a MoE programme.
    pub fn moe_layer_count(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(l, LayerDecodeRequirements::Moe(_)))
            .count()
    }

    pub fn dense_layer_count(&self) -> usize {
        self.layers.len() - self.moe_layer_count()
    }
}
