//! Fixture A — a tiny direct routed-MoE layer, in both storage arrangements.
//!
//! Small enough that a wrong answer is inspectable by hand, and shaped so that
//! **no two axes share a size**: hidden 4, intermediate 3, population 5, top-k
//! 2. Equal dimensions hide transposes, and a transposed operand that still
//! multiplies is the failure this fixture most needs to catch — an earlier
//! draft used a population of 4 and could not have detected a transposed
//! router at all.
//!
//! The same weights are laid out twice — `gate`/`up` as separate regions, and
//! as one concatenated `gate_up_fused` region. Both must produce identical
//! output. That is the property that says the executor follows its bound
//! recipe rather than assuming a physical arrangement.

use crate::format::capability::binding::RepresentationIdentity;
use crate::format::capability::component::ComponentContract;
use crate::format::capability::coordinate::BankCoordinate;
use crate::format::lyrw2::region_format::RegionFormat;
use crate::format::lyrw2::region_role::RegionRole;
use larql_compute::{Activation, MoeTopKWeightPolicy};

use super::super::bank::{BoundBankOperation, BoundExpert};
use super::super::consts::FUSED_PROJECTION_HALVES;
use super::super::expert_kernel::ExpertKernel;
use super::super::operation::BoundMoeOperation;
use super::super::projection::{BoundProjection, ProjectionArrangement};
use super::super::reduction::BoundReduction;
use super::super::router::{BoundExpertScaling, BoundRouter, RouterKernel};
use super::super::tensor::BoundTensor;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

pub const HIDDEN: usize = 4;
pub const INTERMEDIATE: usize = 3;
pub const POPULATION: usize = 5;
pub const TOP_K: usize = 2;
pub const ACTIVATION: Activation = Activation::Silu;
pub const LAYER: u32 = 0;
pub const BANK_ID: u16 = 0;

/// Catalogue variant these operands claim to come from.
const VARIANT: &str = "fixture-a";

/// The router is not a bank region and so has no registry role; it is a
/// manifest-addressed tensor and names itself.
const ROUTER_REGION_SET: &str = "router";

// ── Weight values ──────────────────────────────────────────────────────────
//
// Deterministic and shared with the oracle. Sharing the *data* is fine — the
// oracle's independence is about arithmetic, not about inventing its own
// numbers. Coprime multipliers keep the values from repeating across axes, so
// a swapped index changes the answer.

pub fn gate_at(expert: usize, unit: usize, h: usize) -> f32 {
    0.1 * ((expert * 7 + unit * 3 + h) % 11) as f32 - 0.5
}

pub fn up_at(expert: usize, unit: usize, h: usize) -> f32 {
    0.1 * ((expert * 5 + unit * 2 + h * 3) % 9) as f32 - 0.4
}

pub fn down_at(expert: usize, h: usize, unit: usize) -> f32 {
    0.1 * ((expert * 3 + h * 5 + unit * 2) % 7) as f32 - 0.3
}

pub fn router_at(expert: usize, h: usize) -> f32 {
    0.1 * ((expert * 2 + h * 4) % 13) as f32 - 0.6
}

/// A residual that selects a non-trivial pair of experts.
pub fn input() -> Vec<f32> {
    vec![0.35, -0.72, 0.18, 0.94]
}

fn bytes(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values.flat_map(f32::to_le_bytes).collect()
}

/// Fixture A of the conformance programme: one direct routed-MoE layer.
///
/// Owns the bytes so bound tensors can borrow them.
pub struct DirectMoeFixture {
    gate: Vec<Vec<u8>>,
    up: Vec<Vec<u8>>,
    gate_up: Vec<Vec<u8>>,
    down: Vec<Vec<u8>>,
    router: Vec<u8>,
}

impl Default for DirectMoeFixture {
    fn default() -> Self {
        Self::new()
    }
}

impl DirectMoeFixture {
    pub fn new() -> Self {
        let experts = 0..POPULATION;
        Self {
            gate: experts
                .clone()
                .map(|e| {
                    bytes(
                        (0..INTERMEDIATE)
                            .flat_map(move |i| (0..HIDDEN).map(move |h| gate_at(e, i, h))),
                    )
                })
                .collect(),
            up: experts
                .clone()
                .map(|e| {
                    bytes(
                        (0..INTERMEDIATE)
                            .flat_map(move |i| (0..HIDDEN).map(move |h| up_at(e, i, h))),
                    )
                })
                .collect(),
            // Concatenated, not interleaved: every gate row, then every up row.
            gate_up: experts
                .clone()
                .map(|e| {
                    let gate = (0..INTERMEDIATE)
                        .flat_map(move |i| (0..HIDDEN).map(move |h| gate_at(e, i, h)));
                    let up = (0..INTERMEDIATE)
                        .flat_map(move |i| (0..HIDDEN).map(move |h| up_at(e, i, h)));
                    bytes(gate.chain(up))
                })
                .collect(),
            down: experts
                .map(|e| {
                    bytes(
                        (0..HIDDEN)
                            .flat_map(move |h| (0..INTERMEDIATE).map(move |i| down_at(e, h, i))),
                    )
                })
                .collect(),
            router: bytes((0..POPULATION).flat_map(|e| (0..HIDDEN).map(move |h| router_at(e, h)))),
        }
    }

    /// Build the bound operation for one storage arrangement.
    pub fn operation(&self, arrangement: ProjectionArrangement) -> BoundMoeOperation<'_> {
        let experts = (0..POPULATION)
            .map(|e| BoundExpert {
                expert_id: e as u32,
                projection: self.projection(e, arrangement),
                down: region(
                    RegionRole::Down,
                    &self.down[e],
                    ComponentContract::matrix(HIDDEN as u32, INTERMEDIATE as u32),
                ),
            })
            .collect();

        BoundMoeOperation {
            router: BoundRouter {
                weight: tensor(
                    ROUTER_REGION_SET,
                    &self.router,
                    ComponentContract::matrix(POPULATION as u32, HIDDEN as u32),
                ),
                top_k: TOP_K,
                selected_weight: MoeTopKWeightPolicy::RenormalizedSoftmax,
                scaling: BoundExpertScaling::None,
                kernel: RouterKernel::default(),
            },
            transforms: Vec::new(),
            banks: vec![BoundBankOperation {
                bank: BankCoordinate::new(LAYER, BANK_ID),
                experts,
                intermediate_dim: INTERMEDIATE,
                hidden_dim: HIDDEN,
                activation: ACTIVATION,
                kernel: ExpertKernel::default(),
            }],
            reduction: BoundReduction::WeightedSum,
            residual_dim: HIDDEN,
        }
    }

    fn projection(&self, expert: usize, arrangement: ProjectionArrangement) -> BoundProjection<'_> {
        let rows = ComponentContract::matrix(INTERMEDIATE as u32, HIDDEN as u32);
        match arrangement {
            ProjectionArrangement::Decomposed => BoundProjection::Decomposed {
                gate: region(RegionRole::Gate, &self.gate[expert], rows.clone()),
                up: region(RegionRole::Up, &self.up[expert], rows),
            },
            ProjectionArrangement::Fused => BoundProjection::Fused {
                gate_up: region(
                    RegionRole::GateUpFused,
                    &self.gate_up[expert],
                    ComponentContract::matrix(
                        (INTERMEDIATE * FUSED_PROJECTION_HALVES) as u32,
                        HIDDEN as u32,
                    ),
                ),
            },
        }
    }
}

/// Bind a bank region, named by its registry role.
fn region(role: RegionRole, bytes: &[u8], contract: ComponentContract) -> BoundTensor<'_> {
    tensor(&role.name(), bytes, contract)
}

fn tensor<'a>(region_set: &str, bytes: &'a [u8], contract: ComponentContract) -> BoundTensor<'a> {
    BoundTensor::direct(
        RepresentationIdentity::new(region_set, VARIANT),
        bytes,
        RegionFormat::F32,
        contract,
    )
    .expect("fixture A operands are well-formed")
}
