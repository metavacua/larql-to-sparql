//! A parameterised synthetic MoE layer, laid out as it would be on disk.
//!
//! Fixture A is fixed-size and hand-checkable. This is its scaling companion:
//! same semantics, any shape, and — crucially — a **contiguous byte image**
//! rather than a scatter of separate allocations.
//!
//! The byte image is what makes one builder serve three very different
//! consumers:
//!
//! ```text
//! benches/vindex3_bound_execute     bind over a Vec       cost vs population
//! tests/vindex3_allocation_guard    bind over a Vec       allocations per token
//! examples/vindex3_residency_probe  bind over an mmap     pages actually faulted
//! ```
//!
//! `bind` takes the bytes as an argument rather than reading its own, so the
//! same layout can be bound over heap memory or a file mapping with identical
//! content. The residency probe depends on that: it must measure the layout
//! the other two measure, not a lookalike.
//!
//! # Layout
//!
//! ```text
//! [router] [expert 0: gate_up | down] [expert 1: gate_up | down] ...
//! ```
//!
//! Each expert's regions are adjacent on purpose. Expert-major locality is the
//! property a routed read pattern depends on, and interleaving roles across
//! experts instead would scatter every selected expert across the whole file.

use core::ops::Range;

use crate::format::capability::binding::RepresentationIdentity;
use crate::format::capability::component::ComponentContract;
use crate::format::capability::coordinate::BankCoordinate;
use crate::format::lyrw2::region_format::RegionFormat;
use crate::format::lyrw2::region_role::RegionRole;
use larql_compute::{Activation, MoeTopKWeightPolicy};

use super::super::bank::{BoundBankOperation, BoundExpert};
use super::super::consts::{F32_BYTES, FUSED_PROJECTION_HALVES};
use super::super::expert_kernel::ExpertKernel;
use super::super::operation::BoundMoeOperation;
use super::super::projection::BoundProjection;
use super::super::reduction::BoundReduction;
use super::super::router::{BoundExpertScaling, BoundRouter, RouterKernel};
use super::super::tensor::BoundTensor;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Catalogue variant these operands claim to come from.
const VARIANT: &str = "synthetic";
/// The router is manifest-addressed, not a bank region, so it has no role.
const ROUTER_REGION_SET: &str = "router";
/// Modulus for the weight generator — prime, so values do not repeat on any
/// axis of a realistic shape.
const WEIGHT_MODULUS: usize = 199;
/// Seeds keeping the three region classes numerically distinct.
const SEED_GATE: usize = 0;
const SEED_UP: usize = 1_000;
const SEED_DOWN: usize = 2_000;
const SEED_ROUTER: usize = 3_000;
const SEED_RESIDUAL: usize = 4_000;

/// Deterministic pseudo-weights. Content is irrelevant to timing and
/// residency; determinism keeps runs comparable.
pub fn weight(seed: usize, index: usize) -> f32 {
    (((seed * 31 + index * 17) % WEIGHT_MODULUS) as f32) / WEIGHT_MODULUS as f32 - 0.5
}

/// The dimensions of a synthetic layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyntheticShape {
    pub population: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub top_k: usize,
}

impl SyntheticShape {
    /// Elements in one expert's fused gate+up region.
    pub fn projection_elements(&self) -> usize {
        FUSED_PROJECTION_HALVES * self.intermediate * self.hidden
    }

    /// Elements in one expert's down region.
    pub fn down_elements(&self) -> usize {
        self.hidden * self.intermediate
    }

    /// Bytes one expert occupies, both regions together.
    pub fn expert_bytes(&self) -> usize {
        (self.projection_elements() + self.down_elements()) * F32_BYTES
    }

    pub fn router_bytes(&self) -> usize {
        self.population * self.hidden * F32_BYTES
    }

    /// Total bytes of the layer's byte image.
    pub fn total_bytes(&self) -> usize {
        self.router_bytes() + self.population * self.expert_bytes()
    }

    /// Bytes a single token's routed read touches: the router, plus `top_k`
    /// experts. The residency question in one number.
    pub fn selected_bytes(&self) -> usize {
        self.router_bytes() + self.top_k.min(self.population) * self.expert_bytes()
    }
}

/// Where each region sits in the byte image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertExtents {
    pub gate_up: Range<usize>,
    pub down: Range<usize>,
}

/// A synthetic layer: its byte image and the extents that address it.
#[derive(Debug, Clone)]
pub struct SyntheticLayer {
    pub shape: SyntheticShape,
    pub bytes: Vec<u8>,
    pub router: Range<usize>,
    pub experts: Vec<ExpertExtents>,
}

impl SyntheticLayer {
    /// Build the byte image for `shape`.
    pub fn build(shape: SyntheticShape) -> Self {
        let mut bytes = Vec::with_capacity(shape.total_bytes());

        let router_start = bytes.len();
        extend(&mut bytes, shape.population * shape.hidden, SEED_ROUTER);
        let router = router_start..bytes.len();

        let experts = (0..shape.population)
            .map(|e| {
                let gate_up_start = bytes.len();
                // Gate rows first, then up rows — concatenated, not
                // interleaved, matching what the reader expects.
                extend(&mut bytes, shape.intermediate * shape.hidden, SEED_GATE + e);
                extend(&mut bytes, shape.intermediate * shape.hidden, SEED_UP + e);
                let gate_up = gate_up_start..bytes.len();

                let down_start = bytes.len();
                extend(&mut bytes, shape.down_elements(), SEED_DOWN + e);
                ExpertExtents {
                    gate_up,
                    down: down_start..bytes.len(),
                }
            })
            .collect();

        Self {
            shape,
            bytes,
            router,
            experts,
        }
    }

    /// A residual of the right width for this layer.
    pub fn residual(&self) -> Vec<f32> {
        (0..self.shape.hidden)
            .map(|i| weight(SEED_RESIDUAL, i))
            .collect()
    }

    /// Bind an operation over `image`, which must hold this layer's bytes.
    ///
    /// Separate from `self.bytes` so the same layout can be bound over a file
    /// mapping. The residency probe relies on measuring exactly the layout the
    /// benches measure.
    pub fn bind<'a>(&self, image: &'a [u8]) -> BoundMoeOperation<'a> {
        let shape = self.shape;
        let experts = self
            .experts
            .iter()
            .enumerate()
            .map(|(e, extents)| BoundExpert {
                expert_id: e as u32,
                projection: BoundProjection::Fused {
                    gate_up: tensor(
                        &RegionRole::GateUpFused.name(),
                        &image[extents.gate_up.clone()],
                        FUSED_PROJECTION_HALVES * shape.intermediate,
                        shape.hidden,
                    ),
                },
                down: tensor(
                    &RegionRole::Down.name(),
                    &image[extents.down.clone()],
                    shape.hidden,
                    shape.intermediate,
                ),
            })
            .collect();

        BoundMoeOperation {
            router: BoundRouter {
                weight: tensor(
                    ROUTER_REGION_SET,
                    &image[self.router.clone()],
                    shape.population,
                    shape.hidden,
                ),
                top_k: shape.top_k,
                selected_weight: MoeTopKWeightPolicy::RenormalizedSoftmax,
                scaling: BoundExpertScaling::None,
                kernel: RouterKernel::default(),
            },
            transforms: Vec::new(),
            banks: vec![BoundBankOperation {
                bank: BankCoordinate::new(0, 0),
                experts,
                intermediate_dim: shape.intermediate,
                hidden_dim: shape.hidden,
                activation: Activation::Silu,
                kernel: ExpertKernel::default(),
            }],
            reduction: BoundReduction::WeightedSum,
            residual_dim: shape.hidden,
        }
    }
}

fn extend(out: &mut Vec<u8>, count: usize, seed: usize) {
    out.extend((0..count).flat_map(|i| weight(seed, i).to_le_bytes()));
}

fn tensor<'a>(region_set: &str, data: &'a [u8], rows: usize, cols: usize) -> BoundTensor<'a> {
    BoundTensor::direct(
        RepresentationIdentity::new(region_set, VARIANT),
        data,
        RegionFormat::F32,
        ComponentContract::matrix(rows as u32, cols as u32),
    )
    .expect("synthetic operands are well-formed")
}
