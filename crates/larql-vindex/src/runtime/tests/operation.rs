//! Colocated tests for `operation` — shape agreement across bound operands.
//!
//! `validate` is the load-path gate. Everything it catches would otherwise
//! surface mid-token as an indexing failure, or not at all.

use crate::format::capability::coordinate::BankCoordinate;
use crate::format::lyrw2::region_role::RegionRole;
use larql_compute::{Activation, MoeTopKWeightPolicy};

use super::support::ascending;
use crate::runtime::bank::{BoundBankOperation, BoundExpert};
use crate::runtime::error::ExecutionError;
use crate::runtime::expert_kernel::ExpertKernel;
use crate::runtime::operation::BoundMoeOperation;
use crate::runtime::projection::BoundProjection;
use crate::runtime::reduction::BoundReduction;
use crate::runtime::router::{BoundExpertScaling, BoundRouter, RouterKernel};
use crate::runtime::transform::{BoundTransform, TransformStage};

const RESIDUAL: u32 = 6;
const LATENT: u32 = 3;
const INTERMEDIATE: u32 = 2;
const POPULATION: u32 = 4;
const TOP_K: usize = 2;
const ROUTER: &str = "router";
const LATENT_IN: &str = "latent_in";
const LATENT_OUT: &str = "latent_out";

fn bank(hidden: u32) -> BoundBankOperation<'static> {
    BoundBankOperation {
        bank: BankCoordinate::new(0, 0),
        experts: (0..POPULATION)
            .map(|expert_id| BoundExpert {
                expert_id,
                projection: BoundProjection::Decomposed {
                    gate: ascending(&RegionRole::Gate.name(), INTERMEDIATE, hidden),
                    up: ascending(&RegionRole::Up.name(), INTERMEDIATE, hidden),
                },
                down: ascending(&RegionRole::Down.name(), hidden, INTERMEDIATE),
            })
            .collect(),
        intermediate_dim: INTERMEDIATE as usize,
        hidden_dim: hidden as usize,
        activation: Activation::Silu,
        kernel: ExpertKernel::default(),
    }
}

fn router(hidden: u32) -> BoundRouter<'static> {
    BoundRouter {
        weight: ascending(ROUTER, POPULATION, hidden),
        top_k: TOP_K,
        selected_weight: MoeTopKWeightPolicy::RenormalizedSoftmax,
        scaling: BoundExpertScaling::None,
        kernel: RouterKernel::default(),
    }
}

/// A direct operation: the bank reads the residual width.
pub fn direct() -> BoundMoeOperation<'static> {
    BoundMoeOperation {
        router: router(RESIDUAL),
        transforms: Vec::new(),
        banks: vec![bank(RESIDUAL)],
        reduction: BoundReduction::WeightedSum,
        residual_dim: RESIDUAL as usize,
    }
}

/// A latent operation: the bank reads a narrower routed width.
pub fn latent() -> BoundMoeOperation<'static> {
    BoundMoeOperation {
        router: router(LATENT),
        transforms: vec![
            BoundTransform {
                stage: TransformStage::RoutedInput,
                weight: ascending(LATENT_IN, LATENT, RESIDUAL),
            },
            BoundTransform {
                stage: TransformStage::RoutedOutput,
                weight: ascending(LATENT_OUT, RESIDUAL, LATENT),
            },
        ],
        banks: vec![bank(LATENT)],
        reduction: BoundReduction::WeightedSum,
        residual_dim: RESIDUAL as usize,
    }
}

// ── Bank input width ───────────────────────────────────────────────────────

#[test]
fn a_direct_operation_runs_its_banks_at_the_residual_width() {
    let op = direct();
    assert_eq!(op.bank_input_dim(), RESIDUAL as usize);
    assert!(!op.is_latent());
}

#[test]
fn a_latent_operation_runs_its_banks_at_the_projected_width() {
    let op = latent();
    assert_eq!(op.bank_input_dim(), LATENT as usize);
    assert!(op.is_latent());
    assert_eq!(op.residual_dim, RESIDUAL as usize, "residual is unchanged");
}

#[test]
fn the_last_routed_input_transform_fixes_the_bank_width() {
    // A chained input stage: the width the banks see is the final projection's
    // output, not the first one's.
    let mut op = latent();
    op.transforms.insert(
        0,
        BoundTransform {
            stage: TransformStage::RoutedInput,
            weight: ascending(LATENT_IN, RESIDUAL, RESIDUAL),
        },
    );
    assert_eq!(op.bank_input_dim(), LATENT as usize);
}

// ── Validation ─────────────────────────────────────────────────────────────

#[test]
fn a_consistent_direct_operation_validates() {
    direct().validate().unwrap();
}

#[test]
fn a_consistent_latent_operation_validates() {
    latent().validate().unwrap();
}

#[test]
fn a_bank_that_expects_the_residual_width_behind_a_latent_transform_is_refused() {
    // The mistake a direct-MoE assumption produces: experts sized for the
    // residual, fed the projected vector.
    let mut op = latent();
    op.banks = vec![bank(RESIDUAL)];
    let err = op.validate().unwrap_err();
    let ExecutionError::DimensionMismatch {
        expected, found, ..
    } = err
    else {
        panic!("expected a dimension mismatch, got {err}");
    };
    assert_eq!((expected, found), (LATENT as usize, RESIDUAL as usize));
}

#[test]
fn a_bank_holding_only_part_of_the_population_is_valid() {
    // Sharding. The router addresses the whole population; this bank carries a
    // subset of it. An earlier `validate` demanded equality and would have
    // refused every expert-server slice.
    let mut op = direct();
    op.banks[0].experts.truncate(2);
    op.banks[0].experts[0].expert_id = 2;
    op.banks[0].experts[1].expert_id = 3;
    op.validate().expect("a shard is a legitimate operation");
    assert!(!op.holds_full_population());
}

#[test]
fn a_bank_holding_everything_reports_full_population() {
    let op = direct();
    op.validate().unwrap();
    assert!(op.holds_full_population());
}

#[test]
fn an_expert_the_router_cannot_address_is_refused() {
    // The genuine fault the equality check was reaching for: an expert with an
    // id past the router's rows can never be selected, so its presence means
    // the router and the bank disagree about which population this is.
    let mut op = direct();
    op.banks[0].experts[0].expert_id = POPULATION + 5;
    let err = op.validate().unwrap_err();
    assert!(matches!(err, ExecutionError::ExpertOutOfRange { .. }));
}

#[test]
fn a_router_addressing_more_experts_than_are_bound_is_valid() {
    // Same shape as sharding, stated from the router's side.
    let mut op = direct();
    op.router = BoundRouter {
        weight: ascending(ROUTER, POPULATION + 4, RESIDUAL),
        ..op.router
    };
    op.validate().unwrap();
    assert!(!op.holds_full_population());
}

#[test]
fn a_router_scoring_at_the_wrong_width_is_refused() {
    let mut op = direct();
    op.router = BoundRouter {
        weight: ascending(ROUTER, POPULATION, RESIDUAL + 1),
        ..op.router
    };
    assert!(op.validate().is_err());
}

#[test]
fn an_operation_with_no_banks_validates_vacuously() {
    // Nothing to disagree about. Whether an empty operation is *useful* is a
    // planning question, not a shape one.
    let mut op = direct();
    op.banks.clear();
    op.validate().unwrap();
}

// ── Description ────────────────────────────────────────────────────────────

#[test]
fn a_direct_operation_describes_itself_as_direct() {
    let text = direct().describe();
    assert!(text.contains("direct"), "{text}");
    assert!(text.contains(BoundReduction::WeightedSum.name()), "{text}");
    assert!(text.contains(&format!("top-{TOP_K}")), "{text}");
}

#[test]
fn a_latent_operation_describes_itself_as_latent() {
    assert!(latent().describe().contains("latent"));
}
