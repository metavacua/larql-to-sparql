//! Colocated tests for `bank` — expert lookup and shape validation.

use crate::format::capability::coordinate::BankCoordinate;
use crate::format::lyrw2::region_role::RegionRole;
use larql_compute::Activation;

use super::support::ascending;
use crate::runtime::bank::{BoundBankOperation, BoundExpert};
use crate::runtime::error::ExecutionError;
use crate::runtime::expert_kernel::ExpertKernel;
use crate::runtime::projection::BoundProjection;

const LAYER: u32 = 7;
const BANK_ID: u16 = 1;
const INTERMEDIATE: u32 = 2;
const HIDDEN: u32 = 3;
/// The router's addressable population these banks live inside.
const POPULATION: usize = 128;

fn expert(expert_id: u32) -> BoundExpert<'static> {
    BoundExpert {
        expert_id,
        projection: BoundProjection::Decomposed {
            gate: ascending(&RegionRole::Gate.name(), INTERMEDIATE, HIDDEN),
            up: ascending(&RegionRole::Up.name(), INTERMEDIATE, HIDDEN),
        },
        down: ascending(&RegionRole::Down.name(), HIDDEN, INTERMEDIATE),
    }
}

fn bank(ids: &[u32]) -> BoundBankOperation<'static> {
    BoundBankOperation {
        bank: BankCoordinate::new(LAYER, BANK_ID),
        experts: ids.iter().copied().map(expert).collect(),
        intermediate_dim: INTERMEDIATE as usize,
        hidden_dim: HIDDEN as usize,
        activation: Activation::Silu,
        kernel: ExpertKernel::default(),
    }
}

// ── Lookup ─────────────────────────────────────────────────────────────────

#[test]
fn an_expert_is_found_by_its_id_not_its_position() {
    // Sparse ids are the normal case for a sharded bank: expert 40 may be the
    // first one this shard holds.
    let bank = bank(&[40, 41, 42]);
    assert_eq!(bank.expert(41, POPULATION).unwrap().expert_id, 41);
    assert_eq!(bank.population(), 3);
}

#[test]
fn a_selected_expert_the_bank_does_not_hold_is_refused_not_skipped() {
    // Skipping would drop a fraction of the FFN contribution and produce a
    // token that looks entirely reasonable.
    let err = bank(&[0, 1]).expert(9, POPULATION).unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::SelectedExpertNotResident {
            expert: 9,
            resident: 2,
            ..
        }
    ));
    assert!(err.to_string().contains('9'));
}

#[test]
fn an_empty_bank_refuses_every_id() {
    assert_eq!(bank(&[]).population(), 0);
    assert!(bank(&[]).expert(0, POPULATION).is_err());
}

// ── Validation ─────────────────────────────────────────────────────────────

#[test]
fn a_consistent_bank_validates() {
    bank(&[0, 1, 2]).validate().unwrap();
}

#[test]
fn an_expert_whose_down_is_transposed_is_refused() {
    // `down` contracts the intermediate axis away, so it is [hidden,
    // intermediate]. The swap still multiplies when the two happen to be
    // equal, which is why the fixture keeps them different.
    let mut broken = bank(&[0]);
    broken.experts[0].down = ascending(&RegionRole::Down.name(), INTERMEDIATE, HIDDEN);
    assert!(broken.validate().is_err());
}

#[test]
fn a_bank_whose_declared_intermediate_disagrees_with_its_experts_is_refused() {
    let mut broken = bank(&[0]);
    broken.intermediate_dim = (INTERMEDIATE + 1) as usize;
    let err = broken.validate().unwrap_err();
    assert!(matches!(err, ExecutionError::DimensionMismatch { .. }));
}

#[test]
fn one_broken_expert_fails_the_whole_bank() {
    let mut broken = bank(&[0, 1, 2]);
    broken.experts[2].down = ascending(&RegionRole::Down.name(), HIDDEN, INTERMEDIATE + 1);
    assert!(broken.validate().is_err());
}

#[test]
fn an_expert_validates_against_the_shape_it_is_given() {
    expert(0)
        .validate(INTERMEDIATE as usize, HIDDEN as usize)
        .unwrap();
    assert!(expert(0).validate(INTERMEDIATE as usize, 99).is_err());
}

// ── Description ────────────────────────────────────────────────────────────

#[test]
fn a_bank_describes_its_coordinate_population_and_shape() {
    let text = bank(&[0, 1]).describe();
    assert!(
        text.contains(&BankCoordinate::new(LAYER, BANK_ID).describe()),
        "{text}"
    );
    assert!(text.contains('2'), "{text}");
    assert!(text.contains("2×3"), "{text}");
}
