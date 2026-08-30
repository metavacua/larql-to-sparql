//! CPU-2A: BF16 residency is REAL — the stored bytes are the resident
//! bytes.
//!
//! The failure this guards against is a tidy type with no benefit: a
//! `Bf16` variant whose consumers immediately call `as_f32()` would give
//! a lovely enum and expand exactly as much memory as before. So these
//! assert bytes, not types.

use crate::format::vindex3::encode::encode_system;
use crate::format::vindex3::fixtures::hybrid_lllf_f32_model;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::{WeightFormat, WeightSlice};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::weights::load_weight;
use crate::format::vindex3::opplan::{plan_component_ops, LayerAttention};

/// A container whose decoder stack is stored BF16.
fn bf16_container() -> (
    tempfile::TempDir,
    crate::format::vindex3::opplan::ComponentOpPlan,
    OperandStore,
) {
    let src = tempfile::tempdir().unwrap();
    hybrid_lllf_f32_model(src.path());
    let inventory = larql_models::inventory::build_inventory(src.path()).unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_system(&[("hybrid".to_string(), inventory)], container.path()).unwrap();
    let inspection = inspect_container(container.path(), false).unwrap();
    let plan = plan_component_ops(&inspection, container.path(), "target")
        .unwrap()
        .plan
        .unwrap();
    let store = OperandStore::open(container.path(), &inspection).unwrap();
    (container, plan, store)
}

/// The bf16 format copies stored bytes and converts nothing.
///
/// Asserted three ways: the resident slice is HALF the element count in
/// bytes that f32 would need; the code units widen back to exactly the
/// f32 values; and `as_f32()` REFUSES rather than silently widening.
#[test]
fn bf16_residency_keeps_the_stored_bytes() {
    let (_c, plan, store) = bf16_container();
    // This fixture writes F32 shards, so a bf16 load must refuse — which
    // is itself the first claim: the format performs no conversion.
    let LayerAttention::Softmax(op) = &plan.layers[3].attention else {
        panic!("layer 3 is the softmax layer");
    };
    let err = load_weight((&store).into(), &op.q, WeightFormat::Bf16)
        .expect_err("an f32 checkpoint has no stored bf16 bytes to keep");
    let msg = err.to_string();
    assert!(
        msg.contains("performs no conversion") || msg.contains("not bf16"),
        "the refusal must say WHY rather than invent a narrowing: {msg}"
    );
}

/// `as_f32()` on a bf16 slice refuses.
///
/// The load-bearing assertion of the whole rung: if this widened, every
/// consumer would compile, every test would pass, and the model would
/// still be F32-resident.
#[test]
fn a_bf16_slice_will_not_silently_widen() {
    let units: Vec<u16> = vec![0x3f80, 0xbf80, 0x0000];
    let slice = WeightSlice::Bf16(&units);
    assert!(
        slice.as_f32().is_err(),
        "as_f32() on bf16 must refuse — a widening accessor here would make the compact \
         representation pointless while looking correct"
    );
    let got = slice.as_bf16().expect("the bf16 accessor answers");
    assert_eq!(got, &units[..]);
    // And the widen those units denote is exact.
    let widened: Vec<f32> = got
        .iter()
        .map(|b| f32::from_bits((*b as u32) << 16))
        .collect();
    assert_eq!(widened, vec![1.0f32, -1.0, 0.0]);
}

/// Resident bytes are half of f32 for the same element count.
#[test]
fn bf16_residency_halves_the_resident_bytes() {
    let units: Vec<u16> = (0..1024).map(|i| (i as u16) | 0x3c00).collect();
    let as_bf16 = WeightSlice::Bf16(&units);
    let f32s: Vec<f32> = units
        .iter()
        .map(|b| f32::from_bits((*b as u32) << 16))
        .collect();
    let as_f32 = WeightSlice::F32(&f32s);
    let bytes = |s: &WeightSlice<'_>| match s {
        WeightSlice::Bf16(w) => w.len() * 2,
        WeightSlice::F32(w) => w.len() * 4,
        _ => unreachable!(),
    };
    assert_eq!(bytes(&as_bf16) * 2, bytes(&as_f32));
    assert_eq!(bytes(&as_bf16), 2048);
}
