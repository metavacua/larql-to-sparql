//! Unit cover for the split-scale binding helpers.
//!
//! The numerical question — does the row walk actually decode the right rows
//! — is a GPU-level one and lives with the layout parity tests. What is
//! checked here is the addressing contract those dispatches depend on: one
//! base buffer, one offset per selected expert, and a loud refusal when a
//! split-scale bank arrives without the stream that makes it decodable.

use super::*;
use crate::moe_zero_copy::ResolvedExpert;
use crate::MetalBackend;

/// A resolved expert carrying `scales` as its gate/up exponent binding.
fn expert(
    backend: &MetalBackend,
    expert_id: usize,
    payload: &[u8],
    scales: Option<(Buffer, u64)>,
) -> ResolvedExpert {
    let buf = backend.bufs().get_bytes(payload);
    ResolvedExpert {
        gate_up: (buf.clone(), 0),
        down: (buf, 0),
        gate_up_scales: scales.clone(),
        down_scales: scales,
        expert_id,
        weight: 1.0,
    }
}

fn backend() -> MetalBackend {
    MetalBackend::new().expect("Metal backend required")
}

#[test]
fn scale_offsets_are_read_per_expert_not_derived_from_the_payload() {
    let b = backend();
    let scale_buf = b.bufs().get_bytes(&[7u8; 512]);
    // Offsets deliberately NOT payload/16: the whole point of the second
    // table is that the exponent stream sits where the writer put it.
    let resolved = vec![
        expert(&b, 0, &[1u8; 256], Some((scale_buf.clone(), 40))),
        expert(&b, 1, &[2u8; 256], Some((scale_buf, 104))),
    ];
    let offsets = MetalBackend::scale_offsets(&resolved, |r| r.gate_up_scales.as_ref());
    assert_eq!(offsets, vec![40, 104]);
}

#[test]
fn scale_base_accepts_one_shared_buffer() {
    let b = backend();
    let scale_buf = b.bufs().get_bytes(&[7u8; 512]);
    let resolved = vec![
        expert(&b, 0, &[1u8; 256], Some((scale_buf.clone(), 0))),
        expert(&b, 1, &[2u8; 256], Some((scale_buf.clone(), 64))),
    ];
    let base = MetalBackend::scale_base(&resolved, |r| r.gate_up_scales.as_ref(), "gate_up");
    assert_eq!(base.gpu_address(), scale_buf.gpu_address());
}

#[test]
#[should_panic(expected = "has no gate_up exponent stream")]
fn a_split_scale_bank_without_exponents_refuses() {
    let b = backend();
    // `None` here is the state the resolution step exists to prevent: an
    // MXFP4 bank whose scales never resolved. Reaching the encoder with it
    // must not produce a dispatch that decodes against whatever buffer
    // happens to be bound at slot 2.
    let resolved = vec![expert(&b, 3, &[1u8; 256], None)];
    let _ = MetalBackend::scale_base(&resolved, |r| r.gate_up_scales.as_ref(), "gate_up");
}

#[test]
#[should_panic(expected = "span more than one buffer")]
fn exponents_in_two_buffers_refuse() {
    let b = backend();
    let first = b.bufs().get_bytes(&[7u8; 512]);
    let second = b.bufs().get_bytes(&[9u8; 512]);
    assert_ne!(
        first.gpu_address(),
        second.gpu_address(),
        "fixture needs two genuinely distinct buffers to test the refusal"
    );
    let resolved = vec![
        expert(&b, 0, &[1u8; 256], Some((first, 0))),
        expert(&b, 1, &[2u8; 256], Some((second, 0))),
    ];
    let _ = MetalBackend::scale_base(&resolved, |r| r.gate_up_scales.as_ref(), "gate_up");
}

#[test]
fn the_down_stream_is_checked_independently_of_gate_up() {
    let b = backend();
    let scale_buf = b.bufs().get_bytes(&[7u8; 512]);
    let mut r = expert(&b, 0, &[1u8; 256], Some((scale_buf.clone(), 8)));
    // Gate/up present, down absent — the two are separate regions bound by
    // separate `pair_id`s, so one being reachable says nothing about the
    // other.
    r.down_scales = None;
    let resolved = vec![r];
    assert_eq!(
        MetalBackend::scale_base(&resolved, |r| r.gate_up_scales.as_ref(), "gate_up").gpu_address(),
        scale_buf.gpu_address()
    );
    let refused = std::panic::catch_unwind(|| {
        MetalBackend::scale_base(&resolved, |r| r.down_scales.as_ref(), "down")
    });
    assert!(refused.is_err(), "a missing down stream must refuse too");
}

#[test]
fn the_identity_row_walk_is_what_a_non_fused_operand_gets() {
    use larql_compute::MoeFusedRowLayout;
    use larql_models::quant::mxfp4::FusedHalf;

    // The down projection binds the identity walk. Pin that it really is the
    // identity, and that it is NOT what either fused layout hands back for
    // the up half — otherwise the down dispatch could be reading gate rows
    // and this test would not notice.
    assert_eq!(
        (ROW_BASE_IDENTITY, ROW_STRIDE_IDENTITY),
        (0, 1),
        "down contracts over stored rows in order"
    );
    for layout in [
        MoeFusedRowLayout::ContiguousHalves,
        MoeFusedRowLayout::Interleaved,
    ] {
        let (base, stride) = layout.row_walk(FusedHalf::Up, 64);
        assert_ne!(
            (base as u32, stride as u32),
            (ROW_BASE_IDENTITY, ROW_STRIDE_IDENTITY),
            "{layout:?} up half must differ from the identity walk"
        );
    }
}
