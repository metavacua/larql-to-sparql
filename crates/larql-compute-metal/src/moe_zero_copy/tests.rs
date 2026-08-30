//! Cover for what `resolve_selected_experts` does when resolution misses.
//!
//! The interesting behaviour is not the happy path — the layout parity test
//! drives that end to end — but the fork in what a miss *means*:
//!
//! - an inline-scale bank can be served by the staged dispatcher, so a miss
//!   is `None` and the caller drops back to it;
//! - a split-scale bank cannot be staged by anything in this backend, so a
//!   miss is a refusal here, at the cause, rather than a `None` that surfaces
//!   later as an assert about kernel binding arity and sends the reader to
//!   the wrong file.
//!
//! Both halves are asserted, because the whole point is that they differ.

use super::*;
use crate::MetalBackend;
use larql_compute::{
    Activation, MoeExpertScales, MoeFusedRowLayout, MoeGateRule, MoeLayerWeights, MoeRoutingPolicy,
    MoeWeightLayout, QuantFormat,
};

const HIDDEN: usize = 256;
const INTER: usize = 128;
const TOP_K: usize = 1;
const NUM_EXPERTS: usize = 1;

/// Bytes one expert's fused gate/up region must hold at these dimensions.
const GATE_UP_BYTES: usize = 2 * INTER * (HIDDEN / 32 * 16);
/// Bytes one expert's down region must hold.
const DOWN_BYTES: usize = HIDDEN * (INTER / 32 * 16);

fn backend() -> MetalBackend {
    MetalBackend::new().expect("Metal backend required")
}

fn scratch(metal: &MetalBackend) -> MoeScratch {
    MoeScratch::new_public_with_format(metal, TOP_K, HIDDEN, INTER, QuantFormat::MXFP4, HIDDEN)
}

/// A layer whose only interesting property is where it keeps its scales.
fn layer<'a>(scales: MoeExpertScales<'a>, payload: &'a [&'a [u8]]) -> MoeLayerWeights<'a> {
    MoeLayerWeights {
        experts_gate_up: payload.to_vec(),
        experts_down: payload.to_vec(),
        expert_scales: scales,
        fused_row_layout: MoeFusedRowLayout::Interleaved,
        routing_policy: MoeRoutingPolicy::gemma4_hybrid(),
        weight_layout: MoeWeightLayout::default(),
        expert_data_format: QuantFormat::MXFP4,
        router_proj: &[],
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: true,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts: NUM_EXPERTS,
        top_k: TOP_K,
        intermediate_size: INTER,
        router_bias: &[],
        experts_gate_up_bias: &[],
        experts_down_bias: &[],
        gate_rule: MoeGateRule::Gated(Activation::GeluTanh),
    }
}

fn paired<'a>(gate_up: &'a [u8], down: &'a [u8]) -> MoeExpertScales<'a> {
    MoeExpertScales::Paired {
        gate_up: vec![gate_up],
        down: vec![down],
    }
}

// ── An inline-scale bank: a miss is a fallback, not a failure ──

#[test]
fn an_inline_bank_with_no_bytes_falls_back_to_the_staged_path() {
    let metal = backend();
    let s = scratch(&metal);
    let moe = layer(MoeExpertScales::Inline, &[]);
    let got = metal.resolve_selected_experts(&s, &moe, &[0], &[1.0], |_| None);
    assert!(
        got.is_none(),
        "an inline bank that cannot be bound zero-copy has a staged path to \
         fall back to, so this must be None rather than a refusal"
    );
}

#[test]
fn an_inline_bank_with_short_slices_falls_back() {
    let metal = backend();
    let s = scratch(&metal);
    let moe = layer(MoeExpertScales::Inline, &[]);
    let short = vec![0u8; 16];
    let got = metal.resolve_selected_experts(&s, &moe, &[0], &[1.0], |_| {
        Some((short.as_slice(), short.as_slice()))
    });
    assert!(got.is_none());
}

#[test]
fn an_inline_bank_outside_a_registered_region_falls_back() {
    let metal = backend();
    let s = scratch(&metal);
    let moe = layer(MoeExpertScales::Inline, &[]);
    // Full extents, but never registered — resolvable in size, not in place.
    let gu = vec![0u8; GATE_UP_BYTES];
    let dn = vec![0u8; DOWN_BYTES];
    let got = metal.resolve_selected_experts(&s, &moe, &[0], &[1.0], |_| {
        Some((gu.as_slice(), dn.as_slice()))
    });
    assert!(got.is_none());
}

#[test]
fn no_selected_experts_is_not_a_binding() {
    let metal = backend();
    let s = scratch(&metal);
    let moe = layer(MoeExpertScales::Inline, &[]);
    let got = metal.resolve_selected_experts(&s, &moe, &[], &[], |_| None);
    assert!(got.is_none(), "an empty selection binds nothing");
}

// ── A split-scale bank: a miss is a refusal, at the cause ──

#[test]
#[should_panic(expected = "has no weight bytes")]
fn a_split_bank_with_no_bytes_refuses_rather_than_deferring() {
    let metal = backend();
    let s = scratch(&metal);
    let gu_s = vec![0u8; 64];
    let dn_s = vec![0u8; 64];
    let moe = layer(paired(&gu_s, &dn_s), &[]);
    let _ = metal.resolve_selected_experts(&s, &moe, &[0], &[1.0], |_| None);
}

#[test]
#[should_panic(expected = "weight slices are short")]
fn a_split_bank_with_short_slices_refuses() {
    let metal = backend();
    let s = scratch(&metal);
    let gu_s = vec![0u8; 64];
    let dn_s = vec![0u8; 64];
    let moe = layer(paired(&gu_s, &dn_s), &[]);
    let short = vec![0u8; 16];
    let _ = metal.resolve_selected_experts(&s, &moe, &[0], &[1.0], |_| {
        Some((short.as_slice(), short.as_slice()))
    });
}

#[test]
#[should_panic(expected = "not inside a registered region")]
fn a_split_bank_outside_a_registered_region_refuses() {
    let metal = backend();
    let s = scratch(&metal);
    let gu_s = vec![0u8; 64];
    let dn_s = vec![0u8; 64];
    let moe = layer(paired(&gu_s, &dn_s), &[]);
    let gu = vec![0u8; GATE_UP_BYTES];
    let dn = vec![0u8; DOWN_BYTES];
    let _ = metal.resolve_selected_experts(&s, &moe, &[0], &[1.0], |_| {
        Some((gu.as_slice(), dn.as_slice()))
    });
}

#[test]
#[should_panic(expected = "e8m0 stream is not inside a registered region")]
fn registered_payloads_with_unregistered_exponents_refuse() {
    let metal = backend();
    let s = scratch(&metal);
    // Payloads registered, exponents not: the failure this separates out.
    // Without it the dispatch would bind whatever sits at slot 2 and decode
    // every group against it.
    let payload = vec![0u8; GATE_UP_BYTES + DOWN_BYTES];
    assert!(metal.bufs().register_region(&payload[..]));
    let gu_s = vec![0u8; 64];
    let dn_s = vec![0u8; 64];
    let moe = layer(paired(&gu_s, &dn_s), &[]);
    let _ = metal.resolve_selected_experts(&s, &moe, &[0], &[1.0], |_| {
        Some((&payload[..GATE_UP_BYTES], &payload[GATE_UP_BYTES..]))
    });
}

// ── Ragged vs grouped: experts in separate buffers ──

/// Experts whose regions land in *different* buffers cannot use the grouped
/// offset-table dispatch, so they fall to a per-expert one. That arm had no
/// test: `single_base` is true for every fixture that packs its experts into
/// one region, which is all of them.
///
/// This asserts the two arms agree bit for bit. They should — the grouped
/// kernel's reduction body is a copy of `q6k_matvec`'s — and if they ever
/// stop, the ragged arm is the one production silently falls to when a
/// layer's experts span two mmaps.
#[test]
fn ragged_and_grouped_zero_copy_agree_on_q6k() {
    use larql_compute::cpu::ops::q4_common::quantize_q6_k;

    const EXPERTS: usize = 2;
    const RAGGED_TOP_K: usize = 2;
    // Q6_K's super-block is 256 elements, so a 128-wide intermediate would
    // be padded to 256 in the *down* region only — and a fixture that built
    // down at the unpadded width fails the extent check in
    // `resolve_selected_experts`, drops to the staged path, and quietly
    // compares that path to itself. Use a block-multiple width instead.
    const Q6K_INTER: usize = 256;
    let metal = backend();

    let synth = |len: usize, seed: f32| -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.013 + seed).sin()) * 0.2)
            .collect()
    };

    // Q6_K wants block-multiple rows; hidden and inter both are here.
    let mut gate_up: Vec<Vec<u8>> = Vec::new();
    let mut down: Vec<Vec<u8>> = Vec::new();
    for e in 0..EXPERTS {
        let mut gu = synth(Q6K_INTER * HIDDEN, 0.21 + e as f32 * 0.13);
        gu.extend(synth(Q6K_INTER * HIDDEN, 0.51 + e as f32 * 0.17));
        gate_up.push(quantize_q6_k(&gu));
        down.push(quantize_q6_k(&synth(
            HIDDEN * Q6K_INTER,
            0.83 + e as f32 * 0.07,
        )));
    }

    // One region holding both experts → grouped. Two regions, one per
    // expert → ragged. Same bytes either way.
    let mut packed = Vec::new();
    let mut spans = Vec::new();
    for e in 0..EXPERTS {
        let g = packed.len();
        packed.extend_from_slice(&gate_up[e]);
        let d = packed.len();
        packed.extend_from_slice(&down[e]);
        spans.push((g, gate_up[e].len(), d, down[e].len()));
    }
    let per_expert: Vec<Vec<u8>> = (0..EXPERTS)
        .map(|e| {
            let mut v = gate_up[e].clone();
            v.extend_from_slice(&down[e]);
            v
        })
        .collect();

    assert!(metal.bufs().register_region(&packed[..]));
    for region in &per_expert {
        assert!(metal.bufs().register_region(&region[..]));
    }

    let router = synth(EXPERTS * HIDDEN, 0.31);
    let norm: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.0005)).collect();
    let per_expert_scale = vec![1.0f32; EXPERTS];
    let router_scale = vec![1.0f32; HIDDEN];
    let mut moe = layer(MoeExpertScales::Inline, &[]);
    moe.expert_data_format = QuantFormat::Q6_K;
    moe.fused_row_layout = MoeFusedRowLayout::ContiguousHalves;
    moe.num_experts = EXPERTS;
    moe.intermediate_size = Q6K_INTER;
    moe.top_k = RAGGED_TOP_K;
    moe.router_proj = &router;
    moe.router_scale = &router_scale;
    moe.router_per_expert_scale = &per_expert_scale;
    moe.pre_experts_norm = &norm;
    moe.post_ffn1_norm = &norm;
    moe.post_experts_norm = &norm;
    moe.experts_gate_up = (0..EXPERTS).map(|e| gate_up[e].as_slice()).collect();
    moe.experts_down = (0..EXPERTS).map(|e| down[e].as_slice()).collect();

    let s = MoeScratch::new_public_with_format(
        &metal,
        RAGGED_TOP_K,
        HIDDEN,
        Q6K_INTER,
        QuantFormat::Q6_K,
        moe.gate_up_cols(HIDDEN),
    );
    let h = synth(HIDDEN, 0.9);

    // The precondition the ragged arm needs, asserted rather than assumed:
    // an earlier version of this test packed the "ragged" experts in a way
    // that still resolved to one base, so both arms took the grouped path
    // and the comparison below compared a path to itself.
    let base_of = |bytes: &[u8]| {
        metal
            .bufs()
            .resolve_region(bytes)
            .expect("per-expert region must resolve")
            .0
            .gpu_address()
    };
    assert_ne!(
        base_of(&per_expert[0][..gate_up[0].len()]),
        base_of(&per_expert[1][..gate_up[1].len()]),
        "the per-expert regions share a base buffer, so this fixture cannot \
         reach the ragged arm it exists to cover"
    );

    let grouped = metal.moe_block_for_layer(&h, &moe, 1e-6, &s, |e: usize| {
        let (g, gl, d, dl) = spans[e];
        Some((&packed[g..g + gl], &packed[d..d + dl]))
    });
    let ragged = metal.moe_block_for_layer(&h, &moe, 1e-6, &s, |e: usize| {
        let split = gate_up[e].len();
        Some((&per_expert[e][..split], &per_expert[e][split..]))
    });

    assert!(
        grouped.iter().any(|v| v.abs() > 0.0),
        "both arms all-zero — the comparison would be vacuous"
    );
    assert_eq!(grouped.len(), ragged.len());
    for (i, (a, b)) in grouped.iter().zip(&ragged).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "element {i}: grouped={a} ragged={b} — the per-expert arm must \
             reduce identically to the grouped one"
        );
    }
}
