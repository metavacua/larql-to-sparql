#![cfg(target_os = "macos")]

//! Rung E of the GPU-dataflow routing ladder: semantic closure of the
//! host boundary. Three independent proofs:
//!
//! 1. PARITY — the full production CPU-routed layer (CPU route, CPU
//!    resolve, `set_bytes` offsets/weights, CPU bias staging) against
//!    the descriptor-driven candidate (GPU route → gather → descriptor
//!    bindings → GPU bias staging → GPU-buffer weights). Exact route,
//!    tight output tolerance (weights differ only in fp provenance).
//! 2. POISON — every buffer the legacy path host-staged is filled with
//!    1e30 garbage before the candidate encodes; output must be
//!    BITWISE identical. An accidental host-staging read explodes,
//!    never coincidentally passes. Host-visible route values are also
//!    scrambled after computation — the candidate's signature has no
//!    parameter to receive them (type-level closure).
//! 3. WITNESS — `route_witness` counters: the control encode MUST move
//!    them (the witness's positive control), the candidate encode MUST
//!    NOT. This proves the route-dependent host work is ABSENT, not
//!    merely output-irrelevant — the fact rung F's perf claim needs.
//!
//! E proves no ROUTE-DEPENDENT host work survives. It does NOT claim:
//! no per-layer host work, one-CB execution, no waits, or any
//! performance improvement — those are rung F's subject.

#[path = "common/mod.rs"]
mod common;
use common::get_metal;

use larql_compute::cpu::ops::q4_common::quantize_q6_k;
use larql_compute::{
    MoeExpertScalePolicy, MoeExpertScales, MoeFusedRowLayout, MoeGateRule, MoeInputSource,
    MoeLayerWeights, MoePostExpertNormPolicy, MoeRouterNormPolicy, MoeRoutingPolicy,
    MoeTopKWeightPolicy, MoeWeightLayout, QuantFormat,
};
use larql_compute_metal::{route_witness, MetalBackend};

const PAGE: usize = 16384;
const NUM_EXPERTS: usize = 32;
const HIDDEN: usize = 256;
const INTER: usize = 256;
const TOP_K: usize = 4;
const ROW_BYTES: usize = HIDDEN / 256 * 210;
const GU_EXPERT_BYTES: usize = 2 * INTER * ROW_BYTES;
const DN_EXPERT_BYTES: usize = HIDDEN * (INTER / 256 * 210);
/// gpt-oss clamped-GLU parameters — engages the gate/up bias path.
const GLU_LIMIT: f32 = 7.0;
const GLU_ALPHA: f32 = 1.702;

fn aligned_backing(size: usize) -> (Vec<u8>, usize) {
    let mem = vec![0u8; size + PAGE];
    let off = mem.as_ptr().align_offset(PAGE);
    (mem, off)
}

struct Fixture {
    _gu_mem: Vec<u8>,
    _dn_mem: Vec<u8>,
    gu_ptr: *const u8,
    dn_ptr: *const u8,
    router_w: Vec<f32>,
    router_bias: Vec<f32>,
    gate_up_bias: Vec<f32>,
    down_bias: Vec<f32>,
}

fn build_fixture(metal: &MetalBackend) -> Fixture {
    let gu_size = NUM_EXPERTS * GU_EXPERT_BYTES;
    let dn_size = NUM_EXPERTS * DN_EXPERT_BYTES;
    let (mut gu_mem, gu_off) = aligned_backing(gu_size);
    let (mut dn_mem, dn_off) = aligned_backing(dn_size);
    for e in 0..NUM_EXPERTS {
        let gu_vals: Vec<f32> = (0..2 * INTER * HIDDEN)
            .map(|i| ((e * 977 + i) as f32 * 0.011).sin() * 0.3)
            .collect();
        let dn_vals: Vec<f32> = (0..HIDDEN * INTER)
            .map(|i| ((e * 613 + i) as f32 * 0.017).cos() * 0.3)
            .collect();
        let gq = quantize_q6_k(&gu_vals);
        let dq = quantize_q6_k(&dn_vals);
        assert_eq!(gq.len(), GU_EXPERT_BYTES);
        assert_eq!(dq.len(), DN_EXPERT_BYTES);
        gu_mem[gu_off + e * GU_EXPERT_BYTES..gu_off + (e + 1) * GU_EXPERT_BYTES]
            .copy_from_slice(&gq);
        dn_mem[dn_off + e * DN_EXPERT_BYTES..dn_off + (e + 1) * DN_EXPERT_BYTES]
            .copy_from_slice(&dq);
    }
    let gu_region = &gu_mem[gu_off..gu_off + gu_size];
    let dn_region = &dn_mem[dn_off..dn_off + dn_size];
    assert!(metal.bufs().register_region(gu_region));
    assert!(metal.bufs().register_region(dn_region));

    Fixture {
        gu_ptr: gu_region.as_ptr(),
        dn_ptr: dn_region.as_ptr(),
        router_w: (0..NUM_EXPERTS * HIDDEN)
            .map(|i| ((i as f32) * 0.0007).sin() * 0.05)
            .collect(),
        router_bias: (0..NUM_EXPERTS)
            .map(|e| (e as f32 * 0.9).sin() * 0.3)
            .collect(),
        // Distinctive biases per (expert, row, half): a staging slip is a
        // large numeric change, not noise.
        gate_up_bias: (0..NUM_EXPERTS * 2 * INTER)
            .map(|i| ((i as f32) * 0.023).sin() * 0.2)
            .collect(),
        down_bias: (0..NUM_EXPERTS * HIDDEN)
            .map(|i| ((i as f32) * 0.029).cos() * 0.2)
            .collect(),
        _gu_mem: gu_mem,
        _dn_mem: dn_mem,
    }
}

impl Fixture {
    fn moe(&self) -> MoeLayerWeights<'_> {
        MoeLayerWeights {
            expert_scales: MoeExpertScales::Inline,
            fused_row_layout: MoeFusedRowLayout::ContiguousHalves,
            experts_gate_up: (0..NUM_EXPERTS)
                .map(|e| unsafe {
                    std::slice::from_raw_parts(
                        self.gu_ptr.add(e * GU_EXPERT_BYTES),
                        GU_EXPERT_BYTES,
                    )
                })
                .collect(),
            experts_down: (0..NUM_EXPERTS)
                .map(|e| unsafe {
                    std::slice::from_raw_parts(
                        self.dn_ptr.add(e * DN_EXPERT_BYTES),
                        DN_EXPERT_BYTES,
                    )
                })
                .collect(),
            routing_policy: MoeRoutingPolicy {
                expert_input: MoeInputSource::Residual,
                router_input: MoeInputSource::Residual,
                router_norm: MoeRouterNormPolicy::None,
                selected_weight: MoeTopKWeightPolicy::RawSoftmax,
                expert_scale: MoeExpertScalePolicy::None,
                post_expert_norm: MoePostExpertNormPolicy::None,
            },
            weight_layout: MoeWeightLayout::default(),
            expert_data_format: QuantFormat::Q6_K,
            router_proj: &self.router_w,
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &[],
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts: NUM_EXPERTS,
            top_k: TOP_K,
            intermediate_size: INTER,
            router_bias: &self.router_bias,
            experts_gate_up_bias: &self.gate_up_bias,
            experts_down_bias: &self.down_bias,
            gate_rule: MoeGateRule::ClampedGlu {
                limit: GLU_LIMIT,
                alpha: GLU_ALPHA,
            },
        }
    }
}

fn router_input(seed: u32) -> Vec<f32> {
    (0..HIDDEN)
        .map(|i| (((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32 * 1e-9).sin())
        .collect()
}

fn h_post_attn() -> Vec<f32> {
    (0..HIDDEN)
        .map(|i| ((i as f32) * 0.041).sin() * 0.5)
        .collect()
}

fn max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
    let max_abs = a.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
        / max_abs
}

/// Proofs 1 + 3 — full production parity AND the witness contract, on
/// the same encodes: control moves every counter class (positive
/// control of the witness), candidate moves none, outputs agree.
#[test]
fn e_parity_and_witness_closure() {
    let metal = get_metal();
    let fx = build_fixture(&metal);
    let moe = fx.moe();
    let table = metal
        .build_expert_descriptor_table(&moe, INTER, HIDDEN)
        .expect("table builds");
    let h = h_post_attn();

    for seed in 0..4u32 {
        let x = router_input(seed);

        let before = route_witness::snapshot();
        let control = metal
            .moe_layer_forward_control(&x, &moe, &h)
            .expect("control forward");
        let control_delta = before.delta(&route_witness::snapshot());
        assert!(
            control_delta.host_resolves >= 1
                && control_delta.bias_copies >= 2
                && control_delta.weight_binds >= 1
                && control_delta.offset_binds >= 1,
            "witness positive control failed — legacy path did not move \
             the counters it instruments: {control_delta:?}"
        );

        let before = route_witness::snapshot();
        let candidate = metal
            .moe_layer_forward_descriptor(&x, &moe, &table, &h, false)
            .expect("candidate forward");
        let candidate_delta = before.delta(&route_witness::snapshot());
        assert!(
            candidate_delta.is_zero(),
            "route-dependent host work SURVIVED in the descriptor path: \
             {candidate_delta:?}"
        );

        // Route agreement (exact) is implied by B's gates; re-check here
        // so a parity failure below is immediately attributable.
        let (cpu_ids, _) = larql_compute::cpu::ops::moe::moe_route_from_router_input(&x, &moe);
        let (gpu_ids, _) = metal.moe_route_gpu(&x, &moe).expect("route");
        assert_eq!(cpu_ids, gpu_ids, "seed {seed}: route diverged");

        let rel = max_rel_diff(&control, &candidate);
        assert!(
            rel < 1e-4,
            "seed {seed}: full-layer output diverges (rel={rel:.3e}) — \
             route agrees, so suspect descriptor bindings or staging"
        );
        // The layer must actually do something.
        assert!(
            control.iter().zip(&h).any(|(c, r)| (c - r).abs() > 1e-3),
            "fixture degenerated: expert contribution ~zero"
        );
    }
}

/// Proof 2 — POISON. Garbage in every legacy host-staged buffer, and
/// scrambled host-visible route values, must not move the output by a
/// single bit.
#[test]
fn e_poison_host_route_and_staging_are_irrelevant() {
    let metal = get_metal();
    let fx = build_fixture(&metal);
    let moe = fx.moe();
    let table = metal
        .build_expert_descriptor_table(&moe, INTER, HIDDEN)
        .expect("table builds");
    let h = h_post_attn();
    let x = router_input(99);

    let clean = metal
        .moe_layer_forward_descriptor(&x, &moe, &table, &h, false)
        .expect("clean forward");

    // Dramatic host-route poison: compute what the CPU router WOULD have
    // produced, then wreck it — reversed IDs, all-zero, nonsense weights.
    // The candidate's signature has no parameter that could receive any
    // of these (type-level closure); executing it after the wreckage
    // must be bit-identical.
    let (mut ids, mut weights) =
        larql_compute::cpu::ops::moe::moe_route_from_router_input(&x, &moe);
    ids.reverse();
    ids.fill(0);
    weights.fill(1.0e30);
    std::hint::black_box((&ids, &weights));

    let poisoned = metal
        .moe_layer_forward_descriptor(&x, &moe, &table, &h, true)
        .expect("poisoned forward");

    let clean_bits: Vec<u32> = clean.iter().map(|v| v.to_bits()).collect();
    let poisoned_bits: Vec<u32> = poisoned.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        clean_bits, poisoned_bits,
        "output moved under poison — a host-staged value or host route \
         value leaked into the descriptor path"
    );
}
