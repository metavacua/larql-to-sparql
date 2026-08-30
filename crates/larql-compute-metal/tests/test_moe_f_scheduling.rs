#![cfg(target_os = "macos")]

//! Rung F of the GPU-dataflow routing ladder: the queue stays fed.
//!
//! A pure SCHEDULING experiment — no kernel rewrites, no gather removal,
//! no fusion, no format change. Both arms run the identical A→E
//! production encodes over a 24-layer chained token (layer i+1's router
//! input IS layer i's output buffer; no readback, no host staging
//! between layers). Only WHEN work is submitted differs:
//!
//! - JIT: one command buffer per layer, commit + wait each — the cadence
//!   production decode has today.
//! - PRE-ENCODED: all layers in ONE command buffer — the shape rung E's
//!   semantic closure makes legal.
//!
//! Acceptance is the SHAPE, not a tok/s number (the fixture is small;
//! end-to-end throughput lands when the real decode loop integrates the
//! descriptor path): cmd_bufs 24 → 1, inter-CB bubble collapses, GPU
//! busy comparable, outputs BITWISE equal (a submission policy must not
//! change numerics), witness counters zero in both arms.

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
const LAYERS: usize = 24;
const ROW_BYTES: usize = HIDDEN / 256 * 210;
const GU_EXPERT_BYTES: usize = 2 * INTER * ROW_BYTES;
const DN_EXPERT_BYTES: usize = HIDDEN * (INTER / 256 * 210);

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
        // Scale small so a 24-layer residual chain stays numerically tame.
        let gu_vals: Vec<f32> = (0..2 * INTER * HIDDEN)
            .map(|i| ((e * 977 + i) as f32 * 0.011).sin() * 0.05)
            .collect();
        let dn_vals: Vec<f32> = (0..HIDDEN * INTER)
            .map(|i| ((e * 613 + i) as f32 * 0.017).cos() * 0.05)
            .collect();
        gu_mem[gu_off + e * GU_EXPERT_BYTES..gu_off + (e + 1) * GU_EXPERT_BYTES]
            .copy_from_slice(&quantize_q6_k(&gu_vals));
        dn_mem[dn_off + e * DN_EXPERT_BYTES..dn_off + (e + 1) * DN_EXPERT_BYTES]
            .copy_from_slice(&quantize_q6_k(&dn_vals));
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
        gate_up_bias: (0..NUM_EXPERTS * 2 * INTER)
            .map(|i| ((i as f32) * 0.023).sin() * 0.02)
            .collect(),
        down_bias: (0..NUM_EXPERTS * HIDDEN)
            .map(|i| ((i as f32) * 0.029).cos() * 0.02)
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
                limit: 7.0,
                alpha: 1.702,
            },
        }
    }
}

/// The F gate: identical numerics under both submission policies, the
/// starvation bubble collapses, and no route-dependent host work in
/// either arm. Ungated (fast at this fixture size) — the shape claim is
/// a correctness property; the `#[ignore]`d probe below reports the
/// timing magnitudes.
#[test]
fn pre_encoded_token_matches_jit_bitwise_and_kills_the_bubble() {
    let metal = get_metal();
    let fx = build_fixture(&metal);
    let moe = fx.moe();
    let table = metal
        .build_expert_descriptor_table(&moe, INTER, HIDDEN)
        .expect("table builds");
    let x: Vec<f32> = (0..HIDDEN).map(|i| ((i as f32) * 0.013).sin()).collect();

    // Warmup both arms (pipeline/JIT costs belong to neither).
    for pre in [false, true] {
        metal
            .moe_token_forward_descriptor(&x, &moe, &table, LAYERS, pre)
            .expect("warmup");
    }

    let before = route_witness::snapshot();
    let jit = metal
        .moe_token_forward_descriptor(&x, &moe, &table, LAYERS, false)
        .expect("JIT arm");
    let pre = metal
        .moe_token_forward_descriptor(&x, &moe, &table, LAYERS, true)
        .expect("pre-encoded arm");
    let delta = before.delta(&route_witness::snapshot());

    assert!(
        delta.is_zero(),
        "route-dependent host work appeared in the F path: {delta:?}"
    );
    assert_eq!(jit.cmd_bufs, LAYERS);
    assert_eq!(pre.cmd_bufs, 1);

    // A submission policy must not change the numbers.
    let jit_bits: Vec<u32> = jit.out.iter().map(|v| v.to_bits()).collect();
    let pre_bits: Vec<u32> = pre.out.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        jit_bits, pre_bits,
        "pre-encoding changed the token's numerics — scheduling leaked \
         into semantics"
    );
    assert!(
        jit.out.iter().all(|v| v.is_finite()),
        "fixture degenerated: non-finite output"
    );

    // The starvation bubble must collapse, not merely shrink.
    eprintln!(
        "JIT  {} CB  wall {:7.3} ms  busy {:7.3} ms  bubble {:7.3} ms",
        jit.cmd_bufs, jit.wall_ms, jit.gpu_busy_ms, jit.bubble_ms
    );
    eprintln!(
        "PRE  {} CB  wall {:7.3} ms  busy {:7.3} ms  bubble {:7.3} ms",
        pre.cmd_bufs, pre.wall_ms, pre.gpu_busy_ms, pre.bubble_ms
    );
    assert!(
        pre.bubble_ms <= jit.bubble_ms * 0.1,
        "pre-encoding did not collapse the inter-CB bubble: JIT \
         {:.3} ms vs pre-encoded {:.3} ms — a scheduling dependency \
         survives (E closed the semantic class, so look for waits, \
         readbacks or contents() in the loop)",
        jit.bubble_ms,
        pre.bubble_ms,
    );
}

/// Timing magnitudes over repeats, for the record. `--ignored --nocapture`.
#[test]
#[ignore = "timing measurement; run explicitly with --ignored --nocapture"]
fn scheduling_transition_magnitudes() {
    let metal = get_metal();
    let fx = build_fixture(&metal);
    let moe = fx.moe();
    let table = metal
        .build_expert_descriptor_table(&moe, INTER, HIDDEN)
        .expect("table builds");
    let x: Vec<f32> = (0..HIDDEN).map(|i| ((i as f32) * 0.013).sin()).collect();
    for pre in [false, true] {
        metal
            .moe_token_forward_descriptor(&x, &moe, &table, LAYERS, pre)
            .expect("warmup");
    }
    const REPEATS: usize = 30;
    println!("\n=== F scheduling transition, {LAYERS} chained layers ===");
    for (label, pre) in [("JIT ", false), ("PRE ", true)] {
        let mut wall = Vec::with_capacity(REPEATS);
        let mut bubble = Vec::with_capacity(REPEATS);
        let mut busy = Vec::with_capacity(REPEATS);
        let mut cbs = 0;
        for _ in 0..REPEATS {
            let s = metal
                .moe_token_forward_descriptor(&x, &moe, &table, LAYERS, pre)
                .expect("run");
            wall.push(s.wall_ms);
            bubble.push(s.bubble_ms);
            busy.push(s.gpu_busy_ms);
            cbs = s.cmd_bufs;
        }
        let med = |v: &mut Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        println!(
            "{label} {cbs:2} CB  wall {:7.3} ms  busy {:7.3} ms  bubble {:7.3} ms",
            med(&mut wall),
            med(&mut busy),
            med(&mut bubble),
        );
    }
}
