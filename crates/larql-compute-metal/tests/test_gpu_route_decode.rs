//! Cover the S2 GPU-dataflow route arm of `decode/moe_interleave.rs`.
//!
//! `handle_moe_interleave`'s merged-CB arm is gated on
//! `moe_gpu_route::gpu_route_enabled()`, which resolves `LARQL_GPU_ROUTE`
//! through a process-wide `OnceLock` — "read once; a decode-path A/B
//! switch, not a runtime toggle". Inside any test binary that has already
//! decoded once with the variable unset, the arm is permanently dead, so
//! no amount of test ordering in `test_metal_decode_synthetic` can reach
//! it.
//!
//! This file is therefore its own integration-test binary, on the same
//! reasoning (and the same precedent) as `test_decode_diag.rs`: a fresh
//! process, the variable set before the first decode, so the production
//! arm actually runs.
//!
//! What it establishes is that the arm *fires and produces a healthy
//! token* — the route's numerical parity against the CPU-routed control is
//! owned by the ladder's rung-E gate, and the scheduling claim by rung F.
//! Here the point is that this code path executes at all in-crate.

#![cfg(target_os = "macos")]

extern crate blas_src;

use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};
use larql_compute::{
    Activation, MoeExpertScalePolicy, MoeExpertScales, MoeFusedRowLayout, MoeGateRule,
    MoeInputSource, MoeLayerWeights, MoePostExpertNormPolicy, MoeRouterNormPolicy,
    MoeRoutingPolicy, MoeTopKWeightPolicy, MoeWeightLayout, QuantFormat,
};
use larql_compute_metal::{route_witness, MetalBackend};

/// The synthetic layer builder, shared with `test_metal_decode_synthetic`
/// rather than duplicated — `FullPipelineLayer` has ~30 fields and a second
/// hand-written copy would drift.
#[path = "metal_decode_synthetic/common.rs"]
mod common;

const PAGE: usize = 16384;
const HIDDEN: usize = 256;
const INTER: usize = 256;
const NUM_EXPERTS: usize = 8;
const TOP_K: usize = 2;
// 4 x 64 = 256 = one Q4_K superblock. `wo` reduces over Q_DIM, so at
// NUM_Q_HEADS = 2 the O-projection dispatched zero superblocks and
// emitted zeros, silently (issue #227). `stages::quant_matvec::encode`
// now refuses that geometry instead of returning a zero vector.
const NUM_Q_HEADS: usize = 4;
const NUM_KV_HEADS: usize = 1;
const HEAD_DIM: usize = 64;
const Q_DIM: usize = NUM_Q_HEADS * HEAD_DIM;
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM;
const ROW_BYTES: usize = HIDDEN / 256 * 210;
const GU_EXPERT_BYTES: usize = 2 * INTER * ROW_BYTES;
const DN_EXPERT_BYTES: usize = HIDDEN * (INTER / 256 * 210);

fn synth(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (seed + i as f32 * 0.013).sin() * 0.2)
        .collect()
}

fn aligned_backing(size: usize) -> (Vec<u8>, usize) {
    let mem = vec![0u8; size + PAGE];
    let off = mem.as_ptr().align_offset(PAGE);
    (mem, off)
}

/// Q6_K expert bank in registered, page-aligned memory — the zero-copy
/// resolve the descriptor arm needs refuses anything else.
struct Bank {
    _gu_mem: Vec<u8>,
    _dn_mem: Vec<u8>,
    gu_ptr: *const u8,
    dn_ptr: *const u8,
    router_w: Vec<f32>,
    pre_norm: Vec<f32>,
}

fn build_bank(metal: &MetalBackend) -> Bank {
    let gu_size = NUM_EXPERTS * GU_EXPERT_BYTES;
    let dn_size = NUM_EXPERTS * DN_EXPERT_BYTES;
    let (mut gu_mem, gu_off) = aligned_backing(gu_size);
    let (mut dn_mem, dn_off) = aligned_backing(dn_size);
    for e in 0..NUM_EXPERTS {
        let gq = quantize_q6_k(
            &(0..2 * INTER * HIDDEN)
                .map(|i| ((e * 977 + i) as f32 * 0.011).sin() * 0.3)
                .collect::<Vec<f32>>(),
        );
        let dq = quantize_q6_k(
            &(0..HIDDEN * INTER)
                .map(|i| ((e * 613 + i) as f32 * 0.017).cos() * 0.3)
                .collect::<Vec<f32>>(),
        );
        gu_mem[gu_off + e * GU_EXPERT_BYTES..gu_off + (e + 1) * GU_EXPERT_BYTES]
            .copy_from_slice(&gq);
        dn_mem[dn_off + e * DN_EXPERT_BYTES..dn_off + (e + 1) * DN_EXPERT_BYTES]
            .copy_from_slice(&dq);
    }
    let gu_region = &gu_mem[gu_off..gu_off + gu_size];
    let dn_region = &dn_mem[dn_off..dn_off + dn_size];
    assert!(metal.bufs().register_region(gu_region));
    assert!(metal.bufs().register_region(dn_region));

    Bank {
        gu_ptr: gu_region.as_ptr(),
        dn_ptr: dn_region.as_ptr(),
        router_w: (0..NUM_EXPERTS * HIDDEN)
            .map(|i| ((i as f32) * 0.0007).sin() * 0.05)
            .collect(),
        pre_norm: (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect(),
        _gu_mem: gu_mem,
        _dn_mem: dn_mem,
    }
}

impl Bank {
    /// The gpt-oss shape the GPU route was built for: one pre-experts RMS
    /// norm feeding router and experts alike, identity combine, no
    /// post-expert norm.
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
                expert_input: MoeInputSource::PreExpertsNorm,
                router_input: MoeInputSource::PreExpertsNorm,
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
            pre_experts_norm: &self.pre_norm,
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts: NUM_EXPERTS,
            top_k: TOP_K,
            intermediate_size: INTER,
            router_bias: &[],
            experts_gate_up_bias: &[],
            experts_down_bias: &[],
            gate_rule: MoeGateRule::Gated(Activation::Silu),
        }
    }
}

/// The full production arm: a real decode token whose MoE layer takes the
/// GPU-dataflow route.
///
/// The assertions are the ones that distinguish "the arm ran" from "the
/// arm was skipped and something else produced a plausible number":
/// `GPU_ROUTE_LAYERS` must advance (fired evidence, never inferred from
/// silence), and the route-witness counters that count *host* route work
/// must not move at all.
#[test]
fn gpu_route_arm_fires_on_a_real_decode_token() {
    // Set BEFORE the first `gpu_route_enabled()` call: the OnceLock caches
    // whatever it sees first, and this binary exists to make that value 1.
    unsafe { std::env::set_var("LARQL_GPU_ROUTE", "1") };

    let Some(metal) = MetalBackend::new() else {
        return;
    };
    let bank = build_bank(&metal);

    let wq = quantize_q4_k(&synth(Q_DIM * HIDDEN, 0.1));
    let wk = quantize_q4_k(&synth(KV_DIM * HIDDEN, 0.2));
    let wv = quantize_q4_k(&synth(KV_DIM * HIDDEN, 0.3));
    let wo = quantize_q4_k(&synth(HIDDEN * Q_DIM, 0.4));
    let norm_w: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.001)).collect();

    // Empty gate/up/down: `has_dense_ffn()` must be false or the inline
    // preconditions refuse the layer as a hybrid dense+MoE shape, and the
    // merged-CB arm this binary exists to reach would never run.
    let mut layer = common::build_synth_layer(&wq, &wk, &wv, &wo, &[], &[], &[], &norm_w);
    layer.moe = Some(bank.moe());

    let witness_before = route_witness::snapshot();
    let layers_before = route_witness::GPU_ROUTE_LAYERS.load(std::sync::atomic::Ordering::Relaxed);

    let x = synth(HIDDEN, 0.9);
    let out = MetalBackend::decode_token_q4k_moe(
        &metal,
        std::slice::from_ref(&layer),
        &x,
        HIDDEN,
        INTER,
        Q_DIM,
        KV_DIM,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        10_000.0,
        1e-6,
        |_layer_idx, _expert_idx| None,
        None, // head — this test is about the route arm, not TOKEN-B1
    )
    .expect("decode_token_q4k_moe returns a hidden state for a MoE layer");

    assert_eq!(out.len(), HIDDEN);
    assert!(out.iter().all(|v| v.is_finite()), "route produced NaN/Inf");

    let layers_after = route_witness::GPU_ROUTE_LAYERS.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        layers_after > layers_before,
        "GPU_ROUTE_LAYERS did not advance — the merged-CB arm was skipped, \
         so this test proved nothing about it. Check the preconditions \
         with LARQL_MOE_INLINE_DIAG=1 before reading the output above."
    );

    let witness = witness_before.delta(&route_witness::snapshot());
    assert_eq!(
        (
            witness.host_resolves,
            witness.bias_copies,
            witness.weight_binds,
            witness.offset_binds
        ),
        (0, 0, 0, 0),
        "route-dependent HOST work ran on the GPU-route arm: {witness:?}"
    );
}
