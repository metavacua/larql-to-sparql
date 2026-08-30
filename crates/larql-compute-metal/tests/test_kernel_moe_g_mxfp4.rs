#![cfg(target_os = "macos")]

//! Rung G of the GPU-dataflow routing ladder: native MXFP4 as a
//! descriptor VARIANT — the representation binding changes, nothing
//! else does.
//!
//! Both arms run the production split-scale MXFP4 grouped kernel:
//!
//! - CONTROL: today's CPU-resolved path (`resolve_selected_experts` →
//!   `encode_mxfp4_gate_up`/`down` with `set_bytes` payload AND e8m0
//!   offset tables) — the encode-time host decision.
//! - CANDIDATE: GPU route → descriptor gather (which now also expands
//!   the e8m0 scale-offset tables) → the same kernel bound from
//!   gathered buffers, halves selected by row walk.
//!
//! Expert arithmetic is identical (same kernel, same bytes, same
//! reduction order both arms), but the ROUTING WEIGHTS differ in fp
//! provenance (CPU vs GPU softmax), so the full-layer bar is rung E's:
//! exact route, output rel < 1e-4. Bitwise applies where the inputs are
//! identical — the candidate-vs-candidate negative control below. Q6_K
//! is the execution/scheduling control for the performance question;
//! the numerical oracle for MXFP4 is MXFP4's own production arithmetic,
//! not cross-format agreement.
//!
//! Negative control: nudging ONE descriptor's `gate_up_scale_off` must
//! diverge the candidate in the victim slot only — proving the
//! independent exponent-stream indirection is live, not decorative.

#[path = "common/mod.rs"]
mod common;
use common::get_metal;

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
/// MXFP4 geometry: 32-element groups, 16 payload bytes + 1 e8m0 byte each.
const GROUPS_PER_ROW: usize = HIDDEN / 32;
const ROW_PAYLOAD: usize = GROUPS_PER_ROW * 16;
const ROW_SCALES: usize = GROUPS_PER_ROW;
const GU_PAYLOAD_BYTES: usize = 2 * INTER * ROW_PAYLOAD;
const GU_SCALE_BYTES: usize = 2 * INTER * ROW_SCALES;
const DN_PAYLOAD_BYTES: usize = HIDDEN * (INTER / 32) * 16;
const DN_SCALE_BYTES: usize = HIDDEN * (INTER / 32);

fn aligned_backing(size: usize) -> (Vec<u8>, usize) {
    let mem = vec![0u8; size + PAGE];
    let off = mem.as_ptr().align_offset(PAGE);
    (mem, off)
}

struct Region {
    mem: Vec<u8>,
    off: usize,
    per_expert: usize,
}

impl Region {
    /// One registered region holding `NUM_EXPERTS` contiguous slices,
    /// filled by `fill(byte_index) -> byte`.
    fn build(metal: &MetalBackend, per_expert: usize, fill: impl Fn(usize) -> u8) -> Self {
        let size = NUM_EXPERTS * per_expert;
        let (mut mem, off) = aligned_backing(size);
        for (i, b) in mem[off..off + size].iter_mut().enumerate() {
            *b = fill(i);
        }
        assert!(metal.bufs().register_region(&mem[off..off + size]));
        Self {
            mem,
            off,
            per_expert,
        }
    }
    fn slice(&self, e: usize) -> &[u8] {
        &self.mem[self.off + e * self.per_expert..self.off + (e + 1) * self.per_expert]
    }
}

struct Fixture {
    gu_payload: Region,
    dn_payload: Region,
    gu_scales: Region,
    dn_scales: Region,
    router_w: Vec<f32>,
    router_bias: Vec<f32>,
    gate_up_bias: Vec<f32>,
    down_bias: Vec<f32>,
}

fn build_fixture(metal: &MetalBackend) -> Fixture {
    // Any 4-bit code is a valid FP4 value; e8m0 bytes ≤ 127 keep the
    // scale ≤ 2^0, so every decoded number is finite and bounded —
    // valid-by-construction data without needing an encoder.
    let payload = |i: usize| (i.wrapping_mul(2654435761) >> 7) as u8;
    let scales = |i: usize| 110 + ((i.wrapping_mul(40503) >> 5) % 18) as u8; // 110..=127
    Fixture {
        gu_payload: Region::build(metal, GU_PAYLOAD_BYTES, payload),
        dn_payload: Region::build(metal, DN_PAYLOAD_BYTES, |i| payload(i ^ 0x5a5a)),
        gu_scales: Region::build(metal, GU_SCALE_BYTES, scales),
        dn_scales: Region::build(metal, DN_SCALE_BYTES, |i| scales(i ^ 0x33)),
        router_w: (0..NUM_EXPERTS * HIDDEN)
            .map(|i| ((i as f32) * 0.0007).sin() * 0.05)
            .collect(),
        router_bias: (0..NUM_EXPERTS)
            .map(|e| (e as f32 * 0.9).sin() * 0.3)
            .collect(),
        gate_up_bias: (0..NUM_EXPERTS * 2 * INTER)
            .map(|i| ((i as f32) * 0.023).sin() * 0.2)
            .collect(),
        down_bias: (0..NUM_EXPERTS * HIDDEN)
            .map(|i| ((i as f32) * 0.029).cos() * 0.2)
            .collect(),
    }
}

impl Fixture {
    fn moe(&self) -> MoeLayerWeights<'_> {
        MoeLayerWeights {
            expert_scales: MoeExpertScales::Paired {
                gate_up: (0..NUM_EXPERTS).map(|e| self.gu_scales.slice(e)).collect(),
                down: (0..NUM_EXPERTS).map(|e| self.dn_scales.slice(e)).collect(),
            },
            fused_row_layout: MoeFusedRowLayout::ContiguousHalves,
            experts_gate_up: (0..NUM_EXPERTS).map(|e| self.gu_payload.slice(e)).collect(),
            experts_down: (0..NUM_EXPERTS).map(|e| self.dn_payload.slice(e)).collect(),
            routing_policy: MoeRoutingPolicy {
                expert_input: MoeInputSource::Residual,
                router_input: MoeInputSource::Residual,
                router_norm: MoeRouterNormPolicy::None,
                selected_weight: MoeTopKWeightPolicy::RawSoftmax,
                expert_scale: MoeExpertScalePolicy::None,
                post_expert_norm: MoePostExpertNormPolicy::None,
            },
            weight_layout: MoeWeightLayout::default(),
            expert_data_format: QuantFormat::MXFP4,
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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
    let max_abs = a.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
        / max_abs
}

/// G's core assertion: the MXFP4 descriptor binding matches the
/// production CPU-resolved MXFP4 encode at rung E's full-layer bar
/// (exact route, rel < 1e-4 — weights differ only in fp provenance),
/// and the witness contract holds — nothing but the representation
/// binding changed.
#[test]
fn mxfp4_descriptor_arm_matches_production_bitwise() {
    let metal = get_metal();
    let fx = build_fixture(&metal);
    let moe = fx.moe();
    let table = metal
        .build_expert_descriptor_table(&moe, INTER, HIDDEN)
        .expect("split-scale table builds (C1b class)");
    assert!(table.gate_up_scale_base.is_some() && table.down_scale_base.is_some());
    // Gate–claim congruence: the arm this parity licenses is the arm the
    // backend will actually encode. The vectorised default only fires on
    // a 16-aligned table, so pin that this fixture IS one — otherwise a
    // silent demotion would leave the default arm parity-ungated.
    assert!(
        table.payload_offsets_vec16,
        "fixture payload offsets must be 16-byte aligned so the parity \
         gate exercises the default (vectorised) arm"
    );
    let h = h_post_attn();

    for seed in 0..4u32 {
        let x = router_input(seed);
        let before = route_witness::snapshot();
        let control = metal
            .moe_layer_forward_control(&x, &moe, &h)
            .expect("production MXFP4 forward");
        let control_delta = before.delta(&route_witness::snapshot());
        assert!(
            control_delta.host_resolves >= 1 && control_delta.offset_binds >= 1,
            "witness positive control failed on the MXFP4 legacy path: \
             {control_delta:?}"
        );

        let before = route_witness::snapshot();
        let candidate = metal
            .moe_layer_forward_descriptor(&x, &moe, &table, &h, false)
            .expect("descriptor MXFP4 forward");
        assert!(
            before.delta(&route_witness::snapshot()).is_zero(),
            "route-dependent host work in the MXFP4 descriptor path"
        );

        assert!(
            control.iter().all(|v| v.is_finite()),
            "fixture degenerated: non-finite control output"
        );
        let (cpu_ids, _) = larql_compute::cpu::ops::moe::moe_route_from_router_input(&x, &moe);
        let (gpu_ids, _) = metal.moe_route_gpu(&x, &moe).expect("route");
        assert_eq!(cpu_ids, gpu_ids, "seed {seed}: route diverged");
        let rel = max_rel_diff(&control, &candidate);
        assert!(
            rel < 1e-4,
            "seed {seed}: MXFP4 descriptor arm diverges from the \
             production encode (rel={rel:.3e}) — route agrees, so \
             suspect gather/scale-offset binding, never MXFP4 \
             arithmetic (same kernel both arms)"
        );
    }
}

/// Negative control — the independent e8m0 stream indirection must be
/// LIVE: moving one selected expert's scale offset by one row of
/// exponents diverges the candidate in the victim's contribution;
/// restoring it restores bitwise equality.
#[test]
fn negative_control_scale_offset_mutation_diverges_then_restores() {
    use larql_compute_metal::moe_descriptor::GpuExpertDescriptor;

    let metal = get_metal();
    let fx = build_fixture(&metal);
    let moe = fx.moe();
    let table = metal
        .build_expert_descriptor_table(&moe, INTER, HIDDEN)
        .expect("table builds");
    let h = h_post_attn();
    let x = router_input(1);

    let clean = metal
        .moe_layer_forward_descriptor(&x, &moe, &table, &h, false)
        .expect("clean forward");

    // Victim: the top-1 expert of this route (guaranteed selected).
    let (ids, _) = larql_compute::cpu::ops::moe::moe_route_from_router_input(&x, &moe);
    let victim = ids[0];
    let nudge = |delta: i64| unsafe {
        let p = (table.descs.contents() as *mut GpuExpertDescriptor).add(victim);
        (*p).gate_up_scale_off = ((*p).gate_up_scale_off as i64 + delta) as u32;
    };

    nudge(ROW_SCALES as i64);
    let perturbed = metal
        .moe_layer_forward_descriptor(&x, &moe, &table, &h, false)
        .expect("perturbed forward");
    assert_ne!(
        bits(&clean),
        bits(&perturbed),
        "instrument BLIND: a one-row e8m0 offset error left the output \
         bitwise-identical — the scale indirection is not live"
    );

    nudge(-(ROW_SCALES as i64));
    let restored = metal
        .moe_layer_forward_descriptor(&x, &moe, &table, &h, false)
        .expect("restored forward");
    assert_eq!(bits(&clean), bits(&restored), "restore failed");
}
