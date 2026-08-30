#![cfg(target_os = "macos")]

//! Rung B of the GPU-dataflow routing ladder: fused route selection
//! parity (softmax → deterministic top-k → weight policy).
//!
//! GPU arm: `MetalBackend::moe_route_gpu` — projection (rung A) chained
//! into `moe_router_select` (one single-TG dispatch) in one command
//! buffer. CPU oracle: `moe_route_from_router_input`, the exact
//! production routing function.
//!
//! Gate structure, in the order the discipline demands:
//! 1. negative control — a boundary-crossing perturbation must CHANGE
//!    the selected IDs, then restore-green;
//! 2. engineered exact tie — equal scores around the selection boundary
//!    resolve to the lower expert id on BOTH arms (the tie contract);
//! 3. policy parity — exact IDs/order + tight weight tolerance across
//!    the weight-policy matrix (selection semantics and normalisation
//!    semantics are independent);
//! 4. long-stream parity — hundreds of evolving router inputs;
//! 5. margin diagnostic — every mismatch reports the k-th vs (k+1)-th
//!    probability margin, so a rare future failure is immediately
//!    classifiable as near-tie jitter vs a genuine kernel defect.

#[path = "common/mod.rs"]
mod common;
use common::get_metal;

use larql_compute::cpu::ops::moe::moe_route_from_router_input;
use larql_compute::{
    Activation, MoeExpertScalePolicy, MoeExpertScales, MoeFusedRowLayout, MoeGateRule,
    MoeInputSource, MoeLayerWeights, MoePostExpertNormPolicy, MoeRouterNormPolicy,
    MoeRoutingPolicy, MoeTopKWeightPolicy, MoeWeightLayout, QuantFormat,
};

/// gpt-oss-20b router shape: 32 experts over hidden 2880, top-4.
const NUM_EXPERTS: usize = 32;
const HIDDEN: usize = 2880;
const TOP_K: usize = 4;

/// Weight comparison bar, relative to the largest selected weight.
/// Selection (IDs/order) is exact; only the weights carry fp noise
/// (CPU serial vs GPU tree accumulation, `exp` implementations).
const WEIGHT_REL_TOL: f32 = 1e-4;

/// Owned router tables; `moe()` borrows them into a `MoeLayerWeights`.
struct RouterFixture {
    w: Vec<f32>,
    bias: Vec<f32>,
    scale: Vec<f32>,
}

fn synth_fixture(num_experts: usize, hidden: usize) -> RouterFixture {
    RouterFixture {
        w: (0..num_experts * hidden)
            .map(|i| {
                let f = i as f32;
                ((f * 0.0003).sin() + 0.3 * (f * 0.0011).cos()) * 0.05
            })
            .collect(),
        bias: (0..num_experts)
            .map(|e| ((e as f32) * 0.7).sin() * 0.5)
            .collect(),
        // Distinctive per-expert scales: a dropped multiply can't hide
        // behind a table of 1.0s.
        scale: (0..num_experts).map(|e| 0.5 + 0.05 * e as f32).collect(),
    }
}

fn policy(
    selected_weight: MoeTopKWeightPolicy,
    expert_scale: MoeExpertScalePolicy,
) -> MoeRoutingPolicy {
    // Upstream fields (inputs/norms) act before `router_in` exists and
    // are irrelevant to both arms under test; pinned to the identity
    // choices.
    MoeRoutingPolicy {
        expert_input: MoeInputSource::Residual,
        router_input: MoeInputSource::Residual,
        router_norm: MoeRouterNormPolicy::None,
        selected_weight,
        expert_scale,
        post_expert_norm: MoePostExpertNormPolicy::None,
    }
}

/// Routing-only `MoeLayerWeights`: expert tables empty (neither arm
/// touches them), router tables borrowed from the fixture.
fn moe<'a>(
    fx: &'a RouterFixture,
    num_experts: usize,
    top_k: usize,
    routing_policy: MoeRoutingPolicy,
    with_bias: bool,
) -> MoeLayerWeights<'a> {
    MoeLayerWeights {
        expert_scales: MoeExpertScales::Inline,
        fused_row_layout: MoeFusedRowLayout::ContiguousHalves,
        experts_gate_up: Vec::new(),
        experts_down: Vec::new(),
        routing_policy,
        weight_layout: MoeWeightLayout::default(),
        expert_data_format: QuantFormat::BF16,
        router_proj: &fx.w,
        router_scale: &[],
        router_per_expert_scale: &fx.scale,
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts,
        top_k,
        intermediate_size: 0,
        router_bias: if with_bias { &fx.bias } else { &[] },
        experts_gate_up_bias: &[],
        experts_down_bias: &[],
        gate_rule: MoeGateRule::Gated(Activation::Silu),
    }
}

fn router_input(hidden: usize, seed: u32) -> Vec<f32> {
    (0..hidden)
        .map(|i| (((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as f32 * 1e-9).sin())
        .collect()
}

/// Selection margin from the oracle's own probabilities: distance
/// between the k-th selected and the first excluded expert. The number
/// that classifies any mismatch as near-tie jitter vs kernel defect.
fn selection_margin(router_in: &[f32], m: &MoeLayerWeights<'_>) -> f32 {
    let mut logits = larql_compute::cpu::ops::moe::moe_router_logits(
        router_in,
        m.router_proj,
        m.router_bias,
        m.num_experts,
    );
    larql_compute::cpu::ops::moe::moe_softmax(&mut logits);
    let mut sorted = logits;
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    sorted[m.top_k - 1] - sorted[m.top_k]
}

fn max_weight_rel_diff(cpu: &[f32], gpu: &[f32]) -> f32 {
    let max_abs = cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-9);
    cpu.iter()
        .zip(gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
        / max_abs
}

/// NEGATIVE CONTROL — a single-weight perturbation lifting the expert
/// ranked k+1 across the k-th boundary must CHANGE the GPU-selected IDs;
/// restoring the weight must restore ID equality. Runs before parity is
/// trusted: a gate that has never failed proves nothing.
#[test]
fn negative_control_boundary_crossing_changes_selected_ids() {
    let metal = get_metal();
    let fx = synth_fixture(NUM_EXPERTS, HIDDEN);
    let x = router_input(HIDDEN, 1);
    let pol = policy(MoeTopKWeightPolicy::RawSoftmax, MoeExpertScalePolicy::None);

    let m = moe(&fx, NUM_EXPERTS, TOP_K, pol, true);
    let (cpu_ids, _) = moe_route_from_router_input(&x, &m);

    // Rank all experts on the oracle's logits to find the boundary pair.
    let logits = larql_compute::cpu::ops::moe::moe_router_logits(&x, &fx.w, &fx.bias, NUM_EXPERTS);
    let mut rank: Vec<usize> = (0..NUM_EXPERTS).collect();
    rank.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap().then(a.cmp(&b)));
    let kth = rank[TOP_K - 1];
    let boundary = rank[TOP_K];
    let gap = logits[kth] - logits[boundary];
    assert!(gap > 0.0, "fixture degenerate: no boundary gap");

    let j = (0..HIDDEN)
        .max_by(|&a, &b| x[a].abs().partial_cmp(&x[b].abs()).unwrap())
        .unwrap();
    let mut fx_perturbed = synth_fixture(NUM_EXPERTS, HIDDEN);
    fx_perturbed.w[boundary * HIDDEN + j] += 1.5 * gap / x[j];

    let m_perturbed = moe(&fx_perturbed, NUM_EXPERTS, TOP_K, pol, true);
    let (gpu_ids, _) = metal
        .moe_route_gpu(&x, &m_perturbed)
        .expect("GPU route dispatch");
    assert_ne!(
        cpu_ids, gpu_ids,
        "instrument BLIND: a boundary-crossing perturbation did not \
         change the selected IDs — the parity assertions below cannot \
         be trusted"
    );
    assert!(
        gpu_ids.contains(&boundary),
        "expert {boundary} was lifted across the boundary but is absent \
         from the perturbed route {gpu_ids:?}"
    );

    // Restore-green.
    let (gpu_ids, _) = metal.moe_route_gpu(&x, &m).expect("GPU route dispatch");
    assert_eq!(
        cpu_ids, gpu_ids,
        "restored input fails ID parity — the control leaked into the \
         baseline arm"
    );
}

/// THE TIE CONTRACT — duplicate router rows produce bit-identical
/// logits within each arm, so both arms face an exact tie and must
/// resolve it the same way: lower expert id first. Covers a tie INSIDE
/// the selection and a tie ACROSS the boundary (where the contract
/// decides membership, not just order).
#[test]
fn engineered_exact_tie_resolves_to_lower_expert_id_on_both_arms() {
    let metal = get_metal();
    let e = 8usize;
    let h = 64usize;
    let mut fx = synth_fixture(e, h);
    // Rows 2 and 5 identical → exactly equal logits. Bias empty so the
    // tie survives to selection.
    let row2: Vec<f32> = fx.w[2 * h..3 * h].to_vec();
    fx.w[5 * h..6 * h].copy_from_slice(&row2);
    // Push the tied pair decisively to the top: add a common large
    // component through the weights' first column.
    fx.w[2 * h] += 10.0;
    fx.w[5 * h] += 10.0;
    let x: Vec<f32> = {
        let mut x = router_input(h, 7);
        x[0] = 1.0; // ensure the +10.0 column is live
        x
    };
    let pol = policy(MoeTopKWeightPolicy::RawSoftmax, MoeExpertScalePolicy::None);

    // Tie INSIDE the selection: k=3 takes both tied experts; contract
    // fixes their order as (2, 5).
    let m = moe(&fx, e, 3, pol, false);
    let (cpu_ids, _) = moe_route_from_router_input(&x, &m);
    let (gpu_ids, _) = metal.moe_route_gpu(&x, &m).expect("GPU route dispatch");
    assert_eq!(cpu_ids[0], 2, "CPU tie order: lower id first");
    assert_eq!(cpu_ids[1], 5, "CPU tie order: higher id second");
    assert_eq!(cpu_ids, gpu_ids, "tie order diverges inside selection");

    // Tie ACROSS the boundary: k=1 must pick expert 2 on both arms.
    let m = moe(&fx, e, 1, pol, false);
    let (cpu_ids, _) = moe_route_from_router_input(&x, &m);
    let (gpu_ids, _) = metal.moe_route_gpu(&x, &m).expect("GPU route dispatch");
    assert_eq!(cpu_ids, vec![2], "CPU boundary tie: lower id wins");
    assert_eq!(gpu_ids, vec![2], "GPU boundary tie: lower id wins");
}

/// POLICY PARITY — selection semantics and normalisation semantics are
/// independent; the kernel takes both as inputs rather than baking in
/// one model's combination. Exact IDs/order, weights within tolerance,
/// across the full (weight policy × scale policy) matrix, bias on/off.
#[test]
fn route_parity_across_weight_policy_matrix() {
    let metal = get_metal();
    let fx = synth_fixture(NUM_EXPERTS, HIDDEN);

    let cases = [
        (
            MoeTopKWeightPolicy::RawSoftmax,
            MoeExpertScalePolicy::None,
            true,
        ),
        (
            MoeTopKWeightPolicy::RawSoftmax,
            MoeExpertScalePolicy::PerExpert,
            false,
        ),
        (
            MoeTopKWeightPolicy::RenormalizedSoftmax,
            MoeExpertScalePolicy::None,
            false,
        ),
        (
            MoeTopKWeightPolicy::RenormalizedSoftmax,
            MoeExpertScalePolicy::PerExpert,
            true,
        ),
    ];
    for (weight_pol, scale_pol, with_bias) in cases {
        let m = moe(
            &fx,
            NUM_EXPERTS,
            TOP_K,
            policy(weight_pol, scale_pol),
            with_bias,
        );
        for seed in 0..8u32 {
            let x = router_input(HIDDEN, 100 + seed);
            let (cpu_ids, cpu_w) = moe_route_from_router_input(&x, &m);
            let (gpu_ids, gpu_w) = metal.moe_route_gpu(&x, &m).expect("GPU route dispatch");

            assert_eq!(
                cpu_ids,
                gpu_ids,
                "IDs diverge under ({weight_pol:?}, {scale_pol:?}, bias={with_bias}) \
                 seed {seed}; margin {:.3e}",
                selection_margin(&x, &m),
            );
            let rel = max_weight_rel_diff(&cpu_w, &gpu_w);
            assert!(
                rel < WEIGHT_REL_TOL,
                "weights diverge (rel={rel:.3e}) under \
                 ({weight_pol:?}, {scale_pol:?}, bias={with_bias}) seed {seed}: \
                 cpu={cpu_w:?} gpu={gpu_w:?}"
            );
        }
    }
}

/// LONG-STREAM PARITY — routing errors are discrete and rare; a
/// single-input gate can pass on coincidence. 500 evolving inputs
/// through the gpt-oss policy; every mismatch reports its selection
/// margin so a failure is immediately classifiable (near-tie jitter ≈
/// instrument floor vs real margin = kernel defect).
#[test]
fn long_stream_route_parity_with_margin_diagnostic() {
    let metal = get_metal();
    let fx = synth_fixture(NUM_EXPERTS, HIDDEN);
    let pol = policy(MoeTopKWeightPolicy::RawSoftmax, MoeExpertScalePolicy::None);
    let m = moe(&fx, NUM_EXPERTS, TOP_K, pol, true);

    const STEPS: u32 = 500;
    let mut mismatches = Vec::new();
    let mut min_margin = f32::INFINITY;
    for step in 0..STEPS {
        let x = router_input(HIDDEN, 1000 + step);
        let (cpu_ids, cpu_w) = moe_route_from_router_input(&x, &m);
        let (gpu_ids, gpu_w) = metal.moe_route_gpu(&x, &m).expect("GPU route dispatch");

        let margin = selection_margin(&x, &m);
        min_margin = min_margin.min(margin);
        if cpu_ids != gpu_ids {
            eprintln!(
                "step {step}: ID mismatch cpu={cpu_ids:?} gpu={gpu_ids:?} \
                 margin={margin:.3e}"
            );
            mismatches.push((step, margin));
            continue;
        }
        let rel = max_weight_rel_diff(&cpu_w, &gpu_w);
        if rel >= WEIGHT_REL_TOL {
            eprintln!("step {step}: weight divergence rel={rel:.3e} margin={margin:.3e}");
            mismatches.push((step, margin));
        }
    }
    eprintln!(
        "long-stream: {STEPS} steps, {} mismatches, min selection margin {min_margin:.3e}",
        mismatches.len(),
    );
    assert!(
        mismatches.is_empty(),
        "route parity broke on {}/{STEPS} steps; margins above classify \
         each: ulp-scale ⇒ near-tie instrument floor, real margin ⇒ \
         kernel defect. mismatches: {mismatches:?}",
        mismatches.len(),
    );
}

/// Shapes the single-TG kernel does not cover return `None` — the
/// caller falls back to CPU routing, never a wrong dispatch.
#[test]
fn rejects_shapes_outside_kernel_contract() {
    let metal = get_metal();
    let pol = policy(MoeTopKWeightPolicy::RawSoftmax, MoeExpertScalePolicy::None);

    // num_experts above the one-thread-per-expert ceiling.
    let e_big = 257usize;
    let h = 16usize;
    let fx_big = synth_fixture(e_big, h);
    let m = moe(&fx_big, e_big, 4, pol, false);
    assert!(metal.moe_route_gpu(&router_input(h, 1), &m).is_none());

    let fx = synth_fixture(8, h);
    // top_k = 0 and top_k > MAX_TOP_K.
    let m = moe(&fx, 8, 0, pol, false);
    assert!(metal.moe_route_gpu(&router_input(h, 1), &m).is_none());
    let m = moe(&fx, 8, 33, pol, false);
    assert!(metal.moe_route_gpu(&router_input(h, 1), &m).is_none());

    // Engaged-but-short per-expert scale table: malformed router
    // description, refused rather than reproduced.
    let mut fx_short = synth_fixture(8, h);
    fx_short.scale.truncate(3);
    let m = moe(
        &fx_short,
        8,
        2,
        policy(
            MoeTopKWeightPolicy::RawSoftmax,
            MoeExpertScalePolicy::PerExpert,
        ),
        false,
    );
    assert!(metal.moe_route_gpu(&router_input(h, 1), &m).is_none());
}
