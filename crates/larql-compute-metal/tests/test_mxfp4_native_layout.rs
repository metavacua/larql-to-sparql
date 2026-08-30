//! Does the served path actually *obey* the stored fused-row arrangement?
//!
//! The container has been able to record `RegionSchema.layout` since the
//! schema-4 work, and to validate and refuse on it. Nothing executed against
//! it. This is the step where being wrong stops being a refusal and starts
//! being numbers, so the design is a three-arm one and the third arm is the
//! load-bearing one:
//!
//! | arm | bytes | declared layout | expectation |
//! |---|---|---|---|
//! | 1 | interleaved | `Interleaved` | the reference |
//! | 2 | contiguous halves | `ContiguousHalves` | bit-identical to arm 1 |
//! | 3 | interleaved | `ContiguousHalves` | finite, materially divergent |
//!
//! Arms 1 and 2 hold **the same logical weights** in two physical
//! arrangements, so the engine is its own oracle across the rearrangement and
//! no tolerance has to be argued about — same rows, same groups, same
//! reduction order per row, only different addresses.
//!
//! Arm 3 is the control. Without it, arms 1 and 2 agreeing would only show
//! that two code paths do the same thing, which they would also do if the
//! layout field were ignored entirely. Arm 3 is what makes the agreement mean
//! "the declaration was read".
//!
//! ## Two things the fixture does on purpose
//!
//! - **Exponents vary per group.** With one exponent everywhere, a kernel
//!   that read the scale stream at the wrong offset would get the right
//!   answer anyway and this file would pass while proving nothing.
//! - **Scales are placed after every payload**, so a scale region's offset is
//!   nowhere near `payload_offset / 16`. That is the derivation the split-scale
//!   kernel used to perform; a revert to it fails here rather than in a model.

#![cfg(target_os = "macos")]

use larql_compute::{
    Activation, MoeExpertScales, MoeFusedRowLayout, MoeGateRule, MoeLayerWeights, MoeRoutingPolicy,
    MoeWeightLayout, QuantFormat,
};
use larql_compute_metal::{MetalBackend, MoeScratch};
use larql_models::quant::mxfp4::{FusedHalf, MXFP4_GROUP_BYTES, MXFP4_GROUP_ELEMS};

const HIDDEN: usize = 256;
const INTER: usize = 128;
const TOP_K: usize = 2;
const NUM_EXPERTS: usize = 4;
const EPS: f32 = 1e-6;

/// e8m0 bias — exponent byte `b` decodes to `2^(b - 127)`.
const E8M0_BIAS: i32 = 127;
/// How far the fixture's per-group exponents stray from the bias. Small
/// enough that products stay well inside f32, wide enough that reading the
/// wrong group's exponent changes the answer by orders of magnitude.
const EXPONENT_SPREAD: i32 = 3;

/// fp4 (e2m1) magnitudes, sign in bit 3 — the value set a nibble can hold.
const FP4_MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Deterministic mixing for fixture indices.
///
/// Not decoration. The first version of this file derived every index as
/// `seed * k % m` with `k` and `m` sharing a factor, so whole families of
/// seeds collapsed to identical bytes: two banks meant to be unrelated came
/// out byte-identical, and exponents varied per group but not per row —
/// which would have let a kernel read the wrong row's exponents and still
/// produce the right answer. Mixing removes the arithmetic relationship
/// between a seed and the bytes it produces.
fn mix(x: usize) -> usize {
    let mut v = (x as u64) ^ 0x9E37_79B9_7F4A_7C15;
    v ^= v >> 30;
    v = v.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    v ^= v >> 27;
    v = v.wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^= v >> 31;
    v as usize
}

/// One row's MXFP4 encoding: nibble payload plus one exponent byte per group.
struct Row {
    payload: Vec<u8>,
    scales: Vec<u8>,
}

/// Encode row `r` of a synthetic matrix with `cols` columns.
///
/// Codes come straight from the fp4 alphabet, so the encoding is exact and
/// any disagreement between arms is an addressing difference rather than a
/// rounding one.
fn encode_row(seed: usize, cols: usize) -> Row {
    let groups = cols / MXFP4_GROUP_ELEMS;
    let mut payload = Vec::with_capacity(groups * MXFP4_GROUP_BYTES);
    let mut scales = Vec::with_capacity(groups);
    for g in 0..groups {
        // Vary the exponent by row AND group: a kernel indexing the scale
        // stream by the wrong row, or with a derived offset, lands on a
        // different exponent and the product moves by a power of two.
        let step = (mix(seed * 2 + g) % (2 * EXPONENT_SPREAD as usize + 1)) as i32;
        scales.push((E8M0_BIAS + step - EXPONENT_SPREAD) as u8);
        for b in 0..MXFP4_GROUP_BYTES {
            let cell = mix(((seed * MXFP4_GROUP_ELEMS + g) * MXFP4_GROUP_BYTES) + b);
            let lo = (cell % FP4_MAGNITUDES.len()) as u8;
            let hi = ((cell >> 8) % FP4_MAGNITUDES.len()) as u8;
            // Sign bits from the same draw, so cancellation is exercised
            // without correlating sign to magnitude.
            let lo = lo | (((cell >> 16) & 1) << 3) as u8;
            let hi = hi | (((cell >> 17) & 1) << 3) as u8;
            payload.push(lo | (hi << 4));
        }
    }
    Row { payload, scales }
}

/// One expert's operands in both physical arrangements of the fused rows.
struct Expert {
    gate_up_interleaved: Row,
    gate_up_contiguous: Row,
    down: Row,
}

/// Seed offset separating one expert's three operands. Large enough that no
/// two of them share a row encoding.
const OPERAND_SEED_STRIDE: usize = 500;
/// Seed offset separating one expert bank from the next.
const EXPERT_SEED_STRIDE: usize = 1_000;
/// Seed offset producing an entirely unrelated bank — the calibration weights.
const UNRELATED_BANK_SEED: usize = 900_000;

fn build_expert(expert: usize) -> Expert {
    build_expert_from(expert, 0)
}

fn build_expert_from(expert: usize, bank_seed: usize) -> Expert {
    let base = bank_seed + expert * EXPERT_SEED_STRIDE;
    // The SAME logical rows feed both arrangements — that is what makes arms
    // 1 and 2 comparable at all.
    let gate: Vec<Row> = (0..INTER).map(|j| encode_row(base + j, HIDDEN)).collect();
    let up: Vec<Row> = (0..INTER)
        .map(|j| encode_row(base + OPERAND_SEED_STRIDE + j, HIDDEN))
        .collect();

    let mut interleaved = Row {
        payload: Vec::new(),
        scales: Vec::new(),
    };
    for j in 0..INTER {
        for half in [FusedHalf::Gate, FusedHalf::Up] {
            let row = match half {
                FusedHalf::Gate => &gate[j],
                FusedHalf::Up => &up[j],
            };
            interleaved.payload.extend_from_slice(&row.payload);
            interleaved.scales.extend_from_slice(&row.scales);
        }
    }

    let mut contiguous = Row {
        payload: Vec::new(),
        scales: Vec::new(),
    };
    for rows in [&gate, &up] {
        for row in rows {
            contiguous.payload.extend_from_slice(&row.payload);
            contiguous.scales.extend_from_slice(&row.scales);
        }
    }

    // Down is not fused, so it is identical in both arms.
    let mut down = Row {
        payload: Vec::new(),
        scales: Vec::new(),
    };
    for j in 0..HIDDEN {
        let row = encode_row(base + 2 * OPERAND_SEED_STRIDE + j, INTER);
        down.payload.extend_from_slice(&row.payload);
        down.scales.extend_from_slice(&row.scales);
    }

    Expert {
        gate_up_interleaved: interleaved,
        gate_up_contiguous: contiguous,
        down,
    }
}

/// Byte ranges of one expert's four streams inside the packed region.
#[derive(Clone, Copy)]
struct Ranges {
    gate_up: (usize, usize),
    down: (usize, usize),
    gate_up_scales: (usize, usize),
    down_scales: (usize, usize),
}

/// Pack every expert into one region, **payloads first and scales after**.
///
/// The ordering is the point: it puts each scale stream at an offset with no
/// arithmetic relationship to its payload's, so the kernel can only find it
/// by being told.
fn pack(experts: &[Expert], fused: impl Fn(&Expert) -> &Row) -> (Vec<u8>, Vec<Ranges>) {
    let mut region = Vec::new();
    let mut payload_spans = Vec::with_capacity(experts.len());
    for e in experts {
        let gu = fused(e);
        let gu_start = region.len();
        region.extend_from_slice(&gu.payload);
        let dn_start = region.len();
        region.extend_from_slice(&e.down.payload);
        payload_spans.push((
            (gu_start, gu.payload.len()),
            (dn_start, e.down.payload.len()),
        ));
    }
    let mut ranges = Vec::with_capacity(experts.len());
    for (e, (gate_up, down)) in experts.iter().zip(payload_spans) {
        let gu = fused(e);
        let gus_start = region.len();
        region.extend_from_slice(&gu.scales);
        let dns_start = region.len();
        region.extend_from_slice(&e.down.scales);
        ranges.push(Ranges {
            gate_up,
            down,
            gate_up_scales: (gus_start, gu.scales.len()),
            down_scales: (dns_start, e.down.scales.len()),
        });
    }
    (region, ranges)
}

fn slice(region: &[u8], (start, len): (usize, usize)) -> &[u8] {
    &region[start..start + len]
}

fn synth(len: usize, seed: f32, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32 * 0.017 + seed).sin()) * scale)
        .collect()
}

/// Run the MoE block over `region` with `layout` declared for its rows.
fn run(
    metal: &MetalBackend,
    region: &[u8],
    ranges: &[Ranges],
    layout: MoeFusedRowLayout,
    router: &[f32],
    norm: &[f32],
    h: &[f32],
) -> Vec<f32> {
    let per_expert_scale = vec![1.0f32; NUM_EXPERTS];
    let router_scale = vec![1.0f32; HIDDEN];
    let moe = MoeLayerWeights {
        experts_gate_up: ranges.iter().map(|r| slice(region, r.gate_up)).collect(),
        experts_down: ranges.iter().map(|r| slice(region, r.down)).collect(),
        expert_scales: MoeExpertScales::Paired {
            gate_up: ranges
                .iter()
                .map(|r| slice(region, r.gate_up_scales))
                .collect(),
            down: ranges
                .iter()
                .map(|r| slice(region, r.down_scales))
                .collect(),
        },
        fused_row_layout: layout,
        routing_policy: MoeRoutingPolicy::gemma4_hybrid(),
        weight_layout: MoeWeightLayout::default(),
        expert_data_format: QuantFormat::MXFP4,
        router_proj: router,
        router_scale: &router_scale,
        router_per_expert_scale: &per_expert_scale,
        router_norm: &[],
        router_norm_parameter_free: true,
        router_input_scalar: 1.0,
        pre_experts_norm: norm,
        post_ffn1_norm: norm,
        post_experts_norm: norm,
        num_experts: NUM_EXPERTS,
        top_k: TOP_K,
        intermediate_size: INTER,
        router_bias: &[],
        experts_gate_up_bias: &[],
        experts_down_bias: &[],
        gate_rule: MoeGateRule::Gated(Activation::GeluTanh),
    };
    let scratch = MoeScratch::new_public_with_format(
        metal,
        TOP_K,
        HIDDEN,
        INTER,
        QuantFormat::MXFP4,
        moe.gate_up_cols(HIDDEN),
    );
    let get_expert = |e: usize| -> Option<(&[u8], &[u8])> {
        let r = ranges.get(e)?;
        Some((slice(region, r.gate_up), slice(region, r.down)))
    };
    metal.moe_block_for_layer(h, &moe, EPS, &scratch, get_expert)
}

/// Largest elementwise difference, relative to the reference's own scale.
fn relative_divergence(reference: &[f32], other: &[f32]) -> f32 {
    let scale = reference.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    reference
        .iter()
        .zip(other)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs() / scale))
}

#[test]
fn the_declared_fused_row_layout_is_obeyed_and_a_wrong_one_diverges() {
    let Some(metal) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };

    let experts: Vec<Expert> = (0..NUM_EXPERTS).map(build_expert).collect();
    let (interleaved_region, interleaved_ranges) = pack(&experts, |e| &e.gate_up_interleaved);
    let (contiguous_region, contiguous_ranges) = pack(&experts, |e| &e.gate_up_contiguous);

    // The arrangement really must differ, or all three arms are the same arm.
    assert_ne!(
        interleaved_region, contiguous_region,
        "the two packings produced identical bytes — the fixture cannot \
         distinguish the layouts it exists to distinguish"
    );

    assert!(metal.bufs().register_region(&interleaved_region[..]));
    assert!(metal.bufs().register_region(&contiguous_region[..]));

    let router = synth(NUM_EXPERTS * HIDDEN, 0.31, 0.05);
    let norm: Vec<f32> = (0..HIDDEN).map(|i| 1.0 + (i as f32 * 0.0005)).collect();
    let h = synth(HIDDEN, 0.9, 0.5);

    let arm1 = run(
        &metal,
        &interleaved_region,
        &interleaved_ranges,
        MoeFusedRowLayout::Interleaved,
        &router,
        &norm,
        &h,
    );
    let arm2 = run(
        &metal,
        &contiguous_region,
        &contiguous_ranges,
        MoeFusedRowLayout::ContiguousHalves,
        &router,
        &norm,
        &h,
    );
    let arm3 = run(
        &metal,
        &interleaved_region,
        &interleaved_ranges,
        MoeFusedRowLayout::ContiguousHalves,
        &router,
        &norm,
        &h,
    );

    assert!(
        arm1.iter().any(|v| v.abs() > 0.0),
        "the reference arm produced all zeros — every comparison below is vacuous"
    );
    assert!(
        arm1.iter().all(|v| v.is_finite()),
        "the reference arm is not finite"
    );

    // Right layout, two physical arrangements of the same weights: the same
    // rows reduced in the same order, so this is an equality and not a
    // tolerance.
    assert_eq!(arm1.len(), arm2.len());
    for (i, (a, b)) in arm1.iter().zip(arm2.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "element {i}: interleaved={a} contiguous={b} — the same weights \
             stored two ways must reduce to the same value, so the row walk \
             is not being applied"
        );
    }

    // Wrong layout: the failure mode is a plausible answer, not a crash, so
    // both halves of this are load-bearing. Finite first — a NaN would mean
    // the arm broke rather than mis-read.
    assert!(
        arm3.iter().all(|v| v.is_finite()),
        "the wrong-layout arm went non-finite; that is a broken dispatch, not \
         the silent mis-read this control is meant to catch"
    );
    // How divergent is "materially divergent"? Not a number picked to pass.
    // The reference scale is what serving *entirely different weights* does
    // to this output — the worst a bank can be wrong. A wrong row layout is
    // only a real failure if it lands in that regime rather than perturbing
    // the answer slightly.
    let unrelated_experts: Vec<Expert> = (0..NUM_EXPERTS)
        .map(|e| build_expert_from(e, UNRELATED_BANK_SEED))
        .collect();
    let (unrelated_region, unrelated_ranges) = pack(&unrelated_experts, |e| &e.gate_up_interleaved);
    assert!(metal.bufs().register_region(&unrelated_region[..]));
    let unrelated = run(
        &metal,
        &unrelated_region,
        &unrelated_ranges,
        MoeFusedRowLayout::Interleaved,
        &router,
        &norm,
        &h,
    );

    let wrong_layout = relative_divergence(&arm1, &arm3);
    let different_weights = relative_divergence(&arm1, &unrelated);
    assert!(
        different_weights > 0.1,
        "the calibration arm barely moved the output ({different_weights:.2e} \
         relative), so it cannot stand for `as wrong as possible` and the \
         comparison below would be measuring nothing"
    );
    // Same order of magnitude as swapping the weights outright.
    assert!(
        wrong_layout > different_weights / 10.0,
        "declaring the wrong row layout moved the output by {wrong_layout:.2e} \
         relative, against {different_weights:.2e} for entirely different \
         weights — too small to be the gate/up mixture, so the layout field is \
         not reaching the kernel"
    );
}
