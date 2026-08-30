#![cfg(target_os = "macos")]

//! Rung C of the GPU-dataflow routing ladder: the expert descriptor.
//!
//! C1 — construction: CPU inspection proves exact offsets against where
//!      the fixture PLACED each expert (anti-symmetric layout: quadratic
//!      stagger, so no uniform-stride assumption can pass by accident;
//!      payload ≠ scale ≠ bias offsets by construction).
//! C2 — GPU lookup: `selected_ids → exact descriptors` with a
//!      non-monotonic route `[17, 2, 29, 5]`, so a kernel that confuses
//!      `slot` with `expert_id`, or assumes contiguous experts, fails
//!      loudly.
//! C3 — bias lookup: asymmetric per-expert bias bank, zero payload
//!      involvement — selection alone determines the staged rows, and no
//!      CPU staging exists between bank upload and readback (the route
//!      is consumed entirely by GPU kernels).
//! C4 — refusal: malformed banks (ragged, unregistered, cross-buffer,
//!      contradictory bias dims, missing scale streams) return `None`
//!      totally — never a partially valid table, never a silent CPU
//!      reconstruction inside the GPU-dataflow path.

#[path = "common/mod.rs"]
mod common;
use common::get_metal;

use larql_compute::{
    Activation, MoeExpertScales, MoeFusedRowLayout, MoeGateRule, MoeLayerWeights, MoeRoutingPolicy,
    MoeWeightLayout, QuantFormat,
};
use larql_compute_metal::MetalBackend;

/// Apple Silicon page size — `register_region` requires page-aligned
/// region starts.
const PAGE: usize = 16384;

const NUM_EXPERTS: usize = 32;
/// The non-monotonic route every lookup gate uses: strictly out of
/// order so slot ≠ expert_id everywhere.
const ROUTE: [u32; 4] = [17, 2, 29, 5];

/// Page-aligned backing memory: (backing vec, aligned start offset).
fn aligned_backing(size: usize) -> (Vec<u8>, usize) {
    let mem = vec![0u8; size + PAGE];
    let off = mem.as_ptr().align_offset(PAGE);
    (mem, off)
}

/// Anti-symmetric placement: expert `e`'s slice starts at
/// `lead + e·len + e²·stagger` — non-uniform stride, so an offset table
/// that secretly encodes `base + e·stride` cannot match it.
fn placement(e: usize, len: usize, lead: usize, stagger: usize) -> usize {
    lead + e * len + e * e * stagger
}

struct Bank {
    /// Backing allocations — keep alive: regions are no-copy views.
    _gu_mem: Vec<u8>,
    _dn_mem: Vec<u8>,
    gu_region_ptr: *const u8,
    dn_region_ptr: *const u8,
    gu_slices: Vec<(usize, usize)>, // (placement, len) within region
    dn_slices: Vec<(usize, usize)>,
    gate_up_bias: Vec<f32>,
    down_bias: Vec<f32>,
}

const GU_LEN: usize = 96;
const DN_LEN: usize = 64;
const GU_LEAD: usize = 256;
const DN_LEAD: usize = 512;
const GU_STAGGER: usize = 32;
const DN_STAGGER: usize = 16;
const INTER: usize = 8;
const HIDDEN: usize = 4;

fn build_bank(metal: &MetalBackend, num_experts: usize) -> Bank {
    let gu_size = placement(num_experts, GU_LEN, GU_LEAD, GU_STAGGER);
    let dn_size = placement(num_experts, DN_LEN, DN_LEAD, DN_STAGGER);
    let (gu_mem, gu_off) = aligned_backing(gu_size);
    let (dn_mem, dn_off) = aligned_backing(dn_size);
    let gu_region = &gu_mem[gu_off..gu_off + gu_size];
    let dn_region = &dn_mem[dn_off..dn_off + dn_size];
    assert!(
        metal.bufs().register_region(gu_region),
        "gu region registers"
    );
    assert!(
        metal.bufs().register_region(dn_region),
        "dn region registers"
    );

    // Asymmetric bias bank: `bank[e][fused_row r] = 1000·e + r`, so every
    // (expert, row, half) triple has a unique value — a wrong expert, a
    // swapped gate/up half, or an off-by-one row all produce distinct
    // wrong numbers, not coincidental matches.
    let gate_up_bias: Vec<f32> = (0..num_experts * 2 * INTER)
        .map(|i| {
            let e = i / (2 * INTER);
            let r = i % (2 * INTER);
            (1000 * e + r) as f32
        })
        .collect();
    let down_bias: Vec<f32> = (0..num_experts * HIDDEN)
        .map(|i| (7000 + i) as f32)
        .collect();

    Bank {
        gu_region_ptr: gu_region.as_ptr(),
        dn_region_ptr: dn_region.as_ptr(),
        gu_slices: (0..num_experts)
            .map(|e| (placement(e, GU_LEN, GU_LEAD, GU_STAGGER), GU_LEN))
            .collect(),
        dn_slices: (0..num_experts)
            .map(|e| (placement(e, DN_LEN, DN_LEAD, DN_STAGGER), DN_LEN))
            .collect(),
        gate_up_bias,
        down_bias,
        _gu_mem: gu_mem,
        _dn_mem: dn_mem,
    }
}

impl Bank {
    fn gu_slice(&self, e: usize) -> &[u8] {
        let (pos, len) = self.gu_slices[e];
        unsafe { std::slice::from_raw_parts(self.gu_region_ptr.add(pos), len) }
    }
    fn dn_slice(&self, e: usize) -> &[u8] {
        let (pos, len) = self.dn_slices[e];
        unsafe { std::slice::from_raw_parts(self.dn_region_ptr.add(pos), len) }
    }

    fn moe(&self, num_experts: usize) -> MoeLayerWeights<'_> {
        MoeLayerWeights {
            expert_scales: MoeExpertScales::Inline,
            fused_row_layout: MoeFusedRowLayout::ContiguousHalves,
            experts_gate_up: (0..num_experts).map(|e| self.gu_slice(e)).collect(),
            experts_down: (0..num_experts).map(|e| self.dn_slice(e)).collect(),
            routing_policy: MoeRoutingPolicy::top_k_softmax(),
            weight_layout: MoeWeightLayout::default(),
            expert_data_format: QuantFormat::Q6_K,
            router_proj: &[],
            router_scale: &[],
            router_per_expert_scale: &[],
            router_norm: &[],
            router_norm_parameter_free: false,
            router_input_scalar: 1.0,
            pre_experts_norm: &[],
            post_ffn1_norm: &[],
            post_experts_norm: &[],
            num_experts,
            top_k: ROUTE.len(),
            intermediate_size: INTER,
            router_bias: &[],
            experts_gate_up_bias: &self.gate_up_bias,
            experts_down_bias: &self.down_bias,
            gate_rule: MoeGateRule::Gated(Activation::Silu),
        }
    }
}

/// C1 — construction. Every descriptor's offsets equal the positions the
/// fixture placed the slices at; bias offsets follow the flat-bank
/// element layout; anti-symmetry holds (no two experts share an offset,
/// payload ≠ bias domains by construction).
#[test]
fn c1_descriptor_construction_reports_exact_placements() {
    let metal = get_metal();
    let bank = build_bank(&metal, NUM_EXPERTS);
    let moe = bank.moe(NUM_EXPERTS);

    let table = metal
        .build_expert_descriptor_table(&moe, INTER, HIDDEN)
        .expect("well-formed bank builds");

    assert_eq!(table.num_experts, NUM_EXPERTS);
    assert_eq!(table.gate_up_expert_bytes, GU_LEN);
    assert_eq!(table.down_expert_bytes, DN_LEN);
    for e in 0..NUM_EXPERTS {
        let d = &table.descs_host[e];
        assert_eq!(
            d.gate_up_payload_off as usize,
            placement(e, GU_LEN, GU_LEAD, GU_STAGGER),
            "expert {e} gate_up payload offset"
        );
        assert_eq!(
            d.down_payload_off as usize,
            placement(e, DN_LEN, DN_LEAD, DN_STAGGER),
            "expert {e} down payload offset"
        );
        assert_eq!(d.gate_up_scale_off, 0, "inline scales carry no offset");
        assert_eq!(d.down_scale_off, 0);
        assert_eq!(
            d.gate_up_bias_off as usize,
            e * 2 * INTER,
            "expert {e} gate_up bias element offset"
        );
        assert_eq!(d.down_bias_off as usize, e * HIDDEN);
    }
    // Anti-symmetry is a fixture property — verify it held, or the
    // assertions above prove less than they claim.
    let mut all_offsets: Vec<u32> = table
        .descs_host
        .iter()
        .flat_map(|d| [d.gate_up_payload_off, d.down_payload_off])
        .collect();
    all_offsets.sort_unstable();
    all_offsets.dedup();
    assert_eq!(
        all_offsets.len(),
        2 * NUM_EXPERTS,
        "fixture degenerated: payload offsets collide"
    );
}

/// C1b — split-scale construction. A `Paired` bank's e8m0 streams get
/// their own exact, anti-symmetric offsets in a distinct base buffer —
/// the independent-scale-offset fact the native MXFP4 representation
/// exists to state.
#[test]
fn c1b_split_scale_streams_carry_exact_independent_offsets() {
    let metal = get_metal();
    let bank = build_bank(&metal, NUM_EXPERTS);

    const SC_LEN: usize = 8;
    const SC_LEAD: usize = 128;
    const SC_STAGGER: usize = 4;
    let sc_size = placement(NUM_EXPERTS, SC_LEN, SC_LEAD, SC_STAGGER) + SC_LEN;
    // The down half's stagger is 2×, so size its span independently.
    let dn_span = placement(NUM_EXPERTS, SC_LEN, SC_LEAD / 2, SC_STAGGER * 2) + SC_LEN;
    let (sc_mem, sc_off) = aligned_backing(sc_size + dn_span);
    let sc_region = &sc_mem[sc_off..sc_off + sc_size + dn_span];
    assert!(metal.bufs().register_region(sc_region));

    let gu_scales: Vec<&[u8]> = (0..NUM_EXPERTS)
        .map(|e| {
            let p = placement(e, SC_LEN, SC_LEAD, SC_STAGGER);
            &sc_region[p..p + SC_LEN]
        })
        .collect();
    // Down streams in the region's second half — offsets unrelated to
    // gate_up's.
    let dn_scales: Vec<&[u8]> = (0..NUM_EXPERTS)
        .map(|e| {
            let p = sc_size + placement(e, SC_LEN, SC_LEAD / 2, SC_STAGGER * 2);
            &sc_region[p..p + SC_LEN]
        })
        .collect();

    let mut moe = bank.moe(NUM_EXPERTS);
    moe.expert_scales = MoeExpertScales::Paired {
        gate_up: gu_scales,
        down: dn_scales,
    };
    let table = metal
        .build_expert_descriptor_table(&moe, INTER, HIDDEN)
        .expect("well-formed split-scale bank builds");

    assert!(table.gate_up_scale_base.is_some());
    assert!(table.down_scale_base.is_some());
    for e in 0..NUM_EXPERTS {
        let d = &table.descs_host[e];
        assert_eq!(
            d.gate_up_scale_off as usize,
            placement(e, SC_LEN, SC_LEAD, SC_STAGGER),
            "expert {e} gate_up scale offset"
        );
        assert_eq!(
            d.down_scale_off as usize,
            sc_size + placement(e, SC_LEN, SC_LEAD / 2, SC_STAGGER * 2),
            "expert {e} down scale offset"
        );
        assert_ne!(
            d.gate_up_scale_off, d.gate_up_payload_off,
            "expert {e}: scale offset collides with payload offset — the \
             independent-stream fact is not being stated"
        );
    }
}

/// C2 — GPU lookup. The non-monotonic route gathers exactly the
/// descriptors of the routed experts, in slot order.
#[test]
fn c2_gather_resolves_non_monotonic_route_exactly() {
    let metal = get_metal();
    let bank = build_bank(&metal, NUM_EXPERTS);
    let moe = bank.moe(NUM_EXPERTS);
    let table = metal
        .build_expert_descriptor_table(&moe, INTER, HIDDEN)
        .expect("well-formed bank builds");

    let gate_half = (table.gate_up_expert_bytes / 2) as u32;
    let g = metal
        .descriptor_gather_roundtrip(&table, &ROUTE, gate_half)
        .expect("gather dispatch");
    let (gathered, gate0, gate1, down) = (g.descs, g.gate0, g.gate1, g.down);

    assert_eq!(gathered.len(), ROUTE.len());
    for (slot, &id) in ROUTE.iter().enumerate() {
        assert_eq!(
            gathered[slot], table.descs_host[id as usize],
            "slot {slot} must hold expert {id}'s descriptor verbatim"
        );
        // A slot-indexed (rather than id-indexed) lookup would land here:
        assert_ne!(
            gathered[slot], table.descs_host[slot],
            "slot {slot}: descriptor equals descs[slot] — kernel is \
             indexing by slot, not by selected expert id"
        );
    }
    // The expanded offset tables are the gathered descriptors restated in
    // the exact shape the grouped kernels bind — same facts, same slots.
    for (slot, &id) in ROUTE.iter().enumerate() {
        let d = &table.descs_host[id as usize];
        assert_eq!(gate0[slot], d.gate_up_payload_off, "slot {slot} gate0");
        assert_eq!(
            gate1[slot],
            d.gate_up_payload_off + gate_half,
            "slot {slot} gate1 states the ContiguousHalves fact"
        );
        assert_eq!(down[slot], d.down_payload_off, "slot {slot} down");
    }
    // Out-of-range id refuses rather than reads past the table.
    assert!(metal
        .descriptor_gather_roundtrip(&table, &[NUM_EXPERTS as u32], gate_half)
        .is_none());
}

/// C3 — bias lookup. Zero payload involvement: the staged rows are a
/// pure function of the route, through descriptors only. The asymmetric
/// bank makes every (expert, row, half) distinct, and the expected
/// values are derived through the SAME `FusedHalf` contract production's
/// CPU accessors use.
#[test]
fn c3_bias_staging_follows_selection_with_no_cpu_staging() {
    use larql_models::quant::mxfp4::FusedHalf;

    let metal = get_metal();
    let bank = build_bank(&metal, NUM_EXPERTS);
    let moe = bank.moe(NUM_EXPERTS);
    let table = metal
        .build_expert_descriptor_table(&moe, INTER, HIDDEN)
        .expect("well-formed bank builds");

    let (gate, up) = metal
        .bias_stage_roundtrip(&table, &ROUTE, INTER)
        .expect("bias stage dispatch");

    assert_eq!(gate.len(), ROUTE.len() * INTER);
    for (slot, &id) in ROUTE.iter().enumerate() {
        for j in 0..INTER {
            let expect_gate =
                bank.gate_up_bias[id as usize * 2 * INTER + FusedHalf::Gate.fused_row(j)];
            let expect_up = bank.gate_up_bias[id as usize * 2 * INTER + FusedHalf::Up.fused_row(j)];
            assert_eq!(
                gate[slot * INTER + j],
                expect_gate,
                "slot {slot} (expert {id}) gate bias row {j}"
            );
            assert_eq!(
                up[slot * INTER + j],
                expect_up,
                "slot {slot} (expert {id}) up bias row {j}"
            );
        }
    }

    // Selection alone determines the output: a different route stages
    // different rows, with no host writes anywhere between.
    let other: [u32; 4] = [3, 26, 11, 8];
    let (gate2, _) = metal
        .bias_stage_roundtrip(&table, &other, INTER)
        .expect("bias stage dispatch");
    assert_ne!(
        gate, gate2,
        "two disjoint routes staged identical bias rows — staging is not \
         following the selection"
    );
}

/// C4 — refusal is total. Each malformation yields `None` from
/// construction (or the lookup), never a partially valid table.
#[test]
fn c4_malformed_banks_refuse_completely() {
    let metal = get_metal();
    let bank = build_bank(&metal, NUM_EXPERTS);

    // Ragged payload slices: expert 3's gate_up one byte short.
    {
        let mut moe = bank.moe(NUM_EXPERTS);
        moe.experts_gate_up[3] = &bank.gu_slice(3)[..GU_LEN - 1];
        assert!(
            metal
                .build_expert_descriptor_table(&moe, INTER, HIDDEN)
                .is_none(),
            "ragged slice lengths must refuse"
        );
    }

    // Unregistered memory: a slice that lives outside any region.
    {
        let rogue = vec![0u8; GU_LEN];
        let mut moe = bank.moe(NUM_EXPERTS);
        moe.experts_gate_up[0] = &rogue;
        assert!(
            metal
                .build_expert_descriptor_table(&moe, INTER, HIDDEN)
                .is_none(),
            "unregistered slice must refuse"
        );
    }

    // Cross-buffer bank: expert 0's gate_up in a second registered
    // region — descriptors address ONE base per stream.
    {
        let (mem2, off2) = aligned_backing(GU_LEN * 2);
        let region2 = &mem2[off2..off2 + GU_LEN * 2];
        assert!(metal.bufs().register_region(region2));
        let mut moe = bank.moe(NUM_EXPERTS);
        moe.experts_gate_up[0] = &region2[..GU_LEN];
        assert!(
            metal
                .build_expert_descriptor_table(&moe, INTER, HIDDEN)
                .is_none(),
            "cross-buffer experts must refuse"
        );
    }

    // Bias bank contradicting the stated dims.
    {
        let short_bias: Vec<f32> = bank.gate_up_bias[..NUM_EXPERTS * 2 * INTER - 1].to_vec();
        let mut moe = bank.moe(NUM_EXPERTS);
        moe.experts_gate_up_bias = &short_bias;
        assert!(
            metal
                .build_expert_descriptor_table(&moe, INTER, HIDDEN)
                .is_none(),
            "bias length contradicting dims must refuse"
        );
    }

    // Split-scale bank with a missing stream: the exponent streams are
    // part of the representation — a table without them describes a
    // different bank.
    {
        let scales = [0u8; NUM_EXPERTS];
        let mut moe = bank.moe(NUM_EXPERTS);
        moe.expert_scales = MoeExpertScales::Paired {
            // One stream per expert for gate_up…
            gate_up: (0..NUM_EXPERTS).map(|e| &scales[e..e + 1]).collect(),
            // …but a truncated down table.
            down: (0..NUM_EXPERTS - 1).map(|e| &scales[e..e + 1]).collect(),
        };
        assert!(
            metal
                .build_expert_descriptor_table(&moe, INTER, HIDDEN)
                .is_none(),
            "missing scale streams must refuse"
        );
    }

    // Empty bank.
    {
        let mut moe = bank.moe(NUM_EXPERTS);
        moe.experts_gate_up = Vec::new();
        moe.experts_down = Vec::new();
        moe.num_experts = 0;
        assert!(metal
            .build_expert_descriptor_table(&moe, INTER, HIDDEN)
            .is_none());
    }
}
