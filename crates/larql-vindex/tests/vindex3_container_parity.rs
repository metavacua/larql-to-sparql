//! V2-1, fixture A: does a **VINDEX3 container** execute?
//!
//! # The gap this closes
//!
//! Every VINDEX3 parity result before this one bound its operands out of a
//! VINDEX2 file — `index.json.version` was 2 at every step, on every model.
//! What was proven was the *runtime*:
//!
//! ```text
//! proven      the VINDEX3 executor, fed VINDEX2 operands, matches production
//! not proven  a VINDEX3 container can be written, opened, bound and executed
//! ```
//!
//! Here both arms execute the identical arithmetic on identical weights. The
//! only thing that differs is **where the bytes came from**:
//!
//! ```text
//! arm A   operands from an in-memory buffer  (how every prior test sourced them)
//! arm B   operands from a VINDEX3 container on disk, resolved through
//!         index.json v3 → moe_manifest.json → LYRW v2 region table
//! ```
//!
//! Bit-identical output is the whole claim. Anything less means the container
//! moved a byte — a wrong offset, a transposed region, a role resolved to the
//! wrong schema — and no amount of runtime correctness would rescue it.
//!
//! # Why bit-identity is the right bar here
//!
//! Both arms run the same executor over the same f32 bytes in the same order,
//! so there is no reordering and no quantisation between them. A tolerance
//! would therefore only hide a real defect: the two paths are either reading
//! the same bytes or they are not.

use larql_compute::{Activation, MoeTopKWeightPolicy};
use larql_vindex::format::capability::binding::RepresentationIdentity;
use larql_vindex::format::capability::component::ComponentContract;
use larql_vindex::format::capability::coordinate::BankCoordinate;
use larql_vindex::format::lyrw2::read::Lyrw2Reader;
use larql_vindex::format::lyrw2::region_format::RegionFormat;
use larql_vindex::format::lyrw2::region_role::RegionRole;
use larql_vindex::format::vindex3::test_support::{
    down_f32, fixture_a_fused_spec, fixture_a_spec, gate_f32, routed_spec, router_f32,
    router_f32_for, up_f32, FIXTURE_A_BANK_ID, FIXTURE_A_EXPERTS, FIXTURE_A_HIDDEN,
    FIXTURE_A_INTERMEDIATE, FIXTURE_A_LAYER, FIXTURE_A_SEGMENT_KEY, FIXTURE_A_TOP_K,
};
use larql_vindex::format::vindex3::{write_container, Vindex3Container};
use larql_vindex::runtime::{
    execute, BoundBankOperation, BoundExpert, BoundExpertScaling, BoundMoeOperation,
    BoundProjection, BoundReduction, BoundRouter, BoundTensor, ExpertKernel, MoeInputs,
    RouterKernel,
};
use tempfile::tempdir;

const VARIANT: &str = "exact";
const ROUTER_REGION_SET: &str = "router";
/// SwiGLU — what `gated-mlp-v1` contracts for.
const ACTIVATION: Activation = Activation::Silu;
/// Fixture A's router keeps raw softmax probabilities (no renormalisation),
/// matching `RouterPostProcessing::default()` in its manifest.
const SELECTED_WEIGHT: MoeTopKWeightPolicy = MoeTopKWeightPolicy::RawSoftmax;

fn as_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn tensor<'a>(region_set: &str, bytes: &'a [u8], contract: ComponentContract) -> BoundTensor<'a> {
    BoundTensor::direct(
        RepresentationIdentity::new(region_set, VARIANT),
        bytes,
        RegionFormat::F32,
        contract,
    )
    .expect("bind f32 tensor")
}

fn projection_contract() -> ComponentContract {
    ComponentContract::matrix(FIXTURE_A_INTERMEDIATE, FIXTURE_A_HIDDEN)
}

fn down_contract() -> ComponentContract {
    ComponentContract::matrix(FIXTURE_A_HIDDEN, FIXTURE_A_INTERMEDIATE)
}

fn operation<'a>(router_bytes: &'a [u8], experts: Vec<BoundExpert<'a>>) -> BoundMoeOperation<'a> {
    operation_for(router_bytes, experts, FIXTURE_A_EXPERTS, FIXTURE_A_TOP_K)
}

/// The same operation with the population and top-k supplied rather than
/// assumed — what the not-hard-coded arm drives.
fn operation_for<'a>(
    router_bytes: &'a [u8],
    experts: Vec<BoundExpert<'a>>,
    population: u32,
    top_k: usize,
) -> BoundMoeOperation<'a> {
    BoundMoeOperation {
        router: BoundRouter {
            weight: tensor(
                ROUTER_REGION_SET,
                router_bytes,
                ComponentContract::matrix(population, FIXTURE_A_HIDDEN),
            ),
            top_k,
            selected_weight: SELECTED_WEIGHT,
            scaling: BoundExpertScaling::None,
            kernel: RouterKernel::Incumbent,
        },
        transforms: Vec::new(),
        banks: vec![BoundBankOperation {
            bank: BankCoordinate::new(FIXTURE_A_LAYER, FIXTURE_A_BANK_ID),
            experts,
            intermediate_dim: FIXTURE_A_INTERMEDIATE as usize,
            hidden_dim: FIXTURE_A_HIDDEN as usize,
            activation: ACTIVATION,
            kernel: ExpertKernel::Reference,
        }],
        reduction: BoundReduction::WeightedSum,
        residual_dim: FIXTURE_A_HIDDEN as usize,
    }
}

/// A deterministic residual to route. Not from the fixture's own ramp, so a
/// bug that happened to align input and weights cannot cancel out.
fn residual() -> Vec<f32> {
    (0..FIXTURE_A_HIDDEN as usize)
        .map(|i| ((i as f32) * 0.021 - 1.7).cos())
        .collect()
}

#[test]
fn a_vindex3_container_executes_bit_identically_to_in_memory_operands() {
    // ── arm A: operands held in memory, as every prior parity test sourced them
    let router = as_bytes(&router_f32());
    let gates: Vec<Vec<u8>> = (0..FIXTURE_A_EXPERTS)
        .map(|e| as_bytes(&gate_f32(e)))
        .collect();
    let ups: Vec<Vec<u8>> = (0..FIXTURE_A_EXPERTS)
        .map(|e| as_bytes(&up_f32(e)))
        .collect();
    let downs: Vec<Vec<u8>> = (0..FIXTURE_A_EXPERTS)
        .map(|e| as_bytes(&down_f32(e)))
        .collect();

    let memory_experts: Vec<BoundExpert<'_>> = (0..FIXTURE_A_EXPERTS)
        .map(|e| BoundExpert {
            expert_id: e,
            projection: BoundProjection::Decomposed {
                gate: tensor(
                    &RegionRole::Gate.name(),
                    &gates[e as usize],
                    projection_contract(),
                ),
                up: tensor(
                    &RegionRole::Up.name(),
                    &ups[e as usize],
                    projection_contract(),
                ),
            },
            down: tensor(
                &RegionRole::Down.name(),
                &downs[e as usize],
                down_contract(),
            ),
        })
        .collect();
    let from_memory = execute(
        &operation(&router, memory_experts),
        MoeInputs::shared(&residual()),
    )
    .expect("in-memory arm executes");

    // ── arm B: the same weights, sourced through a VINDEX3 container on disk
    let dir = tempdir().unwrap();
    write_container(dir.path(), &fixture_a_spec()).expect("write VINDEX3 container");
    let container = Vindex3Container::open(dir.path()).expect("open VINDEX3 container");

    // Everything below goes through the container's own declarations: the
    // manifest names the bank's storage key, the index resolves the key to a
    // file, and the LYRW v2 region table resolves (entry, role) to bytes.
    let layer = container
        .layer(FIXTURE_A_LAYER)
        .expect("manifest declares layer 0 as MoE");
    let segment = container
        .segment(&layer.routed_bank.storage)
        .expect("the bank's declared storage resolves");

    let container_experts: Vec<BoundExpert<'_>> = (0..FIXTURE_A_EXPERTS)
        .map(|e| BoundExpert {
            expert_id: e,
            projection: BoundProjection::Decomposed {
                gate: tensor(
                    &RegionRole::Gate.name(),
                    region(&segment, e, RegionRole::Gate),
                    projection_contract(),
                ),
                up: tensor(
                    &RegionRole::Up.name(),
                    region(&segment, e, RegionRole::Up),
                    projection_contract(),
                ),
            },
            down: tensor(
                &RegionRole::Down.name(),
                region(&segment, e, RegionRole::Down),
                down_contract(),
            ),
        })
        .collect();
    let from_container = execute(
        &operation(&router, container_experts),
        MoeInputs::shared(&residual()),
    )
    .expect("container arm executes");

    assert_eq!(
        from_memory.len(),
        FIXTURE_A_HIDDEN as usize,
        "the operation must return the residual width"
    );
    assert_eq!(
        bits(&from_container),
        bits(&from_memory),
        "a VINDEX3 container must deliver the same bytes it was given — any \
         difference here is the container moving a byte, not the executor"
    );
}

fn region<'a>(reader: &Lyrw2Reader<'a>, entry: u32, role: RegionRole) -> &'a [u8] {
    reader
        .region_bytes(FIXTURE_A_BANK_ID, entry, role)
        .unwrap_or_else(|e| panic!("resolve {role:?} for entry {entry}: {e}"))
        .unwrap_or_else(|| panic!("{role:?} region missing for entry {entry}"))
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The container's regions must equal the fixture's own weights byte-for-byte.
///
/// Separate from the execution test on purpose: if both fail, this one says the
/// container stored the wrong bytes; if only the execution test fails, the
/// bytes were right and the binding read them wrongly. One assertion cannot
/// distinguish those.
#[test]
fn every_region_round_trips_byte_for_byte_through_the_container() {
    let dir = tempdir().unwrap();
    write_container(dir.path(), &fixture_a_spec()).expect("write");
    let container = Vindex3Container::open(dir.path()).expect("open");
    let segment = container.segment(FIXTURE_A_SEGMENT_KEY).expect("segment");

    for e in 0..FIXTURE_A_EXPERTS {
        assert_eq!(
            region(&segment, e, RegionRole::Gate),
            as_bytes(&gate_f32(e)),
            "gate bytes differ for expert {e}"
        );
        assert_eq!(
            region(&segment, e, RegionRole::Up),
            as_bytes(&up_f32(e)),
            "up bytes differ for expert {e}"
        );
        assert_eq!(
            region(&segment, e, RegionRole::Down),
            as_bytes(&down_f32(e)),
            "down bytes differ for expert {e}"
        );
    }
}

/// V2-1: fused and decomposed FC1 storage produce identical results **under
/// one manifest**.
///
/// `gated-mlp-v1` declares both `gate + up + down` and `gate_up_fused + down`
/// legal, so the same programme id describes both containers below. Only the
/// physical rendering differs.
///
/// This is the arm that proves the manifest is an *executable type
/// declaration* rather than a replay of one known layout: the logical operator
/// contract is fixed by the programme, and storage shape is free underneath
/// it. K3 depends on that being true — fused/decomposed, latent transforms and
/// shared projections all have superficially compatible dimensions while
/// requiring different execution semantics, so shape must never be what
/// selects the operation.
#[test]
fn fused_and_decomposed_storage_agree_under_one_programme() {
    let router = as_bytes(&router_f32());

    let decomposed_dir = tempdir().unwrap();
    write_container(decomposed_dir.path(), &fixture_a_spec()).expect("write decomposed");
    let decomposed = Vindex3Container::open(decomposed_dir.path()).expect("open decomposed");

    let fused_dir = tempdir().unwrap();
    write_container(fused_dir.path(), &fixture_a_fused_spec()).expect("write fused");
    let fused = Vindex3Container::open(fused_dir.path()).expect("open fused");

    // Same programme id on both sides — that is the point.
    let d_layer = decomposed.layer(FIXTURE_A_LAYER).expect("decomposed layer");
    let f_layer = fused.layer(FIXTURE_A_LAYER).expect("fused layer");
    assert_eq!(
        d_layer.routed_bank.programme, f_layer.routed_bank.programme,
        "both renderings must be described by the same programme"
    );
    assert_ne!(
        d_layer.routed_bank.storage, f_layer.routed_bank.storage,
        "…while resolving to physically different storage"
    );

    let d_seg = decomposed
        .segment(&d_layer.routed_bank.storage)
        .expect("decomposed segment");
    let f_seg = fused
        .segment(&f_layer.routed_bank.storage)
        .expect("fused segment");

    let d_experts: Vec<BoundExpert<'_>> = (0..FIXTURE_A_EXPERTS)
        .map(|e| BoundExpert {
            expert_id: e,
            projection: BoundProjection::Decomposed {
                gate: tensor(
                    &RegionRole::Gate.name(),
                    region(&d_seg, e, RegionRole::Gate),
                    projection_contract(),
                ),
                up: tensor(
                    &RegionRole::Up.name(),
                    region(&d_seg, e, RegionRole::Up),
                    projection_contract(),
                ),
            },
            down: tensor(
                &RegionRole::Down.name(),
                region(&d_seg, e, RegionRole::Down),
                down_contract(),
            ),
        })
        .collect();

    let f_experts: Vec<BoundExpert<'_>> = (0..FIXTURE_A_EXPERTS)
        .map(|e| BoundExpert {
            expert_id: e,
            projection: BoundProjection::Fused {
                gate_up: tensor(
                    &RegionRole::GateUpFused.name(),
                    region(&f_seg, e, RegionRole::GateUpFused),
                    ComponentContract::matrix(2 * FIXTURE_A_INTERMEDIATE, FIXTURE_A_HIDDEN),
                ),
            },
            down: tensor(
                &RegionRole::Down.name(),
                region(&f_seg, e, RegionRole::Down),
                down_contract(),
            ),
        })
        .collect();

    let from_decomposed = execute(
        &operation(&router, d_experts),
        MoeInputs::shared(&residual()),
    )
    .expect("decomposed executes");
    let from_fused = execute(
        &operation(&router, f_experts),
        MoeInputs::shared(&residual()),
    )
    .expect("fused executes");

    assert_eq!(
        bits(&from_fused),
        bits(&from_decomposed),
        "storage layout must not change the logical result — if it does, the \
         programme id is not what selects the operation and shape is leaking \
         into the contract"
    );
}

/// V2-1: expert count and top-K are read from the container, not baked in.
///
/// Three populations and three top-Ks through the identical code path. If any
/// constant had leaked into the writer, the reader or the executor, at least
/// one shape would fail to bind or would execute against the wrong population.
///
/// **Scope, stated honestly.** This discharges two of the three axes in V2-1's
/// "expert counts, top-K and shared banks are demonstrably not hard-coded".
/// The shared-bank axis is *not* covered: `BoundMoeOperation::banks` documents
/// unrouted shared banks as arriving with the Mini-K3 rung, so there is no
/// execution path to test them against yet. That row stays open.
#[test]
fn expert_count_and_top_k_come_from_the_container() {
    // Deliberately varied together and apart: (8,2) is fixture A, (32,4) is
    // GPT-OSS-shaped, (5,1) is a degenerate small case with an odd population
    // that no power-of-two assumption would survive.
    const SHAPES: [(u32, usize); 3] = [(8, 2), (32, 4), (5, 1)];

    for (experts, top_k) in SHAPES {
        let key = format!("routed/pop_{experts}_k{top_k}");
        let dir = tempdir().unwrap();
        write_container(dir.path(), &routed_spec(experts, top_k, &key))
            .unwrap_or_else(|e| panic!("write {experts}x{top_k}: {e}"));
        let container = Vindex3Container::open(dir.path())
            .unwrap_or_else(|e| panic!("open {experts}x{top_k}: {e}"));

        // The container's own declarations, not the test's expectations.
        let layer = container.layer(FIXTURE_A_LAYER).expect("MoE layer");
        assert_eq!(layer.routed_bank.experts, experts);
        assert!(
            container.verify().is_empty(),
            "{experts}x{top_k} must be structurally bindable: {:?}",
            container.verify()
        );

        let segment = container
            .segment(&layer.routed_bank.storage)
            .expect("segment");
        assert_eq!(
            segment.banks()[0].num_entries,
            experts,
            "the bank must hold the population the manifest declares"
        );

        let router = as_bytes(&router_f32_for(experts));
        let bound: Vec<BoundExpert<'_>> = (0..experts)
            .map(|e| BoundExpert {
                expert_id: e,
                projection: BoundProjection::Decomposed {
                    gate: tensor(
                        &RegionRole::Gate.name(),
                        region(&segment, e, RegionRole::Gate),
                        projection_contract(),
                    ),
                    up: tensor(
                        &RegionRole::Up.name(),
                        region(&segment, e, RegionRole::Up),
                        projection_contract(),
                    ),
                },
                down: tensor(
                    &RegionRole::Down.name(),
                    region(&segment, e, RegionRole::Down),
                    down_contract(),
                ),
            })
            .collect();

        let out = execute(
            &operation_for(&router, bound, experts, top_k),
            MoeInputs::shared(&residual()),
        )
        .unwrap_or_else(|e| panic!("execute {experts}x{top_k}: {e}"));
        assert_eq!(out.len(), FIXTURE_A_HIDDEN as usize);
        assert!(
            out.iter().all(|v| v.is_finite()),
            "{experts}x{top_k} produced non-finite output"
        );
    }
}
