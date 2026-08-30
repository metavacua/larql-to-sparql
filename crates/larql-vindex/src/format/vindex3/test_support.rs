//! Conformance fixture A as a real VINDEX3 container.
//!
//! ```text
//! hidden 256 · experts 8 · top_k 2 · shared 0 · programme gated-mlp-v1
//! ```
//!
//! Dimensions are the ones frozen for fixture A in the experiments programme
//! (§1.1 "A — Direct routed MoE (control)"). It is the control precisely
//! because nothing about it is interesting: residual-space experts, plain
//! softmax top-k, no shared bank, no transforms. Anything that fails here
//! fails for a container reason rather than an architecture one.
//!
//! Weights are a deterministic ramp, not random. The fixture's purpose is
//! byte-level parity between two source paths, so reproducibility matters and
//! realism does not.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use super::write::{ContainerSpec, SegmentSource};
use crate::format::lyrw2::bank::{BankDescriptor, BankKind};
use crate::format::lyrw2::browse_mode::BrowseMode;
use crate::format::lyrw2::plan::Lyrw2Plan;
use crate::format::lyrw2::region_format::{Packing, RegionFormat};
use crate::format::lyrw2::region_role::RegionRole;
use crate::format::lyrw2::region_schema::RegionSchema;
use crate::format::lyrw2::write::Lyrw2Writer;
use crate::format::moe_manifest::bank_ref::BankRef;
use crate::format::moe_manifest::{
    Combine, InputSpace, MoeLayer, MoeManifest, Reduction, Router, RouterPostProcessing,
    ScoreActivation, Selection, MOE_MANIFEST_SCHEMA_VERSION,
};

pub const FIXTURE_A_HIDDEN: u32 = 256;
pub const FIXTURE_A_INTERMEDIATE: u32 = 256;
pub const FIXTURE_A_EXPERTS: u32 = 8;
pub const FIXTURE_A_TOP_K: usize = 2;
pub const FIXTURE_A_LAYER: u32 = 0;
pub const FIXTURE_A_SEGMENT_KEY: &str = "routed/layer_000";
/// Segment key for the fused-FC1 rendering of the same weights.
pub const FIXTURE_A_FUSED_SEGMENT_KEY: &str = "routed/layer_000_fused";
pub const FIXTURE_A_PROGRAMME: &str = "gated-mlp-v1";
pub const FIXTURE_A_ROUTER_KEY: &str = "router.weight";
pub const FIXTURE_A_BANK_ID: u16 = 0;

/// Region schema ordinals within the bank, in write order.
///
/// `gated-mlp-v1` declares **both** `gate+up+down` and `gate_up_fused+down`
/// legal (`Programme::role_alternatives`), so one manifest can describe either
/// physical layout. That is what makes the fused/decomposed parity arm
/// meaningful: the logical operator contract is fixed by the programme id, and
/// the storage shape is free underneath it.
const SCHEMA_GATE: u16 = 0;
const SCHEMA_UP: u16 = 1;
const SCHEMA_DOWN: u16 = 2;
const FIXTURE_A_SCHEMA_COUNT: u16 = 3;
const FIXTURE_A_FUSED_SCHEMA_COUNT: u16 = 2;
/// Gate and up halves inside a fused FC1 region.
const FUSED_PROJECTION_HALVES: u32 = 2;

/// Ramp parameters. Small and irrational-ish so a transposition or an off-by-
/// one row shows up as a large difference rather than a plausible one.
const RAMP_SCALE: f32 = 0.013;
const RAMP_BIAS: f32 = -0.37;
const EXPERT_STRIDE: f32 = 0.101;

fn ramp(expert: u32, index: usize, len: usize) -> f32 {
    let t = index as f32 / len as f32;
    (t * RAMP_SCALE + RAMP_BIAS + expert as f32 * EXPERT_STRIDE).sin()
}

/// Deterministic `gate` weights for one expert, `[intermediate, hidden]`.
pub fn gate_f32(expert: u32) -> Vec<f32> {
    let len = (FIXTURE_A_INTERMEDIATE * FIXTURE_A_HIDDEN) as usize;
    (0..len).map(|i| ramp(expert, i, len)).collect()
}

/// Deterministic `up` weights for one expert, `[intermediate, hidden]`.
pub fn up_f32(expert: u32) -> Vec<f32> {
    let len = (FIXTURE_A_INTERMEDIATE * FIXTURE_A_HIDDEN) as usize;
    (0..len).map(|i| ramp(expert + 2, i, len)).collect()
}

/// Deterministic `down` bytes for one expert, f32 row-major.
pub fn down_f32(expert: u32) -> Vec<f32> {
    let len = (FIXTURE_A_HIDDEN * FIXTURE_A_INTERMEDIATE) as usize;
    (0..len).map(|i| ramp(expert + 1, i, len)).collect()
}

/// Router weight `[experts, hidden]`, f32 row-major, for an arbitrary
/// population — the axis V2-1 asks be demonstrably not hard-coded.
pub fn router_f32_for(experts: u32) -> Vec<f32> {
    let len = (experts * FIXTURE_A_HIDDEN) as usize;
    (0..len).map(|i| ramp(0, i, len)).collect()
}

/// Router weight for fixture A's frozen population.
pub fn router_f32() -> Vec<f32> {
    router_f32_for(FIXTURE_A_EXPERTS)
}

/// Gate rows followed by up rows — the layout `BoundProjection::Fused`
/// contracts for, and the only ordering `gate_up_fused` may mean.
pub fn gate_up_fused_f32(expert: u32) -> Vec<f32> {
    let mut v = gate_f32(expert);
    v.extend(up_f32(expert));
    v
}

fn as_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A uniquely-named scratch path. Callers run in parallel and this module
/// compiles into the library, so it cannot reach for `tempfile`.
fn staging_path() -> std::path::PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join("vindex3-fixture-a");
    std::fs::create_dir_all(&dir).expect("staging dir");
    dir.join(format!(
        "seg-{}-{}.lyrw",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

fn bank_of(experts: u32) -> BankDescriptor {
    BankDescriptor {
        bank_id: FIXTURE_A_BANK_ID,
        kind: BankKind::Routed,
        num_entries: experts,
        input_dim: FIXTURE_A_HIDDEN,
        intermediate_dim: FIXTURE_A_INTERMEDIATE,
        output_dim: FIXTURE_A_HIDDEN,
        region_schema_count: FIXTURE_A_SCHEMA_COUNT,
        browse: BrowseMode::Direct,
    }
}

fn schemas() -> Vec<RegionSchema> {
    vec![
        RegionSchema::unpaired(
            SCHEMA_GATE,
            RegionRole::Gate,
            RegionFormat::F32,
            Packing::RowMajor,
            FIXTURE_A_INTERMEDIATE,
            FIXTURE_A_HIDDEN,
        ),
        RegionSchema::unpaired(
            SCHEMA_UP,
            RegionRole::Up,
            RegionFormat::F32,
            Packing::RowMajor,
            FIXTURE_A_INTERMEDIATE,
            FIXTURE_A_HIDDEN,
        ),
        RegionSchema::unpaired(
            SCHEMA_DOWN,
            RegionRole::Down,
            RegionFormat::F32,
            Packing::RowMajor,
            FIXTURE_A_HIDDEN,
            FIXTURE_A_INTERMEDIATE,
        ),
    ]
}

/// Write fixture A's routed bank as one LYRW v2 segment, returning its bytes.
///
/// Staged through a uniquely-named file rather than a shared path: callers run
/// in parallel, and a fixed filename had them deleting each other's segment
/// mid-read. Uniqueness comes from pid plus a process-local counter — this
/// module compiles into the library, so it cannot reach for `tempfile`.
pub fn segment_bytes_for(experts: u32) -> Vec<u8> {
    let path = staging_path();

    let plan = Lyrw2Plan::single_segment(FIXTURE_A_LAYER, bank_of(experts), schemas());
    let mut writer = Lyrw2Writer::create(&path, plan).expect("create lyrw2 writer");
    for expert in 0..experts {
        writer
            .write_region(&as_bytes(&gate_f32(expert)))
            .expect("write gate");
        writer
            .write_region(&as_bytes(&up_f32(expert)))
            .expect("write up");
        writer
            .write_region(&as_bytes(&down_f32(expert)))
            .expect("write down");
    }
    writer.finish().expect("finish lyrw2 segment");

    let bytes = std::fs::read(&path).expect("read back segment");
    let _ = std::fs::remove_file(&path);
    bytes
}

fn fused_bank() -> BankDescriptor {
    BankDescriptor {
        region_schema_count: FIXTURE_A_FUSED_SCHEMA_COUNT,
        ..bank_of(FIXTURE_A_EXPERTS)
    }
}

fn fused_schemas() -> Vec<RegionSchema> {
    vec![
        // A fused operand must declare its row arrangement — `unpaired`
        // defaults to `Unspecified`, which is right for a legacy container
        // being read and refused for one being written.
        RegionSchema {
            layout: crate::format::lyrw2::region_layout::RegionLayout::ContiguousHalves,
            ..RegionSchema::unpaired(
                SCHEMA_GATE,
                RegionRole::GateUpFused,
                RegionFormat::F32,
                Packing::RowMajor,
                FUSED_PROJECTION_HALVES * FIXTURE_A_INTERMEDIATE,
                FIXTURE_A_HIDDEN,
            )
        },
        RegionSchema::unpaired(
            SCHEMA_UP,
            RegionRole::Down,
            RegionFormat::F32,
            Packing::RowMajor,
            FIXTURE_A_HIDDEN,
            FIXTURE_A_INTERMEDIATE,
        ),
    ]
}

/// The same weights as [`fixture_a_segment_bytes`], stored fused.
pub fn fixture_a_fused_segment_bytes() -> Vec<u8> {
    let path = staging_path();
    let plan = Lyrw2Plan::single_segment(FIXTURE_A_LAYER, fused_bank(), fused_schemas());
    let mut writer = Lyrw2Writer::create(&path, plan).expect("create lyrw2 writer");
    for expert in 0..FIXTURE_A_EXPERTS {
        writer
            .write_region(&as_bytes(&gate_up_fused_f32(expert)))
            .expect("write gate_up_fused");
        writer
            .write_region(&as_bytes(&down_f32(expert)))
            .expect("write down");
    }
    writer.finish().expect("finish fused segment");
    let bytes = std::fs::read(&path).expect("read back fused segment");
    let _ = std::fs::remove_file(&path);
    bytes
}

/// Fixture A rendered with fused FC1 storage, under the *same* programme id.
pub fn fixture_a_fused_spec() -> ContainerSpec {
    let mut spec = fixture_a_spec();
    spec.manifest.layers[0].routed_bank.storage = FIXTURE_A_FUSED_SEGMENT_KEY.to_string();
    spec.segments = vec![SegmentSource {
        key: FIXTURE_A_FUSED_SEGMENT_KEY.to_string(),
        bytes: fixture_a_fused_segment_bytes(),
    }];
    spec
}

fn manifest_for(experts: u32, top_k: usize, storage: &str) -> MoeManifest {
    MoeManifest {
        schema_version: MOE_MANIFEST_SCHEMA_VERSION,
        layers: vec![MoeLayer {
            layer: FIXTURE_A_LAYER,
            input_space: InputSpace::Residual,
            router: Router {
                scores: FIXTURE_A_ROUTER_KEY.to_string(),
                activation: ScoreActivation::Softmax,
                selection: Selection::TopK { k: top_k },
                post: RouterPostProcessing::default(),
                shared_expert_sink: false,
            },
            transforms: None,
            routed_bank: BankRef {
                experts,
                programme: FIXTURE_A_PROGRAMME.to_string(),
                storage: storage.to_string(),
                expert_dims: None,
            },
            shared_bank: None,
            reduction: Reduction::GateWeightedSum,
            routed_output_norm: None,
            combine: Combine::ResidualAdd,
        }],
    }
}

/// A routed-MoE container with an arbitrary population and top-k.
///
/// Exists to discharge V2-1's "expert counts and top-K are demonstrably not
/// hard-coded" row: the same code path builds an 8-expert top-2 container and
/// a 32-expert top-4 one, and the executor reads both out of their own
/// declarations rather than a constant.
pub fn routed_spec(experts: u32, top_k: usize, storage: &str) -> ContainerSpec {
    ContainerSpec {
        model: format!("routed-{experts}x{top_k}"),
        family: "direct-moe".to_string(),
        hidden_size: FIXTURE_A_HIDDEN as usize,
        num_layers: 1,
        manifest: manifest_for(experts, top_k, storage),
        segments: vec![SegmentSource {
            key: storage.to_string(),
            bytes: segment_bytes_for(experts),
        }],
    }
}

/// The complete fixture-A container specification (frozen dimensions).
pub fn fixture_a_spec() -> ContainerSpec {
    let mut spec = routed_spec(FIXTURE_A_EXPERTS, FIXTURE_A_TOP_K, FIXTURE_A_SEGMENT_KEY);
    spec.model = "fixture-a".to_string();
    spec
}

/// A container whose declared segment holds `bytes` verbatim.
///
/// Lets a test put a deliberately unusable segment behind a well-formed index
/// and manifest — the shape `verify` exists to catch, and one `open` cannot,
/// since it reads segment files without parsing them.
pub fn spec_with_segment_bytes(bytes: Vec<u8>) -> ContainerSpec {
    let mut spec = fixture_a_spec();
    spec.segments = vec![SegmentSource {
        key: FIXTURE_A_SEGMENT_KEY.to_string(),
        bytes,
    }];
    spec
}

/// A routed bank carrying gate and up but **no down** — well-formed LYRW v2
/// that no gated-MLP programme can be satisfied by.
pub fn gate_only_segment_bytes() -> Vec<u8> {
    let path = staging_path();
    let bank = BankDescriptor {
        region_schema_count: 2,
        ..bank_of(FIXTURE_A_EXPERTS)
    };
    let schemas = vec![
        RegionSchema::unpaired(
            SCHEMA_GATE,
            RegionRole::Gate,
            RegionFormat::F32,
            Packing::RowMajor,
            FIXTURE_A_INTERMEDIATE,
            FIXTURE_A_HIDDEN,
        ),
        RegionSchema::unpaired(
            SCHEMA_UP,
            RegionRole::Up,
            RegionFormat::F32,
            Packing::RowMajor,
            FIXTURE_A_INTERMEDIATE,
            FIXTURE_A_HIDDEN,
        ),
    ];
    let plan = Lyrw2Plan::single_segment(FIXTURE_A_LAYER, bank, schemas);
    let mut writer = Lyrw2Writer::create(&path, plan).expect("create writer");
    for expert in 0..FIXTURE_A_EXPERTS {
        writer
            .write_region(&as_bytes(&gate_f32(expert)))
            .expect("gate");
        writer.write_region(&as_bytes(&up_f32(expert))).expect("up");
    }
    writer.finish().expect("finish");
    let bytes = std::fs::read(&path).expect("read back");
    let _ = std::fs::remove_file(&path);
    bytes
}

/// Fixture-A weights rendered as the frozen fixture-A segment.
pub fn fixture_a_segment_bytes() -> Vec<u8> {
    segment_bytes_for(FIXTURE_A_EXPERTS)
}

/// Segment keys the spec declares — handy for asserting the index covers them.
pub fn fixture_a_segment_keys() -> BTreeMap<String, u32> {
    BTreeMap::from([(FIXTURE_A_SEGMENT_KEY.to_string(), 1)])
}
