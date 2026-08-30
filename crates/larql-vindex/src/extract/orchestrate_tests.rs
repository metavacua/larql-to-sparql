//! Orchestration tests: the target is chosen once, above the writers.
//!
//! The properties that matter are about *routing*, not about bytes — the
//! adapter's own tests already prove byte fidelity. What is proven here:
//!
//! ```text
//! Native + capable      → a container exists, with every routed layer
//! Native + incapable    → error, never a quiet LegacyKQuant
//! Auto                  → legacy, even from a capable source
//! Legacy                → legacy, and writes no container
//! ```

use std::collections::HashMap;

use larql_models::quant::mxfp4::{MXFP4_GROUP_BYTES, MXFP4_GROUP_ELEMS};

use super::*;
use crate::extract::native_moe::NativeMoeLayer;
use crate::extract::target::{ExtractionRequest, ExtractionTarget};
use crate::format::vindex3::import::routed_storage_key;
use crate::format::vindex3::Vindex3Container;
use crate::format::weights::write_f32::WeightSource;

const EXPERTS: usize = 3;
const HIDDEN: usize = 64;
const INTERMEDIATE: usize = 96;
const TOP_K: usize = 2;
const LAYERS: usize = 3;

const GU_OUT: usize = INTERMEDIATE * 2;
const GU_GROUPS: usize = HIDDEN / MXFP4_GROUP_ELEMS;
const DN_OUT: usize = HIDDEN;
const DN_GROUPS: usize = INTERMEDIATE / MXFP4_GROUP_ELEMS;
const GU_PAYLOAD: usize = GU_OUT * GU_GROUPS * MXFP4_GROUP_BYTES;
const GU_SCALE: usize = GU_OUT * GU_GROUPS;
const DN_PAYLOAD: usize = DN_OUT * DN_GROUPS * MXFP4_GROUP_BYTES;
const DN_SCALE: usize = DN_OUT * DN_GROUPS;

fn byte_at(layer: u32, expert: usize, stream_id: u8, i: usize) -> u8 {
    (layer as usize * 211 + expert * 37 + stream_id as usize * 101 + i) as u8
}

fn stream(layer: u32, stream_id: u8, per_expert: usize) -> Vec<u8> {
    (0..EXPERTS)
        .flat_map(|e| (0..per_expert).map(move |i| byte_at(layer, e, stream_id, i)))
        .collect()
}

struct TestSource {
    arch: Box<dyn larql_models::ModelArchitecture>,
    raw: HashMap<String, (Vec<u8>, Vec<usize>)>,
    layers: usize,
}

impl WeightSource for TestSource {
    fn get_tensor(&self, _key: &str) -> Option<(Vec<f32>, usize, usize)> {
        None
    }
    fn get_vector(&self, _key: &str) -> Option<Vec<f32>> {
        None
    }
    fn arch(&self) -> &dyn larql_models::ModelArchitecture {
        &*self.arch
    }
    fn num_layers(&self) -> usize {
        self.layers
    }
    fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)> {
        None
    }
    fn vector_names(&self) -> Vec<String> {
        Vec::new()
    }
    fn get_packed_bf16(&self, _key: &str) -> Option<Vec<u8>> {
        None
    }
    fn get_raw_u8(&self, key: &str) -> Option<(Vec<u8>, Vec<usize>)> {
        self.raw.get(key).cloned()
    }
}

fn arch_json(model_type: &str) -> serde_json::Value {
    serde_json::json!({
        "model_type": model_type,
        "num_hidden_layers": LAYERS,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 128,
        "num_local_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
        "rope_theta": 150000.0,
    })
}

/// A packed-MXFP4 source with every stream present.
fn capable_source() -> TestSource {
    let arch = larql_models::detect_from_json(&arch_json("gpt_oss"));
    let mut raw = HashMap::new();
    for layer in 0..LAYERS as u32 {
        let l = layer as usize;
        raw.insert(
            arch.packed_gate_up_blocks_key(l).unwrap(),
            (
                stream(layer, 0, GU_PAYLOAD),
                vec![EXPERTS, GU_OUT, GU_GROUPS, MXFP4_GROUP_BYTES],
            ),
        );
        raw.insert(
            arch.packed_gate_up_scales_key(l).unwrap(),
            (stream(layer, 1, GU_SCALE), vec![EXPERTS, GU_OUT, GU_GROUPS]),
        );
        raw.insert(
            arch.packed_down_blocks_key(l).unwrap(),
            (
                stream(layer, 2, DN_PAYLOAD),
                vec![EXPERTS, DN_OUT, DN_GROUPS, MXFP4_GROUP_BYTES],
            ),
        );
        raw.insert(
            arch.packed_down_scales_key(l).unwrap(),
            (stream(layer, 3, DN_SCALE), vec![EXPERTS, DN_OUT, DN_GROUPS]),
        );
    }
    TestSource {
        arch,
        raw,
        layers: LAYERS,
    }
}

/// Packed-MXFP4 by architecture, but the source serves no raw bytes — a
/// dequantised view.
fn dequantised_source() -> TestSource {
    TestSource {
        arch: larql_models::detect_from_json(&arch_json("gpt_oss")),
        raw: HashMap::new(),
        layers: LAYERS,
    }
}

fn non_moe_source() -> TestSource {
    TestSource {
        arch: larql_models::detect_from_json(&serde_json::json!({
            "model_type": "llama",
            "num_hidden_layers": LAYERS,
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 128,
        })),
        raw: HashMap::new(),
        layers: LAYERS,
    }
}

// ── inspection reports this orchestrator's reach ─────────────────────────

#[test]
fn a_capable_source_reports_reachable_scale_streams() {
    let caps = inspect_capabilities(&capable_source()).unwrap();
    assert!(caps.split_scale_streams);
    assert!(caps.scale_streams_available);
    assert!(caps.gate_up_layout.is_declared());
    assert!(caps.region_format.is_some());
}

/// A non-MoE model has no native route here, and saying so is not an error.
#[test]
fn a_non_moe_source_reports_no_region_format() {
    let caps = inspect_capabilities(&non_moe_source()).unwrap();
    assert!(caps.region_format.is_none());
    assert!(!caps.split_scale_streams);
}

/// The distinction the capability split exists for: the architecture says
/// packed-MXFP4, but *this source* cannot serve raw bytes. Inspection must
/// surface that rather than claim availability from the format class.
#[test]
fn a_dequantised_source_is_not_natively_extractable() {
    let err = inspect_capabilities(&dequantised_source())
        .expect_err("a declared key the source cannot serve is a refusal");
    assert!(err.to_string().contains("raw bytes"), "{err}");
}

// ── routing ──────────────────────────────────────────────────────────────

#[test]
fn native_on_a_capable_source_writes_a_container_with_every_routed_layer() {
    let src = capable_source();
    let dir = tempfile::tempdir().unwrap();
    let outcome = extract_expert_banks(
        &src,
        dir.path(),
        ExtractionRequest::Native,
        "gpt-oss-fixture",
    )
    .unwrap();

    assert_eq!(
        outcome,
        ExpertBankOutcome::NativeContainer { layers: LAYERS }
    );
    assert_eq!(outcome.target(), ExtractionTarget::NativeV3);

    // The container is real and complete, not merely returned.
    let container = Vindex3Container::open(dir.path()).unwrap();
    assert!(container.verify().is_empty(), "{:?}", container.verify());
    for layer in 0..LAYERS as u32 {
        assert!(
            container.segment(&routed_storage_key(layer)).is_ok(),
            "layer {layer} missing from the container"
        );
    }
}

/// `Auto` stays on the legacy route **even from a capable source**, so no
/// existing caller is opted in while native is unqualified. It must also
/// leave no container behind.
#[test]
fn auto_stays_legacy_even_when_native_would_succeed() {
    let src = capable_source();
    // The same source really can go native.
    let dir_native = tempfile::tempdir().unwrap();
    assert!(matches!(
        extract_expert_banks(&src, dir_native.path(), ExtractionRequest::Native, "m").unwrap(),
        ExpertBankOutcome::NativeContainer { .. }
    ));

    let dir_auto = tempfile::tempdir().unwrap();
    let outcome =
        extract_expert_banks(&src, dir_auto.path(), ExtractionRequest::Auto, "m").unwrap();
    assert_eq!(outcome, ExpertBankOutcome::LegacyInline);
    assert_eq!(outcome.target(), ExtractionTarget::LegacyKQuant);
    assert!(
        Vindex3Container::open(dir_auto.path()).is_err(),
        "Auto must not leave a container behind"
    );
}

/// The legacy route writes no second artifact: the k-quant writer emits
/// expert banks inline as part of the model weights.
#[test]
fn legacy_writes_no_container() {
    let src = capable_source();
    let dir = tempfile::tempdir().unwrap();
    let outcome = extract_expert_banks(&src, dir.path(), ExtractionRequest::Legacy, "m").unwrap();
    assert_eq!(outcome, ExpertBankOutcome::LegacyInline);
    assert!(Vindex3Container::open(dir.path()).is_err());
}

/// The load-bearing property: an unsatisfiable `Native` request errors and
/// is never spelled as the other target. A caller that received
/// `LegacyInline` here would go on to compare the Q6_K control against
/// itself.
#[test]
fn native_on_an_incapable_source_errors_rather_than_downgrading() {
    let src = non_moe_source();
    let dir = tempfile::tempdir().unwrap();
    let result = extract_expert_banks(&src, dir.path(), ExtractionRequest::Native, "m");

    let err = result.expect_err("a non-MoE source cannot satisfy a native request");
    let msg = err.to_string();
    assert!(msg.contains("region format"), "{msg}");
    assert!(msg.contains("legacy"), "{msg}");
    assert!(
        Vindex3Container::open(dir.path()).is_err(),
        "a refused native request must leave no partial container"
    );
}

/// `Auto` never errors, whatever the source can do — it made no demand to
/// be refused. This is what keeps "told no" and "never asked" distinct at
/// the call site.
#[test]
fn auto_succeeds_on_a_source_that_could_not_go_native() {
    let src = non_moe_source();
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(
        extract_expert_banks(&src, dir.path(), ExtractionRequest::Auto, "m").unwrap(),
        ExpertBankOutcome::LegacyInline
    );
}

// ── hybrid stacks: dense layers are skipped, not refused ─────────────────

/// An architecture that declares packed experts for only some layers — a
/// routed/dense hybrid. Only `family` and `config` are required by the
/// trait, so this stays a stub rather than a second implementation.
struct HybridArch {
    inner: Box<dyn larql_models::ModelArchitecture>,
    routed: Vec<usize>,
}

impl HybridArch {
    fn new(routed: Vec<usize>) -> Self {
        Self {
            inner: larql_models::detect_from_json(&arch_json("gpt_oss")),
            routed,
        }
    }
    fn routes(&self, layer: usize) -> bool {
        self.routed.contains(&layer)
    }
}

impl larql_models::ModelArchitecture for HybridArch {
    fn family(&self) -> &str {
        self.inner.family()
    }
    fn config(&self) -> &larql_models::ModelConfig {
        self.inner.config()
    }
    fn expert_format(&self) -> larql_models::ExpertFormat {
        larql_models::ExpertFormat::PackedMxfp4
    }
    // Delegated, not defaulted: the adapter reads these to describe the
    // operand, and the trait's `0` defaults would produce a source that
    // fails its own validity check for reasons unrelated to the routing
    // this stub exists to exercise.
    fn is_moe(&self) -> bool {
        self.inner.is_moe()
    }
    fn num_experts(&self) -> usize {
        self.inner.num_experts()
    }
    fn num_experts_per_token(&self) -> usize {
        self.inner.num_experts_per_token()
    }
    fn moe_intermediate_size(&self) -> usize {
        self.inner.moe_intermediate_size()
    }
    fn packed_gate_up_blocks_key(&self, layer: usize) -> Option<String> {
        self.routes(layer)
            .then(|| self.inner.packed_gate_up_blocks_key(layer))
            .flatten()
    }
    fn packed_gate_up_scales_key(&self, layer: usize) -> Option<String> {
        self.inner.packed_gate_up_scales_key(layer)
    }
    fn packed_down_blocks_key(&self, layer: usize) -> Option<String> {
        self.inner.packed_down_blocks_key(layer)
    }
    fn packed_down_scales_key(&self, layer: usize) -> Option<String> {
        self.inner.packed_down_scales_key(layer)
    }
}

fn hybrid_source(routed: Vec<usize>) -> TestSource {
    let mut src = capable_source();
    src.arch = Box::new(HybridArch::new(routed));
    src
}

/// A dense layer inside a hybrid stack is skipped, and the container holds
/// exactly the routed ones. Refusing here would make every hybrid model
/// unextractable natively.
#[test]
fn dense_layers_are_skipped_and_only_routed_banks_are_written() {
    let src = hybrid_source(vec![0, 2]);
    let dir = tempfile::tempdir().unwrap();
    let outcome =
        extract_expert_banks(&src, dir.path(), ExtractionRequest::Native, "hybrid").unwrap();

    assert_eq!(outcome, ExpertBankOutcome::NativeContainer { layers: 2 });

    let container = Vindex3Container::open(dir.path()).unwrap();
    assert!(container.segment(&routed_storage_key(0)).is_ok());
    assert!(
        container.segment(&routed_storage_key(1)).is_err(),
        "layer 1 is dense and owes the container nothing"
    );
    assert!(container.segment(&routed_storage_key(2)).is_ok());
}

/// Capability inspection keeps looking past a dense layer rather than
/// concluding at layer 0 that the model has no native banks.
#[test]
fn inspection_looks_past_dense_layers() {
    let caps = inspect_capabilities(&hybrid_source(vec![2])).unwrap();
    assert!(
        caps.scale_streams_available,
        "layer 2 routes, so the source is natively extractable"
    );
}

/// A model that declares packed experts but has no routed layer at all is
/// not natively extractable — and that is a refusal to go native, not an
/// error during inspection.
#[test]
fn a_model_with_no_routed_layer_is_not_natively_extractable() {
    let src = hybrid_source(vec![]);
    let caps = inspect_capabilities(&src).unwrap();
    assert!(
        caps.region_format.is_none(),
        "no layer produced expert bytes"
    );

    let dir = tempfile::tempdir().unwrap();
    let msg = extract_expert_banks(&src, dir.path(), ExtractionRequest::Native, "empty")
        .expect_err("nothing to extract natively")
        .to_string();
    assert!(msg.contains("region format"), "{msg}");
    // And Auto still proceeds, because it demanded nothing.
    assert_eq!(
        extract_expert_banks(&src, dir.path(), ExtractionRequest::Auto, "empty").unwrap(),
        ExpertBankOutcome::LegacyInline
    );
}

// ── the container the orchestrator writes is the adapter's bytes ─────────

/// The orchestrator composes rather than reimplements: what lands in the
/// container is exactly what the adapter produced, checked through the
/// real reader on a container the orchestrator wrote end to end.
#[test]
fn the_orchestrators_container_holds_the_adapters_bytes() {
    use crate::format::lyrw2::region_role::RegionRole;

    let src = capable_source();
    let dir = tempfile::tempdir().unwrap();
    extract_expert_banks(
        &src,
        dir.path(),
        ExtractionRequest::Native,
        "gpt-oss-fixture",
    )
    .unwrap();

    let container = Vindex3Container::open(dir.path()).unwrap();
    for layer in 0..LAYERS as u32 {
        let native = NativeMoeLayer::read(&src, src.arch(), layer)
            .unwrap()
            .unwrap();
        let expected = native.as_source();
        let reader = container.segment(&routed_storage_key(layer)).unwrap();

        for e in 0..EXPERTS {
            assert_eq!(
                reader
                    .region_bytes(0, e as u32, RegionRole::GateUpFused)
                    .unwrap()
                    .unwrap(),
                expected.experts_gate_up[e],
                "L{layer} e{e} gate_up"
            );
            assert_eq!(
                reader
                    .paired_region_bytes(0, e as u32, RegionRole::Down, RegionRole::Scales)
                    .unwrap()
                    .unwrap(),
                match &expected.scales {
                    crate::format::vindex3::import::ExpertScaleStreams::Paired { down, .. } =>
                        down[e],
                    _ => panic!("a packed-MXFP4 layer must be paired"),
                },
                "L{layer} e{e} down scales"
            );
        }
    }
}
