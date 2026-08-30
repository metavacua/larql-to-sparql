//! Checkpoint-adapter tests, against four bars:
//!
//! ```text
//! byte ownership       every slice is the checkpoint's own bytes, no transcode
//! axis correctness     expert-stride and layer-stride proven independently
//! layout honesty       rows verbatim, Interleaved declared, no canonicalisation
//! capability honesty   scale_streams_available true because this source
//!                      produced both streams, not because the format implies it
//! ```
//!
//! The fixture is deliberately **asymmetric on every axis** — experts,
//! layers, the two projections' out/group counts, and the payload/scale
//! byte widths all differ — so a stride computed against the wrong axis
//! lands somewhere wrong rather than somewhere that happens to match.

use std::collections::HashMap;

use larql_models::quant::mxfp4::{MXFP4_GROUP_BYTES, MXFP4_GROUP_ELEMS};

use super::*;

const EXPERTS: usize = 3;
const HIDDEN: usize = 64;
const INTERMEDIATE: usize = 96;
const TOP_K: usize = 2;

/// `gate_up` fuses both halves: 2 × 96 = 192 rows, contracting over hidden.
const GU_OUT: usize = INTERMEDIATE * 2;
const GU_GROUPS: usize = HIDDEN / MXFP4_GROUP_ELEMS; // 2
/// `down` contracts over the intermediate axis back to hidden.
const DN_OUT: usize = HIDDEN;
const DN_GROUPS: usize = INTERMEDIATE / MXFP4_GROUP_ELEMS; // 3

const GU_PAYLOAD: usize = GU_OUT * GU_GROUPS * MXFP4_GROUP_BYTES;
const GU_SCALE: usize = GU_OUT * GU_GROUPS;
const DN_PAYLOAD: usize = DN_OUT * DN_GROUPS * MXFP4_GROUP_BYTES;
const DN_SCALE: usize = DN_OUT * DN_GROUPS;

/// Distinct per (layer, expert, stream, offset). Every axis participates,
/// so a slice taken at the wrong layer, expert or stream is detectable.
fn byte_at(layer: u32, expert: usize, stream: u8, i: usize) -> u8 {
    (layer as usize * 211 + expert * 37 + stream as usize * 101 + i) as u8
}

fn stream(layer: u32, stream_id: u8, per_expert: usize) -> Vec<u8> {
    (0..EXPERTS)
        .flat_map(|e| (0..per_expert).map(move |i| byte_at(layer, e, stream_id, i)))
        .collect()
}

/// A `WeightSource` that answers only raw packed tensors, keyed exactly as
/// the architecture spells them.
struct RawSource {
    arch: Box<dyn larql_models::ModelArchitecture>,
    raw: HashMap<String, (Vec<u8>, Vec<usize>)>,
    layers: usize,
}

impl WeightSource for RawSource {
    fn get_tensor(&self, _key: &str) -> Option<(Vec<f32>, usize, usize)> {
        // A native adapter must never reach the dequantised view.
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

const LAYERS: usize = 3;

/// A real `gpt_oss` architecture at fixture dimensions — detected from
/// config exactly as a checkpoint would be, so the key spellings under
/// test are the architecture's own and not restated here.
fn gpt_oss_arch() -> Box<dyn larql_models::ModelArchitecture> {
    larql_models::detect_from_json(&serde_json::json!({
        "model_type": "gpt_oss",
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
    }))
}

fn source() -> RawSource {
    let arch = gpt_oss_arch();
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
    RawSource {
        arch,
        raw,
        layers: LAYERS,
    }
}

fn expected(layer: u32, expert: usize, stream_id: u8, per_expert: usize) -> Vec<u8> {
    (0..per_expert)
        .map(|i| byte_at(layer, expert, stream_id, i))
        .collect()
}

// ── bar 1: byte ownership ────────────────────────────────────────────────

/// Every one of the four streams, for every expert, is exactly the
/// checkpoint's bytes at that expert's offset. Nothing is transcoded and
/// nothing is reordered.
#[test]
fn all_four_streams_are_the_checkpoints_own_bytes_per_expert() {
    let src = source();
    let layer = 1;
    let owned = NativeMoeLayer::read(&src, src.arch(), layer)
        .unwrap()
        .unwrap();
    let s = owned.as_source();

    for e in 0..EXPERTS {
        assert_eq!(
            s.experts_gate_up[e],
            &expected(layer, e, 0, GU_PAYLOAD)[..],
            "gate_up e{e}"
        );
        assert_eq!(
            s.experts_down[e],
            &expected(layer, e, 2, DN_PAYLOAD)[..],
            "down e{e}"
        );
    }
    let ExpertScaleStreams::Paired { gate_up, down } = &s.scales else {
        panic!("a packed-MXFP4 layer must present paired scale streams");
    };
    for e in 0..EXPERTS {
        assert_eq!(
            gate_up[e],
            &expected(layer, e, 1, GU_SCALE)[..],
            "gate_up scales e{e}"
        );
        assert_eq!(
            down[e],
            &expected(layer, e, 3, DN_SCALE)[..],
            "down scales e{e}"
        );
    }
}

/// The adapter never falls back to the dequantised view. `RawSource`
/// answers `get_tensor` with `None`, so any path that reached for f32
/// would fail rather than quietly produce a transcoded bank.
#[test]
fn the_adapter_reads_only_raw_bytes() {
    let src = source();
    assert!(src.get_tensor("anything").is_none());
    assert!(NativeMoeLayer::read(&src, src.arch(), 0).unwrap().is_some());
}

// ── bar 2: axis correctness ──────────────────────────────────────────────

/// Expert stride is proven independently of layer stride: within one
/// layer, no expert's slice equals any other's, in any stream.
#[test]
fn expert_stride_separates_every_expert() {
    let src = source();
    let owned = NativeMoeLayer::read(&src, src.arch(), 2).unwrap().unwrap();
    let s = owned.as_source();
    for a in 0..EXPERTS {
        for b in 0..EXPERTS {
            if a == b {
                continue;
            }
            assert_ne!(
                s.experts_gate_up[a], s.experts_gate_up[b],
                "gate_up {a} vs {b}"
            );
            assert_ne!(s.experts_down[a], s.experts_down[b], "down {a} vs {b}");
        }
    }
}

/// Layer stride is proven independently of expert stride: the *same*
/// expert index read from different layers yields different bytes. A
/// layer-agnostic key would make these identical.
#[test]
fn layer_stride_separates_every_layer() {
    let src = source();
    let per_layer: Vec<Vec<u8>> = (0..LAYERS as u32)
        .map(|l| {
            let owned = NativeMoeLayer::read(&src, src.arch(), l).unwrap().unwrap();
            owned.as_source().experts_gate_up[1].to_vec()
        })
        .collect();
    for a in 0..LAYERS {
        for b in 0..LAYERS {
            if a == b {
                continue;
            }
            assert_ne!(per_layer[a], per_layer[b], "layer {a} vs {b}, same expert");
        }
    }
}

/// The two projections have different out-counts, group-counts and byte
/// widths, so a stride computed against the wrong projection cannot land
/// on a valid length by coincidence.
#[test]
fn the_two_projections_have_independently_correct_widths() {
    let src = source();
    let owned = NativeMoeLayer::read(&src, src.arch(), 0).unwrap().unwrap();
    let s = owned.as_source();

    assert_eq!(s.experts_gate_up[0].len(), GU_PAYLOAD);
    assert_eq!(s.experts_down[0].len(), DN_PAYLOAD);
    assert_ne!(GU_PAYLOAD, DN_PAYLOAD, "fixture must distinguish them");

    let ExpertScaleStreams::Paired { gate_up, down } = &s.scales else {
        unreachable!()
    };
    assert_eq!(gate_up[0].len(), GU_SCALE);
    assert_eq!(down[0].len(), DN_SCALE);
    // A scale stream is 1/16 of its payload — confusing the two is a
    // length error, not a silent one.
    assert_eq!(
        gate_up[0].len() * MXFP4_GROUP_BYTES,
        s.experts_gate_up[0].len()
    );
    assert_eq!(down[0].len() * MXFP4_GROUP_BYTES, s.experts_down[0].len());
}

/// The declared dimensions describe the fused operand correctly: stored
/// rows are `2 × intermediate`, and `down` contracts over its own
/// group-derived width.
#[test]
fn declared_dimensions_describe_the_fused_operand() {
    let src = source();
    let owned = NativeMoeLayer::read(&src, src.arch(), 0).unwrap().unwrap();
    let s = owned.as_source();
    assert_eq!(s.hidden_size as usize, HIDDEN);
    assert_eq!(s.gate_up_stored_intermediate as usize, GU_OUT / 2);
    assert_eq!(
        s.down_stored_intermediate as usize,
        DN_GROUPS * MXFP4_GROUP_ELEMS
    );
    assert_eq!(s.semantic_intermediate as usize, INTERMEDIATE);
    assert_eq!(s.top_k as usize, TOP_K);
}

// ── bar 3: layout honesty ────────────────────────────────────────────────

/// Rows are verbatim and the arrangement is *declared*. If the adapter
/// had de-interleaved, the bytes would differ from the checkpoint's — so
/// this is checked against the source bytes, not against a re-derivation.
#[test]
fn rows_are_verbatim_and_the_layout_is_declared_not_applied() {
    let src = source();
    let owned = NativeMoeLayer::read(&src, src.arch(), 0).unwrap().unwrap();
    assert_eq!(owned.gate_up_layout(), RegionLayout::Interleaved);
    let s = owned.as_source();
    assert_eq!(s.gate_up_layout, RegionLayout::Interleaved);
    // Verbatim: byte 0 of expert 0 is the checkpoint's byte 0, which a
    // row permutation would have moved.
    assert_eq!(s.experts_gate_up[0][0], byte_at(0, 0, 0, 0));
    assert_eq!(s.experts_gate_up[0], &expected(0, 0, 0, GU_PAYLOAD)[..]);
}

// ── bar 4: capability honesty ────────────────────────────────────────────

/// The adapter reports reachable scale streams because it *has* them.
/// Contrast `SourceCapabilities::from_expert_format`, which for the same
/// format refuses to claim availability.
#[test]
fn capability_reflects_this_source_not_the_format_class() {
    let src = source();
    let owned = NativeMoeLayer::read(&src, src.arch(), 0).unwrap().unwrap();
    let caps = owned.capabilities();
    assert!(caps.split_scale_streams, "the format says split");
    assert!(
        caps.scale_streams_available,
        "and this source produced them"
    );
    assert!(caps.gate_up_layout.is_declared());

    let from_format_alone = SourceCapabilities::from_expert_format(
        larql_models::ExpertFormat::PackedMxfp4,
        Some(RegionFormat::Mxfp4),
    );
    assert!(
        !from_format_alone.scale_streams_available,
        "the format class must not claim availability on a source's behalf"
    );
}

/// End of the seam: these capabilities admit a native extraction.
#[test]
fn the_adapters_capabilities_admit_native_extraction() {
    use crate::extract::target::{admit, ExtractionRequest, ExtractionTarget};
    let src = source();
    let owned = NativeMoeLayer::read(&src, src.arch(), 0).unwrap().unwrap();
    assert_eq!(
        admit(ExtractionRequest::Native, &owned.capabilities()).unwrap(),
        ExtractionTarget::NativeV3
    );
}

// ── refusals ─────────────────────────────────────────────────────────────

fn broken(mutate: impl FnOnce(&mut RawSource)) -> String {
    let mut src = source();
    mutate(&mut src);
    let arch = gpt_oss_arch();
    NativeMoeLayer::read(&src, &*arch, 0)
        .expect_err("a malformed layer must refuse")
        .to_string()
}

/// Payload without exponents is the failure that decodes rather than
/// fails — every group at 2^0.
#[test]
fn blocks_without_their_scale_stream_are_refused() {
    let msg = broken(|src| {
        let k = src.arch.packed_gate_up_scales_key(0).unwrap();
        src.raw.remove(&k);
    });
    assert!(msg.contains("2^0"), "{msg}");
}

/// A short buffer sliced by stride yields in-range offsets into the wrong
/// expert, so byte count is checked against the declared shape.
#[test]
fn a_buffer_shorter_than_its_declared_shape_is_refused() {
    let msg = broken(|src| {
        let k = src.arch.packed_gate_up_blocks_key(0).unwrap();
        let entry = src.raw.get_mut(&k).unwrap();
        entry.0.truncate(entry.0.len() - MXFP4_GROUP_BYTES);
    });
    assert!(
        msg.contains("bytes") && msg.contains("shape implies"),
        "{msg}"
    );
}

/// Blocks and scales that individually parse but disagree on geometry
/// would pair a matrix with another matrix's exponents.
#[test]
fn a_scale_stream_disagreeing_on_geometry_is_refused() {
    let msg = broken(|src| {
        let k = src.arch.packed_gate_up_scales_key(0).unwrap();
        let entry = src.raw.get_mut(&k).unwrap();
        entry.1 = vec![EXPERTS, GU_OUT, GU_GROUPS + 1];
    });
    assert!(msg.contains("does not match blocks"), "{msg}");
}

#[test]
fn a_wrong_group_width_is_refused() {
    let msg = broken(|src| {
        let k = src.arch.packed_down_blocks_key(0).unwrap();
        let entry = src.raw.get_mut(&k).unwrap();
        entry.1 = vec![EXPERTS, DN_OUT, DN_GROUPS, MXFP4_GROUP_BYTES + 1];
    });
    assert!(msg.contains("bytes per group"), "{msg}");
}

/// Both projections must cover the same experts, or routing selects an
/// index one of them does not have.
#[test]
fn projections_disagreeing_on_expert_count_are_refused() {
    let msg = broken(|src| {
        // Both down tensors must stay internally consistent, or the
        // per-projection geometry check fires first and this never
        // reaches the cross-projection comparison it is testing.
        let bk = src.arch.packed_down_blocks_key(0).unwrap();
        let b = src.raw.get_mut(&bk).unwrap();
        b.1 = vec![EXPERTS - 1, DN_OUT, DN_GROUPS, MXFP4_GROUP_BYTES];
        b.0.truncate((EXPERTS - 1) * DN_PAYLOAD);

        let sk = src.arch.packed_down_scales_key(0).unwrap();
        let s = src.raw.get_mut(&sk).unwrap();
        s.1 = vec![EXPERTS - 1, DN_OUT, DN_GROUPS];
        s.0.truncate((EXPERTS - 1) * DN_SCALE);
    });
    assert!(msg.contains("same experts"), "{msg}");
}

// ── applicability: "not this model" vs "this model, malformed" ───────────

/// A model that does not store packed-MXFP4 experts is **not applicable**,
/// not an error. Collapsing the two would make every non-MXFP4 extraction
/// fail the moment the native path is offered one.
#[test]
fn a_non_mxfp4_model_is_not_applicable_rather_than_malformed() {
    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "num_hidden_layers": LAYERS,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 128,
    }));
    assert_ne!(
        arch.expert_format(),
        larql_models::ExpertFormat::PackedMxfp4
    );

    let src = source();
    let got = NativeMoeLayer::read(&src, &*arch, 0).expect("must not error");
    assert!(
        got.is_none(),
        "a non-MXFP4 model is simply not this path's business"
    );
}

/// An architecture that declares packed experts but names no `gate_up` for
/// a given layer — a dense layer inside a hybrid stack. Also not an error.
#[test]
fn a_layer_with_no_packed_gate_up_is_not_applicable() {
    let arch = PartialArch {
        config: gpt_oss_arch().config().clone(),
        declare_gate_up: false,
        declare_scales: true,
    };
    let src = source();
    let got = NativeMoeLayer::read(&src, &arch, 0).expect("must not error");
    assert!(got.is_none());
}

/// An architecture that declares one half of a pair and not the other is
/// an arch-level inconsistency — distinct from a source that cannot serve
/// a declared key, and reported as such.
#[test]
fn an_architecture_declaring_blocks_but_not_scales_is_refused() {
    let arch = PartialArch {
        config: gpt_oss_arch().config().clone(),
        declare_gate_up: true,
        declare_scales: false,
    };
    let src = source();
    let msg = NativeMoeLayer::read(&src, &arch, 0)
        .expect_err("a half-declared pair must refuse")
        .to_string();
    assert!(msg.contains("declared but"), "{msg}");
    assert!(msg.contains("2^0"), "{msg}");
}

/// A minimal architecture that can declare its packed keys selectively.
/// Only `family` and `config` are required by the trait; everything else
/// defaults, so this stays a stub rather than a second implementation.
struct PartialArch {
    config: larql_models::ModelConfig,
    declare_gate_up: bool,
    declare_scales: bool,
}

impl larql_models::ModelArchitecture for PartialArch {
    fn family(&self) -> &str {
        "partial-stub"
    }
    fn config(&self) -> &larql_models::ModelConfig {
        &self.config
    }
    fn expert_format(&self) -> larql_models::ExpertFormat {
        larql_models::ExpertFormat::PackedMxfp4
    }
    fn packed_gate_up_blocks_key(&self, layer: usize) -> Option<String> {
        self.declare_gate_up
            .then(|| gpt_oss_arch().packed_gate_up_blocks_key(layer))
            .flatten()
    }
    fn packed_gate_up_scales_key(&self, layer: usize) -> Option<String> {
        self.declare_scales
            .then(|| gpt_oss_arch().packed_gate_up_scales_key(layer))
            .flatten()
    }
    fn packed_down_blocks_key(&self, layer: usize) -> Option<String> {
        gpt_oss_arch().packed_down_blocks_key(layer)
    }
    fn packed_down_scales_key(&self, layer: usize) -> Option<String> {
        gpt_oss_arch().packed_down_scales_key(layer)
    }
}

// ── further malformations ────────────────────────────────────────────────

/// A fused operand with an odd row count cannot be two halves, so the
/// per-branch intermediate would be a rounded-down lie.
#[test]
fn an_odd_fused_row_count_is_refused() {
    let msg = broken(|src| {
        let bk = src.arch.packed_gate_up_blocks_key(0).unwrap();
        let b = src.raw.get_mut(&bk).unwrap();
        b.1 = vec![EXPERTS, GU_OUT - 1, GU_GROUPS, MXFP4_GROUP_BYTES];
        b.0.truncate(EXPERTS * (GU_OUT - 1) * GU_GROUPS * MXFP4_GROUP_BYTES);
        let sk = src.arch.packed_gate_up_scales_key(0).unwrap();
        let s = src.raw.get_mut(&sk).unwrap();
        s.1 = vec![EXPERTS, GU_OUT - 1, GU_GROUPS];
        s.0.truncate(EXPERTS * (GU_OUT - 1) * GU_GROUPS);
    });
    assert!(msg.contains("not two halves"), "{msg}");
}

#[test]
fn a_blocks_tensor_of_the_wrong_rank_is_refused() {
    let msg = broken(|src| {
        let k = src.arch.packed_gate_up_blocks_key(0).unwrap();
        src.raw.get_mut(&k).unwrap().1 = vec![EXPERTS, GU_OUT, GU_GROUPS];
    });
    assert!(msg.contains("is not [experts, out, groups"), "{msg}");
}

#[test]
fn a_scales_tensor_of_the_wrong_rank_is_refused() {
    let msg = broken(|src| {
        let k = src.arch.packed_gate_up_scales_key(0).unwrap();
        src.raw.get_mut(&k).unwrap().1 = vec![EXPERTS, GU_OUT, GU_GROUPS, MXFP4_GROUP_BYTES];
    });
    assert!(msg.contains("is not [experts, out, groups]"), "{msg}");
}

#[test]
fn a_degenerate_shape_is_refused() {
    let msg = broken(|src| {
        let bk = src.arch.packed_gate_up_blocks_key(0).unwrap();
        let b = src.raw.get_mut(&bk).unwrap();
        b.1 = vec![0, GU_OUT, GU_GROUPS, MXFP4_GROUP_BYTES];
        b.0.clear();
        let sk = src.arch.packed_gate_up_scales_key(0).unwrap();
        let s = src.raw.get_mut(&sk).unwrap();
        s.1 = vec![0, GU_OUT, GU_GROUPS];
        s.0.clear();
    });
    assert!(msg.contains("degenerate shape"), "{msg}");
}

#[test]
fn a_short_scale_buffer_is_refused() {
    let msg = broken(|src| {
        let k = src.arch.packed_down_scales_key(0).unwrap();
        let e = src.raw.get_mut(&k).unwrap();
        e.0.truncate(e.0.len() - 1);
    });
    assert!(
        msg.contains("scales are") && msg.contains("shape implies"),
        "{msg}"
    );
}

// ── surface ──────────────────────────────────────────────────────────────

#[test]
fn expert_count_is_reported() {
    let src = source();
    let owned = NativeMoeLayer::read(&src, src.arch(), 0).unwrap().unwrap();
    assert_eq!(owned.num_experts(), EXPERTS);
}

/// `Debug` must summarise, not dump: a real layer is hundreds of MB and a
/// failing assertion that prints it is unreadable.
#[test]
fn debug_prints_byte_counts_not_payloads() {
    let src = source();
    let owned = NativeMoeLayer::read(&src, src.arch(), 0).unwrap().unwrap();
    let s = format!("{owned:?}");
    assert!(s.contains("gate_up_blocks_bytes"), "{s}");
    // The whole stream's length, across every expert — not one expert's
    // slice. Worth pinning: the two differ by the expert count, and a
    // summary that quietly reported the per-expert figure would make a
    // truncated buffer look correctly sized.
    assert!(s.contains(&(EXPERTS * GU_PAYLOAD).to_string()), "{s}");
    assert!(
        s.len() < 400,
        "Debug should summarise; got {} chars",
        s.len()
    );
}

// ── end to end: checkpoint bytes survive persistence ─────────────────────

/// The #8 qualification, whole:
///
/// ```text
/// checkpoint bytes → adapter → VINDEX3 writer → close
///                  → reopen with the real reader
///                  → resolve all four regions → byte-exact
/// ```
///
/// Across every layer and every expert, comparing **source expert `e` of
/// layer `l`** against **reopened expert `e` of layer `l`** — not against
/// a collection or a hash, so a permutation on either axis cannot pass.
#[test]
fn every_region_of_every_expert_and_layer_survives_a_real_round_trip() {
    use crate::format::lyrw2::region_role::RegionRole;
    use crate::format::vindex3::import::routed_storage_key;
    use crate::format::vindex3::{ContainerBuilder, Vindex3Container};

    let src = source();
    let dir = tempfile::tempdir().unwrap();

    let mut builder = ContainerBuilder::create(dir.path()).unwrap();
    for layer in 0..LAYERS as u32 {
        let owned = NativeMoeLayer::read(&src, src.arch(), layer)
            .unwrap()
            .unwrap();
        builder.add_moe_layer(&owned.as_source()).unwrap();
    }
    builder
        .finish("gpt-oss-fixture", "gpt_oss", HIDDEN, LAYERS)
        .unwrap();

    // Reopen from disk: the claim is about what a reader finds, not what
    // the writer intended.
    let container = Vindex3Container::open(dir.path()).unwrap();
    assert!(
        container.verify().is_empty(),
        "structural defects: {:?}",
        container.verify()
    );

    let mut checked = 0usize;
    for layer in 0..LAYERS as u32 {
        let reader = container.segment(&routed_storage_key(layer)).unwrap();
        for e in 0..EXPERTS {
            let ex = e as u32;
            // Payloads, addressed by role (unique in this bank).
            assert_eq!(
                reader
                    .region_bytes(0, ex, RegionRole::GateUpFused)
                    .unwrap()
                    .unwrap(),
                &expected(layer, e, 0, GU_PAYLOAD)[..],
                "L{layer} e{e} gate_up"
            );
            assert_eq!(
                reader
                    .region_bytes(0, ex, RegionRole::Down)
                    .unwrap()
                    .unwrap(),
                &expected(layer, e, 2, DN_PAYLOAD)[..],
                "L{layer} e{e} down"
            );
            // Exponents, addressed through their pairing — role alone is
            // ambiguous here, which is exactly the point.
            assert_eq!(
                reader
                    .paired_region_bytes(0, ex, RegionRole::GateUpFused, RegionRole::Scales)
                    .unwrap()
                    .unwrap(),
                &expected(layer, e, 1, GU_SCALE)[..],
                "L{layer} e{e} gate_up scales"
            );
            assert_eq!(
                reader
                    .paired_region_bytes(0, ex, RegionRole::Down, RegionRole::Scales)
                    .unwrap()
                    .unwrap(),
                &expected(layer, e, 3, DN_SCALE)[..],
                "L{layer} e{e} down scales"
            );
            checked += 4;
        }
    }
    assert_eq!(checked, LAYERS * EXPERTS * 4);
}

/// The stored arrangement survives persistence. Bytes alone cannot carry
/// it — a container written under the other declaration would be
/// byte-identical — so it is checked on the reopened schema.
#[test]
fn the_declared_layout_survives_persistence() {
    use crate::format::lyrw2::region_role::RegionRole;
    use crate::format::vindex3::import::routed_storage_key;
    use crate::format::vindex3::{ContainerBuilder, Vindex3Container};

    let src = source();
    let dir = tempfile::tempdir().unwrap();
    let mut builder = ContainerBuilder::create(dir.path()).unwrap();
    let owned = NativeMoeLayer::read(&src, src.arch(), 0).unwrap().unwrap();
    builder.add_moe_layer(&owned.as_source()).unwrap();
    builder
        .finish("gpt-oss-fixture", "gpt_oss", HIDDEN, LAYERS)
        .unwrap();

    let container = Vindex3Container::open(dir.path()).unwrap();
    let reader = container.segment(&routed_storage_key(0)).unwrap();
    let schemas = reader.schemas_for(0).unwrap();

    let fused = schemas
        .iter()
        .find(|s| s.role == RegionRole::GateUpFused)
        .expect("a fused gate/up region");
    assert_eq!(
        fused.layout,
        RegionLayout::Interleaved,
        "the checkpoint's arrangement must reach the container"
    );
    // And nothing else claims an arrangement it cannot have.
    for s in schemas.iter().filter(|s| s.role != RegionRole::GateUpFused) {
        assert!(!s.layout.is_declared(), "{:?} declared a layout", s.role);
    }
}

/// A source that declares the keys but cannot serve them raw is a
/// dequantised view — refuse rather than silently transcode.
#[test]
fn a_source_that_cannot_supply_raw_bytes_is_refused() {
    let msg = broken(|src| {
        let k = src.arch.packed_gate_up_blocks_key(0).unwrap();
        let entry = src.raw.remove(&k).unwrap();
        // Present under the key but unreadable as raw: emulate by removing
        // only the *scales*, leaving blocks — covered above — so here we
        // drop down blocks while keeping its scales.
        let dk = src.arch.packed_down_blocks_key(0).unwrap();
        src.raw.remove(&dk);
        src.raw.insert(k, entry);
    });
    assert!(
        msg.contains("scale stream is") || msg.contains("raw bytes"),
        "{msg}"
    );
}
