//! Composition gate tests: what the routed-container open admits and
//! refuses, proven against containers built by the real writer.

use super::*;
use crate::test_utils::make_test_gemma4_moe_weights;
use larql_vindex::format::lyrw2::region_format::RegionFormat;
use larql_vindex::format::lyrw2::region_layout::RegionLayout;
use larql_vindex::format::vindex3::{ContainerBuilder, ExpertScaleStreams, MoeLayerSource};
use tempfile::TempDir;

/// Build a container from the fixture model's own expert bytes.
///
/// `skip` omits that layer, so a test can produce a container that is
/// well-formed but cannot serve the model it is composed with — the
/// distinction the coverage check exists to make.
fn container_for(weights: &larql_models::ModelWeights, skip: Option<usize>) -> TempDir {
    let dir = TempDir::new().unwrap();
    let mut builder = ContainerBuilder::create(dir.path()).unwrap();
    let arch = &*weights.arch;
    let mut wrote = false;
    for layer in 0..weights.num_layers {
        if Some(layer) == skip {
            continue;
        }
        let Some(moe) = build_moe_weights(weights, arch, layer) else {
            continue;
        };
        let source = MoeLayerSource {
            layer: layer as u32,
            experts_gate_up: moe.experts_gate_up.clone(),
            experts_down: moe.experts_down.clone(),
            format: RegionFormat::Q4K,
            // Q4_K packs its scales inside each super-block.
            scales: ExpertScaleStreams::Inline,
            // A fact about the bytes this fixture writes, not about any
            // checkpoint: `build_moe_weights` yields `[all gate | all up]`.
            gate_up_layout: RegionLayout::ContiguousHalves,
            hidden_size: weights.hidden_size as u32,
            gate_up_stored_intermediate: moe.intermediate_size as u32,
            down_stored_intermediate: moe.inter_padded() as u32,
            semantic_intermediate: moe.intermediate_size as u32,
            top_k: moe.top_k as u32,
        };
        if builder.add_moe_layer(&source).is_ok() {
            wrote = true;
        }
    }
    assert!(wrote, "fixture produced no routed layers to import");
    builder
        .finish(
            weights.arch.family(),
            weights.arch.family(),
            weights.hidden_size,
            weights.num_layers,
        )
        .unwrap();
    dir
}

#[test]
fn a_container_missing_a_routed_layer_is_refused_before_generation() {
    // Coverage is checked against the model, so an omitted layer must be
    // caught at open — not at the token where that layer first runs, by
    // which point part of an answer has already been produced.
    let weights = make_test_gemma4_moe_weights();
    let full = container_for(&weights, None);
    ContainerRoutedBackend::open(full.path(), &weights, true)
        .expect("a complete container must compose");

    let missing = container_for(&weights, Some(0));
    let err = ContainerRoutedBackend::open(missing.path(), &weights, true)
        .err()
        .expect("a container missing a routed layer must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("no bank for it") || msg.contains("layer 0"),
        "the refusal must name the missing layer, got: {msg}"
    );
}

#[test]
fn a_container_describing_another_model_is_refused() {
    let weights = make_test_gemma4_moe_weights();
    let dir = container_for(&weights, None);

    // Rewrite the family in place: same banks, different identity.
    let index_path = dir.path().join("index.json");
    let text = std::fs::read_to_string(&index_path).unwrap();
    let mut json: serde_json::Value = serde_json::from_str(&text).unwrap();
    json["family"] = serde_json::Value::String("not-this-model".into());
    std::fs::write(&index_path, serde_json::to_string_pretty(&json).unwrap()).unwrap();

    let err = ContainerRoutedBackend::open(dir.path(), &weights, true)
        .err()
        .expect("composing a container for another model must be refused");
    assert!(
        err.to_string().contains("not-this-model"),
        "the refusal must name both sides, got: {err}"
    );
}

#[test]
fn a_composed_route_reports_both_artifacts() {
    // Provenance is not decoration: a composed run is two directories and
    // a result recorded against only one of them is unreproducible.
    let weights = make_test_gemma4_moe_weights();
    let dir = container_for(&weights, None);
    let backend = ContainerRoutedBackend::open(dir.path(), &weights, true).unwrap();
    let line = backend.describe(std::path::Path::new("/models/spine.vindex"));
    assert!(line.contains("/models/spine.vindex"), "{line}");
    assert!(line.contains(&dir.path().display().to_string()), "{line}");
}

// ── Tamper controls for the native split-scale open ─────────────────────
//
// The byte authority moved from the model's slices to the container's
// declared format; these prove the move did not weaken validation. Each
// tamper is a container the real writer happily produces (the writer
// checks uniformity, not format arithmetic) that the composed open must
// still refuse — and the untampered sibling must admit, or the refusals
// prove nothing.

/// How a synthetic native container deviates from a servable one.
#[derive(Clone, Copy, PartialEq)]
enum Tamper {
    None,
    /// Payloads sized for MXFP4 but declared as the spine's format —
    /// the declaration is the byte authority, so it must stop being one.
    WrongDeclaredFormat,
    /// A split-scale format shipped with no scale partners at all.
    MissingScalePair,
    /// Scale partners present but one byte short per expert.
    ShortScalePair,
    /// Payloads one MXFP4 group short.
    ShortPayload,
}

/// Build an MXFP4 split-scale container for the fixture model, `tamper`ed.
fn mxfp4_container_for(weights: &larql_models::ModelWeights, tamper: Tamper) -> TempDir {
    use larql_models::quant::mxfp4::{FUSED_HALVES, MXFP4_GROUP_BYTES, MXFP4_GROUP_ELEMS};

    let dir = TempDir::new().unwrap();
    let mut builder = ContainerBuilder::create(dir.path()).unwrap();
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let mut wrote = false;
    for layer in 0..weights.num_layers {
        let Some(moe) = build_moe_weights(weights, arch, layer) else {
            continue;
        };
        let inter = moe.intermediate_size;
        let row = |cols: usize| (cols / MXFP4_GROUP_ELEMS) * MXFP4_GROUP_BYTES;
        let payload_cut = if tamper == Tamper::ShortPayload {
            MXFP4_GROUP_BYTES
        } else {
            0
        };
        let scale_cut = usize::from(tamper == Tamper::ShortScalePair);
        let per_expert = |len: usize| vec![vec![0x11u8; len]; moe.num_experts];
        let gate_up = per_expert(FUSED_HALVES * inter * row(hidden) - payload_cut);
        let down = per_expert(hidden * row(inter) - payload_cut);
        let gu_scales = per_expert(FUSED_HALVES * inter * (hidden / MXFP4_GROUP_ELEMS) - scale_cut);
        let dn_scales = per_expert(hidden * (inter / MXFP4_GROUP_ELEMS) - scale_cut);

        fn as_slices(v: &[Vec<u8>]) -> Vec<&[u8]> {
            v.iter().map(|e| e.as_slice()).collect()
        }
        let source = MoeLayerSource {
            layer: layer as u32,
            experts_gate_up: as_slices(&gate_up),
            experts_down: as_slices(&down),
            format: if tamper == Tamper::WrongDeclaredFormat {
                RegionFormat::Q4K
            } else {
                RegionFormat::Mxfp4
            },
            scales: if tamper == Tamper::MissingScalePair {
                ExpertScaleStreams::Inline
            } else {
                ExpertScaleStreams::Paired {
                    gate_up: as_slices(&gu_scales),
                    down: as_slices(&dn_scales),
                }
            },
            gate_up_layout: RegionLayout::ContiguousHalves,
            hidden_size: hidden as u32,
            gate_up_stored_intermediate: inter as u32,
            down_stored_intermediate: inter as u32,
            semantic_intermediate: inter as u32,
            top_k: moe.top_k as u32,
        };
        builder.add_moe_layer(&source).unwrap();
        wrote = true;
    }
    assert!(wrote, "fixture produced no routed layers to import");
    builder
        .finish(
            weights.arch.family(),
            weights.arch.family(),
            weights.hidden_size,
            weights.num_layers,
        )
        .unwrap();
    dir
}

fn refusal_for(weights: &larql_models::ModelWeights, tamper: Tamper) -> String {
    let dir = mxfp4_container_for(weights, tamper);
    ContainerRoutedBackend::open(dir.path(), weights, true)
        .err()
        .expect("the tampered container must be refused at open")
        .to_string()
}

#[test]
fn a_native_split_scale_container_admits() {
    // The restore arm: every tamper below must be the ONLY reason its
    // container is refused, which this untampered sibling proves.
    let weights = make_test_gemma4_moe_weights();
    let dir = mxfp4_container_for(&weights, Tamper::None);
    ContainerRoutedBackend::open(dir.path(), &weights, true)
        .expect("an untampered native split-scale container must admit");
}

#[test]
fn a_wrong_declared_format_is_refused_by_the_declarations_own_arithmetic() {
    // Declaring the spine's format hands byte authority back to the model's
    // slices, which these MXFP4-sized payloads cannot satisfy.
    let weights = make_test_gemma4_moe_weights();
    let msg = refusal_for(&weights, Tamper::WrongDeclaredFormat);
    assert!(msg.contains("routed region bytes"), "{msg}");
}

#[test]
fn a_split_scale_bank_with_no_scale_partner_is_refused() {
    let weights = make_test_gemma4_moe_weights();
    let msg = refusal_for(&weights, Tamper::MissingScalePair);
    assert!(msg.to_lowercase().contains("scale"), "{msg}");
}

#[test]
fn a_scale_partner_of_the_wrong_length_is_refused() {
    let weights = make_test_gemma4_moe_weights();
    let msg = refusal_for(&weights, Tamper::ShortScalePair);
    assert!(msg.contains("scale partner bytes"), "{msg}");
}

#[test]
fn a_payload_one_group_short_is_refused() {
    let weights = make_test_gemma4_moe_weights();
    let msg = refusal_for(&weights, Tamper::ShortPayload);
    assert!(msg.contains("routed region bytes"), "{msg}");
}
