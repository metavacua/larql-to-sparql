//! Config, coverage, and loader tests.
//!
//! The coverage fixture is the real checkpoint's tensor-name inventory
//! (`test_data/moss_tts_realtime_tensor_names.txt`, 403 names dumped from
//! `OpenMOSS-Team/MOSS-TTS-Realtime` rev `7568278`) — names only, no
//! weights, so the step-1 accounting gate runs in CI. Loader tests build
//! synthetic miniature checkpoints; nothing below depends on the release's
//! dimensions.

use safetensors::tensor::{serialize_to_file, TensorView};
use safetensors::Dtype;

use crate::config::ModelArchitecture;
use crate::detect::detect_from_json;

use super::config::MossTtsRealtimeConfig;
use super::coverage::classify_checkpoint_keys;
use super::keys::{local_embed_table, local_final_norm, local_layer_prefix, local_lm_head};
use super::loader::load_moss_tts_aux_from_safetensors;

/// The real checkpoint's tensor names.
const REAL_TENSOR_NAMES: &str =
    include_str!("../../../test_data/moss_tts_realtime_tensor_names.txt");

/// Real checkpoint layer counts, for the fixture test only.
const REAL_BACKBONE_LAYERS: usize = 28;
const REAL_LOCAL_LAYERS: usize = 4;
const REAL_RVQ: usize = 16;
/// 11 members per decoder layer (q/k/v/o, q/k norm, gate/up/down, two norms)
/// plus embedding and final norm.
const LAYER_MEMBERS: usize = 11;

fn config_json(
    rvq: usize,
    audio_vocab: usize,
    pad: usize,
    hidden: usize,
    backbone_layers: usize,
    local_layers: usize,
) -> serde_json::Value {
    serde_json::json!({
        "model_type": "moss_tts_realtime",
        "rvq": rvq,
        "audio_vocab_size": audio_vocab,
        "audio_pad_token": pad,
        "language_config": {
            "model_type": "qwen3",
            "hidden_size": hidden,
            "num_hidden_layers": backbone_layers,
            "intermediate_size": hidden * 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": hidden / 2,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "vocab_size": 100
        },
        "local_config": {
            "hidden_size": hidden,
            "num_hidden_layers": local_layers,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": hidden / 2,
            "intermediate_size": hidden * 2,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "max_position_embeddings": rvq + 1
        }
    })
}

fn real_shaped_setup() -> (Box<dyn ModelArchitecture>, MossTtsRealtimeConfig) {
    let json = config_json(
        REAL_RVQ,
        1027,
        1024,
        2048,
        REAL_BACKBONE_LAYERS,
        REAL_LOCAL_LAYERS,
    );
    let arch = detect_from_json(&json);
    let config = MossTtsRealtimeConfig::from_config_json(&json).unwrap();
    (arch, config)
}

// ── Config parsing ───────────────────────────────────────────────────────

#[test]
fn config_parses_and_derives_counts() {
    let json = config_json(16, 1027, 1024, 2048, 28, 4);
    let config = MossTtsRealtimeConfig::from_config_json(&json).unwrap();
    assert_eq!(config.rvq, 16);
    assert_eq!(config.audio_vocab_size, 1027);
    assert_eq!(config.audio_pad_token, 1024);
    assert_eq!(config.local.num_layers, 4);
    assert_eq!(config.local.max_position_embeddings, 17);
    assert_eq!(config.backbone_audio_tables(), 16);
    assert_eq!(config.top_level_tables(), 17);
    assert_eq!(config.local_embed_tables(), 15);
    assert_eq!(config.lm_heads(), 16);
}

#[test]
fn config_missing_field_errors_by_name() {
    let mut json = config_json(2, 11, 8, 8, 1, 1);
    json.as_object_mut().unwrap().remove("rvq");
    let err = MossTtsRealtimeConfig::from_config_json(&json).unwrap_err();
    assert!(
        err.to_string().contains("rvq"),
        "error names the field: {err}"
    );
}

#[test]
fn config_missing_local_config_errors() {
    let mut json = config_json(2, 11, 8, 8, 1, 1);
    json.as_object_mut().unwrap().remove("local_config");
    let err = MossTtsRealtimeConfig::from_config_json(&json).unwrap_err();
    assert!(err.to_string().contains("local_config"));
}

#[test]
fn config_missing_float_field_errors_by_name() {
    let mut json = config_json(2, 11, 8, 8, 1, 1);
    json["local_config"]
        .as_object_mut()
        .unwrap()
        .remove("rms_norm_eps");
    let err = MossTtsRealtimeConfig::from_config_json(&json).unwrap_err();
    assert!(
        err.to_string().contains("rms_norm_eps"),
        "error names the field: {err}"
    );
}

#[test]
fn audio_special_ids_derive_from_pad() {
    let config = MossTtsRealtimeConfig::from_config_json(&config_json(2, 11, 8, 8, 1, 1)).unwrap();
    assert_eq!(config.audio_bos_token(), 9);
    assert_eq!(config.audio_eos_token(), 10);
    // The layout invariant the derivation rests on: pad + specials fit
    // the audio vocabulary exactly.
    assert!(config.audio_eos_token() < config.audio_vocab_size);
}

// ── Keys: layer-member derivation from the architecture ──────────────────

#[test]
fn layer_member_suffixes_derive_from_moss_arch() {
    let (arch, _) = synth_setup();
    let members = super::keys::LayerMemberSuffixes::from_arch(arch.as_ref()).unwrap();
    assert_eq!(members.q_proj, "self_attn.q_proj.weight");
    assert_eq!(members.q_norm, "self_attn.q_norm.weight");
    assert_eq!(members.down_proj, "mlp.down_proj.weight");
    assert_eq!(
        members.post_attention_layernorm,
        "post_attention_layernorm.weight"
    );
    assert_eq!(members.all().len(), 11);
}

#[test]
fn layer_member_suffixes_require_qk_norms() {
    // A Qwen3-shaped depth transformer needs QK norms; an architecture
    // without them (llama) must be refused by name.
    let llama = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "intermediate_size": 16,
        "num_attention_heads": 2,
    }));
    let err = super::keys::LayerMemberSuffixes::from_arch(llama.as_ref()).unwrap_err();
    assert!(err.to_string().contains("q_norm"), "got: {err}");
}

#[test]
fn layer_member_suffixes_reject_prefix_violating_arch() {
    // Defensive path: an architecture whose key methods do not start
    // with its own layer prefix cannot be decomposed into suffixes.
    struct BadArch {
        config: crate::config::ModelConfig,
    }
    impl ModelArchitecture for BadArch {
        fn family(&self) -> &str {
            "bad"
        }
        fn config(&self) -> &crate::config::ModelConfig {
            &self.config
        }
        fn attn_q_key(&self, _layer: usize) -> String {
            "unprefixed.q_proj.weight".to_string()
        }
        fn attn_q_norm_key(&self, layer: usize) -> Option<String> {
            Some(format!(
                "{}self_attn.q_norm.weight",
                self.layer_prefix(layer)
            ))
        }
        fn attn_k_norm_key(&self, layer: usize) -> Option<String> {
            Some(format!(
                "{}self_attn.k_norm.weight",
                self.layer_prefix(layer)
            ))
        }
    }
    let (arch, _) = synth_setup();
    let bad = BadArch {
        config: arch.config().clone(),
    };
    let err = super::keys::LayerMemberSuffixes::from_arch(&bad).unwrap_err();
    assert!(err.to_string().contains("layer prefix"), "got: {err}");
}

// ── Coverage: the step-1 gate against the real inventory ─────────────────

#[test]
fn real_checkpoint_inventory_is_fully_accounted() {
    let (arch, config) = real_shaped_setup();
    let names: Vec<&str> = REAL_TENSOR_NAMES
        .lines()
        .filter(|l| !l.is_empty())
        .collect();
    assert_eq!(names.len(), 403, "fixture should carry the full inventory");

    let coverage = classify_checkpoint_keys(arch.as_ref(), &config, &names).unwrap();

    assert_eq!(
        coverage.unexplained,
        Vec::<String>::new(),
        "every tensor must be consumed or justified"
    );
    assert_eq!(
        coverage.missing,
        Vec::<String>::new(),
        "every expected tensor must exist"
    );
    // Backbone: per-layer members + embedding + final norm.
    assert_eq!(
        coverage.backbone.len(),
        REAL_BACKBONE_LAYERS * LAYER_MEMBERS + 2
    );
    // Aux: audio tables + local embeds + local layers + final norm + heads.
    assert_eq!(
        coverage.aux.len(),
        REAL_RVQ + (REAL_RVQ - 1) + REAL_LOCAL_LAYERS * LAYER_MEMBERS + 1 + REAL_RVQ
    );
    // Exactly one justified skip: the duplicated text table.
    assert_eq!(coverage.justified_skips.len(), 1);
    assert!(coverage.is_fully_accounted());
}

#[test]
fn coverage_flags_stray_and_missing_keys() {
    let (arch, config) = real_shaped_setup();
    let mut names: Vec<String> = REAL_TENSOR_NAMES
        .lines()
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect();
    let removed = names.remove(0);
    names.push("some.unmodelled.tensor".to_string());

    let coverage = classify_checkpoint_keys(arch.as_ref(), &config, &names).unwrap();
    assert_eq!(
        coverage.unexplained,
        vec!["some.unmodelled.tensor".to_string()]
    );
    assert_eq!(coverage.missing, vec![removed]);
    assert!(!coverage.is_fully_accounted());
}

// ── Loader: synthetic miniature checkpoint ───────────────────────────────

struct SynthTensor {
    name: String,
    shape: Vec<usize>,
    data: Vec<f32>,
}

fn zeros(shape: &[usize]) -> Vec<f32> {
    vec![0.0; shape.iter().product()]
}

/// A rank-2 table whose pad row is zero and every other row is `fill`.
fn table_with_zero_pad_row(rows: usize, cols: usize, pad_row: usize, fill: f32) -> Vec<f32> {
    let mut data = vec![fill; rows * cols];
    data[pad_row * cols..(pad_row + 1) * cols].fill(0.0);
    data
}

fn synth_checkpoint(hidden: usize, rvq: usize, audio_vocab: usize, pad: usize) -> Vec<SynthTensor> {
    let mut tensors = Vec::new();
    let mut push = |name: String, shape: Vec<usize>, data: Vec<f32>| {
        tensors.push(SynthTensor { name, shape, data });
    };
    let text_vocab = 5;
    let text_table: Vec<f32> = (0..text_vocab * hidden).map(|i| i as f32).collect();
    push(
        "language_model.embed_tokens.weight".to_string(),
        vec![text_vocab, hidden],
        text_table.clone(),
    );
    push(
        "embed_tokens.0.weight".to_string(),
        vec![text_vocab, hidden],
        text_table,
    );
    for table in 1..=rvq {
        push(
            format!("embed_tokens.{table}.weight"),
            vec![audio_vocab, hidden],
            table_with_zero_pad_row(audio_vocab, hidden, pad, table as f32),
        );
    }
    for table in 0..rvq - 1 {
        push(
            local_embed_table(table),
            vec![audio_vocab, hidden],
            vec![1.0; audio_vocab * hidden],
        );
    }
    let layer_prefix = local_layer_prefix(0);
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        push(
            format!("{layer_prefix}self_attn.{proj}.weight"),
            vec![hidden, hidden],
            zeros(&[hidden, hidden]),
        );
    }
    for norm in ["q_norm", "k_norm"] {
        push(
            format!("{layer_prefix}self_attn.{norm}.weight"),
            vec![hidden / 2],
            zeros(&[hidden / 2]),
        );
    }
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        push(
            format!("{layer_prefix}mlp.{proj}.weight"),
            vec![hidden * 2, hidden],
            zeros(&[hidden * 2, hidden]),
        );
    }
    for norm in ["input_layernorm", "post_attention_layernorm"] {
        push(
            format!("{layer_prefix}{norm}.weight"),
            vec![hidden],
            zeros(&[hidden]),
        );
    }
    push(local_final_norm(), vec![hidden], zeros(&[hidden]));
    for head in 0..rvq {
        push(
            local_lm_head(head),
            vec![audio_vocab, hidden],
            vec![2.0; audio_vocab * hidden],
        );
    }
    tensors
}

fn write_checkpoint(dir: &std::path::Path, tensors: &[SynthTensor]) {
    let bytes: Vec<Vec<u8>> = tensors
        .iter()
        .map(|t| t.data.iter().flat_map(|v| v.to_le_bytes()).collect())
        .collect();
    let views: Vec<TensorView<'_>> = tensors
        .iter()
        .zip(&bytes)
        .map(|(t, b)| TensorView::new(Dtype::F32, t.shape.clone(), b).unwrap())
        .collect();
    let pairs: Vec<(&str, &TensorView<'_>)> = tensors
        .iter()
        .zip(&views)
        .map(|(t, v)| (t.name.as_str(), v))
        .collect();
    serialize_to_file(pairs, None, &dir.join("model.safetensors")).unwrap();
}

const SYNTH_HIDDEN: usize = 8;
const SYNTH_RVQ: usize = 2;
const SYNTH_AUDIO_VOCAB: usize = 11;
const SYNTH_PAD: usize = 8;

fn synth_setup() -> (Box<dyn ModelArchitecture>, MossTtsRealtimeConfig) {
    let json = config_json(SYNTH_RVQ, SYNTH_AUDIO_VOCAB, SYNTH_PAD, SYNTH_HIDDEN, 1, 1);
    let arch = detect_from_json(&json);
    let config = MossTtsRealtimeConfig::from_config_json(&json).unwrap();
    (arch, config)
}

#[test]
fn loader_roundtrips_synthetic_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let tensors = synth_checkpoint(SYNTH_HIDDEN, SYNTH_RVQ, SYNTH_AUDIO_VOCAB, SYNTH_PAD);
    write_checkpoint(dir.path(), &tensors);
    let (arch, config) = synth_setup();

    let aux = load_moss_tts_aux_from_safetensors(dir.path(), arch.as_ref(), config).unwrap();

    assert_eq!(aux.audio_embed_tables.len(), SYNTH_RVQ);
    assert_eq!(aux.local.embed_tables.len(), SYNTH_RVQ - 1);
    assert_eq!(aux.local.lm_heads.len(), SYNTH_RVQ);
    assert_eq!(aux.local.layers.len(), 1);
    assert_eq!(
        aux.audio_embed_tables[0].dim(),
        (SYNTH_AUDIO_VOCAB, SYNTH_HIDDEN)
    );
    assert_eq!(aux.local.layers[0].q_norm.len(), SYNTH_HIDDEN / 2);
    assert_eq!(aux.local.final_norm.len(), SYNTH_HIDDEN);
    // Non-pad rows carried through the dtype conversion.
    assert_eq!(aux.audio_embed_tables[1][(0, 0)], 2.0);
}

#[test]
fn loader_rejects_nonzero_pad_row() {
    let dir = tempfile::tempdir().unwrap();
    let mut tensors = synth_checkpoint(SYNTH_HIDDEN, SYNTH_RVQ, SYNTH_AUDIO_VOCAB, SYNTH_PAD);
    let table = tensors
        .iter_mut()
        .find(|t| t.name == "embed_tokens.1.weight")
        .unwrap();
    table.data[SYNTH_PAD * SYNTH_HIDDEN] = 0.5;
    write_checkpoint(dir.path(), &tensors);
    let (arch, config) = synth_setup();

    let err = load_moss_tts_aux_from_safetensors(dir.path(), arch.as_ref(), config).unwrap_err();
    assert!(err.to_string().contains("pad row"), "got: {err}");
}

#[test]
fn loader_rejects_broken_text_table_duplication() {
    let dir = tempfile::tempdir().unwrap();
    let mut tensors = synth_checkpoint(SYNTH_HIDDEN, SYNTH_RVQ, SYNTH_AUDIO_VOCAB, SYNTH_PAD);
    let table = tensors
        .iter_mut()
        .find(|t| t.name == "embed_tokens.0.weight")
        .unwrap();
    table.data[0] += 1.0;
    write_checkpoint(dir.path(), &tensors);
    let (arch, config) = synth_setup();

    let err = load_moss_tts_aux_from_safetensors(dir.path(), arch.as_ref(), config).unwrap_err();
    assert!(err.to_string().contains("byte-identical"), "got: {err}");
}

// ── Adapter: the depth transformer as an ordinary Qwen3 model ────────────

#[test]
fn adapter_reexpresses_local_transformer_as_qwen3() {
    let dir = tempfile::tempdir().unwrap();
    let tensors = synth_checkpoint(SYNTH_HIDDEN, SYNTH_RVQ, SYNTH_AUDIO_VOCAB, SYNTH_PAD);
    write_checkpoint(dir.path(), &tensors);
    let (arch, config) = synth_setup();
    let aux =
        load_moss_tts_aux_from_safetensors(dir.path(), arch.as_ref(), config.clone()).unwrap();

    let depth = super::adapter::depth_transformer_model(aux.local, &config).unwrap();

    assert_eq!(depth.weights.arch.family(), "qwen3");
    assert_eq!(depth.weights.num_layers, config.local.num_layers);
    assert_eq!(depth.weights.hidden_size, config.local.hidden_size);
    assert_eq!(depth.weights.vocab_size, config.audio_vocab_size);
    // The core is addressable through the synthesized arch's own keys.
    let inner_arch = &depth.weights.arch;
    assert!(depth
        .weights
        .tensors
        .contains_key(&inner_arch.attn_q_key(0)));
    assert!(depth
        .weights
        .tensors
        .contains_key(&inner_arch.ffn_down_key(0)));
    assert!(depth
        .weights
        .vectors
        .contains_key(&inner_arch.attn_q_norm_key(0).unwrap()));
    assert!(depth
        .weights
        .vectors
        .contains_key(inner_arch.final_norm_key()));
    // The audio boundary stays outside the core.
    assert_eq!(depth.embed_tables.len(), config.local_embed_tables());
    assert_eq!(depth.lm_heads.len(), config.lm_heads());
    // Placeholders are zero-valued: accidental use is conspicuous.
    assert!(depth.weights.embed.iter().all(|&v| v == 0.0));
}

/// Full load of the real checkpoint — assertions (text-table duplication,
/// pad-row zero) run on real bytes. NOT FOR CI: needs the ~4.3 GB local
/// snapshot. Run with:
///   MOSS_TTS_REALTIME_DIR=<snapshot dir> \
///   cargo test -p larql-models real_checkpoint_aux_load -- --ignored
#[test]
#[ignore]
fn real_checkpoint_aux_load() {
    let dir = std::env::var("MOSS_TTS_REALTIME_DIR")
        .expect("set MOSS_TTS_REALTIME_DIR to the checkpoint snapshot directory");
    let config_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(std::path::Path::new(&dir).join("config.json")).unwrap(),
    )
    .unwrap();
    let arch = detect_from_json(&config_json);
    let config = MossTtsRealtimeConfig::from_config_json(&config_json).unwrap();

    let aux = load_moss_tts_aux_from_safetensors(&dir, arch.as_ref(), config.clone()).unwrap();

    assert_eq!(aux.audio_embed_tables.len(), config.backbone_audio_tables());
    assert_eq!(aux.local.embed_tables.len(), config.local_embed_tables());
    assert_eq!(aux.local.lm_heads.len(), config.lm_heads());
    assert_eq!(aux.local.layers.len(), config.local.num_layers);
}

#[test]
fn loader_rejects_empty_directory() {
    let dir = tempfile::tempdir().unwrap();
    let (arch, config) = synth_setup();
    let err = load_moss_tts_aux_from_safetensors(dir.path(), arch.as_ref(), config).unwrap_err();
    assert!(err.to_string().contains("safetensors"), "got: {err}");
}

#[test]
fn loader_rejects_unexpected_rank() {
    let dir = tempfile::tempdir().unwrap();
    let mut tensors = synth_checkpoint(SYNTH_HIDDEN, SYNTH_RVQ, SYNTH_AUDIO_VOCAB, SYNTH_PAD);
    let table = tensors
        .iter_mut()
        .find(|t| t.name == "embed_tokens.1.weight")
        .unwrap();
    table.shape = vec![SYNTH_AUDIO_VOCAB, SYNTH_HIDDEN / 2, 2];
    write_checkpoint(dir.path(), &tensors);
    let (arch, config) = synth_setup();

    let err = load_moss_tts_aux_from_safetensors(dir.path(), arch.as_ref(), config).unwrap_err();
    assert!(err.to_string().contains("rank"), "got: {err}");
}

#[test]
fn loader_requires_backbone_embed_for_dup_check() {
    let dir = tempfile::tempdir().unwrap();
    let mut tensors = synth_checkpoint(SYNTH_HIDDEN, SYNTH_RVQ, SYNTH_AUDIO_VOCAB, SYNTH_PAD);
    tensors.retain(|t| t.name != "language_model.embed_tokens.weight");
    write_checkpoint(dir.path(), &tensors);
    let (arch, config) = synth_setup();

    let err = load_moss_tts_aux_from_safetensors(dir.path(), arch.as_ref(), config).unwrap_err();
    assert!(
        err.to_string()
            .contains("language_model.embed_tokens.weight"),
        "got: {err}"
    );
}

#[test]
fn loader_rejects_pad_token_outside_vocab() {
    let dir = tempfile::tempdir().unwrap();
    let tensors = synth_checkpoint(SYNTH_HIDDEN, SYNTH_RVQ, SYNTH_AUDIO_VOCAB, SYNTH_PAD);
    write_checkpoint(dir.path(), &tensors);
    // Same checkpoint, but a config whose pad id cannot index the tables.
    let json = config_json(
        SYNTH_RVQ,
        SYNTH_AUDIO_VOCAB,
        SYNTH_AUDIO_VOCAB + 5,
        SYNTH_HIDDEN,
        1,
        1,
    );
    let arch = detect_from_json(&json);
    let config = MossTtsRealtimeConfig::from_config_json(&json).unwrap();

    let err = load_moss_tts_aux_from_safetensors(dir.path(), arch.as_ref(), config).unwrap_err();
    assert!(err.to_string().contains("out of range"), "got: {err}");
}

#[test]
fn loader_handles_dup_check_across_files() {
    // The duplication assertion must work when the text table and the
    // backbone embedding live in different shards.
    let dir = tempfile::tempdir().unwrap();
    let mut tensors = synth_checkpoint(SYNTH_HIDDEN, SYNTH_RVQ, SYNTH_AUDIO_VOCAB, SYNTH_PAD);
    let text_table = {
        let position = tensors
            .iter()
            .position(|t| t.name == "embed_tokens.0.weight")
            .unwrap();
        tensors.remove(position)
    };
    write_checkpoint(dir.path(), &tensors);
    // Second shard carrying only the text table.
    let bytes: Vec<u8> = text_table
        .data
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let view = TensorView::new(Dtype::F32, text_table.shape.clone(), &bytes).unwrap();
    serialize_to_file(
        vec![(text_table.name.as_str(), &view)],
        None,
        &dir.path().join("model-2.safetensors"),
    )
    .unwrap();
    let (arch, config) = synth_setup();

    let aux = load_moss_tts_aux_from_safetensors(dir.path(), arch.as_ref(), config).unwrap();
    assert_eq!(aux.audio_embed_tables.len(), SYNTH_RVQ);
}

#[test]
fn loader_reports_missing_aux_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let mut tensors = synth_checkpoint(SYNTH_HIDDEN, SYNTH_RVQ, SYNTH_AUDIO_VOCAB, SYNTH_PAD);
    tensors.retain(|t| t.name != local_lm_head(1));
    write_checkpoint(dir.path(), &tensors);
    let (arch, config) = synth_setup();

    let err = load_moss_tts_aux_from_safetensors(dir.path(), arch.as_ref(), config).unwrap_err();
    assert!(err.to_string().contains(&local_lm_head(1)), "got: {err}");
}
