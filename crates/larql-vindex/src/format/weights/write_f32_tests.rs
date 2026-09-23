//! Colocated tests for `write_f32` — the split-file f32 weight writer.
//!
//! The writer is verified primarily by ROUND-TRIP through the f32
//! loader (`load::f32`): write a small synthetic model's weights, load
//! them back, and require numerical equality plus manifest and
//! index.json correctness. Writer-only branches (MoE experts, MLA
//! absorption, tier gating, error paths) are pinned against the files
//! and manifest directly.

use std::collections::HashMap;
use std::path::Path;

use ndarray::Array2;

use larql_models::ModelWeights;

use super::load::load_model_weights;
use super::write_f32::*;
use crate::config::dtype::{encode_floats, StorageDtype};
use crate::config::types::QuantFormat;
use crate::config::{VindexConfig, VindexLayerInfo};
use crate::extract::callbacks::SilentBuildCallbacks;
use crate::format::filenames::{
    ATTN_WEIGHTS_BIN, DOWN_WEIGHTS_BIN, EMBEDDINGS_BIN, GATE_VECTORS_BIN, INDEX_JSON, LM_HEAD_BIN,
    NORMS_BIN, UP_WEIGHTS_BIN, WEIGHT_MANIFEST_JSON,
};
use crate::index::SilentLoadCallbacks;

// ── Synthetic model geometry ──
const HIDDEN: usize = 4;
const NUM_LAYERS: usize = 2;
const INTERMEDIATE: usize = 6;
const VOCAB: usize = 5;
const NUM_Q_HEADS: usize = 2;
const NUM_KV_HEADS: usize = 1;
const HEAD_DIM: usize = 2;
const GATE_FEATURES: usize = 2;
/// Base offsets so every tensor's values are distinct.
const VALUE_STEP: f32 = 0.25;

fn qwen2_arch_json() -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen2",
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": NUM_Q_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "rope_theta": 10000.0,
        "vocab_size": VOCAB,
    })
}

/// Deterministic distinct fill: `base + i * VALUE_STEP`.
fn fill(rows: usize, cols: usize, base: f32) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |(r, c)| {
        base + (r * cols + c) as f32 * VALUE_STEP
    })
}

fn empty_model_weights(arch_json: &serde_json::Value) -> ModelWeights {
    let arch = larql_models::detect_from_json(arch_json);
    let cfg = arch.config();
    let embed = fill(VOCAB, HIDDEN, 900.0);
    let lm_head = fill(VOCAB, HIDDEN, 950.0);
    ModelWeights {
        tensors: HashMap::new(),
        vectors: HashMap::new(),
        raw_bytes: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: HashMap::new(),
        packed_byte_ranges: HashMap::new(),
        embed: embed.into_shared(),
        lm_head: lm_head.into_shared(),
        position_embed: None,
        num_layers: cfg.num_layers,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        vocab_size: VOCAB,
        head_dim: cfg.head_dim,
        num_q_heads: cfg.num_q_heads,
        num_kv_heads: cfg.num_kv_heads,
        rope_base: cfg.rope_base,
        arch,
    }
}

/// Fully-populated qwen2 model: attention projections + biases, norms,
/// dense FFN, distinct lm_head/embed.
fn qwen2_model_weights() -> ModelWeights {
    let mut w = empty_model_weights(&qwen2_arch_json());
    let q_dim = NUM_Q_HEADS * HEAD_DIM;
    let kv_dim = NUM_KV_HEADS * HEAD_DIM;
    for layer in 0..NUM_LAYERS {
        let base = layer as f32 * 100.0;
        let arch = &w.arch;
        let tensors: Vec<(String, Array2<f32>)> = vec![
            (arch.attn_q_key(layer), fill(q_dim, HIDDEN, base + 1.0)),
            (arch.attn_k_key(layer), fill(kv_dim, HIDDEN, base + 2.0)),
            (arch.attn_v_key(layer), fill(kv_dim, HIDDEN, base + 3.0)),
            (arch.attn_o_key(layer), fill(HIDDEN, q_dim, base + 4.0)),
            (
                arch.ffn_up_key(layer),
                fill(INTERMEDIATE, HIDDEN, base + 5.0),
            ),
            (
                arch.ffn_down_key(layer),
                fill(HIDDEN, INTERMEDIATE, base + 6.0),
            ),
        ];
        let vectors: Vec<(Option<String>, usize, f32)> = vec![
            (arch.attn_q_bias_key(layer), q_dim, base + 7.0),
            (arch.attn_k_bias_key(layer), kv_dim, base + 8.0),
            (arch.attn_v_bias_key(layer), kv_dim, base + 9.0),
            (Some(arch.input_layernorm_key(layer)), HIDDEN, base + 10.0),
            (
                Some(arch.post_attention_layernorm_key(layer)),
                HIDDEN,
                base + 11.0,
            ),
        ];
        for (key, t) in tensors {
            w.tensors.insert(key, t.into_shared());
        }
        for (key, len, b) in vectors {
            let key = key.expect("qwen2 declares this vector key");
            w.vectors
                .insert(key, (0..len).map(|i| b + i as f32 * VALUE_STEP).collect());
        }
    }
    let final_norm = w.arch.final_norm_key().to_string();
    w.vectors
        .insert(final_norm, (0..HIDDEN).map(|i| 800.0 + i as f32).collect());
    w
}

fn gate_layer_matrix(layer: usize) -> Vec<f32> {
    (0..GATE_FEATURES * HIDDEN)
        .map(|i| 500.0 + layer as f32 * 50.0 + i as f32)
        .collect()
}

/// Write the index.json + embeddings.bin + gate_vectors.bin the writer
/// and loader expect around the weight files.
fn write_vindex_scaffolding(dir: &Path, weights: &ModelWeights) {
    let layer_bytes = (GATE_FEATURES * HIDDEN * 4) as u64;
    let config = VindexConfig {
        version: 2,
        model: "test/write-f32".into(),
        family: "test".into(),
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN,
        intermediate_size: INTERMEDIATE,
        vocab_size: VOCAB,
        embed_scale: 1.0,
        layers: (0..NUM_LAYERS)
            .map(|layer| VindexLayerInfo {
                layer,
                num_features: GATE_FEATURES,
                offset: layer as u64 * layer_bytes,
                length: layer_bytes,
                num_experts: None,
                num_features_per_expert: None,
            })
            .collect(),
        down_top_k: 1,
        has_model_weights: false,
        source: None,
        checksums: None,
        extract_level: crate::ExtractLevel::All,
        dtype: StorageDtype::F32,
        quant: QuantFormat::None,
        layer_bands: crate::LayerBands::for_family("test", NUM_LAYERS),
        model_config: None,
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    };
    std::fs::write(
        dir.join(INDEX_JSON),
        serde_json::to_string_pretty(&config).unwrap(),
    )
    .unwrap();
    let embed_floats: Vec<f32> = weights.embed.iter().copied().collect();
    std::fs::write(
        dir.join(EMBEDDINGS_BIN),
        encode_floats(&embed_floats, StorageDtype::F32),
    )
    .unwrap();
    let gate: Vec<f32> = (0..NUM_LAYERS).flat_map(gate_layer_matrix).collect();
    std::fs::write(
        dir.join(GATE_VECTORS_BIN),
        encode_floats(&gate, StorageDtype::F32),
    )
    .unwrap();
}

fn manifest_entries(dir: &Path) -> Vec<WeightEntry> {
    let text = std::fs::read_to_string(dir.join(WEIGHT_MANIFEST_JSON)).unwrap();
    serde_json::from_str(&text).unwrap()
}

// ── Round trip ──

#[test]
fn round_trip_through_f32_loader_is_numerically_exact() {
    let tmp = tempfile::tempdir().unwrap();
    let weights = qwen2_model_weights();
    write_vindex_scaffolding(tmp.path(), &weights);

    write_model_weights(&weights, tmp.path(), &mut SilentBuildCallbacks).expect("write");
    let loaded = load_model_weights(tmp.path(), &mut SilentLoadCallbacks).expect("load");

    // Every written 2D tensor comes back bit-exact.
    for (key, tensor) in &weights.tensors {
        let got = loaded
            .tensors
            .get(key)
            .unwrap_or_else(|| panic!("tensor {key} missing after round trip"));
        assert_eq!(got.shape(), tensor.shape(), "{key} shape");
        assert_eq!(
            got.as_slice().unwrap(),
            tensor.as_slice().unwrap(),
            "{key} data"
        );
    }
    // Every written 1D vector (norms + qwen2 attention biases).
    for (key, vector) in &weights.vectors {
        assert_eq!(loaded.vectors.get(key), Some(vector), "{key}");
    }
    // lm_head + embed.
    assert_eq!(
        loaded.lm_head.as_slice().unwrap(),
        weights.lm_head.as_slice().unwrap()
    );
    assert_eq!(
        loaded.embed.as_slice().unwrap(),
        weights.embed.as_slice().unwrap()
    );
    // Gate vectors surface as FFN gate tensors.
    for layer in 0..NUM_LAYERS {
        let key = weights.arch.ffn_gate_key(layer);
        let got = loaded.tensors.get(&key).expect("gate tensor");
        assert_eq!(got.shape(), &[GATE_FEATURES, HIDDEN]);
        assert_eq!(got.as_slice().unwrap(), gate_layer_matrix(layer));
    }
}

#[test]
fn manifest_and_config_are_correct_after_write() {
    let tmp = tempfile::tempdir().unwrap();
    let weights = qwen2_model_weights();
    write_vindex_scaffolding(tmp.path(), &weights);
    write_model_weights(&weights, tmp.path(), &mut SilentBuildCallbacks).unwrap();

    let entries = manifest_entries(tmp.path());
    // 6 tensors + 5 vectors per layer, + final norm + lm_head.
    assert_eq!(entries.len(), NUM_LAYERS * 11 + 2);

    // Per-file contiguity: offsets start at 0 and are gapless, and each
    // entry's byte length matches its f32 shape.
    let mut next_offset: HashMap<&str, u64> = HashMap::new();
    for e in &entries {
        assert!(!e.file.is_empty(), "{} has no file", e.key);
        assert!(
            e.kind == kind::TENSOR || e.kind == kind::VECTOR,
            "{} kind {}",
            e.key,
            e.kind
        );
        let expected_len = e.shape.iter().product::<usize>() as u64 * 4;
        assert_eq!(e.length, expected_len, "{} length", e.key);
        let cursor = next_offset.entry(e.file.as_str()).or_insert(0);
        assert_eq!(e.offset, *cursor, "{} offset in {}", e.key, e.file);
        *cursor += e.length;
    }
    // Physical file sizes match the manifest's account of them.
    for (file, total) in &next_offset {
        let on_disk = std::fs::metadata(tmp.path().join(file)).unwrap().len();
        assert_eq!(on_disk, *total, "{file} size");
    }

    // index.json updated in place.
    let config: VindexConfig =
        serde_json::from_str(&std::fs::read_to_string(tmp.path().join(INDEX_JSON)).unwrap())
            .unwrap();
    assert!(config.has_model_weights);
    let model_cfg = config.model_config.expect("model_config recorded");
    assert_eq!(model_cfg.model_type, "qwen2");
    assert_eq!(model_cfg.num_q_heads, NUM_Q_HEADS);
    assert_eq!(model_cfg.num_kv_heads, NUM_KV_HEADS);
}

// ── Tier / skip gating ──

#[test]
fn attention_level_skips_ffn_and_lm_head() {
    let tmp = tempfile::tempdir().unwrap();
    let weights = qwen2_model_weights();
    write_vindex_scaffolding(tmp.path(), &weights);
    let opts = WriteWeightsOptions {
        level: crate::ExtractLevel::Attention,
        ..Default::default()
    };
    write_model_weights_with_opts(&weights, tmp.path(), &mut SilentBuildCallbacks, opts).unwrap();

    assert!(tmp.path().join(ATTN_WEIGHTS_BIN).exists());
    assert!(tmp.path().join(NORMS_BIN).exists());
    assert!(!tmp.path().join(UP_WEIGHTS_BIN).exists());
    assert!(!tmp.path().join(DOWN_WEIGHTS_BIN).exists());
    assert!(!tmp.path().join(LM_HEAD_BIN).exists());
}

#[test]
fn skip_attn_still_writes_norms() {
    let tmp = tempfile::tempdir().unwrap();
    let weights = qwen2_model_weights();
    write_vindex_scaffolding(tmp.path(), &weights);
    let opts = WriteWeightsOptions {
        skip_attn: true,
        ..Default::default()
    };
    write_model_weights_with_opts(&weights, tmp.path(), &mut SilentBuildCallbacks, opts).unwrap();

    assert!(!tmp.path().join(ATTN_WEIGHTS_BIN).exists());
    assert!(tmp.path().join(NORMS_BIN).exists());
    assert!(tmp.path().join(UP_WEIGHTS_BIN).exists());
    // Norm entries survive; attention projections are absent.
    let entries = manifest_entries(tmp.path());
    assert!(entries.iter().all(|e| e.file != ATTN_WEIGHTS_BIN));
    assert!(entries.iter().any(|e| e.file == NORMS_BIN));
}

#[test]
fn ffn_compact_and_skip_ffn_suppress_up_down_files() {
    for opts in [
        WriteWeightsOptions {
            ffn_compact: true,
            ..Default::default()
        },
        WriteWeightsOptions {
            skip_ffn: true,
            ..Default::default()
        },
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let weights = qwen2_model_weights();
        write_vindex_scaffolding(tmp.path(), &weights);
        write_model_weights_with_opts(&weights, tmp.path(), &mut SilentBuildCallbacks, opts)
            .unwrap();
        assert!(!tmp.path().join(UP_WEIGHTS_BIN).exists());
        assert!(!tmp.path().join(DOWN_WEIGHTS_BIN).exists());
        assert!(tmp.path().join(ATTN_WEIGHTS_BIN).exists());
    }
}

// ── WeightSource impl on ModelWeights ──

#[test]
fn model_weights_source_trait_surface() {
    let mut weights = qwen2_model_weights();
    weights
        .raw_bytes
        .insert("packed.experts".into(), vec![1, 2, 3, 4]);
    let source: &dyn WeightSource = &weights;

    let q_key = weights.arch.attn_q_key(0);
    let (data, rows, cols) = source.get_tensor(&q_key).expect("q tensor");
    assert_eq!((rows, cols), (NUM_Q_HEADS * HEAD_DIM, HIDDEN));
    assert_eq!(data.len(), rows * cols);
    assert!(source.get_tensor("no.such.tensor").is_none());

    let norm_key = weights.arch.final_norm_key().to_string();
    assert!(source.get_vector(&norm_key).is_some());
    assert!(source.get_vector("no.such.vector").is_none());

    assert_eq!(source.num_layers(), NUM_LAYERS);
    let (_, head_rows, head_cols) = source.lm_head().expect("lm_head");
    assert_eq!((head_rows, head_cols), (VOCAB, HIDDEN));

    let names = source.vector_names();
    assert!(names.contains(&norm_key));
    assert_eq!(names.len(), weights.vectors.len());

    assert_eq!(
        source.get_packed_bf16("packed.experts"),
        Some(vec![1, 2, 3, 4])
    );
    assert!(source.get_packed_bf16("absent").is_none());
}

// ── Sparse-source writer behavior ──

/// Missing tensors are skipped (manifest simply has no entry), and the
/// BitNet `*_sub_norm.weight` vectors are picked up from the source map
/// even though no arch norm-key lists them.
#[test]
fn missing_tensors_skip_and_sub_norms_are_captured() {
    let tmp = tempfile::tempdir().unwrap();
    let mut weights = qwen2_model_weights();
    // Drop layer 1's FFN entirely — dense writer must skip, not fail.
    let up1 = weights.arch.ffn_up_key(1);
    let down1 = weights.arch.ffn_down_key(1);
    weights.tensors.remove(&up1);
    weights.tensors.remove(&down1);
    // BitNet-style sub-layer norms on layer 0 (keyed off layer_prefix).
    let attn_sub = format!("{}attn_sub_norm.weight", weights.arch.layer_prefix(0));
    let ffn_sub = format!("{}ffn_sub_norm.weight", weights.arch.layer_prefix(0));
    weights.vectors.insert(attn_sub.clone(), vec![1.0; HIDDEN]);
    weights.vectors.insert(ffn_sub.clone(), vec![2.0; HIDDEN]);

    write_vindex_scaffolding(tmp.path(), &weights);
    write_model_weights(&weights, tmp.path(), &mut SilentBuildCallbacks).unwrap();

    let entries = manifest_entries(tmp.path());
    assert!(entries.iter().all(|e| e.key != up1 && e.key != down1));
    assert!(entries.iter().any(|e| e.key == weights.arch.ffn_up_key(0)));
    for key in [&attn_sub, &ffn_sub] {
        let e = entries
            .iter()
            .find(|e| &e.key == key)
            .expect("sub-norm entry");
        assert_eq!(e.file, NORMS_BIN);
        assert_eq!(e.kind, kind::VECTOR);
    }
}

// ── Error paths ──

#[test]
fn mla_without_full_geometry_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    // deepseek_v2 with lora ranks but NO qk_nope/qk_rope/v_head_dim —
    // absorption cannot run, the standard-attention guard must fire.
    let weights = empty_model_weights(&serde_json::json!({
        "model_type": "deepseek_v2",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": NUM_Q_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "kv_lora_rank": 4,
        "q_lora_rank": 4,
        "vocab_size": VOCAB,
    }));
    assert!(weights.arch.uses_mla());
    let err = write_model_weights(&weights, tmp.path(), &mut SilentBuildCallbacks)
        .expect_err("MLA without geometry must be rejected");
    assert!(err.to_string().contains("latent attention"), "{err}");
}

#[test]
fn missing_index_json_is_an_error() {
    let tmp = tempfile::tempdir().unwrap();
    let weights = qwen2_model_weights();
    // No scaffolding: the final config update has nothing to read.
    let err = write_model_weights(&weights, tmp.path(), &mut SilentBuildCallbacks)
        .expect_err("must fail without index.json");
    assert!(!err.to_string().is_empty());
}

#[test]
fn ffn_compact_on_moe_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let weights = empty_model_weights(&serde_json::json!({
        "model_type": "mixtral",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": NUM_Q_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "num_local_experts": 2,
        "num_experts_per_tok": 2,
        "vocab_size": VOCAB,
    }));
    let opts = WriteWeightsOptions {
        ffn_compact: true,
        ..Default::default()
    };
    let err = write_model_weights_with_opts(&weights, tmp.path(), &mut SilentBuildCallbacks, opts)
        .expect_err("MoE + ffn_compact must be refused");
    assert!(err.to_string().contains("ffn_compact"), "{err}");
}

// ── MoE expert writing ──

#[test]
fn moe_writer_emits_expert_and_router_entries() {
    const NUM_EXPERTS: usize = 2;
    // Two layers, but tensors only on layer 0 — layer 1 exercises the
    // "expert key resolves but tensor is absent" skip branches.
    const MOE_LAYERS: usize = 2;
    let tmp = tempfile::tempdir().unwrap();
    let mut weights = empty_model_weights(&serde_json::json!({
        "model_type": "mixtral",
        "hidden_size": HIDDEN,
        "num_hidden_layers": MOE_LAYERS,
        "intermediate_size": INTERMEDIATE,
        "num_attention_heads": NUM_Q_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "num_local_experts": NUM_EXPERTS,
        "num_experts_per_tok": 2,
        "vocab_size": VOCAB,
    }));
    for expert in 0..NUM_EXPERTS {
        let base = 10.0 * expert as f32;
        let up_key = weights.arch.expert_ffn_up_key(0, expert).unwrap();
        let down_key = weights.arch.expert_ffn_down_key(0, expert).unwrap();
        weights
            .tensors
            .insert(up_key, fill(INTERMEDIATE, HIDDEN, base + 1.0).into_shared());
        weights.tensors.insert(
            down_key,
            fill(HIDDEN, INTERMEDIATE, base + 2.0).into_shared(),
        );
    }
    let router_key = weights.arch.moe_router_key(0).unwrap();
    weights.tensors.insert(
        router_key.clone(),
        fill(NUM_EXPERTS, HIDDEN, 70.0).into_shared(),
    );

    // Minimal index.json (the writer's final config update needs it).
    write_vindex_scaffolding(tmp.path(), &qwen2_model_weights());

    write_model_weights(&weights, tmp.path(), &mut SilentBuildCallbacks).unwrap();
    let entries = manifest_entries(tmp.path());

    // Experts' up tensors + router live in up_weights.bin; down tensors
    // in down_weights.bin. Verify bytes at the recorded offsets.
    let up_bytes = std::fs::read(tmp.path().join(UP_WEIGHTS_BIN)).unwrap();
    let down_bytes = std::fs::read(tmp.path().join(DOWN_WEIGHTS_BIN)).unwrap();
    for expert in 0..NUM_EXPERTS {
        for (key, file_bytes, file) in [
            (
                weights.arch.expert_ffn_up_key(0, expert).unwrap(),
                &up_bytes,
                UP_WEIGHTS_BIN,
            ),
            (
                weights.arch.expert_ffn_down_key(0, expert).unwrap(),
                &down_bytes,
                DOWN_WEIGHTS_BIN,
            ),
        ] {
            let entry = entries
                .iter()
                .find(|e| e.key == key)
                .unwrap_or_else(|| panic!("no manifest entry for {key}"));
            assert_eq!(entry.file, file);
            let raw = &file_bytes[entry.offset as usize..(entry.offset + entry.length) as usize];
            let decoded = crate::config::dtype::decode_floats(raw, StorageDtype::F32);
            let expected = weights.tensors.get(&key).unwrap();
            assert_eq!(decoded, expected.as_slice().unwrap());
        }
    }
    let router = entries.iter().find(|e| e.key == router_key).unwrap();
    assert_eq!(router.file, UP_WEIGHTS_BIN);
    assert_eq!(router.shape, vec![NUM_EXPERTS, HIDDEN]);
    // Layer 1 contributed nothing.
    let layer1_router = weights.arch.moe_router_key(1).unwrap();
    assert!(entries.iter().all(|e| e.key != layer1_router));
}

// ── MLA absorption ──

#[test]
fn mla_projections_are_absorbed_into_standard_qkvo() {
    // DeepSeek-V2-style geometry, tiny: 2 Q heads over 1 KV head.
    const MLA_HIDDEN: usize = 8;
    const QK_NOPE: usize = 2;
    const QK_ROPE: usize = 2;
    const V_HEAD: usize = 2;
    const KV_LORA: usize = 4;
    const Q_LORA: usize = 4;
    const QK_HD: usize = QK_NOPE + QK_ROPE;

    let tmp = tempfile::tempdir().unwrap();
    let mut weights = empty_model_weights(&serde_json::json!({
        "model_type": "deepseek_v2",
        "hidden_size": MLA_HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": MLA_HIDDEN,
        "num_attention_heads": NUM_Q_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": QK_HD,
        "kv_lora_rank": KV_LORA,
        "q_lora_rank": Q_LORA,
        "qk_nope_head_dim": QK_NOPE,
        "qk_rope_head_dim": QK_ROPE,
        "v_head_dim": V_HEAD,
        "vocab_size": VOCAB,
    }));
    assert!(weights.arch.uses_mla(), "fixture must exercise MLA");
    let inserts = [
        (
            weights.arch.mla_kv_a_key(0).unwrap(),
            fill(KV_LORA + QK_ROPE, MLA_HIDDEN, 1.0),
        ),
        (
            weights.arch.mla_kv_b_key(0).unwrap(),
            fill(NUM_KV_HEADS * (QK_NOPE + V_HEAD), KV_LORA, 2.0),
        ),
        (
            weights.arch.mla_q_a_key(0).unwrap(),
            fill(Q_LORA, MLA_HIDDEN, 3.0),
        ),
        (
            weights.arch.mla_q_b_key(0).unwrap(),
            fill(NUM_Q_HEADS * QK_HD, Q_LORA, 4.0),
        ),
        (
            weights.arch.attn_o_key(0),
            fill(MLA_HIDDEN, NUM_Q_HEADS * V_HEAD, 5.0),
        ),
    ];
    for (key, t) in inserts {
        weights.tensors.insert(key, t.into_shared());
    }

    // Attention-only tier: no FFN / lm_head needed, no index.json update
    // hazards beyond the standard one.
    let scaffold = qwen2_model_weights();
    write_vindex_scaffolding(tmp.path(), &scaffold);
    let opts = WriteWeightsOptions {
        level: crate::ExtractLevel::Attention,
        ..Default::default()
    };
    write_model_weights_with_opts(&weights, tmp.path(), &mut SilentBuildCallbacks, opts).unwrap();

    let entries = manifest_entries(tmp.path());
    let shape_of = |key: &str| -> Vec<usize> {
        entries
            .iter()
            .find(|e| e.key == key)
            .unwrap_or_else(|| panic!("missing {key}"))
            .shape
            .clone()
    };
    // Absorbed tensors land under the standard names with dense shapes.
    assert_eq!(
        shape_of(&weights.arch.attn_q_key(0)),
        vec![NUM_Q_HEADS * QK_HD, MLA_HIDDEN]
    );
    assert_eq!(
        shape_of(&weights.arch.attn_k_key(0)),
        vec![NUM_KV_HEADS * QK_HD, MLA_HIDDEN]
    );
    assert_eq!(
        shape_of(&weights.arch.attn_v_key(0)),
        vec![NUM_KV_HEADS * V_HEAD, MLA_HIDDEN]
    );
    // O is passed through untouched — verify bytes.
    let o_key = weights.arch.attn_o_key(0);
    let o_entry = entries.iter().find(|e| e.key == o_key).unwrap();
    let attn_bytes = std::fs::read(tmp.path().join(ATTN_WEIGHTS_BIN)).unwrap();
    let raw = &attn_bytes[o_entry.offset as usize..(o_entry.offset + o_entry.length) as usize];
    let decoded = crate::config::dtype::decode_floats(raw, StorageDtype::F32);
    assert_eq!(
        decoded,
        weights.tensors.get(&o_key).unwrap().as_slice().unwrap()
    );
}

// ── StreamingWeights ──

mod streaming {
    use super::*;
    use safetensors::tensor::TensorView;
    use safetensors::Dtype;

    const Q_KEY: &str = "model.layers.0.self_attn.q_proj.weight";
    const NORM_KEY: &str = "model.norm.weight";
    const OUTPUT_KEY: &str = "output.weight";
    const BF16_KEY: &str = "experts.packed";
    const Q_ROWS: usize = 2;
    const Q_COLS: usize = 3;

    fn f32_bytes(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// One serialized safetensors shard with a 2D f32, 1D f32, lm_head
    /// fallback, and a BF16 packed tensor.
    fn shard() -> (Vec<u8>, HashMap<String, (usize, String)>) {
        let q = f32_bytes(&(0..Q_ROWS * Q_COLS).map(|i| i as f32).collect::<Vec<_>>());
        let norm = f32_bytes(&[1.0, 2.0, 3.0]);
        let output = f32_bytes(&[4.0, 5.0, 6.0, 7.0]);
        let bf16 = vec![0x80u8, 0x3f, 0x00, 0x40, 0x40, 0x40, 0x80, 0x40];
        let views = vec![
            (
                Q_KEY,
                TensorView::new(Dtype::F32, vec![Q_ROWS, Q_COLS], &q).unwrap(),
            ),
            (
                NORM_KEY,
                TensorView::new(Dtype::F32, vec![3], &norm).unwrap(),
            ),
            (
                OUTPUT_KEY,
                TensorView::new(Dtype::F32, vec![2, 2], &output).unwrap(),
            ),
            (
                BF16_KEY,
                TensorView::new(Dtype::BF16, vec![2, 2], &bf16).unwrap(),
            ),
        ];
        let blob = safetensors::serialize(views, None).unwrap();
        let index = [Q_KEY, NORM_KEY, OUTPUT_KEY, BF16_KEY]
            .into_iter()
            .map(|k| (k.to_string(), (0usize, k.to_string())))
            .collect();
        (blob, index)
    }

    #[test]
    fn streaming_source_reads_tensors_vectors_and_lm_head() {
        let (blob, index) = shard();
        let arch = larql_models::detect_from_json(&qwen2_arch_json());
        let shards: Vec<&[u8]> = vec![&blob];
        let source = StreamingWeights {
            shard_mmaps: &shards,
            tensor_index: &index,
            arch: &*arch,
            num_layers: 1,
        };

        let (data, rows, cols) = source.get_tensor(Q_KEY).expect("2D tensor");
        assert_eq!((rows, cols), (Q_ROWS, Q_COLS));
        assert_eq!(
            data,
            (0..Q_ROWS * Q_COLS).map(|i| i as f32).collect::<Vec<_>>()
        );
        // Rank mismatches return None rather than mis-shaping.
        assert!(source.get_tensor(NORM_KEY).is_none());
        assert!(source.get_vector(Q_KEY).is_none());
        assert_eq!(source.get_vector(NORM_KEY), Some(vec![1.0, 2.0, 3.0]));
        assert!(source.get_tensor("nope").is_none());

        // lm_head falls back from lm_head.weight to output.weight.
        let (head, r, c) = source.lm_head().expect("lm_head fallback");
        assert_eq!((r, c), (2, 2));
        assert_eq!(head, vec![4.0, 5.0, 6.0, 7.0]);

        // vector_names picks norm/bias-shaped keys only, sorted.
        assert_eq!(source.vector_names(), vec![NORM_KEY.to_string()]);
        assert_eq!(source.num_layers(), 1);
        assert_eq!(source.arch().config().hidden_size, HIDDEN);
    }

    #[test]
    fn streaming_decodes_f16_bf16_and_rejects_unsupported_dtypes() {
        // f16 1.0 = 0x3C00, 2.0 = 0x4000; bf16 1.0 = 0x3F80, 2.0 = 0x4000.
        let f16 = vec![0x00u8, 0x3C, 0x00, 0x40];
        let bf16 = vec![0x80u8, 0x3F, 0x00, 0x40];
        let ints = vec![0u8; 8];
        let views = vec![
            (
                "half",
                TensorView::new(Dtype::F16, vec![1, 2], &f16).unwrap(),
            ),
            (
                "brain",
                TensorView::new(Dtype::BF16, vec![1, 2], &bf16).unwrap(),
            ),
            (
                "ints",
                TensorView::new(Dtype::I32, vec![1, 2], &ints).unwrap(),
            ),
        ];
        let blob = safetensors::serialize(views, None).unwrap();
        let index: HashMap<String, (usize, String)> = ["half", "brain", "ints"]
            .into_iter()
            .map(|k| (k.to_string(), (0usize, k.to_string())))
            .collect();
        let arch = larql_models::detect_from_json(&qwen2_arch_json());
        let shards: Vec<&[u8]> = vec![&blob];
        let source = StreamingWeights {
            shard_mmaps: &shards,
            tensor_index: &index,
            arch: &*arch,
            num_layers: 1,
        };

        assert_eq!(source.get_tensor("half"), Some((vec![1.0, 2.0], 1, 2)));
        assert_eq!(source.get_tensor("brain"), Some((vec![1.0, 2.0], 1, 2)));
        assert!(source.get_tensor("ints").is_none(), "I32 is unsupported");
        // No lm_head.weight / output.weight anywhere → None.
        assert!(source.lm_head().is_none());
    }

    #[test]
    fn streaming_packed_bf16_requires_bf16_dtype() {
        let (blob, index) = shard();
        let arch = larql_models::detect_from_json(&qwen2_arch_json());
        let shards: Vec<&[u8]> = vec![&blob];
        let source = StreamingWeights {
            shard_mmaps: &shards,
            tensor_index: &index,
            arch: &*arch,
            num_layers: 1,
        };
        let raw = source.get_packed_bf16(BF16_KEY).expect("bf16 bytes");
        assert_eq!(raw.len(), 2 * 2 * 2);
        assert!(
            source.get_packed_bf16(Q_KEY).is_none(),
            "f32 is not packed bf16"
        );
        assert!(source.get_packed_bf16("missing").is_none());
    }
}
