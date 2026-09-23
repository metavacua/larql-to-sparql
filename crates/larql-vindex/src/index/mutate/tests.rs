//! Colocated tests for the INSERT / DELETE / save-to-disk mutation paths.
//!
//! Covers both storage modes: heap indexes built via `VectorIndex::new`
//! and mmap indexes produced by `save_vindex` → `load_vindex` round
//! trips into a `tempfile::tempdir`.

use std::path::Path;

use ndarray::{Array1, Array2};

use crate::config::dtype::StorageDtype;
use crate::config::types::QuantFormat;
use crate::config::{VindexConfig, VindexLayerInfo};
use crate::format::filenames::{DOWN_META_BIN, GATE_VECTORS_BIN, INDEX_JSON};
use crate::index::{FeatureMeta, SilentLoadCallbacks, VectorIndex};

// Fixture geometry — small enough to eyeball, big enough to exercise
// multi-layer / multi-feature paths.
const NUM_LAYERS: usize = 2;
const HIDDEN: usize = 4;
const NUM_FEATURES: usize = 3;

/// Fixed vocab for the round-trip tokenizer. `down_meta.bin` stores
/// token IDs only; `load_vindex` re-derives the strings by decoding
/// through tokenizer.json, so metas and the tokenizer must agree.
/// Index 0 is reserved for `[UNK]` (id 0 also means "empty slot" in
/// the binary format).
const VOCAB: &[&str] = &[
    "[UNK]",
    "paris",
    "rome",
    "tokyo",
    "lima",
    "oslo",
    "madrid",
    "nairobi",
    "berlin",
    "paris_alt",
    "rome_alt",
    "tokyo_alt",
    "lima_alt",
    "oslo_alt",
    "madrid_alt",
    "nairobi_alt",
    "berlin_alt",
];

fn token_id(token: &str) -> u32 {
    VOCAB
        .iter()
        .position(|t| *t == token)
        .unwrap_or_else(|| panic!("{token} not in test vocab")) as u32
}

/// Write a WordLevel tokenizer.json over `VOCAB` so `load_vindex` can
/// decode the ids in down_meta.bin back to the original strings.
fn write_tokenizer(dir: &Path) {
    let vocab_json: String = VOCAB
        .iter()
        .enumerate()
        .map(|(id, tok)| format!("\"{tok}\": {id}"))
        .collect::<Vec<_>>()
        .join(", ");
    let json = format!(
        r#"{{
            "version": "1.0",
            "model": {{"type": "WordLevel", "vocab": {{{vocab_json}}}, "unk_token": "[UNK]"}},
            "pre_tokenizer": {{"type": "Whitespace"}},
            "added_tokens": []
        }}"#
    );
    std::fs::write(dir.join("tokenizer.json"), json).unwrap();
}

fn meta(token: &str, c_score: f32) -> FeatureMeta {
    FeatureMeta {
        top_token: token.to_string(),
        top_token_id: token_id(token),
        c_score,
        top_k: vec![larql_models::TopKEntry {
            token: format!("{token}_alt"),
            token_id: token_id(&format!("{token}_alt")),
            logit: 1.0,
        }],
    }
}

/// Deterministic gate value for (layer, feature, dim) — distinct per slot.
fn gate_val(layer: usize, feature: usize, dim: usize) -> f32 {
    (layer * 100 + feature * 10 + dim) as f32
}

fn gate_matrix(layer: usize) -> Array2<f32> {
    Array2::from_shape_fn((NUM_FEATURES, HIDDEN), |(f, d)| gate_val(layer, f, d))
}

/// Heap-mode index: full gate matrices on both layers; metadata on
/// layer 0 = [paris, <empty>, rome], layer 1 = [tokyo, lima, oslo].
fn heap_index() -> VectorIndex {
    let gate = vec![Some(gate_matrix(0)), Some(gate_matrix(1))];
    let down_meta = vec![
        Some(vec![
            Some(meta("paris", 5.0)),
            None,
            Some(meta("rome", 3.0)),
        ]),
        Some(vec![
            Some(meta("tokyo", 5.0)),
            Some(meta("lima", 1.0)),
            Some(meta("oslo", 3.0)),
        ]),
    ];
    VectorIndex::new(gate, down_meta, NUM_LAYERS, HIDDEN)
}

fn test_config() -> VindexConfig {
    VindexConfig {
        version: 2,
        model: "test/mutate-model".into(),
        family: "test".into(),
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN,
        intermediate_size: HIDDEN * 2,
        vocab_size: 16,
        embed_scale: 1.0,
        layers: Vec::new(),
        down_top_k: 1,
        has_model_weights: false,
        source: None,
        checksums: None,
        extract_level: crate::ExtractLevel::Browse,
        dtype: StorageDtype::F32,
        quant: QuantFormat::None,
        layer_bands: crate::LayerBands::for_family("test", NUM_LAYERS),
        model_config: None,
        fp4: None,
        ffn_layout: None,
        bitnet_layout: None,
    }
}

/// Save a heap fixture and reload it in mmap mode.
fn saved_and_reloaded(dir: &Path) -> (VectorIndex, VindexConfig) {
    let idx = heap_index();
    let mut config = test_config();
    idx.save_vindex(dir, &mut config).expect("save_vindex");
    write_tokenizer(dir);
    let loaded = VectorIndex::load_vindex(dir, &mut SilentLoadCallbacks).expect("load_vindex");
    (loaded, config)
}

// ── Metadata mutation (heap) ──

#[test]
fn set_feature_meta_grows_layers_and_features() {
    let mut idx = VectorIndex::new(vec![None; NUM_LAYERS], Vec::new(), NUM_LAYERS, HIDDEN);
    // Layer 1 / feature 2 forces growth of both the layer slot vec and
    // the per-layer feature vec.
    idx.set_feature_meta(1, 2, meta("berlin", 2.0));
    let got = idx.feature_meta(1, 2).expect("meta stored");
    assert_eq!(got.top_token, "berlin");
    assert_eq!(got.c_score, 2.0);
    // Intermediate slots padded with None.
    assert!(idx.feature_meta(1, 0).is_none());
    assert!(idx.feature_meta(0, 0).is_none());
}

#[test]
fn delete_feature_meta_clears_and_ignores_out_of_range() {
    let mut idx = heap_index();
    assert!(idx.feature_meta(0, 0).is_some());
    idx.delete_feature_meta(0, 0);
    assert!(idx.feature_meta(0, 0).is_none());
    // Out-of-range feature and layer are silent no-ops.
    idx.delete_feature_meta(0, 99);
    idx.delete_feature_meta(99, 0);
}

#[test]
fn delete_then_query_no_longer_finds_feature() {
    let mut idx = heap_index();
    assert_eq!(idx.find_features(Some("rome"), None, None), vec![(0, 2)]);
    idx.delete_feature_meta(0, 2);
    assert!(idx.find_features(Some("rome"), None, None).is_empty());
}

#[test]
fn set_gate_vector_heap_updates_row_and_rejects_bad_shapes() {
    let mut idx = heap_index();
    let new_row = Array1::from_vec(vec![9.0, 8.0, 7.0, 6.0]);
    idx.set_gate_vector(0, 1, &new_row);
    assert_eq!(idx.gate_vector(0, 1).unwrap(), vec![9.0, 8.0, 7.0, 6.0]);
    // Wrong length → no-op.
    let short = Array1::from_vec(vec![1.0]);
    idx.set_gate_vector(0, 0, &short);
    assert_eq!(
        idx.gate_vector(0, 0).unwrap(),
        (0..HIDDEN).map(|d| gate_val(0, 0, d)).collect::<Vec<_>>()
    );
    // Out-of-range feature / layer → no-op, no panic.
    idx.set_gate_vector(0, 99, &new_row);
    idx.set_gate_vector(99, 0, &new_row);
}

#[test]
fn down_and_up_overrides_round_trip() {
    let mut idx = heap_index();
    assert!(idx.down_override_at(0, 0).is_none());
    assert!(idx.up_override_at(0, 0).is_none());

    idx.set_down_vector(0, 0, vec![1.0, 2.0, 3.0, 4.0]);
    idx.set_up_vector(1, 2, vec![5.0; HIDDEN]);

    assert_eq!(idx.down_override_at(0, 0), Some(&[1.0, 2.0, 3.0, 4.0][..]));
    assert_eq!(idx.up_override_at(1, 2), Some(&[5.0; HIDDEN][..]));
    assert_eq!(idx.down_overrides().len(), 1);
    assert_eq!(idx.up_overrides().len(), 1);
    assert!(idx.down_override_at(1, 2).is_none());
    assert!(idx.up_override_at(0, 0).is_none());
}

// ── find_free_feature / find_features (heap) ──

#[test]
fn find_free_feature_heap_prefers_empty_slot_then_weakest() {
    let idx = heap_index();
    // Layer 0 has an explicit empty slot at feature 1.
    assert_eq!(idx.find_free_feature(0), Some(1));
    // Layer 1 is full → weakest c_score wins (lima, 1.0, feature 1).
    assert_eq!(idx.find_free_feature(1), Some(1));
    // Layer with no metadata at all → None.
    assert_eq!(idx.find_free_feature(99), None);
}

#[test]
fn find_features_filters_by_entity_layer_and_relation() {
    let idx = heap_index();
    // Entity match on top_token, case-insensitive.
    assert_eq!(idx.find_features(Some("PARIS"), None, None), vec![(0, 0)]);
    // Entity match via top_k alternates ("rome_alt").
    assert_eq!(
        idx.find_features(Some("rome_alt"), None, None),
        vec![(0, 2)]
    );
    // Layer filter excludes other layers.
    assert!(idx.find_features(Some("paris"), None, Some(1)).is_empty());
    assert_eq!(
        idx.find_features(Some("tokyo"), None, Some(1)),
        vec![(1, 0)]
    );
    // Relation matching is not implemented → any relation filter
    // rejects everything.
    assert!(idx
        .find_features(Some("paris"), Some("capital_of"), None)
        .is_empty());
    // No filters at all → every feature with metadata.
    assert_eq!(idx.find_features(None, None, None).len(), 5);
}

// ── save → load round trip (mmap mode) ──

#[test]
fn save_vindex_then_load_round_trips_gates_and_meta() {
    let tmp = tempfile::tempdir().unwrap();
    let (loaded, config) = saved_and_reloaded(tmp.path());

    for name in [GATE_VECTORS_BIN, DOWN_META_BIN, INDEX_JSON] {
        assert!(tmp.path().join(name).exists(), "{name} missing after save");
    }
    // Config layer info was refreshed by save.
    assert_eq!(config.layers.len(), NUM_LAYERS);
    for (i, info) in config.layers.iter().enumerate() {
        assert_eq!(info.layer, i);
        assert_eq!(info.num_features, NUM_FEATURES);
        assert_eq!(info.length, (NUM_FEATURES * HIDDEN * 4) as u64);
    }
    assert_eq!(config.layers[1].offset, config.layers[0].length);

    assert!(loaded.is_mmap());
    assert_eq!(loaded.num_features(0), NUM_FEATURES);
    // Gate values byte-identical.
    for layer in 0..NUM_LAYERS {
        for feature in 0..NUM_FEATURES {
            let expected: Vec<f32> = (0..HIDDEN).map(|d| gate_val(layer, feature, d)).collect();
            assert_eq!(loaded.gate_vector(layer, feature).unwrap(), expected);
        }
    }
    // Metadata round trip, including the None gap and top_k alternates.
    assert_eq!(loaded.feature_meta(0, 0).unwrap().top_token, "paris");
    assert!(loaded.feature_meta(0, 1).is_none());
    let rome = loaded.feature_meta(0, 2).unwrap();
    assert_eq!(rome.top_token, "rome");
    assert_eq!(rome.top_k[0].token, "rome_alt");
    assert_eq!(loaded.feature_meta(1, 1).unwrap().c_score, 1.0);
}

#[test]
fn insert_then_query_on_loaded_index() {
    let tmp = tempfile::tempdir().unwrap();
    let (mut loaded, _config) = saved_and_reloaded(tmp.path());

    // INSERT: claim the free slot on layer 0, install meta + gate row.
    let slot = loaded.find_free_feature(0).expect("free slot");
    assert_eq!(slot, 1, "mmap path should find the empty slot");
    loaded.set_feature_meta(0, slot, meta("madrid", 9.0));
    let row = Array1::from_vec(vec![4.0, 3.0, 2.0, 1.0]);
    loaded.set_gate_vector(0, slot, &row);

    // set_gate_vector on an mmap index promotes the layer to heap.
    assert!(loaded.gate.gate_vectors[0].is_some(), "layer 0 promoted");
    assert_eq!(
        loaded.gate_vector(0, slot).unwrap(),
        vec![4.0, 3.0, 2.0, 1.0]
    );
    // Untouched features on the promoted layer keep their mmap values.
    assert_eq!(
        loaded.gate_vector(0, 0).unwrap(),
        (0..HIDDEN).map(|d| gate_val(0, 0, d)).collect::<Vec<_>>()
    );
    // Query finds the inserted feature.
    assert_eq!(
        loaded.find_features(Some("madrid"), None, None),
        vec![(0, slot)]
    );
}

#[test]
fn find_free_feature_mmap_weakest_when_full() {
    let tmp = tempfile::tempdir().unwrap();
    let (loaded, _config) = saved_and_reloaded(tmp.path());
    // Layer 1 is full in the fixture → weakest c_score (feature 1).
    assert_eq!(loaded.find_free_feature(1), Some(1));
    // Out-of-range layer → 0 features → None.
    assert_eq!(loaded.find_free_feature(99), None);
}

#[test]
fn mutate_loaded_index_and_save_again_round_trips_the_edit() {
    let src = tempfile::tempdir().unwrap();
    let (mut loaded, mut config) = saved_and_reloaded(src.path());

    // Mutate layer 0 (promotes to heap); layer 1 stays mmap-backed, so
    // the second save covers both the heap and the mmap write branches.
    let row = Array1::from_vec(vec![-1.0, -2.0, -3.0, -4.0]);
    loaded.set_gate_vector(0, 2, &row);
    loaded.set_feature_meta(0, 2, meta("nairobi", 8.0));

    let dst = tempfile::tempdir().unwrap();
    loaded
        .save_vindex(dst.path(), &mut config)
        .expect("re-save");
    write_tokenizer(dst.path());

    let reloaded = VectorIndex::load_vindex(dst.path(), &mut SilentLoadCallbacks).unwrap();
    assert_eq!(
        reloaded.gate_vector(0, 2).unwrap(),
        vec![-1.0, -2.0, -3.0, -4.0]
    );
    assert_eq!(reloaded.feature_meta(0, 2).unwrap().top_token, "nairobi");
    // Layer 1 came through the mmap branch unchanged.
    assert_eq!(
        reloaded.gate_vector(1, 0).unwrap(),
        (0..HIDDEN).map(|d| gate_val(1, 0, d)).collect::<Vec<_>>()
    );
}

#[test]
fn save_gate_vectors_with_config_preserves_expert_geometry() {
    let tmp = tempfile::tempdir().unwrap();
    let idx = heap_index();
    let mut config = test_config();
    // Pretend layer 0 carries MoE geometry from a previous build.
    config.layers = vec![VindexLayerInfo {
        layer: 0,
        num_features: NUM_FEATURES,
        offset: 0,
        length: 0,
        num_experts: Some(4),
        num_features_per_expert: Some(2),
    }];
    let infos = idx
        .save_gate_vectors_with_config(tmp.path(), &config)
        .expect("save");
    assert_eq!(infos.len(), NUM_LAYERS);
    assert_eq!(infos[0].num_experts, Some(4));
    assert_eq!(infos[0].num_features_per_expert, Some(2));
    // Layer 1 had no prior geometry → stays dense.
    assert_eq!(infos[1].num_experts, None);
}

#[test]
fn save_config_writes_readable_index_json() {
    let tmp = tempfile::tempdir().unwrap();
    let config = test_config();
    VectorIndex::save_config(&config, tmp.path()).expect("save_config");
    let text = std::fs::read_to_string(tmp.path().join(INDEX_JSON)).unwrap();
    let parsed: VindexConfig = serde_json::from_str(&text).unwrap();
    assert_eq!(parsed.num_layers, NUM_LAYERS);
    assert_eq!(parsed.model, "test/mutate-model");
}

#[test]
fn save_gate_vectors_errors_on_missing_directory() {
    let idx = heap_index();
    let missing = Path::new("/nonexistent-larql-test-dir/deeper");
    assert!(idx.save_gate_vectors(missing).is_err());
}
