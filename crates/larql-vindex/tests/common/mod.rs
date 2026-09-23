//! Shared fixture builders for the v1 conformance suite
//! (`conformance_v1_*.rs`). One canonical minimal loadable vindex
//! directory — gate_vectors.bin + down_meta.bin + tokenizer.json +
//! index.json — that corruption tests then damage one artifact at a
//! time.
//!
//! Kept deliberately tiny: the corruption tests care about loader
//! behaviour, not model realism.
//!
//! Each `conformance_v1_*` target compiles this module separately and
//! uses a different subset of it, so per-target dead-code warnings are
//! expected noise.
#![allow(dead_code)]

use std::path::Path;

use larql_vindex::{FeatureMeta, SilentLoadCallbacks, VectorIndex, VindexConfig, VindexError};
use ndarray::Array2;

/// Layers in the synthetic fixture.
pub const FIXTURE_LAYERS: usize = 2;
/// Features per layer in the synthetic fixture.
pub const FIXTURE_FEATURES: usize = 4;
/// Hidden dimension of the synthetic fixture.
pub const FIXTURE_HIDDEN: usize = 8;
/// down_top_k written into the fixture config.
pub const FIXTURE_DOWN_TOP_K: usize = 1;

/// Minimal valid tokenizer.json (empty BPE vocab) — enough for
/// `load_vindex_tokenizer` to succeed.
pub const MINIMAL_TOKENIZER_JSON: &str =
    r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;

/// A deterministic in-memory index with metadata on every feature.
pub fn build_synthetic_index() -> VectorIndex {
    let mut state = 7u64;
    let mut gate_vectors = Vec::with_capacity(FIXTURE_LAYERS);
    let mut down_meta = Vec::with_capacity(FIXTURE_LAYERS);
    for _ in 0..FIXTURE_LAYERS {
        let gate = Array2::from_shape_fn((FIXTURE_FEATURES, FIXTURE_HIDDEN), |_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        });
        gate_vectors.push(Some(gate));
        let metas: Vec<Option<FeatureMeta>> = (0..FIXTURE_FEATURES)
            .map(|i| {
                Some(FeatureMeta {
                    top_token: format!("tok{i}"),
                    top_token_id: i as u32,
                    c_score: 0.5,
                    top_k: vec![],
                })
            })
            .collect();
        down_meta.push(Some(metas));
    }
    VectorIndex::new(gate_vectors, down_meta, FIXTURE_LAYERS, FIXTURE_HIDDEN)
}

/// Write the canonical minimal loadable vindex into `dir` and return
/// the config that was saved. Artifact set: `gate_vectors.bin`,
/// `down_meta.bin`, `tokenizer.json`, `index.json`.
pub fn save_minimal_vindex(dir: &Path) -> VindexConfig {
    std::fs::create_dir_all(dir).unwrap();
    let index = build_synthetic_index();
    let layer_infos = index.save_gate_vectors(dir).unwrap();
    index.save_down_meta(dir).unwrap();
    std::fs::write(dir.join("tokenizer.json"), MINIMAL_TOKENIZER_JSON).unwrap();
    let config = VindexConfig {
        version: 2,
        model: "conformance-v1".into(),
        family: "synthetic".into(),
        num_layers: FIXTURE_LAYERS,
        hidden_size: FIXTURE_HIDDEN,
        intermediate_size: FIXTURE_FEATURES,
        vocab_size: 16,
        embed_scale: 1.0,
        layers: layer_infos,
        down_top_k: FIXTURE_DOWN_TOP_K,
        ..Default::default()
    };
    VectorIndex::save_config(&config, dir).unwrap();
    config
}

/// Load a vindex directory through the production entry point.
pub fn load_dir(dir: &Path) -> Result<VectorIndex, VindexError> {
    let mut cb = SilentLoadCallbacks;
    VectorIndex::load_vindex(dir, &mut cb)
}

/// The fixture's tokenizer, for the readers that take one directly.
pub fn tokenizer() -> tokenizers::Tokenizer {
    tokenizers::Tokenizer::from_bytes(MINIMAL_TOKENIZER_JSON.as_bytes()).unwrap()
}
