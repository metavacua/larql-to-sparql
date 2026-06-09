//! Synthetic test fixtures for engine and layer-graph unit tests.
//!
//! The generic `make_test_weights()` ModelWeights builder lives in
//! [`larql_models::test_fixtures`] (gated behind the `test-utils`
//! feature) so downstream test crates — `larql-compute`, this crate,
//! and others — can construct realistic fixtures without disk I/O.
//! Inference re-exports it here so existing
//! `crate::test_utils::make_test_weights` callers don't change.
//!
//! Inference-specific helpers stay here:
//! - `make_test_vindex(weights)` — in-memory VectorIndex with random gate vectors
//! - `make_test_tokenizer(vocab_size)` — WordLevel tokenizer mapping token N to "[N]"
//! - `make_test_q4k_*`, `make_gemma3_*`, `make_starcoder2_*` etc. — arch-specific
//!   fixtures that pull in vindex/tokenizer machinery.
//!
//! Dimensions for `make_test_weights`: vocab=32, hidden=16,
//! intermediate=32, 2 q-heads, 1 kv-head, head_dim=8, 2 layers.
//! Forward pass ≈ 10 ms on CPU.

pub use larql_models::test_fixtures::make_test_weights;

use larql_models::{detect_from_json, ModelWeights, WeightArray};
use ndarray::Array2;
use std::collections::HashMap;

<<<<<<< HEAD:crates/larql-inference/src/test_utils.rs
=======
/// Build a synthetic `ModelWeights` with all tensors populated.
/// Uses `TinyModelArch` key conventions (e.g. `"0.attn.q_proj.weight"`).
pub fn make_test_weights() -> ModelWeights {
    make_test_weights_with(SyntheticDims::tiny())
}

/// Caller-supplied dimensions for [`make_test_weights_with`].
/// Useful when you need the same synthetic-weights factory at a
/// larger shape — e.g. for benchmarking the attention-service
/// runner under realistic dims, without paying for a real
/// model checkpoint.
#[derive(Clone, Copy, Debug)]
pub struct SyntheticDims {
    pub vocab: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_layers: usize,
}

impl SyntheticDims {
    /// Tiny: vocab=32, hidden=16, 2 layers. Default for unit tests.
    /// Forward pass < 1 ms on CPU.
    pub fn tiny() -> Self {
        Self {
            vocab: 32,
            hidden: 16,
            intermediate: 32,
            num_q_heads: 2,
            num_kv_heads: 1,
            head_dim: 8,
            num_layers: 2,
        }
    }

    /// Gemma-3-4B-shaped: vocab=262144, hidden=2560, 33 layers,
    /// num_q=8, num_kv=4, head_dim=320, intermediate=10240.
    /// Useful for realistic-shape benches; weight build is
    /// noticeable (~1 GB f32) so prefer caching with `OnceLock`.
    pub fn gemma_3_4b_like() -> Self {
        Self {
            vocab: 32,
            hidden: 2560,
            intermediate: 10240,
            num_q_heads: 8,
            num_kv_heads: 4,
            head_dim: 320,
            num_layers: 33,
        }
    }

    /// Smaller Gemma-shaped fixture for benches that don't need the
    /// full 33 layers — same hidden/heads but only 4 layers.
    /// Build cost ≈ 100 MB f32; useful for parametric sweeps.
    pub fn gemma_3_4b_4layer() -> Self {
        Self {
            vocab: 32,
            hidden: 2560,
            intermediate: 10240,
            num_q_heads: 8,
            num_kv_heads: 4,
            head_dim: 320,
            num_layers: 4,
        }
    }
}

/// Build synthetic weights at caller-supplied dimensions. The body
/// is identical to [`make_test_weights`] but parameterised. Used by
/// the `larql-server` benchmark harness to reach realistic Gemma
/// shapes without a real checkpoint.
pub fn make_test_weights_with(dims: SyntheticDims) -> ModelWeights {
    let arch_json = serde_json::json!({
        "model_type": "tinymodel",
        "hidden_size": dims.hidden,
        "num_hidden_layers": dims.num_layers,
        "intermediate_size": dims.intermediate,
        "head_dim": dims.head_dim,
        "num_attention_heads": dims.num_q_heads,
        "num_key_value_heads": dims.num_kv_heads,
        "vocab_size": dims.vocab,
    });
    let arch = detect_from_json(&arch_json);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut rng_state = 0xdeadbeef_u64;

    // LCG giving values in [-scale, +scale]
    let mut rand_mat = |rows: usize, cols: usize, scale: f32| -> WeightArray {
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (rng_state as u32) as f32 / u32::MAX as f32 * 2.0 * scale - scale
            })
            .collect();
        Array2::from_shape_vec((rows, cols), data)
            .unwrap()
            .into_shared()
    };

    let hidden = dims.hidden;
    // Embed + lm_head
    let embed = rand_mat(dims.vocab, hidden, 0.1);
    let lm_head = rand_mat(dims.vocab, hidden, 0.1);
    tensors.insert(arch.embed_key().to_string(), embed.clone());

    // Final norm (ones → valid unweighted RMSNorm fallback)
    vectors.insert(arch.final_norm_key().to_string(), vec![1.0; hidden]);

    let q_dim = dims.num_q_heads * dims.head_dim;
    let kv_dim = dims.num_kv_heads * dims.head_dim;
    let inter = dims.intermediate;

    for layer in 0..dims.num_layers {
        // Attention projections
        tensors.insert(arch.attn_q_key(layer), rand_mat(q_dim, hidden, 0.1));
        tensors.insert(arch.attn_k_key(layer), rand_mat(kv_dim, hidden, 0.1));
        tensors.insert(arch.attn_v_key(layer), rand_mat(kv_dim, hidden, 0.1));
        tensors.insert(arch.attn_o_key(layer), rand_mat(hidden, q_dim, 0.1));
        // FFN — missing tensors cause panic, so always provide them
        tensors.insert(arch.ffn_gate_key(layer), rand_mat(inter, hidden, 0.1));
        tensors.insert(arch.ffn_up_key(layer), rand_mat(inter, hidden, 0.1));
        tensors.insert(arch.ffn_down_key(layer), rand_mat(hidden, inter, 0.1));
        // Layer norms
        vectors.insert(arch.input_layernorm_key(layer), vec![1.0; hidden]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![1.0; hidden]);
    }

    ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        packed_mmaps: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: HashMap::new(),
        embed: larql_models::embed::EmbedMatrix::from_array(embed),
        embed_quant: None,
        lm_head,
        lm_head_quant: None,
        position_embed: None,
        quant_tensors: HashMap::new(),
        arch,
        num_layers: dims.num_layers,
        hidden_size: hidden,
        intermediate_size: inter,
        vocab_size: dims.vocab,
        head_dim: dims.head_dim,
        num_q_heads: dims.num_q_heads,
        num_kv_heads: dims.num_kv_heads,
        rope_base: 10_000.0,
    }
}

>>>>>>> ianblenke/main:crates/larql-inference/src/engines/test_utils.rs
/// Build an in-memory `VectorIndex` with random gate vectors per layer.
/// The VectorIndex has no Q4K or interleaved data — `predict_honest` falls
/// through to the CPU path, and `WalkFfn` routes through the sparse fallback
/// that uses `weights.tensors`.
pub fn make_test_vindex(weights: &ModelWeights) -> larql_vindex::VectorIndex {
    let n_features = weights.intermediate_size;
    let hidden = weights.hidden_size;

    // Each layer gets an independent LCG seed so gate matrices are distinct.
    let gate_vectors: Vec<Option<Array2<f32>>> = (0..weights.num_layers)
        .map(|l| {
            let mut state = 0xabcdef_u64.wrapping_add(l as u64 * 0x9e3779b97f4a7c15);
            let data: Vec<f32> = (0..n_features * hidden)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (state as u32) as f32 / u32::MAX as f32 * 0.1 - 0.05
                })
                .collect();
            Some(Array2::from_shape_vec((n_features, hidden), data).unwrap())
        })
        .collect();

    let down_meta = vec![None; weights.num_layers];
    larql_vindex::VectorIndex::new(gate_vectors, down_meta, weights.num_layers, hidden)
}

/// Extend an existing `VectorIndex` with an `interleaved.bin`-shaped
/// f32 FFN payload.
///
/// Layout per layer: `[gate(I × H) | up(I × H) | down(H × I)]` packed
/// as little-endian f32. Same format the `build_interleaved` example
/// produces, so the `interleaved_gate` / `interleaved_up` /
/// `interleaved_down` / `up_layer_matrix` / `down_layer_matrix`
/// accessors all observe the data.
///
/// Reuses `weights.tensors` for the matrices so the f32 walk paths
/// agree bit-for-bit with the dense forward pass under the same
/// weights.
pub fn attach_interleaved_f32_to_test_vindex(
    weights: &ModelWeights,
    index: &mut larql_vindex::VectorIndex,
) {
    let arch = &*weights.arch;
    let mut payload: Vec<u8> = Vec::new();
    for layer in 0..weights.num_layers {
        for key in [
            arch.ffn_gate_key(layer),
            arch.ffn_up_key(layer),
            arch.ffn_down_key(layer),
        ] {
            let tensor = weights
                .tensors
                .get(&key)
                .unwrap_or_else(|| panic!("missing tensor {key} in test weights"));
            let slice = tensor.as_slice().expect("contiguous row-major");
            payload.extend(slice.iter().flat_map(|v| v.to_le_bytes()));
        }
    }
    let mmap = arc_mmap_from_bytes(&payload);
    let storage = std::sync::Arc::make_mut(&mut index.storage);
    storage.set_interleaved_f32(mmap);
}

/// Extend an existing `VectorIndex` with feature-major f32 up/down
/// projections (the `up_features.bin` + `down_features.bin` layout).
///
/// `up_layer_matrix` and `down_layer_matrix` read from this storage,
/// distinct from the `interleaved.bin` layout used by `interleaved_up`
/// / `interleaved_down`. Tests that exercise the `walk_ffn_sparse`
/// fast path (which dispatches via `up_layer_matrix` /
/// `down_layer_matrix` when both return Some) need this fixture.
pub fn attach_feature_major_f32_to_test_vindex(
    weights: &ModelWeights,
    index: &mut larql_vindex::VectorIndex,
) {
    let arch = &*weights.arch;
    let mut up_payload: Vec<u8> = Vec::new();
    let mut down_payload: Vec<u8> = Vec::new();
    for layer in 0..weights.num_layers {
        // up_features layout: per-layer [intermediate × hidden] f32.
        let up = weights
            .tensors
            .get(&arch.ffn_up_key(layer))
            .unwrap_or_else(|| panic!("missing ffn_up tensor"));
        let up_slice = up.as_slice().expect("contiguous row-major");
        up_payload.extend(up_slice.iter().flat_map(|v| v.to_le_bytes()));
        // down_features layout: per-layer [intermediate × hidden] f32 —
        // note the transpose vs the in-memory `[hidden × intermediate]`
        // shape. Walk through manually so the on-disk layout is
        // intermediate-major.
        let down = weights
            .tensors
            .get(&arch.ffn_down_key(layer))
            .unwrap_or_else(|| panic!("missing ffn_down tensor"));
        let h = weights.hidden_size;
        let i = weights.intermediate_size;
        // down: [hidden × intermediate] → write as [intermediate × hidden]
        // by transposing rows/cols at write time.
        for inter in 0..i {
            for hid in 0..h {
                let val = down[[hid, inter]];
                down_payload.extend_from_slice(&val.to_le_bytes());
            }
        }
    }
    let up_mmap = arc_mmap_from_bytes(&up_payload);
    let down_mmap = arc_mmap_from_bytes(&down_payload);
    let storage = std::sync::Arc::make_mut(&mut index.storage);
    storage.set_up_features(up_mmap);
    storage.set_down_features(down_mmap);
}

/// Bundled f32-interleaved fixture: same as [`TestFixtures`] but with
/// the test vindex extended via [`attach_interleaved_f32_to_test_vindex`].
/// Use for tests that need `up_layer_matrix` / `down_layer_matrix` /
/// `interleaved_*` accessors to return `Some` (e.g.
/// `GuidedWalkLayerGraph`, the priority-6 routing branch in `WalkFfn`).
pub struct InterleavedF32TestFixtures {
    pub weights: ModelWeights,
    pub tokenizer: tokenizers::Tokenizer,
    pub index: larql_vindex::VectorIndex,
}

impl InterleavedF32TestFixtures {
    pub fn build() -> Self {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let mut index = make_test_vindex(&weights);
        attach_interleaved_f32_to_test_vindex(&weights, &mut index);
        Self {
            weights,
            tokenizer,
            index,
        }
    }
}

/// Build a `tokenizers::Tokenizer` with a vocabulary of `vocab_size` tokens.
/// Token N decodes to `"[N]"`, so token IDs from `make_test_weights()` all
/// decode to valid (if meaningless) strings.
pub fn make_test_tokenizer(vocab_size: usize) -> tokenizers::Tokenizer {
    // WordLevel::builder().vocab() requires an AHashMap.
    // Build a simple BPE-less tokenizer via JSON serialization instead.
    let mut vocab_json = serde_json::Map::new();
    for i in 0..vocab_size as u64 {
        vocab_json.insert(format!("[{i}]"), serde_json::Value::Number(i.into()));
    }
    // Add UNK token at the end
    vocab_json.insert("[UNK]".into(), serde_json::Value::Number(vocab_size.into()));

    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocab_json,
            "unk_token": "[UNK]"
        }
    });

    let bytes = serde_json::to_vec(&tokenizer_json).expect("JSON serialization failed");
    tokenizers::Tokenizer::from_bytes(&bytes).expect("synthetic tokenizer construction failed")
}

/// All three synthetic fixtures bundled together. Build once per test module
/// via `OnceLock`; each field is cheaply borrowed.
pub struct TestFixtures {
    pub weights: ModelWeights,
    pub tokenizer: tokenizers::Tokenizer,
    pub index: larql_vindex::VectorIndex,
}

impl TestFixtures {
    pub fn build() -> Self {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let index = make_test_vindex(&weights);
        Self {
            weights,
            tokenizer,
            index,
        }
    }
}

/// Serialise the synthetic `make_test_weights()` model + matching
/// vindex + tokenizer to an on-disk directory that any code path
/// reaching for `larql_vindex::load_vindex_config` /
/// `load_model_weights` will accept.
///
/// Replaces the previous "set `LARQL_MODEL` to a real Gemma snapshot"
/// pattern: tests can call this with a `tempfile::TempDir` and exercise
/// the full disk-loading pipeline without depending on multi-gigabyte
/// model artifacts in `~/.cache`.
///
/// The fixture is **synthetic**: the weights produce garbage logits.
/// Tests asserting plumbing (correct files written, correct error on
/// missing config, correct dispatch on backend type, etc.) work fine;
/// tests asserting semantic content ("model predicts Paris") still
/// need a real model and don't belong in `tests/`.
///
/// Layout written:
/// ```text
/// dir/
///   index.json              -- VindexConfig with has_model_weights=true
///   tokenizer.json          -- WordLevel "[0]".."[VOCAB-1]" tokenizer
///   embeddings.bin          -- VOCAB × HIDDEN f32 (from weights.embed)
///   weight_manifest.json    -- per-tensor offset/length manifest
///   attn_weights.bin        -- per-layer Q/K/V/O + norms
///   up_weights.bin          -- per-layer gate + up
///   down_weights.bin        -- per-layer down
///   norms.bin               -- final norm
///   lm_head.bin             -- output projection
///   gate_vectors.bin        -- vindex gate matrices (from make_test_vindex)
///   down_meta.bin           -- vindex down metadata (empty per layer)
/// ```
pub fn write_synthetic_model_dir(dir: &std::path::Path) -> Result<(), String> {
    use larql_vindex::{
        write_model_weights, ExtractLevel, MoeConfig, StorageDtype, VindexConfig, VindexModelConfig,
    };

    std::fs::create_dir_all(dir).map_err(|e| format!("create_dir_all: {e}"))?;

    let weights = make_test_weights();
    let tokenizer = make_test_tokenizer(weights.vocab_size);
    let index = make_test_vindex(&weights);

    // ── tokenizer.json ────────────────────────────────────────────────
    // Write a tokenizer that encodes `[N]` to id N *as a single token*
    // — `make_test_tokenizer`'s Whitespace pre-tokenizer would split
    // `[1]` into `[`, `1`, `]`, all of which UNK, blowing up the
    // embedding lookup with id=vocab_size. The on-disk fixture uses a
    // pre-tokenizer-free variant so test prompts like `EXPLAIN INFER
    // "[1]"` lookup directly. `tokenizer` is kept above for any caller
    // that needs the in-memory shape.
    let _ = &tokenizer; // returned by make_test_tokenizer; not the on-disk shape
    let tok_path = dir.join("tokenizer.json");
    std::fs::write(&tok_path, synthetic_tokenizer_json(weights.vocab_size))
        .map_err(|e| format!("write tokenizer.json: {e}"))?;

    // ── model_config + index.json ─────────────────────────────────────
    // `has_model_weights=true` is the gate the loader checks; without
    // it `load_model_weights` errors with "rebuild with extract --level
    // all". model_config carries the arch fields detect_from_json needs
    // to reconstruct the tinymodel arch on the loader side.
    let model_config = VindexModelConfig {
        model_type: "tinymodel".into(),
        head_dim: weights.head_dim,
        num_q_heads: weights.num_q_heads,
        num_kv_heads: weights.num_kv_heads,
        rope_base: weights.rope_base,
        sliding_window: None,
        moe: None::<MoeConfig>,
        global_head_dim: None,
        num_global_kv_heads: None,
        partial_rotary_factor: None,
        sliding_window_pattern: None,
        layer_types: None,
        attention_k_eq_v: false,
        num_kv_shared_layers: None,
        per_layer_embed_dim: None,
        rope_local_base: None,
        query_pre_attn_scalar: None,
        final_logit_softcapping: None,
        attention_multiplier: None,
        residual_multiplier: None,
        logits_scaling: None,
        norm_eps: None,
    };

    let mut config = VindexConfig {
        version: 2,
        model: "synthetic/tinymodel".into(),
        family: "tinymodel".into(),
        source: None,
        checksums: None,
        num_layers: weights.num_layers,
        hidden_size: weights.hidden_size,
        intermediate_size: weights.intermediate_size,
        vocab_size: weights.vocab_size,
        embed_scale: 1.0,
        extract_level: ExtractLevel::All,
        dtype: StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: 5,
        has_model_weights: true,
        model_config: Some(model_config),
        fp4: None,
        ffn_layout: None,
    };

    // Writes index.json + gate_vectors.bin + down_meta.bin.
    // `save_vindex` mutates `config` to record layer manifests.
    index
        .save_vindex(dir, &mut config)
        .map_err(|e| format!("save_vindex: {e}"))?;

    // ── Model weights (attn / up / down / norms / lm_head) ────────────
    let mut cb = larql_vindex::SilentBuildCallbacks;
    write_model_weights(&weights, dir, &mut cb).map_err(|e| format!("write_model_weights: {e}"))?;

    // ── Embeddings (vocab × hidden f32, little-endian) ────────────────
    let embed_slice = weights.embed.as_slice().ok_or("embed not contiguous")?;
    let mut embed_bytes = Vec::with_capacity(embed_slice.len() * 4);
    for &v in embed_slice {
        embed_bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(dir.join("embeddings.bin"), &embed_bytes)
        .map_err(|e| format!("write embeddings.bin: {e}"))?;

    Ok(())
}

/// Serialise the synthetic `make_test_q4k_weights()` model + matching
/// Q4_K vindex to an on-disk directory that the strict
/// `open_inference_vindex` loader will accept.
///
/// Companion to [`write_synthetic_model_dir`]. Use this when a test
/// needs to exercise the Q4_K loader resolution order (attn_weights_q4k
/// → interleaved_kquant → lm_head_q4) without a real Gemma snapshot on
/// disk.
///
/// Layout written:
/// ```text
/// dir/
///   index.json                       -- VindexConfig with quant=Q4K
///   tokenizer.json                   -- WordLevel "[0]".."[VOCAB-1]"
///   gate_vectors.bin                 -- empty per-layer (vindex contract)
///   down_meta.bin                    -- empty per-layer
///   attn_weights_q4k.bin             -- Q/K/V/O quantised per layer
///   attn_weights_q4k_manifest.json
///   interleaved_kquant.bin              -- [gate|up|down] per layer
///   interleaved_kquant_manifest.json
///   lm_head_q4.bin                   -- tied embed quantised
///   norms.bin                        -- f32 norms (unchanged from non-Q4 path)
/// ```
pub fn write_synthetic_q4k_model_dir(dir: &std::path::Path) -> Result<(), String> {
    write_synthetic_q4k_model_dir_layers(dir, Q4K_TEST_NUM_LAYERS)
}

/// Layer-parametrised sibling of [`write_synthetic_q4k_model_dir`]. Same
/// on-disk layout, but the serialised model has `num_layers` decoder
/// layers — for tests that need a depth-fraction layer index to land
/// inside the model (e.g. the LQL FR3 relation resolver's probe layer,
/// which clamps to ≥3 and so needs a model deeper than the 2-layer default).
pub fn write_synthetic_q4k_model_dir_layers(
    dir: &std::path::Path,
    num_layers: usize,
) -> Result<(), String> {
    use larql_vindex::{
        write_model_weights_kquant, ExtractLevel, MoeConfig, SilentBuildCallbacks, StorageDtype,
        VindexConfig, VindexModelConfig,
    };

    std::fs::create_dir_all(dir).map_err(|e| format!("create_dir_all: {e}"))?;

    let weights = make_test_q4k_weights_layers(num_layers);

    // ── tokenizer.json ────────────────────────────────────────────────
    std::fs::write(
        dir.join("tokenizer.json"),
        synthetic_tokenizer_json(weights.vocab_size),
    )
    .map_err(|e| format!("write tokenizer.json: {e}"))?;

    // ── model_config + index.json ─────────────────────────────────────
    let model_config = VindexModelConfig {
        model_type: "gemma3_text".into(),
        head_dim: weights.head_dim,
        num_q_heads: weights.num_q_heads,
        num_kv_heads: weights.num_kv_heads,
        rope_base: weights.rope_base,
        sliding_window: None,
        moe: None::<MoeConfig>,
        global_head_dim: None,
        num_global_kv_heads: None,
        partial_rotary_factor: None,
        sliding_window_pattern: None,
        layer_types: None,
        attention_k_eq_v: false,
        num_kv_shared_layers: None,
        per_layer_embed_dim: None,
        rope_local_base: None,
        query_pre_attn_scalar: None,
        final_logit_softcapping: None,
        attention_multiplier: None,
        residual_multiplier: None,
        logits_scaling: None,
        norm_eps: None,
    };

    let mut config = VindexConfig {
        version: 2,
        model: "synthetic/gemma3_q4k".into(),
        family: "gemma3".into(),
        source: None,
        checksums: None,
        num_layers: weights.num_layers,
        hidden_size: weights.hidden_size,
        intermediate_size: weights.intermediate_size,
        vocab_size: weights.vocab_size,
        embed_scale: 1.0,
        extract_level: ExtractLevel::All,
        dtype: StorageDtype::F32,
        quant: larql_vindex::QuantFormat::Q4K,
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: 5,
        has_model_weights: true,
        model_config: Some(model_config),
        fp4: None,
        ffn_layout: None,
    };

    // Use an empty in-memory index for `save_vindex` (writes the
    // mandatory gate_vectors.bin + down_meta.bin + index.json scaffolding).
    let empty_index = larql_vindex::VectorIndex::new(
        vec![None; weights.num_layers],
        vec![None; weights.num_layers],
        weights.num_layers,
        weights.hidden_size,
    );
    empty_index
        .save_vindex(dir, &mut config)
        .map_err(|e| format!("save_vindex: {e}"))?;

    // ── Q4K weights (attn_weights_q4k + interleaved_kquant + lm_head_q4 + norms) ──
    let mut cb = SilentBuildCallbacks;
    write_model_weights_kquant(&weights, dir, &mut cb)
        .map_err(|e| format!("write_model_weights_kquant: {e}"))?;

    // ── Embeddings (required by `load_model_weights_kquant` — the Q4K
    //    writer doesn't emit them on its own). ─────────────────────
    let embed_slice = weights.embed.as_slice().ok_or("embed not contiguous")?;
    let mut embed_bytes = Vec::with_capacity(embed_slice.len() * 4);
    for &v in embed_slice {
        embed_bytes.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(dir.join("embeddings.bin"), &embed_bytes)
        .map_err(|e| format!("write embeddings.bin: {e}"))?;

    Ok(())
}

/// Build a tokenizer JSON whose vocab is `[0]`..`[vocab_size-1]` and
/// whose `pre_tokenizer` is **null** — so bracketed forms encode as a
/// single token instead of being split into `[`, `N`, `]` (all UNK)
/// by [`make_test_tokenizer`]'s Whitespace pre-tokenizer.
///
/// Used only by [`write_synthetic_model_dir`] so on-disk-fixture
/// callers can write test prompts like `"[1]"` and have them
/// encode to a single in-vocab id. `make_test_tokenizer` is kept
/// in its prior shape for backward-compatibility with in-memory
/// fixture consumers.
///
/// `[UNK]` is mapped to **id 0** (a real, in-range vocab slot) so any
/// stray UNK from text the loader processes through the model still
/// hits a valid embedding row — saves the embed lookup from panicking
/// with "Index N must be less than axis length N" when something
/// outside the bracket form sneaks into encoding.
/// Build the on-disk tokenizer JSON whose vocab is `[0]`..`[vocab_size-1]`
/// and whose `pre_tokenizer` is **null** — bracketed forms encode as a
/// single token. Public so tests can build a matching in-memory
/// `Tokenizer` without going through `write_synthetic_model_dir`.
pub fn synthetic_tokenizer_json(vocab_size: usize) -> String {
    let mut vocab_json = serde_json::Map::new();
    for i in 0..vocab_size as u64 {
        vocab_json.insert(format!("[{i}]"), serde_json::Value::Number(i.into()));
    }
    vocab_json.insert("[UNK]".into(), serde_json::Value::Number(0.into()));

    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocab_json,
            "unk_token": "[UNK]"
        }
    });
    serde_json::to_string(&tokenizer_json).expect("synthetic tokenizer json")
}

// ── Alternate-arch fixtures ─────────────────────────────────────────────
//
// `make_test_weights` uses the `tinymodel` arch which leaves many optional
// branches dormant (no bias keys, no QK norm, no post norms, gated FFN
// only). The fixtures below pin those branches by routing through a
// real arch impl that enables them. Each fixture provides exactly the
// tensors + vectors the matching forward path needs to reach finite
// output without panicking.

fn rand_mat_seeded(rows: usize, cols: usize, scale: f32, seed: u64) -> WeightArray {
    let mut state = seed;
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state as u32) as f32 / u32::MAX as f32 * 2.0 * scale - scale
        })
        .collect();
    Array2::from_shape_vec((rows, cols), data)
        .unwrap()
        .into_shared()
}

// `make_gemma3_test_weights` and `make_starcoder2_test_weights` moved
// to `larql_models::test_fixtures` (ADR-0022 Step 2e) so larql-compute
// can run arch-specific spine tests too. Re-exported here so existing
// `crate::test_utils::make_*_test_weights` paths in inference test
// modules and downstream test crates (larql-kv) keep working.
pub use larql_models::test_fixtures::{make_gemma3_test_weights, make_starcoder2_test_weights};

// ── Q4_K-aware synthetic fixtures moved to `larql_models::test_fixtures` ──
// (ADR-0022 Step 3b) so larql-compute's kquant_forward tests can
// construct realistic Q4K-sized ModelWeights. Re-exported for existing
// `crate::test_utils::*` callers.
pub use larql_models::test_fixtures::{
    arc_mmap_from_bytes, make_test_q4k_weights, make_test_q4k_weights_layers,
    make_test_q4k_weights_silu, Q4K_TEST_HIDDEN, Q4K_TEST_INTER, Q4K_TEST_NUM_LAYERS,
    Q4K_TEST_VOCAB,
};
/// Build a fully-populated synthetic `VectorIndex` that satisfies the
/// cached + direct-matvec decode contract on the Q4_K weights from
/// [`make_test_q4k_weights`]. Quantises Q/K/V/O and gate/up/down to
/// Q4_K bytes via `quantize_q4_k`, installs them as the attn +
/// interleaved Q4_K storage, and synthesises a Q4_K lm_head view from
/// the (tied) embeddings.
pub fn make_test_q4k_vindex(weights: &ModelWeights) -> larql_vindex::VectorIndex {
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;

    let num_layers = weights.num_layers;
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;

    let q4k_for = |key: &str| -> Vec<u8> {
        let tensor = weights
            .tensors
            .get(key)
            .unwrap_or_else(|| panic!("missing tensor {key} in test weights"));
        let slice = tensor.as_slice().expect("contiguous row-major");
        quantize_q4_k(slice)
    };

    let mut attn_payload: Vec<u8> = Vec::new();
    let mut attn_manifest: Vec<(usize, usize, String)> = Vec::new();
    for layer in 0..num_layers {
        for key in [
            arch.attn_q_key(layer),
            arch.attn_k_key(layer),
            arch.attn_v_key(layer),
            arch.attn_o_key(layer),
        ] {
            let bytes = q4k_for(&key);
            let offset = attn_payload.len();
            let length = bytes.len();
            attn_payload.extend_from_slice(&bytes);
            attn_manifest.push((offset, length, "Q4_K".to_string()));
        }
    }

    let mut ffn_payload: Vec<u8> = Vec::new();
    let mut ffn_manifest: Vec<(usize, usize, String)> = Vec::new();
    for layer in 0..num_layers {
        for key in [
            arch.ffn_gate_key(layer),
            arch.ffn_up_key(layer),
            arch.ffn_down_key(layer),
        ] {
            let bytes = q4k_for(&key);
            let offset = ffn_payload.len();
            let length = bytes.len();
            ffn_payload.extend_from_slice(&bytes);
            ffn_manifest.push((offset, length, "Q4_K".to_string()));
        }
    }

    let gate_vectors = vec![None; num_layers];
    let down_meta = vec![None; num_layers];
    let mut index = larql_vindex::VectorIndex::new(gate_vectors, down_meta, num_layers, hidden);
    index.vocab_size = weights.vocab_size;

    let attn_mmap = arc_mmap_from_bytes(&attn_payload);
    let ffn_mmap = arc_mmap_from_bytes(&ffn_payload);
    {
        let storage = std::sync::Arc::make_mut(&mut index.storage);
        storage.set_attn_kquant(attn_mmap, Some(attn_manifest));
        storage.set_interleaved_kquant(ffn_mmap, Some(ffn_manifest));
    }

    // Synth Q4_K lm_head from tied embedding (same lifecycle as
    // `synthesize_lm_head_kquant` on a real tied-embedding vindex).
    let lm_head_slice = weights
        .lm_head
        .as_slice()
        .expect("lm_head contiguous row-major");
    let lm_head_q4 = quantize_q4_k(lm_head_slice);
    let lm_head_mmap = arc_mmap_from_bytes(&lm_head_q4);
    {
        let storage = std::sync::Arc::make_mut(&mut index.storage);
        storage.set_lm_head_kquant_mmap(lm_head_mmap);
    }

    // Also populate the f32 lm_head view so callers reaching
    // `lm_head_knn_backend_skip_q4k` get a non-empty fallback when the
    // backend's Q4_K stride-32 / f16 GEMV paths aren't implemented
    // (e.g. `MockGpuBackend` delegating to `CpuBackend`'s default
    // `q4k_matvec_stride32 → None`). Without this, `forced_logits` and
    // anything else that routes through that helper short-circuits on
    // "vindex lm_head returned no scores".
    let lm_head_f32_bytes: Vec<u8> = lm_head_slice.iter().flat_map(|v| v.to_le_bytes()).collect();
    let lm_head_f32_mmap = arc_mmap_from_bytes(&lm_head_f32_bytes);
    {
        let storage = std::sync::Arc::make_mut(&mut index.storage);
        storage.set_lm_head_f32(lm_head_f32_mmap);
    }
    index
}

/// Like [`make_test_q4k_vindex`] but with no FFN mmap — simulates a
/// pure-MoE model where dense FFN weights don't exist.
pub fn make_test_q4k_vindex_attn_only(weights: &ModelWeights) -> larql_vindex::VectorIndex {
    use larql_compute::cpu::ops::q4_common::quantize_q4_k;

    let num_layers = weights.num_layers;
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;

    let q4k_for = |key: &str| -> Vec<u8> {
        let tensor = weights
            .tensors
            .get(key)
            .unwrap_or_else(|| panic!("missing tensor {key} in test weights"));
        let slice = tensor.as_slice().expect("contiguous row-major");
        quantize_q4_k(slice)
    };

    let mut attn_payload: Vec<u8> = Vec::new();
    let mut attn_manifest: Vec<(usize, usize, String)> = Vec::new();
    for layer in 0..num_layers {
        for key in [
            arch.attn_q_key(layer),
            arch.attn_k_key(layer),
            arch.attn_v_key(layer),
            arch.attn_o_key(layer),
        ] {
            let bytes = q4k_for(&key);
            let offset = attn_payload.len();
            let length = bytes.len();
            attn_payload.extend_from_slice(&bytes);
            attn_manifest.push((offset, length, "Q4_K".to_string()));
        }
    }

    let gate_vectors = vec![None; num_layers];
    let down_meta = vec![None; num_layers];
    let mut index = larql_vindex::VectorIndex::new(gate_vectors, down_meta, num_layers, hidden);
    index.vocab_size = weights.vocab_size;

    let attn_mmap = arc_mmap_from_bytes(&attn_payload);
    {
        let storage = std::sync::Arc::make_mut(&mut index.storage);
        storage.set_attn_kquant(attn_mmap, Some(attn_manifest));
    }

    index
}

/// Minimum Q4_K-aligned hidden / intermediate / expert-intermediate
/// for the Gemma 4 hybrid-MoE fixture. Q4_K requires multiples of 256.
pub const GEMMA4_MOE_HIDDEN: usize = 256;
pub const GEMMA4_MOE_INTER: usize = 256;
pub const GEMMA4_MOE_NUM_EXPERTS: usize = 4;
pub const GEMMA4_MOE_TOP_K: usize = 2;

/// Build a synthetic Gemma 4 hybrid-MoE `ModelWeights`.
///
<<<<<<< HEAD:crates/larql-inference/src/test_utils.rs
/// `enable_moe_block=true` plus all the per-layer dense attention + dense
/// FFN tensors a Gemma 4 26B-A4B variant carries, plus the per-layer MoE
/// pieces:
///
/// - Router projection (`vectors[layers.L.router.proj.weight]`).
/// - Packed BF16 expert `gate_up` (`raw_bytes[layers.L.experts.gate_up_proj]`).
/// - Packed BF16 expert `down`    (`raw_bytes[layers.L.experts.down_proj]`).
///
/// All weights are deterministic LCG ramps. Values are math-meaningless;
/// the fixture's job is to satisfy the runtime checks
/// (`arch.is_hybrid_moe()=true`, `weights.get_packed_bytes(...)` non-None,
/// `weights.vectors[router_key]` non-None) so the MoE forward branches
/// in `pipeline_layer::build_moe_weights`,
/// `vindex/kquant_forward/hidden.rs::run_moe_layer_cpu`, and
/// `vindex/kquant_forward/remote_ffn.rs` execute end-to-end.
pub fn make_test_gemma4_moe_weights() -> ModelWeights {
    let num_q = 4usize;
    let num_kv = 2usize;
    let head_dim = GEMMA4_MOE_HIDDEN / num_q;
    let num_layers = 2usize;
=======
/// Enables the dormant branches in `attention/{block, gpu}.rs` and
/// `forward/layer.rs` that tinymodel never reaches:
/// - **QK norm** — `attn_q_norm_key` / `attn_k_norm_key` return Some,
///   so `rms_norm_heads` runs in `run_attention_block_core`
/// - **post norms** — `has_post_norms()` is true; pre/post FFN norm keys
///   are populated, the FFN dispatch routes through the post-norm arm
/// - **GeluTanh activation** — `activation()` is `GeluTanh`, exercising
///   the gelu-tanh gate-up branches in `ffn/weight.rs` and `attention`
/// - **`embed_scale = sqrt(hidden)`** — non-1.0 embed scaling
/// - **`norm_weight_offset = 1.0`** — non-zero offset added to every
///   norm weight at runtime (the saved weights are deltas)
/// - **sliding-window/global pattern** — layer 0 is a sliding-window
///   layer, layer 5 (and multiples of 6) is global; both are exercised
///   in the 2-layer fixture only at layer 0/1 (both sliding), but
///   `is_sliding_window_layer()` still returns sensible values.
pub fn make_gemma3_test_weights() -> ModelWeights {
    const VOCAB: usize = 32;
    const HIDDEN: usize = 16;
    const INTER: usize = 32;
    const NUM_Q: usize = 2;
    const NUM_KV: usize = 1;
    const HEAD_DIM: usize = 8;
    const NUM_LAYERS: usize = 2;
>>>>>>> ianblenke/main:crates/larql-inference/src/engines/test_utils.rs

    let arch_json = serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": GEMMA4_MOE_HIDDEN,
            "intermediate_size": GEMMA4_MOE_INTER,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_q,
            "num_key_value_heads": num_kv,
            "head_dim": head_dim,
            "vocab_size": GEMMA4_MOE_HIDDEN,
            "enable_moe_block": true,
            "num_experts": GEMMA4_MOE_NUM_EXPERTS,
            "top_k_experts": GEMMA4_MOE_TOP_K,
            "moe_intermediate_size": GEMMA4_MOE_INTER,
            "rope_theta": 10000.0,
        }
    });
    let arch = detect_from_json(&arch_json);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut raw_bytes: HashMap<String, Vec<u8>> = HashMap::new();

    let mut seed = 0xb000_1eef_u64;
    let mut next_seed = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };

    let hidden = GEMMA4_MOE_HIDDEN;
    let inter = GEMMA4_MOE_INTER;
    let moe_inter = GEMMA4_MOE_INTER;
    let vocab = GEMMA4_MOE_HIDDEN;

    let embed = rand_mat_seeded(vocab, hidden, 0.05, next_seed());
    let lm_head = embed.clone();
    tensors.insert(arch.embed_key().to_string(), embed.clone());

    vectors.insert(arch.final_norm_key().to_string(), vec![1.0; hidden]);

    let q_dim = num_q * head_dim;
    let kv_dim = num_kv * head_dim;

    for layer in 0..num_layers {
        tensors.insert(
            arch.attn_q_key(layer),
            rand_mat_seeded(q_dim, hidden, 0.05, next_seed()),
        );
        tensors.insert(
            arch.attn_k_key(layer),
            rand_mat_seeded(kv_dim, hidden, 0.05, next_seed()),
        );
        tensors.insert(
            arch.attn_v_key(layer),
            rand_mat_seeded(kv_dim, hidden, 0.05, next_seed()),
        );
        tensors.insert(
            arch.attn_o_key(layer),
            rand_mat_seeded(hidden, q_dim, 0.05, next_seed()),
        );

<<<<<<< HEAD:crates/larql-inference/src/test_utils.rs
        // Hybrid: every layer also carries a dense MLP alongside MoE.
=======
        // FFN — gated (SwiGLU/GeluTanh)
>>>>>>> ianblenke/main:crates/larql-inference/src/engines/test_utils.rs
        tensors.insert(
            arch.ffn_gate_key(layer),
            rand_mat_seeded(inter, hidden, 0.05, next_seed()),
        );
        tensors.insert(
            arch.ffn_up_key(layer),
            rand_mat_seeded(inter, hidden, 0.05, next_seed()),
        );
        tensors.insert(
            arch.ffn_down_key(layer),
            rand_mat_seeded(hidden, inter, 0.05, next_seed()),
        );

        // Gemma 4 four-norm layout.
        vectors.insert(arch.input_layernorm_key(layer), vec![0.5; hidden]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![0.5; hidden]);
        if let Some(k) = arch.pre_feedforward_layernorm_key(layer) {
            vectors.insert(k, vec![0.5; hidden]);
        }
        if let Some(k) = arch.post_feedforward_layernorm_key(layer) {
            vectors.insert(k, vec![0.5; hidden]);
        }
        if let Some(k) = arch.attn_q_norm_key(layer) {
            vectors.insert(k, vec![0.5; head_dim]);
        }
        if let Some(k) = arch.attn_k_norm_key(layer) {
            vectors.insert(k, vec![0.5; head_dim]);
        }
        if let Some(k) = arch.layer_scalar_key(layer) {
            vectors.insert(k, vec![1.0]);
        }

        // ── MoE pieces ───────────────────────────────────────────────
        let router_key = arch
            .moe_router_key(layer)
            .expect("Gemma 4 MoE arch must produce a router key");
        let router_proj: Vec<f32> = (0..GEMMA4_MOE_NUM_EXPERTS * hidden)
            .map(|i| ((i as f32) * 0.001).sin() * 0.05)
            .collect();
        vectors.insert(router_key, router_proj);

        // Packed BF16 expert gate_up: num_experts × [2*moe_inter, hidden].
        // BF16 = top 16 bits of the f32 little-endian representation; the
        // per-byte ramp keeps every block non-degenerate without
        // saturating the activation.
        let gate_up_floats_per_expert = 2 * moe_inter * hidden;
        let total_gate_up_bytes = GEMMA4_MOE_NUM_EXPERTS * gate_up_floats_per_expert * 2;
        let mut gate_up_blob = vec![0u8; total_gate_up_bytes];
        for (i, chunk) in gate_up_blob.chunks_exact_mut(2).enumerate() {
            let v = (((i & 0xff) as f32 * 0.001 - 0.128) * 0.1).to_bits();
            chunk[0] = (v >> 16) as u8;
            chunk[1] = (v >> 24) as u8;
        }
        let gate_up_key = arch
            .packed_experts_gate_up_key(layer)
            .expect("Gemma 4 MoE arch must produce a packed gate_up key");
        raw_bytes.insert(gate_up_key, gate_up_blob);

        let down_floats_per_expert = hidden * moe_inter;
        let total_down_bytes = GEMMA4_MOE_NUM_EXPERTS * down_floats_per_expert * 2;
        let mut down_blob = vec![0u8; total_down_bytes];
        for (i, chunk) in down_blob.chunks_exact_mut(2).enumerate() {
            let v = (((i & 0xff) as f32 * 0.0007 - 0.09) * 0.1).to_bits();
            chunk[0] = (v >> 16) as u8;
            chunk[1] = (v >> 24) as u8;
        }
        let down_key = arch
            .packed_experts_down_key(layer)
            .expect("Gemma 4 MoE arch must produce a packed down key");
        raw_bytes.insert(down_key, down_blob);
    }

    ModelWeights {
        tensors,
        vectors,
        raw_bytes,
        packed_mmaps: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: HashMap::new(),
        embed: larql_models::embed::EmbedMatrix::from_array(embed),
        embed_quant: None,
        lm_head,
        lm_head_quant: None,
        position_embed: None,
        quant_tensors: HashMap::new(),
        arch,
        num_layers,
        hidden_size: hidden,
        intermediate_size: inter,
        vocab_size: vocab,
        head_dim,
        num_q_heads: num_q,
        num_kv_heads: num_kv,
        rope_base: 10_000.0,
    }
}

<<<<<<< HEAD:crates/larql-inference/src/test_utils.rs
// `synthetic_e2b_like_arch_json` + `make_synthetic_e2b_like_weights`
// moved to `larql_models::test_fixtures` (ADR-0022 Step 2e2). Re-exported
// so existing `crate::test_utils::*` callers (forward/ple.rs tests) and
// downstream test crates keep working.
pub use larql_models::test_fixtures::{
    make_synthetic_e2b_like_weights, synthetic_e2b_like_arch_json,
};
/// Bundled fixture for Q4_K decode-path tests. Mirrors `TestFixtures`.
pub struct Q4KTestFixtures {
    pub weights: ModelWeights,
    pub tokenizer: tokenizers::Tokenizer,
    pub index: larql_vindex::VectorIndex,
}
=======
/// Build a synthetic `ModelWeights` configured as a Starcoder2-style arch.
///
/// Enables the dormant branches:
/// - **LayerNorm (not RMSNorm)** — `norm_type()` is `LayerNorm`
/// - **Non-gated FFN** — `ffn_type()` is `Standard` (non-gated),
///   exercising the `else` arm in `ffn/weight.rs::dense_ffn_forward_backend`
/// - **FFN bias** — `ffn_up_bias_key` / `ffn_down_bias_key` return Some,
///   so the `add_bias` calls fire
/// - **Attention bias** — `attn_q_bias_key` / `attn_k_bias_key` /
///   `attn_v_bias_key` / `attn_o_bias_key` return Some
/// - **c_fc/c_proj naming** — `ffn_up_key` / `ffn_down_key` use
///   `mlp.c_fc.weight` / `mlp.c_proj.weight` rather than gate/up/down
/// - **GeluTanh activation** — `activation()` is `GeluTanh`
pub fn make_starcoder2_test_weights() -> ModelWeights {
    const VOCAB: usize = 32;
    const HIDDEN: usize = 16;
    const INTER: usize = 32;
    const NUM_Q: usize = 2;
    const NUM_KV: usize = 1;
    const HEAD_DIM: usize = 8;
    const NUM_LAYERS: usize = 2;
>>>>>>> ianblenke/main:crates/larql-inference/src/engines/test_utils.rs

impl Q4KTestFixtures {
    pub fn build() -> Self {
        let weights = make_test_q4k_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let index = make_test_q4k_vindex(&weights);
        Self {
            weights,
            tokenizer,
            index,
        }
<<<<<<< HEAD:crates/larql-inference/src/test_utils.rs
    }
}

#[cfg(test)]
mod synthetic_model_dir_tests {
    use super::*;
    use larql_vindex::{load_vindex_config, SilentLoadCallbacks};

    #[test]
    fn write_then_load_round_trips() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_synthetic_model_dir(dir.path()).expect("write fixture");

        // 1. Config round-trips with the flags the EXPLAIN INFER pipeline gates on.
        let config = load_vindex_config(dir.path()).expect("load_vindex_config");
        assert!(
            config.has_model_weights,
            "fixture must set has_model_weights=true"
        );
        assert_eq!(config.quant, larql_vindex::QuantFormat::None);
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.hidden_size, 16);
        let mc = config.model_config.as_ref().expect("model_config");
        assert_eq!(mc.model_type, "tinymodel");
        assert_eq!(mc.head_dim, 8);

        // 2. Weights load via the same path InferenceWeights::load uses.
        let mut cb = SilentLoadCallbacks;
        let weights = larql_vindex::load_model_weights(dir.path(), &mut cb)
            .expect("load_model_weights against synthetic fixture");
        assert_eq!(weights.num_layers, 2);
        assert_eq!(weights.hidden_size, 16);
        assert_eq!(weights.vocab_size, 32);
        // Round-tripped tensors must be retrievable by the arch-keyed
        // names the forward pass walks — pick a representative entry.
        assert!(
            weights.tensors.contains_key(&weights.arch.attn_q_key(0)),
            "expected attn_q tensor for layer 0 after round-trip"
        );
        assert!(weights.tensors.contains_key(&weights.arch.ffn_gate_key(0)));
    }

    #[test]
    fn tokenizer_file_is_present_and_loadable() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_synthetic_model_dir(dir.path()).expect("write fixture");
        let tok_path = dir.path().join("tokenizer.json");
        assert!(tok_path.exists(), "tokenizer.json must be written");
        let _ = tokenizers::Tokenizer::from_file(&tok_path).expect("tokenizer round-trips");
    }

    #[test]
    fn embeddings_bin_has_expected_size() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_synthetic_model_dir(dir.path()).expect("write fixture");
        let bytes = std::fs::read(dir.path().join("embeddings.bin")).expect("embeddings.bin");
        // 32 vocab × 16 hidden × 4 bytes = 2048
        assert_eq!(bytes.len(), 32 * 16 * 4);
    }
}

// ── MockGpuBackend — Q4-capable mock for the GPU decode/prefill paths ────────
//
// Production Metal-only paths (`gpu/decode_loop.rs`, `gpu/prefill.rs`,
// `gpu/forced_logits.rs`, `gpu/mod.rs`, `vindex/kquant_forward/metal.rs`)
// short-circuit when `backend.supports(Capability::DecodeToken | PrefillQ4)`
// returns false — which is the case for `CpuBackend`. To exercise the
// actual function bodies under test we need a backend that advertises
// those capabilities and returns shape-correct (but content-garbage) data
// from `decode_token` / `prefill_kquant`.
//
// Math methods delegate to a wrapped `CpuBackend` so test code that
// happens to read intermediate tensors gets non-garbage values where it
// can; the canned-shape returns from `decode_token` / `prefill_kquant` are
// fine for coverage because the calling code's contract is just
// `Some(Vec<f32>)` of the right length.

/// Minimal Q4-capable compute backend for tests. Delegates math to
/// `CpuBackend` and overrides `supports` + `decode_token` + `prefill_kquant`
/// so the GPU paths in `larql-inference` execute end-to-end. Output
/// values are zeros — tests assert *shape* and *that the call returned
/// Some*, not numerical correctness.
pub struct MockGpuBackend {
    inner: larql_compute::CpuBackend,
    kv_len: std::sync::atomic::AtomicUsize,
}

impl Default for MockGpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MockGpuBackend {
    pub fn new() -> Self {
        Self {
            inner: larql_compute::CpuBackend,
            kv_len: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl larql_compute::MatMul for MockGpuBackend {
    fn matmul(
        &self,
        a: ndarray::ArrayView2<f32>,
        b: ndarray::ArrayView2<f32>,
    ) -> ndarray::Array2<f32> {
        self.inner.matmul(a, b)
    }
    fn matmul_transb(
        &self,
        a: ndarray::ArrayView2<f32>,
        b: ndarray::ArrayView2<f32>,
    ) -> ndarray::Array2<f32> {
        self.inner.matmul_transb(a, b)
    }
}

impl larql_compute::QuantMatVec for MockGpuBackend {
    fn supports_quant(&self, format: larql_compute::QuantFormat) -> bool {
        self.inner.supports_quant(format)
    }
}

impl larql_compute::DecodeBackend for MockGpuBackend {
    fn has_kv_cache(&self) -> bool {
        true
    }

    fn reset_kv_cache(&self) {
        self.kv_len.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    fn kv_cache_len(&self) -> usize {
        self.kv_len.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn truncate_kv_cache(&self, len: usize) {
        self.kv_len.store(len, std::sync::atomic::Ordering::Relaxed);
    }

    fn preallocate_kv_cache_per_layer(&self, _shapes: &[(usize, usize)], _max_seq: usize) {
        // No-op — we don't actually hold a cache, just a length counter.
    }

    fn decode_token(
        &self,
        _layers: &[larql_compute::FullPipelineLayer<'_>],
        _x: &[f32],
        hidden: usize,
        _inter: usize,
    ) -> Option<Vec<f32>> {
        self.kv_len
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(vec![0.0f32; hidden])
    }

    fn decode_token_with_moe(
        &self,
        _layers: &[larql_compute::FullPipelineLayer<'_>],
        _x: &[f32],
        hidden: usize,
        _inter: usize,
        moe_fn: &mut dyn FnMut(usize, &[f32]) -> Vec<f32>,
    ) -> Option<Vec<f32>> {
        // Invoke the MoE callback once with a zero residual so the
        // expert dispatch path runs end-to-end.
        let _ = moe_fn(0, &vec![0.0f32; hidden]);
        self.kv_len
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(vec![0.0f32; hidden])
    }

    fn decode_token_q4k_moe<'w>(
        &self,
        _layers: &[larql_compute::FullPipelineLayer<'_>],
        _x: &[f32],
        hidden: usize,
        _inter: usize,
        _norm_eps: f32,
        get_expert: &dyn Fn(usize, usize) -> Option<(&'w [u8], &'w [u8])>,
    ) -> Option<Vec<f32>> {
        let _ = get_expert(0, 0);
        self.kv_len
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(vec![0.0f32; hidden])
    }

    fn prefill_kquant(
        &self,
        _layers: &[larql_compute::FullPipelineLayer<'_>],
        _x: &[f32],
        hidden: usize,
        _inter: usize,
        seq_len: usize,
        _use_qk_norm: bool,
        _softcap: f32,
    ) -> Option<Vec<f32>> {
        self.kv_len
            .store(seq_len, std::sync::atomic::Ordering::Relaxed);
        Some(vec![0.0f32; seq_len * hidden])
    }
}

impl larql_compute::ComputeBackend for MockGpuBackend {
    fn name(&self) -> &str {
        "mock-gpu"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn supports(&self, cap: larql_compute::backend::Capability) -> bool {
        use larql_compute::backend::Capability::*;
        matches!(
            cap,
            DecodeToken
                | DecodeMoe
                | DecodeQ4KMoe
                | PrefillQ4
                | FullPipelineQ4
                | QuantMatVec
                | Q4VecMat
                | Q4PairBatch
        )
    }
}

#[cfg(test)]
mod mock_gpu_backend_tests {
    use super::*;
    use larql_compute::backend::Capability;
    use larql_compute::prelude::*;

    #[test]
    fn mock_advertises_decode_token_capability() {
        let mock = MockGpuBackend::new();
        assert!(mock.supports(Capability::DecodeToken));
        assert!(mock.supports(Capability::PrefillQ4));
        assert!(mock.supports(Capability::DecodeQ4KMoe));
        assert_eq!(mock.name(), "mock-gpu");
    }

    #[test]
    fn mock_decode_token_returns_hidden_sized_vector() {
        let mock = MockGpuBackend::new();
        let out = mock.decode_token(&[], &[], 8, 16).expect("Some");
        assert_eq!(out.len(), 8);
        assert_eq!(mock.kv_cache_len(), 1);
    }

    #[test]
    fn mock_prefill_q4_returns_seq_x_hidden_vector() {
        let mock = MockGpuBackend::new();
        let out = mock
            .prefill_kquant(&[], &[], 4, 16, 3, false, 0.0)
            .expect("Some");
        assert_eq!(out.len(), 3 * 4);
        assert_eq!(mock.kv_cache_len(), 3);
    }

    #[test]
    fn mock_reset_clears_kv_len() {
        let mock = MockGpuBackend::new();
        let _ = mock.prefill_kquant(&[], &[], 4, 16, 5, false, 0.0);
        assert_eq!(mock.kv_cache_len(), 5);
        mock.reset_kv_cache();
        assert_eq!(mock.kv_cache_len(), 0);
    }

    #[test]
    fn mock_truncate_sets_kv_len() {
        let mock = MockGpuBackend::new();
        let _ = mock.prefill_kquant(&[], &[], 4, 16, 10, false, 0.0);
        mock.truncate_kv_cache(3);
        assert_eq!(mock.kv_cache_len(), 3);
    }

    #[test]
    fn mock_decode_with_moe_invokes_callback() {
        let mock = MockGpuBackend::new();
        let mut callback_fired = false;
        let mut moe_fn = |_layer: usize, _h: &[f32]| -> Vec<f32> {
            callback_fired = true;
            vec![0.0f32; 8]
        };
        let _ = mock.decode_token_with_moe(&[], &[], 8, 16, &mut moe_fn);
        assert!(callback_fired);
    }

    #[test]
    fn mock_decode_q4k_moe_invokes_expert_lookup() {
        let mock = MockGpuBackend::new();
        let lookup_count = std::cell::Cell::new(0);
        let bytes = [0u8; 16];
        let get_expert = |_layer: usize, _expert: usize| -> Option<(&[u8], &[u8])> {
            lookup_count.set(lookup_count.get() + 1);
            Some((&bytes[..], &bytes[..]))
        };
        let _ = mock.decode_token_q4k_moe(&[], &[], 8, 16, 1e-6, &get_expert);
        assert!(lookup_count.get() >= 1);
=======
        if let Some(k) = arch.attn_k_bias_key(layer) {
            vectors.insert(k, vec![0.01; kv_dim]);
        }
        if let Some(k) = arch.attn_v_bias_key(layer) {
            vectors.insert(k, vec![0.01; kv_dim]);
        }
        if let Some(k) = arch.attn_o_bias_key(layer) {
            vectors.insert(k, vec![0.01; HIDDEN]);
        }

        // FFN — non-gated (c_fc + c_proj). The keys map through
        // arch.ffn_up_key/ffn_down_key which return c_fc/c_proj naming.
        tensors.insert(
            arch.ffn_up_key(layer),
            rand_mat_seeded(INTER, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.ffn_down_key(layer),
            rand_mat_seeded(HIDDEN, INTER, 0.1, next_seed()),
        );
        // Add gate too — gated-only code paths may probe it regardless
        // of arch's ffn_type; safer to populate it so a stray lookup
        // doesn't panic the test.
        tensors.insert(
            arch.ffn_gate_key(layer),
            rand_mat_seeded(INTER, HIDDEN, 0.1, next_seed()),
        );

        // FFN biases — Starcoder2 has them.
        if let Some(k) = arch.ffn_up_bias_key(layer) {
            vectors.insert(k, vec![0.01; INTER]);
        }
        if let Some(k) = arch.ffn_down_bias_key(layer) {
            vectors.insert(k, vec![0.01; HIDDEN]);
        }

        // Layer norms — Starcoder2 uses LayerNorm, norm_weight_offset=0,
        // so weights are the actual scale.
        vectors.insert(arch.input_layernorm_key(layer), vec![1.0; HIDDEN]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![1.0; HIDDEN]);
    }

    ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        packed_mmaps: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: HashMap::new(),
        embed: larql_models::embed::EmbedMatrix::from_array(embed),
        embed_quant: None,
        lm_head,
        lm_head_quant: None,
        position_embed: None,
        quant_tensors: HashMap::new(),
        arch,
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        vocab_size: VOCAB,
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        rope_base: 10_000.0,
>>>>>>> ianblenke/main:crates/larql-inference/src/engines/test_utils.rs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma3_fixture_has_qk_norm_and_post_norm_keys() {
        let w = make_gemma3_test_weights();
        // QK norm keys present
        for layer in 0..w.num_layers {
            let qk = w.arch.attn_q_norm_key(layer).expect("Gemma3 has q_norm");
            let kk = w.arch.attn_k_norm_key(layer).expect("Gemma3 has k_norm");
            assert!(w.vectors.contains_key(&qk), "missing {qk}");
            assert!(w.vectors.contains_key(&kk), "missing {kk}");
        }
        // post-norms branch
        assert!(
            w.arch.has_post_norms(),
            "Gemma3 fixture must have post_norms=true"
        );
        // embed scale is sqrt(hidden)
        let expected = (w.hidden_size as f32).sqrt();
        assert!((w.arch.embed_scale() - expected).abs() < 1e-6);
        // norm offset is 1.0
        assert_eq!(w.arch.norm_weight_offset(), 1.0);
        // family
        assert_eq!(w.arch.family(), "gemma3");
    }

    #[test]
    fn starcoder2_fixture_has_layernorm_and_biases() {
        use larql_models::{FfnType, NormType};
        let w = make_starcoder2_test_weights();
        assert_eq!(w.arch.family(), "starcoder2");
        // LayerNorm (not RMSNorm)
        assert!(matches!(w.arch.norm_type(), NormType::LayerNorm));
        // Standard (non-gated) FFN
        assert!(matches!(w.arch.ffn_type(), FfnType::Standard));
        // FFN keys use c_fc/c_proj naming
        let up_key = w.arch.ffn_up_key(0);
        assert!(
            up_key.contains("c_fc"),
            "expected c_fc in {up_key}, got {up_key}"
        );
        let down_key = w.arch.ffn_down_key(0);
        assert!(down_key.contains("c_proj"), "expected c_proj in {down_key}");
        // Attention biases present
        for layer in 0..w.num_layers {
            let qb = w
                .arch
                .attn_q_bias_key(layer)
                .expect("Starcoder2 has q bias");
            assert!(w.vectors.contains_key(&qb), "missing {qb}");
        }
        // FFN biases present
        for layer in 0..w.num_layers {
            let upb = w
                .arch
                .ffn_up_bias_key(layer)
                .expect("Starcoder2 has up bias");
            assert!(w.vectors.contains_key(&upb), "missing {upb}");
        }
    }

    #[test]
    fn gemma3_fixture_shape_matches_dims() {
        let w = make_gemma3_test_weights();
        assert_eq!(w.num_layers, 2);
        assert_eq!(w.embed.shape(), [w.vocab_size, w.hidden_size]);
        assert_eq!(w.lm_head.shape(), &[w.vocab_size, w.hidden_size]);
    }

    #[test]
    fn starcoder2_fixture_shape_matches_dims() {
        let w = make_starcoder2_test_weights();
        assert_eq!(w.num_layers, 2);
        assert_eq!(w.embed.shape(), [w.vocab_size, w.hidden_size]);
        assert_eq!(w.lm_head.shape(), &[w.vocab_size, w.hidden_size]);
    }
}
