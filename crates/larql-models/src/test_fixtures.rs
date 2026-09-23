//! Shared test fixtures for `ModelWeights` consumers.
//!
//! Gated behind the `test-utils` feature so production builds never
//! pull in the synthetic builders. Downstream test crates
//! (`larql-compute`, `larql-inference`, `larql-vindex`, `larql-kv`)
//! depend on `larql-models` with `features = ["test-utils"]` under
//! `[dev-dependencies]` to construct realistic `ModelWeights` without
//! disk I/O.
//!
//! Architecture-specific fixtures (Gemma 3, StarCoder2, Q4K, MoE, E2B)
//! still live in `crates/larql-inference/src/test_utils.rs` because
//! they pull in inference-side concepts (vindex, tokenizer, mock GPU
//! backends). Only the generic `TinyModel` builder lives here — it's
//! the one the moved-down forward-pass tests in `larql-compute` need.

use crate::{detect_from_json, ModelWeights, WeightArray};
use ndarray::Array2;
use std::collections::HashMap;

/// Build a synthetic `ModelWeights` with all tensors populated.
///
/// Uses `TinyModelArch` key conventions
/// (e.g. `"0.attn.q_proj.weight"`). Dimensions: vocab=32, hidden=16,
/// intermediate=32, 2 q-heads, 1 kv-head, head_dim=8, 2 layers.
/// Forward pass ≈ 10 ms on CPU.
pub fn make_test_weights() -> ModelWeights {
    const VOCAB: usize = 32;
    const HIDDEN: usize = 16;
    const INTER: usize = 32;
    const NUM_Q: usize = 2;
    const NUM_KV: usize = 1;
    const HEAD_DIM: usize = 8;
    const NUM_LAYERS: usize = 2;

    let arch_json = serde_json::json!({
        "model_type": "tinymodel",
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "intermediate_size": INTER,
        "head_dim": HEAD_DIM,
        "num_attention_heads": NUM_Q,
        "num_key_value_heads": NUM_KV,
        "vocab_size": VOCAB,
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

    // Embed + lm_head
    let embed = rand_mat(VOCAB, HIDDEN, 0.1);
    let lm_head = rand_mat(VOCAB, HIDDEN, 0.1);
    tensors.insert(arch.embed_key().to_string(), embed.clone());

    // Final norm (ones → valid unweighted RMSNorm fallback)
    vectors.insert(arch.final_norm_key().to_string(), vec![1.0; HIDDEN]);

    let q_dim = NUM_Q * HEAD_DIM;
    let kv_dim = NUM_KV * HEAD_DIM;

    for layer in 0..NUM_LAYERS {
        // Attention projections
        tensors.insert(arch.attn_q_key(layer), rand_mat(q_dim, HIDDEN, 0.1));
        tensors.insert(arch.attn_k_key(layer), rand_mat(kv_dim, HIDDEN, 0.1));
        tensors.insert(arch.attn_v_key(layer), rand_mat(kv_dim, HIDDEN, 0.1));
        tensors.insert(arch.attn_o_key(layer), rand_mat(HIDDEN, q_dim, 0.1));
        // FFN — missing tensors cause panic, so always provide them
        tensors.insert(arch.ffn_gate_key(layer), rand_mat(INTER, HIDDEN, 0.1));
        tensors.insert(arch.ffn_up_key(layer), rand_mat(INTER, HIDDEN, 0.1));
        tensors.insert(arch.ffn_down_key(layer), rand_mat(HIDDEN, INTER, 0.1));
        // Layer norms
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
        embed,
        lm_head,
        position_embed: None,
        arch,
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        vocab_size: VOCAB,
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        rope_base: 10_000.0,
    }
}

// ── Seeded RNG helper shared by arch-specific fixtures ──

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

/// Build a synthetic `ModelWeights` configured as a Gemma 3-style arch.
///
/// Enables the dormant branches in `attention/{block, gpu}.rs` and
/// `forward/layer.rs` that tinymodel never reaches:
/// - **QK norm** — `attn_q_norm_key` / `attn_k_norm_key` return Some
/// - **post norms** — `has_post_norms()` is true; pre/post FFN norm keys
///   are populated, the FFN dispatch routes through the post-norm arm
/// - **GeluTanh activation** — `activation()` is `GeluTanh`
/// - **`embed_scale = sqrt(hidden)`** — non-1.0 embed scaling
/// - **`norm_weight_offset = 1.0`** — non-zero offset added to every
///   norm weight at runtime
pub fn make_gemma3_test_weights() -> ModelWeights {
    const HIDDEN: usize = 16;
    const INTER: usize = 32;
    const NUM_Q: usize = 2;
    const NUM_KV: usize = 1;
    const HEAD_DIM: usize = 8;
    const VOCAB: usize = 32;
    const NUM_LAYERS: usize = 2;
    gemma3_test_weights_inner(
        serde_json::json!({
            "model_type": "gemma3",
            "hidden_size": HIDDEN,
            "num_hidden_layers": NUM_LAYERS,
            "intermediate_size": INTER,
            "head_dim": HEAD_DIM,
            "num_attention_heads": NUM_Q,
            "num_key_value_heads": NUM_KV,
            "vocab_size": VOCAB,
            "rope_theta": 10000.0,
            "residual_multiplier": 0.5,
        }),
        NUM_LAYERS,
    )
}

/// Like [`make_gemma3_test_weights`] but with the structured per-layer-type
/// `rope_scaling` form (linear factor 8 on full-attention layers, default on
/// sliding) and **6 layers**, so layer 5 is a global/full-attention layer with
/// `position_divisor = 8`. Used to exercise the scaled-RoPE path — the engine
/// prefill and full-recompute attention must agree there (regression guard for
/// the 2026-05-28 prefill-RoPE divergence).
pub fn make_gemma3_rope_scaled_test_weights() -> ModelWeights {
    const HIDDEN: usize = 16;
    const INTER: usize = 32;
    const NUM_Q: usize = 2;
    const NUM_KV: usize = 1;
    const HEAD_DIM: usize = 8;
    const VOCAB: usize = 32;
    const NUM_LAYERS: usize = 6;
    gemma3_test_weights_inner(
        serde_json::json!({
            "model_type": "gemma3",
            "text_config": {
                "model_type": "gemma3_text",
                "hidden_size": HIDDEN,
                "num_hidden_layers": NUM_LAYERS,
                "intermediate_size": INTER,
                "head_dim": HEAD_DIM,
                "num_attention_heads": NUM_Q,
                "num_key_value_heads": NUM_KV,
                "vocab_size": VOCAB,
                "rope_theta": 10000.0,
                "residual_multiplier": 0.5,
                "sliding_window": 1024,
                "rope_scaling": {
                    "full_attention": {"rope_type": "linear", "factor": 8.0},
                    "sliding_attention": {"rope_type": "default"},
                },
            },
        }),
        NUM_LAYERS,
    )
}

fn gemma3_test_weights_inner(arch_json: serde_json::Value, num_layers: usize) -> ModelWeights {
    const VOCAB: usize = 32;
    const HIDDEN: usize = 16;
    const INTER: usize = 32;
    const NUM_Q: usize = 2;
    const NUM_KV: usize = 1;
    const HEAD_DIM: usize = 8;
    let arch = detect_from_json(&arch_json);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    let q_dim = NUM_Q * HEAD_DIM;
    let kv_dim = NUM_KV * HEAD_DIM;

    let embed = rand_mat_seeded(VOCAB, HIDDEN, 0.1, 0x9e3779b9);
    let lm_head = rand_mat_seeded(VOCAB, HIDDEN, 0.1, 0xa1b2c3d4);
    tensors.insert(arch.embed_key().to_string(), embed.clone());

    // Gemma 3: norm_weight_offset=1.0; saved weight is delta off identity.
    vectors.insert(arch.final_norm_key().to_string(), vec![0.0; HIDDEN]);

    let mut seed_counter: u64 = 0xdeadbeef;
    let mut next_seed = || {
        seed_counter = seed_counter.wrapping_add(0x9e3779b97f4a7c15);
        seed_counter
    };

    for layer in 0..num_layers {
        tensors.insert(
            arch.attn_q_key(layer),
            rand_mat_seeded(q_dim, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.attn_k_key(layer),
            rand_mat_seeded(kv_dim, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.attn_v_key(layer),
            rand_mat_seeded(kv_dim, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.attn_o_key(layer),
            rand_mat_seeded(HIDDEN, q_dim, 0.1, next_seed()),
        );

        tensors.insert(
            arch.ffn_gate_key(layer),
            rand_mat_seeded(INTER, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.ffn_up_key(layer),
            rand_mat_seeded(INTER, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.ffn_down_key(layer),
            rand_mat_seeded(HIDDEN, INTER, 0.1, next_seed()),
        );

        // Layer norms — input + post-attention. Gemma 3 norm_weight_offset=1.0
        // means saved weights are deltas; zeros → identity at runtime.
        vectors.insert(arch.input_layernorm_key(layer), vec![0.0; HIDDEN]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![0.0; HIDDEN]);
        if let Some(k) = arch.pre_feedforward_layernorm_key(layer) {
            vectors.insert(k, vec![0.0; HIDDEN]);
        }
        if let Some(k) = arch.post_feedforward_layernorm_key(layer) {
            vectors.insert(k, vec![0.0; HIDDEN]);
        }

        // QK norm — per-head dim weights.
        if let Some(k) = arch.attn_q_norm_key(layer) {
            vectors.insert(k, vec![0.0; HEAD_DIM]);
        }
        if let Some(k) = arch.attn_k_norm_key(layer) {
            vectors.insert(k, vec![0.0; HEAD_DIM]);
        }
    }

    ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        packed_mmaps: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: HashMap::new(),
        embed,
        lm_head,
        position_embed: None,
        arch,
        num_layers,
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        vocab_size: VOCAB,
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        rope_base: 10_000.0,
    }
}

/// Build a synthetic `ModelWeights` configured as a Starcoder2-style arch.
///
/// Enables the dormant branches:
/// - **Non-gated FFN** — `ffn_type()` is `NonGated`, exercising the
///   `else` arm in `ffn/weight.rs::dense_ffn_forward_backend`
/// - **FFN bias** — `ffn_up_bias_key` / `ffn_down_bias_key` return Some
/// - **Attention bias** — `attn_*_bias_key` all return Some
/// - **Gelu activation** — `activation()` is `Gelu`
pub fn make_starcoder2_test_weights() -> ModelWeights {
    const VOCAB: usize = 32;
    const HIDDEN: usize = 16;
    const INTER: usize = 32;
    const NUM_Q: usize = 2;
    const NUM_KV: usize = 1;
    const HEAD_DIM: usize = 8;
    const NUM_LAYERS: usize = 2;

    let arch_json = serde_json::json!({
        "model_type": "starcoder2",
        "hidden_size": HIDDEN,
        "num_hidden_layers": NUM_LAYERS,
        "intermediate_size": INTER,
        "head_dim": HEAD_DIM,
        "num_attention_heads": NUM_Q,
        "num_key_value_heads": NUM_KV,
        "vocab_size": VOCAB,
        "residual_multiplier": 0.5,
        "attention_multiplier": 2.0,
    });
    let arch = detect_from_json(&arch_json);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    let q_dim = NUM_Q * HEAD_DIM;
    let kv_dim = NUM_KV * HEAD_DIM;

    let embed = rand_mat_seeded(VOCAB, HIDDEN, 0.1, 0x12345678);
    let lm_head = rand_mat_seeded(VOCAB, HIDDEN, 0.1, 0x87654321);
    tensors.insert(arch.embed_key().to_string(), embed.clone());

    vectors.insert(arch.final_norm_key().to_string(), vec![1.0; HIDDEN]);

    let mut seed_counter: u64 = 0xfeedbabe;
    let mut next_seed = || {
        seed_counter = seed_counter.wrapping_add(0x9e3779b97f4a7c15);
        seed_counter
    };

    for layer in 0..NUM_LAYERS {
        tensors.insert(
            arch.attn_q_key(layer),
            rand_mat_seeded(q_dim, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.attn_k_key(layer),
            rand_mat_seeded(kv_dim, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.attn_v_key(layer),
            rand_mat_seeded(kv_dim, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.attn_o_key(layer),
            rand_mat_seeded(HIDDEN, q_dim, 0.1, next_seed()),
        );

        if let Some(k) = arch.attn_q_bias_key(layer) {
            vectors.insert(k, vec![0.01; q_dim]);
        }
        if let Some(k) = arch.attn_k_bias_key(layer) {
            vectors.insert(k, vec![0.01; kv_dim]);
        }
        if let Some(k) = arch.attn_v_bias_key(layer) {
            vectors.insert(k, vec![0.01; kv_dim]);
        }
        if let Some(k) = arch.attn_o_bias_key(layer) {
            vectors.insert(k, vec![0.01; HIDDEN]);
        }

        tensors.insert(
            arch.ffn_up_key(layer),
            rand_mat_seeded(INTER, HIDDEN, 0.1, next_seed()),
        );
        tensors.insert(
            arch.ffn_down_key(layer),
            rand_mat_seeded(HIDDEN, INTER, 0.1, next_seed()),
        );
        tensors.insert(
            arch.ffn_gate_key(layer),
            rand_mat_seeded(INTER, HIDDEN, 0.1, next_seed()),
        );

        if let Some(k) = arch.ffn_up_bias_key(layer) {
            vectors.insert(k, vec![0.01; INTER]);
        }
        if let Some(k) = arch.ffn_down_bias_key(layer) {
            vectors.insert(k, vec![0.01; HIDDEN]);
        }

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
        embed,
        lm_head,
        position_embed: None,
        arch,
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN,
        intermediate_size: INTER,
        vocab_size: VOCAB,
        head_dim: HEAD_DIM,
        num_q_heads: NUM_Q,
        num_kv_heads: NUM_KV,
        rope_base: 10_000.0,
    }
}

// ── Gemma 4 E2B-like synthetic fixture (PLE-aware) ──

/// Tiny synthetic Gemma-4-E2B-shaped arch with PLE + KV sharing.
///
/// Same shape as `crates/larql-models/tests/test_architectures.rs::gemma4_e2b_arch`
/// but smaller (4 layers, hidden=8) so weights fit in-memory cheaply.
/// Shared with `layer_graph::pipeline_layer::tests` and the `forward::ple::tests`
/// module — both need `has_per_layer_embeddings()=true` AND valid PLE tensor
/// keys populated in `weights.tensors` / `weights.vectors`.
pub fn synthetic_e2b_like_arch_json() -> serde_json::Value {
    serde_json::json!({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "global_head_dim": 8,
            "vocab_size": 32,
            "sliding_window": 4,
            "hidden_size_per_layer_input": 4,
            "num_kv_shared_layers": 2,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0
                },
                "sliding_attention": {"rope_theta": 10000.0}
            },
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention"
            ]
        }
    })
}

/// Weight amplitude for the synthetic E2B-like fixture's LCG-seeded tensors.
pub const E2B_TEST_WEIGHT_SCALE: f32 = 0.05;
/// Per-layer `layer_scalar` value for the synthetic E2B-like fixture.
/// Deliberately neither 0.0 (skipped as absent) nor 1.0 (skipped as
/// identity) so `apply_layer_scalar` visibly changes the hidden state —
/// a forward path that drops the scalar diverges instead of matching.
pub const E2B_TEST_LAYER_SCALAR: f32 = 0.75;

/// Build minimal `ModelWeights` matching the synthetic E2B-like arch.
/// Tensors carry deterministic non-zero LCG values (not zeros) so the
/// PLE and `layer_scalar` contributions are non-trivial: parity tests
/// between forward paths fail if either step is dropped, not just when
/// the tensor keys are missing.
pub fn make_synthetic_e2b_like_weights() -> ModelWeights {
    let arch = detect_from_json(&synthetic_e2b_like_arch_json());
    let num_layers = 4;
    let hidden = 8;
    let intermediate = 16;
    let head_dim = 4;
    let global_head_dim = 8;
    let num_q_heads = 2;
    let num_kv_heads = 1;
    let vocab_size = 32;
    let ple_dim = 4;

    let mut tensors: std::collections::HashMap<String, WeightArray> =
        std::collections::HashMap::new();
    let mut vectors: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();

    let mut seed = 0xe2b_0000_u64;
    let mut next_seed = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };
    let scale = E2B_TEST_WEIGHT_SCALE;

    let embed = rand_mat_seeded(vocab_size, hidden, scale, next_seed());
    let lm_head = embed.clone();
    tensors.insert(arch.embed_key().to_string(), embed.clone());
    vectors.insert(arch.final_norm_key().to_string(), vec![1.0; hidden]);

    if let Some(k) = arch.per_layer_model_projection_key() {
        tensors.insert(
            k,
            rand_mat_seeded(num_layers * ple_dim, hidden, scale, next_seed()),
        );
    }
    if let Some(k) = arch.per_layer_embed_key() {
        tensors.insert(
            k,
            rand_mat_seeded(vocab_size, num_layers * ple_dim, scale, next_seed()),
        );
    }
    if let Some(k) = arch.per_layer_projection_norm_key() {
        vectors.insert(k, vec![1.0; ple_dim]);
    }

    for layer in 0..num_layers {
        let layer_head_dim = if arch.is_sliding_window_layer(layer) {
            head_dim
        } else {
            global_head_dim
        };
        let q_dim = num_q_heads * layer_head_dim;
        let kv_dim = num_kv_heads * layer_head_dim;
        tensors.insert(
            arch.attn_q_key(layer),
            rand_mat_seeded(q_dim, hidden, scale, next_seed()),
        );
        tensors.insert(
            arch.attn_k_key(layer),
            rand_mat_seeded(kv_dim, hidden, scale, next_seed()),
        );
        tensors.insert(
            arch.attn_v_key(layer),
            rand_mat_seeded(kv_dim, hidden, scale, next_seed()),
        );
        tensors.insert(
            arch.attn_o_key(layer),
            rand_mat_seeded(hidden, q_dim, scale, next_seed()),
        );
        tensors.insert(
            arch.ffn_gate_key(layer),
            rand_mat_seeded(intermediate, hidden, scale, next_seed()),
        );
        tensors.insert(
            arch.ffn_up_key(layer),
            rand_mat_seeded(intermediate, hidden, scale, next_seed()),
        );
        tensors.insert(
            arch.ffn_down_key(layer),
            rand_mat_seeded(hidden, intermediate, scale, next_seed()),
        );
        vectors.insert(arch.input_layernorm_key(layer), vec![1.0; hidden]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![1.0; hidden]);
        if let Some(k) = arch.per_layer_input_gate_key(layer) {
            tensors.insert(k, rand_mat_seeded(ple_dim, hidden, scale, next_seed()));
        }
        if let Some(k) = arch.per_layer_projection_key(layer) {
            tensors.insert(k, rand_mat_seeded(hidden, ple_dim, scale, next_seed()));
        }
        if let Some(k) = arch.post_per_layer_input_norm_key(layer) {
            vectors.insert(k, vec![1.0; hidden]);
        }
        if let Some(k) = arch.layer_scalar_key(layer) {
            vectors.insert(k, vec![E2B_TEST_LAYER_SCALAR]);
        }
    }

    ModelWeights {
        tensors,
        vectors,
        raw_bytes: std::collections::HashMap::new(),
        packed_mmaps: std::collections::HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: std::collections::HashMap::new(),
        embed,
        lm_head,
        position_embed: None,
        arch,
        num_layers,
        hidden_size: hidden,
        intermediate_size: intermediate,
        vocab_size,
        head_dim,
        num_q_heads,
        num_kv_heads,
        rope_base: 10_000.0,
    }
}

// ── Q4_K-aware synthetic fixtures (Step 3b) ──

// ── Q4_K-aware synthetic fixture ─────────────────────────────────────────
//
// `make_test_weights` uses hidden=16, below Q4_K's 256-element
// super-block minimum. The cached / direct-matvec decode paths in
// `vindex/kquant_forward/cached.rs` require a vindex with real
// `attn_kquant_layer_data` + `interleaved_kquant_layer_data` manifests,
// so unit tests for those paths can't fit the tiny fixture. The
// helpers below build a hidden=256, intermediate=256 Gemma 3-style
// fixture with synthetic Q4_K bytes that round-trip through
// `larql_compute::cpu::ops::q4_common::quantize_q4_k`.

/// Hidden dimension for the Q4_K test fixture — minimum Q4_K-safe
/// multiple of 256.
pub const Q4K_TEST_HIDDEN: usize = 256;
/// Intermediate dimension for the Q4_K test fixture.
pub const Q4K_TEST_INTER: usize = 256;
/// Vocabulary size for the Q4_K test fixture.
pub const Q4K_TEST_VOCAB: usize = 256;
/// Layer count for the Q4_K test fixture.
pub const Q4K_TEST_NUM_LAYERS: usize = 2;
/// Query-head count for the Q4_K test fixtures.
pub const Q4K_TEST_NUM_Q: usize = 4;
/// K/V-head count for the Q4_K test fixtures (GQA reps = 2).
pub const Q4K_TEST_NUM_KV: usize = 2;
/// Wide FFN width for the Q4_K fixture: threshold tests that route the
/// walk's parallel Q4K-down branch need `hits ≥ 512` while staying
/// below the full-K gemv rewrite at 80% density — so
/// `intermediate > 512 / 0.8 = 640`; 768 is the next 256-multiple
/// (Q4_K super-block constraint).
pub const Q4K_TEST_INTER_WIDE: usize = 768;

/// Build a synthetic `ModelWeights` sized to satisfy Q4_K's 256-element
/// super-block constraint. Uses Gemma 3 architecture so the
/// `has_post_norms` + `GeluTanh` branches in the cached decode path
/// are exercised. `Q4K_TEST_NUM_LAYERS` layers.
pub fn make_test_q4k_weights() -> ModelWeights {
    make_test_q4k_weights_layers(Q4K_TEST_NUM_LAYERS)
}

/// Layer-parametrised sibling of [`make_test_q4k_weights`]. Identical
/// arch / dims, but with `num_layers` decoder layers — for tests that
/// need a depth-fraction layer index to land inside the model (e.g. the
/// LQL FR3 relation resolver, whose probe layer clamps to ≥3, so it needs
/// a model deeper than the default 2-layer fixture).
pub fn make_test_q4k_weights_layers(num_layers: usize) -> ModelWeights {
    let num_q = 4usize;
    let num_kv = 2usize;
    let head_dim = Q4K_TEST_HIDDEN / num_q;

    let arch_json = serde_json::json!({
        "model_type": "gemma3_text",
        "hidden_size": Q4K_TEST_HIDDEN,
        "num_hidden_layers": num_layers,
        "intermediate_size": Q4K_TEST_INTER,
        "head_dim": head_dim,
        "num_attention_heads": num_q,
        "num_key_value_heads": num_kv,
        "vocab_size": Q4K_TEST_VOCAB,
        "hidden_activation": "gelu_pytorch_tanh",
        "rope_theta": 10000.0,
    });
    q4k_test_weights_from_json(
        arch_json,
        num_layers,
        Q4K_TEST_INTER,
        Q4K_TEST_HIDDEN,
        Q4K_TEST_NUM_Q,
        Q4K_TEST_NUM_KV,
    )
}

/// Wide-FFN sibling of [`make_test_q4k_weights`]: same Gemma 3 arch and
/// hidden size, but `intermediate_size = Q4K_TEST_INTER_WIDE` (768) and a
/// single layer (the wide FFN triples quantisation cost per layer). Built
/// for walk-engine threshold tests that need a route of ≥ 512 features
/// without triggering the 80%-density full-K gemv rewrite.
pub fn make_test_q4k_weights_wide() -> ModelWeights {
    let num_q = 4usize;
    let num_kv = 2usize;
    let head_dim = Q4K_TEST_HIDDEN / num_q;
    let num_layers = 1usize;

    let arch_json = serde_json::json!({
        "model_type": "gemma3_text",
        "hidden_size": Q4K_TEST_HIDDEN,
        "num_hidden_layers": num_layers,
        "intermediate_size": Q4K_TEST_INTER_WIDE,
        "head_dim": head_dim,
        "num_attention_heads": num_q,
        "num_key_value_heads": num_kv,
        "vocab_size": Q4K_TEST_VOCAB,
        "hidden_activation": "gelu_pytorch_tanh",
        "rope_theta": 10000.0,
    });
    q4k_test_weights_from_json(
        arch_json,
        num_layers,
        Q4K_TEST_INTER_WIDE,
        Q4K_TEST_HIDDEN,
        Q4K_TEST_NUM_Q,
        Q4K_TEST_NUM_KV,
    )
}

/// Rope-scaled sibling of [`make_test_q4k_weights`]: Gemma-3 arch at the
/// same Q4_K-compatible dims, with `sliding_window` + flat linear
/// `rope_scaling` (factor 8) and **6 layers**, so the 5:1 local:global
/// pattern puts a full-attention layer (position divisor 8) inside the
/// model. This is the fixture for direct-vs-staged decode parity on the
/// scaled-RoPE path — [`make_gemma3_rope_scaled_test_weights`] covers the
/// same arch shape but at hidden=16, below Q4_K's 256-element super-block
/// floor, so it cannot drive the direct-matvec kernels.
pub fn make_test_q4k_weights_rope_scaled() -> ModelWeights {
    let num_q = 4usize;
    let num_kv = 2usize;
    let head_dim = Q4K_TEST_HIDDEN / num_q;
    let num_layers = 6usize;

    let arch_json = serde_json::json!({
        "model_type": "gemma3_text",
        "hidden_size": Q4K_TEST_HIDDEN,
        "num_hidden_layers": num_layers,
        "intermediate_size": Q4K_TEST_INTER,
        "head_dim": head_dim,
        "num_attention_heads": num_q,
        "num_key_value_heads": num_kv,
        "vocab_size": Q4K_TEST_VOCAB,
        "hidden_activation": "gelu_pytorch_tanh",
        "rope_theta": 10000.0,
        "sliding_window": 512,
        "rope_scaling": {"rope_type": "linear", "factor": 8.0},
    });
    q4k_test_weights_from_json(
        arch_json,
        num_layers,
        Q4K_TEST_INTER,
        Q4K_TEST_HIDDEN,
        Q4K_TEST_NUM_Q,
        Q4K_TEST_NUM_KV,
    )
}

fn q4k_test_weights_from_json(
    arch_json: serde_json::Value,
    num_layers: usize,
    intermediate: usize,
    hidden: usize,
    num_q: usize,
    num_kv: usize,
) -> ModelWeights {
    let head_dim = hidden / num_q;
    let arch = detect_from_json(&arch_json);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    let mut seed = 0xc0ffee_u64;
    let mut next_seed = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };

    let embed = rand_mat_seeded(Q4K_TEST_VOCAB, hidden, 0.05, next_seed());
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
        tensors.insert(
            arch.ffn_gate_key(layer),
            rand_mat_seeded(intermediate, hidden, 0.05, next_seed()),
        );
        tensors.insert(
            arch.ffn_up_key(layer),
            rand_mat_seeded(intermediate, hidden, 0.05, next_seed()),
        );
        tensors.insert(
            arch.ffn_down_key(layer),
            rand_mat_seeded(hidden, intermediate, 0.05, next_seed()),
        );

        vectors.insert(arch.input_layernorm_key(layer), vec![0.5; hidden]);
        vectors.insert(arch.post_attention_layernorm_key(layer), vec![0.5; hidden]);
        if let Some(k) = arch.pre_feedforward_layernorm_key(layer) {
            vectors.insert(k, vec![0.5; hidden]);
        }
        if let Some(k) = arch.post_feedforward_layernorm_key(layer) {
            vectors.insert(k, vec![0.5; hidden]);
        }
        // QK-norm, on the architectures that declare it (Gemma 3 / 4).
        // These were declared-but-absent until 2026-08, which made the
        // fixture claim an architecture it did not actually carry: every
        // consumer that resolves the weight got `None` and silently
        // skipped the stage, so a backend disagreeing about what to do
        // with a missing QK-norm weight had nothing pinning it. Sized per
        // head_dim, not hidden — QK-norm normalises within a head.
        if let Some(k) = arch.attn_q_norm_key(layer) {
            vectors.insert(k, vec![0.5; head_dim]);
        }
        if let Some(k) = arch.attn_k_norm_key(layer) {
            vectors.insert(k, vec![0.5; head_dim]);
        }
    }

    ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        packed_mmaps: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: HashMap::new(),
        embed,
        lm_head,
        position_embed: None,
        arch,
        num_layers,
        hidden_size: hidden,
        intermediate_size: intermediate,
        vocab_size: Q4K_TEST_VOCAB,
        head_dim,
        num_q_heads: num_q,
        num_kv_heads: num_kv,
        rope_base: 10_000.0,
    }
}

/// SiLU sibling of [`make_test_q4k_weights`].
///
/// Uses the TinyModel architecture so the FFN activation is `Silu` and
/// the FFN type is `Gated`. Dimensions match the Q4_K constraints
/// (`Q4K_TEST_HIDDEN` is a multiple of 256) so the same `make_test_q4k_vindex`
/// can wrap the result. Needed by tests that exercise the SiLU branch in
/// quantised forward paths (e.g. `walk_ffn_kquant_dequant`'s `silu_gate_up`
/// arm) without depending on a Gemma3 fixture.
pub fn make_test_q4k_weights_silu() -> ModelWeights {
    let num_q = 4usize;
    let num_kv = 2usize;
    let head_dim = Q4K_TEST_HIDDEN / num_q;

    let arch_json = serde_json::json!({
        "model_type": "tinymodel",
        "hidden_size": Q4K_TEST_HIDDEN,
        "num_hidden_layers": Q4K_TEST_NUM_LAYERS,
        "intermediate_size": Q4K_TEST_INTER,
        "head_dim": head_dim,
        "num_attention_heads": num_q,
        "num_key_value_heads": num_kv,
        "vocab_size": Q4K_TEST_VOCAB,
    });
    let arch = detect_from_json(&arch_json);

    let mut tensors: HashMap<String, WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    let mut seed = 0xdeadc0de_u64;
    let mut next_seed = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        seed
    };

    let embed = rand_mat_seeded(Q4K_TEST_VOCAB, Q4K_TEST_HIDDEN, 0.05, next_seed());
    let lm_head = embed.clone();
    tensors.insert(arch.embed_key().to_string(), embed.clone());

    vectors.insert(
        arch.final_norm_key().to_string(),
        vec![1.0; Q4K_TEST_HIDDEN],
    );

    let q_dim = num_q * head_dim;
    let kv_dim = num_kv * head_dim;

    for layer in 0..Q4K_TEST_NUM_LAYERS {
        tensors.insert(
            arch.attn_q_key(layer),
            rand_mat_seeded(q_dim, Q4K_TEST_HIDDEN, 0.05, next_seed()),
        );
        tensors.insert(
            arch.attn_k_key(layer),
            rand_mat_seeded(kv_dim, Q4K_TEST_HIDDEN, 0.05, next_seed()),
        );
        tensors.insert(
            arch.attn_v_key(layer),
            rand_mat_seeded(kv_dim, Q4K_TEST_HIDDEN, 0.05, next_seed()),
        );
        tensors.insert(
            arch.attn_o_key(layer),
            rand_mat_seeded(Q4K_TEST_HIDDEN, q_dim, 0.05, next_seed()),
        );
        tensors.insert(
            arch.ffn_gate_key(layer),
            rand_mat_seeded(Q4K_TEST_INTER, Q4K_TEST_HIDDEN, 0.05, next_seed()),
        );
        tensors.insert(
            arch.ffn_up_key(layer),
            rand_mat_seeded(Q4K_TEST_INTER, Q4K_TEST_HIDDEN, 0.05, next_seed()),
        );
        tensors.insert(
            arch.ffn_down_key(layer),
            rand_mat_seeded(Q4K_TEST_HIDDEN, Q4K_TEST_INTER, 0.05, next_seed()),
        );

        vectors.insert(arch.input_layernorm_key(layer), vec![1.0; Q4K_TEST_HIDDEN]);
        vectors.insert(
            arch.post_attention_layernorm_key(layer),
            vec![1.0; Q4K_TEST_HIDDEN],
        );
    }

    ModelWeights {
        tensors,
        vectors,
        raw_bytes: HashMap::new(),
        packed_mmaps: HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_byte_ranges: HashMap::new(),
        embed,
        lm_head,
        position_embed: None,
        arch,
        num_layers: Q4K_TEST_NUM_LAYERS,
        hidden_size: Q4K_TEST_HIDDEN,
        intermediate_size: Q4K_TEST_INTER,
        vocab_size: Q4K_TEST_VOCAB,
        head_dim,
        num_q_heads: num_q,
        num_kv_heads: num_kv,
        rope_base: 10_000.0,
    }
}

/// Wrap a byte payload in an anonymous read-only mmap. Used to build
/// in-memory test vindexes without touching the filesystem.
///
/// Public so inference-side fixtures (`make_test_q4k_vindex` etc.)
/// that stay in `larql-inference/src/test_utils.rs` can reuse it.
pub fn arc_mmap_from_bytes(payload: &[u8]) -> std::sync::Arc<memmap2::Mmap> {
    let mut anon = memmap2::MmapMut::map_anon(payload.len().max(1)).expect("anon mmap");
    if !payload.is_empty() {
        anon.copy_from_slice(payload);
    }
    let mmap = anon.make_read_only().expect("freeze");
    std::sync::Arc::new(mmap)
}

// ── Gemma 4 hybrid-MoE fixture ──────────────────────────────────────
//
// Minimum Q4_K-aligned hidden / intermediate / expert-intermediate for
// the Gemma 4 hybrid-MoE fixture. Q4_K requires multiples of 256.

/// Hidden dimension for the Gemma 4 MoE test fixture.
pub const GEMMA4_MOE_HIDDEN: usize = 256;
/// Intermediate dimension for the Gemma 4 MoE test fixture.
pub const GEMMA4_MOE_INTER: usize = 256;
/// Expert count for the Gemma 4 MoE test fixture.
pub const GEMMA4_MOE_NUM_EXPERTS: usize = 4;
/// Top-k experts for the Gemma 4 MoE test fixture.
pub const GEMMA4_MOE_TOP_K: usize = 2;

/// Build a synthetic Gemma 4 hybrid-MoE `ModelWeights`.
///
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
/// in `pipeline_layer::build_moe_weights` and the kquant_forward MoE
/// guards execute end-to-end on the substrate side.
pub fn make_test_gemma4_moe_weights() -> ModelWeights {
    let num_q = 4usize;
    let num_kv = 2usize;
    let head_dim = GEMMA4_MOE_HIDDEN / num_q;
    let num_layers = 2usize;

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

        let router_key = arch
            .moe_router_key(layer)
            .expect("Gemma 4 MoE arch must produce a router key");
        let router_proj: Vec<f32> = (0..GEMMA4_MOE_NUM_EXPERTS * hidden)
            .map(|i| ((i as f32) * 0.001).sin() * 0.05)
            .collect();
        vectors.insert(router_key, router_proj);

        // Packed BF16 expert gate_up.
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
        embed,
        lm_head,
        position_embed: None,
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

/// Gemma-3 Q4_K fixture at **caller-chosen attention dimensions**.
///
/// Every other Q4_K fixture is pinned to `hidden = 256, num_q = 4`, i.e.
/// `head_dim = 64`. That made shape sensitivity untestable: a kernel with
/// a `head_dim` assumption would be wrong on every fixture and right on
/// every real model, which is exactly the signature of the Metal
/// batched-prefill divergence (see `larql-kv`'s
/// `gpu_engine_parity::gemma3_prefill_gap`). This builder is the knob for
/// bisecting that axis, and for any future "does this kernel assume a
/// shape?" question.
///
/// `hidden` must be a multiple of 256 (Q4_K super-block) and divisible by
/// `num_q`; `num_q` must be divisible by `num_kv`.
pub fn make_test_q4k_weights_with_dims(
    hidden: usize,
    num_q: usize,
    num_kv: usize,
    num_layers: usize,
) -> ModelWeights {
    assert!(
        hidden.is_multiple_of(256),
        "Q4_K needs a hidden size that is a multiple of its 256-element super-block, got {hidden}"
    );
    assert!(
        num_q != 0 && hidden.is_multiple_of(num_q),
        "hidden {hidden} must divide evenly into {num_q} query heads"
    );
    assert!(
        num_kv != 0 && num_q.is_multiple_of(num_kv),
        "{num_q} query heads must group evenly onto {num_kv} K/V heads"
    );
    let head_dim = hidden / num_q;

    let arch_json = serde_json::json!({
        "model_type": "gemma3_text",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": hidden,
        "head_dim": head_dim,
        "num_attention_heads": num_q,
        "num_key_value_heads": num_kv,
        "vocab_size": Q4K_TEST_VOCAB,
        "hidden_activation": "gelu_pytorch_tanh",
        "rope_theta": 10000.0,
    });
    q4k_test_weights_from_json(arch_json, num_layers, hidden, hidden, num_q, num_kv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_test_weights_basic_shape() {
        let w = make_test_weights();
        assert_eq!(w.hidden_size, 16);
        assert_eq!(w.intermediate_size, 32);
        assert_eq!(w.vocab_size, 32);
        assert_eq!(w.num_layers, 2);
        assert_eq!(w.num_q_heads, 2);
        assert_eq!(w.num_kv_heads, 1);
        assert_eq!(w.head_dim, 8);
    }

    #[test]
    fn make_test_weights_embed_matrix_shape() {
        let w = make_test_weights();
        assert_eq!(w.embed.shape(), &[32, 16]);
        assert_eq!(w.lm_head.shape(), &[32, 16]);
    }

    #[test]
    fn make_test_weights_per_layer_tensors_present() {
        let w = make_test_weights();
        for layer in 0..w.num_layers {
            assert!(
                w.tensors.contains_key(&w.arch.attn_q_key(layer)),
                "missing q-proj for layer {layer}"
            );
            assert!(
                w.tensors.contains_key(&w.arch.attn_k_key(layer)),
                "missing k-proj for layer {layer}"
            );
            assert!(
                w.tensors.contains_key(&w.arch.attn_v_key(layer)),
                "missing v-proj for layer {layer}"
            );
            assert!(
                w.tensors.contains_key(&w.arch.attn_o_key(layer)),
                "missing o-proj for layer {layer}"
            );
            assert!(
                w.tensors.contains_key(&w.arch.ffn_gate_key(layer)),
                "missing ffn-gate for layer {layer}"
            );
            assert!(
                w.tensors.contains_key(&w.arch.ffn_up_key(layer)),
                "missing ffn-up for layer {layer}"
            );
            assert!(
                w.tensors.contains_key(&w.arch.ffn_down_key(layer)),
                "missing ffn-down for layer {layer}"
            );
            assert!(
                w.vectors.contains_key(&w.arch.input_layernorm_key(layer)),
                "missing input-norm for layer {layer}"
            );
            assert!(
                w.vectors
                    .contains_key(&w.arch.post_attention_layernorm_key(layer)),
                "missing post-attn-norm for layer {layer}"
            );
        }
    }

    #[test]
    fn make_test_weights_final_norm_is_ones() {
        let w = make_test_weights();
        let final_norm = w
            .vectors
            .get(w.arch.final_norm_key())
            .expect("final norm missing");
        assert_eq!(final_norm.len(), w.hidden_size);
        assert!(final_norm.iter().all(|v| (*v - 1.0).abs() < 1e-9));
    }

    #[test]
    fn make_test_weights_deterministic_across_calls() {
        // The LCG seed is fixed (0xdeadbeef), so two independent calls
        // must produce identical weight tensors. Pin this so future
        // refactors don't accidentally introduce per-call randomness.
        let a = make_test_weights();
        let b = make_test_weights();
        for layer in 0..a.num_layers {
            let key = a.arch.attn_q_key(layer);
            let ta = a.tensors.get(&key).unwrap();
            let tb = b.tensors.get(&key).unwrap();
            assert_eq!(ta.shape(), tb.shape());
            for (x, y) in ta.iter().zip(tb.iter()) {
                assert!(
                    (x - y).abs() < f32::EPSILON,
                    "non-deterministic tensor at layer {layer}"
                );
            }
        }
    }

    #[test]
    fn make_test_weights_values_in_expected_range() {
        // All weights are LCG-sampled in [-0.1, 0.1]. Pin the magnitude
        // so future scale tweaks are caught.
        let w = make_test_weights();
        for v in w.embed.iter() {
            assert!(
                v.abs() <= 0.1 + 1e-6,
                "embed value outside [-0.1, 0.1]: {v}"
            );
        }
    }
}
