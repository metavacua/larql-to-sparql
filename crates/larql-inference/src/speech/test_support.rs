//! A synthetic miniature speech model — the MOSS execution shape
//! (backbone + summed audio tables + depth transformer + per-codebook
//! heads) at toy dimensions, for exercising the generation machinery in
//! CI without a checkpoint.
//!
//! Follows the crate's [`crate::test_utils`] precedent: an always-compiled
//! support module so integration tests under `tests/` can use it. Every
//! dimension is read off the built weights or spelled as a named
//! constant here — nothing is tied to the release checkpoint.

use ndarray::Array2;

use larql_models::speech::moss_tts_realtime::{
    DepthTransformerModel, LocalTransformerConfig, MossTtsRealtimeConfig,
};
use larql_models::test_fixtures::make_test_weights;
use larql_models::ModelWeights;

/// Codebooks per frame in the synthetic model.
pub const SYNTH_RVQ: usize = 3;
/// Audio-side vocabulary per codebook, specials included.
pub const SYNTH_AUDIO_VOCAB: usize = 16;
/// The audio pad id; BOS and EOS are derived from it by
/// [`MossTtsRealtimeConfig`] exactly as on the real checkpoint.
pub const SYNTH_AUDIO_PAD: usize = 10;
/// The text id fed once the text stream is exhausted. Any in-vocabulary
/// id works; the value only has to be consistent across compared runs.
pub const SYNTH_TEXT_PAD: u32 = 1;

/// The full synthetic bundle, ready for a
/// [`crate::speech::moss_session::MossSpeech`] borrow.
pub struct SynthSpeechModel {
    pub backbone: ModelWeights,
    pub depth: DepthTransformerModel,
    pub audio_tables: Vec<Array2<f32>>,
    pub config: MossTtsRealtimeConfig,
}

/// Deterministic filler: the same LCG family the crate's other synthetic
/// fixtures use, seeded per tensor so tables differ from each other.
fn filled(rows: usize, cols: usize, seed: u64, scale: f32) -> Array2<f32> {
    let mut state = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 32) as u32 as f32 / u32::MAX as f32 * 2.0 * scale - scale
    })
}

/// Build the synthetic speech model. Deterministic: two calls produce
/// identical weights.
pub fn synth_speech_model() -> SynthSpeechModel {
    let backbone = make_test_weights();
    let depth_weights = make_test_weights();
    let hidden = backbone.hidden_size;
    assert_eq!(
        depth_weights.hidden_size, hidden,
        "depth hidden size must match the backbone (no projection exists between them)"
    );

    let audio_tables: Vec<Array2<f32>> = (0..SYNTH_RVQ)
        .map(|table| filled(SYNTH_AUDIO_VOCAB, hidden, 0x1000 + table as u64, 0.05))
        .collect();
    let embed_tables: Vec<Array2<f32>> = (0..SYNTH_RVQ - 1)
        .map(|table| filled(SYNTH_AUDIO_VOCAB, hidden, 0x2000 + table as u64, 0.05))
        .collect();
    let lm_heads: Vec<Array2<f32>> = (0..SYNTH_RVQ)
        .map(|head| filled(SYNTH_AUDIO_VOCAB, hidden, 0x3000 + head as u64, 0.5))
        .collect();

    let local = LocalTransformerConfig {
        hidden_size: hidden,
        num_layers: depth_weights.num_layers,
        num_q_heads: depth_weights.num_q_heads,
        num_kv_heads: depth_weights.num_kv_heads,
        head_dim: depth_weights.head_dim,
        intermediate_size: depth_weights.intermediate_size,
        norm_eps: 1e-6,
        rope_base: depth_weights.rope_base,
        max_position_embeddings: SYNTH_RVQ + 1,
    };
    let config = MossTtsRealtimeConfig {
        rvq: SYNTH_RVQ,
        audio_vocab_size: SYNTH_AUDIO_VOCAB,
        audio_pad_token: SYNTH_AUDIO_PAD,
        local,
    };

    SynthSpeechModel {
        backbone,
        depth: DepthTransformerModel {
            weights: depth_weights,
            embed_tables,
            lm_heads,
        },
        audio_tables,
        config,
    }
}

/// A turn matrix at the synthetic model's shape: `rows` text ids on
/// column 0, every audio channel at pad — the structural minimum the
/// prompt contract requires.
pub fn synth_turn_matrix(rows: usize, config: &MossTtsRealtimeConfig) -> Array2<u32> {
    let mut matrix =
        Array2::<u32>::from_elem((rows, 1 + config.rvq), config.audio_pad_token as u32);
    for row in 0..rows {
        matrix[[row, 0]] = (row % 8) as u32;
    }
    matrix
}

/// A word-level tokenizer carrying the MOSS protocol specials
/// (`<|audio_pad|>`, `<|text_pad|>`) as added tokens — enough for the
/// prompt builder's structural contract (splice rows, lead placement,
/// single-id specials) to be exercised without the release tokenizer.
/// Unknown words collapse to `[UNK]`, which the prompt structure does
/// not care about.
pub fn synth_moss_tokenizer() -> tokenizers::Tokenizer {
    let words = ["hello", "world", "one", "two", "three", "four", "five"];
    let mut vocab = serde_json::Map::new();
    for (id, word) in words.iter().enumerate() {
        vocab.insert((*word).to_string(), serde_json::Value::Number(id.into()));
    }
    let unk_id = words.len() as u64;
    vocab.insert("[UNK]".into(), serde_json::Value::Number(unk_id.into()));
    let specials = ["<|audio_pad|>", "<|text_pad|>"];
    let added_tokens: Vec<serde_json::Value> = specials
        .iter()
        .enumerate()
        .map(|(offset, content)| {
            serde_json::json!({
                "id": unk_id + 1 + offset as u64,
                "content": content,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            })
        })
        .collect();
    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "[UNK]"
        }
    });
    let bytes = serde_json::to_vec(&tokenizer_json).expect("tokenizer json");
    tokenizers::Tokenizer::from_bytes(&bytes).expect("synthetic moss tokenizer")
}
