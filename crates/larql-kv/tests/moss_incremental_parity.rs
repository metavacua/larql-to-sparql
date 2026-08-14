//! Funnel step 5 gate, checkpoint half: on the real MOSS-TTS-Realtime
//! model, the token stream from incremental text feeding is identical
//! to batch generation on the same text — for every chunking, in the
//! reference's operating mode (sampled) and in greedy.
//!
//! The CI half of this gate (`larql-inference/tests/test_moss_session.rs`)
//! pins the session mechanics on a synthetic model; this test closes the
//! same equality with the release checkpoint, tokenizer and prompt
//! construction included: `build_prompt` (batch) versus `build_turn` +
//! a [`MossSession`] fed in chunks.
//!
//! NOT FOR CI. Run:
//!
//! ```text
//! MOSS_TTS_REALTIME_DIR=<hf snapshot dir> \
//! cargo test --release -p larql-kv --test moss_incremental_parity -- --ignored --nocapture
//! ```

mod common;

use common::model_dir_from_env;
use larql_inference::ffn::WeightFfn;
use larql_inference::larql_models::loading::safetensors::load_model_dir;
use larql_inference::larql_models::speech::moss_tts_realtime::{
    depth_transformer_model, load_moss_tts_aux_from_safetensors, MossTtsRealtimeConfig,
};
use larql_inference::speech::moss_prompt::{build_prompt, build_turn};
use larql_inference::speech::moss_realtime::{generate_frames_streaming, MossGeneration};
use larql_inference::speech::moss_sampling::{DecodeMode, MossSampling};
use larql_inference::speech::moss_session::{MossSession, MossSpeech};
use larql_inference::tokenizer::load_tokenizer;

/// The novel fixture text: long enough that the queue outlives the
/// 12-token lead by a wide margin.
const TEXT: &str = "The quick brown fox jumps over the lazy dog while the \
                    orchestra tunes quietly in the next room.";
/// Bounds the greedy run (greedy does not terminate on novel text) and
/// backstops the sampled runs.
const MAX_FRAMES: usize = 150;
/// Seed shared by every compared run.
const SEED: u64 = 0;

struct LoadedModel {
    weights: larql_inference::larql_models::ModelWeights,
    audio_tables: Vec<ndarray::Array2<f32>>,
    depth: larql_inference::larql_models::speech::moss_tts_realtime::DepthTransformerModel,
    config: MossTtsRealtimeConfig,
    tokenizer: larql_inference::tokenizers::Tokenizer,
}

fn load() -> LoadedModel {
    let model_dir = model_dir_from_env();
    let weights = load_model_dir(&model_dir).expect("backbone load");
    let config_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(std::path::Path::new(&model_dir).join("config.json")).unwrap(),
    )
    .unwrap();
    let config = MossTtsRealtimeConfig::from_config_json(&config_json).unwrap();
    let aux = load_moss_tts_aux_from_safetensors(&model_dir, weights.arch.as_ref(), config.clone())
        .expect("aux load");
    let depth = depth_transformer_model(aux.local, &config).expect("depth adapter");
    let tokenizer = load_tokenizer(std::path::Path::new(&model_dir)).expect("tokenizer");
    LoadedModel {
        weights,
        audio_tables: aux.audio_embed_tables,
        depth,
        config,
        tokenizer,
    }
}

fn run_batch(model: &LoadedModel, mode: DecodeMode) -> MossGeneration {
    let prompt = build_prompt(&model.tokenizer, &model.config, None, TEXT).expect("prompt");
    let backend = larql_inference::cpu_engine_backend();
    let ffn = WeightFfn {
        weights: &model.weights,
    };
    let depth_ffn = WeightFfn {
        weights: &model.depth.weights,
    };
    generate_frames_streaming(
        backend.as_ref(),
        &model.weights,
        &ffn,
        &model.audio_tables,
        &model.depth,
        &depth_ffn,
        &model.config,
        &prompt.prefill_matrix,
        &prompt.text_queue,
        prompt.text_pad_id,
        MAX_FRAMES,
        mode,
        SEED,
        None,
        None,
        |_, _| {},
    )
    .expect("batch generation")
}

fn run_incremental(model: &LoadedModel, mode: DecodeMode, chunk: usize) -> MossGeneration {
    let turn = build_turn(&model.tokenizer, &model.config, None).expect("turn");
    let text_ids: Vec<u32> = model
        .tokenizer
        .encode(TEXT, false)
        .expect("encode")
        .get_ids()
        .to_vec();
    let backend = larql_inference::cpu_engine_backend();
    let ffn = WeightFfn {
        weights: &model.weights,
    };
    let depth_ffn = WeightFfn {
        weights: &model.depth.weights,
    };
    let speech = MossSpeech {
        backend: backend.as_ref(),
        backbone: &model.weights,
        backbone_ffn: &ffn,
        audio_tables: &model.audio_tables,
        depth: &model.depth,
        depth_ffn: &depth_ffn,
        config: &model.config,
        backbone_attn: None,
        depth_attn: None,
    };
    let mut session = MossSession::new(
        speech,
        turn.matrix,
        turn.text_pad_id,
        MAX_FRAMES,
        mode,
        SEED,
    );
    let mut sink = |_: usize, _: &[u32]| {};
    for piece in text_ids.chunks(chunk) {
        session.push_text_ids(piece, &mut sink).expect("push");
    }
    session.end_text(&mut sink).expect("end_text");
    session.finish(&mut sink).expect("finish");
    session.into_generation()
}

fn assert_same(batch: &MossGeneration, incremental: &MossGeneration, label: &str) {
    assert_eq!(
        batch.frames.len(),
        incremental.frames.len(),
        "{label}: frame count"
    );
    let mut mismatches = 0usize;
    for (index, (batch_frame, incremental_frame)) in batch
        .frames
        .iter()
        .zip(incremental.frames.iter())
        .enumerate()
    {
        if batch_frame != incremental_frame {
            mismatches += 1;
            if mismatches <= 5 {
                eprintln!(
                    "{label}: frame {index} differs: {batch_frame:?} vs {incremental_frame:?}"
                );
            }
        }
    }
    assert_eq!(mismatches, 0, "{label}: frames must match exactly");
    assert_eq!(batch.eos_at, incremental.eos_at, "{label}: EOS position");
}

#[test]
#[ignore]
fn moss_incremental_stream_matches_batch() {
    let model = load();

    // The reference's operating mode first: sampled, chunk sizes from
    // token-by-token up to all-at-once.
    let sampled = DecodeMode::Sampled(MossSampling::default());
    let batch = run_batch(&model, sampled);
    println!(
        "batch (sampled, seed {SEED}): {} frames, eos={:?}",
        batch.frames.len(),
        batch.eos_at
    );
    assert!(
        batch.eos_at.is_some(),
        "sampled batch generation should terminate on the fixture text"
    );
    for chunk in [1usize, 5, usize::MAX] {
        let effective = chunk.min(512);
        let incremental = run_incremental(&model, sampled, effective);
        assert_same(&batch, &incremental, &format!("sampled chunk {effective}"));
        println!(
            "incremental chunk {effective}: identical ({} frames)",
            incremental.frames.len()
        );
    }

    // Greedy, capped: exercises the non-sampled arm on the same session.
    let greedy_batch = run_batch(&model, DecodeMode::Greedy);
    let greedy_incremental = run_incremental(&model, DecodeMode::Greedy, 7);
    assert_same(&greedy_batch, &greedy_incremental, "greedy chunk 7");
    println!(
        "greedy chunk 7: identical ({} frames, capped at {MAX_FRAMES})",
        greedy_incremental.frames.len()
    );
}
