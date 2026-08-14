//! Funnel step 5 gate, CI half: incremental push-text generation is
//! identical to batch generation on the same text — for every chunking.
//!
//! Runs on the synthetic miniature speech model
//! ([`larql_inference::speech::test_support`]), so the *session
//! mechanics* (pending queue, prefill trigger at the text lead, step
//! order, RNG draw order, pad drain) are pinned in CI. The same
//! equality against the real checkpoint is the ignored
//! `moss_incremental_parity` test in `larql-kv`.

use larql_inference::ffn::WeightFfn;
use larql_inference::speech::moss_prompt::{append_text_lead, DELAY_TOKENS_LEN};
use larql_inference::speech::moss_realtime::{generate_frames_streaming, MossGeneration};
use larql_inference::speech::moss_sampling::{DecodeMode, MossSampling};
use larql_inference::speech::moss_session::{MossSession, MossSpeech};
use larql_inference::speech::test_support::{
    synth_speech_model, synth_turn_matrix, SynthSpeechModel, SYNTH_TEXT_PAD,
};

/// Rows in the synthetic turn matrix (stands in for system + assistant
/// header rows).
const TURN_ROWS: usize = 4;
/// Frame cap for every run: bounds the non-terminating cases and keeps
/// the test fast.
const MAX_FRAMES: usize = 12;
/// RNG seed shared by every compared run.
const SEED: u64 = 7;

/// A deterministic in-vocabulary text id sequence.
fn text_ids(count: usize) -> Vec<u32> {
    (0..count as u32)
        .map(|index| (index * 5 + 2) % 29)
        .collect()
}

/// Batch reference: the prompt path `build_prompt` takes, spelled on the
/// synthetic turn matrix — lead into the prefill, remainder queued.
fn run_batch(model: &SynthSpeechModel, ids: &[u32], mode: DecodeMode) -> MossGeneration {
    let backend = larql_inference::cpu_engine_backend();
    let backbone_ffn = WeightFfn {
        weights: &model.backbone,
    };
    let depth_ffn = WeightFfn {
        weights: &model.depth.weights,
    };
    let turn = synth_turn_matrix(TURN_ROWS, &model.config);
    let lead = ids.len().min(DELAY_TOKENS_LEN);
    let prefill = append_text_lead(&turn, &ids[..lead], &model.config);
    generate_frames_streaming(
        backend.as_ref(),
        &model.backbone,
        &backbone_ffn,
        &model.audio_tables,
        &model.depth,
        &depth_ffn,
        &model.config,
        &prefill,
        &ids[lead..],
        SYNTH_TEXT_PAD,
        MAX_FRAMES,
        mode,
        SEED,
        None,
        None,
        |_, _| {},
    )
    .expect("batch generation")
}

/// Incremental: the same text pushed in `chunk`-sized pieces through a
/// session, then ended and drained.
fn run_incremental(
    model: &SynthSpeechModel,
    ids: &[u32],
    mode: DecodeMode,
    chunk: usize,
) -> MossGeneration {
    let backend = larql_inference::cpu_engine_backend();
    let backbone_ffn = WeightFfn {
        weights: &model.backbone,
    };
    let depth_ffn = WeightFfn {
        weights: &model.depth.weights,
    };
    let speech = MossSpeech {
        backend: backend.as_ref(),
        backbone: &model.backbone,
        backbone_ffn: &backbone_ffn,
        audio_tables: &model.audio_tables,
        depth: &model.depth,
        depth_ffn: &depth_ffn,
        config: &model.config,
        backbone_attn: None,
        depth_attn: None,
    };
    let turn = synth_turn_matrix(TURN_ROWS, &model.config);
    let mut session = MossSession::new(speech, turn, SYNTH_TEXT_PAD, MAX_FRAMES, mode, SEED);
    let mut observed: Vec<(usize, Vec<u32>)> = Vec::new();
    let mut observer = |index: usize, codes: &[u32]| observed.push((index, codes.to_vec()));
    for piece in ids.chunks(chunk.max(1)) {
        session.push_text_ids(piece, &mut observer).expect("push");
    }
    session.end_text(&mut observer).expect("end_text");
    session.finish(&mut observer).expect("finish");

    // The callback saw every frame, in order, exactly once.
    let frames: Vec<Vec<u32>> = session.frames().to_vec();
    let seen: Vec<Vec<u32>> = observed.iter().map(|(_, codes)| codes.clone()).collect();
    assert_eq!(seen, frames, "observer must see the frames as stored");
    for (position, (index, _)) in observed.iter().enumerate() {
        assert_eq!(*index, position, "frame indexes must be sequential");
    }
    session.into_generation()
}

fn assert_same(batch: &MossGeneration, incremental: &MossGeneration, label: &str) {
    assert_eq!(batch.frames, incremental.frames, "{label}: frames differ");
    assert_eq!(
        batch.eos_at, incremental.eos_at,
        "{label}: EOS position differs"
    );
}

#[test]
fn incremental_equals_batch_across_chunkings_sampled() {
    let model = synth_speech_model();
    let ids = text_ids(20);
    let mode = DecodeMode::Sampled(MossSampling::default());
    let batch = run_batch(&model, &ids, mode);
    assert!(
        batch.frames.len() > 3,
        "the fixture must generate enough frames to exercise stepping"
    );
    for chunk in [1, 3, ids.len()] {
        let incremental = run_incremental(&model, &ids, mode, chunk);
        assert_same(&batch, &incremental, &format!("sampled, chunk {chunk}"));
    }
}

#[test]
fn incremental_equals_batch_greedy() {
    let model = synth_speech_model();
    let ids = text_ids(20);
    let batch = run_batch(&model, &ids, DecodeMode::Greedy);
    let incremental = run_incremental(&model, &ids, DecodeMode::Greedy, 4);
    assert_same(&batch, &incremental, "greedy, chunk 4");
}

#[test]
fn short_text_prefills_on_end_text() {
    // Text shorter than the lead: nothing may generate until end_text
    // declares the stream over, and the result still matches batch.
    let model = synth_speech_model();
    let ids = text_ids(5);
    let mode = DecodeMode::Sampled(MossSampling::default());
    let batch = run_batch(&model, &ids, mode);

    let backend = larql_inference::cpu_engine_backend();
    let backbone_ffn = WeightFfn {
        weights: &model.backbone,
    };
    let depth_ffn = WeightFfn {
        weights: &model.depth.weights,
    };
    let speech = MossSpeech {
        backend: backend.as_ref(),
        backbone: &model.backbone,
        backbone_ffn: &backbone_ffn,
        audio_tables: &model.audio_tables,
        depth: &model.depth,
        depth_ffn: &depth_ffn,
        config: &model.config,
        backbone_attn: None,
        depth_attn: None,
    };
    let turn = synth_turn_matrix(TURN_ROWS, &model.config);
    let mut session = MossSession::new(speech, turn, SYNTH_TEXT_PAD, MAX_FRAMES, mode, SEED);
    let mut sink = |_: usize, _: &[u32]| {};
    session.push_text_ids(&ids[..2], &mut sink).expect("push");
    session.push_text_ids(&ids[2..], &mut sink).expect("push");
    assert!(
        session.frames().is_empty(),
        "below the text lead, nothing generates until the text ends"
    );
    session.end_text(&mut sink).expect("end_text");
    assert!(
        !session.frames().is_empty(),
        "end_text must fire the prefill with the short lead"
    );
    session.finish(&mut sink).expect("finish");
    assert_same(&batch, &session.into_generation(), "short text");
}

#[test]
#[should_panic(expected = "finish before end_text")]
fn finish_requires_end_text() {
    let model = synth_speech_model();
    let backend = larql_inference::cpu_engine_backend();
    let backbone_ffn = WeightFfn {
        weights: &model.backbone,
    };
    let depth_ffn = WeightFfn {
        weights: &model.depth.weights,
    };
    let speech = MossSpeech {
        backend: backend.as_ref(),
        backbone: &model.backbone,
        backbone_ffn: &backbone_ffn,
        audio_tables: &model.audio_tables,
        depth: &model.depth,
        depth_ffn: &depth_ffn,
        config: &model.config,
        backbone_attn: None,
        depth_attn: None,
    };
    let turn = synth_turn_matrix(TURN_ROWS, &model.config);
    let mut session = MossSession::new(
        speech,
        turn,
        SYNTH_TEXT_PAD,
        MAX_FRAMES,
        DecodeMode::Greedy,
        SEED,
    );
    let mut sink = |_: usize, _: &[u32]| {};
    let _ = session.finish(&mut sink);
}
