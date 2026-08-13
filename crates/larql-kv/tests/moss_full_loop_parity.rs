//! Funnel step 4 (`docs/tts-funnel.md` §3): the full generation loop.
//!
//! Steps 2–3 verified the pieces with teacher forcing. This is the
//! chained, self-generated run: LARQL prefls the reference's exact
//! prompt, then alternates backbone `decode_step_from_hidden` and cached
//! depth micro-steps on its OWN hiddens and its OWN sampled codebooks,
//! consuming the same text queue, stopping on its own audio-EOS
//! detection. No tokenizer anywhere; ids in, ids out.
//!
//! Gate: the complete greedy frame sequence — every codebook of every
//! frame, the EOS position, and the emitted (EOS-cropped) sequence —
//! is identical to the reference dump's. Any single divergent codebook
//! anywhere in the 138-frame chain would cascade, so exact equality here
//! closes the whole execution chain end-to-end.
//!
//! NOT FOR CI. Run:
//!
//! ```text
//! MOSS_TTS_REALTIME_DIR=<hf snapshot dir> \
//! MOSS_PARITY_BIN_DIR=<parity-dump/bin dir> \
//! cargo test --release -p larql-kv --test moss_full_loop_parity -- --ignored --nocapture
//! ```

mod common;

use common::{model_dir_from_env, BinFixtures};
use larql_inference::ffn::WeightFfn;
use larql_inference::larql_models::loading::safetensors::load_model_dir;
use larql_inference::larql_models::speech::moss_tts_realtime::{
    depth_transformer_model, load_moss_tts_aux_from_safetensors, MossTtsRealtimeConfig,
};
use larql_inference::speech::moss_realtime::generate_frames_greedy;

/// The reference consumed this many text ids during prefill
/// (`delay_tokens_len`); the queue starts after them. Derived from the
/// dump rather than assumed: prefill_ids' text column beyond the prompt
/// matrix carries exactly the lead.
fn text_lead_len(prompt_rows: usize, prefill_rows: usize) -> usize {
    prefill_rows - prompt_rows
}

#[test]
#[ignore]
fn moss_full_greedy_loop_matches_reference_dump() {
    let model_dir = model_dir_from_env();
    let fixtures = BinFixtures::open_from_env();

    let weights = load_model_dir(&model_dir).expect("backbone load");
    let config_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(std::path::Path::new(&model_dir).join("config.json")).unwrap(),
    )
    .unwrap();
    let config = MossTtsRealtimeConfig::from_config_json(&config_json).unwrap();
    let aux = load_moss_tts_aux_from_safetensors(&model_dir, weights.arch.as_ref(), config.clone())
        .expect("aux load");
    let audio_tables = aux.audio_embed_tables;
    let depth = depth_transformer_model(aux.local, &config).expect("depth adapter");

    let prefill_ids = fixtures.i64_matrix_as_u32("prefill_ids");
    let prompt_matrix = fixtures.i64_matrix_as_u32("prompt_matrix");
    let reference_frames = fixtures.i64_matrix_as_u32("frames");
    let reference_emitted = fixtures.i64_matrix_as_u32("emitted_tokens");
    let lead = text_lead_len(prompt_matrix.nrows(), prefill_ids.nrows());

    // text_ids is rank-1 in the dump; read via shape-aware path.
    let text_ids_shape = fixtures.shape("text_ids").to_vec();
    assert_eq!(text_ids_shape.len(), 1);
    let text_ids: Vec<u32> = {
        let bytes = std::fs::read(
            std::path::Path::new(&std::env::var("MOSS_PARITY_BIN_DIR").unwrap())
                .join("text_ids.bin"),
        )
        .unwrap();
        bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as u32)
            .collect()
    };
    let text_queue = &text_ids[lead..];

    // The pad id the reference actually fed once text ran out: read off
    // the dump's own step ids rather than assumed.
    let step_ids = fixtures.i64_matrix_as_u32("step_ids");
    let text_pad_id = step_ids[[step_ids.nrows() - 1, 0]];

    let backend = larql_inference::cpu_engine_backend();
    let ffn = WeightFfn { weights: &weights };
    let depth_ffn = WeightFfn {
        weights: &depth.weights,
    };
    let generation = generate_frames_greedy(
        backend.as_ref(),
        &weights,
        &ffn,
        &audio_tables,
        &depth,
        &depth_ffn,
        &config,
        &prefill_ids,
        text_queue,
        text_pad_id,
        reference_frames.nrows() + 8,
    )
    .expect("generation");

    let frames_total = reference_frames.nrows();
    assert_eq!(
        generation.frames.len(),
        frames_total,
        "frame count mismatch: LARQL {} vs reference {frames_total}",
        generation.frames.len()
    );
    let mut mismatches = 0usize;
    for (frame_index, frame) in generation.frames.iter().enumerate() {
        for (codebook, &code) in frame.iter().enumerate() {
            if code != reference_frames[[frame_index, codebook]] {
                mismatches += 1;
                if mismatches <= 5 {
                    eprintln!(
                        "frame {frame_index} codebook {codebook}: {code} != {}",
                        reference_frames[[frame_index, codebook]]
                    );
                }
            }
        }
    }
    assert_eq!(mismatches, 0, "self-generated frames must match exactly");
    assert_eq!(
        generation.eos_at,
        Some(frames_total - 1),
        "EOS must land on the reference's final frame"
    );
    assert_eq!(
        generation.emitted().len(),
        reference_emitted.nrows(),
        "emitted (EOS-cropped) length must match the reference crop"
    );
    println!(
        "full-loop parity: {frames_total} frames self-generated identically; \
         EOS at {frames_total}-1; emitted {} frames",
        generation.emitted().len()
    );
}
