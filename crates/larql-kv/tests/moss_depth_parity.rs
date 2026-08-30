//! Funnel step 3 (`docs/tts-funnel.md` §3): depth-transformer parity.
//!
//! The nested generation step — backbone hidden in, sixteen per-codebook
//! logit vectors out — executed with NO new runtime machinery: the depth
//! transformer is re-expressed as an ordinary 4-layer Qwen3
//! (`depth_transformer_model`) and each micro-step prefix runs through
//! the same `prefill_from_hidden` the backbone used in step 2. Causal
//! attention makes the incremental prefix forward mathematically
//! identical to the reference's cached micro-step decode.
//!
//! Teacher-forced from the step-0 dump: micro-step inputs come from the
//! dump's backbone hiddens and sampled frames, so this test isolates the
//! depth transformer alone. Two gates per micro-step, over every frame:
//! logits within tolerance, and greedy argmax equal to the reference's
//! sampled codebook.
//!
//! NOT FOR CI: needs the checkpoint and the step-0 dump. Run:
//!
//! ```text
//! MOSS_TTS_REALTIME_DIR=<hf snapshot dir> \
//! MOSS_PARITY_BIN_DIR=<parity-dump/bin dir> \
//! cargo test --release -p larql-kv --test moss_depth_parity -- --ignored --nocapture
//! ```

mod common;

use ndarray::Array2;

use common::{max_abs_diff, model_dir_from_env, BinFixtures};
use larql_compute::forward::ops::apply_norm;
use larql_inference::ffn::WeightFfn;
use larql_inference::larql_models::speech::moss_tts_realtime::{
    depth_transformer_model, load_moss_tts_aux_from_safetensors, MossTtsRealtimeConfig,
};
use larql_inference::KvEngine;
use larql_kv::engines::standard::StandardEngine;

/// First-gate tolerance — exists to be tightened, not relaxed.
const LOGIT_MAX_ABS: f32 = 5e-3;

#[test]
#[ignore]
fn moss_depth_transformer_matches_reference_logits() {
    let model_dir = model_dir_from_env();
    let fixtures = BinFixtures::open_from_env();

    // ── The inner model, loaded and re-expressed as a Qwen3 ──
    let config_json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(std::path::Path::new(&model_dir).join("config.json")).unwrap(),
    )
    .unwrap();
    let config = MossTtsRealtimeConfig::from_config_json(&config_json).unwrap();
    let arch = larql_inference::larql_models::detect_from_json(&config_json);
    let aux =
        load_moss_tts_aux_from_safetensors(&model_dir, arch.as_ref(), config.clone()).unwrap();
    let depth = depth_transformer_model(aux.local, &config).unwrap();
    let weights = &depth.weights;
    let ffn = WeightFfn { weights };

    // ── Teacher-forcing material from the dump ──
    let prefill_hidden = fixtures.f32_matrix("prefill_hidden");
    let step_hiddens = fixtures.f32_matrix("step_hiddens");
    let frames = fixtures.i64_matrix_as_u32("frames");
    let reference_logits = fixtures.f32_rank3("local_logits");
    let frames_total = frames.nrows();
    assert_eq!(reference_logits.len(), frames_total);
    assert_eq!(step_hiddens.nrows(), frames_total - 1);
    let rvq = config.rvq;
    let hidden_size = config.local.hidden_size;

    let mut worst_logit_diff = 0.0f32;
    let mut argmax_mismatches = 0usize;

    for frame in 0..frames_total {
        // Micro-step 0's input: the backbone hidden that produced this
        // frame (prefill's last row for frame 0, then per-step hiddens).
        let backbone_hidden = if frame == 0 {
            prefill_hidden.row(prefill_hidden.nrows() - 1)
        } else {
            step_hiddens.row(frame - 1)
        };

        // The full teacher-forced depth sequence for this frame:
        // position 0 = backbone hidden, position i = embedding of the
        // reference's sampled codebook i-1 through table i-1.
        let mut sequence = Array2::<f32>::zeros((rvq, hidden_size));
        sequence.row_mut(0).assign(&backbone_hidden);
        for position in 1..rvq {
            let code = frames[[frame, position - 1]] as usize;
            sequence
                .row_mut(position)
                .assign(&depth.embed_tables[position - 1].row(code));
        }

        for micro in 0..rvq {
            // Fresh engine per prefix: the reference rebuilds the local
            // cache every frame, and causal attention makes the prefix
            // forward equal to its cached decode.
            let prefix = sequence.slice(ndarray::s![0..=micro, ..]).to_owned();
            let mut engine = StandardEngine::new(None);
            let last = engine
                .prefill_from_hidden(weights, &ffn, &prefix)
                .expect("depth prefill");
            let normed = apply_norm(
                weights,
                &last,
                weights.arch.final_norm_key(),
                weights.arch.norm_weight_offset(),
            );
            let logits = normed.dot(&depth.lm_heads[micro].t());

            let reference = {
                let mut row = Array2::<f32>::zeros((1, logits.ncols()));
                row.row_mut(0).assign(&reference_logits[frame].row(micro));
                row
            };
            let diff = max_abs_diff(&logits, &reference);
            worst_logit_diff = worst_logit_diff.max(diff);

            let ours = argmax(&logits);
            let theirs = frames[[frame, micro]] as usize;
            if ours != theirs {
                argmax_mismatches += 1;
                eprintln!(
                    "frame {frame} micro {micro}: argmax {ours} != reference {theirs} \
                     (logit max |Δ| {diff:e})"
                );
            }
        }
    }

    println!(
        "depth parity: {frames_total} frames × {rvq} micro-steps — \
         worst logit |Δ| = {worst_logit_diff:e}, argmax mismatches = {argmax_mismatches}"
    );
    assert_eq!(argmax_mismatches, 0, "greedy codebook choices must match");
    assert!(
        worst_logit_diff <= LOGIT_MAX_ABS,
        "logit divergence: {worst_logit_diff:e} > {LOGIT_MAX_ABS:e}"
    );
}

fn argmax(logits: &Array2<f32>) -> usize {
    logits
        .row(0)
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}
