//! MOSS-TTS-Realtime generation loop — text ids in, audio frames out.
//!
//! One outer step per 80 ms acoustic frame (`docs/tts-funnel.md` §1.3):
//!
//! ```text
//! [text id | previous frame's rvq ids]   (one row, 1+rvq columns)
//!         │  summed through 1+rvq embedding tables
//!         ▼
//! backbone decode_step_from_hidden (persistent KV)
//!         ▼  final-normed hidden
//! depth transformer: cached micro-steps over its own fresh KV
//!   position 0 = the hidden row, then one sampled codebook per position,
//!   head i read at position i
//!         ▼
//! frame [rvq ids]; codebook 0 == audio EOS stops the loop
//! ```
//!
//! The depth stage runs the dispatch helpers directly with a per-frame
//! handle set — the cache lives exactly one frame, matching the
//! reference's rebuilt `StaticCache` — and is *cached*, not
//! prefix-recomputed: micro-step `i` appends one position.
//!
//! No tokenizer exists anywhere in this loop. Text ids arrive from the
//! caller; audio ids leave to the caller. The batch entry points below
//! are wrappers over [`MossSession`] — one loop implementation serves
//! batch and incremental, so the step-5 "incremental equals batch" gate
//! holds by construction.
//!
//! [`MossSession`]: crate::speech::moss_session::MossSession

use ndarray::Array2;

use larql_compute::forward::ops::apply_norm;
use larql_models::speech::moss_tts_realtime::{DepthTransformerModel, MossTtsRealtimeConfig};
use larql_models::ModelWeights;

use rand::rngs::StdRng;

use crate::ffn::FfnBackend;
use crate::kv_dispatch::helpers::{
    kv_decode_step_from_hidden_via_dispatch, kv_prefill_from_hidden_via_dispatch,
};
use crate::kv_engine::EngineError;
use crate::speech::moss_sampling::{
    apply_repetition_penalty, filter_logits, sample_multinomial, DecodeMode,
};
use crate::speech::moss_session::{MossSession, MossSpeech};
use crate::EngineBackend;
use larql_compute::KvIndex;

/// Where one frame's compute went — the step-5 performance surface.
#[derive(Debug, Clone, Copy)]
pub struct FrameStages {
    /// The backbone decode step (absent for frame 0, which rides the
    /// prefill).
    pub backbone_seconds: f64,
    /// The 16 depth micro-steps + heads.
    pub depth_seconds: f64,
}

/// A completed (or frame-capped) generation.
#[derive(Debug)]
pub struct MossGeneration {
    /// Every generated frame, EOS frame included, `rvq` ids each.
    pub frames: Vec<Vec<u32>>,
    /// Index of the frame whose codebook 0 was audio EOS, if reached.
    pub eos_at: Option<usize>,
    /// Seconds spent in the prefill forward.
    pub prefill_seconds: f64,
    /// Per-frame stage split, one entry per frame.
    pub stage_timings: Vec<FrameStages>,
}

impl MossGeneration {
    /// Frames up to (excluding) the EOS frame — what a codec should
    /// decode, matching the reference's crop.
    pub fn emitted(&self) -> &[Vec<u32>] {
        match self.eos_at {
            Some(index) => &self.frames[..index],
            None => &self.frames,
        }
    }
}

/// Generate audio frames greedily until audio EOS or `max_frames`.
///
/// `prefill_matrix` is the `[T, 1+rvq]` prompt (system + voice-clone
/// splice + the text lead with audio BOS placed by the caller — temporal
/// protocol belongs to the caller, not this loop). `audio_tables` are the
/// backbone's per-codebook input tables
/// (`MossTtsRealtimeAuxWeights::audio_embed_tables`), summed with the
/// text embedding per position. `text_queue` holds the not-yet-consumed
/// text ids; once exhausted, `text_pad_id` is fed, per the reference.
/// The backbone `engine` must support multimodal decode
/// (`supports_multimodal`).
#[allow(clippy::too_many_arguments)]
pub fn generate_frames_greedy(
    backend: &dyn EngineBackend,
    backbone: &ModelWeights,
    backbone_ffn: &dyn FfnBackend,
    audio_tables: &[Array2<f32>],
    depth: &DepthTransformerModel,
    depth_ffn: &dyn FfnBackend,
    config: &MossTtsRealtimeConfig,
    prefill_matrix: &Array2<u32>,
    text_queue: &[u32],
    text_pad_id: u32,
    max_frames: usize,
) -> Result<MossGeneration, EngineError> {
    generate_frames_greedy_streaming(
        backend,
        backbone,
        backbone_ffn,
        audio_tables,
        depth,
        depth_ffn,
        config,
        prefill_matrix,
        text_queue,
        text_pad_id,
        max_frames,
        |_, _| {},
    )
}

/// [`generate_frames_greedy`] with a per-frame observer: `on_frame(index,
/// codes)` fires the moment each frame exists, before the next backbone
/// step — the streaming surface. Frames are the model's realtime unit
/// (80 ms each at 12.5 Hz), so a consumer feeding a codec + ring buffer
/// hangs directly off this callback.
#[allow(clippy::too_many_arguments)]
pub fn generate_frames_greedy_streaming(
    backend: &dyn EngineBackend,
    backbone: &ModelWeights,
    backbone_ffn: &dyn FfnBackend,
    audio_tables: &[Array2<f32>],
    depth: &DepthTransformerModel,
    depth_ffn: &dyn FfnBackend,
    config: &MossTtsRealtimeConfig,
    prefill_matrix: &Array2<u32>,
    text_queue: &[u32],
    text_pad_id: u32,
    max_frames: usize,
    on_frame: impl FnMut(usize, &[u32]),
) -> Result<MossGeneration, EngineError> {
    generate_frames_streaming(
        backend,
        backbone,
        backbone_ffn,
        audio_tables,
        depth,
        depth_ffn,
        config,
        prefill_matrix,
        text_queue,
        text_pad_id,
        max_frames,
        DecodeMode::Greedy,
        0,
        None,
        None,
        on_frame,
    )
}

/// The general form: greedy or sampled ([`DecodeMode`]), seeded for
/// reproducibility in sampled mode (the reference's operating mode —
/// greedy does not terminate on novel text).
#[allow(clippy::too_many_arguments)]
pub fn generate_frames_streaming(
    backend: &dyn EngineBackend,
    backbone: &ModelWeights,
    backbone_ffn: &dyn FfnBackend,
    audio_tables: &[Array2<f32>],
    depth: &DepthTransformerModel,
    depth_ffn: &dyn FfnBackend,
    config: &MossTtsRealtimeConfig,
    prefill_matrix: &Array2<u32>,
    text_queue: &[u32],
    text_pad_id: u32,
    max_frames: usize,
    mode: DecodeMode,
    seed: u64,
    backbone_attn: Option<&dyn KvIndex>,
    depth_attn: Option<&dyn KvIndex>,
    mut on_frame: impl FnMut(usize, &[u32]),
) -> Result<MossGeneration, EngineError> {
    let rvq = config.rvq;
    assert_eq!(
        prefill_matrix.ncols(),
        1 + rvq,
        "prefill matrix must carry one text column plus one per codebook"
    );
    assert_eq!(
        audio_tables.len(),
        rvq,
        "one backbone audio input table per codebook"
    );

    let speech = MossSpeech {
        backend,
        backbone,
        backbone_ffn,
        audio_tables,
        depth,
        depth_ffn,
        config,
        backbone_attn,
        depth_attn,
    };
    let mut session = MossSession::from_prefill_matrix(
        speech,
        prefill_matrix.clone(),
        text_pad_id,
        max_frames,
        mode,
        seed,
    );
    let mut observer = |index: usize, codes: &[u32]| on_frame(index, codes);
    session.push_text_ids(text_queue, &mut observer)?;
    session.end_text(&mut observer)?;
    session.finish(&mut observer)?;
    Ok(session.into_generation())
}

/// One frame through the depth transformer: cached micro-steps over a
/// per-frame handle set, one codebook per head.
///
/// `prior_frames` is the per-codebook repetition history (frame-major);
/// empty on the first frame, which matches the reference disabling the
/// penalty for the prefill frame.
#[allow(clippy::too_many_arguments)]
pub(crate) fn depth_frame(
    depth: &DepthTransformerModel,
    ffn: &dyn FfnBackend,
    backend: &dyn EngineBackend,
    attn_index: Option<&dyn KvIndex>,
    backbone_hidden: &Array2<f32>,
    mode: DecodeMode,
    prior_frames: &[Vec<u32>],
    rng: &mut StdRng,
) -> Result<Vec<u32>, EngineError> {
    let weights = &depth.weights;
    let view = larql_models::WeightsView::dense(weights);
    let rvq = depth.lm_heads.len();

    let mut frame = Vec::with_capacity(rvq);

    // Micro-step 0: the raw backbone hidden is the whole prefix.
    let (mut hidden, mut handles) = kv_prefill_from_hidden_via_dispatch(
        backend,
        view,
        ffn,
        backbone_hidden,
        None,
        None,
        attn_index,
    )
    .map_err(EngineError::Execution)?
    .ok_or_else(|| EngineError::BackendFailure {
        details: "depth prefill declined by backend".into(),
    })?;

    for micro in 0..rvq {
        let code = pick_code(depth, micro, weights, &hidden, mode, prior_frames, rng);
        frame.push(code);
        if micro + 1 == rvq {
            break;
        }
        // Micro-step i+1: embed the code just sampled through table i.
        let embed_row = {
            let table = &depth.embed_tables[micro];
            let mut row = Array2::<f32>::zeros((1, table.ncols()));
            row.row_mut(0).assign(&table.row(code as usize));
            row
        };
        hidden = kv_decode_step_from_hidden_via_dispatch(
            backend,
            view,
            ffn,
            &mut handles,
            &embed_row,
            micro + 1,
            None,
            attn_index,
        )
        .map_err(EngineError::Execution)?
        .ok_or_else(|| EngineError::BackendFailure {
            details: "depth decode step declined by backend".into(),
        })?;
    }

    Ok(frame)
}

fn pick_code(
    depth: &DepthTransformerModel,
    head: usize,
    weights: &ModelWeights,
    hidden: &Array2<f32>,
    mode: DecodeMode,
    prior_frames: &[Vec<u32>],
    rng: &mut StdRng,
) -> u32 {
    let normed = backbone_final_norm(weights, hidden);
    let logits_row = normed.dot(&depth.lm_heads[head].t());
    let mut logits: Vec<f32> = logits_row.row(0).to_vec();
    match mode {
        DecodeMode::Greedy => argmax(&logits) as u32,
        DecodeMode::Sampled(sampling) => {
            // Codebook `head`'s own history across prior frames.
            let history: Vec<u32> = prior_frames.iter().map(|frame| frame[head]).collect();
            apply_repetition_penalty(
                &mut logits,
                &history,
                sampling.repetition_penalty,
                sampling.repetition_window,
            );
            filter_logits(&mut logits, &sampling);
            sample_multinomial(&logits, rng) as u32
        }
    }
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in logits.iter().enumerate() {
        if value > best_value {
            best_value = value;
            best = index;
        }
    }
    best
}

pub(crate) fn backbone_final_norm(weights: &ModelWeights, hidden: &Array2<f32>) -> Array2<f32> {
    apply_norm(
        weights,
        hidden,
        weights.arch.final_norm_key(),
        weights.arch.norm_weight_offset(),
    )
}
