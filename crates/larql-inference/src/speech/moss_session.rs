//! The resumable MOSS generation session — the incremental push-text
//! protocol (`docs/tts-funnel.md` §1.4, funnel step 5).
//!
//! The reference's streaming contract, restated in ids: text ids
//! accumulate in a pending queue; the prefill fires once
//! [`DELAY_TOKENS_LEN`] ids are pending (or the text has ended with
//! fewer), consuming the lead and emitting the first frame; every
//! further pending id drives exactly one backbone step and emits one
//! frame; when the text is over, [`MossSession::finish`] feeds
//! `<|text_pad|>` until audio EOS. Tokenisation policy stays with the
//! caller — the session speaks ids, never strings, so a chat model's
//! token stream and a tokenized prompt are the same input.
//!
//! Batch generation ([`generate_frames_streaming`]) is re-expressed on
//! this session — same prefill, same step order, same RNG draw sequence —
//! so "incremental equals batch" holds by construction and is pinned by
//! the step-5 gate tests rather than by two loops kept manually in sync.
//!
//! One deliberate deviation from the reference: when text ends before
//! the prefill, the reference stuffs *all* pending ids into the prefill
//! lead, but only its string-buffering front end can accumulate more
//! than `DELAY_TOKENS_LEN` there (token pushes drain eagerly). The
//! session always takes `min(pending, DELAY_TOKENS_LEN)` as the lead —
//! identical to the reference for every id-level push sequence, and the
//! only choice under which chunking cannot change the output.
//!
//! [`generate_frames_streaming`]: crate::speech::moss_realtime::generate_frames_streaming

use std::collections::VecDeque;

use ndarray::Array2;

use larql_compute::forward::embed::embed_tables_sum;
use larql_compute::KvIndex;
use larql_models::speech::moss_tts_realtime::{DepthTransformerModel, MossTtsRealtimeConfig};
use larql_models::ModelWeights;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::ffn::FfnBackend;
use crate::kv_dispatch::helpers::{
    kv_decode_step_from_hidden_via_dispatch, kv_prefill_from_hidden_via_dispatch,
};
use crate::kv_dispatch::KvHandle;
use crate::kv_engine::EngineError;
use crate::speech::moss_prompt::{append_text_lead, DELAY_TOKENS_LEN};
use crate::speech::moss_realtime::{backbone_final_norm, depth_frame, FrameStages, MossGeneration};
use crate::speech::moss_sampling::DecodeMode;
use crate::EngineBackend;

/// The loaded speech model, borrowed as one unit: everything a session
/// executes against, none of it mutated.
#[derive(Clone, Copy)]
pub struct MossSpeech<'a> {
    pub backend: &'a dyn EngineBackend,
    pub backbone: &'a ModelWeights,
    pub backbone_ffn: &'a dyn FfnBackend,
    /// Backbone-side per-codebook input tables, summed with the text
    /// embedding each step.
    pub audio_tables: &'a [Array2<f32>],
    pub depth: &'a DepthTransformerModel,
    pub depth_ffn: &'a dyn FfnBackend,
    pub config: &'a MossTtsRealtimeConfig,
    /// Packed (Q4K-direct) attention for the backbone, when quantised.
    pub backbone_attn: Option<&'a dyn KvIndex>,
    /// Packed attention for the depth transformer, when quantised.
    pub depth_attn: Option<&'a dyn KvIndex>,
}

/// Where the prefill matrix comes from.
enum PromptSource {
    /// Incremental: the turn matrix (system + voice splice + assistant
    /// header) waits for the text lead to accumulate; the session
    /// assembles lead rows + audio BOS itself at prefill time.
    Turn(Array2<u32>),
    /// Batch: a complete prefill matrix, lead and BOS already placed by
    /// the caller — ready the moment the session is driven.
    Prefill(Array2<u32>),
    /// Consumed by the prefill.
    Spent,
}

/// Post-prefill backbone state: one KV handle per layer plus the next
/// absolute position.
struct Live {
    handles: Vec<KvHandle>,
    abs_position: usize,
}

/// A resumable generation: text ids in as they become available, frames
/// out as they are earned. Frames are the model's realtime unit (80 ms
/// at 12.5 Hz), so a consumer feeding a codec + ring buffer hangs
/// directly off the `on_frame` callbacks.
pub struct MossSession<'a> {
    speech: MossSpeech<'a>,
    mode: DecodeMode,
    rng: StdRng,
    text_pad_id: u32,
    max_frames: usize,
    source: PromptSource,
    pending: VecDeque<u32>,
    text_ended: bool,
    live: Option<Live>,
    frames: Vec<Vec<u32>>,
    eos_at: Option<usize>,
    prefill_seconds: f64,
    stage_timings: Vec<FrameStages>,
}

/// The per-frame observer: `(frame_index, codes)`, fired the moment the
/// frame exists, before any further compute.
pub type OnFrame<'f> = &'f mut dyn FnMut(usize, &[u32]);

impl<'a> MossSession<'a> {
    /// An incremental session for one turn. `turn_matrix` is the prompt
    /// *without* the text lead ([`crate::speech::moss_prompt::build_turn`]);
    /// the session appends the lead and audio BOS once enough text has
    /// been pushed.
    pub fn new(
        speech: MossSpeech<'a>,
        turn_matrix: Array2<u32>,
        text_pad_id: u32,
        max_frames: usize,
        mode: DecodeMode,
        seed: u64,
    ) -> Self {
        Self::with_source(
            speech,
            PromptSource::Turn(turn_matrix),
            text_pad_id,
            max_frames,
            mode,
            seed,
        )
    }

    /// A session whose prefill matrix is already complete (text lead and
    /// audio BOS placed) — the batch entry point. The prefill fires on
    /// the first drive regardless of pending text.
    pub fn from_prefill_matrix(
        speech: MossSpeech<'a>,
        prefill_matrix: Array2<u32>,
        text_pad_id: u32,
        max_frames: usize,
        mode: DecodeMode,
        seed: u64,
    ) -> Self {
        Self::with_source(
            speech,
            PromptSource::Prefill(prefill_matrix),
            text_pad_id,
            max_frames,
            mode,
            seed,
        )
    }

    fn with_source(
        speech: MossSpeech<'a>,
        source: PromptSource,
        text_pad_id: u32,
        max_frames: usize,
        mode: DecodeMode,
        seed: u64,
    ) -> Self {
        Self {
            speech,
            mode,
            rng: StdRng::seed_from_u64(seed),
            text_pad_id,
            max_frames,
            source,
            pending: VecDeque::new(),
            text_ended: false,
            live: None,
            frames: Vec::new(),
            eos_at: None,
            prefill_seconds: 0.0,
            stage_timings: Vec::new(),
        }
    }

    /// Push text ids and generate every frame they earn. Ids beyond
    /// audio EOS (or the frame cap) stay queued and unconsumed.
    pub fn push_text_ids(&mut self, ids: &[u32], on_frame: OnFrame<'_>) -> Result<(), EngineError> {
        self.pending.extend(ids.iter().copied());
        self.drain(on_frame)
    }

    /// Declare the text complete. If the prefill has not fired yet (text
    /// shorter than the lead), it fires now with everything pending.
    pub fn end_text(&mut self, on_frame: OnFrame<'_>) -> Result<(), EngineError> {
        self.text_ended = true;
        self.drain(on_frame)
    }

    /// Feed `<|text_pad|>` until audio EOS or the frame cap — the
    /// reference's `finish`. Requires [`MossSession::end_text`] first.
    pub fn finish(&mut self, on_frame: OnFrame<'_>) -> Result<(), EngineError> {
        assert!(
            self.text_ended,
            "finish before end_text: the text stream must be closed first"
        );
        self.drain(on_frame)?;
        if self.live.is_none() {
            return Ok(());
        }
        while !self.is_finished() && self.frames.len() < self.max_frames {
            self.step_text(self.text_pad_id, on_frame)?;
        }
        Ok(())
    }

    /// True once codebook 0 has produced audio EOS.
    pub fn is_finished(&self) -> bool {
        self.eos_at.is_some()
    }

    /// Every frame generated so far, EOS frame included.
    pub fn frames(&self) -> &[Vec<u32>] {
        &self.frames
    }

    /// Close the session into the batch result type.
    pub fn into_generation(self) -> MossGeneration {
        MossGeneration {
            frames: self.frames,
            eos_at: self.eos_at,
            prefill_seconds: self.prefill_seconds,
            stage_timings: self.stage_timings,
        }
    }

    /// Prefill if ready, then one step per pending id until EOS or the
    /// frame cap.
    fn drain(&mut self, on_frame: OnFrame<'_>) -> Result<(), EngineError> {
        self.prefill_if_ready(on_frame)?;
        if self.live.is_none() {
            return Ok(());
        }
        while !self.is_finished() && self.frames.len() < self.max_frames {
            let Some(id) = self.pending.pop_front() else {
                break;
            };
            self.step_text(id, on_frame)?;
        }
        Ok(())
    }

    fn prefill_if_ready(&mut self, on_frame: OnFrame<'_>) -> Result<(), EngineError> {
        if self.live.is_some() {
            return Ok(());
        }
        let ready = match &self.source {
            PromptSource::Prefill(_) => true,
            PromptSource::Turn(_) => {
                self.pending.len() >= DELAY_TOKENS_LEN
                    || (self.text_ended && !self.pending.is_empty())
            }
            PromptSource::Spent => unreachable!("prefill already consumed"),
        };
        if !ready {
            return Ok(());
        }
        let matrix = match std::mem::replace(&mut self.source, PromptSource::Spent) {
            PromptSource::Prefill(matrix) => matrix,
            PromptSource::Turn(turn) => {
                let lead_len = self.pending.len().min(DELAY_TOKENS_LEN);
                let lead: Vec<u32> = self.pending.drain(..lead_len).collect();
                append_text_lead(&turn, &lead, self.speech.config)
            }
            PromptSource::Spent => unreachable!(),
        };

        let speech = self.speech;
        let backbone_tables = backbone_tables(speech);
        let phase = larql_compute::phase_timing::start();
        let embeds = embed_tables_sum(&backbone_tables, &matrix);
        larql_compute::phase_timing::finish(phase, "prefill.embed_sum");
        let prefill_start = std::time::Instant::now();
        let (last, handles) = kv_prefill_from_hidden_via_dispatch(
            speech.backend,
            larql_models::WeightsView::dense(speech.backbone),
            speech.backbone_ffn,
            &embeds,
            None,
            None,
            speech.backbone_attn,
        )
        .map_err(EngineError::Execution)?
        .ok_or_else(|| EngineError::BackendFailure {
            details: "backbone prefill declined by backend".into(),
        })?;
        self.prefill_seconds = prefill_start.elapsed().as_secs_f64();
        self.live = Some(Live {
            handles,
            abs_position: embeds.nrows(),
        });

        // Frame 0 rides the prefill: no backbone step of its own.
        let hidden = backbone_final_norm(speech.backbone, &last);
        self.emit_frame(&hidden, 0.0, on_frame)
    }

    /// One outer step: embed `[text_id | previous frame]`, advance the
    /// backbone, emit the frame off the new hidden.
    fn step_text(&mut self, text_id: u32, on_frame: OnFrame<'_>) -> Result<(), EngineError> {
        let speech = self.speech;
        let rvq = speech.config.rvq;
        let previous = self
            .frames
            .last()
            .expect("step_text before the prefill frame");
        let mut step_ids = Array2::<u32>::zeros((1, 1 + rvq));
        step_ids[[0, 0]] = text_id;
        for (column, &code) in previous.iter().enumerate() {
            step_ids[[0, column + 1]] = code;
        }
        let backbone_tables = backbone_tables(speech);
        let step_embed = embed_tables_sum(&backbone_tables, &step_ids);
        let live = self.live.as_mut().expect("step_text before prefill");
        let backbone_start = std::time::Instant::now();
        let step_hidden = kv_decode_step_from_hidden_via_dispatch(
            speech.backend,
            larql_models::WeightsView::dense(speech.backbone),
            speech.backbone_ffn,
            &mut live.handles,
            &step_embed,
            live.abs_position,
            None,
            speech.backbone_attn,
        )
        .map_err(EngineError::Execution)?
        .ok_or_else(|| EngineError::BackendFailure {
            details: "backbone decode step declined by backend".into(),
        })?;
        live.abs_position += 1;
        let backbone_seconds = backbone_start.elapsed().as_secs_f64();
        let hidden = backbone_final_norm(speech.backbone, &step_hidden);
        self.emit_frame(&hidden, backbone_seconds, on_frame)
    }

    /// The depth stage on a final-normed backbone hidden: sample the
    /// frame, record timings, detect EOS, notify.
    fn emit_frame(
        &mut self,
        hidden: &Array2<f32>,
        backbone_seconds: f64,
        on_frame: OnFrame<'_>,
    ) -> Result<(), EngineError> {
        let speech = self.speech;
        let depth_start = std::time::Instant::now();
        let frame = depth_frame(
            speech.depth,
            speech.depth_ffn,
            speech.backend,
            speech.depth_attn,
            hidden,
            self.mode,
            &self.frames,
            &mut self.rng,
        )?;
        self.stage_timings.push(FrameStages {
            backbone_seconds,
            depth_seconds: depth_start.elapsed().as_secs_f64(),
        });
        let is_eos = frame[0] as usize == speech.config.audio_eos_token();
        on_frame(self.frames.len(), &frame);
        self.frames.push(frame);
        if is_eos {
            self.eos_at = Some(self.frames.len() - 1);
        }
        Ok(())
    }
}

/// Text table first, then the per-codebook tables — the summed-embedding
/// input order of `docs/tts-funnel.md` §1.3.
fn backbone_tables<'a>(speech: MossSpeech<'a>) -> Vec<ndarray::ArrayView2<'a, f32>> {
    std::iter::once(speech.backbone.embed.view())
        .chain(speech.audio_tables.iter().map(|table| table.view()))
        .collect()
}
