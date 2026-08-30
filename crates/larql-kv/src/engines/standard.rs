//! StandardEngine — the production K/V cache, wrapped as a `KvEngine`.
//!
//! Step 3c (2026-05-16): migrated from direct `kv_prefill_run` /
//! `kv_decode_step_run` calls to dispatch through
//! [`larql_inference::EngineBackend`] via
//! [`kv_prefill_via_dispatch`] / [`kv_decode_step_via_dispatch`].
//! Cache state is now `Vec<KvHandle>` (one per layer) instead of
//! `KvCache`. Bit-parity with the legacy path is preserved (verified
//! in this file's parity tests + `larql-kv`'s end-to-end suite).
//!
//! Output is bit-identical to today's `--kv-cache standard` (with
//! `window_size: None`) and `--kv-cache markov-bounded`
//! (with `window_size: Some(N)`).

use ndarray::Array2;

use crate::{EngineInfo, KvEngine};
use larql_inference::async_compute_backend::AsyncComputeBackend;
use larql_inference::ffn::FfnBackend;
use larql_inference::kv_dispatch::helpers::{
    kv_decode_step_from_hidden_via_dispatch, kv_decode_step_from_hidden_via_dispatch_async,
    kv_decode_step_via_dispatch, kv_decode_step_via_dispatch_async,
    kv_prefill_from_hidden_via_dispatch, kv_prefill_from_hidden_via_dispatch_async,
    kv_prefill_via_dispatch, kv_prefill_via_dispatch_async,
};
use larql_inference::kv_engine::EngineError;
use larql_inference::model::ModelWeights;
use larql_inference::{cpu_engine_backend, EngineBackend, KvHandle};

/// The new position's content for one decode step: a token id (text
/// path — embedded inside the dispatch helper) or a pre-built hidden
/// row (multi-modal / summed-embedding path, MOSS-TTS-Realtime).
enum StepInput<'a> {
    Token(u32),
    Hidden(&'a Array2<f32>),
}

/// Backend slot — `StandardEngine` accepts either a synchronous
/// [`EngineBackend`] (the default `--kv-cache standard` path) or an
/// [`AsyncComputeBackend`] (opt-in via [`StandardEngine::with_async_backend`]).
///
/// The async variant routes prefill/decode through the async helpers
/// in [`larql_inference::kv_dispatch::helpers`]. At Step A3 of the
/// `async-compute-backend.md` migration, async output is bit-identical
/// to sync on CPU; the win is on Metal once Step A4's deferred dispatch
/// lands.
enum BackendSlot {
    Sync(Box<dyn EngineBackend>),
    Async(Box<dyn AsyncComputeBackend>),
}

impl BackendSlot {
    fn name(&self) -> &str {
        match self {
            BackendSlot::Sync(b) => b.name(),
            BackendSlot::Async(b) => b.name(),
        }
    }
}

/// Which dispatch shape populated `handles` at prefill. Decode must
/// follow the recorded mode: the two shapes are not interchangeable
/// (a coarse handle is one whole-model `CpuQ4kCacheHandle`/Metal cache;
/// per-layer handles are one `CpuKvHandle` per layer), and inferring
/// the mode from `handles.len()` conflates "coarse handle" with
/// "1-layer model" — the misdispatch panicked in the backend downcast.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PrefillDispatchMode {
    /// Backend coarse (fused) path: `handles` is a single whole-model
    /// cache handle from `coarse_prefill`.
    Coarse,
    /// Per-layer dispatch: `handles` has one entry per layer.
    PerLayer,
}

/// Production K/V cache engine. `window_size: None` = unbounded growth
/// (the `--kv-cache standard` flag); `Some(N)` = sliding window (the
/// `--kv-cache markov-bounded --context-window N` flag combo).
///
/// Windowed quant note: the coarse trait surface has no window
/// parameter, so when `window_size` is `Some(N)` the quant entry points
/// decline the coarse path and run per-layer dispatch (which enforces
/// the window via `clip_kv`) — correctness over speed. See
/// [`Self::prefill_quant`].
pub struct StandardEngine {
    window_size: Option<usize>,
    /// One handle per layer (per-layer mode) or a single whole-model
    /// handle (coarse mode); populated by `prefill*`. `None` before
    /// prefill or if the engine has been reset.
    handles: Option<Vec<KvHandle>>,
    /// Which dispatch shape populated `handles` — set together with
    /// `handles` at every prefill entry point; `None` iff `handles`
    /// is `None`. Decode entry points follow this, never handle count.
    prefill_mode: Option<PrefillDispatchMode>,
    /// Tracks the absolute token position of the next token to be
    /// decoded. Set at the end of `prefill` to `prompt_ids.len()`;
    /// incremented after each `decode_step`. The legacy `KvCache` had
    /// its own `next_position` field; this engine tracks it directly.
    abs_position: usize,
    /// Logical position of each resident row, per layer. Kept congruent
    /// with `handles` by every operation that adds or removes rows —
    /// prefill, decode, excise, splice, replace. Empty on the coarse
    /// path, which has no per-layer rows to describe.
    row_positions: larql_inference::kv_row_positions::KvRowPositions,
    backend: BackendSlot,
    /// Engine-owned f32 dequant scratch for the per-layer fallback Q4K path
    /// (`prefill_quant`/`decode_step_quant` populate it; `do_prefill`/
    /// `do_decode_step` resolve attention/FFN through a
    /// `WeightsView::with_scratch` over it). Empty on the dense path. Keeps
    /// `weights` immutable so the engine can hold `Arc<ModelWeights>`.
    dequant_scratch: larql_inference::DequantScratch,
    /// Set when a failed decode step left K/V that could not be rewound,
    /// carrying the rendered cause. While set, decode entry points refuse
    /// with [`EngineError::InvariantViolation`] rather than compute from a
    /// cache that describes no token sequence.
    ///
    /// Cleared by a successful prefill, which replaces the cache outright —
    /// so re-prefilling is the documented way back, and the engine is never
    /// permanently dead.
    invalidated: Option<String>,
}

impl StandardEngine {
    pub fn new(window_size: Option<usize>) -> Self {
        Self::with_backend(window_size, cpu_engine_backend())
    }

    pub fn with_backend(window_size: Option<usize>, backend: Box<dyn EngineBackend>) -> Self {
        Self {
            window_size,
            handles: None,
            prefill_mode: None,
            abs_position: 0,
            row_positions: Default::default(),
            backend: BackendSlot::Sync(backend),
            dequant_scratch: larql_inference::DequantScratch::new(),
            invalidated: None,
        }
    }

    /// Construct with an [`AsyncComputeBackend`]. The engine routes
    /// prefill/decode through async dispatch; output is bit-identical
    /// to [`Self::with_backend`] at Step A3 (parallel-validated) and
    /// faster on Metal once Step A4's deferred dispatch lands.
    pub fn with_async_backend(
        window_size: Option<usize>,
        backend: Box<dyn AsyncComputeBackend>,
    ) -> Self {
        Self {
            window_size,
            handles: None,
            prefill_mode: None,
            abs_position: 0,
            row_positions: Default::default(),
            backend: BackendSlot::Async(backend),
            dequant_scratch: larql_inference::DequantScratch::new(),
            invalidated: None,
        }
    }

    fn cache_memory_bytes(&self) -> usize {
        // Two homes, never both: per-layer dispatch puts the K/V in the
        // handles this engine owns; the coarse path hands back a sentinel
        // handle and keeps the cache inside the backend. Summing both is
        // safe — a backend that answers `backend_resident_kv_bytes` with
        // a non-zero figure is by contract reporting K/V that no handle
        // can see, so there is nothing to double-count.
        let handle_bytes: usize = self
            .handles
            .as_ref()
            .map(|handles| {
                // `resident_bytes` (not the per-layer formula) so a
                // whole-model handle reports all its layers, not one.
                handles.iter().map(|h| h.resident_bytes()).sum()
            })
            .unwrap_or(0);
        handle_bytes + self.backend_resident_kv_bytes()
    }

    /// K/V the backend holds internally (Metal's coarse pipeline). Zero
    /// for the async slot: `AsyncComputeBackend` carries no `KvDispatch`,
    /// and no async backend currently owns a cache of its own.
    fn backend_resident_kv_bytes(&self) -> usize {
        match &self.backend {
            BackendSlot::Sync(b) => b.as_ref().backend_resident_kv_bytes(),
            BackendSlot::Async(_) => 0,
        }
    }

    /// Shared prefill body — both `prefill` (index=None) and
    /// `prefill_quant` (index=Some) route through here. Matches on the
    /// `BackendSlot` to pick sync vs async dispatch.
    ///
    /// This is the engine's policy point for the dispatch ring's three
    /// outcomes: a refusal terminates as
    /// [`EngineError::Execution`] and no hidden state is produced; a
    /// declining backend stays [`EngineError::BackendFailure`], exactly
    /// as the bare `None` it replaced.
    fn do_prefill(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
        index: Option<&larql_inference::larql_vindex::VectorIndex>,
    ) -> Result<Array2<f32>, EngineError> {
        let view = larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch);
        let (hidden, handles) = match &self.backend {
            BackendSlot::Sync(b) => {
                kv_prefill_via_dispatch(b.as_ref(), view, ffn, token_ids, self.window_size, index)
                    .map_err(EngineError::Execution)?
                    .ok_or_else(|| EngineError::BackendFailure {
                        details: "kv_prefill_via_dispatch returned None".into(),
                    })?
            }
            BackendSlot::Async(b) => kv_prefill_via_dispatch_async(
                b.as_ref(),
                view,
                ffn,
                token_ids,
                self.window_size,
                index,
            )
            .map_err(EngineError::Execution)?
            .ok_or_else(|| EngineError::BackendFailure {
                details: "kv_prefill_via_dispatch_async returned None".into(),
            })?,
        };
        self.handles = Some(handles);
        self.prefill_mode = Some(PrefillDispatchMode::PerLayer);
        // A completed prefill replaces the cache outright, so whatever an
        // earlier failed step left behind is gone with it.
        self.invalidated = None;
        self.abs_position = token_ids.len();
        self.reset_row_positions_after_prefill();
        Ok(hidden)
    }

    /// Multi-modal prefill: accept pre-built initial hidden state from
    /// the host (built via `embed_plan` on an `EmbeddingPlan` that may
    /// contain `Precomputed` vision/audio chunks). Same body as
    /// `do_prefill` minus the embed call; `abs_position` advances by
    /// `initial_hidden.nrows()` instead of by token count. See
    /// ADR-0023 for the seam decision.
    fn do_prefill_from_hidden(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        initial_hidden: &Array2<f32>,
    ) -> Result<Array2<f32>, EngineError> {
        let view = larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch);
        // `token_ids: None` — MM hidden rows (vision/audio) have no 1:1
        // token identities, so PLE inputs cannot be derived here. PLE
        // architectures (Gemma 4 E-series) must prefill via the token
        // entry point (`prefill`), which threads `Some(token_ids)`.
        let (hidden, handles) = match &self.backend {
            BackendSlot::Sync(b) => kv_prefill_from_hidden_via_dispatch(
                b.as_ref(),
                view,
                ffn,
                initial_hidden,
                None,
                self.window_size,
                None,
            ),
            BackendSlot::Async(b) => kv_prefill_from_hidden_via_dispatch_async(
                b.as_ref(),
                view,
                ffn,
                initial_hidden,
                None,
                self.window_size,
                None,
            ),
        }
        .map_err(EngineError::Execution)?
        .ok_or_else(|| EngineError::BackendFailure {
            details: "do_prefill_from_hidden returned None (empty hidden input or \
                      backend dispatch failure)"
                .into(),
        })?;
        self.handles = Some(handles);
        self.prefill_mode = Some(PrefillDispatchMode::PerLayer);
        // A completed prefill replaces the cache outright, so whatever an
        // earlier failed step left behind is gone with it.
        self.invalidated = None;
        // Critical: position pointer must be derived from the hidden
        // row count, NOT from any token count — the input may contain
        // vision rows that aren't tokens. Decode-loop correctness
        // depends on this; off-by-one here garbles the entire
        // continuation. Pinned by the StandardEngine entry-point
        // agreement test in this file's tests module.
        self.abs_position = initial_hidden.nrows();
        self.reset_row_positions_after_prefill();
        Ok(hidden)
    }

    /// Shared decode-step body — both `decode_step` (index=None) and
    /// `decode_step_quant` (index=Some) route through here.
    ///
    /// Same policy point as [`Self::do_prefill`]: a refusal terminates as
    /// [`EngineError::Execution`] and no hidden state escapes.
    ///
    /// **Transactional.** A decode step mutates the cache before it can
    /// know whether it will finish: each layer's attention appends the new
    /// token's K/V, and only then does the FFN get the chance to refuse. A
    /// step that does not complete therefore rewinds every handle to the
    /// length it had on entry, so a caller that fixes the refusal's cause
    /// can drive the same token through the same engine.
    ///
    /// When the rewind cannot be trusted — see [`Self::rewind_is_sound`] —
    /// the error is wrapped as [`EngineError::StateInvalidated`] and the
    /// engine records it, so every later decode refuses instead of
    /// computing from a cache that describes no token sequence. `prefill`
    /// clears the condition, because it replaces the cache outright.
    ///
    /// `abs_position` advances only on success, on every path.
    fn do_decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
        index: Option<&larql_inference::larql_vindex::VectorIndex>,
    ) -> Result<Array2<f32>, EngineError> {
        self.do_decode_step_input(weights, ffn, StepInput::Token(token_id), index)
    }

    /// One decode step whose new position is a pre-built hidden row —
    /// the decode-time peer of `do_prefill_from_hidden`. Same rewind and
    /// invalidation semantics as `do_decode_step`; only the dispatch
    /// intent differs.
    fn do_decode_step_input(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        input: StepInput<'_>,
        index: Option<&larql_inference::larql_vindex::VectorIndex>,
    ) -> Result<Array2<f32>, EngineError> {
        if let Some(what) = &self.invalidated {
            return Err(EngineError::InvariantViolation {
                what: format!("decode_step called on an invalidated engine: {what}"),
            });
        }
        let view = larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch);
        let window = self.window_size;
        let abs_position = self.abs_position;
        let backend = &self.backend;
        let handles = self
            .handles
            .as_mut()
            .ok_or_else(|| EngineError::InvariantViolation {
                what: "decode_step called before prefill (handles missing)".into(),
            })?;

        // Snapshot before the first append. Recorded per layer rather than
        // assumed to be uniform: nothing in the trait promises every layer
        // caches the same number of rows, and a wrong assumption here would
        // rewind to a length no layer ever had.
        let entry_lengths: Vec<usize> = handles.iter().map(|h| h.cached_len()).collect();
        let rewindable = Self::rewind_is_sound(window, &entry_lengths);

        let outcome = match (backend, &input) {
            (BackendSlot::Sync(b), StepInput::Token(token_id)) => kv_decode_step_via_dispatch(
                b.as_ref(),
                view,
                ffn,
                handles,
                *token_id,
                abs_position,
                window,
                index,
            ),
            (BackendSlot::Sync(b), StepInput::Hidden(hidden_row)) => {
                kv_decode_step_from_hidden_via_dispatch(
                    b.as_ref(),
                    view,
                    ffn,
                    handles,
                    hidden_row,
                    abs_position,
                    window,
                    index.map(|v| v as &dyn larql_compute::KvIndex),
                )
            }
            (BackendSlot::Async(b), StepInput::Token(token_id)) => {
                kv_decode_step_via_dispatch_async(
                    b.as_ref(),
                    view,
                    ffn,
                    handles,
                    *token_id,
                    abs_position,
                    window,
                    index,
                )
            }
            (BackendSlot::Async(b), StepInput::Hidden(hidden_row)) => {
                kv_decode_step_from_hidden_via_dispatch_async(
                    b.as_ref(),
                    view,
                    ffn,
                    handles,
                    hidden_row,
                    abs_position,
                    window,
                    index.map(|v| v as &dyn larql_compute::KvIndex),
                )
            }
        };

        let failure = match outcome {
            Ok(Some(hidden)) => {
                self.record_decode_append(self.abs_position as u64);
                self.abs_position += 1;
                return Ok(hidden);
            }
            // A declining backend leaves the same half-applied step a refusal
            // does, so it gets the same treatment. Distinguishing them here
            // would make the cache's integrity depend on which of two
            // unrelated things went wrong.
            Ok(None) => EngineError::BackendFailure {
                details: "decode step via dispatch returned None".into(),
            },
            Err(refusal) => EngineError::Execution(refusal),
        };

        let rewound = rewindable && Self::rewind(backend, handles, &entry_lengths);
        if rewound {
            return Err(failure);
        }
        let invalidated = failure.invalidating_engine_state();
        self.invalidated = Some(invalidated.to_string());
        Err(invalidated)
    }

    /// Whether rewinding a failed decode step would restore the exact cache
    /// the step started from.
    ///
    /// Unbounded caches only ever append, so truncating to the recorded
    /// length is exact. A windowed cache is different: a step that reaches
    /// the window drops its oldest row to make room, and that row is gone.
    /// Row *count* cannot see this — append-then-drop leaves it unchanged —
    /// so the only sound test is whether every layer had room to spare
    /// before the step began.
    fn rewind_is_sound(window: Option<usize>, entry_lengths: &[usize]) -> bool {
        match window {
            None => true,
            // `len < w` is "the step's own append still fits", i.e. it will not
            // push the layer past `w` and trigger the evicting clip.
            Some(w) => entry_lengths.iter().all(|&len| len < w),
        }
    }

    /// Truncate every handle back to its recorded length. All-or-nothing:
    /// one backend that cannot rewind makes the whole cache untrustworthy,
    /// because the layers are only meaningful together.
    fn rewind(backend: &BackendSlot, handles: &mut [KvHandle], entry_lengths: &[usize]) -> bool {
        handles
            .iter_mut()
            .zip(entry_lengths)
            .all(|(handle, &len)| match backend {
                BackendSlot::Sync(b) => b.as_ref().truncate_kv(handle, len),
                BackendSlot::Async(b) => b.as_ref().truncate_kv(handle, len),
            })
    }
    /// Rebuild the position map to describe the handles as they stand
    /// after a prefill: one contiguous run per layer, ending at
    /// `abs_position`.
    ///
    /// Sound *only* here. A prefilled cache is contiguous by
    /// construction — a windowed prefill drops its oldest rows, which
    /// shortens the run without perforating it — so the resident rows
    /// and the next position do determine the positions at this one
    /// moment. Every later mutation has to say what it did rather than
    /// have it inferred, which is why nothing else calls this.
    ///
    /// The coarse path keeps an empty map: it has a single whole-model
    /// handle and no per-layer rows to describe.
    fn reset_row_positions_after_prefill(&mut self) {
        self.row_positions = match self.prefill_mode {
            Some(PrefillDispatchMode::PerLayer) => {
                let rows: Vec<usize> = self
                    .handles
                    .as_ref()
                    .map(|hs| hs.iter().map(|h| h.cached_len()).collect())
                    .unwrap_or_default();
                larql_inference::kv_row_positions::KvRowPositions::tails_ending_at(
                    self.abs_position as u64,
                    &rows,
                )
            }
            _ => Default::default(),
        };
    }

    /// Record the row every layer just appended at `position`, then
    /// mirror whatever the window clip dropped.
    ///
    /// The clip happens inside dispatch, below this engine, and it keeps
    /// the physical tail. Reproducing that here keeps the map an honest
    /// description of the cache — including where the physical tail is
    /// the wrong set of rows. Correcting it means changing the clip, not
    /// the map: see `clip_layer_to_logical_window`.
    fn record_decode_append(&mut self, position: u64) {
        if !matches!(self.prefill_mode, Some(PrefillDispatchMode::PerLayer)) {
            return;
        }
        let Some(handles) = self.handles.as_ref() else {
            return;
        };
        for (layer, handle) in handles.iter().enumerate() {
            let Some(map) = self.row_positions.layer_mut(layer) else {
                continue;
            };
            // An append out of order means the engine's position moved
            // backwards under rows that stayed put — an inconsistency
            // this map exists to expose, so let it stand rather than
            // patch it silently.
            if map.append(position).is_ok() {
                map.retain_tail(handle.cached_len());
            }
        }
    }

    /// Per-layer handles, or `None` on the coarse path / before prefill.
    fn layer_handles(&self) -> Option<&[KvHandle]> {
        match self.prefill_mode? {
            PrefillDispatchMode::PerLayer => self.handles.as_deref(),
            PrefillDispatchMode::Coarse => None,
        }
    }
}

impl KvEngine for StandardEngine {
    fn name(&self) -> &str {
        "standard"
    }

    fn info(&self) -> EngineInfo {
        let config = match self.window_size {
            Some(w) => format!("window={w}"),
            None => "window=full".into(),
        };
        let mem = self.cache_memory_bytes();
        EngineInfo {
            name: "standard".into(),
            description: format!(
                "production K/V tensor cache — full FP32 K/V per layer (mem={:.1}MB)",
                mem as f64 / 1_048_576.0,
            ),
            backend: self.backend.name().to_string(),
            config,
        }
    }

    fn prefill(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        self.do_prefill(weights, ffn, token_ids, None)
    }

    fn supports_multimodal(&self) -> bool {
        // StandardEngine is the first (Phase 1d) engine to implement
        // `prefill_from_hidden`. Six other engines inherit the default
        // `false` per ADR-0023 §"Default-false debt". When each of them
        // gains real MM support (or the eventual `prefill →
        // embed_tokens_pub + prefill_from_hidden` collapse lands across
        // the board), the override here becomes redundant and the
        // trait method itself can be removed.
        true
    }

    fn prefill_from_hidden(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        initial_hidden: &Array2<f32>,
    ) -> Result<Array2<f32>, EngineError> {
        self.do_prefill_from_hidden(weights, ffn, initial_hidden)
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.do_decode_step(weights, ffn, token_id, None)
    }

    fn decode_step_from_hidden(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        hidden_row: &Array2<f32>,
    ) -> Result<Array2<f32>, EngineError> {
        self.do_decode_step_input(weights, ffn, StepInput::Hidden(hidden_row), None)
    }

    fn prefill_quant(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
        backend: &dyn larql_inference::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        // Try the backend's coarse (fused) prefill intent first — this
        // is the production-speed Q4K path on CPU (~24 tok/s on Gemma
        // 3 4B vs ~0.4 tok/s through per-layer dispatch). Quant-agnostic:
        // the backend inspects `index` to pick the right kernel.
        //
        // WINDOW: ask the backend to honour this engine's window on the
        // fused path. `coarse_*_windowed` fails closed — a backend that
        // cannot bound BOTH attention and K/V to `window_size` answers
        // `None`, and we fall through to the per-layer path below, which
        // enforces the window via `clip_kv`.
        //
        // That decline is the current answer on every backend, so this
        // is behaviour-preserving today. It replaces a blanket
        // `window_size.is_none()` gate: the reason a windowed engine
        // leaves the fast path is now the backend's own capability
        // answer rather than a rule stated here, so a backend that
        // implements the window starts getting the fused path without
        // this engine changing. That matters — on a host-delegating
        // backend the per-layer route runs the whole forward on the CPU
        // and costs ~2.4x, while the window makes attention *cheaper*.
        let coarse = match &self.backend {
            BackendSlot::Sync(b) => b.as_ref().coarse_prefill_windowed(
                weights,
                token_ids,
                Some(index),
                self.window_size,
            ),
            BackendSlot::Async(b) => b.as_ref().coarse_prefill_windowed(
                weights,
                token_ids,
                Some(index),
                self.window_size,
            ),
        };
        if let Some((hidden, handle)) = coarse {
            // Store as a single-element handles vec — the `KvHandle`
            // wraps the backend's whole-model cache (not per-layer).
            self.handles = Some(vec![handle]);
            self.prefill_mode = Some(PrefillDispatchMode::Coarse);
            self.abs_position = token_ids.len();
            self.reset_row_positions_after_prefill();
            // Same reasoning as the per-layer prefills: the cache is new.
            self.invalidated = None;
            return Ok(hidden);
        }
        // Backend doesn't have a coarse path (e.g. f32 model, or
        // hybrid-MoE / cross-layer-KV models that don't fit the cached
        // shape), or the engine is windowed. Fall back to per-layer
        // dispatch, dequantising attention into the engine-owned scratch
        // (`do_prefill` resolves it through a `WeightsView::with_scratch`)
        // — `weights` stays immutable.
        larql_inference::vindex::ensure_attn_tensors_dequantised(
            &mut self.dequant_scratch,
            weights,
            index,
        );
        // The caller passes `NullFfn` on the quant path (the bench/CLI
        // contract: engines route FFN internally from the vindex — see
        // `NoCacheEngine::prefill_quant`). Forwarding it verbatim would
        // run an identity FFN through `run_ffn` (h + normed(h) garbage)
        // on exactly the archs that decline coarse, so substitute the
        // real Q4K FFN walk built from the vindex.
        let walk_ffn = larql_inference::vindex::WalkFfn::from_config(
            weights,
            index,
            larql_inference::vindex::WalkFfnConfig::dense(weights.num_layers),
        )
        .with_backend(backend);
        self.do_prefill(weights, &walk_ffn, token_ids, Some(index))
    }

    fn decode_step_quant(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
        backend: &dyn larql_inference::ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        // Decode must follow the dispatch mode RECORDED at prefill —
        // never the handle count: a 1-layer model's per-layer prefill
        // also yields exactly one handle, and feeding that per-layer
        // handle to the coarse path is a shape misdispatch (the backend
        // downcast used to panic on it).
        let mode = self
            .prefill_mode
            .ok_or_else(|| EngineError::InvariantViolation {
                what: "decode_step called before prefill (handles missing)".into(),
            })?;
        // The coarse branch below bypasses `do_decode_step`, so it needs the
        // invalidation guard in its own right — a guard that only covers the
        // path that *sets* the flag protects the wrong half.
        if let Some(what) = &self.invalidated {
            return Err(EngineError::InvariantViolation {
                what: format!("decode_step_quant called on an invalidated engine: {what}"),
            });
        }
        if mode == PrefillDispatchMode::Coarse {
            let handles = self
                .handles
                .as_mut()
                .ok_or_else(|| EngineError::InvariantViolation {
                    what: "decode_step called before prefill (handles missing)".into(),
                })?;
            // Invariant: Coarse mode stores exactly one whole-model
            // handle (set together with the mode in `prefill_quant`).
            let handle = &mut handles[0];
            // Windowed variant, matching the prefill that minted this
            // handle: a sequence that reached the fused path under a
            // window must keep decoding under it.
            let coarse = match &self.backend {
                BackendSlot::Sync(b) => b.as_ref().coarse_decode_step_windowed(
                    weights,
                    token_id,
                    Some(index),
                    handle,
                    self.abs_position,
                    self.window_size,
                ),
                BackendSlot::Async(b) => b.as_ref().coarse_decode_step_windowed(
                    weights,
                    token_id,
                    Some(index),
                    handle,
                    self.abs_position,
                    self.window_size,
                ),
            };
            return match coarse {
                Some(h) => {
                    self.abs_position += 1;
                    Ok(h)
                }
                // A coarse-prefilled cache CANNOT flow through the
                // per-layer path (whole-model handle vs one handle per
                // layer), so a coarse decode failure is terminal —
                // erroring beats the old behaviour of falling through
                // and panicking in the per-layer downcast.
                None => Err(EngineError::BackendFailure {
                    details: "coarse_decode_step failed on a coarse-prefilled cache; \
                              the whole-model handle cannot be decoded per-layer"
                        .into(),
                }),
            };
        }
        // Per-layer dispatch (recorded mode: PerLayer). Dequantise
        // attention into the engine-owned scratch (idempotent — persists
        // across decode steps); `do_decode_step` resolves it through a
        // `WeightsView::with_scratch`.
        larql_inference::vindex::ensure_attn_tensors_dequantised(
            &mut self.dequant_scratch,
            weights,
            index,
        );
        // Same FFN substitution as `prefill_quant` — the caller's FFN
        // is `NullFfn` by contract on the quant path; route the real
        // Q4K FFN walk from the vindex.
        let walk_ffn = larql_inference::vindex::WalkFfn::from_config(
            weights,
            index,
            larql_inference::vindex::WalkFfnConfig::dense(weights.num_layers),
        )
        .with_backend(backend);
        self.do_decode_step(weights, &walk_ffn, token_id, Some(index))
    }

    /// Resident-weights variants (task #16): the caller has already made the
    /// client weights f32-resident, so these take `&weights` (no `&mut` for
    /// `ensure_attn_tensors_dequantised`) and just thread `index` to the
    /// per-layer dispatch. That lets the FFN backend borrow the same `&weights`
    /// concurrently (no borrow conflict) while a Q4K-direct attention kernel
    /// reads packed bytes from `index`. CPU uses per-layer handles (the coarse
    /// path is Metal-only), so these route straight to `do_*`.
    fn prefill_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        self.do_prefill(weights, ffn, token_ids, Some(index))
    }

    fn decode_step_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_inference::larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.do_decode_step(weights, ffn, token_id, Some(index))
    }

    fn per_layer_kv_mut(
        &mut self,
    ) -> Option<&mut dyn larql_inference::kv_engine::PerLayerKvAccess> {
        // Only the per-layer dispatch shape can offer row access.
        matches!(self.prefill_mode, Some(PrefillDispatchMode::PerLayer)).then_some(self)
    }

    fn memory_bytes(&self) -> usize {
        self.cache_memory_bytes()
    }

    fn window_tokens(&self) -> usize {
        self.handles
            .as_ref()
            .and_then(|h| h.first())
            .map(|h| h.cached_len())
            .unwrap_or(0)
    }

    fn cold_bytes(&self) -> usize {
        // Standard cache does not have a separate cold tier — the K/V
        // tensors are the state. Sliding-window evictions drop data
        // entirely; nothing is moved to cold.
        0
    }

    fn dispatch_path(&self) -> Option<larql_inference::kv_engine::DispatchPath> {
        use larql_inference::kv_engine::DispatchPath;
        // `prefill_mode` is the authority (see its declaration): handle
        // count cannot distinguish a coarse handle from a 1-layer model.
        self.prefill_mode.map(|mode| match mode {
            PrefillDispatchMode::Coarse => DispatchPath::Coarse,
            PrefillDispatchMode::PerLayer => DispatchPath::PerLayer,
        })
    }
}

// ─── Per-layer K/V access (substrate only) ───────────────────────────────────
//
// Mechanical row surgery for policy wrappers. `StandardEngine` still
// reports `logical_source_masking: false` — it does not decide which
// rows leave, which layers are touched, or how a removal is undone.
// Only the dense/per-layer prefill path offers this; the coarse quant
// path has a single whole-model handle with no per-layer granularity,
// and `prefill_mode` is what distinguishes them.

impl larql_inference::kv_engine::PerLayerKvAccess for StandardEngine {
    fn layer_count(&self) -> Option<usize> {
        match self.prefill_mode? {
            PrefillDispatchMode::PerLayer => self.handles.as_ref().map(Vec::len),
            PrefillDispatchMode::Coarse => None,
        }
    }

    fn logical_next_position(&self) -> usize {
        self.abs_position
    }

    fn resident_rows(&self, layer: usize) -> Option<usize> {
        self.layer_handles()?.get(layer).map(|h| h.cached_len())
    }

    fn read_layer_kv(&self, layer: usize) -> Option<(Array2<f32>, Array2<f32>)> {
        let BackendSlot::Sync(backend) = &self.backend else {
            return None;
        };
        backend.read_kv_to_host(self.layer_handles()?.get(layer)?)
    }

    fn excise_kv_rows(
        &mut self,
        layer: usize,
        rows: std::ops::Range<usize>,
    ) -> Option<larql_inference::kv_engine::ExcisedKvRows> {
        if rows.start >= rows.end {
            return None;
        }
        let BackendSlot::Sync(backend) = &self.backend else {
            // Async dispatch has no host-readback surface here.
            return None;
        };
        match self.prefill_mode? {
            PrefillDispatchMode::PerLayer => {}
            PrefillDispatchMode::Coarse => return None,
        }
        let handles = self.handles.as_mut()?;
        let handle = handles.get(layer)?;
        let (k, v) = backend.read_kv_to_host(handle)?;
        if rows.end > k.nrows() {
            return None;
        }
        // The map must already agree with the cache, or the positions
        // this hands back describe rows that were never there.
        let map = self.row_positions.layer_mut(layer)?;
        if map.len() != k.nrows() {
            return None;
        }

        let removed = larql_inference::kv_engine::ExcisedRows {
            k: k.slice(ndarray::s![rows.clone(), ..]).to_owned(),
            v: v.slice(ndarray::s![rows.clone(), ..]).to_owned(),
        };
        let kept: Vec<usize> = (0..k.nrows()).filter(|i| !rows.contains(i)).collect();
        // Rebuild first: a failed rebuild must leave the map describing
        // the cache that is still there.
        let rebuilt = rebuild_handle(backend.as_ref(), layer, &k, &v, &kept)?;
        let positions = map.excise(rows).ok()?;
        handles[layer] = rebuilt;
        larql_inference::kv_engine::ExcisedKvRows::new(removed, positions)
    }

    fn replace_layer_kv(
        &mut self,
        layer: usize,
        k: &Array2<f32>,
        v: &Array2<f32>,
        positions: &[u64],
    ) -> bool {
        if k.nrows() != v.nrows() || k.ncols() != v.ncols() || positions.len() != k.nrows() {
            return false;
        }
        let candidate = larql_inference::kv_row_positions::LayerRowPositions::from_positions(
            positions.to_vec(),
        );
        if candidate.check_ascending().is_err() {
            return false;
        }
        let BackendSlot::Sync(backend) = &self.backend else {
            return false;
        };
        let Some(PrefillDispatchMode::PerLayer) = self.prefill_mode else {
            return false;
        };
        let Some(handles) = self.handles.as_mut() else {
            return false;
        };
        if layer >= handles.len() || self.row_positions.layer(layer).is_none() {
            return false;
        }
        let all: Vec<usize> = (0..k.nrows()).collect();
        match rebuild_handle(backend.as_ref(), layer, k, v, &all) {
            Some(rebuilt) => {
                handles[layer] = rebuilt;
                // Both halves land together, after everything that could
                // have failed already has.
                *self
                    .row_positions
                    .layer_mut(layer)
                    .expect("layer presence checked above") = candidate;
                true
            }
            None => false,
        }
    }

    fn set_logical_next_position(&mut self, position: usize) -> bool {
        // Rows are not re-rotated by this, so moving the next position
        // *behind* a resident row would leave the cache holding a
        // position it claims has not happened yet — and the next append
        // could not ascend from it. Refuse rather than record it.
        if let Some(max) = self.row_positions.max_position() {
            if (position as u64) <= max {
                return false;
            }
        }
        self.abs_position = position;
        true
    }

    fn row_positions(&self) -> Option<&larql_inference::kv_row_positions::KvRowPositions> {
        matches!(self.prefill_mode, Some(PrefillDispatchMode::PerLayer))
            .then_some(&self.row_positions)
    }

    fn clip_layer_to_logical_window(&mut self, layer: usize, window: u64) -> Option<usize> {
        let BackendSlot::Sync(backend) = &self.backend else {
            return None;
        };
        match self.prefill_mode? {
            PrefillDispatchMode::PerLayer => {}
            PrefillDispatchMode::Coarse => return None,
        }
        let keep = self
            .row_positions
            .layer(layer)?
            .rows_within_window(self.abs_position as u64, window);
        let handles = self.handles.as_mut()?;
        let handle = handles.get(layer)?;
        let (k, v) = backend.read_kv_to_host(handle)?;
        let map = self.row_positions.layer_mut(layer)?;
        if map.len() != k.nrows() {
            return None;
        }
        if keep.len() == map.len() {
            return Some(0);
        }
        let dropped = map.len() - keep.len();
        let rebuilt = rebuild_handle(backend.as_ref(), layer, &k, &v, &keep)?;
        map.retain_rows(&keep).ok()?;
        handles[layer] = rebuilt;
        Some(dropped)
    }

    fn splice_kv_rows(
        &mut self,
        layer: usize,
        at: usize,
        rows: &larql_inference::kv_engine::ExcisedKvRows,
    ) -> bool {
        let BackendSlot::Sync(backend) = &self.backend else {
            return false;
        };
        let Some(PrefillDispatchMode::PerLayer) = self.prefill_mode else {
            return false;
        };
        let Some(handles) = self.handles.as_mut() else {
            return false;
        };
        let Some(handle) = handles.get(layer) else {
            return false;
        };
        let Some((k, v)) = backend.read_kv_to_host(handle) else {
            return false;
        };
        let kv = rows.kv();
        if at > k.nrows() || kv.k.ncols() != k.ncols() {
            return false;
        }
        let Some(map) = self.row_positions.layer_mut(layer) else {
            return false;
        };
        if map.len() != k.nrows() {
            return false;
        }
        // Splice into a copy first. A splice at the wrong offset leaves
        // positions out of order, and finding that out after the handle
        // has been rebuilt would mean the refusal came too late.
        let mut candidate = map.clone();
        if candidate.splice(at, rows.logical_positions()).is_err() {
            return false;
        }

        let total = k.nrows() + rows.len();
        let kv_dim = k.ncols();
        let mut rebuilt = backend.alloc_kv_buffer(layer, total, kv_dim);
        let push = |handle: &mut KvHandle, kr: &[f32], vr: &[f32], pos: usize| {
            backend.append_kv(handle, kr, vr, pos);
        };
        for i in 0..at {
            push(&mut rebuilt, row_of(&k, i), row_of(&v, i), i);
        }
        for i in 0..rows.len() {
            push(&mut rebuilt, row_of(&kv.k, i), row_of(&kv.v, i), at + i);
        }
        for i in at..k.nrows() {
            push(&mut rebuilt, row_of(&k, i), row_of(&v, i), rows.len() + i);
        }
        handles[layer] = rebuilt;
        *map = candidate;
        true
    }
}

/// Rebuild a layer handle holding only `kept` rows of `(k, v)`.
fn rebuild_handle(
    backend: &dyn EngineBackend,
    layer: usize,
    k: &Array2<f32>,
    v: &Array2<f32>,
    kept: &[usize],
) -> Option<KvHandle> {
    let mut rebuilt = backend.alloc_kv_buffer(layer, kept.len().max(1), k.ncols());
    for (physical, &i) in kept.iter().enumerate() {
        backend.append_kv(&mut rebuilt, row_of(k, i), row_of(v, i), physical);
    }
    Some(rebuilt)
}

fn row_of(a: &Array2<f32>, i: usize) -> &[f32] {
    a.row(i)
        .to_slice()
        .expect("host-read K/V rows are contiguous")
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_inference::ffn::WeightFfn;
    use larql_inference::forward::hidden_to_raw_logits;
    use larql_inference::test_utils::make_test_weights;

    // ── Rewind soundness ────────────────────────────────────────────────
    //
    // The predicate that decides whether a failed decode step can be undone.
    // Tested directly because the interesting cases are boundary conditions on
    // the window, and driving each of them through a real refusal would need a
    // refusing route per case while proving the same three facts.

    #[test]
    fn an_unbounded_cache_is_always_rewindable() {
        // Unbounded caches only append, so truncating to the recorded length
        // restores the exact prior state whatever the lengths were.
        assert!(StandardEngine::rewind_is_sound(None, &[0, 7, 4096]));
        assert!(StandardEngine::rewind_is_sound(None, &[]));
    }

    #[test]
    fn a_windowed_cache_is_rewindable_only_with_room_to_spare() {
        const W: usize = 4;
        // Every layer strictly below the window: the step's own append fits,
        // so no eviction fires and the truncate is exact.
        assert!(StandardEngine::rewind_is_sound(Some(W), &[0, 1, 3]));
        // A layer *at* the window evicts to make room, and the evicted row is
        // gone — length would come back while the contents had shifted.
        assert!(!StandardEngine::rewind_is_sound(Some(W), &[3, 4]));
        assert!(!StandardEngine::rewind_is_sound(Some(W), &[W]));
    }

    #[test]
    fn one_unrewindable_layer_condemns_the_whole_step() {
        // All-or-nothing: the layers are only meaningful together, so a single
        // layer at the window makes the cache untrustworthy even if every
        // other layer had room.
        const W: usize = 8;
        assert!(!StandardEngine::rewind_is_sound(Some(W), &[0, 0, 0, W, 0]));
    }

    #[test]
    fn engine_name() {
        assert_eq!(StandardEngine::new(None).name(), "standard");
    }

    #[test]
    fn engine_info_unbounded() {
        let info = StandardEngine::new(None).info();
        assert!(info.config.contains("full"));
    }

    #[test]
    fn engine_info_windowed() {
        let info = StandardEngine::new(Some(128)).info();
        assert!(info.config.contains("128"));
    }

    #[test]
    fn memory_zero_before_prefill() {
        let eng = StandardEngine::new(None);
        assert_eq!(eng.memory_bytes(), 0);
        assert_eq!(eng.window_tokens(), 0);
        assert_eq!(eng.cold_bytes(), 0);
    }

    #[test]
    fn prefill_populates_cache_and_returns_hidden() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = StandardEngine::new(None);
        let h = engine
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .expect("prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > 0, "cache should be populated");
        assert!(engine.window_tokens() >= 3);
    }

    // ─── Phase 1d.3a: StandardEngine entry-point agreement ───────────────
    //
    // Verifies that `prefill(tokens)` and
    // `prefill_from_hidden(embed_tokens_pub(tokens))` agree on BOTH:
    //   (a) the returned hidden state (catches dispatch-level drift),
    //   (b) the post-prefill `abs_position` (catches the off-by-one
    //       that would silently garble decode-loop continuation).
    //
    // The `abs_position` check is the load-bearing one — the new line
    // in `do_prefill_from_hidden` (`self.abs_position = initial_hidden.nrows()`)
    // is the only genuinely new logic in this PR. If it's wrong, the
    // first decoded token after MM prefill gets the wrong RoPE position
    // and the entire continuation is garbled, but the prefill itself
    // looks fine. This test pins that one line.

    #[test]
    fn standard_supports_multimodal() {
        let engine = StandardEngine::new(None);
        assert!(
            engine.supports_multimodal(),
            "StandardEngine is the Phase 1d MM-capable engine"
        );
    }

    #[test]
    fn prefill_and_prefill_from_hidden_agree_on_hidden_and_abs_position() {
        use larql_inference::forward::embed_tokens_pub;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let tokens = [0u32, 1, 2, 3];

        let mut engine_text = StandardEngine::new(None);
        let h_text = engine_text
            .prefill(&weights, &ffn, &tokens)
            .expect("prefill text");
        let abs_text = engine_text.abs_position;

        let mut engine_hidden = StandardEngine::new(None);
        let initial_hidden = embed_tokens_pub(&weights, &tokens);
        let h_hidden = engine_hidden
            .prefill_from_hidden(&weights, &ffn, &initial_hidden)
            .expect("prefill_from_hidden");
        let abs_hidden = engine_hidden.abs_position;

        // (a) hidden state must match bit-identically — same dispatch
        // path, just with the embed hoisted out of the engine.
        assert_eq!(
            h_text, h_hidden,
            "prefill(tokens) and prefill_from_hidden(embed_tokens_pub(tokens)) \
             must produce identical hidden state"
        );
        // (b) `abs_position` must be set from the hidden's row count.
        // For a text-only input where hidden.nrows() == tokens.len(),
        // both paths land on the same value. Phase 1d MM (where vision
        // rows expand the hidden) WILL diverge from token count —
        // that's the whole point of deriving it from `initial_hidden.nrows()`.
        assert_eq!(
            abs_text, abs_hidden,
            "abs_position must agree between text and from-hidden paths \
             (text=tokens.len(), hidden=nrows; for text-only input they coincide)"
        );
        assert_eq!(
            abs_hidden,
            tokens.len(),
            "abs_position after from-hidden prefill must equal input row count"
        );
    }

    #[test]
    fn prefill_from_hidden_abs_position_derives_from_nrows_not_tokens() {
        // Specifically pin the contract that `abs_position` is set from
        // the hidden's row count. For MM, the host's hidden will have
        // more rows than the text token count (image marker + 256 vision
        // rows + text). Synthesize that shape and verify the engine
        // records the full row count.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };

        // 7 rows that are NOT pure tokens — emulate "text + 3 vision +
        // text". Just any Array2 with finite values that the layer
        // graph can run through.
        let mm_rows = 7usize;
        let mut hidden = Array2::<f32>::zeros((mm_rows, weights.hidden_size));
        for r in 0..mm_rows {
            for c in 0..weights.hidden_size {
                hidden[[r, c]] = ((r * 13 + c * 7) % 17) as f32 * 0.01 - 0.08;
            }
        }
        let mut engine = StandardEngine::new(None);
        let _ = engine.prefill_from_hidden(&weights, &ffn, &hidden);
        assert_eq!(
            engine.abs_position, mm_rows,
            "abs_position must = initial_hidden.nrows(), not any token count"
        );
    }

    #[test]
    fn decode_step_produces_finite_logits() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = StandardEngine::new(None);
        engine.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
        let h = engine.decode_step(&weights, &ffn, 2).expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(hidden_to_raw_logits(&weights, &h)
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn cache_grows_with_decode_steps() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = StandardEngine::new(None);
        engine.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        let after_prefill = engine.memory_bytes();
        engine.decode_step(&weights, &ffn, 1).expect("decode 1");
        let after_one = engine.memory_bytes();
        engine.decode_step(&weights, &ffn, 2).expect("decode 2");
        let after_two = engine.memory_bytes();
        assert!(after_one > after_prefill);
        assert!(after_two > after_one);
    }

    #[test]
    fn sliding_window_clips_cache() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let window = 2usize;
        let mut engine = StandardEngine::new(Some(window));
        // Prefill with 4 tokens — cache should clip to last `window` per layer.
        engine
            .prefill(&weights, &ffn, &[0u32, 1, 2, 3])
            .expect("prefill 4 tokens");
        assert!(
            engine.window_tokens() <= window,
            "expected window_tokens ≤ {window}, got {}",
            engine.window_tokens()
        );
    }

    #[test]
    fn decode_step_without_prefill_returns_none() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = StandardEngine::new(None);
        assert!(engine.decode_step(&weights, &ffn, 0).is_err());
    }

    // ── Step 4 parity gate ─────────────────────────────────────────────────
    //
    // `StandardEngine` is the engine-trait wrapper over the production K/V
    // cache. Driven through `generate_with_engine`, its token output must
    // be bit-identical to `generate_cached_backend` on the same inputs.
    // This is the unification's bit-parity gate (spec §8.4); failure here
    // blocks Step 5 (default flip).

    use crate::generation::{generate_cached_backend, generate_with_engine};
    use larql_inference::test_utils::make_test_tokenizer;

    fn run_legacy(
        weights: &larql_inference::ModelWeights,
        tokenizer: &larql_inference::tokenizers::Tokenizer,
        ffn: &WeightFfn<'_>,
        prompt: &[u32],
        max: usize,
        window: Option<usize>,
    ) -> Vec<u32> {
        generate_cached_backend(
            weights,
            tokenizer,
            ffn,
            prompt,
            max,
            None,
            window,
            |_, _| {},
        )
    }

    fn run_engine(
        weights: &larql_inference::ModelWeights,
        tokenizer: &larql_inference::tokenizers::Tokenizer,
        ffn: &WeightFfn<'_>,
        prompt: &[u32],
        max: usize,
        window: Option<usize>,
    ) -> Vec<u32> {
        let mut engine = crate::AnyEngine::Kv(Box::new(StandardEngine::new(window)));
        generate_with_engine(&mut engine, weights, tokenizer, ffn, prompt, max, |_, _| {})
    }

    // The five parity tests below assert bit-exact equality between
    // two code paths that route the same matmuls through different
    // dispatch wrappers. BLAS on Windows runs successive matmuls with
    // different reduction orders (parallel accumulation), so the two
    // paths drift by a few times 1e-3 — enough to flip argmax in a
    // token stream and break hidden-state bit-equality. Linux/macOS
    // BLAS is deterministic and the property holds there; we keep the
    // strict check on those platforms and skip on Windows rather than
    // weaken to a fuzzy tolerance that wouldn't catch real bugs.

    #[cfg(not(windows))]
    #[test]
    fn parity_standard_unbounded_matches_legacy() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let prompt = &[2u32, 3, 5, 7];
        let max = 6;
        let legacy = run_legacy(&weights, &tokenizer, &ffn, prompt, max, None);
        let engine = run_engine(&weights, &tokenizer, &ffn, prompt, max, None);
        assert_eq!(
            engine, legacy,
            "engine dispatch must produce identical tokens to generate_cached_backend (window=None)"
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn parity_standard_windowed_matches_legacy() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let prompt = &[1u32, 2, 3, 4, 5];
        let max = 5;
        // Window smaller than prompt → exercises prefill-time clipping.
        let window = Some(3);
        let legacy = run_legacy(&weights, &tokenizer, &ffn, prompt, max, window);
        let engine = run_engine(&weights, &tokenizer, &ffn, prompt, max, window);
        assert_eq!(
            engine, legacy,
            "engine dispatch must produce identical tokens to generate_cached_backend (sliding window)"
        );
    }

    #[test]
    fn parity_standard_short_prompt_long_window_matches_legacy() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let prompt = &[0u32, 1];
        let max = 4;
        let window = Some(64); // window > prompt — exercises decode-time growth past prompt
        let legacy = run_legacy(&weights, &tokenizer, &ffn, prompt, max, window);
        let engine = run_engine(&weights, &tokenizer, &ffn, prompt, max, window);
        assert_eq!(
            engine, legacy,
            "engine dispatch must produce identical tokens at short-prompt long-window edge case"
        );
    }

    /// Gemma-4 PLE + layer_scalar parity (issue #98 regression, engine
    /// level): StandardEngine's dispatch path must apply the same
    /// per-layer PLE + `layer_scalar` sequence as the legacy
    /// `kv_prefill_run` / `kv_decode_step_run` reference. The synthetic
    /// E2B-like fixture carries non-zero PLE tensors and a non-identity
    /// `layer_scalar`, so dropping either step diverges bit-visibly.
    /// Several decode steps deep because #98's signature was "first
    /// token fine, later tokens garbage".
    #[cfg(not(windows))]
    #[test]
    fn standard_engine_matches_legacy_on_ple_arch() {
        use crate::generation::{kv_decode_step_run, kv_prefill_run};
        use larql_inference::forward::NoopHook;
        use larql_inference::test_utils::make_synthetic_e2b_like_weights;

        const DECODE_STEPS: usize = 4;

        let weights = make_synthetic_e2b_like_weights();
        let ffn = WeightFfn { weights: &weights };
        let prompt = [0u32, 1, 2];

        let mut engine = StandardEngine::new(None);
        let h_engine = engine
            .prefill(&weights, &ffn, &prompt)
            .expect("engine PLE prefill");
        let (h_legacy, mut cache) = kv_prefill_run(
            larql_inference::WeightsView::dense(&weights),
            &ffn,
            &prompt,
            None,
            Some(&larql_compute::CpuBackend),
            &mut NoopHook,
        )
        .expect("legacy PLE prefill");
        let bits = |h: &Array2<f32>| h.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();
        assert_eq!(
            bits(&h_engine),
            bits(&h_legacy),
            "PLE prefill hidden must match legacy bit-for-bit"
        );

        for step in 0..DECODE_STEPS {
            let token = (3 + step) as u32;
            let h_engine = engine
                .decode_step(&weights, &ffn, token)
                .expect("engine PLE decode");
            let h_legacy = kv_decode_step_run(
                &weights,
                &ffn,
                &mut cache,
                token,
                Some(&larql_compute::CpuBackend),
                &mut NoopHook,
            )
            .expect("legacy PLE decode");
            assert_eq!(
                bits(&h_engine),
                bits(&h_legacy),
                "PLE decode step {step} hidden must match legacy bit-for-bit"
            );
        }
    }

    // ── A5 parity gate ──────────────────────────────────────────────
    //
    // `StandardEngine::with_async_backend(CpuBackend)` must produce
    // bit-identical token streams to `StandardEngine::new(CpuBackend)`.
    // CpuBackend's `AsyncComputeBackend` impl is a degenerate
    // `Ready<T>` wrapper around the sync `KvDispatch` (`A2`), so
    // bit-parity is the trait-shape correctness contract for engine
    // opt-in. Spec: `async-compute-backend.md` §10.5.

    use larql_compute::CpuBackend;
    use larql_inference::AsyncComputeBackend;

    fn run_engine_async(
        weights: &larql_inference::ModelWeights,
        tokenizer: &larql_inference::tokenizers::Tokenizer,
        ffn: &WeightFfn<'_>,
        prompt: &[u32],
        max: usize,
        window: Option<usize>,
    ) -> Vec<u32> {
        let backend: Box<dyn AsyncComputeBackend> = Box::new(CpuBackend);
        let mut engine = crate::AnyEngine::Kv(Box::new(StandardEngine::with_async_backend(
            window, backend,
        )));
        generate_with_engine(&mut engine, weights, tokenizer, ffn, prompt, max, |_, _| {})
    }

    #[cfg(not(windows))]
    #[test]
    fn async_parity_standard_unbounded_matches_sync_engine() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let prompt = &[2u32, 3, 5, 7];
        let max = 6;
        let sync = run_engine(&weights, &tokenizer, &ffn, prompt, max, None);
        let asynch = run_engine_async(&weights, &tokenizer, &ffn, prompt, max, None);
        assert_eq!(
            sync, asynch,
            "with_async_backend must produce identical tokens to with_backend (CpuBackend, window=None)"
        );
    }

    #[test]
    fn async_parity_standard_windowed_matches_sync_engine() {
        let weights = make_test_weights();
        let tokenizer = make_test_tokenizer(weights.vocab_size);
        let ffn = WeightFfn { weights: &weights };
        let prompt = &[1u32, 2, 3, 4, 5];
        let max = 5;
        let window = Some(3);
        let sync = run_engine(&weights, &tokenizer, &ffn, prompt, max, window);
        let asynch = run_engine_async(&weights, &tokenizer, &ffn, prompt, max, window);
        assert_eq!(
            sync, asynch,
            "with_async_backend must produce identical tokens to with_backend (CpuBackend, sliding window)"
        );
    }

    #[test]
    fn async_engine_reports_backend_name() {
        let backend: Box<dyn AsyncComputeBackend> = Box::new(CpuBackend);
        let engine = StandardEngine::with_async_backend(None, backend);
        // info() reports the underlying ComputeBackend::name() regardless
        // of which slot variant the engine holds. CpuBackend returns
        // "cpu (BLAS + C Q4 kernel)" or similar — just assert the prefix.
        assert!(
            engine.info().backend.starts_with("cpu"),
            "expected backend name to start with \"cpu\", got {:?}",
            engine.info().backend
        );
    }

    /// Multi-step parity proof: 64 decode steps through both sync and
    /// async dispatch, asserting that *every* intermediate hidden state
    /// is bit-identical. Catches subtle drift that the short-run tests
    /// above would miss — e.g. a one-time K/V append difference, a
    /// per-step accumulating error, a divergence that only surfaces
    /// after many steps.
    ///
    /// This is the accuracy proof for A5: with `Ready*`-wrapped CPU
    /// async, the two paths must produce identical output over a long
    /// generation, not just a 4-token sample.
    #[cfg(not(windows))]
    #[test]
    fn async_parity_long_run_no_drift() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let prompt: Vec<u32> = (0..16).collect();
        let max_steps = 64;

        let mut sync_engine = StandardEngine::new(None);
        let sync_h0 = sync_engine
            .prefill(&weights, &ffn, &prompt)
            .expect("sync prefill");

        let backend: Box<dyn AsyncComputeBackend> = Box::new(CpuBackend);
        let mut async_engine = StandardEngine::with_async_backend(None, backend);
        let async_h0 = async_engine
            .prefill(&weights, &ffn, &prompt)
            .expect("async prefill");

        assert_eq!(
            sync_h0, async_h0,
            "prefill hidden must match bit-for-bit between sync and async dispatch"
        );

        let mut token = 1u32;
        for step in 0..max_steps {
            let sync_h = sync_engine
                .decode_step(&weights, &ffn, token)
                .expect("sync decode_step");
            let async_h = async_engine
                .decode_step(&weights, &ffn, token)
                .expect("async decode_step");
            assert_eq!(
                sync_h, async_h,
                "hidden mismatch at decode step {step} (token={token})"
            );
            let logits = larql_inference::forward::hidden_to_raw_logits(&weights, &sync_h);
            token = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
        }
        assert_eq!(
            sync_engine.window_tokens(),
            async_engine.window_tokens(),
            "post-run cache size must match"
        );
        assert_eq!(
            sync_engine.memory_bytes(),
            async_engine.memory_bytes(),
            "post-run cache memory must match"
        );
    }

    /// Sliding-window variant of the long-run parity test. Different
    /// code path through `clip_kv` per step; same accuracy contract.
    #[cfg(not(windows))]
    #[test]
    fn async_parity_long_run_windowed_no_drift() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let prompt: Vec<u32> = (0..8).collect();
        let max_steps = 64;
        let window = Some(4);

        let mut sync_engine = StandardEngine::new(window);
        sync_engine.prefill(&weights, &ffn, &prompt).unwrap();

        let backend: Box<dyn AsyncComputeBackend> = Box::new(CpuBackend);
        let mut async_engine = StandardEngine::with_async_backend(window, backend);
        async_engine.prefill(&weights, &ffn, &prompt).unwrap();

        let mut token = 1u32;
        for step in 0..max_steps {
            let sync_h = sync_engine.decode_step(&weights, &ffn, token).unwrap();
            let async_h = async_engine.decode_step(&weights, &ffn, token).unwrap();
            assert_eq!(
                sync_h, async_h,
                "windowed hidden mismatch at decode step {step}"
            );
            let logits = larql_inference::forward::hidden_to_raw_logits(&weights, &sync_h);
            token = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
        }
        // Sliding window clips at the helper level — both should end at
        // the same window size.
        assert_eq!(sync_engine.window_tokens(), async_engine.window_tokens());
    }

    // ── Q4K paths via Q4K fixture ─────────────────────────────────────────
    //
    // `prefill_quant` / `decode_step_quant` first try the backend's
    // `coarse_prefill` / `coarse_decode_step`. The Q4K-equipped fixture
    // (`make_test_q4k_vindex` + `make_test_q4k_weights`) satisfies the
    // cached-decode contract, so an UNWINDOWED engine takes the coarse
    // path on `CpuBackend` (verified by
    // `coarse_prefill_records_coarse_mode_and_decodes_coarse`); the
    // per-layer dequant fallback is exercised by the windowed tests
    // further down (windowed engines decline coarse by design).

    // ── dispatch_path reporting ───────────────────────────────────────────

    #[test]
    fn dispatch_path_is_none_before_prefill() {
        // Nothing has chosen a shape yet — reporting one would be a guess.
        assert_eq!(StandardEngine::new(None).dispatch_path(), None);
        assert_eq!(StandardEngine::new(Some(4)).dispatch_path(), None);
    }

    #[test]
    fn dispatch_path_reports_coarse_when_the_backend_took_the_fused_path() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let mut engine = StandardEngine::new(None);
        engine
            .prefill_quant(&weights, &NullFfn, &index, &[0u32, 1, 2], &*backend)
            .expect("prefill_quant");
        assert_eq!(
            engine.dispatch_path(),
            Some(larql_inference::kv_engine::DispatchPath::Coarse),
            "unwindowed Q4K prefill takes the coarse path on CpuBackend"
        );
    }

    /// The window gate's observable consequence. A windowed engine MUST
    /// decline coarse (the coarse surface has no window parameter, so
    /// taking it would silently attend the full context while `info()`
    /// advertised `window=N`). Pinning the reported shape means a future
    /// change that lets a windowed config onto the fused path fails here
    /// rather than quietly returning full-context answers.
    #[test]
    fn windowed_engine_never_reports_coarse() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let mut engine = StandardEngine::new(Some(2));
        engine
            .prefill_quant(&weights, &NullFfn, &index, &[0u32, 1, 2], &*backend)
            .expect("windowed prefill_quant");
        assert_eq!(
            engine.dispatch_path(),
            Some(larql_inference::kv_engine::DispatchPath::PerLayer),
            "a windowed engine must decline the window-less coarse surface"
        );
    }

    #[test]
    fn dense_prefill_reports_per_layer() {
        use larql_inference::ffn::WeightFfn;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let mut engine = StandardEngine::new(None);
        engine
            .prefill(&weights, &WeightFfn { weights: &weights }, &[0u32, 1, 2])
            .expect("dense prefill");
        assert_eq!(
            engine.dispatch_path(),
            Some(larql_inference::kv_engine::DispatchPath::PerLayer),
            "the dense (no-vindex) path is per-layer by construction"
        );
    }

    #[test]
    fn prefill_quant_cpu_fallback_runs_via_dequant() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = StandardEngine::new(None);
        let h = engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2], &*backend)
            .expect("prefill_quant Q4K cpu fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > 0);
    }

    #[test]
    fn decode_step_quant_cpu_fallback_extends_cache() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = StandardEngine::new(None);
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill_quant");
        let mem_before = engine.memory_bytes();
        let h = engine
            .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
            .expect("decode_step_quant Q4K cpu fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(
            engine.memory_bytes() > mem_before,
            "K/V cache should grow after Q4K decode step"
        );
    }

    #[test]
    fn decode_step_quant_without_prefill_returns_none() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = StandardEngine::new(None);
        // self.handles is None → decode_step_quant returns an
        // InvariantViolation error at the `self.handles.as_mut()` guard.
        assert!(engine
            .decode_step_quant(&weights, &ffn, &index, 0, &*backend)
            .is_err());
    }

    // ── Empty-prompt guards ────────────────────────────────────────────────
    //
    // `prefill` and `prefill_quant` reject an empty token slice with
    // `EngineError::EmptyPrompt` *before* touching the backend. Pinning
    // both arms keeps the early-return invariant covered (the dispatch
    // helpers would otherwise return None and the caller could not tell
    // "empty input" from "backend failed").

    #[test]
    fn prefill_empty_prompt_returns_empty_prompt_error() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = StandardEngine::new(None);
        let err = engine.prefill(&weights, &ffn, &[]).unwrap_err();
        assert!(
            matches!(err, EngineError::EmptyPrompt),
            "empty prompt must map to EngineError::EmptyPrompt, got {err:?}"
        );
    }

    #[test]
    fn prefill_quant_empty_prompt_returns_empty_prompt_error() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = StandardEngine::new(None);
        let err = engine
            .prefill_quant(&weights, &ffn, &index, &[], &*backend)
            .unwrap_err();
        assert!(
            matches!(err, EngineError::EmptyPrompt),
            "empty prompt must map to EngineError::EmptyPrompt, got {err:?}"
        );
    }

    // ── Resident-weights quant path (task #16) ─────────────────────────────
    //
    // `prefill_resident` / `decode_step_resident` are the
    // `--moe-shards`/`LARQL_Q4K_DIRECT_ATTN` entry points: they take
    // `&ModelWeights` (immutable — the caller has already made the client
    // weights f32-resident) and thread `index` to the per-layer dispatch.
    // With `LARQL_Q4K_DIRECT_ATTN` unset (the default in tests) the CPU
    // backend ignores the index and reads f32 from `weights.tensors`, so
    // driving these with f32 test weights + a Q4K vindex exercises the
    // resident code path (`do_prefill` / `do_decode_step` with
    // `Some(index)`) without needing the Metal-only direct kernel.

    #[test]
    fn prefill_resident_populates_cache_and_returns_hidden() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let ffn = NullFfn;
        let mut engine = StandardEngine::new(None);
        let h = engine
            .prefill_resident(&weights, &ffn, &index, &[0u32, 1, 2])
            .expect("prefill_resident");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
        assert!(engine.memory_bytes() > 0, "cache should be populated");
    }

    #[test]
    fn prefill_resident_empty_prompt_returns_empty_prompt_error() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let ffn = NullFfn;
        let mut engine = StandardEngine::new(None);
        let err = engine
            .prefill_resident(&weights, &ffn, &index, &[])
            .unwrap_err();
        assert!(matches!(err, EngineError::EmptyPrompt));
    }

    #[test]
    fn decode_step_resident_extends_cache() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let ffn = NullFfn;
        let mut engine = StandardEngine::new(None);
        engine
            .prefill_resident(&weights, &ffn, &index, &[0u32, 1])
            .expect("prefill_resident");
        let mem_before = engine.memory_bytes();
        let h = engine
            .decode_step_resident(&weights, &ffn, &index, 2)
            .expect("decode_step_resident");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
        assert!(
            engine.memory_bytes() > mem_before,
            "K/V cache should grow after a resident decode step"
        );
    }

    #[test]
    fn decode_step_resident_without_prefill_returns_error() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let ffn = NullFfn;
        let mut engine = StandardEngine::new(None);
        // No prefill → handles is None → InvariantViolation at the guard.
        assert!(engine
            .decode_step_resident(&weights, &ffn, &index, 0)
            .is_err());
    }

    // ── Quant paths through the async backend slot ─────────────────────────
    //
    // `prefill_quant` / `decode_step_quant` both branch on the
    // `BackendSlot` to call `coarse_prefill` / `coarse_decode_step`.
    // The sync-slot arms are covered by the Q4K fixture tests above; these
    // drive the *async* slot arms. `CpuBackend`'s coarse path returns None
    // either way, so the engine still falls through to the per-layer
    // dequant fallback — but the async match arm (and its `coarse_*` call)
    // is now executed.

    #[test]
    fn prefill_quant_async_slot_falls_back_via_dequant() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        use larql_inference::AsyncComputeBackend;
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let async_backend: Box<dyn AsyncComputeBackend> = Box::new(CpuBackend);
        let ffn = NullFfn;
        let mut engine = StandardEngine::with_async_backend(None, async_backend);
        let h = engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2], &*backend)
            .expect("prefill_quant async-slot fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > 0);
    }

    #[test]
    fn decode_step_quant_async_slot_falls_back_via_dequant() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        use larql_inference::AsyncComputeBackend;
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let async_backend: Box<dyn AsyncComputeBackend> = Box::new(CpuBackend);
        let ffn = NullFfn;
        let mut engine = StandardEngine::with_async_backend(None, async_backend);
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill_quant async-slot");
        let mem_before = engine.memory_bytes();
        let h = engine
            .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
            .expect("decode_step_quant async-slot fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(
            engine.memory_bytes() > mem_before,
            "K/V cache should grow after async-slot Q4K decode step"
        );
    }

    // ── Windowed quant path honors the window (bug fix) ───────────────────
    //
    // The coarse trait surface (`coarse_prefill` / `coarse_decode_step`)
    // has no window parameter, so a windowed engine that took the coarse
    // path silently attended over the FULL context while `info()`
    // reported `window=N`. The fix declines coarse whenever
    // `window_size.is_some()` and routes through the per-layer path,
    // whose dispatch enforces the window via `clip_kv`. These tests
    // assert the actual row accounting, not just shape.

    /// Window used by the windowed-quant tests — deliberately smaller
    /// than the prompt so prefill-time clipping is exercised.
    const QUANT_WINDOW: usize = 2;
    /// Decode steps run after prefill in the windowed-quant tests.
    const QUANT_DECODE_STEPS: usize = 3;

    #[test]
    fn prefill_quant_windowed_bounds_cached_rows_to_window() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let prompt = [0u32, 1, 2, 3, 4]; // 5 tokens > QUANT_WINDOW
        let mut engine = StandardEngine::new(Some(QUANT_WINDOW));
        engine
            .prefill_quant(&weights, &ffn, &index, &prompt, &*backend)
            .expect("windowed prefill_quant");
        // The load-bearing assertion: the cache must hold at most
        // `window` rows. On the (window-less) coarse path this is the
        // full prompt length — the bug this test pins.
        assert!(
            engine.window_tokens() <= QUANT_WINDOW,
            "windowed quant prefill must bound cached rows to window={QUANT_WINDOW}, \
             got {} (coarse path ignores the window)",
            engine.window_tokens()
        );
        // Decode past the window: the bound must hold on every step.
        for step in 0..QUANT_DECODE_STEPS {
            let token = (5 + step) as u32;
            engine
                .decode_step_quant(&weights, &ffn, &index, token, &*backend)
                .expect("windowed decode_step_quant");
            assert!(
                engine.window_tokens() <= QUANT_WINDOW,
                "window bound violated after decode step {step}: {} rows",
                engine.window_tokens()
            );
        }
    }

    /// Windowed quant output must be bit-identical to the per-layer
    /// windowed path — because post-fix it IS the per-layer windowed
    /// path. The reference drives the engine internals directly with an
    /// explicitly-constructed `WalkFfn` (the substitution
    /// `prefill_quant` performs itself), so any divergence means the
    /// quant entry points stopped enforcing the window or stopped
    /// routing the real FFN.
    #[test]
    fn windowed_quant_path_matches_per_layer_windowed_reference() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let prompt = [0u32, 1, 2, 3, 4];
        let bits = |h: &Array2<f32>| h.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();

        // Engine A: public quant entry points, caller passes NullFfn
        // (the bench-harness contract — engines route FFN internally).
        let ffn = NullFfn;
        let mut engine_a = StandardEngine::new(Some(QUANT_WINDOW));
        let h_a = engine_a
            .prefill_quant(&weights, &ffn, &index, &prompt, &*backend)
            .expect("engine A windowed prefill_quant");

        // Engine B: the per-layer windowed reference — same dequant
        // scratch handling + explicit WalkFfn, driven through the
        // internal per-layer body.
        let mut engine_b = StandardEngine::new(Some(QUANT_WINDOW));
        larql_inference::vindex::ensure_attn_tensors_dequantised(
            &mut engine_b.dequant_scratch,
            &weights,
            &index,
        );
        let walk_ffn = larql_inference::vindex::WalkFfn::from_config(
            &weights,
            &index,
            larql_inference::vindex::WalkFfnConfig::dense(weights.num_layers),
        )
        .with_backend(&*backend);
        let h_b = engine_b
            .do_prefill(&weights, &walk_ffn, &prompt, Some(&index))
            .expect("engine B per-layer windowed prefill");
        assert_eq!(
            bits(&h_a),
            bits(&h_b),
            "windowed quant prefill must match the per-layer windowed reference bit-for-bit"
        );

        for step in 0..QUANT_DECODE_STEPS {
            let token = (5 + step) as u32;
            let h_a = engine_a
                .decode_step_quant(&weights, &ffn, &index, token, &*backend)
                .expect("engine A windowed decode_step_quant");
            let h_b = engine_b
                .do_decode_step(&weights, &walk_ffn, token, Some(&index))
                .expect("engine B per-layer windowed decode");
            assert_eq!(
                bits(&h_a),
                bits(&h_b),
                "windowed quant decode step {step} must match the per-layer reference bit-for-bit"
            );
        }
    }

    // ── Per-layer quant fallback routes the real FFN (bug fix) ────────────
    //
    // The bench/CLI passes `NullFfn` on the quant path (engines route
    // FFN internally from the vindex — see `NoCacheEngine`). The
    // per-layer quant fallback used to forward the caller's `NullFfn`
    // verbatim into `run_ffn`, silently producing `h + normed(h)`
    // garbage on exactly the archs that decline coarse. Post-fix the
    // fallback substitutes a `WalkFfn` built from the vindex; this test
    // bit-compares against an explicitly-constructed `WalkFfn` driven
    // through the same internals.

    /// Window NARROWER than the prompt, so the backend declines the
    /// fused path and both engines run the per-layer quant walk this
    /// test is about.
    ///
    /// It used to be a window *wider* than the whole run, on the premise
    /// that "coarse is declined whenever windowed". That premise is gone:
    /// a backend now accepts a window it can honour, and a window wider
    /// than the prompt is trivially honourable — so engine A took the
    /// fused path while the reference stayed per-layer and the two
    /// legitimately disagreed. Narrower-than-prompt is the property that
    /// actually forces per-layer now.
    ///
    /// The window then clips, but it clips BOTH engines identically
    /// (same `window_size`, same `do_prefill` internals), so the FFN
    /// routing this test isolates is unaffected.
    const FORCE_PER_LAYER_WINDOW: usize = 2;

    #[test]
    fn quant_fallback_with_null_ffn_matches_explicit_walk_ffn_reference() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let prompt = [0u32, 1, 2];
        let bits = |h: &Array2<f32>| h.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();

        // Engine A: public quant entry points with NullFfn.
        let ffn = NullFfn;
        let mut engine_a = StandardEngine::new(Some(FORCE_PER_LAYER_WINDOW));
        let h_a = engine_a
            .prefill_quant(&weights, &ffn, &index, &prompt, &*backend)
            .expect("engine A prefill_quant");
        // Pin the premise: this test is about the PER-LAYER fallback, so
        // it is only meaningful if engine A actually took it. Without
        // this, a future widening of what the fused path accepts would
        // make the comparison silently test nothing.
        assert_eq!(
            engine_a.dispatch_path(),
            Some(larql_inference::kv_engine::DispatchPath::PerLayer),
            "engine A must be on the per-layer path for this comparison to mean anything"
        );

        // Engine B: same internals, explicit WalkFfn.
        let mut engine_b = StandardEngine::new(Some(FORCE_PER_LAYER_WINDOW));
        larql_inference::vindex::ensure_attn_tensors_dequantised(
            &mut engine_b.dequant_scratch,
            &weights,
            &index,
        );
        let walk_ffn = larql_inference::vindex::WalkFfn::from_config(
            &weights,
            &index,
            larql_inference::vindex::WalkFfnConfig::dense(weights.num_layers),
        )
        .with_backend(&*backend);
        let h_b = engine_b
            .do_prefill(&weights, &walk_ffn, &prompt, Some(&index))
            .expect("engine B reference prefill");
        assert_eq!(
            bits(&h_a),
            bits(&h_b),
            "quant-fallback prefill with NullFfn must substitute the real WalkFfn \
             (bit-parity with the explicit-WalkFfn reference)"
        );

        for step in 0..QUANT_DECODE_STEPS {
            let token = (3 + step) as u32;
            let h_a = engine_a
                .decode_step_quant(&weights, &ffn, &index, token, &*backend)
                .expect("engine A decode_step_quant");
            let h_b = engine_b
                .do_decode_step(&weights, &walk_ffn, token, Some(&index))
                .expect("engine B reference decode");
            assert_eq!(
                bits(&h_a),
                bits(&h_b),
                "quant-fallback decode step {step} with NullFfn must match the \
                 explicit-WalkFfn reference bit-for-bit"
            );
        }
    }

    // ── Prefill-mode tracking (bug fix) ───────────────────────────────────
    //
    // `decode_step_quant` used to infer "coarse handle" from
    // `handles.len() == 1`, which conflates it with "1-layer model":
    // a 1-layer model prefilled via the per-layer path hit the coarse
    // downcast (`cpu_q4k_cache_mut`) with a `CpuKvHandle` and panicked.
    // The engine now records which mode prefill used and decode follows
    // the recorded mode.

    #[test]
    fn one_layer_per_layer_prefill_then_decode_step_quant_does_not_panic() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights_layers};
        /// A single decoder layer — the count that used to be
        /// indistinguishable from the coarse single-handle shape.
        const ONE_LAYER: usize = 1;
        let weights = make_test_q4k_weights_layers(ONE_LAYER);
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let prompt = [0u32, 1, 2];
        let bits = |h: &Array2<f32>| h.iter().map(|v| v.to_bits()).collect::<Vec<u32>>();

        let walk_ffn = larql_inference::vindex::WalkFfn::from_config(
            &weights,
            &index,
            larql_inference::vindex::WalkFfnConfig::dense(weights.num_layers),
        )
        .with_backend(&*backend);

        // Engine A: per-layer prefill via the resident entry point
        // (per-layer handles — for a 1-layer model that is exactly one
        // handle), then decode through the public quant entry point.
        let mut engine_a = StandardEngine::new(None);
        engine_a
            .prefill_resident(&weights, &walk_ffn, &index, &prompt)
            .expect("engine A per-layer prefill_resident");
        assert_eq!(
            engine_a.handles.as_ref().map(|h| h.len()),
            Some(ONE_LAYER),
            "1-layer per-layer prefill must produce exactly one per-layer handle"
        );
        // Pre-fix: panics in `cpu_q4k_cache_mut` (coarse downcast on a
        // per-layer `CpuKvHandle`). Post-fix: follows the recorded
        // per-layer mode.
        let ffn = NullFfn;
        let h_a = engine_a
            .decode_step_quant(&weights, &ffn, &index, 3, &*backend)
            .expect("decode_step_quant after per-layer prefill on a 1-layer model");

        // Engine B: pure per-layer reference for the same step.
        let mut engine_b = StandardEngine::new(None);
        engine_b
            .prefill_resident(&weights, &walk_ffn, &index, &prompt)
            .expect("engine B per-layer prefill_resident");
        larql_inference::vindex::ensure_attn_tensors_dequantised(
            &mut engine_b.dequant_scratch,
            &weights,
            &index,
        );
        let h_b = engine_b
            .do_decode_step(&weights, &walk_ffn, 3, Some(&index))
            .expect("engine B per-layer reference decode");
        assert_eq!(
            bits(&h_a),
            bits(&h_b),
            "1-layer decode_step_quant must produce per-layer parity, not a coarse \
             misdispatch"
        );
    }

    /// After a *coarse* quant prefill the engine must keep decoding
    /// coarse — the recorded mode, not the handle count, drives the
    /// dispatch. (Unwindowed engine + the coarse-capable Q4K fixture
    /// takes the coarse path on CPU; the handle is the whole-model
    /// `CpuQ4kCacheHandle`.)
    #[test]
    fn coarse_prefill_records_coarse_mode_and_decodes_coarse() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let prompt = [0u32, 1, 2];
        let mut engine = StandardEngine::new(None);
        engine
            .prefill_quant(&weights, &ffn, &index, &prompt, &*backend)
            .expect("coarse prefill_quant");
        assert_eq!(
            engine.prefill_mode,
            Some(PrefillDispatchMode::Coarse),
            "unwindowed quant prefill on a coarse-capable backend must record Coarse"
        );
        let cached_before = engine.window_tokens();
        engine
            .decode_step_quant(&weights, &ffn, &index, 3, &*backend)
            .expect("coarse decode_step_quant");
        assert_eq!(
            engine.window_tokens(),
            cached_before + 1,
            "coarse decode must extend the whole-model cache by exactly one row"
        );
    }
}
