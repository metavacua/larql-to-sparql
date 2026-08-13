//! `WindowedCheckpointEngine` — window-based KV cache with boundary-checkpoint replay.
//!
//! Window lifecycle:
//!   1. `process(tokens)` — extends the active window's K,V via
//!      `rs_extend_from_checkpoint`. Auto-closes when the window fills.
//!   2. `close_window()` — saves last-position K,V to `CheckpointStore`,
//!      appends token IDs to `TokenArchive`, resets active window.
//!   3. `replay_window(id)` — reconstructs a window's full K,V by replaying
//!      archived tokens from the prior checkpoint.
//!   4. `stats()` — total bytes, windows, compression ratio vs full KV.
//!
//! Memory at 370K tokens (Gemma 3 4B, W=512):
//!   Checkpoints ≈ 278 KB/window × N_windows
//!   Token archive = 4 bytes/token
//!   Total ≈ 30 MB  vs  25.8 GB for Standard KV  (≈2,000×)

// `EngineStats` is pure data + a `String`/`format!`-only `summary()` --
// no VectorIndex/Instant/env touch -- and stays portable.
// `WindowedCheckpointEngine` (`impl KvEngine`, upstream-gated) and every
// import it alone needs are native.
#[cfg(not(target_arch = "wasm32"))]
use larql_compute::ComputeBackend;
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::{cpu_engine_backend, EngineBackend};
#[cfg(not(target_arch = "wasm32"))]
use larql_vindex::VectorIndex;
#[cfg(not(target_arch = "wasm32"))]
use ndarray::Array2;
use serde::Serialize;

#[cfg(not(target_arch = "wasm32"))]
use super::checkpoint_store::CheckpointStore;
#[cfg(not(target_arch = "wasm32"))]
use super::extend::{
    empty_prior, rs_extend_from_checkpoint_backend, rs_extend_from_checkpoint_quant,
    rs_extend_inplace, truncate_kv_rows,
};
#[cfg(not(target_arch = "wasm32"))]
use super::token_archive::TokenArchive;
#[cfg(not(target_arch = "wasm32"))]
use crate::engines::markov_residual::ensure_attn_tensors_dequantised;
#[cfg(not(target_arch = "wasm32"))]
use crate::{EngineInfo, KvEngine};
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::attention::SharedKV;
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::ffn::FfnBackend;
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::kv_engine::EngineError;
#[cfg(not(target_arch = "wasm32"))]
use larql_inference::model::ModelWeights;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

// ─── EngineStats ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct EngineStats {
    pub total_tokens: usize,
    pub archived_windows: usize,
    pub current_window_id: usize,
    pub current_window_tokens: usize,
    pub checkpoint_bytes: usize,
    pub archive_bytes: usize,
    pub total_boundary_bytes: usize,
    pub equivalent_kv_bytes: usize,
    pub compression_ratio: f64,
}

impl EngineStats {
    pub fn summary(&self) -> String {
        format!(
            "{} windows / {} tokens — {:.0}× compression vs full KV",
            self.archived_windows, self.total_tokens, self.compression_ratio,
        )
    }
}

// ─── Engine ──────────────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
pub struct WindowedCheckpointEngine {
    pub window_size: usize,
    pub checkpoints: CheckpointStore,
    pub archive: TokenArchive,

    pub(super) current_window_id: usize,
    pub(super) current_window_tokens: Vec<u32>,
    /// Per-layer K/V for the current (partial) window.
    ///
    /// Two layouts coexist:
    /// - **Pre-allocated** (dispatch hot path): `Array2` is shaped
    ///   `[window_size, kv_dim]` with only the first
    ///   `current_window_kv_len` rows valid; the rest are zeros. Used by
    ///   `try_prefill_via_dispatch` / `decode_step_via_dispatch` so the
    ///   per-step append is one `slice_mut().assign(row)`, not a fresh
    ///   `Array2::zeros((n+1, kv_dim)) + slice-copy`.
    /// - **Narrow** (CPU walk path): `Array2` is shaped `[n, kv_dim]`,
    ///   matching the arrays returned by `rs_extend_from_checkpoint_*`.
    ///   `current_window_kv_len` equals `n` here, so readers can treat
    ///   the two layouts uniformly via the counter.
    ///
    /// Readers that need the logical length **must** use
    /// `current_window_kv_len`, not `k.shape()[0]`.
    pub(super) current_window_kv: Option<Vec<SharedKV>>,
    /// Logical row count for `current_window_kv`. See field doc above.
    pub(super) current_window_kv_len: usize,
    pub(super) abs_offset: usize,
    /// Hidden state at the last processed token; set by `process()`.
    pub(super) last_hidden: Option<Array2<f32>>,
    pub(super) backend: Box<dyn EngineBackend>,
    pub(super) profiling: bool,
    pub(super) profile: crate::profiler::EngineProfiler,
    /// W1-GPU: handle into the backend's K/V cache, populated when
    /// prefill routes through `coarse_prefill_with_state`. `None` =
    /// legacy CPU walk path.
    pub(super) kv_handle: Option<larql_inference::KvHandle>,
    /// Engine-owned f32 dequant scratch for the per-layer walk fallback
    /// (see `MarkovResidualEngine::dequant_scratch`). Keeps `weights` immutable.
    pub(super) dequant_scratch: larql_inference::DequantScratch,
}

/// Smallest legal window. `window_size == 0` would make the fill check
/// in `process()` (`current_window_tokens.len() >= window_size`) true
/// forever with nothing consumed — an infinite close loop — so it is
/// rejected at construction. `window_size == 1` is legal.
#[cfg(not(target_arch = "wasm32"))]
const MIN_WINDOW_SIZE: usize = 1;

#[cfg(not(target_arch = "wasm32"))]
impl WindowedCheckpointEngine {
    /// # Panics
    ///
    /// Panics if `window_size < MIN_WINDOW_SIZE` (i.e. zero).
    pub fn new(window_size: usize) -> Self {
        Self::with_backend(window_size, cpu_engine_backend())
    }

    /// # Panics
    ///
    /// Panics if `window_size < MIN_WINDOW_SIZE` (i.e. zero).
    pub fn with_backend(window_size: usize, backend: Box<dyn EngineBackend>) -> Self {
        assert!(
            window_size >= MIN_WINDOW_SIZE,
            "WindowedCheckpointEngine window_size must be >= {MIN_WINDOW_SIZE}, got {window_size}"
        );
        Self {
            window_size,
            checkpoints: CheckpointStore::new(),
            archive: TokenArchive::new(),
            current_window_id: 0,
            current_window_tokens: Vec::new(),
            current_window_kv: None,
            current_window_kv_len: 0,
            abs_offset: 0,
            last_hidden: None,
            backend,
            profiling: false,
            profile: crate::profiler::EngineProfiler::default(),
            kv_handle: None,
            dequant_scratch: larql_inference::DequantScratch::new(),
        }
    }

    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.profiling = enabled;
        self
    }

    /// Feed tokens into the engine. Windows auto-close when they fill.
    pub fn process(
        &mut self,
        weights: &ModelWeights,
        tokens: &[u32],
        moe_ffn: Option<&dyn larql_inference::ffn::FfnBackend>,
    ) -> Result<(), EngineError> {
        self.process_with_index(weights, tokens, moe_ffn, None)
    }

    /// `process` with an optional vindex threaded to the per-token attention
    /// steps (Q4K-direct route under `LARQL_Q4K_DIRECT_ATTN` — the
    /// non-standard-engine structural-gap fix).
    pub fn process_with_index(
        &mut self,
        weights: &ModelWeights,
        tokens: &[u32],
        moe_ffn: Option<&dyn larql_inference::ffn::FfnBackend>,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Result<(), EngineError> {
        let mut remaining = tokens;
        // Closing a window archives its tokens and saves its checkpoint, and
        // neither is undoable. `extend_current` rewinds the *current* window
        // exactly, so a failure is retryable right up until the first close —
        // after that the engine holds a stream it cannot complete, and must
        // say so rather than let a caller retry into a duplicated window.
        let mut closed_a_window = false;
        while !remaining.is_empty() {
            let free = self.window_size - self.current_window_tokens.len();
            let take = remaining.len().min(free);
            let (chunk, rest) = remaining.split_at(take);
            if let Err(failure) = self.extend_current(weights, chunk, moe_ffn, index) {
                return Err(if closed_a_window {
                    failure.invalidating_engine_state()
                } else {
                    failure
                });
            }
            remaining = rest;
            if self.current_window_tokens.len() >= self.window_size {
                self.close_window();
                closed_a_window = true;
            }
        }
        Ok(())
    }

    /// Close any partial current window. Call before replay if the window hasn't filled.
    pub fn flush(&mut self) {
        if !self.current_window_tokens.is_empty() {
            self.close_window();
        }
    }

    /// Reconstruct a window's full K,V by replaying its archived tokens from
    /// the prior window's boundary checkpoint.
    ///
    /// For hybrid-MoE models, pass the FFN hook + vindex so the replay
    /// dispatches experts exactly like the live-window path
    /// ([`extend_current`](Self::extend_current)); pass `None`/`None` for dense
    /// models. (Previously this always passed `None` → dense FFN, which would
    /// have produced wrong K/V for an evicted MoE window — the C1 follow-up.)
    pub fn replay_window(
        &self,
        weights: &ModelWeights,
        moe_ffn: Option<&dyn larql_inference::ffn::FfnBackend>,
        index: Option<&larql_vindex::VectorIndex>,
        window_id: usize,
    ) -> Result<(Vec<SharedKV>, usize), EngineError> {
        let (tokens, abs_offset) =
            self.archive
                .retrieve(window_id)
                .ok_or_else(|| EngineError::RetrievalMiss {
                    reason: format!("window {window_id} is not archived"),
                })?;

        let prior = if window_id > 0 && self.checkpoints.contains(window_id - 1) {
            let (ckpt, _) =
                self.checkpoints
                    .load(window_id - 1)
                    .ok_or_else(|| EngineError::RetrievalMiss {
                        reason: format!("checkpoint for window {} is missing", window_id - 1),
                    })?;
            ckpt
        } else {
            empty_prior(weights)
        };

        let mut kv_cache = prior;
        rs_extend_from_checkpoint_backend(
            larql_inference::WeightsView::dense(weights),
            tokens,
            &mut kv_cache,
            abs_offset,
            self.backend.as_ref(),
            moe_ffn,
            index,
        )?;
        let abs_end = abs_offset + tokens.len() - 1;
        Ok((kv_cache, abs_end))
    }

    /// Total storage and context statistics.
    pub fn stats(&self, weights: &ModelWeights) -> EngineStats {
        let arch = &*weights.arch;
        let num_layers = weights.num_layers;
        let kv_dim_sum: usize = (0..num_layers)
            .map(|l| arch.num_kv_heads_for_layer(l) * arch.head_dim_for_layer(l))
            .sum();

        let total_archived = self.archive.total_tokens();
        let current = self.current_window_tokens.len();
        let total_tokens = total_archived + current;

        let equivalent_kv_bytes = total_tokens * kv_dim_sum * 2 * 2;
        let checkpoint_bytes = self.checkpoints.total_bytes();
        let archive_bytes = self.archive.total_bytes();
        let total_boundary_bytes = checkpoint_bytes + archive_bytes;
        let compression_ratio = if total_boundary_bytes == 0 {
            0.0
        } else {
            equivalent_kv_bytes as f64 / total_boundary_bytes as f64
        };

        EngineStats {
            total_tokens,
            archived_windows: self.archive.len(),
            current_window_id: self.current_window_id,
            current_window_tokens: current,
            checkpoint_bytes,
            archive_bytes,
            total_boundary_bytes,
            equivalent_kv_bytes,
            compression_ratio,
        }
    }

    /// Quant-aware equivalent of `process()` — uses
    /// `rs_extend_from_checkpoint_quant` (WalkFfn for FFN; dispatches on
    /// the vindex's format) instead of the f32-backed
    /// `rs_extend_from_checkpoint_backend`.
    fn process_quant(
        &mut self,
        weights: &ModelWeights,
        index: &VectorIndex,
        tokens: &[u32],
        backend: &dyn ComputeBackend,
    ) -> Option<()> {
        let mut remaining = tokens;
        while !remaining.is_empty() {
            let free = self.window_size - self.current_window_tokens.len();
            let take = remaining.len().min(free);
            let (chunk, rest) = remaining.split_at(take);
            self.extend_current_quant(weights, index, chunk, backend)?;
            remaining = rest;
            if self.current_window_tokens.len() >= self.window_size {
                self.close_window();
            }
        }
        Some(())
    }

    fn extend_current_quant(
        &mut self,
        weights: &ModelWeights,
        index: &VectorIndex,
        chunk: &[u32],
        backend: &dyn ComputeBackend,
    ) -> Option<()> {
        if chunk.is_empty() {
            return Some(());
        }

        let prior = if self.current_window_tokens.is_empty() {
            if self.current_window_id > 0 && self.checkpoints.contains(self.current_window_id - 1) {
                let (ckpt, _) = self.checkpoints.load(self.current_window_id - 1)?;
                ckpt
            } else {
                empty_prior(weights)
            }
        } else {
            // Mid-window the shadow MUST exist; seeding from an empty
            // prior here would silently drop every in-window token from
            // attention. Fail upward (typed BackendFailure at the
            // KvEngine boundary) instead.
            self.current_window_kv.take()?
        };

        let abs_start = self.abs_offset + self.current_window_tokens.len();
        let prof = self.profiling.then_some(&mut self.profile);
        let view = larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch);
        let out =
            rs_extend_from_checkpoint_quant(view, index, chunk, prior, abs_start, backend, prof)?;

        self.last_hidden = Some(out.last_hidden);
        // CPU walk path returns narrow `[n, kv_dim]` arrays — counter
        // equals shape[0] here. Hot path (`decode_step_via_dispatch`)
        // will re-normalise to pre-allocated `[window_size, kv_dim]`
        // on the next prefill if needed; mixed-mode within a single
        // window isn't supported (and isn't reachable today since
        // `kv_handle` gates the two paths).
        self.current_window_kv_len = out.kv_cache.first().map_or(0, |(k, _)| k.shape()[0]);
        self.current_window_kv = Some(out.kv_cache);
        self.current_window_tokens.extend_from_slice(chunk);
        Some(())
    }

    fn current_kv_bytes(&self) -> usize {
        // W8: count only the logically valid rows. Buffers may be
        // pre-allocated `[window_size, kv_dim]` so `k.len()` overstates
        // by `(window_size - current_window_kv_len) * kv_dim`.
        let rows = self.current_window_kv_len;
        if rows == 0 {
            return 0;
        }
        self.current_window_kv.as_ref().map_or(0, |kv| {
            kv.iter()
                .map(|(k, v)| (k.shape()[1] + v.shape()[1]) * rows * 4)
                .sum()
        })
    }

    fn extend_current(
        &mut self,
        weights: &ModelWeights,
        chunk: &[u32],
        moe_ffn: Option<&dyn larql_inference::ffn::FfnBackend>,
        index: Option<&larql_vindex::VectorIndex>,
    ) -> Result<(), EngineError> {
        if chunk.is_empty() {
            return Ok(());
        }

        // `prior_len` is the prior's LOGICAL row count — the window-KV counter
        // mid-window, the checkpoint's row count at a window start, or 0.
        let (mut prior, prior_len) = if self.current_window_tokens.is_empty() {
            if self.current_window_id > 0 && self.checkpoints.contains(self.current_window_id - 1) {
                let id = self.current_window_id - 1;
                let (ckpt, _) =
                    self.checkpoints
                        .load(id)
                        .ok_or_else(|| EngineError::RetrievalMiss {
                            reason: format!("checkpoint for window {id} is missing"),
                        })?;
                let len = ckpt.first().map_or(0, |(k, _)| k.shape()[0]);
                (ckpt, len)
            } else {
                (empty_prior(weights), 0)
            }
        } else {
            // Mid-window the shadow MUST exist — see extend_current_quant.
            let shadow =
                self.current_window_kv
                    .take()
                    .ok_or_else(|| EngineError::InvariantViolation {
                        what: "mid-window extend with no K/V shadow".into(),
                    })?;
            (shadow, self.current_window_kv_len)
        };

        let abs_start = self.abs_offset + self.current_window_tokens.len();

        // In-place fast path: append the chunk's K/V rows into the window's
        // doubling-capacity buffers instead of rebuilding an owned `[len+1]`
        // concat every layer every step (O(window) → O(1) per step). Gated to
        // the Q4K-direct route (with the shared `LARQL_MARKOV_INPLACE_KV`
        // toggle); flags-off keeps the unchanged owned-concat path bit-for-bit,
        // which is what `resident_identity_tests` pins. The window's existing
        // `current_window_kv_len` counter already treats the buffers as
        // over-allocated (the dispatch path does too), so close_window /
        // current_kv_bytes need no change.
        let use_inplace = index.is_some()
            && crate::engines::markov_residual::compute::markov_inplace_kv_enabled()
            && larql_compute::options::q4k_direct_attn_enabled();

        // Both arms restore the shadow on failure, which is what makes a
        // refused chunk rewindable: `current_window_kv_len` and
        // `current_window_tokens` are advanced only after the extend returns,
        // so putting the buffers back at `prior_len` rows restores exactly the
        // window this call started from.
        let outcome = if use_inplace {
            // The in-place path only ever writes past `prior_len`, which the
            // counter never advanced past, so the logical window is already
            // intact — nothing to truncate.
            rs_extend_inplace(
                larql_inference::WeightsView::dense(weights),
                chunk,
                &mut prior,
                prior_len,
                abs_start,
                self.backend.as_ref(),
                moe_ffn,
                index,
            )
            .map(|last| (last, prior_len + chunk.len()))
        } else {
            rs_extend_from_checkpoint_backend(
                larql_inference::WeightsView::dense(weights),
                chunk,
                &mut prior,
                abs_start,
                self.backend.as_ref(),
                moe_ffn,
                index,
            )
            .map(|step| {
                // CPU walk path: narrow arrays, counter == shape[0].
                let rows = prior.first().map_or(0, |(k, _)| k.shape()[0]);
                (step.last_hidden, rows)
            })
            .inspect_err(|_| {
                // The owned-concat path replaces each layer's buffer as it
                // goes and reads a prior by `shape()[0]`, so a half-advanced
                // cache would attend over a token whose step never finished.
                truncate_kv_rows(&mut prior, prior_len);
            })
        };

        let (last_hidden, rows) = match outcome {
            Ok(pair) => pair,
            Err(failure) => {
                self.current_window_kv = Some(prior);
                return Err(failure);
            }
        };
        self.last_hidden = Some(last_hidden);
        self.current_window_kv_len = rows;
        self.current_window_kv = Some(prior);
        self.current_window_tokens.extend_from_slice(chunk);
        Ok(())
    }

    pub(super) fn close_window(&mut self) {
        // W10 Phase B: under HOnly the engine-side window shadow is
        // None; pull the last position's K/V back from the backend
        // (Metal kv cache) via KvDispatch::read_kv_row_at. Without
        // HOnly this branch never fires (kv is always Some) and we
        // slice the engine-side shadow as before.
        let n = self.current_window_kv_len;
        let window_len = self.current_window_tokens.len();
        // Absolute stream position of this window's last token — the value
        // recorded *with* the checkpoint, so a later replay knows where it
        // sat. It is no longer an index into anything: the dispatch path now
        // clips the backend handle to the window (issue #200), so the handle
        // holds this window's rows and not the stream's. The row to read back
        // is therefore its last one. Indexing the handle by absolute position
        // was correct only while the window was not being enforced.
        let abs_end = self.abs_offset + window_len - 1;
        let last_kv: Vec<SharedKV> = match self.current_window_kv.take() {
            Some(kv) => {
                if n == 0 {
                    Vec::new()
                } else {
                    // Shadow is window-local: its last logical row is
                    // `n - 1` regardless of how many windows preceded.
                    kv.iter()
                        .map(|(k, v)| {
                            let last_k = k.slice(ndarray::s![n - 1..n, ..]).to_owned();
                            let last_v = v.slice(ndarray::s![n - 1..n, ..]).to_owned();
                            (last_k, last_v)
                        })
                        .collect()
                }
            }
            None => {
                // No CPU shadow — engine ran under HOnly. Read the window's
                // last K/V back from the backend's kv cache. The handle is
                // clipped to the window, so its final row *is* this window's
                // last position; reading an absolute stream index here would
                // now run off the end. If there is no handle or the backend
                // lacks the readback affordance, fall through with an empty
                // checkpoint: the tokens are still archived and the counters
                // reset (a wedged window would otherwise spin `process()`
                // forever), and the mismatched empty checkpoint surfaces as an
                // extend error on the next window instead of silent loss.
                if n == 0 {
                    Vec::new()
                } else if let Some(handle) = self.kv_handle.as_ref() {
                    debug_assert_eq!(
                        n, window_len,
                        "HOnly window shadow counter out of sync with window tokens"
                    );
                    // Window-relative: the clipped handle's last row.
                    let last_row = handle.cached_len().saturating_sub(1);
                    let mut rows = Vec::new();
                    let mut layer = 0;
                    while let Some((k_row, v_row)) = self
                        .backend
                        .as_ref()
                        .read_kv_row_at(handle, layer, last_row)
                    {
                        let kv_dim = k_row.len();
                        let k = Array2::from_shape_vec((1, kv_dim), k_row)
                            .expect("read_kv_row_at returned mismatched length");
                        let v = Array2::from_shape_vec((1, kv_dim), v_row)
                            .expect("read_kv_row_at returned mismatched length");
                        rows.push((k, v));
                        layer += 1;
                    }
                    rows
                } else {
                    Vec::new()
                }
            }
        };
        self.current_window_kv_len = 0;

        self.checkpoints
            .save(self.current_window_id, last_kv, abs_end);
        self.archive.archive(
            self.current_window_id,
            std::mem::take(&mut self.current_window_tokens),
            self.abs_offset,
        );
        self.abs_offset += window_len;
        self.current_window_id += 1;
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl KvEngine for WindowedCheckpointEngine {
    fn name(&self) -> &str {
        "windowed-checkpoint"
    }

    fn info(&self) -> EngineInfo {
        let mem =
            self.checkpoints.total_bytes() + self.archive.total_bytes() + self.current_kv_bytes();
        EngineInfo {
            name: "windowed-checkpoint".into(),
            description: format!(
                "window-boundary KV checkpoints + token replay \
                 (windows={}, tokens={}, mem={:.1}MB)",
                self.archive.len(),
                self.archive.total_tokens() + self.current_window_tokens.len(),
                mem as f64 / 1_048_576.0,
            ),
            backend: self.backend.name().to_string(),
            config: format!("window={}", self.window_size),
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
        self.process(weights, token_ids, Some(ffn))?;
        self.last_hidden
            .clone()
            .ok_or_else(|| EngineError::BackendFailure {
                details: "last_hidden missing after prefill".into(),
            })
    }

    fn decode_step(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.process(weights, &[token_id], Some(ffn))?;
        self.last_hidden
            .clone()
            .ok_or_else(|| EngineError::BackendFailure {
                details: "last_hidden missing after decode_step".into(),
            })
    }

    /// Resident-path decode: threads `index` to the per-token attention
    /// steps' Q4K-direct route (the non-standard-engine structural-gap fix).
    fn decode_step_resident(
        &mut self,
        weights: &ModelWeights,
        ffn: &dyn FfnBackend,
        index: &larql_vindex::VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        self.process_with_index(weights, &[token_id], Some(ffn), Some(index))?;
        self.last_hidden
            .clone()
            .ok_or_else(|| EngineError::BackendFailure {
                details: "last_hidden missing after decode_step".into(),
            })
    }

    fn memory_bytes(&self) -> usize {
        // The last term covers the coarse dispatch path, where the live
        // window's K/V is held by the backend rather than in
        // `current_window_kv` — without it the hot tier reads as 0.0MB
        // and the ratio against a standard cache becomes unbounded.
        self.checkpoints.total_bytes()
            + self.archive.total_bytes()
            + self.current_kv_bytes()
            + self.kv_handle.as_ref().map_or(0, |h| h.resident_bytes())
            + self.backend.backend_resident_kv_bytes()
    }

    fn window_tokens(&self) -> usize {
        self.current_window_tokens.len()
    }

    fn cold_bytes(&self) -> usize {
        self.checkpoints.total_bytes() + self.archive.total_bytes()
    }

    fn dispatch_path(&self) -> Option<larql_inference::kv_engine::DispatchPath> {
        use larql_inference::kv_engine::DispatchPath;
        // `kv_handle` marks the coarse W1-GPU path; `current_window_kv`
        // is the per-layer shadow this engine keeps when it walks
        // layer-by-layer. Neither = nothing prefilled yet.
        match (self.kv_handle.is_some(), self.current_window_kv.is_some()) {
            (true, _) => Some(DispatchPath::Coarse),
            (false, true) => Some(DispatchPath::PerLayer),
            (false, false) => None,
        }
    }

    fn stage_summary(&self) -> Option<crate::DecodeStageSummary> {
        if !self.profiling || self.profile.decode_total.count == 0 {
            return None;
        }
        Some(
            self.profile
                .summary("windowed-checkpoint", self.backend.name()),
        )
    }

    /// Quant prefill — runs the windowed-checkpoint extension regardless
    /// of backend or vindex format. W1-GPU: tries `coarse_prefill_with_state`
    /// first; falls back to the legacy CPU per-layer walk when state
    /// capture isn't available. The engine's window-checkpoint
    /// contract is preserved either way: `current_window_kv` is built
    /// from captured per-layer state (W1-GPU) or computed via walk.
    fn prefill_quant(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_ids: &[u32],
        backend: &dyn ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        if let Some(hidden) = self.try_prefill_via_dispatch(weights, index, token_ids) {
            return Ok(hidden);
        }
        self.kv_handle = None;
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        self.process_quant(weights, index, token_ids, backend)
            .ok_or_else(|| EngineError::BackendFailure {
                details: "process_quant returned None during prefill_quant".into(),
            })?;
        self.last_hidden
            .clone()
            .ok_or_else(|| EngineError::BackendFailure {
                details: "last_hidden missing after prefill_quant".into(),
            })
    }

    fn decode_step_quant(
        &mut self,
        weights: &ModelWeights,
        _ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_id: u32,
        backend: &dyn ComputeBackend,
    ) -> Result<Array2<f32>, EngineError> {
        if self.kv_handle.is_some() {
            return self
                .decode_step_via_dispatch(weights, index, token_id)
                .ok_or_else(|| EngineError::BackendFailure {
                    details: "decode_step_via_dispatch returned None".into(),
                });
        }
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        self.process_quant(weights, index, &[token_id], backend)
            .ok_or_else(|| EngineError::BackendFailure {
                details: "process_quant returned None during decode_step_quant".into(),
            })?;
        self.last_hidden
            .clone()
            .ok_or_else(|| EngineError::BackendFailure {
                details: "last_hidden missing after decode_step_quant".into(),
            })
    }

    // ── Executor-aware migration (Phase 2 of engine-state-vs-execution spec) ──
    //
    // Drive the per-token layer loop through a caller-supplied `LayerExecutor`
    // and honor the caller-supplied `FfnBackend`. The legacy `*_quant` methods
    // construct their own `WalkFfn` and ignore the FFN parameter; remote-FFN
    // deployments (`larql bench --ffn http://shard:8080`) need this path so
    // the engine actually dispatches through the supplied backend.
    //
    // Window-close semantics (checkpoint + archive at window boundaries) are
    // identical to `process_quant` / `extend_current_quant` — the executor only
    // owns per-layer compute; window state is engine state.
    fn prefill_quant_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_ids: &[u32],
    ) -> Result<Array2<f32>, EngineError> {
        use larql_inference::layer_executor::ExecutorDispatchKind;
        if token_ids.is_empty() {
            return Err(EngineError::EmptyPrompt);
        }
        // Spec §3.4: this engine's state policy (windowed checkpoints) is
        // expressible against per-layer dispatch only. Transparent degrade
        // on fused executors until the Phase 3 refusal contract lands.
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.prefill_quant(weights, ffn, index, token_ids, executor.backend());
        }
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        self.process_via_executor(weights, executor, ffn, token_ids)
            .ok_or_else(|| EngineError::BackendFailure {
                details: "process_via_executor returned None during prefill_quant_via_executor"
                    .into(),
            })?;
        self.last_hidden
            .clone()
            .ok_or_else(|| EngineError::BackendFailure {
                details: "last_hidden missing after prefill_quant_via_executor".into(),
            })
    }

    fn decode_step_quant_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        index: &VectorIndex,
        token_id: u32,
    ) -> Result<Array2<f32>, EngineError> {
        use larql_inference::layer_executor::ExecutorDispatchKind;
        if matches!(executor.dispatch_kind(), ExecutorDispatchKind::Fused) {
            return self.decode_step_quant(weights, ffn, index, token_id, executor.backend());
        }
        ensure_attn_tensors_dequantised(&mut self.dequant_scratch, weights, index);
        self.process_via_executor(weights, executor, ffn, &[token_id])
            .ok_or_else(|| EngineError::BackendFailure {
                details: "process_via_executor returned None during decode_step_quant_via_executor"
                    .into(),
            })?;
        self.last_hidden
            .clone()
            .ok_or_else(|| EngineError::BackendFailure {
                details: "last_hidden missing after decode_step_quant_via_executor".into(),
            })
    }
}

// ── Executor-driven window extension ─────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
impl WindowedCheckpointEngine {
    /// Executor-aware analogue of `process_quant`: feeds tokens into the
    /// current window, auto-closes on fill, drives per-layer compute
    /// through `executor` instead of constructing a local `WalkFfn`.
    fn process_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        tokens: &[u32],
    ) -> Option<()> {
        let mut remaining = tokens;
        while !remaining.is_empty() {
            let free = self.window_size - self.current_window_tokens.len();
            let take = remaining.len().min(free);
            let (chunk, rest) = remaining.split_at(take);
            self.extend_current_via_executor(weights, executor, ffn, chunk)?;
            remaining = rest;
            if self.current_window_tokens.len() >= self.window_size {
                self.close_window();
            }
        }
        Some(())
    }

    fn extend_current_via_executor(
        &mut self,
        weights: &ModelWeights,
        executor: &dyn larql_inference::layer_executor::LayerExecutor,
        ffn: &dyn FfnBackend,
        chunk: &[u32],
    ) -> Option<()> {
        use larql_inference::forward::embed_tokens_pub;
        if chunk.is_empty() {
            return Some(());
        }

        let mut kv_cache: Vec<SharedKV> = if self.current_window_tokens.is_empty() {
            if self.current_window_id > 0 && self.checkpoints.contains(self.current_window_id - 1) {
                let (ckpt, _) = self.checkpoints.load(self.current_window_id - 1)?;
                ckpt
            } else {
                super::extend::empty_prior(weights)
            }
        } else {
            // Mid-window the shadow MUST exist — see extend_current_quant.
            self.current_window_kv.take()?
        };

        let num_layers = weights.num_layers;
        if kv_cache.len() != num_layers {
            return None;
        }
        let abs_start = self.abs_offset + self.current_window_tokens.len();
        let mut last_hidden: Option<Array2<f32>> = None;

        for (i, &token_id) in chunk.iter().enumerate() {
            let abs_position = abs_start + i;
            let mut h = embed_tokens_pub(weights, &[token_id]);
            // PLE inputs are per-token — this loop embeds one token at a
            // time, matching the legacy `kv_decode_step_run` recipe exactly.
            let ple_inputs = larql_inference::forward::ple::precompute_per_layer_inputs(
                weights,
                &h,
                &[token_id],
            );

            for (layer, kv_slot) in kv_cache.iter_mut().enumerate() {
                let (h_out, new_kv) = executor.run_decode_layer(
                    larql_inference::WeightsView::with_scratch(weights, &self.dequant_scratch),
                    layer,
                    &h,
                    kv_slot,
                    abs_position,
                    ffn,
                )?;
                // `LayerExecutor::run_decode_layer` returns attention + bare
                // FFN only (`LocalWalkExecutor`, the sole production impl,
                // ends at `run_ffn`); the PLE + layer_scalar tail is the
                // driving loop's responsibility, mirroring the legacy
                // `kv_decode_step_run` sequence.
                h = crate::engines::apply_ple_and_layer_scalar(
                    weights,
                    &h_out,
                    layer,
                    ple_inputs.get(layer),
                );
                *kv_slot = new_kv;
            }
            last_hidden = Some(h);
        }

        self.last_hidden = last_hidden;
        // CPU walk path via executor: kv_cache is narrow arrays.
        self.current_window_kv_len = kv_cache.first().map_or(0, |(k, _)| k.shape()[0]);
        self.current_window_kv = Some(kv_cache);
        self.current_window_tokens.extend_from_slice(chunk);
        Some(())
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_engine_is_empty() {
        let eng = WindowedCheckpointEngine::new(512);
        assert_eq!(eng.window_size, 512);
        assert_eq!(eng.archive.len(), 0);
        assert_eq!(eng.checkpoints.len(), 0);
        assert_eq!(eng.current_window_id, 0);
        assert_eq!(eng.memory_bytes(), 0);
    }

    #[test]
    fn engine_info_backend_is_cpu() {
        let eng = WindowedCheckpointEngine::new(256);
        let info = eng.info();
        assert_eq!(info.name, "windowed-checkpoint");
        assert!(
            info.backend.starts_with("cpu"),
            "expected cpu backend, got {:?}",
            info.backend
        );
        assert_eq!(info.config, "window=256");
        assert!(info.summary().contains("windowed-checkpoint"));
        assert!(info.summary().contains("cpu"));
    }

    #[test]
    fn engine_info_config_contains_window_size() {
        let eng = WindowedCheckpointEngine::new(1024);
        assert!(eng.info().config.contains("1024"));
    }

    #[test]
    fn window_tokens_and_cold_bytes_start_zero() {
        let eng = WindowedCheckpointEngine::new(512);
        assert_eq!(eng.window_tokens(), 0);
        assert_eq!(eng.cold_bytes(), 0);
    }

    // ── prefill / decode cycle ─────────────────────────────────────────────────

    #[test]
    fn prefill_returns_hidden_state() {
        use larql_inference::ffn::WeightFfn;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = WindowedCheckpointEngine::new(512);
        let h = engine
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .expect("prefill failed");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(
            h.iter().all(|v| v.is_finite()),
            "hidden state should be finite"
        );
    }

    #[test]
    fn decode_step_returns_hidden_state() {
        use larql_inference::ffn::WeightFfn;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = WindowedCheckpointEngine::new(512);
        engine.prefill(&weights, &ffn, &[0u32]).expect("prefill");
        let h = engine.decode_step(&weights, &ffn, 1).expect("decode_step");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn window_auto_closes_when_full() {
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let window_size = 3usize;
        let mut engine = WindowedCheckpointEngine::new(window_size);

        // Feed exactly window_size tokens → triggers close
        for tok in 0..window_size as u32 {
            engine
                .process(&weights, &[tok], None)
                .expect("process failed");
        }
        assert_eq!(engine.archive.len(), 1, "one window should be archived");
        assert_eq!(
            engine.current_window_tokens.len(),
            0,
            "current window should be empty"
        );
        assert_eq!(
            engine.checkpoints.len(),
            1,
            "one checkpoint should be saved"
        );
    }

    #[test]
    fn two_full_windows_archives_two() {
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let mut engine = WindowedCheckpointEngine::new(2);

        // 4 tokens = 2 complete windows
        for tok in 0u32..4 {
            engine.process(&weights, &[tok], None).expect("process");
        }
        assert_eq!(engine.archive.len(), 2);
        assert_eq!(engine.checkpoints.len(), 2);
    }

    #[test]
    fn partial_window_after_process() {
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let mut engine = WindowedCheckpointEngine::new(4);

        // 3 tokens < window_size=4 → no close
        engine
            .process(&weights, &[0u32, 1, 2], None)
            .expect("process");
        assert_eq!(engine.archive.len(), 0, "no window closed yet");
        assert_eq!(engine.window_tokens(), 3);
    }

    #[test]
    fn flush_closes_partial_window() {
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let mut engine = WindowedCheckpointEngine::new(4);
        engine.process(&weights, &[0u32, 1], None).expect("process");
        assert_eq!(engine.archive.len(), 0);
        engine.flush();
        assert_eq!(engine.archive.len(), 1, "flush should close partial window");
    }

    #[test]
    fn cold_bytes_grow_after_window_close() {
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let mut engine = WindowedCheckpointEngine::new(2);
        assert_eq!(engine.cold_bytes(), 0);
        engine.process(&weights, &[0u32, 1], None).expect("process"); // closes window
        assert!(
            engine.cold_bytes() > 0,
            "cold tier should grow after window close"
        );
    }

    #[test]
    fn memory_bytes_nonzero_after_prefill() {
        use larql_inference::ffn::WeightFfn;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = WindowedCheckpointEngine::new(512);
        assert_eq!(engine.memory_bytes(), 0);
        engine
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .expect("prefill");
        assert!(engine.memory_bytes() > 0);
    }

    #[test]
    fn logits_from_windowed_checkpoint_are_finite() {
        use larql_inference::ffn::WeightFfn;
        use larql_inference::forward::hidden_to_raw_logits;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = WindowedCheckpointEngine::new(512);
        let h = engine.prefill(&weights, &ffn, &[0u32, 1]).expect("prefill");
        let logits = hidden_to_raw_logits(&weights, &h);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "logits should be finite"
        );
    }

    // ── Q4K paths via Q4K fixture ─────────────────────────────────────────
    //
    // `prefill_quant` first tries `fused_prefill` (Metal fast path); on
    // CPU that returns None (no fused decode kernel), so we fall through
    // to the dequant + cached-decode path. The Q4K fixture has the attn
    // Q4K slices the dequant step needs.

    #[test]
    fn prefill_quant_cpu_runs_via_dequant_path() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = WindowedCheckpointEngine::new(512);
        let h = engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1, 2], &*backend)
            .expect("prefill_quant Q4K cpu fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn decode_step_quant_cpu_extends_state() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = WindowedCheckpointEngine::new(512);
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill_quant");
        let h = engine
            .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
            .expect("decode_step_quant Q4K cpu fallback");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// Flags-ON parity gate for the in-place window K/V fast path: an A/B of the
    /// in-place steady state vs the owned-concat reference, both driving the
    /// resident decode path (`extend_current`) with Q4K-direct attention live.
    /// The two must produce bit-identical hidden states every step — the
    /// in-place append only changes the window-buffer representation (doubling +
    /// views vs fresh owned concat). 13 tokens < window(512), so it stays in one
    /// window (no close). Serialised on `Q4K_FLAG_ENV_LOCK`; path selected via
    /// the shared `LARQL_MARKOV_INPLACE_KV` thread-local override.
    #[test]
    fn decode_inplace_matches_owned_concat_flags_on() {
        use crate::engines::markov_residual::compute::set_markov_env_override;
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};

        let _q4k = crate::engines::Q4kFlagGuard::set(&[
            (larql_compute::options::ENV_Q4K_DIRECT_ATTN, true),
            (larql_compute::options::ENV_Q4K_ATTN_INT8, false),
        ]);

        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let ffn = NullFfn;

        let run = |inplace: bool| -> Vec<Vec<u32>> {
            set_markov_env_override(
                "LARQL_MARKOV_INPLACE_KV",
                Some(if inplace { "1" } else { "0" }),
            );
            let mut engine = WindowedCheckpointEngine::new(512);
            engine
                .prefill(&weights, &ffn, &[0u32, 1, 2])
                .expect("prefill");
            let mut hiddens = Vec::new();
            for tok in 3u32..=12 {
                let h = engine
                    .decode_step_resident(&weights, &ffn, &index, tok)
                    .expect("decode_step_resident");
                assert!(h.iter().all(|v| v.is_finite()));
                hiddens.push(h.iter().map(|v| v.to_bits()).collect());
            }
            hiddens
        };

        let a = run(true);
        let b = run(false);
        assert_eq!(
            a, b,
            "unlimited in-place vs owned-concat hidden states diverged (q4k on)"
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
        let mut engine = WindowedCheckpointEngine::new(512);
        // No prefill → decode falls through fast-path checks and returns None
        // (or some empty hidden) without panicking.
        let _ = engine.decode_step_quant(&weights, &ffn, &index, 0, &*backend);
    }

    // ── Public utility methods (stats, replay_window, summary) ────────────

    #[test]
    fn engine_stats_summary_includes_archived_and_compression() {
        use larql_inference::ffn::WeightFfn;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = WindowedCheckpointEngine::new(512);
        engine
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .expect("prefill");
        let stats = engine.stats(&weights);
        assert!(stats.total_tokens >= 3);
        // EngineStats::summary builds a one-line string that includes
        // window count and token count.
        let s = stats.summary();
        assert!(s.contains("windows"));
        assert!(s.contains("tokens"));
    }

    #[test]
    fn engine_stats_with_empty_engine_handles_zero_division() {
        let weights = larql_inference::test_utils::make_test_weights();
        let engine = WindowedCheckpointEngine::new(512);
        let stats = engine.stats(&weights);
        // No prefill → all counters zero, compression ratio short-circuits
        // to 0.0 (no division by zero).
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.archived_windows, 0);
        assert!(
            stats.compression_ratio == 0.0,
            "compression should be 0 when no boundary bytes archived"
        );
        // Summary still produces a string for the empty case.
        let _ = stats.summary();
    }

    #[test]
    fn replay_window_returns_none_for_missing_window() {
        let weights = larql_inference::test_utils::make_test_weights();
        let engine = WindowedCheckpointEngine::new(512);
        // No windows archived → any window_id returns None at the
        // `self.archive.retrieve(window_id)?` line.
        assert!(engine.replay_window(&weights, None, None, 0).is_err());
        assert!(engine.replay_window(&weights, None, None, 99).is_err());
    }

    #[test]
    fn replay_window_succeeds_after_window_overflow() {
        use larql_inference::ffn::WeightFfn;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        // window=2; prefill 4 tokens → archives at least 1 window.
        let mut engine = WindowedCheckpointEngine::new(2);
        engine
            .prefill(&weights, &ffn, &[0u32, 1, 2, 3])
            .expect("prefill 4 tokens");
        let stats = engine.stats(&weights);
        assert!(
            stats.archived_windows >= 1,
            "expected at least 1 archived window after overflow, got {}",
            stats.archived_windows
        );
        // Replay the first archived window — exercises the
        // `rs_extend_from_checkpoint_backend` path (lines 132-138).
        let replay = engine.replay_window(&weights, None, None, 0);
        assert!(replay.is_ok(), "replay_window(0) should succeed");
        let (kv, abs_end) = replay.unwrap();
        assert!(!kv.is_empty(), "replayed K/V cache should be non-empty");
        assert!(
            abs_end < 4,
            "abs_end {abs_end} should be within the prefill"
        );
    }

    // ── Phase 2: executor-driven path ─────────────────────────────────────

    #[test]
    fn prefill_quant_via_executor_runs_through_local_walk() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = WindowedCheckpointEngine::new(512);
        let h = engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2])
            .expect("executor prefill");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(engine.memory_bytes() > 0);
    }

    #[test]
    fn decode_step_quant_via_executor_extends_state() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        let mut engine = WindowedCheckpointEngine::new(512);
        engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1])
            .expect("prefill");
        let h = engine
            .decode_step_quant_via_executor(&weights, &executor, &ffn, &index, 2)
            .expect("decode");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    /// Drive `rs_extend_from_checkpoint_quant`'s `Some(profiler)` arms
    /// — covers the per-stage `if timing { ... }` blocks and the
    /// profiler accumulator at the end of the function.
    #[test]
    fn process_quant_with_profiling_populates_summary() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = WindowedCheckpointEngine::new(512).with_profiling(true);
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill");
        engine
            .decode_step_quant(&weights, &ffn, &index, 2, &*backend)
            .expect("decode");
        let summary = engine
            .stage_summary()
            .expect("windowed_checkpoint profiler should populate summary");
        assert_eq!(summary.engine, "windowed-checkpoint");
        assert!(summary.steps >= 1);
        assert!(summary.avg_attention_us > 0.0);
        assert!(summary.avg_ffn_us > 0.0);
        assert!(summary.avg_total_decode_us > 0.0);
    }

    /// Counting FFN that records every `forward` call. Proves the executor
    /// path actually dispatches through the caller's `FfnBackend` instead
    /// of constructing a local `WalkFfn` (the legacy coupling the migration
    /// removes).
    struct CountingFfn {
        calls: std::sync::atomic::AtomicUsize,
        hidden: usize,
    }
    impl larql_inference::ffn::FfnBackend for CountingFfn {
        fn forward(&self, _layer: usize, x: &ndarray::Array2<f32>) -> ndarray::Array2<f32> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            ndarray::Array2::zeros((x.shape()[0], self.hidden))
        }
        fn name(&self) -> &str {
            "counting"
        }
    }

    #[test]
    fn executor_path_honors_ffn_parameter() {
        use larql_inference::layer_executor::LocalWalkExecutor;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);

        let ffn = CountingFfn {
            calls: std::sync::atomic::AtomicUsize::new(0),
            hidden: weights.hidden_size,
        };
        let mut engine = WindowedCheckpointEngine::new(512);
        engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2])
            .expect("prefill via executor");

        let call_count = ffn.calls.load(std::sync::atomic::Ordering::SeqCst);
        // 3 tokens × num_layers — one FFN dispatch per (token, layer)
        // because the engine's per-token loop runs every layer through
        // `run_decode_layer`, which in turn invokes the caller's FFN.
        let expected = 3 * weights.num_layers;
        assert_eq!(
            call_count, expected,
            "executor path should dispatch FFN through the supplied backend \
             once per (token, layer); got {call_count} for {expected} \
             expected — engine is likely constructing its own FFN internally",
        );
    }

    // ── window_size validation ────────────────────────────────────────────

    #[test]
    #[should_panic(expected = "window_size must be >= 1")]
    fn zero_window_size_is_rejected_at_construction() {
        let _ = WindowedCheckpointEngine::new(0);
    }

    #[test]
    fn window_size_one_is_legal() {
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let mut engine = WindowedCheckpointEngine::new(1);
        engine.process(&weights, &[0u32, 1], None).expect("process");
        assert_eq!(
            engine.archive.len(),
            2,
            "window_size=1 closes one window per token"
        );
    }

    // ── degenerate close / lost-shadow states ─────────────────────────────

    /// A full window with neither a CPU shadow nor a backend handle is
    /// unrecoverable K/V-wise, but the close must still archive the
    /// tokens and reset the counters — pre-fix it early-returned with
    /// the window intact, and `process()` spun forever re-trying the
    /// close with zero free slots.
    #[test]
    fn close_window_without_shadow_or_handle_recovers_bookkeeping() {
        let mut engine = WindowedCheckpointEngine::new(2);
        engine.current_window_tokens = vec![7, 8];
        engine.current_window_kv = None;
        engine.current_window_kv_len = 2;
        engine.close_window();
        assert_eq!(engine.archive.len(), 1, "tokens archived");
        assert!(engine.current_window_tokens.is_empty(), "window reset");
        assert_eq!(engine.abs_offset, 2);
        assert_eq!(engine.current_window_id, 1);
        let (ckpt, abs_end) = engine.checkpoints.load(0).expect("checkpoint entry");
        assert!(
            ckpt.is_empty(),
            "unrecoverable K/V must yield an empty checkpoint, not stale rows"
        );
        assert_eq!(abs_end, 1);
        let (tokens, abs_start) = engine.archive.retrieve(0).expect("archived tokens");
        assert_eq!(tokens, &[7, 8]);
        assert_eq!(abs_start, 0);
    }

    /// Mid-window decode with the shadow lost must surface an error —
    /// pre-fix it silently seeded attention from an empty prior,
    /// dropping every in-window token from the context.
    #[test]
    fn decode_with_lost_window_shadow_errors_instead_of_dropping_context() {
        use larql_inference::ffn::WeightFfn;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let mut engine = WindowedCheckpointEngine::new(8);
        engine
            .prefill(&weights, &ffn, &[0u32, 1, 2])
            .expect("prefill");
        engine.current_window_kv = None;
        let res = engine.decode_step(&weights, &ffn, 3);
        assert!(
            res.is_err(),
            "mid-window decode without the window shadow must error"
        );
    }

    /// Same contract on the quant walk path (`extend_current_quant`).
    #[test]
    fn decode_step_quant_with_lost_window_shadow_errors() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::test_utils::{make_test_q4k_vindex, make_test_q4k_weights};
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let ffn = NullFfn;
        let mut engine = WindowedCheckpointEngine::new(512);
        engine
            .prefill_quant(&weights, &ffn, &index, &[0u32, 1], &*backend)
            .expect("prefill");
        // Simulate a failed dispatch step: handle dropped, shadow gone,
        // window tokens still present.
        engine.kv_handle = None;
        engine.current_window_kv = None;
        let res = engine.decode_step_quant(&weights, &ffn, &index, 2, &*backend);
        assert!(
            res.is_err(),
            "quant decode without the window shadow must error"
        );
    }

    #[test]
    fn prefill_quant_via_executor_with_small_window_archives() {
        use larql_inference::ffn::NullFfn;
        use larql_inference::layer_executor::LocalWalkExecutor;
        use larql_inference::test_utils::make_test_weights;
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let executor = LocalWalkExecutor::new(&*backend);
        let ffn = NullFfn;
        // window=2, 4 tokens → triggers two window-close cycles via
        // `process_via_executor`. Exercises the prior-checkpoint-load
        // branch in `extend_current_via_executor`.
        let mut engine = WindowedCheckpointEngine::new(2);
        engine
            .prefill_quant_via_executor(&weights, &executor, &ffn, &index, &[0u32, 1, 2, 3])
            .expect("prefill 4 tokens through executor");
        let stats = engine.stats(&weights);
        assert!(
            stats.archived_windows >= 1,
            "expected at least 1 archived window, got {}",
            stats.archived_windows
        );
    }
}
