//! Top-level `UnlimitedContextEngine` — Rust port of
//! `chuk-mlx/src/chuk_lazarus/inference/context/research/unlimited_engine.py`.
//!
//! Window lifecycle:
//!   1. `process(tokens)` — extends active window's K,V via
//!      `rs_extend_from_checkpoint`. When window fills, auto-closes.
//!   2. `close_window()` — saves last-position K,V to `CheckpointStore`,
//!      appends token IDs to `TokenArchive`, resets active window.
//!   3. `replay_window(id)` — reconstructs a window's full K,V by running
//!      a forward pass over the archived tokens from the prior checkpoint.
//!   4. `stats()` — total bytes, windows, compression ratio vs full KV.

use larql_inference::attention::SharedKV;
use larql_inference::model::ModelWeights;
use serde::Serialize;

use super::checkpoint_store::CheckpointStore;
use super::extend::{empty_prior, rs_extend_from_checkpoint};
use super::token_archive::TokenArchive;

/// Storage and context statistics for `UnlimitedContextEngine`.
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
            self.archived_windows, self.total_tokens, self.compression_ratio
        )
    }
}

pub struct UnlimitedContextEngine {
    pub window_size: usize,
    pub checkpoints: CheckpointStore,
    pub archive: TokenArchive,

    current_window_id: usize,
    current_window_tokens: Vec<u32>,
    current_window_kv: Option<Vec<SharedKV>>,
    abs_offset: usize,
}

impl UnlimitedContextEngine {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            checkpoints: CheckpointStore::new(),
            archive: TokenArchive::new(),
            current_window_id: 0,
            current_window_tokens: Vec::new(),
            current_window_kv: None,
            abs_offset: 0,
        }
    }

    /// Feed tokens into the engine. Windows auto-close when they fill.
    ///
    /// Processes in chunks that fit within the current window; whenever the
    /// current window is exactly `window_size` tokens, closes it (saves
    /// checkpoint + archives tokens) and starts a new window.
    pub fn process(&mut self, weights: &ModelWeights, tokens: &[u32]) -> Option<()> {
        let mut remaining = tokens;
        while !remaining.is_empty() {
            let free = self.window_size - self.current_window_tokens.len();
            let take = remaining.len().min(free);
            let (chunk, rest) = remaining.split_at(take);
            self.extend_current(weights, chunk)?;
            remaining = rest;
            if self.current_window_tokens.len() >= self.window_size {
                self.close_window();
            }
        }
        Some(())
    }

    /// Close any partial current window. Call before replay if the current
    /// window hasn't filled naturally.
    pub fn flush(&mut self) {
        if !self.current_window_tokens.is_empty() {
            self.close_window();
        }
    }

    /// Reconstruct a window's full K,V by replaying its archived tokens
    /// from the prior window's boundary checkpoint.
    ///
    /// Returns `(kv_per_layer, abs_end)` where `kv_per_layer[l]` has shape
    /// `(prior_len + |w|, num_kv × head_dim)` and `abs_end` is the
    /// absolute position of the last token in this window.
    ///
    /// For `window_id == 0` (no prior), runs a fresh prefill — bit-exact
    /// with the original processing. For `window_id > 0`, starts from the
    /// saved 1-token checkpoint of the previous window — within-window K,V
    /// are produced by the actual forward pass; the 1-token prior summary
    /// is the only cross-window approximation.
    pub fn replay_window(
        &self,
        weights: &ModelWeights,
        window_id: usize,
    ) -> Option<(Vec<SharedKV>, usize)> {
        let (tokens, abs_offset) = self.archive.retrieve(window_id)?;

        let prior = if window_id > 0 && self.checkpoints.contains(window_id - 1) {
            let (ckpt, _) = self.checkpoints.load(window_id - 1)?;
            ckpt
        } else {
            empty_prior(weights)
        };

        let out = rs_extend_from_checkpoint(weights, tokens, &prior, abs_offset)?;
        let abs_end = abs_offset + tokens.len() - 1;
        Some((out.kv_cache, abs_end))
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

        // Standard KV reference: bf16 (2 bytes per K and V entry)
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

    // ------------------------------------------------------------------
    // internals
    // ------------------------------------------------------------------

    fn extend_current(&mut self, weights: &ModelWeights, chunk: &[u32]) -> Option<()> {
        if chunk.is_empty() {
            return Some(());
        }

        // Seed with prior window's checkpoint on first extend of a new window,
        // or continue from whatever K,V the active window has accumulated.
        let prior = if self.current_window_tokens.is_empty() {
            if self.current_window_id > 0 && self.checkpoints.contains(self.current_window_id - 1)
            {
                let (ckpt, _) = self.checkpoints.load(self.current_window_id - 1)?;
                ckpt
            } else {
                empty_prior(weights)
            }
        } else {
            self.current_window_kv
                .take()
                .unwrap_or_else(|| empty_prior(weights))
        };

        let abs_start = self.abs_offset + self.current_window_tokens.len();
        let out = rs_extend_from_checkpoint(weights, chunk, &prior, abs_start)?;

        self.current_window_kv = Some(out.kv_cache);
        self.current_window_tokens.extend_from_slice(chunk);
        Some(())
    }

    fn close_window(&mut self) {
        let kv = match self.current_window_kv.take() {
            Some(kv) => kv,
            None => return,
        };

        // Extract last-position K,V per layer = next boundary checkpoint.
        let last_kv: Vec<SharedKV> = kv
            .iter()
            .map(|(k, v)| {
                let n = k.shape()[0];
                let last_k = k.slice(ndarray::s![n - 1..n, ..]).to_owned();
                let last_v = v.slice(ndarray::s![n - 1..n, ..]).to_owned();
                (last_k, last_v)
            })
            .collect();

        let window_len = self.current_window_tokens.len();
        let abs_end = self.abs_offset + window_len - 1;

        self.checkpoints.save(self.current_window_id, last_kv, abs_end);
        self.archive.archive(
            self.current_window_id,
            std::mem::take(&mut self.current_window_tokens),
            self.abs_offset,
        );
        self.abs_offset += window_len;
        self.current_window_id += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Engine construction + storage accounting without running a model.
    #[test]
    fn new_engine_is_empty() {
        let eng = UnlimitedContextEngine::new(512);
        assert_eq!(eng.window_size, 512);
        assert_eq!(eng.archive.len(), 0);
        assert_eq!(eng.checkpoints.len(), 0);
        assert_eq!(eng.current_window_id, 0);
    }
}
