//! `RsStoreCodec` — `RsStore` with a codec-encoded cold tier.

use larql_inference::attention::SharedKV;
use ndarray::{s, Array2};

use crate::engines::markov_residual_codec::codec::{decode_block, encode_block, ColdResidualCodec};

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Per-layer encoded cold residuals.
#[derive(Debug, Clone)]
pub struct EncodedColdLayer {
    /// Number of cold positions stored.
    pub n_positions: usize,
    /// Hidden size (constant per layer).
    pub hidden_size: usize,
    /// Encoded payload bytes for `n_positions × hidden_size` elements.
    pub payload: Vec<u8>,
}

impl EncodedColdLayer {
    pub fn empty(hidden_size: usize) -> Self {
        Self {
            n_positions: 0,
            hidden_size,
            payload: Vec::new(),
        }
    }

    /// Append `block` (which must have the same `hidden_size`) to the existing
    /// encoded payload. The codec is applied to `block` once on append.
    pub fn append(&mut self, codec: ColdResidualCodec, block: &Array2<f32>) {
        let cols = block.shape()[1];
        let rows = block.shape()[0];
        assert_eq!(
            cols, self.hidden_size,
            "EncodedColdLayer hidden_size mismatch (have {}, got {cols})",
            self.hidden_size
        );
        if rows == 0 {
            return;
        }
        let block_bytes = encode_block(codec, block);
        self.payload.extend_from_slice(&block_bytes);
        self.n_positions += rows;
    }

    /// Decode the layer back to a 2-D `f32` block.
    pub fn decode(&self, codec: ColdResidualCodec) -> Array2<f32> {
        decode_block(codec, &self.payload, self.n_positions, self.hidden_size)
    }
}

/// `RsStoreCodec` — per-layer hot residuals (f32) + per-layer codec-encoded
/// cold residuals. Mirrors `RsStore` from the `markov_residual` engine, with
/// the cold tier swapped for a byte-packed representation. `hot_kv`
/// caches the K/V projection of the hot tier across decode steps
/// (W2; see `RsStore` doc for invariants).
pub struct RsStoreCodec {
    /// Per-layer residual stream. W8.2: possibly over-allocated; valid
    /// row count is `hot_len`, not `stored[l].shape()[0]`. See the
    /// mirror [`crate::engines::markov_residual::store::RsStore`] doc
    /// for the doubling-capacity contract.
    pub stored: Vec<Array2<f32>>,
    pub cold_encoded: Option<Vec<EncodedColdLayer>>,
    pub cold_kv: Option<Vec<SharedKV>>,
    /// Per-layer hot K/V. Same over-allocation rule as `stored`.
    pub hot_kv: Option<Vec<SharedKV>>,
    pub cold_abs_start: usize,
    pub next_position: usize,
    pub max_window: Option<usize>,
    pub codec: ColdResidualCodec,
    /// W8.2: logical row count for `stored` / `hot_kv`. See field doc
    /// on `stored`.
    pub hot_len: usize,
}

impl RsStoreCodec {
    pub fn memory_bytes(&self) -> usize {
        // W8.2: count only logically valid rows (hot_len), not the
        // pre-allocated capacity.
        let rows = self.hot_len;
        let hot: usize = self.stored.iter().map(|s| rows * s.shape()[1] * 4).sum();
        let cold_enc: usize = self
            .cold_encoded
            .as_ref()
            .map(|layers| layers.iter().map(|l| l.payload.len()).sum())
            .unwrap_or(0);
        let cold_kv: usize = self
            .cold_kv
            .as_ref()
            .map(|kv| kv.iter().map(|(k, v)| (k.len() + v.len()) * 4).sum())
            .unwrap_or(0);
        let hot_kv: usize = self
            .hot_kv
            .as_ref()
            .map(|kv| {
                kv.iter()
                    .map(|(k, v)| (k.shape()[1] + v.shape()[1]) * rows * 4)
                    .sum()
            })
            .unwrap_or(0);
        hot + cold_enc + cold_kv + hot_kv
    }

    pub fn cold_bytes(&self) -> usize {
        let cold_enc: usize = self
            .cold_encoded
            .as_ref()
            .map(|layers| layers.iter().map(|l| l.payload.len()).sum())
            .unwrap_or(0);
        let cold_kv: usize = self
            .cold_kv
            .as_ref()
            .map(|kv| kv.iter().map(|(k, v)| (k.len() + v.len()) * 4).sum())
            .unwrap_or(0);
        cold_enc + cold_kv
    }

    pub fn window_tokens(&self) -> usize {
        // W8.2: use the logical-length counter.
        self.hot_len
    }

    /// Clip the hot tier for `layer` against `max_window`. Returns the
    /// overflow as an `f32` block (the caller is responsible for encoding it
    /// onto the cold tier). Also clips `hot_kv` consistently when
    /// present so the K/V cache stays aligned with the (smaller)
    /// hot residual buffer.
    pub(crate) fn clip_layer_overflow(&mut self, layer: usize) -> Array2<f32> {
        let window = match self.max_window {
            Some(w) => w,
            None => return Array2::zeros((0, self.stored[layer].shape()[1])),
        };
        // W8.2: use logical row count, not pre-allocated capacity.
        let rows = self.hot_len;
        let cols = self.stored[layer].shape()[1];
        if rows <= window {
            return Array2::zeros((0, cols));
        }
        let start = rows - window;
        let s_logical = self.stored[layer].slice(s![..rows, ..]);
        let overflow = s_logical.slice(s![..start, ..]).to_owned();
        self.stored[layer] = s_logical.slice(s![start.., ..]).to_owned();
        // Same `start..` slice keeps hot_kv aligned with stored. The
        // evicted top rows are absorbed into cold_kv by the caller
        // via `snapshot_evicted_hot_kv` (see `rs_decode_step_codec_walk`
        // for the merge-into-cold flow).
        if let Some(kv) = self.hot_kv.as_mut() {
            let (k, v) = &kv[layer];
            let k_logical = k.slice(s![..rows, ..]);
            let v_logical = v.slice(s![..rows, ..]);
            kv[layer] = (
                k_logical.slice(s![start.., ..]).to_owned(),
                v_logical.slice(s![start.., ..]).to_owned(),
            );
        }
        // NB: do NOT update `self.hot_len` here — see RsStore::clip_layer
        // for rationale. Callers must reset via
        // `finalise_hot_len_after_clip()` after the per-layer loop.
        overflow
    }

    /// Reset the logical row count after a window-clip loop. Call once
    /// after `clip_layer_overflow` has been invoked for every layer.
    pub(crate) fn finalise_hot_len_after_clip(&mut self) {
        if let Some(w) = self.max_window {
            self.hot_len = self.hot_len.min(w);
        }
    }

    // NOTE: Unlike [`super::super::markov_residual::store::RsStore`],
    // the codec engine does *not* expose `snapshot_evicted_hot_kv`.
    // Its cold tier is codec-encoded (lossy under e.g. bf16), so the
    // evicted raw K/V diverges from what would be recomputed against
    // the round-tripped cold residual; we always invalidate cold_kv
    // on overflow and let the next step recompute against the
    // codec-decoded residual.
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store(num_layers: usize, seq_len: usize, hidden: usize) -> RsStoreCodec {
        let stored = (0..num_layers)
            .map(|_| Array2::from_elem((seq_len, hidden), 1.0f32))
            .collect();
        RsStoreCodec {
            stored,
            cold_encoded: None,
            cold_kv: None,
            hot_kv: None,
            cold_abs_start: 0,
            next_position: seq_len,
            max_window: None,
            codec: ColdResidualCodec::Bf16,
            hot_len: seq_len,
        }
    }

    #[test]
    fn encoded_layer_empty_starts_at_zero() {
        let l = EncodedColdLayer::empty(16);
        assert_eq!(l.n_positions, 0);
        assert_eq!(l.hidden_size, 16);
        assert!(l.payload.is_empty());
    }

    #[test]
    fn append_block_grows_payload_and_count() {
        let mut l = EncodedColdLayer::empty(4);
        let block = Array2::<f32>::ones((3, 4));
        l.append(ColdResidualCodec::Bf16, &block);
        assert_eq!(l.n_positions, 3);
        // bf16 = 2 bytes per element.
        assert_eq!(l.payload.len(), 3 * 4 * 2);
    }

    #[test]
    fn append_then_decode_roundtrips() {
        let mut l = EncodedColdLayer::empty(2);
        let block = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        l.append(ColdResidualCodec::Bf16, &block);
        let dec = l.decode(ColdResidualCodec::Bf16);
        assert_eq!(dec.shape(), &[2, 2]);
        for (orig, got) in block.iter().zip(dec.iter()) {
            assert!((orig - got).abs() < 0.1);
        }
    }

    #[test]
    fn append_empty_block_is_noop() {
        let mut l = EncodedColdLayer::empty(4);
        let block: Array2<f32> = Array2::zeros((0, 4));
        l.append(ColdResidualCodec::Bf16, &block);
        assert_eq!(l.n_positions, 0);
        assert!(l.payload.is_empty());
    }

    #[test]
    #[should_panic(expected = "hidden_size mismatch")]
    fn append_wrong_hidden_size_panics() {
        let mut l = EncodedColdLayer::empty(4);
        let block: Array2<f32> = Array2::zeros((1, 5)); // wrong hidden
        l.append(ColdResidualCodec::Bf16, &block);
    }

    // ── RsStoreCodec ──────────────────────────────────────────────────────────

    #[test]
    fn memory_bytes_hot_only() {
        let s = make_store(2, 3, 8);
        assert_eq!(s.memory_bytes(), 2 * 3 * 8 * 4);
    }

    #[test]
    fn cold_bytes_zero_when_no_cold() {
        let s = make_store(1, 3, 8);
        assert_eq!(s.cold_bytes(), 0);
    }

    #[test]
    fn window_tokens_matches_stored() {
        let s = make_store(2, 5, 4);
        assert_eq!(s.window_tokens(), 5);
    }

    #[test]
    fn window_tokens_zero_for_empty_store() {
        let s = make_store(0, 0, 4);
        assert_eq!(s.window_tokens(), 0);
    }

    #[test]
    fn clip_layer_no_window_returns_empty_overflow() {
        let mut s = make_store(1, 10, 4);
        let ov = s.clip_layer_overflow(0);
        assert_eq!(ov.shape()[0], 0);
        assert_eq!(s.stored[0].shape()[0], 10);
    }

    #[test]
    fn clip_layer_within_window_returns_empty() {
        let mut s = make_store(1, 4, 4);
        s.max_window = Some(8);
        let ov = s.clip_layer_overflow(0);
        assert_eq!(ov.shape()[0], 0);
        assert_eq!(s.stored[0].shape()[0], 4);
    }

    #[test]
    fn clip_layer_excess_overflow_moves() {
        let mut s = make_store(1, 10, 4);
        s.max_window = Some(3);
        let ov = s.clip_layer_overflow(0);
        assert_eq!(ov.shape()[0], 7);
        assert_eq!(s.stored[0].shape()[0], 3);
    }

    #[test]
    fn memory_includes_cold_payloads_and_kv() {
        let mut s = make_store(1, 0, 4);
        s.cold_encoded = Some(vec![EncodedColdLayer {
            n_positions: 3,
            hidden_size: 4,
            payload: vec![0u8; 24], // 3 × 4 × 2
        }]);
        let cold_only = s.memory_bytes();
        assert_eq!(cold_only, 24);
        assert_eq!(s.cold_bytes(), 24);
    }
}
