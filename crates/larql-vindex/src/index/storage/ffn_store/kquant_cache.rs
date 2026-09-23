//! Q4_K/Q6_K dequant cache — `kquant_ffn_layer` lazily decodes a whole
//! layer to f32 (transposing down from `[hidden, intermediate]` to
//! feature-major), shares the result via `Arc`, and bounds memory
//! via an LRU controlled by `set_kquant_ffn_cache_max_layers`.
//!
//! **The cache is the legacy path.** Production Metal decode bypasses
//! it entirely (`kquant_matmul_transb` streams Q4_K bytes through the
//! GPU). The W2 feature-major down emit (see
//! `format/weights/write_kquant/feature_major_down.rs` + the
//! `kquant_down_feature_scaled_add` dispatch) replaces the cache for
//! per-feature down decode when `down_features_q4k.bin` is present.
//! The cache stays as the fallback for vindexes extracted before
//! W2 landed.
//!
//! Carved out of `ffn_store.rs` in the 2026-04-25 modularity pass.

use std::sync::Arc;

use crate::index::core::VectorIndex;

impl VectorIndex {
    /// Diagnostic: count of populated `kquant_ffn_cache` slots and the
    /// total f32 bytes they hold. Used by perf probes that need to know
    /// whether a decode actually exercised the dequant cache (the hot
    /// path on Metal does NOT — it streams Q4_K bytes through
    /// `kquant_matmul_transb`). Returns `(populated_slots, bytes)`.
    pub fn kquant_ffn_cache_stats(&self) -> (usize, usize) {
        let cache = self.ffn.kquant_ffn_cache.lock().unwrap();
        let mut slots = 0usize;
        let mut bytes = 0usize;
        for slot in cache.iter() {
            for arc in slot.iter().flatten() {
                slots += 1;
                bytes += arc.len() * std::mem::size_of::<f32>();
            }
        }
        (slots, bytes)
    }

    /// Cap the number of layers held in `kquant_ffn_cache`. Mirror of
    /// `set_gate_cache_max_layers` for the FFN dequant cache. `0`
    /// (default) means unbounded. Setting a smaller cap shrinks the
    /// cache eagerly via the LRU.
    ///
    /// Recommended: `8` for a CPU-only Gemma 3 4B server (≈ 840 MB
    /// down-leg ceiling). Metal-backed runs do not need this — the
    /// full-K fast path bypasses the cache entirely. With W2
    /// feature-major down enabled at extract time, the cache is
    /// only used for non-Q4K interleaved fallback paths and can
    /// be capped at 1.
    pub fn set_kquant_ffn_cache_max_layers(&self, max_layers: usize) {
        self.ffn
            .kquant_ffn_cache_max_layers
            .store(max_layers, std::sync::atomic::Ordering::Relaxed);
        if max_layers > 0 {
            let mut cache = self.ffn.kquant_ffn_cache.lock().unwrap();
            let mut lru = self.ffn.kquant_ffn_cache_lru.lock().unwrap();
            while lru.len() > max_layers {
                if let Some(evict) = lru.pop_back() {
                    if evict < cache.len() {
                        cache[evict] = [None, None, None];
                    }
                }
            }
        }
    }

    /// Record an access to a Q4_K-cached layer and evict if the LRU
    /// has grown beyond `kquant_ffn_cache_max_layers`. Must be called
    /// with `cache` already locked by the caller; `just_inserted` is
    /// true when this call just dequantised a fresh layer.
    fn touch_q4k_ffn_cache_lru(
        &self,
        layer: usize,
        just_inserted: bool,
        cache: &mut [[Option<std::sync::Arc<Vec<f32>>>; 3]],
    ) {
        let max = self
            .ffn
            .kquant_ffn_cache_max_layers
            .load(std::sync::atomic::Ordering::Relaxed);
        if max == 0 {
            return;
        }
        let mut lru = self.ffn.kquant_ffn_cache_lru.lock().unwrap();
        if let Some(pos) = lru.iter().position(|&l| l == layer) {
            lru.remove(pos);
        }
        lru.push_front(layer);
        if just_inserted {
            while lru.len() > max {
                if let Some(evict) = lru.pop_back() {
                    if evict < cache.len() && evict != layer {
                        cache[evict] = [None, None, None];
                    }
                }
            }
        }
    }

    /// Dequantise one Q4K/Q6K FFN matrix on demand, caching the result.
    /// `component`: 0=gate, 1=up, 2=down. Returns `None` when no Q4K
    /// interleaved mmap is loaded. First access per (layer, component)
    /// pays a ~200ms–1s dequant cost (varies with intermediate size);
    /// later accesses are a single `Arc` clone.
    ///
    /// **Memory cost.** Caching a 31B layer's up+down is ~1.85GB of f32
    /// heap. For fine-grained inference prefer [`Self::kquant_ffn_row_into`],
    /// which decodes a single feature into a caller-provided buffer
    /// without populating the cache.
    pub fn kquant_ffn_layer(
        &self,
        layer: usize,
        component: usize,
    ) -> Option<std::sync::Arc<Vec<f32>>> {
        if component > 2 {
            return None;
        }
        {
            let mut cache = self.ffn.kquant_ffn_cache.lock().unwrap();
            if let Some(slot) = cache.get(layer) {
                if let Some(ref arc) = slot[component] {
                    let arc = arc.clone();
                    // Hit — bump LRU but don't evict (just_inserted=false).
                    self.touch_q4k_ffn_cache_lru(layer, false, &mut cache);
                    return Some(arc);
                }
            }
        }
        let slices = self.interleaved_kquant_layer_data(layer)?;
        let (bytes, format) = slices[component];
        let intermediate = self.num_features(layer);
        if intermediate == 0 {
            return None;
        }
        // Layout (row padding, down transpose) is owned by the shared
        // decoder — see `kquant_decode.rs`.
        let final_data = super::kquant_decode::decode_component_feature_major(
            bytes,
            format,
            intermediate,
            self.hidden_size,
            component,
        )?;
        let arc = std::sync::Arc::new(final_data);
        {
            let mut cache = self.ffn.kquant_ffn_cache.lock().unwrap();
            if let Some(slot) = cache.get_mut(layer) {
                slot[component] = Some(arc.clone());
            }
            // Fresh insert — bump LRU and evict if over the cap.
            self.touch_q4k_ffn_cache_lru(layer, true, &mut cache);
        }
        Some(arc)
    }

    /// Cache-based scaled-add — decodes the whole layer (`kquant_ffn_layer`)
    /// on first access, then serves `out += alpha * row` from the cached
    /// feature-major matrix. Required for down: it is stored transposed
    /// on disk (`[hidden, intermediate]`), so a per-row decode reads
    /// hidden-dim rows rather than feature vectors.
    ///
    /// Superseded by `kquant_down_feature_scaled_add` when
    /// `down_features_q4k.bin` is present (W2). Stays here as the
    /// fallback for legacy vindexes.
    #[inline]
    pub fn kquant_ffn_row_scaled_add_via_cache(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        let Some(arc) = self.kquant_ffn_layer(layer, component) else {
            return false;
        };
        let hidden = self.hidden_size;
        let row_start = feat * hidden;
        let row_end = row_start + hidden;
        if row_end > arc.len() || out.len() != hidden {
            return false;
        }
        for i in 0..hidden {
            out[i] += alpha * arc[row_start + i];
        }
        true
    }

    /// Lock-free dequant cache for the parallel-batch server path.
    ///
    /// On the first call for a given `(layer, component)` this dequantises
    /// the Q4K data and stores an `Arc<Vec<f32>>` in a per-slot `OnceLock`.
    /// Every subsequent call is a single atomic load + `Arc::clone` —
    /// no mutex, no LRU, no contention between concurrent rayon workers.
    ///
    /// The data layout matches `kquant_ffn_layer` exactly (component=2/down is
    /// transposed to feature-major so callers can do `activation.dot(&view)`
    /// directly without an extra `.t()`).
    ///
    /// Returns `None` only when the vindex has no Q4K interleaved data or
    /// the layer index is out of range.  A `None` stored by `get_or_init`
    /// is permanent for this instance; callers must fall back to fresh
    /// dequant in that case.
    pub fn kquant_ffn_layer_once(&self, layer: usize, component: usize) -> Option<Arc<Vec<f32>>> {
        if component > 2 {
            return None;
        }
        let once = self.ffn.kquant_ffn_once.get(layer)?.get(component)?;

        let result = once.get_or_init(|| {
            let slices = self.interleaved_kquant_layer_data(layer)?;
            let (bytes, format) = slices[component];
            let intermediate = self.num_features(layer);
            if intermediate == 0 {
                return None;
            }
            // Same feature-major layout as `kquant_ffn_layer` — both go
            // through the shared row-padding-aware decoder.
            let final_data = super::kquant_decode::decode_component_feature_major(
                bytes,
                format,
                intermediate,
                self.hidden_size,
                component,
            )?;
            Some(std::sync::Arc::new(final_data))
        });

        result.clone()
    }
}

#[cfg(test)]
mod tests {
    //! Cache-path-only coverage. The dequant happy path lives in
    //! `tests/test_vindex_to_q4k.rs` end-to-end fixtures; here we
    //! pin the cache-hit, LRU, stats, and early-return branches that
    //! don't require real Q4_K-encoded bytes.
    use std::sync::atomic::Ordering;

    use ndarray::Array2;

    use super::*;

    fn fresh(num_layers: usize, hidden: usize) -> VectorIndex {
        let mut v = VectorIndex::empty(num_layers, hidden);
        for layer in 0..num_layers {
            // num_features fallback wants a populated gate matrix; the
            // exact shape doesn't matter for the cache tests.
            v.gate.gate_vectors[layer] = Some(Array2::<f32>::zeros((4, hidden)));
        }
        v
    }

    fn install_cache_entry(v: &VectorIndex, layer: usize, component: usize, data: Vec<f32>) {
        let mut cache = v.ffn.kquant_ffn_cache.lock().unwrap();
        cache[layer][component] = Some(Arc::new(data));
    }

    // ── kquant_ffn_cache_stats ────────────────────────────────────────

    #[test]
    fn cache_stats_zero_when_empty() {
        let v = fresh(3, 8);
        let (slots, bytes) = v.kquant_ffn_cache_stats();
        assert_eq!(slots, 0);
        assert_eq!(bytes, 0);
    }

    #[test]
    fn cache_stats_counts_populated_slots_and_bytes() {
        let v = fresh(3, 8);
        install_cache_entry(&v, 0, 0, vec![0.0_f32; 4]); // 16 bytes
        install_cache_entry(&v, 0, 1, vec![0.0_f32; 8]); // 32 bytes
        install_cache_entry(&v, 2, 2, vec![0.0_f32; 16]); // 64 bytes
        let (slots, bytes) = v.kquant_ffn_cache_stats();
        assert_eq!(slots, 3);
        assert_eq!(bytes, 16 + 32 + 64);
    }

    // ── set_kquant_ffn_cache_max_layers ───────────────────────────────

    #[test]
    fn set_max_layers_zero_unbounded_does_not_evict() {
        let v = fresh(4, 8);
        for layer in 0..4 {
            install_cache_entry(&v, layer, 0, vec![0.0_f32; 4]);
            v.ffn.kquant_ffn_cache_lru.lock().unwrap().push_front(layer);
        }
        v.set_kquant_ffn_cache_max_layers(0);
        let (slots, _) = v.kquant_ffn_cache_stats();
        assert_eq!(slots, 4, "max=0 means unbounded");
    }

    #[test]
    fn set_max_layers_evicts_lru_tail() {
        let v = fresh(5, 8);
        for layer in 0..5 {
            install_cache_entry(&v, layer, 0, vec![0.0_f32; 4]);
            v.ffn.kquant_ffn_cache_lru.lock().unwrap().push_front(layer);
        }
        // After this lru = [4, 3, 2, 1, 0] (front=most-recent).
        v.set_kquant_ffn_cache_max_layers(2);
        let (slots, _) = v.kquant_ffn_cache_stats();
        assert_eq!(slots, 2, "shrinks to cap");
        assert_eq!(v.ffn.kquant_ffn_cache_max_layers.load(Ordering::Relaxed), 2);
        // The two MRU entries (layers 3 and 4) survive; LRU tail (0,1,2) evicted.
        let cache = v.ffn.kquant_ffn_cache.lock().unwrap();
        assert!(cache[3][0].is_some() || cache[4][0].is_some());
        assert!(cache[0][0].is_none());
        assert!(cache[1][0].is_none());
        assert!(cache[2][0].is_none());
    }

    #[test]
    fn set_max_layers_handles_evict_index_oob() {
        // Edge case: lru pop_back yields a layer >= cache.len() (e.g. if
        // the cache was resized). The eviction must skip that slot
        // without panicking.
        let v = fresh(3, 8);
        v.ffn.kquant_ffn_cache_lru.lock().unwrap().push_front(99);
        v.ffn.kquant_ffn_cache_lru.lock().unwrap().push_front(0);
        v.set_kquant_ffn_cache_max_layers(1);
        // Survives without panic; the OoB index is ignored.
    }

    // ── kquant_ffn_layer cache-hit path ───────────────────────────────

    #[test]
    fn q4k_ffn_layer_returns_none_for_invalid_component() {
        let v = fresh(2, 8);
        assert!(v.kquant_ffn_layer(0, 99).is_none());
        // Even with a cache entry "installed" in slot 0, component 99 is rejected early.
        install_cache_entry(&v, 0, 0, vec![1.0_f32; 8]);
        assert!(v.kquant_ffn_layer(0, 99).is_none());
    }

    #[test]
    fn q4k_ffn_layer_returns_cached_arc_on_hit() {
        let v = fresh(2, 8);
        install_cache_entry(&v, 0, 1, vec![1.0_f32; 8]);
        let arc = v.kquant_ffn_layer(0, 1).expect("cache hit returns Some");
        assert_eq!(arc.as_slice(), &[1.0_f32; 8]);
    }

    #[test]
    fn q4k_ffn_layer_cache_hit_bumps_lru() {
        let v = fresh(3, 8);
        v.set_kquant_ffn_cache_max_layers(3);
        install_cache_entry(&v, 0, 0, vec![1.0_f32; 4]);
        install_cache_entry(&v, 1, 0, vec![2.0_f32; 4]);
        install_cache_entry(&v, 2, 0, vec![3.0_f32; 4]);
        // Seed LRU with 0, 1, 2 (front=most-recent).
        {
            let mut lru = v.ffn.kquant_ffn_cache_lru.lock().unwrap();
            lru.push_back(0);
            lru.push_back(1);
            lru.push_back(2);
        }
        // Hit on layer 0 — moves it to the front.
        let _ = v.kquant_ffn_layer(0, 0).unwrap();
        let lru = v.ffn.kquant_ffn_cache_lru.lock().unwrap();
        assert_eq!(lru.front().copied(), Some(0), "hit promotes to MRU");
    }

    #[test]
    fn q4k_ffn_layer_returns_none_with_no_q4k_data() {
        // No interleaved_kquant mmap, no cached entry → None.
        let v = fresh(1, 8);
        assert!(v.kquant_ffn_layer(0, 0).is_none());
    }

    #[test]
    fn q4k_ffn_layer_oob_layer_returns_none() {
        let v = fresh(1, 8);
        assert!(v.kquant_ffn_layer(99, 0).is_none());
    }

    // ── kquant_ffn_row_scaled_add_via_cache ───────────────────────────

    #[test]
    fn row_scaled_add_via_cache_writes_alpha_times_row() {
        let v = fresh(1, 4);
        // Cached layer is a flat feature-major buffer: 2 features × 4 hidden.
        install_cache_entry(
            &v,
            0,
            0,
            vec![1.0, 2.0, 3.0, 4.0, /* feature 1 */ 5.0, 6.0, 7.0, 8.0],
        );
        let mut out = [10.0_f32, 10.0, 10.0, 10.0];
        // Pull feature index 1, alpha = 0.5 → out += 0.5 * [5, 6, 7, 8].
        let ok = v.kquant_ffn_row_scaled_add_via_cache(0, 0, 1, 0.5, &mut out);
        assert!(ok);
        assert_eq!(out, [12.5, 13.0, 13.5, 14.0]);
    }

    #[test]
    fn row_scaled_add_via_cache_returns_false_with_no_cache() {
        let v = fresh(1, 4);
        let mut out = [0.0_f32; 4];
        // No q4k data, no cache → kquant_ffn_layer returns None → false.
        assert!(!v.kquant_ffn_row_scaled_add_via_cache(0, 0, 0, 1.0, &mut out));
    }

    #[test]
    fn row_scaled_add_via_cache_rejects_oob_feature() {
        let v = fresh(1, 4);
        install_cache_entry(&v, 0, 0, vec![0.0_f32; 8]); // 2 features × 4
        let mut out = [0.0_f32; 4];
        // feat=99 → row_end > arc.len() → false.
        assert!(!v.kquant_ffn_row_scaled_add_via_cache(0, 0, 99, 1.0, &mut out));
    }

    #[test]
    fn row_scaled_add_via_cache_rejects_wrong_out_len() {
        let v = fresh(1, 4);
        install_cache_entry(&v, 0, 0, vec![0.0_f32; 8]);
        let mut out = [0.0_f32; 7]; // wrong length
        assert!(!v.kquant_ffn_row_scaled_add_via_cache(0, 0, 0, 1.0, &mut out));
    }

    // ── kquant_ffn_layer_once early returns ───────────────────────────

    // ── row-padded slab regression (non-256-aligned dims) ────────────

    /// Write a one-layer interleaved_kquant.bin + manifest to `dir` the
    /// way the production writer does, and return the logical matrices.
    /// gate/up rows are per-feature constants, down columns per-feature
    /// constants, so any stride error shows up as wildly wrong values
    /// rather than small quant noise.
    fn write_non_aligned_q4k_layer(
        dir: &std::path::Path,
        intermediate: usize,
        hidden: usize,
    ) -> std::io::Result<()> {
        use super::super::kquant_decode::q4k_row_padded_for_tests;
        use crate::format::filenames::{INTERLEAVED_KQUANT_BIN, INTERLEAVED_KQUANT_MANIFEST_JSON};
        use larql_models::quant::ggml::k_quant_padded_cols;

        let gate: Vec<f32> = (0..intermediate)
            .flat_map(|r| std::iter::repeat_n((r + 1) as f32 * 0.5, hidden))
            .collect();
        let up = gate.clone();
        // down is stored [hidden, intermediate]; cell [h][i] = (i+1)*0.25.
        let down: Vec<f32> = (0..hidden)
            .flat_map(|_| (0..intermediate).map(|i| (i + 1) as f32 * 0.25))
            .collect();

        let mut bin: Vec<u8> = Vec::new();
        let mut manifest = Vec::new();
        for (key, data, rows, cols) in [
            ("gate", &gate, intermediate, hidden),
            ("up", &up, intermediate, hidden),
            ("down", &down, hidden, intermediate),
        ] {
            let bytes = q4k_row_padded_for_tests(data, rows, cols);
            manifest.push(serde_json::json!({
                "key": key,
                "shape": [rows, k_quant_padded_cols(cols)],
                "format": "Q4_K",
                "offset": bin.len(),
                "length": bytes.len(),
            }));
            bin.extend_from_slice(&bytes);
        }
        std::fs::write(dir.join(INTERLEAVED_KQUANT_BIN), &bin)?;
        std::fs::write(
            dir.join(INTERLEAVED_KQUANT_MANIFEST_JSON),
            serde_json::to_vec(&manifest).unwrap(),
        )?;
        Ok(())
    }

    #[test]
    fn q4k_ffn_layer_decodes_row_padded_non_aligned_slabs() {
        use super::super::kquant_decode::Q4K_TEST_TOLERANCE;

        // hidden=320 (padded 512) and intermediate=3 (padded 256): both
        // gate/up and the down transpose exercise the padded stride.
        // Before the fix, rows past row 0 read padding zeros / shifted
        // data and this test fails with values far outside tolerance.
        let (intermediate, hidden) = (3usize, 320usize);
        let dir = tempfile::tempdir().unwrap();
        write_non_aligned_q4k_layer(dir.path(), intermediate, hidden).unwrap();

        let mut v = VectorIndex::empty(1, hidden);
        v.gate.gate_vectors[0] = Some(Array2::<f32>::zeros((intermediate, hidden)));
        v.load_interleaved_kquant(dir.path()).unwrap();

        // gate (component 0): row r ≈ (r+1)*0.5 everywhere.
        let gate = v.kquant_ffn_layer(0, 0).expect("gate decodes");
        assert_eq!(gate.len(), intermediate * hidden);
        for r in 0..intermediate {
            let expect = (r + 1) as f32 * 0.5;
            for (c, &val) in gate[r * hidden..(r + 1) * hidden].iter().enumerate() {
                assert!(
                    (val - expect).abs() < Q4K_TEST_TOLERANCE,
                    "gate row {r} col {c}: got {val}, want ~{expect}"
                );
            }
        }

        // down (feature-major after transpose):
        // feature i ≈ (i+1)*0.25 across all hidden positions.
        let down = v
            .kquant_ffn_layer(0, super::super::FFN_DOWN)
            .expect("down decodes");
        assert_eq!(down.len(), intermediate * hidden);
        for i in 0..intermediate {
            let expect = (i + 1) as f32 * 0.25;
            for (h, &val) in down[i * hidden..(i + 1) * hidden].iter().enumerate() {
                assert!(
                    (val - expect).abs() < Q4K_TEST_TOLERANCE,
                    "down feature {i} h {h}: got {val}, want ~{expect}"
                );
            }
        }

        // The lock-free twin must produce the identical layout.
        let once = v.kquant_ffn_layer_once(0, 0).expect("once-path decodes");
        assert_eq!(
            once.as_slice(),
            gate.as_slice(),
            "once path diverges from LRU path"
        );
    }

    #[test]
    fn q4k_ffn_layer_once_invalid_component_returns_none() {
        let v = fresh(1, 8);
        assert!(v.kquant_ffn_layer_once(0, 99).is_none());
    }

    #[test]
    fn q4k_ffn_layer_once_oob_layer_returns_none() {
        let v = fresh(1, 8);
        assert!(v.kquant_ffn_layer_once(99, 0).is_none());
    }

    #[test]
    fn q4k_ffn_layer_once_returns_none_with_no_q4k_data() {
        let v = fresh(2, 8);
        // get_or_init runs and the closure returns None → cached
        // permanently as None. Subsequent call also yields None.
        assert!(v.kquant_ffn_layer_once(0, 0).is_none());
        assert!(v.kquant_ffn_layer_once(0, 0).is_none());
    }
}
