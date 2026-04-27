//! Walk FFN data — mmap'd feature-major down and up projection vectors.
//!
//! Manages down_features.bin and up_features.bin — [intermediate, hidden] per layer,
//! f32 files where each feature's vector is contiguous for zero-copy BLAS access.

use std::sync::Arc;

use crate::error::VindexError;

use super::core::VectorIndex;

use crate::mmap_util::{mmap_demand_paged, mmap_optimized};

/// Feature store methods for VectorIndex.
impl VectorIndex {
    /// Load feature-major down vectors from down_features.bin.
    pub fn load_down_features(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("down_features.bin");
        if !path.exists() {
            return Err(VindexError::Parse(
                "down_features.bin not found. Run: cargo run --release -p larql-vindex --example build_down_features -- <vindex>".into()
            ));
        }
        let file = std::fs::File::open(&path)?;
        // Demand-paged: only the activated feature vectors are read per token.
        let mmap = unsafe { mmap_demand_paged(&file)? };
        self.down_features_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Whether feature-major down vectors are loaded.
    pub fn has_down_features(&self) -> bool {
        self.down_features_mmap.is_some()
    }

    /// Get a feature's contiguous down vector from the mmap'd feature-major file.
    /// Returns `[hidden_size]` f32 slice — zero-copy from mmap.
    pub fn down_feature_vector(&self, layer: usize, feature: usize) -> Option<&[f32]> {
        let mmap = self.down_features_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 || feature >= intermediate { return None; }

        let layer_floats = intermediate * self.hidden_size;
        let layer_offset = layer * layer_floats * 4;
        let feature_offset = feature * self.hidden_size * 4;
        let start = layer_offset + feature_offset;
        let end = start + self.hidden_size * 4;

        if end > mmap.len() { return None; }

        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, self.hidden_size)
        };
        Some(data)
    }

    /// Get the full down matrix for a layer: [intermediate, hidden] zero-copy view.
    pub fn down_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.down_features_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 { return None; }

        let floats_per_layer = intermediate * self.hidden_size;
        let bytes_per_layer = floats_per_layer * 4;
        let start = layer * bytes_per_layer;
        let end = start + bytes_per_layer;
        if end > mmap.len() { return None; }

        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, floats_per_layer)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Load feature-major up vectors from up_features.bin.
    pub fn load_up_features(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("up_features.bin");
        if !path.exists() {
            return Err(VindexError::Parse(
                "up_features.bin not found. Run: cargo run --release -p larql-vindex --example build_up_features -- <vindex>".into()
            ));
        }
        let file = std::fs::File::open(&path)?;
        // Demand-paged: only activated feature vectors are read per token.
        let mmap = unsafe { mmap_demand_paged(&file)? };
        self.up_features_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Get the full up matrix for a layer: [intermediate, hidden] zero-copy view.
    pub fn up_layer_matrix(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.up_features_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 { return None; }
        let floats_per_layer = intermediate * self.hidden_size;
        let bytes_per_layer = floats_per_layer * 4;
        let start = layer * bytes_per_layer;
        let end = start + bytes_per_layer;
        if end > mmap.len() { return None; }
        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, floats_per_layer)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Whether both up and down feature-major mmaps are loaded.
    pub fn has_full_mmap_ffn(&self) -> bool {
        self.down_features_mmap.is_some() && self.up_features_mmap.is_some()
    }

    // ── Interleaved FFN data: gate+up+down packed per layer ──

    /// Load interleaved FFN data: [gate|up|down] per layer in one contiguous file.
    /// Eliminates TLB thrash from 3 separate mmap files.
    pub fn load_interleaved(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("interleaved.bin");
        if !path.exists() {
            return Err(VindexError::Parse(
                "interleaved.bin not found. Run: cargo run --release -p larql-vindex --example build_interleaved -- <vindex>".into()
            ));
        }
        let file = std::fs::File::open(&path)?;
        // Demand-paged: per-layer prefetch issued at query time via prefetch_interleaved_layer.
        let mmap = unsafe { mmap_demand_paged(&file)? };
        self.interleaved_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    /// Whether interleaved FFN data is loaded.
    pub fn has_interleaved(&self) -> bool {
        self.interleaved_mmap.is_some()
    }

    /// Get gate matrix for a layer from the interleaved file: [intermediate, hidden].
    pub fn interleaved_gate(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.interleaved_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 { return None; }
        let matrix_floats = intermediate * self.hidden_size;
        let matrix_bytes = matrix_floats * 4;
        let layer_bytes = matrix_bytes * 3; // gate + up + down
        let start = layer * layer_bytes; // gate is first
        let end = start + matrix_bytes;
        if end > mmap.len() { return None; }
        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, matrix_floats)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Get up matrix for a layer from the interleaved file: [intermediate, hidden].
    pub fn interleaved_up(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.interleaved_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 { return None; }
        let matrix_floats = intermediate * self.hidden_size;
        let matrix_bytes = matrix_floats * 4;
        let layer_bytes = matrix_bytes * 3;
        let start = layer * layer_bytes + matrix_bytes; // up is second
        let end = start + matrix_bytes;
        if end > mmap.len() { return None; }
        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, matrix_floats)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Get down matrix for a layer from the interleaved file: [intermediate, hidden].
    pub fn interleaved_down(&self, layer: usize) -> Option<ndarray::ArrayView2<'_, f32>> {
        let mmap = self.interleaved_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 { return None; }
        let matrix_floats = intermediate * self.hidden_size;
        let matrix_bytes = matrix_floats * 4;
        let layer_bytes = matrix_bytes * 3;
        let start = layer * layer_bytes + matrix_bytes * 2; // down is third
        let end = start + matrix_bytes;
        if end > mmap.len() { return None; }
        let data = unsafe {
            let ptr = mmap[start..end].as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, matrix_floats)
        };
        ndarray::ArrayView2::from_shape((intermediate, self.hidden_size), data).ok()
    }

    /// Prefetch next layer's interleaved data into page cache.
    pub fn prefetch_interleaved_layer(&self, layer: usize) {
        #[cfg(unix)]
        if let Some(ref mmap) = self.interleaved_mmap {
            let intermediate = self.num_features(layer);
            if intermediate == 0 { return; }
            let matrix_bytes = intermediate * self.hidden_size * 4;
            let layer_bytes = matrix_bytes * 3;
            let start = layer * layer_bytes;
            let end = (start + layer_bytes).min(mmap.len());
            if start >= mmap.len() { return; }
            unsafe {
                let ptr = mmap[start..].as_ptr() as *mut libc::c_void;
                libc::madvise(ptr, end - start, libc::MADV_WILLNEED);
            }
        }
    }

    // ── Q4 interleaved: quantized gate+up+down per layer ──

    /// Load Q4_0 interleaved FFN data.
    pub fn load_interleaved_q4(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("interleaved_q4.bin");
        if !path.exists() {
            return Err(VindexError::Parse("interleaved_q4.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_demand_paged(&file)? };
        self.interleaved_q4_mmap = Some(Arc::new(mmap));
        Ok(())
    }

    pub fn has_interleaved_q4(&self) -> bool {
        self.interleaved_q4_mmap.is_some()
    }

    /// Load Q4_K/Q6_K interleaved FFN data (Ollama-compatible, matches attn format).
    ///
    /// Also reads the optional `interleaved_q4k_manifest.json` sidecar emitted
    /// by the streaming Q4 writer. When the manifest is present callers get
    /// per-matrix layout (offsets, lengths, formats) via
    /// [`VectorIndex::interleaved_q4k_layer_data`]. When it's absent — older
    /// vindexes from `build_q4k_weights.rs` — callers fall back to the legacy
    /// uniform-stride path.
    pub fn load_interleaved_q4k(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("interleaved_q4k.bin");
        if !path.exists() {
            return Err(VindexError::Parse("interleaved_q4k.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        // Demand-paged: the q4k forward walk reads only the activated features'
        // byte ranges per layer, not the entire 13 GB file.
        let mmap = unsafe { mmap_demand_paged(&file)? };
        self.interleaved_q4k_mmap = Some(Arc::new(mmap));

        let manifest_path = dir.join("interleaved_q4k_manifest.json");
        if manifest_path.exists() {
            let json: Vec<serde_json::Value> = serde_json::from_str(
                &std::fs::read_to_string(&manifest_path)
                    .map_err(|e| VindexError::Parse(e.to_string()))?,
            )
            .map_err(|e| VindexError::Parse(e.to_string()))?;

            let entries: Vec<(usize, usize, String)> = json
                .iter()
                .map(|e| {
                    let offset = e["offset"].as_u64().unwrap_or(0) as usize;
                    let length = e["length"].as_u64().unwrap_or(0) as usize;
                    let format = e["format"].as_str().unwrap_or("Q4_K").to_string();
                    (offset, length, format)
                })
                .collect();
            self.interleaved_q4k_manifest = Some(entries);
        }
        Ok(())
    }

    pub fn has_interleaved_q4k(&self) -> bool {
        self.interleaved_q4k_mmap.is_some()
    }

    /// Per-layer Q4_K/Q6_K FFN slices — [gate, up, down] with formats.
    ///
    /// Returns `None` when the FFN manifest wasn't present at load time
    /// (caller should fall back to uniform-stride). Returns `Some` iff the
    /// manifest has 3 entries for `layer`; downstream kernels dispatch on
    /// the format string (`"Q4_K"` or `"Q6_K"`).
    pub fn interleaved_q4k_layer_data(&self, layer: usize) -> Option<[(&[u8], &str); 3]> {
        let mmap = self.interleaved_q4k_mmap.as_ref()?;
        let manifest = self.interleaved_q4k_manifest.as_ref()?;
        let base = layer * 3;
        if base + 2 >= manifest.len() {
            return None;
        }
        let mut out: [(&[u8], &str); 3] = [(&[], ""); 3];
        for i in 0..3 {
            let (offset, length, ref format) = manifest[base + i];
            out[i] = (&mmap[offset..offset + length], format.as_str());
        }
        Some(out)
    }

    /// Dequantize one matrix from Q4 interleaved file → f32 Array2.
    /// component: 0=gate, 1=up, 2=down
    fn dequant_q4_matrix(&self, layer: usize, component: usize) -> Option<ndarray::Array2<f32>> {
        let mmap = self.interleaved_q4_mmap.as_ref()?;
        let intermediate = self.num_features(layer);
        if intermediate == 0 { return None; }

        let floats_per_matrix = intermediate * self.hidden_size;
        let q4_bytes_per_matrix = floats_per_matrix / 32 * 18; // Q4_0: 18 bytes per 32 elements
        let q4_bytes_per_layer = q4_bytes_per_matrix * 3;

        let start = layer * q4_bytes_per_layer + component * q4_bytes_per_matrix;
        let end = start + q4_bytes_per_matrix;
        if end > mmap.len() { return None; }

        let q4_data = &mmap[start..end];
        let floats = larql_models::quant::ggml::dequantize_q4_0(q4_data, floats_per_matrix).ok()?;
        ndarray::Array2::from_shape_vec((intermediate, self.hidden_size), floats).ok()
    }

    /// Dequantise one Q4K/Q6K FFN matrix on demand, caching the result.
    /// `component`: 0=gate, 1=up, 2=down. Returns `None` when no Q4K
    /// interleaved mmap is loaded. First access per (layer, component)
    /// pays a ~200ms–1s dequant cost (varies with intermediate size);
    /// later accesses are a single `Arc` clone.
    ///
    /// **Memory cost.** Caching a 31B layer's up+down is ~1.85GB of f32
    /// heap. For fine-grained inference prefer [`Self::q4k_ffn_row_into`],
    /// which decodes a single feature into a caller-provided buffer
    /// without populating the cache.
    pub fn q4k_ffn_layer(&self, layer: usize, component: usize)
        -> Option<std::sync::Arc<Vec<f32>>>
    {
        if component > 2 { return None; }
        {
            let cache = self.q4k_ffn_cache.lock().unwrap();
            if let Some(slot) = cache.get(layer) {
                if let Some(ref arc) = slot[component] {
                    return Some(arc.clone());
                }
            }
        }
        let slices = self.interleaved_q4k_layer_data(layer)?;
        let (bytes, format) = slices[component];
        let intermediate = self.num_features(layer);
        if intermediate == 0 { return None; }
        let hidden = self.hidden_size;
        let n = intermediate * hidden;
        let padded = n.div_ceil(256) * 256;
        let decoded = match format {
            "Q4_K" => larql_models::quant::ggml::dequantize_q4_k(bytes, padded).ok()?,
            "Q6_K" => larql_models::quant::ggml::dequantize_q6_k(bytes, padded).ok()?,
            _ => return None,
        };
        // Gate (0) and up (1) are stored row-major [intermediate, hidden] — row
        // `feat` already contains that feature's weight vector.
        //
        // Down (2) is stored row-major [hidden, intermediate] (the native PyTorch
        // nn.Linear(intermediate, hidden) orientation). To give callers a
        // feature-major view matching gate/up, we transpose here: after the flip
        // arc[feat*hidden..(feat+1)*hidden] is feature `feat`'s down vector.
        let final_data: Vec<f32> = if component == 2 {
            let mut t = vec![0.0f32; n];
            for h in 0..hidden {
                let src_row = &decoded[h * intermediate..(h + 1) * intermediate];
                for (i, &v) in src_row.iter().enumerate() {
                    t[i * hidden + h] = v;
                }
            }
            t
        } else {
            decoded.into_iter().take(n).collect()
        };
        let arc = std::sync::Arc::new(final_data);
        {
            let mut cache = self.q4k_ffn_cache.lock().unwrap();
            if let Some(slot) = cache.get_mut(layer) {
                slot[component] = Some(arc.clone());
            }
        }
        Some(arc)
    }

    /// Cache-based scaled-add — decodes the whole layer (`q4k_ffn_layer`)
    /// on first access, then serves `out += alpha * row` from the cached
    /// feature-major matrix. Required for down: it is stored transposed
    /// on disk (`[hidden, intermediate]`), so a per-row decode reads
    /// hidden-dim rows rather than feature vectors.
    #[inline]
    pub fn q4k_ffn_row_scaled_add_via_cache(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        let Some(arc) = self.q4k_ffn_layer(layer, component) else { return false; };
        let hidden = self.hidden_size;
        let row_start = feat * hidden;
        let row_end = row_start + hidden;
        if row_end > arc.len() || out.len() != hidden { return false; }
        for i in 0..hidden {
            out[i] += alpha * arc[row_start + i];
        }
        true
    }

    /// Cache-based dot — same role as `q4k_ffn_row_scaled_add_via_cache`
    /// but for the up leg. Currently unused (up is row-major on disk so
    /// per-row decode is enough); kept for diagnostics and test parity.
    /// If this works and the per-row version doesn't, the bug is in the
    /// row-offset calculation or per-row byte slicing.
    #[inline]
    pub fn q4k_ffn_row_dot_via_cache(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        x: &[f32],
    ) -> Option<f32> {
        let arc = self.q4k_ffn_layer(layer, component)?;
        let hidden = self.hidden_size;
        let row_start = feat * hidden;
        let row_end = row_start + hidden;
        if row_end > arc.len() { return None; }
        let mut acc = 0.0f32;
        for (i, &xv) in x.iter().enumerate() {
            acc += arc[row_start + i] * xv;
        }
        Some(acc)
    }

    /// Direct Q4K/Q6K matmul — Y = X @ W.T, where W is the FFN matrix
    /// stored as Q4K/Q6K bytes in the vindex. Decodes and FMAs fused,
    /// parallelised across W rows. Zero extra RAM (no f32 cache).
    ///
    /// `x` is `[x_rows, w_cols]` row-major. `component` selects the layer's
    /// gate (0) / up (1) / down (2) Q4K slice. On return the output is
    /// `[x_rows, w_rows]` row-major where `w_rows` equals the slice's
    /// shape-0 (intermediate for gate/up, hidden for down).
    ///
    /// Dispatches to the backend's `q4k_matvec` / `q6k_matvec` when a
    /// compute backend is provided (Metal on Apple Silicon, CPU-SIMD
    /// otherwise) — one submission per X row. Falls back to the rayon
    /// + CPU-NEON scalar path when no backend is attached.
    pub fn q4k_matmul_transb(
        &self,
        layer: usize,
        component: usize,
        x: &[f32],
        x_rows: usize,
        backend: Option<&dyn larql_compute::ComputeBackend>,
    ) -> Option<Vec<f32>> {
        use rayon::prelude::*;
        if component > 2 { return None; }
        let slices = self.interleaved_q4k_layer_data(layer)?;
        let (bytes, format) = slices[component];

        let intermediate = self.num_features(layer);
        let hidden = self.hidden_size;
        let (w_rows, w_cols) = match component {
            0 | 1 => (intermediate, hidden),
            2     => (hidden, intermediate),
            _     => return None,
        };
        if x.len() != x_rows * w_cols { return None; }
        if w_cols % 256 != 0 { return None; }

        // Backend per-row dispatch is *slower* than CPU-NEON here because
        // each q4k_matvec call pays a Metal submission (~15 ms). With x_rows
        // × layers × 3 components we'd spend all our time in dispatch.
        // A batched Metal shader (one submission per layer) would fix this,
        // but we don't have it wired yet — keep the hook for future use.
        let _ = backend;

        let (block_bytes, block_size) = match format {
            "Q4_K" => (144usize, 256usize),
            "Q6_K" => (210usize, 256usize),
            _ => return None,
        };
        let blocks_per_row = w_cols / block_size;
        let bytes_per_w_row = blocks_per_row * block_bytes;

        // CPU fallback: rayon over W rows, NEON per-row dot.
        let mut y_t = vec![0.0f32; w_rows * x_rows];
        y_t.par_chunks_mut(x_rows).enumerate().for_each(|(j, slot)| {
            let w_row_start = j * bytes_per_w_row;
            let w_row = &bytes[w_row_start..w_row_start + bytes_per_w_row];
            for i in 0..x_rows {
                let x_row = &x[i * w_cols..(i + 1) * w_cols];
                slot[i] = match format {
                    "Q4_K" => larql_models::quant::ggml::q4k_row_dot(w_row, x_row).unwrap_or(0.0),
                    "Q6_K" => larql_models::quant::ggml::q6k_row_dot(w_row, x_row).unwrap_or(0.0),
                    _ => 0.0,
                };
            }
        });
        let mut y = vec![0.0f32; x_rows * w_rows];
        for j in 0..w_rows {
            let src_base = j * x_rows;
            for i in 0..x_rows {
                y[i * w_rows + j] = y_t[src_base + i];
            }
        }
        Some(y)
    }

    /// Fused Q4K/Q6K decode + dot with `x` for one feature. Returns `None`
    /// if the row isn't available. This is ~2× faster than the
    /// `q4k_ffn_row_into` → BLAS sdot sequence because it skips the Vec
    /// allocation, the intermediate copy, and keeps the decoded data in
    /// registers.
    #[inline]
    pub fn q4k_ffn_row_dot(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        x: &[f32],
    ) -> Option<f32> {
        if component > 2 || x.len() != self.hidden_size { return None; }
        let slices = self.interleaved_q4k_layer_data(layer)?;
        let (bytes, format) = slices[component];
        let hidden = self.hidden_size;
        if feat >= self.num_features(layer) { return None; }
        match format {
            "Q4_K" => {
                if !hidden.is_multiple_of(256) { return None; }
                let bytes_per_row = (hidden / 256) * 144;
                let start = feat * bytes_per_row;
                let end = start + bytes_per_row;
                if end > bytes.len() { return None; }
                larql_models::quant::ggml::q4k_row_dot(&bytes[start..end], x).ok()
            }
            "Q6_K" => {
                if !hidden.is_multiple_of(256) { return None; }
                let bytes_per_row = (hidden / 256) * 210;
                let start = feat * bytes_per_row;
                let end = start + bytes_per_row;
                if end > bytes.len() { return None; }
                larql_models::quant::ggml::q6k_row_dot(&bytes[start..end], x).ok()
            }
            _ => None,
        }
    }

    /// Fused Q4K/Q6K decode + scaled-add into `out` for one feature.
    /// Counterpart to `q4k_ffn_row_dot` for the down leg.
    #[inline]
    pub fn q4k_ffn_row_scaled_add(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        alpha: f32,
        out: &mut [f32],
    ) -> bool {
        if component > 2 || out.len() != self.hidden_size { return false; }
        let Some(slices) = self.interleaved_q4k_layer_data(layer) else { return false; };
        let (bytes, format) = slices[component];
        let hidden = self.hidden_size;
        if feat >= self.num_features(layer) { return false; }
        match format {
            "Q4_K" => {
                if !hidden.is_multiple_of(256) { return false; }
                let bytes_per_row = (hidden / 256) * 144;
                let start = feat * bytes_per_row;
                let end = start + bytes_per_row;
                if end > bytes.len() { return false; }
                larql_models::quant::ggml::q4k_row_scaled_add(&bytes[start..end], alpha, out).is_ok()
            }
            "Q6_K" => {
                if !hidden.is_multiple_of(256) { return false; }
                let bytes_per_row = (hidden / 256) * 210;
                let start = feat * bytes_per_row;
                let end = start + bytes_per_row;
                if end > bytes.len() { return false; }
                larql_models::quant::ggml::q6k_row_scaled_add(&bytes[start..end], alpha, out).is_ok()
            }
            _ => false,
        }
    }

    /// Decode one row of a Q4K/Q6K FFN matrix directly into `out` without
    /// caching. `component`: 0=gate, 1=up, 2=down; `feat` is the feature
    /// (row) index; `out` must have length `hidden_size`. Returns `false`
    /// when the vindex has no Q4K data or shape is invalid.
    ///
    /// Row-level decode is the small-memory path for very large models
    /// (~30B+) where caching entire dequantised layers blows the RAM
    /// budget. Cost is ~50–70μs per row for hidden≈5376; at K=100 on a
    /// 60-layer model that's ~60 × 100 × 2 decodes × 60μs ≈ 720ms per
    /// forward pass.
    pub fn q4k_ffn_row_into(
        &self,
        layer: usize,
        component: usize,
        feat: usize,
        out: &mut [f32],
    ) -> bool {
        if component > 2 || out.len() != self.hidden_size { return false; }
        let Some(slices) = self.interleaved_q4k_layer_data(layer) else { return false; };
        let (bytes, format) = slices[component];
        let hidden = self.hidden_size;
        if feat >= self.num_features(layer) { return false; }

        match format {
            "Q4_K" => {
                // Q4_K block: 144 bytes for 256 elements.
                if !hidden.is_multiple_of(256) { return false; }
                let blocks_per_row = hidden / 256;
                let bytes_per_row = blocks_per_row * 144;
                let start = feat * bytes_per_row;
                let end = start + bytes_per_row;
                if end > bytes.len() { return false; }
                let row_bytes = &bytes[start..end];
                match larql_models::quant::ggml::dequantize_q4_k(row_bytes, hidden) {
                    Ok(v) => { out.copy_from_slice(&v[..hidden]); true }
                    Err(_) => false,
                }
            }
            "Q6_K" => {
                // Q6_K block: 210 bytes for 256 elements.
                if !hidden.is_multiple_of(256) { return false; }
                let blocks_per_row = hidden / 256;
                let bytes_per_row = blocks_per_row * 210;
                let start = feat * bytes_per_row;
                let end = start + bytes_per_row;
                if end > bytes.len() { return false; }
                let row_bytes = &bytes[start..end];
                match larql_models::quant::ggml::dequantize_q6_k(row_bytes, hidden) {
                    Ok(v) => { out.copy_from_slice(&v[..hidden]); true }
                    Err(_) => false,
                }
            }
            _ => false,
        }
    }

    /// Get gate matrix from Q4 interleaved file, dequantized to f32.
    pub fn interleaved_q4_gate(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.dequant_q4_matrix(layer, 0)
    }

    /// Get up matrix from Q4 interleaved file, dequantized to f32.
    pub fn interleaved_q4_up(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.dequant_q4_matrix(layer, 1)
    }

    /// Get down matrix from Q4 interleaved file, dequantized to f32.
    pub fn interleaved_q4_down(&self, layer: usize) -> Option<ndarray::Array2<f32>> {
        self.dequant_q4_matrix(layer, 2)
    }

    /// Prefetch next layer's Q4 data.
    pub fn prefetch_interleaved_q4_layer(&self, layer: usize) {
        #[cfg(unix)]
        if let Some(ref mmap) = self.interleaved_q4_mmap {
            let intermediate = self.num_features(layer);
            if intermediate == 0 { return; }
            let q4_bytes_per_matrix = intermediate * self.hidden_size / 32 * 18;
            let q4_bytes_per_layer = q4_bytes_per_matrix * 3;
            let start = layer * q4_bytes_per_layer;
            let end = (start + q4_bytes_per_layer).min(mmap.len());
            if start >= mmap.len() { return; }
            unsafe {
                let ptr = mmap[start..].as_ptr() as *mut libc::c_void;
                libc::madvise(ptr, end - start, libc::MADV_WILLNEED);
            }
        }
    }

    // warmup() is in gate.rs (it's a gate cache operation)

    // ── Q4 gate vectors for fast KNN via larql-compute ──

    /// Load Q4_0 gate vectors from gate_vectors_q4.bin.
    ///
    /// File layout: layers packed contiguously, each layer is
    /// [num_features × hidden] in Q4_0 format (18 bytes per 32 elements).
    /// The per-layer feature count comes from gate_mmap_slices (must load
    /// f32/f16 gates first for the slice metadata, or pass feature counts).
    pub fn load_gate_vectors_q4(&mut self, dir: &std::path::Path) -> Result<(), VindexError> {
        let path = dir.join("gate_vectors_q4.bin");
        if !path.exists() {
            return Err(VindexError::Parse("gate_vectors_q4.bin not found".into()));
        }
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { mmap_optimized(&file)? };

        // Compute per-layer byte offsets from feature counts
        let mut slices = Vec::with_capacity(self.num_layers);
        let mut offset = 0usize;
        for layer in 0..self.num_layers {
            let num_features = self.num_features(layer);
            let floats = num_features * self.hidden_size;
            let q4_bytes = floats / 32 * 18; // Q4_0: 18 bytes per 32 elements
            slices.push(super::types::GateQ4Slice {
                byte_offset: offset,
                byte_len: q4_bytes,
                num_features,
            });
            offset += q4_bytes;
        }

        self.gate_q4_mmap = Some(Arc::new(mmap));
        self.gate_q4_slices = slices;
        Ok(())
    }

    /// Whether Q4 gate vectors are loaded.
    pub fn has_gate_q4(&self) -> bool {
        self.gate_q4_mmap.is_some()
    }

    /// Get Q4 data slice for a layer's gate vectors. Returns the raw Q4_0 bytes.
    pub fn gate_q4_data(&self, layer: usize) -> Option<&[u8]> {
        let mmap = self.gate_q4_mmap.as_ref()?;
        let slice = self.gate_q4_slices.get(layer)?;
        if slice.byte_len == 0 { return None; }
        let end = slice.byte_offset + slice.byte_len;
        if end > mmap.len() { return None; }
        Some(&mmap[slice.byte_offset..end])
    }

}
