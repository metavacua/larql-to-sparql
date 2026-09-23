//! Gate rows for layouts with a single dense `gate_proj` per layer.
//!
//! Serves two cases that are distinct in the model and identical here:
//!
//! - **Dense models** — one gate matrix, no experts.
//! - **Hybrid MoE with packed-BF16 experts** (Gemma 4 26B A4B) — experts
//!   are stored separately and the *dense* FFN gate is what KNN walk
//!   routing consults, so the gate rows written are the dense ones.
//!
//! Both record `num_experts: None`, which the reader treats as a
//! single-expert index — the format's existing convention and the one
//! §4.1.1's expert-addressed defaults are built on.

use crate::config::dtype::write_floats;
use crate::config::VindexLayerInfo;
use crate::error::VindexError;
use crate::extract::streaming::context::StreamingContext;
use crate::extract::streaming::tensor_io::{normalize_key, GateSink};

impl StreamingContext<'_> {
    /// Write one layer's gate rows from a single dense gate matrix.
    ///
    /// Returns the byte count appended to `gate_file` (zero when the
    /// layer's gate tensor does not resolve).
    pub(super) fn gate_rows_dense(
        &mut self,
        layer: usize,
        gate_file: &mut GateSink,
        offset: u64,
        prefixes: &[&str],
    ) -> Result<u64, VindexError> {
        let key = normalize_key(&self.arch.ffn_gate_key(layer), prefixes);
        let Some(tensor) = self.tensor_source.get_tensor_f32(&key)? else {
            return Ok(0);
        };

        let num_features = tensor.nrows();
        let data = tensor
            .as_slice()
            .expect("gate tensor from get_tensor_f32 is contiguous");
        let length = write_floats(gate_file, data, self.dtype)?;

        self.layer_infos.push(VindexLayerInfo {
            layer,
            num_features,
            offset,
            length,
            num_experts: None,
            num_features_per_expert: None,
        });
        Ok(length)
    }
}
