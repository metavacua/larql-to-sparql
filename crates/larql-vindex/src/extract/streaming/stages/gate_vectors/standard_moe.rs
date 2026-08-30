//! Gate rows for the standard MoE layout (Mixtral lineage) — one
//! `gate_proj` tensor per expert, unpacked.

use crate::config::dtype::write_floats;
use crate::config::VindexLayerInfo;
use crate::error::VindexError;
use crate::extract::streaming::context::StreamingContext;
use crate::extract::streaming::tensor_io::{normalize_key, GateSink};

use super::summary::{expert_seed, summarise_expert_gate};

impl StreamingContext<'_> {
    /// Write one layer's gate rows from per-expert gate tensors.
    ///
    /// Returns the byte count appended to `gate_file` (zero when no
    /// expert tensor for this layer resolves).
    pub(super) fn gate_rows_standard_moe(
        &mut self,
        layer: usize,
        gate_file: &mut GateSink,
        offset: u64,
        prefixes: &[&str],
    ) -> Result<u64, VindexError> {
        let summary_k = self.summary_features_per_expert;
        let mut total_features = 0usize;
        let mut layer_bytes = 0u64;
        let mut features_per_expert = 0usize;

        for expert in 0..self.n_experts {
            let Some(key) = self.arch.expert_ffn_gate_key(layer, expert) else {
                continue;
            };
            let Some(tensor) = self
                .tensor_source
                .get_tensor_f32(&normalize_key(&key, prefixes))?
            else {
                continue;
            };

            let (data, n_feat) =
                summarise_expert_gate(tensor.view(), summary_k, expert_seed(layer, expert));

            layer_bytes += write_floats(gate_file, &data, self.dtype)?;
            features_per_expert = n_feat;
            total_features += n_feat;
        }

        if total_features == 0 {
            return Ok(0);
        }

        self.layer_infos.push(VindexLayerInfo {
            layer,
            num_features: total_features,
            offset,
            length: layer_bytes,
            num_experts: Some(self.n_experts),
            num_features_per_expert: Some(features_per_expert),
        });
        Ok(layer_bytes)
    }
}
