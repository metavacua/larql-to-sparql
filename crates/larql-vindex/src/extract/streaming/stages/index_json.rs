//! Stage 5 — assemble + write `index.json` (preliminary; checksums
//! added later in `finalize`).

use crate::config::{VindexConfig, VindexModelConfig};
use crate::error::VindexError;
use crate::extract::streaming::context::StreamingContext;
use crate::format::filenames::*;

impl<'a> StreamingContext<'a> {
    /// Stage 5 — assemble + write `index.json` (preliminary; checksums
    /// added later in `finalize`).
    pub(in crate::extract::streaming) fn write_index_json(&mut self) -> Result<(), VindexError> {
        let family = self.arch.family().to_string();
        let layer_infos = std::mem::take(&mut self.layer_infos);
        let config = VindexConfig {
            version: 2,
            model: self.model_name.to_string(),
            family: family.clone(),
            num_layers: self.num_layers,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            embed_scale: self.embed_scale,
            layers: layer_infos,
            down_top_k: self.down_top_k,
            has_model_weights: false,
            source: Some(crate::VindexSource {
                huggingface_repo: Some(self.model_name.to_string()),
                huggingface_revision: None,
                safetensors_sha256: None,
                extracted_at: crate::extract::build_helpers::chrono_now(),
                larql_version: env!("CARGO_PKG_VERSION").to_string(),
                // v1 provenance — populated once the extractor learns
                // to fetch the upstream commit hash + safetensors
                // digests.
                base_model_sha: None,
                extractor_sha: None,
                base_safetensors_sha256: None,
            }),
            checksums: None,
            extract_level: self.extract_level,
            dtype: self.dtype,
            quant: self.quant,
            layer_bands: crate::LayerBands::for_family(&family, self.num_layers),
            // One mapping, not three. This was a verbatim copy of
            // `VindexModelConfig::from_arch` differing only in the MoE
            // branch, and the copies are why `rope_scaling` could be
            // missed in all of them at once — see the note on
            // `VindexModelConfig`.
            model_config: Some({
                let mut mc = VindexModelConfig::from_arch(&*self.arch);
                // The extractor's own MoE observation wins: `n_experts`
                // is what was actually written, which can differ from
                // what the config declares.
                if self.is_moe {
                    if let Some(moe) = mc.moe.as_mut() {
                        moe.num_experts = self.n_experts;
                    }
                } else {
                    mc.moe = None;
                }
                mc
            }),
            fp4: None,
            ffn_layout: None,
            bitnet_layout: None,
        };

        // Write preliminary index.json (needed by write_model_weights which reads dtype from it).
        let config_json =
            serde_json::to_string_pretty(&config).map_err(|e| VindexError::Parse(e.to_string()))?;
        std::fs::write(self.output_dir.join(INDEX_JSON), config_json)?;
        Ok(())
    }
}
