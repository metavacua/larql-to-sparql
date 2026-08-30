//! Stage 6 of the build pipeline — assemble + write `index.json`.

use crate::config::{VindexConfig, VindexModelConfig};
use crate::error::VindexError;
use crate::extract::build_helpers::chrono_now;
use crate::format::filenames::*;

use super::BuildContext;

impl<'a> BuildContext<'a> {
    /// Stage 6 — assemble + write `index.json`. If the extract level
    /// requires it, also write the model weights and re-emit the index
    /// with `has_model_weights = true`. Final pass adds provenance +
    /// checksums.
    pub(super) fn write_index_json(
        &mut self,
        model_name: &str,
        extract_level: crate::ExtractLevel,
    ) -> Result<(), VindexError> {
        let family = self.weights.arch.family().to_string();
        let mut config = VindexConfig {
            version: 2,
            model: model_name.to_string(),
            family: family.clone(),
            num_layers: self.num_layers,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            embed_scale: self.embed_scale,
            layers: std::mem::take(&mut self.layer_infos),
            down_top_k: self.down_top_k,
            has_model_weights: false,
            source: None,
            checksums: None,
            extract_level,
            dtype: self.dtype,
            quant: crate::QuantFormat::None,
            layer_bands: crate::LayerBands::for_family(&family, self.num_layers),
            // See the streaming writer: one mapping, not three.
            model_config: Some({
                let mut mc = VindexModelConfig::from_arch(&*self.weights.arch);
                // Geometry the extractor observed in the tensors wins
                // over the declared config — this writer's original
                // reason for not calling `from_arch`.
                mc.head_dim = self.weights.head_dim;
                mc.num_q_heads = self.weights.num_q_heads;
                mc.num_kv_heads = self.weights.num_kv_heads;
                mc.rope_base = self.weights.rope_base;
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

        // Preliminary write — `write_model_weights` reads the index.
        let config_json =
            serde_json::to_string_pretty(&config).map_err(|e| VindexError::Parse(e.to_string()))?;
        std::fs::write(self.output_dir.join(INDEX_JSON), config_json)?;

        if extract_level != crate::ExtractLevel::Browse {
            let opts = crate::format::weights::WriteWeightsOptions {
                level: crate::ExtractLevel::All,
                ffn_compact: false,
                // Dense-only BitNet: skip the attn + FFN projection
                // tensors (they live in the bitnet/ I2_S artifacts);
                // write only norms + embed + lm_head.
                skip_attn: self.dense_only,
                skip_ffn: self.dense_only,
            };
            crate::format::weights::write_model_weights_with_opts(
                self.weights,
                self.output_dir,
                self.callbacks,
                opts,
            )?;
            config.has_model_weights = true;
        }

        // Final pass — provenance + checksums.
        config.source = Some(crate::VindexSource {
            huggingface_repo: Some(model_name.to_string()),
            huggingface_revision: None,
            safetensors_sha256: None,
            extracted_at: chrono_now(),
            larql_version: env!("CARGO_PKG_VERSION").to_string(),
            // v1 provenance — populated once the extractor learns to
            // fetch the upstream commit hash + safetensors digests.
            base_model_sha: None,
            extractor_sha: None,
            base_safetensors_sha256: None,
        });
        config.checksums = crate::format::checksums::compute_checksums(self.output_dir).ok();

        let config_json =
            serde_json::to_string_pretty(&config).map_err(|e| VindexError::Parse(e.to_string()))?;
        std::fs::write(self.output_dir.join(INDEX_JSON), config_json)?;
        Ok(())
    }
}
