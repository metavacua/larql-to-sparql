//! `CREATE VINDEX "path" ARCHITECTURE "model" EMPTY` — constructs the initial
//! object in the category of vindexes for the named model architecture.

use std::path::{Path, PathBuf};

use crate::ast::VindexInit;
use crate::error::LqlError;
use crate::executor::Session;
use larql_vindex::{
    ExtractLevel, LayerBands, QuantFormat, StorageDtype, VindexConfig, VindexLayerInfo,
    VindexModelConfig,
};

impl Session {
    pub(crate) fn exec_create_vindex(
        &mut self,
        path: &str,
        architecture: &str,
        init: VindexInit,
    ) -> Result<Vec<String>, LqlError> {
        match init {
            VindexInit::Empty => self.exec_create_vindex_empty(path, architecture),
        }
    }

    fn exec_create_vindex_empty(
        &mut self,
        path: &str,
        architecture: &str,
    ) -> Result<Vec<String>, LqlError> {
        let output_dir = PathBuf::from(path);

        // Resolve architecture: try as a local directory first, then via model cache.
        let model_dir = resolve_architecture_path(architecture)
            .map_err(|e| LqlError::exec("failed to resolve architecture", e))?;

        let arch = larql_models::detect_architecture(&model_dir)
            .map_err(|e| LqlError::exec("failed to detect architecture", e))?;

        let cfg = arch.config();
        let family = arch.family().to_string();
        let num_layers = cfg.num_layers;
        let hidden_size = cfg.hidden_size;
        let intermediate_size = cfg.intermediate_size;
        let vocab_size = cfg.vocab_size.unwrap_or(0);
        let embed_scale = arch.embed_scale();

        // Build VindexLayerInfo with num_features = 0 for every layer (initial object).
        let layers: Vec<VindexLayerInfo> = (0..num_layers)
            .map(|l| VindexLayerInfo {
                layer: l,
                num_features: 0,
                offset: 0,
                length: 0,
                num_experts: None,
                num_features_per_expert: None,
            })
            .collect();

        let model_config = Some(VindexModelConfig {
            model_type: cfg.model_type.clone(),
            head_dim: cfg.head_dim,
            num_q_heads: cfg.num_q_heads,
            num_kv_heads: cfg.num_kv_heads,
            rope_base: cfg.rope_base,
            sliding_window: cfg.sliding_window,
            moe: None,
            global_head_dim: cfg.global_head_dim,
            num_global_kv_heads: cfg.num_global_kv_heads,
            partial_rotary_factor: cfg.partial_rotary_factor,
            sliding_window_pattern: cfg.sliding_window_pattern,
            layer_types: cfg.layer_types.clone(),
            attention_k_eq_v: cfg.attention_k_eq_v,
            num_kv_shared_layers: cfg.num_kv_shared_layers,
            per_layer_embed_dim: cfg.per_layer_embed_dim,
            rope_local_base: cfg.rope_local_base,
            query_pre_attn_scalar: cfg.query_pre_attn_scalar,
            final_logit_softcapping: cfg.final_logit_softcapping,
        });

        let config = VindexConfig {
            version: 2,
            model: architecture.to_string(),
            family: family.clone(),
            num_layers,
            hidden_size,
            intermediate_size,
            vocab_size,
            embed_scale,
            extract_level: ExtractLevel::Browse,
            dtype: StorageDtype::F32,
            quant: QuantFormat::None,
            layer_bands: LayerBands::for_family(&family, num_layers),
            layers,
            down_top_k: 10,
            has_model_weights: false,
            source: None,
            checksums: None,
            model_config,
            fp4: None,
            ffn_layout: None,
        };

        std::fs::create_dir_all(&output_dir)
            .map_err(|e| LqlError::exec("failed to create output directory", e))?;

        let config_json = serde_json::to_string_pretty(&config)
            .map_err(|e| LqlError::exec("failed to serialise vindex config", e))?;
        std::fs::write(
            output_dir.join(larql_vindex::format::filenames::INDEX_JSON),
            config_json,
        )
        .map_err(|e| LqlError::exec("failed to write index.json", e))?;

        Ok(vec![format!(
            "Created empty vindex: {} ({} layers, hidden={}, intermediate={}, model: {})",
            output_dir.display(),
            num_layers,
            hidden_size,
            intermediate_size,
            architecture,
        )])
    }
}

/// Resolve the architecture string to a model directory that contains config.json.
/// Accepts a local directory path or a model ID looked up in the HF cache.
fn resolve_architecture_path(architecture: &str) -> Result<PathBuf, larql_models::ModelError> {
    let local = Path::new(architecture);
    if local.is_dir() {
        return Ok(local.to_path_buf());
    }
    // Fall back to model cache resolution (same logic as `USE MODEL`).
    larql_models::resolve_model_path(architecture)
}
