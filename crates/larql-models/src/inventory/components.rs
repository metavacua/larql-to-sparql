//! Generic nested-component topology reader.
//!
//! A multimodal checkpoint nests sibling components as `*_config` objects
//! (`vision_config`, `audio_config`, …). The main `ModelConfig` parser owns
//! `text_config`/`language_config`; every *other* component used to be a
//! presence check, leaving its whole subtree unjudged. This reader extracts
//! the generic topology vocabulary — depth, widths, heads, layer types,
//! norm eps, activation, rope, patch geometry — into a typed
//! [`ComponentTopology`], keyed by no family name.
//!
//! **Consumed-key honesty by construction**: the reader records the exact
//! path of every key it reads into [`ComponentReading::consumed_paths`], and
//! key classification credits precisely that set. There is no second
//! registry to drift — a key is `consumed` iff a read here (or in the main
//! parser) actually stored it.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Config-object suffix that marks a nested component (`vision_config`).
const COMPONENT_CONFIG_SUFFIX: &str = "_config";

/// Components owned by the main `ModelConfig` parser, not this reader.
const MAIN_PARSER_COMPONENTS: &[&str] = &["text_config", "language_config"];

/// One nested component's declared topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentTopology {
    /// Component name (`vision` for `vision_config`).
    pub name: String,
    pub model_type: Option<String>,
    pub hidden_size: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub num_layers: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub head_dim: Option<usize>,
    /// Per-layer attention kinds, verbatim (`window_attention`,
    /// `full_attention`, …) — the vocabulary is the component's own.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_types: Option<Vec<String>>,
    pub norm_eps: Option<f64>,
    /// Norm kind, named by which epsilon spelling the component declares
    /// (`layer_norm_eps` → LayerNorm, `rms_norm_eps` → RMSNorm). Absent
    /// when neither is declared — a fact, not a default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub norm_kind: Option<crate::config::NormType>,
    pub hidden_act: Option<String>,
    pub rope_theta: Option<f64>,
    pub rope_type: Option<String>,
    pub max_position_embeddings: Option<usize>,
    /// Patch/positional geometry for perception towers.
    #[serde(skip_serializing_if = "PatchGeometry::is_empty")]
    pub patch: PatchGeometry,
}

/// Perception-tower patch geometry. All optional; absent fields are absent
/// facts, not defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatchGeometry {
    pub patch_size: Option<usize>,
    pub patch_temporal: Option<usize>,
    pub merge_size: Option<usize>,
    pub pos_emb_height: Option<usize>,
    pub pos_emb_width: Option<usize>,
    pub image_size: Option<usize>,
    pub num_channels: Option<usize>,
}

impl PatchGeometry {
    pub fn is_empty(&self) -> bool {
        self.patch_size.is_none()
            && self.patch_temporal.is_none()
            && self.merge_size.is_none()
            && self.pos_emb_height.is_none()
            && self.pos_emb_width.is_none()
            && self.image_size.is_none()
            && self.num_channels.is_none()
    }
}

/// A component reading: the topology plus the exact key paths consumed to
/// produce it (paths are full, from the config root).
pub struct ComponentReading {
    pub topology: ComponentTopology,
    pub consumed_paths: BTreeSet<String>,
}

/// Read every nested component from a parsed config root, in key order.
pub fn read_components(config: &Value) -> Vec<ComponentReading> {
    let Some(map) = config.as_object() else {
        return Vec::new();
    };
    map.iter()
        .filter(|(key, value)| {
            value.is_object()
                && key.ends_with(COMPONENT_CONFIG_SUFFIX)
                && !MAIN_PARSER_COMPONENTS.contains(&key.as_str())
        })
        .map(|(key, value)| read_component(key, value))
        .collect()
}

/// A cursor that reads keys out of one component object while recording
/// the path of every key it actually resolved — absent keys are never
/// marked, so the consumed set is exactly the present-and-read keys.
struct Cursor<'a> {
    root_key: &'a str,
    object: &'a Value,
    consumed: BTreeSet<String>,
}

impl<'a> Cursor<'a> {
    fn mark(&mut self, rel_path: &str) {
        self.consumed
            .insert(format!("{}.{rel_path}", self.root_key));
    }

    fn get(&mut self, rel_path: &str) -> Option<&'a Value> {
        let mut node = self.object;
        for segment in rel_path.split('.') {
            node = node.get(segment)?;
        }
        self.mark(rel_path);
        Some(node)
    }

    fn usize_at(&mut self, rel_path: &str) -> Option<usize> {
        self.get(rel_path)?.as_u64().map(|v| v as usize)
    }

    fn f64_at(&mut self, rel_path: &str) -> Option<f64> {
        self.get(rel_path)?.as_f64()
    }

    fn string_at(&mut self, rel_path: &str) -> Option<String> {
        self.get(rel_path)?.as_str().map(str::to_string)
    }
}

fn read_component(root_key: &str, object: &Value) -> ComponentReading {
    let mut cursor = Cursor {
        root_key,
        object,
        consumed: BTreeSet::new(),
    };
    let name = root_key
        .strip_suffix(COMPONENT_CONFIG_SUFFIX)
        .unwrap_or(root_key)
        .to_string();
    let layer_types = cursor.get("layer_types").and_then(|lt| {
        lt.as_array().map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
    });
    let topology = ComponentTopology {
        name,
        model_type: cursor.string_at("model_type"),
        hidden_size: cursor.usize_at("hidden_size"),
        intermediate_size: cursor.usize_at("intermediate_size"),
        num_layers: cursor.usize_at("num_hidden_layers"),
        num_attention_heads: cursor.usize_at("num_attention_heads"),
        num_key_value_heads: cursor.usize_at("num_key_value_heads"),
        head_dim: cursor.usize_at("head_dim"),
        layer_types,
        norm_eps: cursor
            .f64_at("layer_norm_eps")
            .or_else(|| cursor.f64_at("rms_norm_eps")),
        // The spelling that matched names the kind; read order above.
        norm_kind: if cursor.f64_at("layer_norm_eps").is_some() {
            Some(crate::config::NormType::LayerNorm)
        } else if cursor.f64_at("rms_norm_eps").is_some() {
            Some(crate::config::NormType::RmsNorm)
        } else {
            None
        },
        hidden_act: cursor
            .string_at("hidden_act")
            .or_else(|| cursor.string_at("hidden_activation")),
        rope_theta: cursor.f64_at("rope_parameters.rope_theta"),
        rope_type: cursor.string_at("rope_parameters.rope_type"),
        max_position_embeddings: cursor.usize_at("max_position_embeddings"),
        patch: PatchGeometry {
            patch_size: cursor.usize_at("patch_size"),
            patch_temporal: cursor.usize_at("patch_temporal"),
            merge_size: cursor.usize_at("merge_size"),
            pos_emb_height: cursor.usize_at("pos_emb_height"),
            pos_emb_width: cursor.usize_at("pos_emb_width"),
            image_size: cursor.usize_at("image_size"),
            num_channels: cursor.usize_at("num_channels"),
        },
    };
    ComponentReading {
        topology,
        consumed_paths: cursor.consumed,
    }
}
