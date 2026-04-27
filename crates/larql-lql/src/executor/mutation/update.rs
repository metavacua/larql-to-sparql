//! `UPDATE EDGES SET ... WHERE ...` — rewrite feature metadata via the
//! patch overlay.

use crate::ast::{Assignment, Condition, Value};
use crate::error::LqlError;
use crate::executor::Session;

impl Session {
    pub(crate) fn exec_update(
        &mut self,
        set: &[Assignment],
        conditions: &[Condition],
    ) -> Result<Vec<String>, LqlError> {
        let entity_filter = conditions
            .iter()
            .find(|c| c.field == "entity")
            .and_then(|c| {
                if let Value::String(ref s) = c.value {
                    Some(s.as_str())
                } else {
                    None
                }
            });
        let layer_filter = conditions
            .iter()
            .find(|c| c.field == "layer")
            .and_then(|c| {
                if let Value::Integer(n) = c.value {
                    Some(n as usize)
                } else {
                    None
                }
            });
        let feature_filter = conditions
            .iter()
            .find(|c| c.field == "feature")
            .and_then(|c| {
                if let Value::Integer(n) = c.value {
                    Some(n as usize)
                } else {
                    None
                }
            });

        // Collect updates, then record
        let mut update_ops: Vec<(usize, usize, larql_vindex::FeatureMeta)> = Vec::new();
        {
            let (_path, _config, patched) = self.require_patched_mut()?;

            // Fast path: explicit (layer, feature) — same shape as DELETE.
            // Bypasses `find_features` so the caller can target a single
            // slot directly without needing to match by entity/relation.
            let matches: Vec<(usize, usize)> =
                if let (Some(layer), Some(feature)) = (layer_filter, feature_filter) {
                    vec![(layer, feature)]
                } else {
                    patched
                        .base()
                        .find_features(entity_filter, None, layer_filter)
                };

            if matches.is_empty() {
                return Ok(vec!["  (no matching features found)".into()]);
            }

            for &(layer, feature) in &matches {
                if let Some(meta) = patched.feature_meta(layer, feature) {
                    let mut new_meta = meta;
                    for assignment in set {
                        match assignment.field.as_str() {
                            "target" | "top_token" => {
                                if let Value::String(ref s) = assignment.value {
                                    new_meta.top_token = s.clone();
                                }
                            }
                            "confidence" | "c_score" => {
                                if let Value::Number(n) = assignment.value {
                                    new_meta.c_score = n as f32;
                                } else if let Value::Integer(n) = assignment.value {
                                    new_meta.c_score = n as f32;
                                }
                            }
                            _ => {}
                        }
                    }
                    patched.update_feature_meta(layer, feature, new_meta.clone());
                    update_ops.push((layer, feature, new_meta));
                }
            }
        }

        // Record to patch session
        for (layer, feature, meta) in &update_ops {
            if let Some(ref mut recording) = self.patch_recording {
                recording.operations.push(larql_vindex::PatchOp::Update {
                    layer: *layer,
                    feature: *feature,
                    gate_vector_b64: None,
                    down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                        top_token: meta.top_token.clone(),
                        top_token_id: meta.top_token_id,
                        c_score: meta.c_score,
                    }),
                });
            }
        }

        Ok(vec![format!(
            "Updated {} features (patch overlay)",
            update_ops.len()
        )])
    }
}
