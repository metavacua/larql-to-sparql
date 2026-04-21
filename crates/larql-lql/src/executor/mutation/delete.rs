//! `DELETE FROM EDGES WHERE ...` — remove features via the patch overlay.

use crate::ast::{Condition, Value};
use crate::error::LqlError;
use crate::executor::Session;

impl Session {
    pub(crate) fn exec_delete(
        &mut self,
        conditions: &[Condition],
    ) -> Result<Vec<String>, LqlError> {
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

        // Collect deletions, then apply
        let deletes: Vec<(usize, usize)>;
        {
            let (_path, _config, patched) = self.require_patched_mut()?;

            if let (Some(layer), Some(feature)) = (layer_filter, feature_filter) {
                patched.delete_feature(layer, feature);
                deletes = vec![(layer, feature)];
            } else {
                let matches = patched
                    .base()
                    .find_features(entity_filter, None, layer_filter);
                if matches.is_empty() {
                    return Ok(vec!["  (no matching features found)".into()]);
                }
                for &(layer, feature) in &matches {
                    patched.delete_feature(layer, feature);
                }
                deletes = matches;
            }
        }

        // Record to patch session
        for &(layer, feature) in &deletes {
            if let Some(ref mut recording) = self.patch_recording {
                recording.operations.push(larql_vindex::PatchOp::Delete {
                    layer,
                    feature,
                    reason: None,
                });
            }
        }

        Ok(vec![format!(
            "Deleted {} features (patch overlay)",
            deletes.len()
        )])
    }
}
