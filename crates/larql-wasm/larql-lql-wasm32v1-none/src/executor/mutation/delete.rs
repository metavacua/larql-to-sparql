//! `DELETE FROM EDGES WHERE ...` — remove features via the patch overlay.

use crate::ast::Condition;
use crate::error::LqlError;
use crate::executor::Session;

use super::{relation_filter_matches, WhereFilters};

impl Session {
    pub(crate) fn exec_delete(
        &mut self,
        conditions: &[Condition],
    ) -> Result<Vec<String>, LqlError> {
        let filters = WhereFilters::from_conditions(conditions);

        // Collect candidates with a readonly borrow before mutating the
        // patch overlay, so relation predicates cannot be dropped silently.
        let deletes = {
            let (_path, _config, patched) = self.require_vindex()?;
            let candidates = filters.resolve_candidates(patched.base());

            let mut matches = Vec::new();
            for (layer, feature) in candidates {
                if relation_filter_matches(
                    self.relation_classifier(),
                    filters.relation,
                    layer,
                    feature,
                )? {
                    matches.push((layer, feature));
                }
            }
            matches
        };

        if deletes.is_empty() {
            return Ok(vec!["  (no matching features found)".into()]);
        }

        {
            let (_path, _config, patched) = self.require_patched_mut()?;
            for &(layer, feature) in &deletes {
                patched.delete_feature(layer, feature);
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
