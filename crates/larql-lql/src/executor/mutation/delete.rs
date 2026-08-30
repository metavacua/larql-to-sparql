//! `DELETE FROM EDGES WHERE ...` — tombstone features via the session
//! overlay (V2's `PatchedVindex`, V3's `KnowledgeOverlay`).
//!
//! One logical operation, written once: candidates resolve through the
//! knowledge seam, the tombstone lands in whichever overlay the
//! binding holds, and the same `PatchOp::Delete` is recorded — so a
//! saved patch replays identically on either backend.

use crate::ast::Condition;
use crate::error::LqlError;
use crate::executor::{Backend, Session};

use super::{relation_filter_matches, WhereFilters};

impl Session {
    pub(crate) fn exec_delete(
        &mut self,
        conditions: &[Condition],
    ) -> Result<Vec<String>, LqlError> {
        let filters = WhereFilters::from_conditions(conditions);

        // Collect candidates with a readonly borrow before mutating the
        // overlay, so relation predicates cannot be dropped silently.
        let deletes = {
            let ctx = self.browse()?;
            let candidates = filters.resolve_candidates(&ctx.source);

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

        match &mut self.backend {
            Backend::Vindex { patched, .. } => {
                for &(layer, feature) in &deletes {
                    patched.delete_feature(layer, feature);
                }
            }
            Backend::Vindex3 { overlay, .. } => {
                for &(layer, feature) in &deletes {
                    overlay.delete_feature(layer, feature);
                }
            }
            _ => unreachable!("browse() refused every other backend above"),
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
