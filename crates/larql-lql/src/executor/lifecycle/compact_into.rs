//! `COMPACT INTO VINDEX "out"` (VINDEX3): semantics-preserving
//! physical reorganisation of the bound container.
//!
//! COMPACT is not a second compiler: it refuses while the session
//! holds ANY overlay state, because removing overlays by materialising
//! meaning is COMPILE's job. The division keeps both operations
//! honest — COMPILE changes how meaning is stored to absorb edits;
//! COMPACT changes how the SAME meaning is stored, and the logical
//! DIFF is its proof instrument (`SemanticDiff(input, output) == ∅`).

use std::path::PathBuf;

use crate::error::LqlError;
use crate::executor::{Backend, Session};
use larql_vindex::format::vindex3::compact::compact_container;

impl Session {
    pub(crate) fn exec_compact_into(&mut self, output: &str) -> Result<Vec<String>, LqlError> {
        let Backend::Vindex3 { path, overlay, .. } = &self.backend else {
            return Err(LqlError::Execution(
                "COMPACT INTO VINDEX reorganises a VINDEX3 container; VINDEX2 compaction \
                 is the tiered COMPACT MINOR / COMPACT MAJOR"
                    .into(),
            ));
        };
        if overlay.num_overrides() > 0 || !overlay.knn_store.is_empty() {
            return Err(LqlError::Execution(
                "the session holds overlay state — COMPACT preserves meaning and cannot \
                 absorb edits; COMPILE CURRENT INTO VINDEX first (materialise the meaning), \
                 then COMPACT the result"
                    .into(),
            ));
        }

        let out = PathBuf::from(output);
        let report =
            compact_container(path, &out).map_err(|e| LqlError::exec("compact failed", e))?;

        let mut lines = vec![
            format!("Compacted: {} (VINDEX3)", out.display()),
            format!(
                "  {} segments carried byte-identically",
                report.carried_segments
            ),
        ];
        if report.dropped.is_empty() {
            lines.push("  nothing to drop — the container was already minimal".into());
        } else {
            lines.push(format!(
                "  {} unreferenced files dropped:",
                report.dropped.len()
            ));
            for name in &report.dropped {
                lines.push(format!("    - {name}"));
            }
        }
        Ok(lines)
    }
}
