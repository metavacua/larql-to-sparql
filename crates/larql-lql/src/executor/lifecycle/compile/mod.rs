//! `COMPILE ... INTO {MODEL, VINDEX}` — dispatch + shared MEMIT fact
//! collection.

use std::path::PathBuf;

use crate::ast::{CompileConflict, CompileTarget, OutputFormat, VindexRef};
use crate::error::LqlError;
use crate::executor::{Backend, Session};

mod bake;
mod into_model;
mod into_vindex;

impl Session {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn exec_compile(
        &mut self,
        vindex: &VindexRef,
        output: &str,
        _format: Option<OutputFormat>,
        target: CompileTarget,
        on_conflict: Option<CompileConflict>,
    ) -> Result<Vec<String>, LqlError> {
        let vindex_path = match vindex {
            VindexRef::Current => {
                match &self.backend {
                    Backend::Vindex { path, .. } => path.clone(),
                    _ => return Err(LqlError::NoBackend),
                }
            }
            VindexRef::Path(p) => PathBuf::from(p),
        };

        match target {
            CompileTarget::Vindex => self.exec_compile_into_vindex(
                &vindex_path,
                output,
                on_conflict.unwrap_or(CompileConflict::LastWins),
            ),
            CompileTarget::Model => self.exec_compile_into_model(&vindex_path, output),
        }
    }
}

// ── Shared MEMIT fact collection (used by INTO MODEL and INTO VINDEX) ──

/// Collect MEMIT facts from BOTH applied patches on the PatchedVindex
/// AND the in-memory `patch_recording` of the current session.
/// Live INSERT ops go to `patch_recording` until SAVE PATCH; MEMIT
/// needs to see them for COMPILE to bake the uncommitted edits.
fn collect_memit_facts_with_recording(
    patched: &larql_vindex::PatchedVindex,
    vindex_path: &std::path::Path,
    recording_ops: &[larql_vindex::PatchOp],
) -> Result<Vec<larql_inference::MemitFact>, LqlError> {
    let tokenizer = larql_vindex::load_vindex_tokenizer(vindex_path)
        .map_err(|e| LqlError::exec("load tokenizer for MEMIT", e))?;

    let mut facts = Vec::new();
    let mut seen = std::collections::HashSet::new();

    let push_fact = |op: &larql_vindex::PatchOp,
                     facts: &mut Vec<larql_inference::MemitFact>,
                     seen: &mut std::collections::HashSet<_>|
     -> Result<(), LqlError> {
        if let larql_vindex::PatchOp::Insert {
            layer, entity, relation, target, ..
        } = op
        {
            let rel_str = relation.as_deref().unwrap_or("relation");
            let key = (entity.clone(), rel_str.to_string(), target.clone(), *layer);
            if !seen.insert(key) {
                return Ok(());
            }
            let rel_words = rel_str.replace(['-', '_'], " ");
            let prompt = format!("The {rel_words} of {entity} is");
            let encoding = tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|e| LqlError::exec("tokenize MEMIT prompt", e))?;
            let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();

            let spaced = format!(" {target}");
            let target_encoding = tokenizer
                .encode(spaced.as_str(), false)
                .map_err(|e| LqlError::exec("tokenize MEMIT target", e))?;
            let target_token_id = target_encoding.get_ids().first().copied().unwrap_or(0);

            facts.push(larql_inference::MemitFact {
                prompt_tokens,
                target_token_id,
                layer: *layer,
                label: format!("{entity} → {target} (L{layer})"),
            });
        }
        Ok(())
    };

    for patch in &patched.patches {
        for op in &patch.operations {
            push_fact(op, &mut facts, &mut seen)?;
        }
    }
    for op in recording_ops {
        push_fact(op, &mut facts, &mut seen)?;
    }

    Ok(facts)
}
