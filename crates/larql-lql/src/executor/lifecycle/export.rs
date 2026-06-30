//! EXPORT PATCH TO "gh://owner/repo" [BRANCH "branch"] [MESSAGE "msg"]
//!
//! Serializes the active patch recording as a Vindexfile + .vlp pair and
//! commits both atomically to a GitHub branch via `createCommitOnBranch`.
//!
//! Files written to the target repo:
//!   patches/{timestamp}.vlp  — the VindexPatch JSON
//!   Vindexfile               — FROM <base_model> + PATCH ./patches/{timestamp}.vlp

use larql_vindex::vindexfile::github::{graphql_commit, FileAddition, GhUrl};
use larql_vindex::{checksums, PatchOp, VindexPatch};

use crate::error::LqlError;
use crate::executor::{Backend, Session};

impl Session {
    pub(crate) fn exec_export_patch(
        &mut self,
        target: &str,
        branch: Option<&str>,
        message: Option<&str>,
    ) -> Result<Vec<String>, LqlError> {
        // ── 1. Require an active patch recording ─────────────────────────
        let recording = self.patch_recording.take().ok_or_else(|| {
            LqlError::Execution(
                "no active patch session — run BEGIN PATCH first, add facts, \
                 then EXPORT PATCH TO \"gh://...\"."
                    .into(),
            )
        })?;
        self.auto_patch = false;

        // ── 2. Parse the gh:// target URL ────────────────────────────────
        let gh = GhUrl::parse(target)
            .map_err(|e| LqlError::Execution(format!("invalid export target: {e}")))?;

        // ── 3. Build VindexPatch from the recording ───────────────────────
        let (model_name, base_checksum) = match &self.backend {
            Backend::Vindex { config, path, .. } => (
                config.model.clone(),
                checksums::compute_base_checksum(path).ok(),
            ),
            Backend::Weight { model_id, .. } => (model_id.clone(), None),
            _ => ("unknown".into(), None),
        };

        let patch = VindexPatch {
            version: 2,
            base_model: model_name.clone(),
            base_checksum,
            created_at: String::new(),
            description: None,
            author: None,
            tags: vec![],
            dependencies: None,
            operations: recording.operations,
        };

        // ── 4. Reject KNN ops — they belong to the SPARQL edge ───────────
        for op in &patch.operations {
            if matches!(op, PatchOp::InsertKnn { .. } | PatchOp::DeleteKnn { .. }) {
                return Err(LqlError::Execution(
                    "KNN operations cannot be exported via gh://; \
                     they serialize via the SPARQL edge. \
                     Use COMPOSE mode inserts for gh:// export."
                        .into(),
                ));
            }
        }

        let (ins, upd, del) = patch.counts();

        // ── 5. Serialize the patch to JSON ───────────────────────────────
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let patch_filename = format!("patches/{timestamp}.vlp");

        let patch_json = serde_json::to_string_pretty(&patch)
            .map_err(|e| LqlError::Execution(format!("patch serialization failed: {e}")))?;

        // ── 6. Build the Vindexfile text ──────────────────────────────────
        let from_line = if model_name == "unknown" || model_name.is_empty() {
            "# FROM <base_model>  -- unknown; fill in manually".to_string()
        } else if larql_vindex::is_hf_path(&model_name) {
            format!("FROM {model_name}")
        } else {
            format!("FROM hf://{model_name}")
        };

        let vindexfile_text = format!("{from_line}\nPATCH ./{patch_filename}\n");

        // ── 7. Commit to GitHub ───────────────────────────────────────────
        let use_branch = branch.unwrap_or(&gh.git_ref);
        let use_branch = if use_branch == "HEAD" {
            "main"
        } else {
            use_branch
        };

        let auto_msg =
            format!("feat(knowledge): export patch ({ins} inserts, {upd} updates, {del} deletes)");
        let commit_message = message.unwrap_or(&auto_msg);

        let files = vec![
            FileAddition {
                path: patch_filename.clone(),
                contents_base64: base64_encode(patch_json.as_bytes()),
            },
            FileAddition {
                path: "Vindexfile".to_string(),
                contents_base64: base64_encode(vindexfile_text.as_bytes()),
            },
        ];

        let commit_oid = graphql_commit(&gh.owner, &gh.repo, use_branch, commit_message, &files)
            .map_err(|e| LqlError::Execution(format!("GitHub commit failed: {e}")))?;

        let short_oid = &commit_oid[..commit_oid.len().min(7)];

        Ok(vec![format!(
            "Exported: gh://{owner}/{repo}@{branch}/{patch}\n  commit {short_oid}\n  {ins} inserts, {upd} updates, {del} deletes",
            owner = gh.owner,
            repo = gh.repo,
            branch = use_branch,
            patch = patch_filename,
        )])
    }
}

fn base64_encode(data: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(data)
}
