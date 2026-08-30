//! HuggingFace API helpers for repo-level state — remote LFS index lookup
//! and repo creation. Both are blocking HTTP calls used by the publish
//! orchestrator before any per-file upload runs.

use std::collections::HashMap;

use crate::error::VindexError;

use super::protocol::{hf_base, repo_type_plural, CONTENT_TYPE_NDJSON, HTTP_STATUS_CONFLICT};

/// List remote files and return `filename → lfs.oid` for every LFS-tracked
/// file at the repo root. Files without an `lfs.oid` (git-tracked small
/// text) are omitted; callers skip only what's in the map.
pub(super) fn fetch_remote_lfs_oids(
    repo_id: &str,
    token: &str,
    repo_type: &str,
) -> Result<HashMap<String, String>, VindexError> {
    let plural = repo_type_plural(repo_type);
    let base = hf_base();
    let url = format!("{base}/api/{plural}/{repo_id}/tree/main?recursive=true");
    let client = reqwest::blocking::Client::new();
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bearer {token}"))
        .send()
        .map_err(|e| VindexError::Parse(format!("HF tree fetch failed: {e}")))?;

    if !resp.status().is_success() {
        // 404 on a fresh repo → no remote files, can't skip anything.
        return Ok(HashMap::new());
    }

    let body: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("HF tree JSON: {e}")))?;
    Ok(parse_lfs_oid_index(&body))
}

/// Walk the HF tree-listing JSON and return `filename → lfs.oid` for
/// every LFS-tracked file. Files without an `lfs.oid` (small text /
/// directories) are omitted. Pulled out as a pure helper so the JSON
/// contract can be unit-tested without an HTTP server.
pub(super) fn parse_lfs_oid_index(body: &serde_json::Value) -> HashMap<String, String> {
    let arr = match body.as_array() {
        Some(a) => a,
        None => return HashMap::new(),
    };

    let mut out = HashMap::new();
    for entry in arr {
        if entry.get("type").and_then(|v| v.as_str()) != Some("file") {
            continue;
        }
        let path = match entry.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => continue,
        };
        if let Some(lfs_oid) = entry
            .get("lfs")
            .and_then(|v| v.get("oid"))
            .and_then(|v| v.as_str())
        {
            out.insert(path.to_string(), lfs_oid.to_string());
        }
    }
    out
}

/// The repo's current HEAD commit sha on `main` — the closest thing a
/// multi-commit publish (one commit per file, §publishing-design.md
/// §1) has to "the revision this publish produced": not a single
/// atomic commit, but "whatever `main` points to immediately after the
/// last file landed." Used to pin `RegistryArtifactRef::revision`
/// mechanically instead of leaving it to a human to look up and retype.
pub(super) fn fetch_repo_head_sha(
    repo_id: &str,
    token: &str,
    repo_type: &str,
) -> Result<String, VindexError> {
    let plural = repo_type_plural(repo_type);
    let base = hf_base();
    let url = format!("{base}/api/{plural}/{repo_id}");
    let client = reqwest::blocking::Client::new();
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bearer {token}"))
        .send()
        .map_err(|e| VindexError::Parse(format!("HF repo-info fetch failed: {e}")))?;

    if !resp.status().is_success() {
        return Err(VindexError::Parse(format!(
            "HF repo-info fetch for {repo_id} returned {}",
            resp.status()
        )));
    }

    let body: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("HF repo-info JSON: {e}")))?;
    body.get("sha")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| {
            VindexError::Parse(format!("HF repo-info for {repo_id} carried no 'sha' field"))
        })
}

pub(super) fn create_hf_repo(
    repo_id: &str,
    token: &str,
    repo_type: &str,
    private: bool,
) -> Result<(), VindexError> {
    let client = reqwest::blocking::Client::new();
    let url = format!("{}/api/repos/create", hf_base());
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .json(&create_hf_repo_body(repo_id, repo_type, private))
        .send()
        .map_err(|e| VindexError::Parse(format!("HF API error: {e}")))?;

    // 409 Conflict = already exists, that's fine
    if resp.status().is_success() || resp.status().as_u16() == HTTP_STATUS_CONFLICT {
        Ok(())
    } else {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        Err(VindexError::Parse(format!(
            "HF repo create failed ({status}): {body}"
        )))
    }
}

/// The `POST /api/repos/create` body. `repo_id`'s owner (everything before
/// the last `/`) becomes `organization` — HF's create-repo API defaults an
/// omitted `organization` to the *token's own* namespace, never the caller's
/// intended one. Without this, `--repo larql/granite-4.1-3b` published from
/// a `chrishayuk`-owned token silently landed under `chrishayuk/granite-4.1-3b`
/// instead: the create call "succeeded" against the wrong namespace, and the
/// very next preupload call (which does target the real `repo_id`) 404'd on
/// a repo that was never actually created. A bare `repo_id` with no `/` (no
/// owner named at all) omits the field, matching the pre-fix behaviour for
/// that case — there is no intended namespace to override.
pub(super) fn create_hf_repo_body(
    repo_id: &str,
    repo_type: &str,
    private: bool,
) -> serde_json::Value {
    let name = repo_id.split('/').next_back().unwrap_or(repo_id);
    let mut body = serde_json::json!({
        "name": name,
        "type": repo_type,
        "private": private,
    });
    if let Some((owner, _)) = repo_id.rsplit_once('/') {
        body["organization"] = serde_json::Value::String(owner.to_string());
    }
    body
}

/// Flip an already-created repo's visibility. Used at RELEASE
/// (docs/vindex-factory.md §7/§8.3): a build publishes PRIVATE, verifies
/// the published bytes, and only then flips PUBLIC — nothing goes live
/// unverified. `PUT /api/{repo_type}s/{repo_id}/settings` with
/// `{"private": bool}` (HF Hub API, confirmed against the OpenAPI spec
/// at huggingface.co/.well-known/openapi.md — the modern `visibility`
/// enum field also works, but `private` is the simpler two-state case
/// this needs).
pub(super) fn update_repo_visibility(
    repo_id: &str,
    token: &str,
    repo_type: &str,
    private: bool,
) -> Result<(), VindexError> {
    let plural = repo_type_plural(repo_type);
    let url = format!("{}/api/{plural}/{repo_id}/settings", hf_base());
    let client = reqwest::blocking::Client::new();
    let resp = client
        .put(&url)
        .header("Authorization", format!("Bearer {token}"))
        .json(&serde_json::json!({ "private": private }))
        .send()
        .map_err(|e| VindexError::Parse(format!("HF API error: {e}")))?;

    if resp.status().is_success() {
        Ok(())
    } else {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        Err(VindexError::Parse(format!(
            "HF repo visibility update failed ({status}): {body}"
        )))
    }
}

/// Every file path in the repo, whether LFS-tracked or not.
///
/// [`fetch_remote_lfs_oids`] deliberately drops non-LFS entries because it
/// only answers "can I skip this upload". Pruning needs the full set: the
/// small `*_manifest.json` siblings of a renamed weight file are git-tracked,
/// not LFS, and leaving them behind is what makes a half-renamed vindex load
/// the wrong pair.
pub(super) fn fetch_remote_file_paths(
    repo_id: &str,
    token: &str,
    repo_type: &str,
) -> Result<Vec<String>, VindexError> {
    let plural = repo_type_plural(repo_type);
    let url = format!(
        "{}/api/{plural}/{repo_id}/tree/main?recursive=true",
        hf_base()
    );
    let resp = reqwest::blocking::Client::new()
        .get(&url)
        .header("Authorization", format!("Bearer {token}"))
        .send()
        .map_err(|e| VindexError::Parse(format!("HF tree fetch failed: {e}")))?;
    if !resp.status().is_success() {
        // Fresh repo → nothing to prune.
        return Ok(Vec::new());
    }
    let body: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("HF tree JSON: {e}")))?;
    Ok(parse_file_paths(&body))
}

/// Pull `path` from every `type == "file"` entry of a tree listing.
pub(super) fn parse_file_paths(body: &serde_json::Value) -> Vec<String> {
    let arr = match body.as_array() {
        Some(a) => a,
        None => return Vec::new(),
    };
    arr.iter()
        .filter(|e| e.get("type").and_then(|v| v.as_str()) == Some("file"))
        .filter_map(|e| e.get("path").and_then(|v| v.as_str()).map(str::to_string))
        .collect()
}

/// Delete `paths` from the repo in one commit.
pub(super) fn delete_remote_files(
    repo_id: &str,
    token: &str,
    repo_type: &str,
    paths: &[String],
) -> Result<(), VindexError> {
    if paths.is_empty() {
        return Ok(());
    }
    let plural = repo_type_plural(repo_type);
    let url = format!("{}/api/{plural}/{repo_id}/commit/main", hf_base());

    let mut ndjson = serde_json::to_string(&serde_json::json!({
        "key": "header",
        "value": { "summary": format!("Prune {} file(s) not in the source vindex", paths.len()) },
    }))
    .unwrap();
    ndjson.push('\n');
    for p in paths {
        ndjson.push_str(
            &serde_json::to_string(&serde_json::json!({
                "key": "deletedFile",
                "value": { "path": p },
            }))
            .unwrap(),
        );
        ndjson.push('\n');
    }

    let resp = reqwest::blocking::Client::new()
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Content-Type", CONTENT_TYPE_NDJSON)
        .body(ndjson)
        .send()
        .map_err(|e| VindexError::Parse(format!("prune commit failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!(
            "prune commit ({status}): {body}"
        )));
    }
    Ok(())
}

/// Repo files that `publish` never writes and must therefore never prune:
/// git plumbing and the model card, which is authored separately
/// (`larql card`) and lives only on the Hub.
pub(super) fn is_prune_exempt(path: &str) -> bool {
    path.starts_with('.') || path == "README.md" || path.ends_with("/README.md")
}
