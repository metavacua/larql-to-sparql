//! HuggingFace Hub integration — download and upload vindexes.
//!
//! Vindexes are stored as HuggingFace dataset repos. Each file in the vindex
//! directory maps 1:1 to a file in the repo. HuggingFace's CDN handles
//! distribution, caching, and access control.
//!
//! ```text
//! # Download a vindex
//! larql> USE "hf://chrishayuk/gemma-3-4b-it-vindex";
//!
//! # Upload a vindex
//! larql publish gemma3-4b.vindex --repo chrishayuk/gemma-3-4b-it-vindex
//! ```

use std::path::{Path, PathBuf};

use crate::error::VindexError;

/// The files that make up a vindex, in priority order for lazy loading.
const VINDEX_CORE_FILES: &[&str] = &[
    "index.json",
    "tokenizer.json",
    "gate_vectors.bin",
    "embeddings.bin",
    "down_meta.bin",
    "down_meta.jsonl",
    "relation_clusters.json",
    "feature_labels.json",
];

const VINDEX_WEIGHT_FILES: &[&str] = &[
    "attn_weights.bin",
    "norms.bin",
    "up_weights.bin",
    "down_weights.bin",
    "lm_head.bin",
    "weight_manifest.json",
];

/// Resolve an `hf://` path to a local directory, downloading if needed.
///
/// Supports:
/// - `hf://user/repo` — downloads the full dataset repo
/// - `hf://user/repo@revision` — specific revision/tag
///
/// Files are cached in the HuggingFace cache directory (~/.cache/huggingface/).
/// Only downloads files that don't already exist locally.
pub fn resolve_hf_vindex(hf_path: &str) -> Result<PathBuf, VindexError> {
    let path = hf_path.strip_prefix("hf://")
        .ok_or_else(|| VindexError::Parse(format!("not an hf:// path: {hf_path}")))?;

    // Parse repo and optional revision
    let (repo_id, revision) = if let Some((repo, rev)) = path.split_once('@') {
        (repo.to_string(), Some(rev.to_string()))
    } else {
        (path.to_string(), None)
    };

    // Use hf-hub to download
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| VindexError::Parse(format!("HuggingFace API init failed: {e}")))?;

    let repo = if let Some(ref rev) = revision {
        api.repo(hf_hub::Repo::with_revision(
            repo_id.clone(),
            hf_hub::RepoType::Dataset,
            rev.clone(),
        ))
    } else {
        api.repo(hf_hub::Repo::new(
            repo_id.clone(),
            hf_hub::RepoType::Dataset,
        ))
    };

    // Download index.json first (small, tells us what we need)
    let index_path = repo.get("index.json")
        .map_err(|e| VindexError::Parse(format!(
            "failed to download index.json from hf://{}: {e}", repo_id
        )))?;

    let vindex_dir = index_path.parent()
        .ok_or_else(|| VindexError::Parse("cannot determine vindex directory".into()))?
        .to_path_buf();

    // Download core files (needed for browse)
    for filename in VINDEX_CORE_FILES {
        if *filename == "index.json" {
            continue; // already downloaded
        }
        let _ = repo.get(filename); // optional file, skip if missing
    }

    Ok(vindex_dir)
}

/// Download additional weight files for inference/compile.
/// Called lazily when INFER or COMPILE is first used.
pub fn download_hf_weights(hf_path: &str) -> Result<(), VindexError> {
    let path = hf_path.strip_prefix("hf://")
        .ok_or_else(|| VindexError::Parse(format!("not an hf:// path: {hf_path}")))?;

    let (repo_id, revision) = if let Some((repo, rev)) = path.split_once('@') {
        (repo.to_string(), Some(rev.to_string()))
    } else {
        (path.to_string(), None)
    };

    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| VindexError::Parse(format!("HuggingFace API init failed: {e}")))?;

    let repo = if let Some(ref rev) = revision {
        api.repo(hf_hub::Repo::with_revision(
            repo_id.clone(),
            hf_hub::RepoType::Dataset,
            rev.clone(),
        ))
    } else {
        api.repo(hf_hub::Repo::new(
            repo_id.clone(),
            hf_hub::RepoType::Dataset,
        ))
    };

    for filename in VINDEX_WEIGHT_FILES {
        let _ = repo.get(filename); // optional, skip if not in repo
    }

    Ok(())
}

/// Upload a local vindex directory to HuggingFace as a dataset repo.
///
/// Requires HF_TOKEN environment variable or ~/.huggingface/token.
pub fn publish_vindex(
    vindex_dir: &Path,
    repo_id: &str,
    callbacks: &mut dyn PublishCallbacks,
) -> Result<String, VindexError> {
    if !vindex_dir.is_dir() {
        return Err(VindexError::NotADirectory(vindex_dir.to_path_buf()));
    }

    // Check index.json exists
    let index_path = vindex_dir.join("index.json");
    if !index_path.exists() {
        return Err(VindexError::Parse(format!(
            "not a vindex directory (no index.json): {}", vindex_dir.display()
        )));
    }

    // Get HF token
    let token = get_hf_token()?;

    callbacks.on_start(repo_id);

    // Create the dataset repo (or confirm it exists)
    create_hf_dataset_repo(repo_id, &token)?;

    // Upload each file
    let mut files: Vec<PathBuf> = std::fs::read_dir(vindex_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    files.sort();

    for file_path in &files {
        let filename = file_path.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        let size = std::fs::metadata(file_path)
            .map(|m| m.len())
            .unwrap_or(0);

        callbacks.on_file_start(&filename, size);

        upload_file_to_hf(repo_id, &token, file_path, &filename)?;

        callbacks.on_file_done(&filename);
    }

    let url = format!("https://huggingface.co/datasets/{}", repo_id);
    callbacks.on_complete(&url);

    Ok(url)
}

/// Callbacks for publish progress.
pub trait PublishCallbacks {
    fn on_start(&mut self, _repo: &str) {}
    fn on_file_start(&mut self, _filename: &str, _size: u64) {}
    fn on_file_done(&mut self, _filename: &str) {}
    fn on_complete(&mut self, _url: &str) {}
}

pub struct SilentPublishCallbacks;
impl PublishCallbacks for SilentPublishCallbacks {}

// ═══════════════════════════════════════════════════════════════
// HuggingFace HTTP API helpers
// ═══════════════════════════════════════════════════════════════

fn get_hf_token() -> Result<String, VindexError> {
    // Try environment variable first
    if let Ok(token) = std::env::var("HF_TOKEN") {
        return Ok(token);
    }

    // Try token file
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let token_path = PathBuf::from(&home).join(".huggingface").join("token");
    if token_path.exists() {
        let token = std::fs::read_to_string(&token_path)?;
        return Ok(token.trim().to_string());
    }

    // Try newer cache location
    let token_path = PathBuf::from(&home).join(".cache").join("huggingface").join("token");
    if token_path.exists() {
        let token = std::fs::read_to_string(&token_path)?;
        return Ok(token.trim().to_string());
    }

    Err(VindexError::Parse(
        "HuggingFace token not found. Set HF_TOKEN or run `huggingface-cli login`.".into()
    ))
}

fn create_hf_dataset_repo(repo_id: &str, token: &str) -> Result<(), VindexError> {
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post("https://huggingface.co/api/repos/create")
        .header("Authorization", format!("Bearer {token}"))
        .json(&serde_json::json!({
            "name": repo_id.split('/').next_back().unwrap_or(repo_id),
            "type": "dataset",
            "private": false,
        }))
        .send()
        .map_err(|e| VindexError::Parse(format!("HF API error: {e}")))?;

    // 409 = already exists, that's fine
    if resp.status().is_success() || resp.status().as_u16() == 409 {
        Ok(())
    } else {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        Err(VindexError::Parse(format!("HF repo create failed ({status}): {body}")))
    }
}

fn upload_file_to_hf(
    repo_id: &str,
    token: &str,
    local_path: &Path,
    remote_filename: &str,
) -> Result<(), VindexError> {
    let data = std::fs::read(local_path)?;

    let url = format!(
        "https://huggingface.co/api/datasets/{}/upload/main/{}",
        repo_id, remote_filename
    );

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(3600)) // 1 hour for large files
        .build()
        .map_err(|e| VindexError::Parse(format!("HTTP client error: {e}")))?;

    let resp = client
        .put(&url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Content-Type", "application/octet-stream")
        .body(data)
        .send()
        .map_err(|e| VindexError::Parse(format!("upload failed: {e}")))?;

    if resp.status().is_success() {
        Ok(())
    } else {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        Err(VindexError::Parse(format!(
            "upload {} failed ({status}): {body}", remote_filename
        )))
    }
}

/// Check if a path is an hf:// reference.
pub fn is_hf_path(path: &str) -> bool {
    path.starts_with("hf://")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_hf_path() {
        assert!(is_hf_path("hf://chrishayuk/gemma-3-4b-it-vindex"));
        assert!(is_hf_path("hf://user/repo@v1.0"));
        assert!(!is_hf_path("./local.vindex"));
        assert!(!is_hf_path("/absolute/path"));
    }

    #[test]
    fn test_parse_hf_path() {
        let path = "hf://chrishayuk/gemma-3-4b-it-vindex@v2.0";
        let stripped = path.strip_prefix("hf://").unwrap();
        let (repo, rev) = stripped.split_once('@').unwrap();
        assert_eq!(repo, "chrishayuk/gemma-3-4b-it-vindex");
        assert_eq!(rev, "v2.0");
    }
}
