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

/// Re-exported from hf-hub 0.5 so callers don't have to depend on
/// `hf_hub` directly. Implement this trait on an `indicatif::ProgressBar`
/// wrapper (or similar) to get per-file progress + resume behaviour out
/// of [`resolve_hf_vindex_with_progress`].
pub use hf_hub::api::Progress as DownloadProgress;

/// Like [`resolve_hf_vindex`], but drives a progress reporter per file.
/// hf-hub handles `.incomplete` partial-file resume internally — if the
/// download is interrupted, the next call picks up from where it left off.
///
/// `progress` is a factory: called once per file about to be downloaded,
/// with the filename. Return a fresh `DownloadProgress` — typically an
/// `indicatif::ProgressBar` fetched from a `MultiProgress`.
pub fn resolve_hf_vindex_with_progress<F, P>(
    hf_path: &str,
    mut progress: F,
) -> Result<PathBuf, VindexError>
where
    F: FnMut(&str) -> P,
    P: DownloadProgress,
{
    let path = hf_path
        .strip_prefix("hf://")
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
        api.repo(hf_hub::Repo::new(repo_id.clone(), hf_hub::RepoType::Dataset))
    };

    let index_path = repo
        .download_with_progress("index.json", progress("index.json"))
        .map_err(|e| {
            VindexError::Parse(format!(
                "failed to download index.json from hf://{repo_id}: {e}"
            ))
        })?;
    let vindex_dir = index_path
        .parent()
        .ok_or_else(|| VindexError::Parse("cannot determine vindex directory".into()))?
        .to_path_buf();

    for filename in VINDEX_CORE_FILES {
        if *filename == "index.json" {
            continue;
        }
        let _ = repo.download_with_progress(filename, progress(filename));
    }
    Ok(vindex_dir)
}

/// Options controlling [`publish_vindex_with_opts`]. Kept as a struct so
/// the signature can grow without breaking callers.
#[derive(Clone, Debug, Default)]
pub struct PublishOptions {
    /// When true, skip uploading LFS-tracked files whose local SHA256
    /// already matches the remote `lfs.oid`. Small files (git-tracked
    /// json / manifest) are always re-uploaded — their text is tiny and
    /// the git blob SHA-1 format isn't directly derivable from the file
    /// content SHA256 without a separate hash.
    pub skip_unchanged: bool,
}

impl PublishOptions {
    pub fn skip_unchanged() -> Self {
        Self { skip_unchanged: true }
    }
}

/// Upload a local vindex directory to HuggingFace as a dataset repo.
///
/// Equivalent to `publish_vindex_with_opts(dir, repo_id, &PublishOptions::default(), cb)`.
/// Requires HF_TOKEN environment variable or ~/.huggingface/token.
pub fn publish_vindex(
    vindex_dir: &Path,
    repo_id: &str,
    callbacks: &mut dyn PublishCallbacks,
) -> Result<String, VindexError> {
    publish_vindex_with_opts(vindex_dir, repo_id, &PublishOptions::default(), callbacks)
}

/// Upload a vindex directory with explicit options. See [`PublishOptions`].
pub fn publish_vindex_with_opts(
    vindex_dir: &Path,
    repo_id: &str,
    opts: &PublishOptions,
    callbacks: &mut dyn PublishCallbacks,
) -> Result<String, VindexError> {
    if !vindex_dir.is_dir() {
        return Err(VindexError::NotADirectory(vindex_dir.to_path_buf()));
    }
    let index_path = vindex_dir.join("index.json");
    if !index_path.exists() {
        return Err(VindexError::Parse(format!(
            "not a vindex directory (no index.json): {}",
            vindex_dir.display()
        )));
    }

    let token = get_hf_token()?;
    callbacks.on_start(repo_id);
    create_hf_dataset_repo(repo_id, &token)?;

    // Pull remote LFS index so we can skip unchanged files. Non-fatal
    // if the tree API errors (brand-new repo returns 404 here) — we just
    // fall back to "upload everything".
    let remote_lfs: std::collections::HashMap<String, String> = if opts.skip_unchanged {
        fetch_remote_lfs_oids(repo_id, &token).unwrap_or_default()
    } else {
        std::collections::HashMap::new()
    };

    let mut files: Vec<PathBuf> = std::fs::read_dir(vindex_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    files.sort();

    for file_path in &files {
        let filename = file_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let size = std::fs::metadata(file_path).map(|m| m.len()).unwrap_or(0);

        // Skip-if-unchanged: compare local SHA256 against remote lfs.oid.
        if opts.skip_unchanged {
            if let Some(remote_sha) = remote_lfs.get(&filename) {
                if let Ok(local_sha) = crate::format::checksums::sha256_file(file_path) {
                    if local_sha == *remote_sha {
                        callbacks.on_file_skipped(&filename, size, remote_sha);
                        continue;
                    }
                }
            }
        }

        callbacks.on_file_start(&filename, size);
        upload_file_to_hf(repo_id, &token, file_path, &filename, callbacks)?;
        callbacks.on_file_done(&filename);
    }

    let url = format!("https://huggingface.co/datasets/{}", repo_id);
    callbacks.on_complete(&url);
    Ok(url)
}

/// List remote files and return `filename → lfs.oid` for every LFS-tracked
/// file at the repo root. Files without an `lfs.oid` (git-tracked small
/// text) are omitted; callers skip only what's in the map.
fn fetch_remote_lfs_oids(
    repo_id: &str,
    token: &str,
) -> Result<std::collections::HashMap<String, String>, VindexError> {
    let url = format!(
        "https://huggingface.co/api/datasets/{repo_id}/tree/main?recursive=true"
    );
    let client = reqwest::blocking::Client::new();
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bearer {token}"))
        .send()
        .map_err(|e| VindexError::Parse(format!("HF tree fetch failed: {e}")))?;

    if !resp.status().is_success() {
        // 404 on a fresh repo → no remote files, can't skip anything.
        return Ok(std::collections::HashMap::new());
    }

    let body: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("HF tree JSON: {e}")))?;
    let arr = match body.as_array() {
        Some(a) => a,
        None => return Ok(std::collections::HashMap::new()),
    };

    let mut out = std::collections::HashMap::new();
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
    Ok(out)
}

/// Callbacks for publish progress.
pub trait PublishCallbacks {
    fn on_start(&mut self, _repo: &str) {}
    fn on_file_start(&mut self, _filename: &str, _size: u64) {}
    /// Fired periodically during the upload with cumulative bytes sent
    /// for the current file. Default no-op. Implement to render a live
    /// progress bar; indicatif wrappers live in the CLI layer to stay
    /// version-agnostic here.
    fn on_file_progress(&mut self, _filename: &str, _bytes_sent: u64, _total_bytes: u64) {}
    fn on_file_done(&mut self, _filename: &str) {}
    /// Fired when [`PublishOptions::skip_unchanged`] matches the remote
    /// `lfs.oid` and the upload is skipped. Default no-op so existing
    /// callbacks don't need to change.
    fn on_file_skipped(&mut self, _filename: &str, _size: u64, _sha256: &str) {}
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

/// Counting `Read` adapter — increments a shared atomic on every read so
/// a poll thread can report upload progress without per-chunk syscalls.
struct CountingReader<R: std::io::Read> {
    inner: R,
    counter: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl<R: std::io::Read> std::io::Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.counter
            .fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(n)
    }
}

fn upload_file_to_hf(
    repo_id: &str,
    token: &str,
    local_path: &Path,
    remote_filename: &str,
    callbacks: &mut dyn PublishCallbacks,
) -> Result<(), VindexError> {
    use std::sync::atomic::Ordering;
    use std::sync::mpsc::TryRecvError;
    use std::time::Duration;

    let size = std::fs::metadata(local_path)?.len();
    let file = std::fs::File::open(local_path)?;

    // Streaming body so a 27 GB server slice doesn't get pulled into RAM.
    let counter = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let reader = CountingReader {
        inner: file,
        counter: counter.clone(),
    };
    let body = reqwest::blocking::Body::sized(reader, size);

    let url = format!(
        "https://huggingface.co/api/datasets/{}/upload/main/{}",
        repo_id, remote_filename
    );

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(3600)) // 1 hour — large slice uploads aren't fast
        .build()
        .map_err(|e| VindexError::Parse(format!("HTTP client error: {e}")))?;

    // Run the upload on a worker thread so this thread can poll the
    // byte counter + fire `on_file_progress` periodically.
    let url_owned = url.clone();
    let token_owned = token.to_string();
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        let result = client
            .put(&url_owned)
            .header("Authorization", format!("Bearer {token_owned}"))
            .header("Content-Type", "application/octet-stream")
            .body(body)
            .send();
        let _ = tx.send(result);
    });

    loop {
        match rx.try_recv() {
            Ok(resp) => {
                let _ = handle.join();
                let resp = resp
                    .map_err(|e| VindexError::Parse(format!("upload failed: {e}")))?;
                if resp.status().is_success() {
                    // Final tick so the bar reads 100 % even if the last
                    // read() didn't line up exactly with `size`.
                    callbacks.on_file_progress(remote_filename, size, size);
                    return Ok(());
                } else {
                    let status = resp.status();
                    let body = resp.text().unwrap_or_default();
                    return Err(VindexError::Parse(format!(
                        "upload {} failed ({status}): {body}",
                        remote_filename
                    )));
                }
            }
            Err(TryRecvError::Empty) => {
                let sent = counter.load(Ordering::Relaxed);
                callbacks.on_file_progress(remote_filename, sent, size);
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(TryRecvError::Disconnected) => {
                let _ = handle.join();
                return Err(VindexError::Parse(
                    "upload worker terminated unexpectedly".into(),
                ));
            }
        }
    }
}

/// Check if a path is an hf:// reference.
pub fn is_hf_path(path: &str) -> bool {
    path.starts_with("hf://")
}

// ═══════════════════════════════════════════════════════════════
// Collections
// ═══════════════════════════════════════════════════════════════

/// One repo in a collection.
#[derive(Clone, Debug)]
pub struct CollectionItem {
    /// Repo id (`owner/name`). Full form including namespace.
    pub repo_id: String,
    /// `"dataset"` (vindex repos) or `"model"`.
    pub repo_type: String,
    /// Optional short note rendered on the collection card.
    pub note: Option<String>,
}

/// Ensure a collection titled `title` exists in `namespace`, then add
/// every item to it. Idempotent: re-runs reuse the slug (matched by
/// case-insensitive title) and treat HTTP 409 on add-item as success.
/// Returns the collection URL on success.
pub fn ensure_collection(
    namespace: &str,
    title: &str,
    description: Option<&str>,
    items: &[CollectionItem],
) -> Result<String, VindexError> {
    let token = get_hf_token()?;
    let slug = match find_collection_slug(namespace, title, &token)? {
        Some(existing) => existing,
        None => create_collection(namespace, title, description, &token)?,
    };
    for item in items {
        add_collection_item(&slug, item, &token)?;
    }
    Ok(format!("https://huggingface.co/collections/{slug}"))
}

fn find_collection_slug(
    namespace: &str,
    title: &str,
    token: &str,
) -> Result<Option<String>, VindexError> {
    let client = reqwest::blocking::Client::new();
    let url = format!("https://huggingface.co/api/users/{namespace}/collections?limit=100");
    let resp = client
        .get(&url)
        .header("Authorization", format!("Bearer {token}"))
        .send()
        .map_err(|e| VindexError::Parse(format!("HF collections list failed: {e}")))?;
    if !resp.status().is_success() {
        if resp.status().as_u16() == 404 {
            return Ok(None);
        }
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!(
            "HF collections list ({status}): {body}"
        )));
    }
    let body: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("HF collections JSON: {e}")))?;
    let arr = match body.as_array() {
        Some(a) => a,
        None => return Ok(None),
    };
    let target = title.to_ascii_lowercase();
    for entry in arr {
        let entry_title = entry.get("title").and_then(|v| v.as_str()).unwrap_or("");
        if entry_title.to_ascii_lowercase() == target {
            if let Some(slug) = entry.get("slug").and_then(|v| v.as_str()) {
                return Ok(Some(slug.to_string()));
            }
        }
    }
    Ok(None)
}

fn create_collection(
    namespace: &str,
    title: &str,
    description: Option<&str>,
    token: &str,
) -> Result<String, VindexError> {
    let client = reqwest::blocking::Client::new();
    let mut body = serde_json::json!({
        "title": title,
        "namespace": namespace,
        "private": false,
    });
    if let Some(desc) = description {
        body["description"] = serde_json::Value::String(desc.to_string());
    }
    let resp = client
        .post("https://huggingface.co/api/collections")
        .header("Authorization", format!("Bearer {token}"))
        .json(&body)
        .send()
        .map_err(|e| VindexError::Parse(format!("HF collection create failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!(
            "HF collection create ({status}): {body}"
        )));
    }
    let json: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("HF collection JSON: {e}")))?;
    let slug = json
        .get("slug")
        .and_then(|v| v.as_str())
        .ok_or_else(|| VindexError::Parse("HF collection response missing slug".into()))?;
    Ok(slug.to_string())
}

fn add_collection_item(
    slug: &str,
    item: &CollectionItem,
    token: &str,
) -> Result<(), VindexError> {
    let client = reqwest::blocking::Client::new();
    let url = format!("https://huggingface.co/api/collections/{slug}/item");
    let mut body = serde_json::json!({
        "item": {
            "type": item.repo_type,
            "id": item.repo_id,
        },
    });
    if let Some(note) = &item.note {
        body["note"] = serde_json::Value::String(note.clone());
    }
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {token}"))
        .json(&body)
        .send()
        .map_err(|e| VindexError::Parse(format!("HF collection add-item failed: {e}")))?;
    if resp.status().is_success() || resp.status().as_u16() == 409 {
        Ok(())
    } else {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        Err(VindexError::Parse(format!(
            "HF collection add-item ({status}): {body}"
        )))
    }
}

/// Cheap HEAD probe — returns `Ok(true)` if the dataset repo exists and
/// is readable, `Ok(false)` on 404, `Err` on other failures. Auth is
/// optional; pass-through when available (lets callers see private
/// repos they own).
pub fn dataset_repo_exists(repo_id: &str) -> Result<bool, VindexError> {
    let token = get_hf_token().ok();
    let url = format!("https://huggingface.co/api/datasets/{repo_id}");
    let client = reqwest::blocking::Client::new();
    let mut req = client.head(&url);
    if let Some(t) = token {
        req = req.header("Authorization", format!("Bearer {t}"));
    }
    let resp = req
        .send()
        .map_err(|e| VindexError::Parse(format!("HF HEAD failed: {e}")))?;
    if resp.status().is_success() {
        Ok(true)
    } else if resp.status().as_u16() == 404 {
        Ok(false)
    } else {
        Err(VindexError::Parse(format!(
            "HF HEAD {repo_id}: {}",
            resp.status()
        )))
    }
}

/// Fetch a collection by slug (or full collection URL) and return its
/// items as `(type, id)` pairs — typically `("dataset", "owner/name")`.
pub fn fetch_collection_items(
    slug_or_url: &str,
) -> Result<Vec<(String, String)>, VindexError> {
    let slug = slug_or_url
        .trim_start_matches("https://huggingface.co/collections/")
        .trim_start_matches("http://huggingface.co/collections/")
        .trim_start_matches("hf://collections/")
        .trim_start_matches('/');
    let token = get_hf_token().ok();
    let url = format!("https://huggingface.co/api/collections/{slug}");
    let client = reqwest::blocking::Client::new();
    let mut req = client.get(&url);
    if let Some(t) = token {
        req = req.header("Authorization", format!("Bearer {t}"));
    }
    let resp = req
        .send()
        .map_err(|e| VindexError::Parse(format!("HF collection fetch failed: {e}")))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!(
            "HF collection fetch ({status}): {body}"
        )));
    }
    let body: serde_json::Value = resp
        .json()
        .map_err(|e| VindexError::Parse(format!("HF collection JSON: {e}")))?;
    let items = body
        .get("items")
        .and_then(|v| v.as_array())
        .ok_or_else(|| VindexError::Parse("collection response missing items".into()))?;
    let mut out = Vec::new();
    for item in items {
        let kind = match item.get("type").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let id = match item.get("id").and_then(|v| v.as_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        out.push((kind, id));
    }
    Ok(out)
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
