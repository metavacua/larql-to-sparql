//! Constants for the HuggingFace HTTP / LFS wire protocol.
//!
//! Every wire-format literal that appears in more than one site, or is
//! a non-obvious magic number (timeout duration, HTTP status code),
//! lives here. JSON keys that appear once inside a single
//! `serde_json::json!({...})` body stay inline — they're
//! self-documenting next to the structural value.

use std::time::Duration;

// ── Base URL ──────────────────────────────────────────────────────────

const HF_BASE_URL: &str = "https://huggingface.co";

/// Test-only env var that overrides the HuggingFace base URL. Tests
/// across `huggingface/{publish,discovery,download}` spin up a
/// `mockito` server and point `hf_base()` at it. Production callers
/// never set this.
pub(in crate::format::huggingface) const TEST_BASE_ENV: &str = "LARQL_HF_TEST_BASE";

/// Base URL for HuggingFace requests. Defaults to
/// `https://huggingface.co`; can be overridden for tests via
/// [`TEST_BASE_ENV`]. Visible to every sibling under
/// `format::huggingface::*` so the `discovery` and `download`
/// modules can route through the same override.
pub(in crate::format::huggingface) fn hf_base() -> String {
    std::env::var(TEST_BASE_ENV).unwrap_or_else(|_| HF_BASE_URL.to_string())
}

// ── Repo type wire values ─────────────────────────────────────────────

/// Value HF expects for `type` when creating a model repo.
pub(super) const REPO_TYPE_MODEL: &str = "model";
/// Value HF expects for `type` when creating a dataset repo.
pub(super) const REPO_TYPE_DATASET: &str = "dataset";

/// Map a repo type (`"model"` / `"dataset"`) to the URL path segment
/// (`"models"` / `"datasets"`). Defaults to `"models"` for unknown
/// values so a typo silently routes to a 404 we can handle, not a
/// fictional `"spaces"` URL.
pub(super) fn repo_type_plural(repo_type: &str) -> &'static str {
    if repo_type == REPO_TYPE_DATASET {
        "datasets"
    } else {
        "models"
    }
}

// ── HTTP behaviour ────────────────────────────────────────────────────

/// One hour. The 4-7 GB Q4K weight files take this long over a typical
/// home connection; anything shorter has us bouncing on slow uploads.
pub(super) const LFS_PUT_TIMEOUT: Duration = Duration::from_secs(3600);

/// How often the main thread polls the byte counter while the worker
/// thread runs the streaming PUT. 100ms is fine-grained enough for a
/// progress bar but cheap enough not to spin.
pub(super) const UPLOAD_PROGRESS_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// HTTP 409 Conflict — what HF returns from `repos/create` when the
/// repo already exists. Not an error for our purposes.
pub(super) const HTTP_STATUS_CONFLICT: u16 = 409;

/// Bytes of the file we hand HF in the preupload `sample` so it can
/// sniff text vs binary. HF accepts a smaller sample for tiny files
/// without complaint.
pub(super) const HF_PREUPLOAD_SAMPLE_BYTES: usize = 512;

// ── Content-Type values ───────────────────────────────────────────────

pub(super) const CONTENT_TYPE_LFS_JSON: &str = "application/vnd.git-lfs+json";
pub(super) const CONTENT_TYPE_NDJSON: &str = "application/x-ndjson";

// ── LFS protocol vocabulary (used in multiple sites) ──────────────────

pub(super) const LFS_OP_UPLOAD: &str = "upload";
pub(super) const LFS_OP_VERIFY: &str = "verify";
pub(super) const LFS_TRANSFER_BASIC: &str = "basic";
pub(super) const HASH_ALGO_SHA256: &str = "sha256";
