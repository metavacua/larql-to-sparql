//! Shared test scaffolding for `tests.rs` and `tests_v3.rs`.
//!
//! Both files exercise the same `hf_hub`-bound resolve/download functions
//! against a mocked HF endpoint and need the same env-var isolation and
//! mock-building helpers. Kept as one file so the two test modules can't
//! drift into subtly different mock shapes.

use super::*;

/// Stub `DownloadProgress` for the *_with_progress tests. We only need
/// the trait to exist so the function type-checks; the stub is never
/// invoked because most callers hit an early-return path, and the
/// V3-completeness tests don't assert on progress ticks.
pub(super) struct NoOpProgress;
impl DownloadProgress for NoOpProgress {
    fn init(&mut self, _size: usize, _filename: &str) {}
    fn update(&mut self, _size: usize) {}
    fn finish(&mut self) {}
}

/// RAII guard for HF_ENDPOINT + HF_HOME + a tempdir cache.
/// Restores prior values on drop.
pub(super) struct HfTestEnv {
    prev_endpoint: Option<String>,
    prev_home: Option<String>,
    prev_hub: Option<String>,
    prev_token: Option<String>,
    // Hold the tempdir so it lives as long as the guard.
    _tmp: tempfile::TempDir,
}
impl HfTestEnv {
    pub(super) fn new(endpoint: &str) -> Self {
        let prev_endpoint = std::env::var("HF_ENDPOINT").ok();
        let prev_home = std::env::var("HF_HOME").ok();
        let prev_hub = std::env::var("HUGGINGFACE_HUB_CACHE").ok();
        let prev_token = std::env::var("HF_TOKEN").ok();

        let tmp = tempfile::tempdir().unwrap();
        std::env::set_var("HF_ENDPOINT", endpoint);
        std::env::set_var("HF_HOME", tmp.path());
        // Clear HUGGINGFACE_HUB_CACHE so HF_HOME wins; clear token
        // so we don't accidentally hit a real auth header.
        std::env::remove_var("HUGGINGFACE_HUB_CACHE");
        std::env::remove_var("HF_TOKEN");

        Self {
            prev_endpoint,
            prev_home,
            prev_hub,
            prev_token,
            _tmp: tmp,
        }
    }
}
impl Drop for HfTestEnv {
    fn drop(&mut self) {
        for (k, prev) in [
            ("HF_ENDPOINT", self.prev_endpoint.take()),
            ("HF_HOME", self.prev_home.take()),
            ("HUGGINGFACE_HUB_CACHE", self.prev_hub.take()),
            ("HF_TOKEN", self.prev_token.take()),
        ] {
            match prev {
                Some(v) => std::env::set_var(k, v),
                None => std::env::remove_var(k),
            }
        }
    }
}

/// Build a mock that satisfies hf-hub's metadata() probe. The
/// requirements are: an ETag (or X-Linked-Etag) header, an
/// X-Repo-Commit header (commit hash), and a Content-Range header
/// of the form "bytes 0-0/<size>". hf-hub's metadata path issues a
/// GET with `Range: bytes=0-0`, then a follow-up GET for the full
/// body; both must succeed for `repo.get()` to return a path.
///
/// `expect` lets the test cap how many times the mock can fire —
/// hf-hub's download path may retry or make follow-up requests we
/// don't strictly model.
pub(super) fn mock_hf_file_resolve(
    server: &mut mockito::ServerGuard,
    path_regex: &str,
    etag: &str,
    body: &[u8],
) -> Vec<mockito::Mock> {
    let len = body.len();
    let cr = format!("bytes 0-0/{len}");
    let meta = server
        .mock("GET", mockito::Matcher::Regex(path_regex.into()))
        .with_status(200)
        .with_header("ETag", &format!("\"{etag}\""))
        .with_header("X-Repo-Commit", "deadbeefcafebabe")
        .with_header("Content-Range", &cr)
        .with_header("Accept-Ranges", "bytes")
        .with_header("Content-Length", &len.to_string())
        .with_body(body)
        .expect_at_least(1)
        .create();
    vec![meta]
}
