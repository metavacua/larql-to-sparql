//! HuggingFace LFS protocol primitives — batch endpoint, signed-URL
//! streaming PUT (single + multipart), verify, commit. Internal-only
//! siblings of [`super::upload::upload_file_to_hf`]; see that file for
//! the orchestration shape.
//!
//! Module layout:
//! - `batch`        — `lfs_batch_upload` + JSON parsing (basic + multipart)
//! - `stream`       — `stream_put_with_progress` (single-PUT to signed URL)
//! - `multipart`    — `upload_multipart` (chunked PUT + completion POST)
//! - `finalize`     — `lfs_verify` + `commit_lfs_file`
//! - `mod` (here)   — `CountingReader`, action types, `upload_lfs` orchestrator
//! - `test_support` — shared test fixtures (cfg(test))

mod batch;
mod finalize;
mod multipart;
mod stream;
#[cfg(test)]
mod test_support;

use std::collections::HashMap;
use std::path::Path;

use crate::error::VindexError;

use super::PublishCallbacks;
use batch::lfs_batch_upload;
use finalize::{commit_lfs_file, lfs_verify};
use multipart::upload_multipart;
use stream::stream_put_with_progress;

/// Counting `Read` adapter — increments a shared atomic on every read so
/// a poll thread can report upload progress without per-chunk syscalls.
pub(super) struct CountingReader<R: std::io::Read> {
    pub(super) inner: R,
    pub(super) counter: std::sync::Arc<portable_atomic::AtomicU64>,
}

impl<R: std::io::Read> std::io::Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.counter
            .fetch_add(n as u64, portable_atomic::Ordering::Relaxed);
        Ok(n)
    }
}

#[derive(Debug)]
pub(super) struct LfsAction {
    pub(super) href: String,
    pub(super) header: HashMap<String, String>,
}

/// What kind of upload HF returned in the batch response.
///
/// Files ≤5 GB: `Single` — one PUT to `LfsAction.href` with
/// `LfsAction.header` as request headers.
///
/// Files >5 GB (when our batch request declared `multipart` capability
/// — see [`crate::format::huggingface::publish::protocol::LFS_TRANSFER_MULTIPART`]):
/// `Multipart` — `href` is the completion endpoint, `header` contains
/// `chunk_size` + numbered `00001` / `00002` / … keys, one pre-signed
/// PUT URL per part.
#[derive(Debug)]
pub(super) enum UploadAction {
    /// Single-PUT (HF basic transfer adapter).
    Single(LfsAction),
    /// Multipart upload (HF custom transfer adapter).
    ///
    /// `completion_href` is the URL we POST the final
    /// `{"oid": "...", "parts": [{partNumber, etag}, ...]}` payload to
    /// once every part has been PUT successfully. `chunk_size` is the
    /// per-part byte budget HF chose (typically 100 MB). `parts` is
    /// the ordered list of pre-signed S3 PUT URLs, one per part.
    Multipart {
        completion_href: String,
        chunk_size: u64,
        parts: Vec<String>,
    },
}

#[derive(Debug)]
pub(super) struct LfsBatchResponse {
    pub(super) upload: Option<UploadAction>,
    pub(super) verify: Option<LfsAction>,
}

/// LFS-mode upload: batch → PUT to signed URL → verify → commit pointer.
#[allow(clippy::too_many_arguments)]
pub(super) fn upload_lfs(
    repo_id: &str,
    token: &str,
    local_path: &Path,
    remote_filename: &str,
    size: u64,
    sha256: &str,
    callbacks: &mut dyn PublishCallbacks,
    repo_type: &str,
) -> Result<(), VindexError> {
    let batch = lfs_batch_upload(repo_id, token, sha256, size, repo_type)?;

    match batch.upload {
        Some(UploadAction::Single(ref upload)) => {
            stream_put_with_progress(
                &upload.href,
                &upload.header,
                local_path,
                size,
                remote_filename,
                callbacks,
            )?;
        }
        Some(UploadAction::Multipart {
            ref completion_href,
            chunk_size,
            ref parts,
        }) => {
            upload_multipart(
                local_path,
                size,
                sha256,
                completion_href,
                chunk_size,
                parts,
                remote_filename,
                callbacks,
            )?;
        }
        None => {
            // Object already on HF's side — tick the bar to 100% so
            // the UX matches the upload path.
            callbacks.on_file_progress(remote_filename, size, size);
        }
    }

    if let Some(ref verify) = batch.verify {
        lfs_verify(&verify.href, &verify.header, token, sha256, size)?;
    }

    commit_lfs_file(repo_id, token, remote_filename, sha256, size, repo_type)
}

#[cfg(test)]
mod tests {
    use super::test_support::{write_temp_bytes, CapturingCallbacks, EnvBaseGuard};
    use super::*;
    use serial_test::serial;
    use std::io::Read;

    // ─── CountingReader ────────────────────────────────────────────

    #[test]
    fn counting_reader_counts_bytes_read() {
        use std::sync::atomic::Ordering;
        let bytes = b"hello world".to_vec();
        let counter = std::sync::Arc::new(portable_atomic::AtomicU64::new(0));
        let mut reader = CountingReader {
            inner: bytes.as_slice(),
            counter: counter.clone(),
        };
        let mut buf = [0u8; 5];
        reader.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, b"hello");
        assert_eq!(counter.load(Ordering::Relaxed), 5);

        let mut rest = Vec::new();
        reader.read_to_end(&mut rest).unwrap();
        assert_eq!(rest, b" world");
        assert_eq!(counter.load(Ordering::Relaxed), 11);
    }

    /// Exercise the derived `Debug` impls on `LfsAction`,
    /// `UploadAction`, and `LfsBatchResponse`. llvm-cov counts the
    /// per-field match arms inside a `#[derive(Debug)]` expansion
    /// against the file's line coverage; without an `{:?}`-format
    /// call, every line of every variant arm shows as uncovered.
    /// Hits both `UploadAction::Single` and `UploadAction::Multipart`
    /// arms (the file's two largest contributors to the derived
    /// coverage hole).
    #[test]
    fn upload_action_debug_format_covers_both_variants() {
        let single = UploadAction::Single(LfsAction {
            href: "https://x/single".into(),
            header: HashMap::new(),
        });
        let multipart = UploadAction::Multipart {
            completion_href: "https://x/complete".into(),
            chunk_size: 100,
            parts: vec!["https://x/p1".into()],
        };
        let response = LfsBatchResponse {
            upload: Some(single),
            verify: Some(LfsAction {
                href: "https://x/verify".into(),
                header: HashMap::new(),
            }),
        };
        // Pin only that the format compiles and produces something
        // non-empty for each variant — exact format text is a derive
        // contract we don't want to lock in.
        assert!(format!("{response:?}").contains("Single"));
        assert!(format!("{multipart:?}").contains("Multipart"));
    }

    #[test]
    fn counting_reader_counter_starts_at_zero() {
        use std::sync::atomic::Ordering;
        let counter = std::sync::Arc::new(portable_atomic::AtomicU64::new(0));
        let mut reader = CountingReader {
            inner: std::io::empty(),
            counter: counter.clone(),
        };
        let mut buf = [0u8; 16];
        let n = reader.read(&mut buf).unwrap();
        assert_eq!(n, 0);
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    // ─── upload_lfs orchestrator (batch → PUT → verify → commit) ───

    #[test]
    #[serial]
    fn upload_lfs_full_path_with_upload_action() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let (_dir, path) = write_temp_bytes(b"payload");

        let put_url = format!("{}/lfs/up", server.url());
        let verify_url = format!("{}/lfs/v", server.url());

        let batch_mock = server
            .mock("POST", "/org/repo.git/info/lfs/objects/batch")
            .with_status(200)
            .with_body(
                serde_json::json!({
                    "objects": [{
                        "actions": {
                            "upload": {"href": put_url, "header": {}},
                            "verify": {"href": verify_url, "header": {}}
                        }
                    }]
                })
                .to_string(),
            )
            .create();
        let put_mock = server.mock("PUT", "/lfs/up").with_status(200).create();
        let verify_mock = server.mock("POST", "/lfs/v").with_status(200).create();
        let commit_mock = server
            .mock("POST", "/api/models/org/repo/commit/main")
            .with_status(200)
            .create();

        let mut cb = CapturingCallbacks::default();
        upload_lfs("org/repo", "t", &path, "p.bin", 7, "sha", &mut cb, "model").unwrap();
        batch_mock.assert();
        put_mock.assert();
        verify_mock.assert();
        commit_mock.assert();
    }

    #[test]
    #[serial]
    fn upload_lfs_skips_put_when_object_already_present() {
        // Batch returns no `actions.upload` ⇒ HF says the LFS object is
        // already stored. upload_lfs must skip the PUT and proceed
        // straight to verify (if present) + commit.
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let (_dir, path) = write_temp_bytes(b"payload");

        let verify_url = format!("{}/lfs/v", server.url());

        let batch_mock = server
            .mock("POST", "/org/repo.git/info/lfs/objects/batch")
            .with_status(200)
            .with_body(
                serde_json::json!({
                    "objects": [{
                        "actions": {
                            "verify": {"href": verify_url, "header": {}}
                        }
                    }]
                })
                .to_string(),
            )
            .create();
        let verify_mock = server.mock("POST", "/lfs/v").with_status(200).create();
        let commit_mock = server
            .mock("POST", "/api/models/org/repo/commit/main")
            .with_status(200)
            .create();

        let mut cb = CapturingCallbacks::default();
        upload_lfs("org/repo", "t", &path, "p.bin", 7, "sha", &mut cb, "model").unwrap();
        batch_mock.assert();
        verify_mock.assert();
        commit_mock.assert();
        assert!(cb
            .progress_calls
            .iter()
            .any(|(_, sent, total)| sent == total && *sent == 7));
    }

    /// End-to-end orchestrator on the multipart branch: batch
    /// returns a multipart-shape `actions.upload`, `upload_lfs`
    /// dispatches into `upload_multipart` (chunked PUT + completion
    /// POST), then runs verify + commit. Pins the integration so a
    /// future refactor that drops the `UploadAction::Multipart` arm
    /// (e.g. accidental match-arm deletion) trips immediately
    /// instead of silently falling through to single-PUT and
    /// failing on >5 GB uploads at runtime.
    #[test]
    #[serial]
    fn upload_lfs_multipart_path_runs_parts_completion_verify_commit() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        // 10 bytes, chunk_size=4 → 3 parts (4, 4, 2).
        let (_dir, path) = write_temp_bytes(b"AAAABBBBCC");

        let complete_url = format!("{}/complete", server.url());
        let part1_url = format!("{}/p1", server.url());
        let part2_url = format!("{}/p2", server.url());
        let part3_url = format!("{}/p3", server.url());
        let verify_url = format!("{}/lfs/v", server.url());

        let batch_mock = server
            .mock("POST", "/org/repo.git/info/lfs/objects/batch")
            .with_status(200)
            .with_body(
                serde_json::json!({
                    "objects": [{
                        "actions": {
                            "upload": {
                                "href": complete_url,
                                "header": {
                                    "chunk_size": "4",
                                    "00001": part1_url,
                                    "00002": part2_url,
                                    "00003": part3_url,
                                },
                            },
                            "verify": {"href": verify_url, "header": {}}
                        }
                    }]
                })
                .to_string(),
            )
            .create();
        let p1_mock = server
            .mock("PUT", "/p1")
            .with_status(200)
            .with_header("ETag", "\"e1\"")
            .create();
        let p2_mock = server
            .mock("PUT", "/p2")
            .with_status(200)
            .with_header("ETag", "\"e2\"")
            .create();
        let p3_mock = server
            .mock("PUT", "/p3")
            .with_status(200)
            .with_header("ETag", "\"e3\"")
            .create();
        let completion_mock = server.mock("POST", "/complete").with_status(200).create();
        let verify_mock = server.mock("POST", "/lfs/v").with_status(200).create();
        let commit_mock = server
            .mock("POST", "/api/models/org/repo/commit/main")
            .with_status(200)
            .create();

        let mut cb = CapturingCallbacks::default();
        upload_lfs(
            "org/repo", "t", &path, "huge.bin", 10, "sha", &mut cb, "model",
        )
        .unwrap();
        batch_mock.assert();
        p1_mock.assert();
        p2_mock.assert();
        p3_mock.assert();
        completion_mock.assert();
        verify_mock.assert();
        commit_mock.assert();
        // Final progress tick = 100%.
        assert_eq!(cb.progress_calls.last().map(|t| (t.1, t.2)), Some((10, 10)));
    }

    #[test]
    #[serial]
    fn upload_lfs_no_verify_action_skips_verify_and_commits() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let (_dir, path) = write_temp_bytes(b"payload");

        let put_url = format!("{}/lfs/up", server.url());

        let batch_mock = server
            .mock("POST", "/org/repo.git/info/lfs/objects/batch")
            .with_status(200)
            .with_body(
                serde_json::json!({
                    "objects": [{
                        "actions": {
                            "upload": {"href": put_url, "header": {}}
                        }
                    }]
                })
                .to_string(),
            )
            .create();
        let put_mock = server.mock("PUT", "/lfs/up").with_status(200).create();
        let commit_mock = server
            .mock("POST", "/api/models/org/repo/commit/main")
            .with_status(200)
            .create();

        let mut cb = CapturingCallbacks::default();
        upload_lfs("org/repo", "t", &path, "p.bin", 7, "sha", &mut cb, "model").unwrap();
        batch_mock.assert();
        put_mock.assert();
        commit_mock.assert();
    }
}
