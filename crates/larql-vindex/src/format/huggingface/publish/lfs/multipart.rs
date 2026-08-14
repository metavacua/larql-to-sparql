//! HF multipart LFS upload — chunked PUT per part + completion POST.
//!
//! Used for files >5 GB where HF's basic transfer adapter would
//! reject the single PUT. The batch response carries a
//! `chunk_size` + numbered `00001` / `00002` / … pre-signed S3 PUT
//! URLs; we PUT each part in order, capture the returned `ETag` from
//! S3, then POST `{"oid": "<sha256>", "parts": [{partNumber, etag},
//! …]}` to the completion endpoint (which the batch response handed
//! us as `upload.href`).
//!
//! Mirrors the multipart branch of
//! `huggingface_hub._commit_api._upload_multi_part`. The protocol is
//! HF's extension to the git-lfs batch API: the batch caller declares
//! `"transfers": ["basic", "multipart"]` (see `batch.rs`), and HF
//! picks `multipart` when the object exceeds the basic-transfer
//! single-PUT ceiling. See
//! [`crate::format::huggingface::publish::protocol::LFS_TRANSFER_MULTIPART`]
//! for the constant.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::VindexError;

use super::super::protocol::{CONTENT_TYPE_LFS_JSON, LFS_PUT_TIMEOUT};
use super::super::PublishCallbacks;
use super::retry::{backoff_for, is_transient_status, retry_after_secs, TRANSIENT_MAX_ATTEMPTS};

/// Streamed multipart LFS upload. Returns successfully once HF's
/// completion endpoint has accepted the assembled parts.
///
/// `local_path` + `size` describe the local file. `sha256` is the
/// object's pre-computed SHA-256 (already used as the OID in the
/// batch request). `completion_href` is `actions.upload.href` from
/// the batch response — POST target for the final assembly call.
/// `chunk_size` is the per-part byte budget HF chose. `parts` is the
/// ordered list of pre-signed S3 PUT URLs (already numerically sorted
/// in `batch.rs`).
///
/// Progress callbacks fire after every part lands; the bar
/// granularity is therefore `chunk_size` (≈100 MB on typical HF
/// responses) rather than the per-byte granularity of the single-PUT
/// path. Reasonable trade for not having to thread a counter through
/// the streaming body across N concurrent uploads.
#[allow(clippy::too_many_arguments)]
pub(super) fn upload_multipart(
    local_path: &Path,
    size: u64,
    sha256: &str,
    completion_href: &str,
    chunk_size: u64,
    parts: &[String],
    remote_filename: &str,
    callbacks: &mut dyn PublishCallbacks,
) -> Result<(), VindexError> {
    if chunk_size == 0 {
        return Err(VindexError::Parse(
            "multipart upload chunk_size must be > 0".into(),
        ));
    }
    if parts.is_empty() {
        return Err(VindexError::Parse(
            "multipart upload parts list is empty".into(),
        ));
    }
    // Sanity-check: expected number of parts = ceil(size / chunk_size).
    // If the server lies (returns too few/many URLs for the size) we
    // can't reconstruct the file. Mirrors
    // `_get_sorted_parts_urls`'s "Invalid server response" check.
    let expected_parts = size.div_ceil(chunk_size) as usize;
    if expected_parts != parts.len() {
        return Err(VindexError::Parse(format!(
            "multipart upload {remote_filename}: server returned {} part URLs but file size \
             {size} / chunk_size {chunk_size} requires {expected_parts} parts",
            parts.len()
        )));
    }

    let client = reqwest::blocking::Client::builder()
        .timeout(LFS_PUT_TIMEOUT)
        .build()
        .map_err(|e| VindexError::Parse(format!("HTTP client error: {e}")))?;

    let mut etags: Vec<String> = Vec::with_capacity(parts.len());
    let mut file = std::fs::File::open(local_path)?;
    let mut bytes_uploaded: u64 = 0;

    for (idx, part_url) in parts.iter().enumerate() {
        let part_number = (idx + 1) as u32;
        let offset = idx as u64 * chunk_size;
        let part_len = chunk_size.min(size - offset);

        // Read exactly `part_len` bytes from the current offset.
        file.seek(SeekFrom::Start(offset)).map_err(|e| {
            VindexError::Parse(format!(
                "multipart upload {remote_filename}: seek to offset {offset} failed: {e}"
            ))
        })?;
        let mut buf = vec![0u8; part_len as usize];
        file.read_exact(&mut buf).map_err(|e| {
            VindexError::Parse(format!(
                "multipart upload {remote_filename}: read {part_len} bytes at offset {offset} failed: {e}"
            ))
        })?;

        // Retry transient failures rather than aborting the publish. HF's S3
        // backend rate-limits a long upload with `503 SlowDown`, and losing
        // the run at part N leaves the repo holding stale files with no
        // replacement — a partially-published repo still serves, so it is
        // worse than a clean failure. The body is re-sent from `buf`, which
        // is already in memory, so a retry costs no extra read.
        let mut attempt: u32 = 1;
        let resp = loop {
            let outcome = client.put(part_url).body(buf.clone()).send();

            let (retryable, detail, resp) = match outcome {
                Ok(r) if r.status().is_success() => break r,
                Ok(r) => {
                    let status = r.status();
                    let wait = retry_after_secs(r.headers());
                    let transient = is_transient_status(status.as_u16());
                    (transient, format!("{status}"), Some((r, wait)))
                }
                // A transport error (reset, timeout, DNS blip) is always
                // worth one more go — nothing was necessarily rejected.
                Err(e) => (true, format!("transport error: {e}"), None),
            };

            if !retryable || attempt >= TRANSIENT_MAX_ATTEMPTS {
                let body = match resp {
                    Some((r, _)) => r.text().unwrap_or_default(),
                    None => String::new(),
                };
                let give_up = if retryable {
                    format!(" after {TRANSIENT_MAX_ATTEMPTS} attempts")
                } else {
                    String::new()
                };
                return Err(VindexError::Parse(format!(
                    "multipart upload {remote_filename} part {part_number}/{} ({detail}){give_up}: {body}",
                    parts.len()
                )));
            }

            let wait = backoff_for(attempt, resp.and_then(|(_, w)| w));
            callbacks.on_retry(
                &format!("{remote_filename} part {part_number}/{}", parts.len()),
                attempt + 1,
                TRANSIENT_MAX_ATTEMPTS,
                &detail,
                wait,
            );
            std::thread::sleep(wait);
            attempt += 1;
        };

        // S3 returns the part's ETag in the response header. The
        // completion POST must reference these ETags in order.
        // S3 quotes the ETag value — preserve the quotes; the
        // multipart-complete API accepts them as-is.
        let etag = resp
            .headers()
            .get(reqwest::header::ETAG)
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| {
                VindexError::Parse(format!(
                    "multipart upload {remote_filename} part {part_number}: \
                     response missing ETag header"
                ))
            })?
            .to_string();
        etags.push(etag);

        bytes_uploaded += part_len;
        callbacks.on_file_progress(remote_filename, bytes_uploaded, size);
    }

    finalize_multipart(&client, completion_href, sha256, &etags, remote_filename)?;

    Ok(())
}

/// POST the completion payload — `{"oid": "<sha256>", "parts":
/// [{"partNumber": N, "etag": "..."}]}` — to HF's completion endpoint.
/// HF then assembles the object on its side and the verify+commit
/// stages can proceed.
fn finalize_multipart(
    client: &reqwest::blocking::Client,
    completion_href: &str,
    sha256: &str,
    etags: &[String],
    remote_filename: &str,
) -> Result<(), VindexError> {
    let parts: Vec<serde_json::Value> = etags
        .iter()
        .enumerate()
        .map(|(idx, etag)| {
            serde_json::json!({
                "partNumber": idx + 1,
                "etag": etag,
            })
        })
        .collect();
    let body = serde_json::json!({
        "oid": sha256,
        "parts": parts,
    });
    let resp = client
        .post(completion_href)
        .header("Content-Type", CONTENT_TYPE_LFS_JSON)
        .header("Accept", CONTENT_TYPE_LFS_JSON)
        .json(&body)
        .send()
        .map_err(|e| {
            VindexError::Parse(format!(
                "multipart upload {remote_filename}: completion POST failed: {e}"
            ))
        })?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(VindexError::Parse(format!(
            "multipart upload {remote_filename} completion ({status}): {body}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{write_temp_bytes, CapturingCallbacks, EnvBaseGuard};
    use super::*;
    use serial_test::serial;

    /// Round-trip: 3 parts of 4 bytes each (one partial at the end).
    /// Each part PUT returns a distinct ETag; finalisation POST
    /// receives them in `partNumber` order.
    #[test]
    #[serial]
    fn upload_multipart_three_parts_finalises_with_etags_in_order() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let payload = b"AAAABBBBCC"; // 10 bytes
        let (_dir, path) = write_temp_bytes(payload);

        let part1 = server
            .mock("PUT", "/p1")
            .match_body(b"AAAA".to_vec())
            .with_status(200)
            .with_header("ETag", "\"etag-1\"")
            .create();
        let part2 = server
            .mock("PUT", "/p2")
            .match_body(b"BBBB".to_vec())
            .with_status(200)
            .with_header("ETag", "\"etag-2\"")
            .create();
        let part3 = server
            .mock("PUT", "/p3")
            .match_body(b"CC".to_vec())
            .with_status(200)
            .with_header("ETag", "\"etag-3\"")
            .create();
        let completion = server
            .mock("POST", "/complete")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!({
                "oid": "sha256-x",
                "parts": [
                    {"partNumber": 1, "etag": "\"etag-1\""},
                    {"partNumber": 2, "etag": "\"etag-2\""},
                    {"partNumber": 3, "etag": "\"etag-3\""}
                ],
            })))
            .with_status(200)
            .create();

        let completion_href = format!("{}/complete", server.url());
        let parts = vec![
            format!("{}/p1", server.url()),
            format!("{}/p2", server.url()),
            format!("{}/p3", server.url()),
        ];
        let mut cb = CapturingCallbacks::default();
        upload_multipart(
            &path,
            10,
            "sha256-x",
            &completion_href,
            4,
            &parts,
            "blob.bin",
            &mut cb,
        )
        .unwrap();

        part1.assert();
        part2.assert();
        part3.assert();
        completion.assert();

        // Progress callbacks: one per part. Final tick should hit 100%.
        let last = cb.progress_calls.last().expect("at least one tick");
        assert_eq!(last, &("blob.bin".to_string(), 10, 10));
        assert_eq!(cb.progress_calls.len(), 3);
    }

    /// Exact divisor: 8 bytes / 4-byte chunks = exactly 2 parts (no
    /// short final part).
    #[test]
    #[serial]
    fn upload_multipart_exact_chunk_alignment() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let payload = b"01234567";
        let (_dir, path) = write_temp_bytes(payload);

        let p1 = server
            .mock("PUT", "/p1")
            .match_body(b"0123".to_vec())
            .with_status(200)
            .with_header("ETag", "\"e1\"")
            .create();
        let p2 = server
            .mock("PUT", "/p2")
            .match_body(b"4567".to_vec())
            .with_status(200)
            .with_header("ETag", "\"e2\"")
            .create();
        let completion = server.mock("POST", "/complete").with_status(200).create();

        let completion_href = format!("{}/complete", server.url());
        let parts = vec![
            format!("{}/p1", server.url()),
            format!("{}/p2", server.url()),
        ];
        let mut cb = CapturingCallbacks::default();
        upload_multipart(&path, 8, "x", &completion_href, 4, &parts, "f", &mut cb).unwrap();
        p1.assert();
        p2.assert();
        completion.assert();
    }

    /// Part URL count mismatch with size/chunk_size combo must error
    /// before any PUT fires. Three URLs but only ceil(8/4)=2 needed.
    #[test]
    #[serial]
    fn upload_multipart_part_count_mismatch_errors_before_put() {
        let server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let (_dir, path) = write_temp_bytes(b"01234567");

        let parts = vec![
            "https://x/p1".to_string(),
            "https://x/p2".to_string(),
            "https://x/p3".to_string(), // one too many
        ];
        let mut cb = CapturingCallbacks::default();
        let err = upload_multipart(&path, 8, "x", "https://x/c", 4, &parts, "f", &mut cb)
            .expect_err("count mismatch must error");
        assert!(err.to_string().contains("server returned 3 part URLs"));
        let _ = server;
    }

    /// `chunk_size == 0` is rejected early.
    #[test]
    fn upload_multipart_zero_chunk_size_errors() {
        let (_dir, path) = write_temp_bytes(b"x");
        let mut cb = CapturingCallbacks::default();
        let err = upload_multipart(
            &path,
            1,
            "x",
            "https://x/c",
            0,
            &["https://x/p".into()],
            "f",
            &mut cb,
        )
        .expect_err("chunk_size=0 must error");
        assert!(err.to_string().contains("chunk_size must be > 0"));
    }

    /// Empty parts list is rejected early.
    #[test]
    fn upload_multipart_empty_parts_errors() {
        let (_dir, path) = write_temp_bytes(b"x");
        let mut cb = CapturingCallbacks::default();
        let err = upload_multipart(&path, 1, "x", "https://x/c", 1, &[], "f", &mut cb)
            .expect_err("empty parts must error");
        assert!(err.to_string().contains("parts list is empty"));
    }

    /// A part PUT that returns 500 surfaces the error with the
    /// remote filename and part number for diagnostics.
    #[test]
    #[serial]
    fn upload_multipart_part_put_500_surfaces_error_with_filename() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let (_dir, path) = write_temp_bytes(b"0123");

        let _p1 = server
            .mock("PUT", "/p1")
            .with_status(500)
            .with_body("boom")
            .create();

        let completion_href = format!("{}/complete", server.url());
        let parts = vec![format!("{}/p1", server.url())];
        let mut cb = CapturingCallbacks::default();
        let err = upload_multipart(
            &path,
            4,
            "x",
            &completion_href,
            4,
            &parts,
            "blob.bin",
            &mut cb,
        )
        .expect_err("500 must error");
        let msg = err.to_string();
        assert!(msg.contains("part 1/1"), "{msg}");
        assert!(msg.contains("blob.bin"), "{msg}");
        assert!(msg.contains("500"), "{msg}");
    }

    /// A part PUT that responds 200 but without an `ETag` header
    /// errors — we'd otherwise POST a bogus completion payload.
    #[test]
    #[serial]
    fn upload_multipart_missing_etag_header_errors() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let (_dir, path) = write_temp_bytes(b"x");

        let _p1 = server.mock("PUT", "/p1").with_status(200).create();

        let completion_href = format!("{}/complete", server.url());
        let parts = vec![format!("{}/p1", server.url())];
        let mut cb = CapturingCallbacks::default();
        let err = upload_multipart(&path, 1, "x", &completion_href, 1, &parts, "f", &mut cb)
            .expect_err("missing ETag must error");
        assert!(err.to_string().contains("missing ETag header"));
    }

    /// Completion POST failure surfaces with the filename. Catches
    /// the case where parts upload fine but HF rejects the assembly
    /// (e.g. SHA mismatch on the server's side).
    #[test]
    #[serial]
    fn upload_multipart_completion_500_surfaces_error() {
        let mut server = mockito::Server::new();
        let _guard = EnvBaseGuard::new(&server.url());
        let (_dir, path) = write_temp_bytes(b"x");

        let _p1 = server
            .mock("PUT", "/p1")
            .with_status(200)
            .with_header("ETag", "\"e1\"")
            .create();
        let _completion = server
            .mock("POST", "/complete")
            .with_status(500)
            .with_body("bad assembly")
            .create();

        let completion_href = format!("{}/complete", server.url());
        let parts = vec![format!("{}/p1", server.url())];
        let mut cb = CapturingCallbacks::default();
        let err = upload_multipart(
            &path,
            1,
            "x",
            &completion_href,
            1,
            &parts,
            "blob.bin",
            &mut cb,
        )
        .expect_err("completion 500 must error");
        let msg = err.to_string();
        assert!(msg.contains("blob.bin"), "{msg}");
        assert!(msg.contains("completion (500"), "{msg}");
    }
}
