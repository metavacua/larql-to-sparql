//! Mode B shard downloader — streams a tar from the donor's `/v1/shard`
//! endpoint, optionally verifies the SHA-256 of the byte stream, and
//! unpacks it into `store_path/{model_id}/layers-{start}-{end}/`.
//!
//! The unpack is atomic: the tar is unpacked into a sibling `.tmp` directory
//! that is renamed onto the final path on success. A partial download leaves
//! a `.tmp` directory behind which the next attempt removes.

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tracing::{info, warn};

const SHARD_ENDPOINT: &str = "/v1/shard";

/// Whole-request timeout for downloading one shard tar (connect + full body):
/// 10 minutes, sized for multi-GB layer tars over LAN links.
const SHARD_DOWNLOAD_TIMEOUT_SECS: u64 = 600;

/// Reject a `model_id` that cannot safely become one path segment.
///
/// `model_id` arrives in the router's `AssignMsg` — it is remote input,
/// not a local configuration value — and every use below joins it into a
/// path that is then `create_dir_all`'d and unpacked into. Without this,
/// a router (or anyone who can reach this node's announce socket) can
/// set `model_id` to `../../../../etc/cron.d` and choose where a tar
/// lands on this filesystem.
///
/// A single path component of the conservative character set is all any
/// real model id needs; anything else is refused rather than sanitised,
/// because silently rewriting an id would make the shard land somewhere
/// the router did not ask for and the mismatch would surface later as a
/// confusing cache miss.
fn validated_model_id(model_id: &str) -> Result<&str, String> {
    if model_id.is_empty() {
        return Err("model_id is empty".into());
    }
    if model_id.len() > MAX_MODEL_ID_LEN {
        return Err(format!(
            "model_id is {} bytes, over the {MAX_MODEL_ID_LEN}-byte limit",
            model_id.len()
        ));
    }
    // `..` and separators are the traversal primitives; `.` alone would
    // resolve to the store root. Windows accepts `\\` as a separator, so
    // reject it too even though this path is unix-first.
    if model_id == "." || model_id == ".." {
        return Err(format!("model_id `{model_id}` is a directory reference"));
    }
    if model_id.contains("..") {
        return Err(format!("model_id `{model_id}` contains `..`"));
    }
    if !model_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    {
        return Err(format!(
            "model_id `{model_id}` has characters outside [A-Za-z0-9._-]"
        ));
    }
    Ok(model_id)
}

/// Upper bound on a `model_id`, so a hostile peer cannot push a path near
/// the filesystem's own limit.
const MAX_MODEL_ID_LEN: usize = 128;

/// Download a shard tar from `origin_url`, verify the hash, atomically unpack
/// to `store_path/{model_id}/layers-{layer_start}-{layer_end}/`.
pub async fn download_and_load_shard(
    origin_url: &str,
    store_path: &str,
    expected_hash: &str,
    model_id: &str,
    layer_start: u32,
    layer_end: u32,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let url = format!(
        "{}{SHARD_ENDPOINT}/{model_id}/{layer_start}-{layer_end}",
        origin_url.trim_end_matches('/')
    );

    let model_id = validated_model_id(model_id).map_err(|e| {
        warn!(%e, "Mode B: refusing shard download — unsafe model_id");
        e
    })?;
    let model_dir = PathBuf::from(store_path).join(model_id);
    let shard_dir = model_dir.join(format!("layers-{layer_start}-{layer_end}"));
    let tmp_dir = model_dir.join(format!(".tmp-layers-{layer_start}-{layer_end}"));

    tokio::fs::create_dir_all(&model_dir).await?;

    // Remove a stale tmp directory from an earlier aborted attempt.
    if tokio::fs::metadata(&tmp_dir).await.is_ok() {
        let _ = tokio::fs::remove_dir_all(&tmp_dir).await;
    }
    // If the final shard already exists, treat as success (idempotent).
    if tokio::fs::metadata(&shard_dir).await.is_ok() {
        info!(dest = %shard_dir.display(), "Mode B: shard already present — skipping download");
        return Ok(());
    }

    info!(url = %url, dest = %shard_dir.display(), "Mode B: downloading shard tar…");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(SHARD_DOWNLOAD_TIMEOUT_SECS))
        .build()?;
    let resp = client.get(&url).send().await?;
    if !resp.status().is_success() {
        return Err(format!("shard download failed: HTTP {} from {url}", resp.status()).into());
    }

    let bytes = resp.bytes().await?;
    info!(
        bytes = bytes.len(),
        "Mode B: download complete — unpacking…"
    );

    let skip_hash = expected_hash.is_empty()
        || expected_hash == "0000000000000000"
        || expected_hash.chars().all(|c| c == '0');

    if !skip_hash {
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let got_hash = format!("{:x}", hasher.finalize());
        if got_hash != expected_hash {
            return Err(
                format!("shard hash mismatch: expected {expected_hash}, got {got_hash}").into(),
            );
        }
        info!("Mode B: hash verified ✓");
    } else {
        warn!("Mode B: hash check skipped (placeholder hash)");
    }

    // Unpack in a blocking task — `tar::Archive` is sync I/O.
    let tmp_dir_for_blocking = tmp_dir.clone();
    let bytes_for_blocking = bytes.clone();
    tokio::task::spawn_blocking(move || -> std::io::Result<()> {
        std::fs::create_dir_all(&tmp_dir_for_blocking)?;
        let cursor = std::io::Cursor::new(bytes_for_blocking);
        let mut archive = tar::Archive::new(cursor);
        archive.unpack(&tmp_dir_for_blocking)?;
        Ok(())
    })
    .await
    .map_err(|e| format!("unpack task join failed: {e}"))??;

    // Atomic rename onto the final path.
    if let Err(e) = tokio::fs::rename(&tmp_dir, &shard_dir).await {
        // Best-effort cleanup of the half-unpacked tmp dir.
        let _ = tokio::fs::remove_dir_all(&tmp_dir).await;
        return Err(format!(
            "atomic rename {} -> {} failed: {e}",
            tmp_dir.display(),
            shard_dir.display()
        )
        .into());
    }

    info!(dest = %shard_dir.display(), "Mode B: shard unpacked — ready");
    Ok(())
}

/// Where a shard lands. `None` for a `model_id` that is not a safe single
/// path segment — callers must refuse rather than fall back, since every
/// fallback here is a write outside the store.
#[allow(dead_code)] // exposed for tests + future external callers
pub fn shard_dest_path(store_path: &str, model_id: &str, start: u32, end: u32) -> Option<PathBuf> {
    let model_id = validated_model_id(model_id).ok()?;
    Some(
        Path::new(store_path)
            .join(model_id)
            .join(format!("layers-{start}-{end}")),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `model_id` is remote input from the router's AssignMsg. These are
    /// the shapes that let a peer choose where bytes land on this disk.
    #[test]
    fn traversing_model_ids_are_refused() {
        for hostile in [
            "../../../../etc/cron.d",
            "..",
            ".",
            "a/../../b",
            "foo/bar",
            "/absolute",
            "back\\slash",
            "trailing/",
            "nul\0byte",
            "",
        ] {
            assert!(
                validated_model_id(hostile).is_err(),
                "accepted hostile model_id {hostile:?}"
            );
            assert!(
                shard_dest_path("/store", hostile, 0, 1).is_none(),
                "built a destination path for hostile model_id {hostile:?}"
            );
        }
        // Over-long ids too — a peer should not get to push the path near
        // the filesystem's own limit.
        assert!(validated_model_id(&"a".repeat(MAX_MODEL_ID_LEN + 1)).is_err());
    }

    /// The negative control: without this, a validator that refused
    /// EVERYTHING would pass the test above and silently break Mode B.
    #[test]
    fn real_model_ids_are_accepted_and_stay_inside_the_store() {
        for good in [
            "gemma3-4b-q4k",
            "gpt-oss-20b.vindex3",
            "Muse_Glimmer-30B",
            "a",
            &"m".repeat(MAX_MODEL_ID_LEN),
        ] {
            assert!(
                validated_model_id(good).is_ok(),
                "rejected real id {good:?}"
            );
            let path = shard_dest_path("/store", good, 0, 1).expect("path for a real id");
            assert!(
                path.starts_with("/store"),
                "{good:?} escaped the store: {}",
                path.display()
            );
        }
    }
    use tempfile::TempDir;

    fn build_tar_in_memory(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut tar = tar::Builder::new(&mut buf);
            for (name, content) in files {
                let mut header = tar::Header::new_gnu();
                header.set_size(content.len() as u64);
                header.set_mode(0o644);
                header.set_cksum();
                tar.append_data(&mut header, name, *content).unwrap();
            }
            tar.finish().unwrap();
        }
        buf
    }

    #[test]
    fn shard_dest_path_combines_segments() {
        let p = shard_dest_path("/mnt/shards", "gemma4-26b", 0, 14).expect("safe id");
        assert!(p.ends_with("gemma4-26b/layers-0-14") || p.ends_with("gemma4-26b\\layers-0-14"));
    }

    #[tokio::test]
    async fn unpacks_tar_into_atomic_destination() {
        // End-to-end: serve a tar from a hyper-axum test server and verify the
        // client unpacks it into the right directory atomically.
        use axum::body::Body;
        use axum::extract::Path;
        use axum::http::{header, StatusCode};
        use axum::response::Response;
        use axum::routing::get;
        use axum::Router;

        async fn serve_tar(Path((_model, _range)): Path<(String, String)>) -> Response {
            let tar = build_tar_in_memory(&[
                ("index.json", b"{\"hello\":\"world\"}"),
                ("layer-0.bin", &[1u8, 2, 3, 4]),
            ]);
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "application/x-tar")
                .body(Body::from(tar))
                .unwrap()
        }

        let app = Router::new().route("/v1/shard/{model_id}/{range}", get(serve_tar));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server_handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let tmp = TempDir::new().unwrap();
        let store = tmp.path().to_str().unwrap();
        let origin = format!("http://{addr}");

        download_and_load_shard(&origin, store, "", "gemma-test", 0, 5)
            .await
            .expect("download must succeed");

        let dest = shard_dest_path(store, "gemma-test", 0, 5).expect("safe id");
        assert!(dest.is_dir(), "shard directory not created at {dest:?}");
        let manifest = std::fs::read(dest.join("index.json")).unwrap();
        assert_eq!(manifest, b"{\"hello\":\"world\"}");
        let layer = std::fs::read(dest.join("layer-0.bin")).unwrap();
        assert_eq!(layer, &[1u8, 2, 3, 4]);

        // tmp directory must have been renamed away.
        let tmp_dir = tmp.path().join("gemma-test").join(".tmp-layers-0-5");
        assert!(
            !tmp_dir.exists(),
            "stale tmp directory survived: {tmp_dir:?}"
        );

        // Idempotent re-call must not fail.
        download_and_load_shard(&origin, store, "", "gemma-test", 0, 5)
            .await
            .expect("re-download must be idempotent");

        server_handle.abort();
    }

    #[tokio::test]
    async fn rejects_hash_mismatch() {
        use axum::body::Body;
        use axum::extract::Path;
        use axum::http::{header, StatusCode};
        use axum::response::Response;
        use axum::routing::get;
        use axum::Router;

        async fn serve_tar(Path((_m, _r)): Path<(String, String)>) -> Response {
            let tar = build_tar_in_memory(&[("a.txt", b"hi")]);
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "application/x-tar")
                .body(Body::from(tar))
                .unwrap()
        }

        let app = Router::new().route("/v1/shard/{model_id}/{range}", get(serve_tar));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let tmp = TempDir::new().unwrap();
        let store = tmp.path().to_str().unwrap();
        let origin = format!("http://{addr}");

        let err = download_and_load_shard(
            &origin,
            store,
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "gemma-test",
            0,
            0,
        )
        .await
        .expect_err("expected hash mismatch error");
        assert!(format!("{err}").contains("hash mismatch"));
    }
}
