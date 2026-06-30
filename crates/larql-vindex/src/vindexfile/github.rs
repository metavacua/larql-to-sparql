//! GitHub REST + GraphQL helpers for the `gh://` Vindexfile resolver
//! and the `EXPORT PATCH TO "gh://..."` write path.
//!
//! URL scheme: `gh://owner/repo@ref/path/to/file`
//! e.g.        `gh://metavacua/larql-to-sparql@main/Vindexfile`
//!             `gh://metavacua/larql-to-sparql@knowledge/patches/foo.vlp`
//!
//! Authentication: `GITHUB_TOKEN` env-var.  Required for private repos
//! and strongly recommended for public ones (avoids the 60 req/hr
//! unauthenticated rate limit).

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::VindexError;

fn io_err(msg: impl Into<String>) -> VindexError {
    VindexError::Io(std::io::Error::other(msg.into()))
}

// ── API base URL (overridable for tests) ────────────────────────────────

/// Env-var used by tests to redirect all GitHub API calls to a local
/// mockito server instead of `https://api.github.com`.
pub(crate) const GH_TEST_BASE_ENV: &str = "LARQL_GH_TEST_BASE";

fn gh_api_base() -> String {
    std::env::var(GH_TEST_BASE_ENV).unwrap_or_else(|_| "https://api.github.com".to_string())
}

// ── URL parsing ─────────────────────────────────────────────────────────

/// Parsed `gh://owner/repo@ref/path` URL.
pub struct GhUrl {
    pub owner: String,
    pub repo: String,
    pub git_ref: String,
    pub path: String,
}

impl GhUrl {
    /// Parse `gh://owner/repo@ref/path`.
    ///
    /// The `@ref` part is optional; if absent, `ref` defaults to `"HEAD"`.
    pub fn parse(raw: &str) -> Result<Self, VindexError> {
        let rest = raw
            .strip_prefix("gh://")
            .ok_or_else(|| VindexError::Parse(format!("not a gh:// URL: {raw}")))?;

        // Split owner/repo@ref/path
        let mut parts = rest.splitn(2, '/');
        let owner = parts
            .next()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| VindexError::Parse(format!("gh:// URL missing owner: {raw}")))?
            .to_string();

        let rest = parts
            .next()
            .ok_or_else(|| VindexError::Parse(format!("gh:// URL missing repo: {raw}")))?;

        // repo may carry @ref
        let (repo_part, path_rest) = rest.split_once('/').unwrap_or((rest, ""));

        let (repo, git_ref) = if let Some((r, rf)) = repo_part.split_once('@') {
            (r.to_string(), rf.to_string())
        } else {
            (repo_part.to_string(), "HEAD".to_string())
        };

        if repo.is_empty() {
            return Err(VindexError::Parse(format!(
                "gh:// URL missing repo name: {raw}"
            )));
        }

        Ok(Self {
            owner,
            repo,
            git_ref,
            path: path_rest.to_string(),
        })
    }
}

// ── REST file download ───────────────────────────────────────────────────

#[derive(Deserialize)]
struct ContentsResponse {
    content: Option<String>,
    encoding: Option<String>,
    download_url: Option<String>,
}

/// Download a file from GitHub and return the path to a local temp copy.
///
/// Uses the REST Contents API (`GET /repos/{owner}/{repo}/contents/{path}?ref={ref}`).
/// Falls back to `download_url` when the file exceeds 1 MB (GitHub's inline limit).
pub fn download_gh_file(gh: &GhUrl) -> Result<PathBuf, VindexError> {
    let token = std::env::var("GITHUB_TOKEN").ok();

    let url = format!(
        "{}/repos/{}/{}/contents/{}?ref={}",
        gh_api_base(),
        gh.owner,
        gh.repo,
        gh.path,
        gh.git_ref
    );

    let mut req = reqwest::blocking::Client::new()
        .get(&url)
        .header("Accept", "application/vnd.github.v3+json")
        .header("User-Agent", "larql-vindex/0.1");

    if let Some(ref tok) = token {
        req = req.bearer_auth(tok);
    }

    let resp = req
        .send()
        .map_err(|e| io_err(format!("gh:// fetch failed for {url}: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        return Err(io_err(format!("gh:// GET {url} → {status}: {body}")));
    }

    let contents: ContentsResponse = resp
        .json()
        .map_err(|e| io_err(format!("gh:// JSON parse failed: {e}")))?;

    let bytes = if contents.encoding.as_deref() == Some("base64") {
        let raw = contents.content.unwrap_or_default().replace('\n', "");
        base64_decode(&raw)?
    } else if let Some(dl_url) = contents.download_url {
        // File > 1 MB — use the direct download URL
        let mut req2 = reqwest::blocking::Client::new()
            .get(&dl_url)
            .header("User-Agent", "larql-vindex/0.1");
        if let Some(ref tok) = token {
            req2 = req2.bearer_auth(tok);
        }
        req2.send()
            .map_err(|e| io_err(format!("gh:// download_url failed: {e}")))?
            .bytes()
            .map_err(|e| io_err(format!("gh:// read bytes failed: {e}")))?
            .to_vec()
    } else {
        return Err(io_err(
            "gh:// response had neither base64 content nor download_url",
        ));
    };

    // Write to a named temp file that persists until the caller is done with it.
    let filename = Path::new(&gh.path)
        .file_name()
        .unwrap_or(std::ffi::OsStr::new("gh_file"))
        .to_string_lossy();
    let tmp_path = std::env::temp_dir().join(format!(
        "larql_gh_{}_{}_{}_{}",
        gh.owner,
        gh.repo,
        gh.git_ref.replace('/', "_"),
        filename
    ));
    std::fs::write(&tmp_path, &bytes)
        .map_err(|e| io_err(format!("gh:// write temp failed: {e}")))?;

    Ok(tmp_path)
}

// ── GraphQL commit ───────────────────────────────────────────────────────

/// A single file addition for a GitHub GraphQL commit.
pub struct FileAddition {
    pub path: String,
    pub contents_base64: String,
}

/// Commit one or more file additions to a GitHub branch atomically via
/// the `createCommitOnBranch` GraphQL mutation.
///
/// Returns the new commit OID on success.
pub fn graphql_commit(
    owner: &str,
    repo: &str,
    branch: &str,
    message: &str,
    files: &[FileAddition],
) -> Result<String, VindexError> {
    let token = std::env::var("GITHUB_TOKEN")
        .map_err(|_| io_err("GITHUB_TOKEN env-var not set — required for gh:// write"))?;

    // Step 1: get the current HEAD OID for expectedHeadOid
    let head_oid = get_branch_head_oid(owner, repo, branch, &token)?;

    // Step 2: build the additions array
    let additions_json = files
        .iter()
        .map(|f| {
            format!(
                r#"{{"path": {}, "contents": {}}}"#,
                serde_json::to_string(&f.path).unwrap(),
                serde_json::to_string(&f.contents_base64).unwrap()
            )
        })
        .collect::<Vec<_>>()
        .join(", ");

    let mutation = r#"
mutation($input: CreateCommitOnBranchInput!) {
  createCommitOnBranch(input: $input) {
    commit { oid }
  }
}
"#;

    let variables = format!(
        r#"{{
  "input": {{
    "branch": {{
      "repositoryNameWithOwner": "{owner}/{repo}",
      "branchName": "{branch}"
    }},
    "message": {{ "headline": {message_json} }},
    "fileChanges": {{ "additions": [{additions}] }},
    "expectedHeadOid": "{head_oid}"
  }}
}}"#,
        owner = owner,
        repo = repo,
        branch = branch,
        message_json = serde_json::to_string(message).unwrap(),
        additions = additions_json,
        head_oid = head_oid
    );

    let body = serde_json::json!({
        "query": mutation,
        "variables": serde_json::from_str::<serde_json::Value>(&variables)
            .map_err(|e| io_err(format!("variables JSON: {e}")))?
    });

    let resp = reqwest::blocking::Client::new()
        .post(format!("{}/graphql", gh_api_base()))
        .bearer_auth(&token)
        .header("User-Agent", "larql-vindex/0.1")
        .json(&body)
        .send()
        .map_err(|e| io_err(format!("GraphQL request failed: {e}")))?;

    let status = resp.status();
    let text = resp
        .text()
        .map_err(|e| io_err(format!("GraphQL response read failed: {e}")))?;

    if !status.is_success() {
        return Err(io_err(format!("GitHub GraphQL → {status}: {text}")));
    }

    // Extract commit OID
    let val: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| io_err(format!("GraphQL JSON: {e}")))?;

    if let Some(errors) = val.get("errors") {
        return Err(io_err(format!("GitHub GraphQL errors: {errors}")));
    }

    let oid = val
        .pointer("/data/createCommitOnBranch/commit/oid")
        .and_then(|v| v.as_str())
        .ok_or_else(|| io_err(format!("unexpected GraphQL response: {text}")))?
        .to_string();

    Ok(oid)
}

// ── Internal helpers ─────────────────────────────────────────────────────

fn get_branch_head_oid(
    owner: &str,
    repo: &str,
    branch: &str,
    token: &str,
) -> Result<String, VindexError> {
    let url = format!(
        "{}/repos/{owner}/{repo}/git/refs/heads/{branch}",
        gh_api_base()
    );

    let resp = reqwest::blocking::Client::new()
        .get(&url)
        .bearer_auth(token)
        .header("Accept", "application/vnd.github.v3+json")
        .header("User-Agent", "larql-vindex/0.1")
        .send()
        .map_err(|e| io_err(format!("GET branch ref failed: {e}")))?;

    if resp.status().as_u16() == 404 {
        // Branch doesn't exist yet — use the default branch HEAD
        return get_default_branch_head_oid(owner, repo, token);
    }

    let val: serde_json::Value = resp
        .json()
        .map_err(|e| io_err(format!("branch ref JSON: {e}")))?;

    val.pointer("/object/sha")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| {
            io_err(format!(
                "branch ref missing sha for {owner}/{repo}@{branch}"
            ))
        })
}

fn get_default_branch_head_oid(
    owner: &str,
    repo: &str,
    token: &str,
) -> Result<String, VindexError> {
    let url = format!("{}/repos/{owner}/{repo}", gh_api_base());
    let resp = reqwest::blocking::Client::new()
        .get(&url)
        .bearer_auth(token)
        .header("Accept", "application/vnd.github.v3+json")
        .header("User-Agent", "larql-vindex/0.1")
        .send()
        .map_err(|e| io_err(format!("GET repo failed: {e}")))?;

    let val: serde_json::Value = resp.json().map_err(|e| io_err(format!("repo JSON: {e}")))?;

    let default_branch = val
        .get("default_branch")
        .and_then(|v| v.as_str())
        .unwrap_or("main")
        .to_string();

    get_branch_head_oid(owner, repo, &default_branch, token)
}

fn base64_decode(s: &str) -> Result<Vec<u8>, VindexError> {
    use std::io::Read;
    // Use the standard base64 alphabet; GitHub uses standard (not URL-safe)
    let mut decoder =
        base64::read::DecoderReader::new(s.as_bytes(), &base64::engine::general_purpose::STANDARD);
    let mut out = Vec::new();
    decoder
        .read_to_end(&mut out)
        .map_err(|e| io_err(format!("base64 decode failed: {e}")))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;
    use serial_test::serial;

    // ── Pure parsing tests (no network, no serial needed) ───────────────

    #[test]
    fn parse_gh_url_with_ref() {
        let u = GhUrl::parse("gh://metavacua/larql-to-sparql@main/Vindexfile").unwrap();
        assert_eq!(u.owner, "metavacua");
        assert_eq!(u.repo, "larql-to-sparql");
        assert_eq!(u.git_ref, "main");
        assert_eq!(u.path, "Vindexfile");
    }

    #[test]
    fn parse_gh_url_without_ref_defaults_to_head() {
        let u = GhUrl::parse("gh://metavacua/larql-to-sparql/Vindexfile").unwrap();
        assert_eq!(u.git_ref, "HEAD");
        assert_eq!(u.path, "Vindexfile");
    }

    #[test]
    fn parse_gh_url_nested_path() {
        let u = GhUrl::parse("gh://metavacua/larql-to-sparql@knowledge/patches/foo.vlp").unwrap();
        assert_eq!(u.path, "patches/foo.vlp");
    }

    #[test]
    fn parse_gh_url_missing_prefix_errors() {
        assert!(GhUrl::parse("https://github.com/owner/repo").is_err());
    }

    #[test]
    fn parse_gh_url_missing_owner_errors() {
        assert!(GhUrl::parse("gh:///repo@main/file").is_err());
    }

    #[test]
    fn parse_gh_url_missing_repo_errors() {
        assert!(GhUrl::parse("gh://owner").is_err());
    }

    #[test]
    fn parse_gh_url_empty_repo_name_errors() {
        assert!(GhUrl::parse("gh://owner/@main/file").is_err());
    }

    #[test]
    fn parse_gh_url_missing_slash_after_repo_gives_empty_path() {
        let u = GhUrl::parse("gh://owner/repo@main").unwrap();
        assert_eq!(u.path, "");
    }

    #[test]
    fn base64_decode_valid() {
        // "hello" in standard base64
        let decoded = base64_decode("aGVsbG8=").unwrap();
        assert_eq!(decoded, b"hello");
    }

    #[test]
    fn base64_decode_invalid_returns_error() {
        assert!(base64_decode("!!!not-base64!!!").is_err());
    }

    // ── RAII env-var guard ───────────────────────────────────────────────

    struct EnvGuard {
        key: &'static str,
        prev: Option<String>,
    }
    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::set_var(key, value);
            Self { key, prev }
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match self.prev.take() {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }

    // ── HTTP-mocked tests ────────────────────────────────────────────────
    //
    // Each test spins up its own mockito::Server on an ephemeral port and
    // overrides LARQL_GH_TEST_BASE to point at it. Tests are serialized
    // via `#[serial]` because env vars are process-global.

    #[test]
    #[serial]
    fn download_gh_file_base64_content_success() {
        let mut server = mockito::Server::new();
        let _base = EnvGuard::set(GH_TEST_BASE_ENV, &server.url());
        // Remove GITHUB_TOKEN so auth header is absent (exercises the no-token branch)
        let _tok = {
            let prev = std::env::var("GITHUB_TOKEN").ok();
            std::env::remove_var("GITHUB_TOKEN");
            prev
        };

        let mock = server
            .mock("GET", "/repos/owner/repo/contents/path/file.txt?ref=main")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "encoding": "base64",
                    "content": base64::engine::general_purpose::STANDARD
                        .encode(b"hello world") + "\n",
                    "download_url": null
                })
                .to_string(),
            )
            .create();

        let gh = GhUrl {
            owner: "owner".into(),
            repo: "repo".into(),
            git_ref: "main".into(),
            path: "path/file.txt".into(),
        };
        let tmp = download_gh_file(&gh).unwrap();
        mock.assert();
        assert_eq!(std::fs::read(&tmp).unwrap(), b"hello world");
    }

    #[test]
    #[serial]
    fn download_gh_file_with_token_sets_auth_header() {
        let mut server = mockito::Server::new();
        let _base = EnvGuard::set(GH_TEST_BASE_ENV, &server.url());
        let _tok = EnvGuard::set("GITHUB_TOKEN", "mytoken");

        let mock = server
            .mock("GET", "/repos/o/r/contents/f?ref=HEAD")
            .match_header("authorization", "Bearer mytoken")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "encoding": "base64",
                    "content": base64::engine::general_purpose::STANDARD.encode(b"data"),
                    "download_url": null
                })
                .to_string(),
            )
            .create();

        let gh = GhUrl {
            owner: "o".into(),
            repo: "r".into(),
            git_ref: "HEAD".into(),
            path: "f".into(),
        };
        download_gh_file(&gh).unwrap();
        mock.assert();
    }

    #[test]
    #[serial]
    fn download_gh_file_http_error_propagates() {
        let mut server = mockito::Server::new();
        let _base = EnvGuard::set(GH_TEST_BASE_ENV, &server.url());

        let mock = server
            .mock("GET", "/repos/o/r/contents/f?ref=HEAD")
            .with_status(404)
            .with_body("not found")
            .create();

        let gh = GhUrl {
            owner: "o".into(),
            repo: "r".into(),
            git_ref: "HEAD".into(),
            path: "f".into(),
        };
        assert!(download_gh_file(&gh).is_err());
        mock.assert();
    }

    #[test]
    #[serial]
    fn download_gh_file_download_url_fallback() {
        // Simulates a >1 MB file: Contents API returns no inline content,
        // just a download_url; function must follow it.
        let mut server = mockito::Server::new();
        let _base = EnvGuard::set(GH_TEST_BASE_ENV, &server.url());

        let dl_path = "/raw/owner/repo/main/big.bin";
        let contents_mock = server
            .mock("GET", "/repos/owner/repo/contents/big.bin?ref=main")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "encoding": null,
                    "content": null,
                    "download_url": format!("{}{}", server.url(), dl_path)
                })
                .to_string(),
            )
            .create();

        let raw_mock = server
            .mock("GET", dl_path)
            .with_status(200)
            .with_body(b"bigdata".as_ref())
            .create();

        let gh = GhUrl {
            owner: "owner".into(),
            repo: "repo".into(),
            git_ref: "main".into(),
            path: "big.bin".into(),
        };
        let tmp = download_gh_file(&gh).unwrap();
        contents_mock.assert();
        raw_mock.assert();
        assert_eq!(std::fs::read(&tmp).unwrap(), b"bigdata");
    }

    #[test]
    #[serial]
    fn download_gh_file_no_content_no_download_url_errors() {
        let mut server = mockito::Server::new();
        let _base = EnvGuard::set(GH_TEST_BASE_ENV, &server.url());

        let _mock = server
            .mock("GET", "/repos/o/r/contents/f?ref=HEAD")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({"encoding": null, "content": null, "download_url": null})
                    .to_string(),
            )
            .create();

        let gh = GhUrl {
            owner: "o".into(),
            repo: "r".into(),
            git_ref: "HEAD".into(),
            path: "f".into(),
        };
        assert!(download_gh_file(&gh).is_err());
    }

    #[test]
    fn graphql_commit_missing_token_errors() {
        // No network needed — token check fails before any HTTP call.
        std::env::remove_var("GITHUB_TOKEN");
        let result = graphql_commit("o", "r", "b", "msg", &[]);
        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn graphql_commit_success() {
        let mut server = mockito::Server::new();
        let _base = EnvGuard::set(GH_TEST_BASE_ENV, &server.url());
        let _tok = EnvGuard::set("GITHUB_TOKEN", "tok");

        // Step 1: GET branch ref → returns sha
        let ref_mock = server
            .mock("GET", "/repos/owner/repo/git/refs/heads/main")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "ref": "refs/heads/main",
                    "object": {"sha": "abc123def456", "type": "commit"}
                })
                .to_string(),
            )
            .create();

        // Step 2: POST GraphQL mutation → returns commit OID
        let gql_mock = server
            .mock("POST", "/graphql")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "data": {
                        "createCommitOnBranch": {
                            "commit": {"oid": "newoid123"}
                        }
                    }
                })
                .to_string(),
            )
            .create();

        let oid = graphql_commit(
            "owner",
            "repo",
            "main",
            "test commit",
            &[FileAddition {
                path: "Vindexfile".into(),
                contents_base64: "Y29udGVudA==".into(),
            }],
        )
        .unwrap();

        ref_mock.assert();
        gql_mock.assert();
        assert_eq!(oid, "newoid123");
    }

    #[test]
    #[serial]
    fn graphql_commit_branch_404_falls_back_to_default() {
        let mut server = mockito::Server::new();
        let _base = EnvGuard::set(GH_TEST_BASE_ENV, &server.url());
        let _tok = EnvGuard::set("GITHUB_TOKEN", "tok");

        // Branch lookup 404s
        let ref_404 = server
            .mock("GET", "/repos/o/r/git/refs/heads/newbranch")
            .with_status(404)
            .create();

        // Fallback: GET repo info → default_branch = "main"
        let repo_mock = server
            .mock("GET", "/repos/o/r")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(serde_json::json!({"default_branch": "main"}).to_string())
            .create();

        // Retry branch lookup with "main"
        let ref_main = server
            .mock("GET", "/repos/o/r/git/refs/heads/main")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(serde_json::json!({"object": {"sha": "headsha"}}).to_string())
            .create();

        // GraphQL commit
        let gql_mock = server
            .mock("POST", "/graphql")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "data": {
                        "createCommitOnBranch": {"commit": {"oid": "oid42"}}
                    }
                })
                .to_string(),
            )
            .create();

        let oid = graphql_commit("o", "r", "newbranch", "msg", &[]).unwrap();
        ref_404.assert();
        repo_mock.assert();
        ref_main.assert();
        gql_mock.assert();
        assert_eq!(oid, "oid42");
    }

    #[test]
    #[serial]
    fn graphql_commit_graphql_errors_propagates() {
        let mut server = mockito::Server::new();
        let _base = EnvGuard::set(GH_TEST_BASE_ENV, &server.url());
        let _tok = EnvGuard::set("GITHUB_TOKEN", "tok");

        let _ref_mock = server
            .mock("GET", "/repos/o/r/git/refs/heads/b")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(serde_json::json!({"object": {"sha": "s"}}).to_string())
            .create();

        let _gql_mock = server
            .mock("POST", "/graphql")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(serde_json::json!({"errors": [{"message": "branch not found"}]}).to_string())
            .create();

        assert!(graphql_commit("o", "r", "b", "msg", &[]).is_err());
    }

    #[test]
    #[serial]
    fn graphql_commit_http_error_propagates() {
        let mut server = mockito::Server::new();
        let _base = EnvGuard::set(GH_TEST_BASE_ENV, &server.url());
        let _tok = EnvGuard::set("GITHUB_TOKEN", "tok");

        let _ref_mock = server
            .mock("GET", "/repos/o/r/git/refs/heads/b")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(serde_json::json!({"object": {"sha": "s"}}).to_string())
            .create();

        let _gql_mock = server
            .mock("POST", "/graphql")
            .with_status(401)
            .with_body("Unauthorized")
            .create();

        assert!(graphql_commit("o", "r", "b", "msg", &[]).is_err());
    }

    #[test]
    #[serial]
    fn graphql_commit_missing_oid_errors() {
        let mut server = mockito::Server::new();
        let _base = EnvGuard::set(GH_TEST_BASE_ENV, &server.url());
        let _tok = EnvGuard::set("GITHUB_TOKEN", "tok");

        let _ref_mock = server
            .mock("GET", "/repos/o/r/git/refs/heads/b")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(serde_json::json!({"object": {"sha": "s"}}).to_string())
            .create();

        // Response has no OID in the expected path
        let _gql_mock = server
            .mock("POST", "/graphql")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(serde_json::json!({"data": {}}).to_string())
            .create();

        assert!(graphql_commit("o", "r", "b", "msg", &[]).is_err());
    }

    #[test]
    #[serial]
    fn get_branch_head_oid_missing_sha_errors() {
        let mut server = mockito::Server::new();
        let _base = EnvGuard::set(GH_TEST_BASE_ENV, &server.url());
        let _tok = EnvGuard::set("GITHUB_TOKEN", "tok");

        let _ref_mock = server
            .mock("GET", "/repos/o/r/git/refs/heads/b")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(serde_json::json!({"object": {}}).to_string()) // no sha
            .create();

        let _gql_mock = server
            .mock("POST", "/graphql")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(serde_json::json!({"data": {}}).to_string())
            .create();

        assert!(graphql_commit("o", "r", "b", "msg", &[]).is_err());
    }
}
