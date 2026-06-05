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
    VindexError::Io(std::io::Error::new(std::io::ErrorKind::Other, msg.into()))
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
        "https://api.github.com/repos/{}/{}/contents/{}?ref={}",
        gh.owner, gh.repo, gh.path, gh.git_ref
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
        .post("https://api.github.com/graphql")
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
    let url = format!("https://api.github.com/repos/{owner}/{repo}/git/refs/heads/{branch}");

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
    let url = format!("https://api.github.com/repos/{owner}/{repo}");
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
    fn parse_gh_url_missing_owner_errors() {
        assert!(GhUrl::parse("gh:///repo@main/file").is_err());
    }

    #[test]
    fn parse_gh_url_missing_slash_after_repo_gives_empty_path() {
        let u = GhUrl::parse("gh://owner/repo@main").unwrap();
        assert_eq!(u.path, "");
    }
}
