//! Tests for [`super`].
//!
//! Split out of `remote.rs` so the implementation file states the
//! behaviour and this one states the evidence for it.

use super::remote::*;

#[test]
fn parse_oid_index_extracts_lfs_files_only() {
    // Mixed entries: LFS file, non-LFS file (small text), directory.
    // Only the LFS entry should appear in the map.
    let body = serde_json::json!([
        {
            "type": "file",
            "path": "weights.bin",
            "lfs": {"oid": "abc123", "size": 1024}
        },
        {
            "type": "file",
            "path": "index.json"
            // no lfs key — git-tracked small text
        },
        {
            "type": "directory",
            "path": "layers",
            "lfs": {"oid": "should-not-appear"}
        }
    ]);
    let map = parse_lfs_oid_index(&body);
    assert_eq!(map.len(), 1);
    assert_eq!(map.get("weights.bin").map(|s| s.as_str()), Some("abc123"));
}

#[test]
fn parse_oid_index_handles_subdir_paths() {
    // HF returns paths like "layers/layer_00.weights" — they should
    // round-trip through the map verbatim (the publish code uses
    // them as filename keys).
    let body = serde_json::json!([
        {
            "type": "file",
            "path": "layers/layer_00.weights",
            "lfs": {"oid": "deadbeef"}
        }
    ]);
    let map = parse_lfs_oid_index(&body);
    assert_eq!(
        map.get("layers/layer_00.weights").map(|s| s.as_str()),
        Some("deadbeef"),
    );
}

#[test]
fn parse_oid_index_non_array_body_yields_empty_map() {
    // Fresh repo / unauth → HF can return a non-array body. Non-fatal:
    // caller falls back to "upload everything".
    let body = serde_json::json!({"error": "not found"});
    assert!(parse_lfs_oid_index(&body).is_empty());
}

#[test]
fn parse_oid_index_empty_array_yields_empty_map() {
    let body = serde_json::json!([]);
    assert!(parse_lfs_oid_index(&body).is_empty());
}

#[test]
fn parse_oid_index_missing_path_skips_entry() {
    // Defensive: malformed entries don't poison the whole walk.
    let body = serde_json::json!([
        {"type": "file", "lfs": {"oid": "x"}},
        {
            "type": "file",
            "path": "good.bin",
            "lfs": {"oid": "y"}
        }
    ]);
    let map = parse_lfs_oid_index(&body);
    assert_eq!(map.len(), 1);
    assert_eq!(map.get("good.bin").map(|s| s.as_str()), Some("y"));
}

// ─── HTTP-mocked integration tests ─────────────────────────────
//
// These set `LARQL_HF_TEST_BASE` to a per-test mockito URL and
// serialize via `#[serial]` because env vars are process-global.

use crate::format::huggingface::publish::protocol::TEST_BASE_ENV;
use serial_test::serial;

/// Port 0 is never a listener — connecting to it fails immediately at
/// the OS level (confirmed: ~0.3s including the client build, no
/// timeout wait needed), which is what the `.send()` transport-error
/// branches below need: a target that is reachable-shaped (a valid
/// `http://host:port` the client will actually attempt) but where the
/// connection itself never completes.
const UNREACHABLE_ENDPOINT: &str = "http://127.0.0.1:0";

/// RAII-style env-var override: sets the var, restores on drop.
struct EnvBaseGuard {
    prev: Option<String>,
}
impl EnvBaseGuard {
    fn new(value: &str) -> Self {
        let prev = std::env::var(TEST_BASE_ENV).ok();
        std::env::set_var(TEST_BASE_ENV, value);
        Self { prev }
    }
}
impl Drop for EnvBaseGuard {
    fn drop(&mut self) {
        match self.prev.take() {
            Some(v) => std::env::set_var(TEST_BASE_ENV, v),
            None => std::env::remove_var(TEST_BASE_ENV),
        }
    }
}

#[test]
fn parse_file_paths_takes_every_file_regardless_of_lfs() {
    // The contrast with `parse_lfs_oid_index`: pruning must see the
    // small git-tracked manifests too, because a renamed weight file
    // leaves its `*_manifest.json` sibling behind as well.
    let body = serde_json::json!([
        {"type": "file", "path": "interleaved_kquant.bin", "lfs": {"oid": "a"}},
        {"type": "file", "path": "interleaved_kquant_manifest.json"},
        {"type": "directory", "path": "layers"},
        {"type": "file", "path": "layers/layer_00.weights", "lfs": {"oid": "b"}}
    ]);
    let paths = parse_file_paths(&body);
    assert_eq!(
        paths,
        vec![
            "interleaved_kquant.bin",
            "interleaved_kquant_manifest.json",
            "layers/layer_00.weights"
        ],
        "directories must be dropped, non-LFS files kept"
    );
}

#[test]
fn parse_file_paths_tolerates_a_non_array_body() {
    assert!(parse_file_paths(&serde_json::json!({"error": "nope"})).is_empty());
}

#[test]
fn prune_exempts_git_plumbing_and_the_model_card() {
    // These are authored outside the vindex (`larql card`, repo
    // creation) so publishing must never treat them as stale.
    for p in [".gitattributes", ".gitignore", "README.md", "sub/README.md"] {
        assert!(is_prune_exempt(p), "{p} must be exempt");
    }
    // Everything the build emits is fair game.
    for p in [
        "interleaved_q4k.bin",
        "attn_weights_q4k_manifest.json",
        "layers/layer_00.weights",
        "index.json",
    ] {
        assert!(!is_prune_exempt(p), "{p} must be prunable");
    }
}

#[test]
#[serial]
fn fetch_remote_file_paths_lists_files_and_skips_directories() {
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());
    let mock = server
        .mock("GET", "/api/models/org/repo/tree/main?recursive=true")
        .match_header("authorization", "Bearer t")
        .with_status(200)
        .with_body(
            serde_json::json!([
                {"type": "file", "path": "index.json"},
                {"type": "directory", "path": "layers"},
                {"type": "file", "path": "layers/layer_00.weights", "lfs": {"oid": "x"}}
            ])
            .to_string(),
        )
        .create();
    let paths = fetch_remote_file_paths("org/repo", "t", "model").unwrap();
    mock.assert();
    assert_eq!(paths, vec!["index.json", "layers/layer_00.weights"]);
}

#[test]
#[serial]
fn fetch_remote_file_paths_dataset_uses_datasets_path_segment() {
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());
    let mock = server
        .mock("GET", "/api/datasets/org/repo/tree/main?recursive=true")
        .with_status(200)
        .with_body("[]")
        .create();
    assert!(fetch_remote_file_paths("org/repo", "t", "dataset")
        .unwrap()
        .is_empty());
    mock.assert();
}

#[test]
#[serial]
fn fetch_remote_file_paths_returns_empty_on_a_missing_repo() {
    // A fresh repo 404s on the tree API; that means "nothing to prune",
    // not an error that should abort a successful publish.
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());
    let mock = server
        .mock("GET", "/api/models/org/new/tree/main?recursive=true")
        .with_status(404)
        .create();
    assert!(fetch_remote_file_paths("org/new", "t", "model")
        .unwrap()
        .is_empty());
    mock.assert();
}

#[test]
#[serial]
fn fetch_remote_file_paths_send_failure_propagates() {
    let _guard = EnvBaseGuard::new(UNREACHABLE_ENDPOINT);
    let err = fetch_remote_file_paths("org/repo", "t", "model").unwrap_err();
    assert!(err.to_string().contains("HF tree fetch failed"), "{err}");
}

#[test]
#[serial]
fn fetch_remote_file_paths_json_parse_error_propagates() {
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());
    let _mock = server
        .mock("GET", "/api/models/org/repo/tree/main?recursive=true")
        .with_status(200)
        .with_body("not json")
        .create();
    let err = fetch_remote_file_paths("org/repo", "t", "model").unwrap_err();
    assert!(err.to_string().contains("HF tree JSON"), "{err}");
}

#[test]
#[serial]
fn delete_remote_files_posts_one_commit_with_a_deleted_entry_per_path() {
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());
    let mock = server
        .mock("POST", "/api/models/org/repo/commit/main")
        .match_header("authorization", "Bearer t")
        .match_body(mockito::Matcher::AllOf(vec![
            mockito::Matcher::Regex(r#""key":"header""#.into()),
            mockito::Matcher::Regex(r#""path":"interleaved_q4k\.bin""#.into()),
            mockito::Matcher::Regex(r#""path":"attn_weights_q4k\.bin""#.into()),
        ]))
        .with_status(200)
        .create();

    delete_remote_files(
        "org/repo",
        "t",
        "model",
        &[
            "interleaved_q4k.bin".to_string(),
            "attn_weights_q4k.bin".to_string(),
        ],
    )
    .unwrap();
    mock.assert();
}

#[test]
#[serial]
fn delete_remote_files_is_a_no_op_for_an_empty_list() {
    // No mock is registered: if this issued a request it would fail to
    // connect, so passing proves no commit was attempted.
    let server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());
    delete_remote_files("org/repo", "t", "model", &[]).unwrap();
}

#[test]
#[serial]
fn delete_remote_files_surfaces_a_rejected_commit() {
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());
    let _mock = server
        .mock("POST", "/api/models/org/repo/commit/main")
        .with_status(403)
        .with_body("forbidden")
        .create();
    let err = delete_remote_files("org/repo", "t", "model", &["a.bin".to_string()])
        .unwrap_err()
        .to_string();
    assert!(err.contains("403"), "error should name the status: {err}");
}

#[test]
#[serial]
fn delete_remote_files_send_failure_propagates() {
    let _guard = EnvBaseGuard::new(UNREACHABLE_ENDPOINT);
    let err = delete_remote_files("org/repo", "t", "model", &["a.bin".to_string()]).unwrap_err();
    assert!(err.to_string().contains("prune commit failed"), "{err}");
}

#[test]
#[serial]
fn fetch_remote_lfs_oids_parses_tree_response() {
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());

    let mock = server
        .mock("GET", "/api/models/org/repo/tree/main?recursive=true")
        .match_header("authorization", "Bearer t")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(
            serde_json::json!([
                {"type": "file", "path": "weights.bin",
                 "lfs": {"oid": "abc"}},
                {"type": "file", "path": "index.json"}
            ])
            .to_string(),
        )
        .create();

    let map = fetch_remote_lfs_oids("org/repo", "t", "model").unwrap();
    mock.assert();
    assert_eq!(map.len(), 1);
    assert_eq!(map.get("weights.bin").map(|s| s.as_str()), Some("abc"));
}

#[test]
#[serial]
fn fetch_remote_lfs_oids_dataset_uses_datasets_path_segment() {
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());

    let mock = server
        .mock("GET", "/api/datasets/org/repo/tree/main?recursive=true")
        .with_status(200)
        .with_body("[]")
        .create();

    let map = fetch_remote_lfs_oids("org/repo", "t", "dataset").unwrap();
    mock.assert();
    assert!(map.is_empty());
}

#[test]
#[serial]
fn fetch_remote_lfs_oids_404_returns_empty_map() {
    // Fresh repo: tree endpoint 404s before the first commit.
    // Caller falls back to "upload everything", so this MUST NOT
    // surface as an error.
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());

    let mock = server
        .mock("GET", "/api/models/org/repo/tree/main?recursive=true")
        .with_status(404)
        .create();

    let map = fetch_remote_lfs_oids("org/repo", "t", "model").unwrap();
    mock.assert();
    assert!(map.is_empty());
}

#[test]
#[serial]
fn fetch_remote_lfs_oids_send_failure_propagates() {
    // No mockito server at all — the endpoint is unreachable, so the
    // `.send()` call itself fails before any status code exists.
    let _guard = EnvBaseGuard::new(UNREACHABLE_ENDPOINT);
    let err = fetch_remote_lfs_oids("org/repo", "t", "model").unwrap_err();
    assert!(err.to_string().contains("HF tree fetch failed"), "{err}");
}

#[test]
#[serial]
fn fetch_remote_lfs_oids_json_parse_error_propagates() {
    // 200 OK but a body that isn't JSON at all (vs. the well-formed
    // wrong-shape body `_non_array_body_yields_empty_map` covers,
    // which `parse_lfs_oid_index` handles gracefully) — `resp.json()`
    // itself must fail here, and that must surface as an error rather
    // than a silently-empty map.
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());

    let _mock = server
        .mock("GET", "/api/models/org/repo/tree/main?recursive=true")
        .with_status(200)
        .with_body("not json")
        .create();

    let err = fetch_remote_lfs_oids("org/repo", "t", "model").unwrap_err();
    assert!(err.to_string().contains("HF tree JSON"), "{err}");
}

// ── fetch_repo_head_sha: the pinned-revision fetch ───────────────────

#[test]
#[serial]
fn fetch_repo_head_sha_reads_the_sha_field() {
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());

    let mock = server
        .mock("GET", "/api/models/org/repo")
        .match_header("authorization", "Bearer t")
        .with_status(200)
        .with_body(r#"{"sha":"deadbeefcafebabe","siblings":[]}"#)
        .create();

    let sha = fetch_repo_head_sha("org/repo", "t", "model").unwrap();
    mock.assert();
    assert_eq!(sha, "deadbeefcafebabe");
}

#[test]
#[serial]
fn fetch_repo_head_sha_dataset_uses_datasets_path_segment() {
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());

    let mock = server
        .mock("GET", "/api/datasets/org/repo")
        .with_status(200)
        .with_body(r#"{"sha":"abc123"}"#)
        .create();

    let sha = fetch_repo_head_sha("org/repo", "t", "dataset").unwrap();
    mock.assert();
    assert_eq!(sha, "abc123");
}

#[test]
#[serial]
fn fetch_repo_head_sha_propagates_a_non_success_status() {
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());

    let _mock = server
        .mock("GET", "/api/models/org/repo")
        .with_status(500)
        .create();

    let err = fetch_repo_head_sha("org/repo", "t", "model").unwrap_err();
    assert!(err.to_string().contains("org/repo"), "{err}");
}

#[test]
#[serial]
fn fetch_repo_head_sha_errors_when_the_sha_field_is_absent() {
    // A malformed or unexpected response body must not be silently
    // treated as "no revision" — a claimed registry entry's pinned
    // revision must be a real fact or a real refusal, never absent.
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());

    let _mock = server
        .mock("GET", "/api/models/org/repo")
        .with_status(200)
        .with_body(r#"{"siblings":[]}"#)
        .create();

    let err = fetch_repo_head_sha("org/repo", "t", "model").unwrap_err();
    assert!(err.to_string().contains("sha"), "{err}");
}

#[test]
#[serial]
fn fetch_repo_head_sha_send_failure_propagates() {
    let _guard = EnvBaseGuard::new(UNREACHABLE_ENDPOINT);
    let err = fetch_repo_head_sha("org/repo", "t", "model").unwrap_err();
    assert!(
        err.to_string().contains("HF repo-info fetch failed"),
        "{err}"
    );
}

#[test]
#[serial]
fn fetch_repo_head_sha_json_parse_error_propagates() {
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());

    let _mock = server
        .mock("GET", "/api/models/org/repo")
        .with_status(200)
        .with_body("not json")
        .create();

    let err = fetch_repo_head_sha("org/repo", "t", "model").unwrap_err();
    assert!(err.to_string().contains("HF repo-info JSON"), "{err}");
}

#[test]
#[serial]
fn fetch_remote_lfs_oids_non_array_body_yields_empty_map() {
    // 200 OK with a JSON object (not array) body — defensive path
    // already covered by the pure parser test, but exercise the
    // full HTTP path here too.
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());

    let mock = server
        .mock("GET", "/api/models/org/repo/tree/main?recursive=true")
        .with_status(200)
        .with_body(r#"{"error": "weird"}"#)
        .create();

    let map = fetch_remote_lfs_oids("org/repo", "t", "model").unwrap();
    mock.assert();
    assert!(map.is_empty());
}

#[test]
#[serial]
fn create_hf_repo_success() {
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());

    let mock = server
        .mock("POST", "/api/repos/create")
        .match_header("authorization", "Bearer t")
        .match_body(mockito::Matcher::PartialJson(
            serde_json::json!({"name": "repo", "type": "model"}),
        ))
        .with_status(200)
        .with_body("{}")
        .create();

    create_hf_repo("org/repo", "t", "model", false).unwrap();
    mock.assert();
}

#[test]
#[serial]
fn create_hf_repo_private_true_sets_the_field() {
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());

    let mock = server
        .mock("POST", "/api/repos/create")
        .match_body(mockito::Matcher::PartialJson(
            serde_json::json!({"private": true}),
        ))
        .with_status(200)
        .with_body("{}")
        .create();

    create_hf_repo("org/repo", "t", "model", true).unwrap();
    mock.assert();
}

#[test]
#[serial]
fn create_hf_repo_409_conflict_is_ok() {
    // 409 Conflict means "already exists" — that's fine, the publish
    // path proceeds to commit. Must NOT surface as an error.
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());

    let mock = server
        .mock("POST", "/api/repos/create")
        .with_status(409)
        .with_body("conflict")
        .create();

    create_hf_repo("org/repo", "t", "model", false).unwrap();
    mock.assert();
}

#[test]
#[serial]
fn create_hf_repo_other_error_propagates() {
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());

    let mock = server
        .mock("POST", "/api/repos/create")
        .with_status(500)
        .with_body("boom")
        .create();

    let err = create_hf_repo("org/repo", "t", "model", false).expect_err("500 must error");
    mock.assert();
    let msg = err.to_string();
    assert!(msg.contains("500"), "{msg}");
}

#[test]
#[serial]
fn create_hf_repo_uses_last_path_segment_as_name() {
    // HF's repos/create body uses just the repo name (the part after
    // the slash), not the full owner/repo. A repo_id without a slash
    // should pass through verbatim.
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());

    let mock = server
        .mock("POST", "/api/repos/create")
        .match_body(mockito::Matcher::PartialJson(
            serde_json::json!({"name": "loose-repo"}),
        ))
        .with_status(200)
        .create();

    create_hf_repo("loose-repo", "t", "model", false).unwrap();
    mock.assert();
}

#[test]
fn create_hf_repo_body_omits_organization_for_a_bare_name() {
    // A repo_id with no `/` names no owner at all — `organization`
    // must be entirely absent so HF defaults the create to the
    // token's own namespace, which is exactly what "no owner
    // requested" should mean.
    let body = create_hf_repo_body("loose-repo", "model", false);
    assert_eq!(body["name"], "loose-repo");
    assert!(
        body.get("organization").is_none(),
        "unexpected organization field: {body}"
    );
}

#[test]
fn create_hf_repo_body_sends_organization_for_a_namespaced_repo() {
    // The real fix: omitting `organization` makes HF default repo
    // creation to the *token's own* namespace, not the caller's
    // intended one — `--repo larql/granite-4.1-3b` from a
    // `chrishayuk`-owned token used to silently create
    // `chrishayuk/granite-4.1-3b` instead, and the next call (which
    // does address the real repo_id) 404'd on a repo that was never
    // created. `organization` must carry the owner verbatim.
    let body = create_hf_repo_body("larql/granite-4.1-3b", "model", false);
    assert_eq!(body["name"], "granite-4.1-3b");
    assert_eq!(body["organization"], "larql");
}

#[test]
#[serial]
fn create_hf_repo_send_failure_propagates() {
    let _guard = EnvBaseGuard::new(UNREACHABLE_ENDPOINT);
    let err = create_hf_repo("org/repo", "t", "model", false).unwrap_err();
    assert!(err.to_string().contains("HF API error"), "{err}");
}

#[test]
#[serial]
fn update_repo_visibility_success() {
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());

    let mock = server
        .mock("PUT", "/api/models/org/repo/settings")
        .match_header("authorization", "Bearer t")
        .match_body(mockito::Matcher::PartialJson(
            serde_json::json!({"private": false}),
        ))
        .with_status(200)
        .with_body("{}")
        .create();

    update_repo_visibility("org/repo", "t", "model", false).unwrap();
    mock.assert();
}

#[test]
#[serial]
fn update_repo_visibility_uses_dataset_plural() {
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());

    let mock = server
        .mock("PUT", "/api/datasets/org/repo/settings")
        .with_status(200)
        .with_body("{}")
        .create();

    update_repo_visibility("org/repo", "t", "dataset", true).unwrap();
    mock.assert();
}

#[test]
#[serial]
fn update_repo_visibility_error_propagates() {
    let mut server = mockito::Server::new();
    let _guard = EnvBaseGuard::new(&server.url());

    let mock = server
        .mock("PUT", "/api/models/org/repo/settings")
        .with_status(403)
        .with_body("forbidden")
        .create();

    let err = update_repo_visibility("org/repo", "t", "model", false).expect_err("403 must error");
    mock.assert();
    let msg = err.to_string();
    assert!(msg.contains("403"), "{msg}");
}

#[test]
#[serial]
fn update_repo_visibility_send_failure_propagates() {
    let _guard = EnvBaseGuard::new(UNREACHABLE_ENDPOINT);
    let err = update_repo_visibility("org/repo", "t", "model", false).unwrap_err();
    assert!(err.to_string().contains("HF API error"), "{err}");
}
