//! Unit tests for the session projection. The handlers themselves are
//! driven over real HTTP in `tests/test_http_sessions.rs`.

use super::*;
use crate::session::SessionSummary;

fn summary() -> SessionSummary {
    SessionSummary {
        id: "s-1".into(),
        model: "m-test".into(),
        created_at: 1_700_000_000,
        last_used_at: 1_700_000_060,
        expires_at: 1_700_003_660,
        patch_names: vec!["alpha".into(), "beta".into()],
        resumptions: 3,
        reused_tokens_total: 276,
    }
}

#[test]
fn renders_identity_lifecycle_and_patch_identities() {
    let json = session_json(&summary(), &SessionContinuation::default());
    assert_eq!(json["object"], SESSION_OBJECT);
    assert_eq!(json["id"], "s-1");
    assert_eq!(json["model"], "m-test");
    assert_eq!(json["created_at"], 1_700_000_000u64);
    assert_eq!(json["last_used_at"], 1_700_000_060u64);
    assert_eq!(json["expires_at"], 1_700_003_660u64);
    assert_eq!(json["state"], STATE_ACTIVE);
    assert_eq!(json["patches"]["active"], 2);
    assert_eq!(json["patches"]["ids"][0], "alpha");
    assert_eq!(json["patches"]["ids"][1], "beta");
}

#[test]
fn absent_continuation_reads_as_unavailable_and_zeroed() {
    let json = session_json(&summary(), &SessionContinuation::default());
    assert_eq!(json["continuation"]["available"], false);
    assert_eq!(json["continuation"]["input_tokens"], 0);
    // Cumulative counters belong to the session, not to whatever state
    // happens to be resident, so they survive an emptied cache.
    assert_eq!(json["continuation"]["resumptions"], 3);
    assert_eq!(json["continuation"]["reused_tokens_total"], 276);
}

#[test]
fn resident_continuation_reports_its_absorbed_prompt() {
    let continuation = SessionContinuation {
        entries: 1,
        input_tokens: 412,
    };
    let json = session_json(&summary(), &continuation);
    assert_eq!(json["continuation"]["available"], true);
    assert_eq!(json["continuation"]["input_tokens"], 412);
}

#[test]
fn a_session_with_no_patches_renders_an_empty_id_list() {
    let mut summary = summary();
    summary.patch_names.clear();
    let json = session_json(&summary, &SessionContinuation::default());
    assert_eq!(json["patches"]["active"], 0);
    assert_eq!(
        json["patches"]["ids"],
        serde_json::Value::Array(Vec::new()),
        "the field is always present, never null"
    );
}

#[test]
fn the_projection_leaks_no_internals() {
    // The contract: metadata only. No overlay contents, no KV bytes, no
    // cache keys — a regression here is how an observability surface
    // quietly becomes a data-exfiltration surface.
    let json = session_json(
        &summary(),
        &SessionContinuation {
            entries: 1,
            input_tokens: 9,
        },
    );
    let mut keys: Vec<&str> = json
        .as_object()
        .expect("object")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec![
            "continuation",
            "created_at",
            "expires_at",
            "id",
            "last_used_at",
            "model",
            "object",
            "patches",
            "state",
        ]
    );
}
