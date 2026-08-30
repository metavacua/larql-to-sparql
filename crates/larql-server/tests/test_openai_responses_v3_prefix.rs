//! N1 prefix stability under REAL request construction — the matrix.
//!
//! The N1 cache resumes only on **exact token-id prefix identity**; no
//! fuzzy matching, no tokenizer tricks. This suite asks whether normal
//! OpenAI-style request assembly preserves that identity, against a V3
//! container whose tokenizer is *template-stable*: WordLevel with a
//! `WhitespaceSplit` pre-tokenizer and a bijective surface ↔ id map
//! (`[0]`…`[25]` plus dedicated scaffold words `User:` / `Assistant:` /
//! `System:`), so any emitted token re-tokenizes to itself when the
//! conversation is re-rendered. Contrast `test_openai_responses_v3.rs`,
//! whose tokenizer (no pre-tokenizer) makes every full prompt a single
//! [UNK] and therefore pins the *fallback* path.
//!
//! Matrix rows covered here:
//! - plain template, growing conversation → resumes (principal gate)
//! - one-token difference inside the cached prefix → rejected (ids level)
//! - changed system message → no resume (and the consumed hit is the
//!   observable `hits - resumptions` gap)
//! - cross-model chain → never resumes, and does NOT consume the entry
//! - tools present on V3 → refused before the cache is touched
//! - template census: which chat templates preserve a *string* prefix
//!   under conversation growth (token-level stability additionally
//!   needs each family's real tokenizer — out of fixture scope)
//!
//! Not representable here (documented, not silently skipped):
//! truncation / context-window movement — the server has no context
//! mover on this path, so there is no request shape that exercises it.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use larql_server::bootstrap::{load_artifact, LoadVindexOptions, LoadedArtifact};
use larql_server::state::AppState;
use larql_vindex::format::vindex3::fixtures::{
    encode_fixture_container, miniature_glimmer, G_VOCAB,
};

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use tower::ServiceExt;

const NEW_TOKENS: usize = 4;
/// First-turn user content (vocabulary-covered).
const TURN_1: &str = "[3]";
/// Second-turn user content.
const TURN_2: &str = "[5]";

/// WordLevel + WhitespaceSplit with a BIJECTIVE surface ↔ id map over
/// the model's whole vocab: ids 0..=25 are `[i]`, and the top three
/// ids are the Plain template's scaffold words. Bijectivity is the
/// stability property: whatever the model emits decodes to a word that
/// re-tokenizes to the same id inside the re-rendered conversation.
fn template_stable_tokenizer_json() -> String {
    const {
        assert!(G_VOCAB >= 4, "fixture vocab too small for scaffold words");
    }
    let scaffold = ["User:", "Assistant:", "System:"];
    let word_ids = G_VOCAB - scaffold.len();
    let mut entries: Vec<String> = (0..word_ids).map(|i| format!("\"[{i}]\":{i}")).collect();
    for (k, word) in scaffold.iter().enumerate() {
        entries.push(format!("\"{word}\":{}", word_ids + k));
    }
    format!(
        "{{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\
         \"normalizer\":null,\"pre_tokenizer\":{{\"type\":\"WhitespaceSplit\"}},\
         \"post_processor\":null,\"decoder\":null,\
         \"model\":{{\"type\":\"WordLevel\",\"vocab\":{{{}}},\"unk_token\":\"[0]\"}}}}",
        entries.join(",")
    )
}

/// One V3 container under a KNOWN directory name (the model id derives
/// from the basename), with the template-stable tokenizer.
fn v3_container_named(root: &Path, name: &str) -> PathBuf {
    let checkpoint = root.join(format!("{name}-checkpoint"));
    let container = root.join(name);
    std::fs::create_dir_all(&checkpoint).unwrap();
    std::fs::create_dir_all(&container).unwrap();
    encode_fixture_container(miniature_glimmer, &checkpoint, &container, name);
    std::fs::write(
        container.join("tokenizer.json"),
        template_stable_tokenizer_json(),
    )
    .unwrap();
    container
}

fn v3_state(containers: &[&Path], kv_entries: usize) -> Arc<AppState> {
    let v3_models = containers
        .iter()
        .map(|c| {
            let artifact =
                load_artifact(&c.to_string_lossy(), LoadVindexOptions::default()).unwrap();
            match artifact {
                LoadedArtifact::V3(m) => Arc::new(*m),
                LoadedArtifact::V2(_) => panic!("a VINDEX3 container must bind as V3"),
            }
        })
        .collect();
    Arc::new(AppState {
        model_set: std::sync::RwLock::new(larql_server::state::ModelSet {
            models: Vec::new(),
            v3_models,
        }),
        router_topology: larql_server::state::RouterTopology::SingleModel,
        lifecycle: std::sync::Mutex::new(larql_server::state::LifecycleState::Idle),
        started_at: std::time::Instant::now(),
        requests_served: std::sync::atomic::AtomicU64::new(0),
        api_key: None,
        sessions: larql_server::session::SessionManager::new(3600),
        describe_cache: larql_server::cache::DescribeCache::new(0),
        infer_timeout: std::time::Duration::from_secs(60),
        responses: larql_server::response_store::ResponseStore::new(),
        v3_kv: larql_server::response_kv::ResponseKvCache::new(
            kv_entries,
            larql_server::response_kv::DEFAULT_TTL_SECS,
        ),
        runtime: Arc::new(larql_server::runtime_stats::RuntimeRecorder::new()),
    })
}

async fn post_responses(app: &axum::Router, body: serde_json::Value) -> axum::http::Response<Body> {
    app.clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/responses")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap()
}

async fn drain(resp: axum::http::Response<Body>) -> (StatusCode, serde_json::Value) {
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| serde_json::json!({"raw": String::from_utf8_lossy(&bytes)}));
    (status, json)
}

fn output_text(envelope: &serde_json::Value) -> String {
    envelope["output"][0]["content"][0]["text"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

fn cached_tokens(envelope: &serde_json::Value) -> u64 {
    envelope["usage"]["input_tokens_details"]["cached_tokens"]
        .as_u64()
        .expect("usage carries cached_tokens")
}

/// Drive turn 1 (store) then a chained turn 2 with `extra` merged into
/// the second request; returns the two envelopes.
async fn chain(
    app: &axum::Router,
    model: Option<&str>,
    extra: serde_json::Value,
) -> (serde_json::Value, serde_json::Value) {
    let mut first_req = serde_json::json!({"input": TURN_1, "max_output_tokens": NEW_TOKENS});
    if let Some(m) = model {
        first_req["model"] = serde_json::json!(m);
    }
    let (status, first) = drain(post_responses(app, first_req).await).await;
    assert_eq!(status, StatusCode::OK, "{first}");

    let mut second_req = serde_json::json!({
        "input": TURN_2,
        "previous_response_id": first["id"],
        "max_output_tokens": NEW_TOKENS,
    });
    if let Some(obj) = extra.as_object() {
        for (k, v) in obj {
            second_req[k] = v.clone();
        }
    }
    let (status, second) = drain(post_responses(app, second_req).await).await;
    assert_eq!(status, StatusCode::OK, "{second}");
    (first, second)
}

// ── the principal gate ───────────────────────────────────────────────────────

/// cold A → cache continuation → B extends A → resumption ENGAGES
/// (`cached_tokens > 0`, resumptions counter advances) → output equals
/// the cold (cache-disabled) run of the identical chain, byte for byte.
#[tokio::test]
async fn growing_plain_conversation_resumes_and_matches_cold_output() {
    let root = tempfile::tempdir().unwrap();
    let container = v3_container_named(root.path(), "prefix-m");

    let warm_state = v3_state(&[&container], 4);
    let warm = larql_server::routes::single_model_router(Arc::clone(&warm_state));
    let (_, warm_second) = chain(&warm, None, serde_json::json!({})).await;

    let cold_state = v3_state(&[&container], 0);
    let cold = larql_server::routes::single_model_router(Arc::clone(&cold_state));
    let (_, cold_second) = chain(&cold, None, serde_json::json!({})).await;

    assert!(
        cached_tokens(&warm_second) > 0,
        "the growing conversation must preserve an exact token-id prefix \
         on the template-stable fixture: {warm_second}"
    );
    assert_eq!(cached_tokens(&cold_second), 0);
    assert_eq!(
        output_text(&warm_second),
        output_text(&cold_second),
        "resumed output must equal cold output"
    );
    assert_eq!(warm_state.v3_kv.hits(), 1);
    assert_eq!(warm_state.v3_kv.resumptions(), 1, "hit AND engaged");
    assert_eq!(
        warm_state.v3_kv.reused_tokens_total(),
        cached_tokens(&warm_second)
    );
}

/// The negative twin at the seam where the contract lives: one token
/// changed INSIDE the cached prefix → resumption rejected, output
/// identical to a fresh prefill.
#[test]
fn one_token_difference_inside_the_prefix_rejects_resumption() {
    let root = tempfile::tempdir().unwrap();
    let container = v3_container_named(root.path(), "prefix-ids");
    let artifact =
        load_artifact(&container.to_string_lossy(), LoadVindexOptions::default()).unwrap();
    let model = match artifact {
        LoadedArtifact::V3(m) => *m,
        LoadedArtifact::V2(_) => panic!("must bind as V3"),
    };
    let sampling = larql_inference::SamplingConfig::default();
    let eos = larql_inference::EosConfig::default();

    let (_, handoff) = larql_server::vindex3::generate_v3_resumable(
        &model,
        &[1, 2, 3],
        None,
        NEW_TOKENS,
        sampling,
        &eos,
        |_, _| {},
    )
    .expect("turn 1");

    // Extend the absorbed ids but flip one token INSIDE the prefix.
    let mut corrupted = handoff.absorbed_ids.clone();
    corrupted[1] = if corrupted[1] == 4 { 5 } else { 4 };
    corrupted.extend_from_slice(&[6, 7]);

    let (resumed, _) = larql_server::vindex3::generate_v3_resumable(
        &model,
        &corrupted,
        Some(handoff),
        NEW_TOKENS,
        sampling,
        &eos,
        |_, _| {},
    )
    .expect("corrupted-prefix turn");
    let (fresh, _) = larql_server::vindex3::generate_v3_resumable(
        &model,
        &corrupted,
        None,
        NEW_TOKENS,
        sampling,
        &eos,
        |_, _| {},
    )
    .expect("fresh turn");

    assert_eq!(
        resumed.reused_prompt_tokens, 0,
        "a one-token difference inside the prefix must reject resumption"
    );
    assert_eq!(resumed.ids, fresh.ids, "rejected resume must equal fresh");
}

// ── negative rows over HTTP ──────────────────────────────────────────────────

/// A changed system message prepends tokens BEFORE the cached prefix,
/// breaking identity at position 0: the resident state is found (hit,
/// consumed — take-once) but resumption must not engage, and the
/// output must equal the cold run of the same altered chain. The
/// surviving `hits - resumptions` gap is exactly what `/v1/stats` is
/// meant to expose.
#[tokio::test]
async fn changed_system_message_defeats_the_prefix_but_not_correctness() {
    let root = tempfile::tempdir().unwrap();
    let container = v3_container_named(root.path(), "prefix-sys");
    let extra = serde_json::json!({"instructions": "[1]"});

    let warm_state = v3_state(&[&container], 4);
    let warm = larql_server::routes::single_model_router(Arc::clone(&warm_state));
    let (_, warm_second) = chain(&warm, None, extra.clone()).await;

    let cold_state = v3_state(&[&container], 0);
    let cold = larql_server::routes::single_model_router(Arc::clone(&cold_state));
    let (_, cold_second) = chain(&cold, None, extra).await;

    assert_eq!(
        cached_tokens(&warm_second),
        0,
        "a changed system message must not resume: {warm_second}"
    );
    assert_eq!(output_text(&warm_second), output_text(&cold_second));
    assert_eq!(warm_state.v3_kv.hits(), 1, "resident state was found");
    assert_eq!(
        warm_state.v3_kv.resumptions(),
        0,
        "…but resumption must not engage — this is the stability gap"
    );
}

/// A chain naming a DIFFERENT model must never resume from another
/// binding's KV — even with an identical tokenizer, where the token-id
/// prefix would happen to match — and must not consume the entry: the
/// rightful chain afterwards still resumes.
#[tokio::test]
async fn cross_model_chain_never_resumes_and_preserves_the_entry() {
    let root = tempfile::tempdir().unwrap();
    let container_a = v3_container_named(root.path(), "prefix-a");
    let container_b = v3_container_named(root.path(), "prefix-b");

    let state = v3_state(&[&container_a, &container_b], 4);
    let app = larql_server::routes::single_model_router(Arc::clone(&state));

    let (first, _) = chain(
        &app,
        Some("prefix-a"),
        serde_json::json!({"model": "prefix-b"}),
    )
    .await;

    // The cross-model second turn above: resident state exists under
    // model a's id, so the lookup is a NON-consuming miss.
    assert_eq!(state.v3_kv.hits(), 0);
    assert_eq!(state.v3_kv.misses(), 1);
    assert_eq!(state.v3_kv.resumptions(), 0);

    // The rightful chain still finds — and engages — the entry.
    let (status, third) = drain(
        post_responses(
            &app,
            serde_json::json!({
                "model": "prefix-a",
                "input": TURN_2,
                "previous_response_id": first["id"],
                "max_output_tokens": NEW_TOKENS,
            }),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{third}");
    assert!(
        cached_tokens(&third) > 0,
        "the rightful chain must still resume: {third}"
    );
    assert_eq!(state.v3_kv.hits(), 1);
    assert_eq!(state.v3_kv.resumptions(), 1);
}

/// A tools request cannot consume the continuation state: resume is
/// gated off for tools BEFORE any cache lookup, and on this
/// JSON-incapable vocabulary the constrained generation then fails
/// closed — either way the entry must survive for the plain chain.
#[tokio::test]
async fn rejected_tools_chain_does_not_consume_the_entry() {
    let root = tempfile::tempdir().unwrap();
    let container = v3_container_named(root.path(), "prefix-tools");
    let state = v3_state(&[&container], 4);
    let app = larql_server::routes::single_model_router(Arc::clone(&state));

    let (status, first) = drain(
        post_responses(
            &app,
            serde_json::json!({"input": TURN_1, "max_output_tokens": NEW_TOKENS}),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{first}");
    assert_eq!(state.v3_kv.len(), 1);

    let (status, rejected) = drain(
        post_responses(
            &app,
            serde_json::json!({
                "input": TURN_2,
                "previous_response_id": first["id"],
                "max_output_tokens": NEW_TOKENS,
                "tools": [{"type": "function", "name": "f",
                           "parameters": {"type": "object"}}],
            }),
        )
        .await,
    )
    .await;
    assert!(status.is_server_error(), "{rejected}");
    assert_eq!(state.v3_kv.len(), 1, "the entry must survive the failure");
    assert_eq!(state.v3_kv.hits() + state.v3_kv.misses(), 0);

    let (status, second) = drain(
        post_responses(
            &app,
            serde_json::json!({
                "input": TURN_2,
                "previous_response_id": first["id"],
                "max_output_tokens": NEW_TOKENS,
            }),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{second}");
    assert!(cached_tokens(&second) > 0, "{second}");
}

// ── template census ──────────────────────────────────────────────────────────

/// Census, not a performance gate: for each chat template, does the
/// rendered turn-1 prompt survive as an exact STRING prefix of the
/// re-rendered grown conversation? String stability is necessary (not
/// sufficient) for token-id stability — a template that rewrites its
/// opening can never preserve an id prefix under any tokenizer. The
/// classification is pinned so silent template drift shows up here.
#[test]
fn template_prefix_survival_census() {
    use larql_inference::prompt::ChatTemplate;
    let templates = [
        ChatTemplate::Plain,
        ChatTemplate::Gemma,
        ChatTemplate::Llama,
        ChatTemplate::ChatML,
        ChatTemplate::Mistral,
    ];

    let mut census = Vec::new();
    for template in templates {
        let turn1 = template.render_messages([("user", TURN_1)]);
        let grown = template.render_messages([
            ("user", TURN_1),
            ("assistant", "[4] [7]"),
            ("user", TURN_2),
        ]);
        let stable = grown.starts_with(&turn1);
        census.push((format!("{template:?}"), stable));
    }
    println!("template prefix survival census (string level): {census:?}");

    // Pinned classification — every current template appends turns
    // after the assistant-open scaffold, so the string prefix survives
    // conversation growth across the board. A `false` here means a
    // template started rewriting its opening: N1 can never resume on
    // it, whatever the tokenizer does.
    for (name, stable) in &census {
        assert!(*stable, "template {name} no longer preserves its prefix");
    }
}
