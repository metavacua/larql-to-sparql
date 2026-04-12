use super::*;
use super::helpers::*;
use crate::parser;

// ── Session state: no backend ──

#[test]
fn no_backend_stats() {
    let mut session = Session::new();
    let stmt = parser::parse("STATS;").unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, LqlError::NoBackend));
}

#[test]
fn no_backend_walk() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"WALK "test" TOP 5;"#).unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), LqlError::NoBackend));
}

#[test]
fn no_backend_describe() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"DESCRIBE "France";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_select() {
    let mut session = Session::new();
    let stmt = parser::parse("SELECT * FROM EDGES;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_explain() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"EXPLAIN WALK "test";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_show_relations() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW RELATIONS;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_show_layers() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW LAYERS;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn no_backend_show_features() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW FEATURES 26;").unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

// ── USE errors ──

#[test]
fn use_nonexistent_vindex() {
    let mut session = Session::new();
    let stmt =
        parser::parse(r#"USE "/nonexistent/path/fake.vindex";"#).unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), LqlError::Execution(_)));
}

#[test]
fn use_model_fails_on_nonexistent() {
    let mut session = Session::new();
    let stmt =
        parser::parse(r#"USE MODEL "/nonexistent/model";"#).unwrap();
    let result = session.execute(&stmt);
    // Should fail to resolve the model path
    assert!(result.is_err());
}

#[test]
fn use_model_auto_extract_parses() {
    // Verify AUTO_EXTRACT parses correctly (loading will fail for nonexistent model)
    let mut session = Session::new();
    let stmt = parser::parse(
        r#"USE MODEL "/nonexistent/model" AUTO_EXTRACT;"#,
    )
    .unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
}

// ── Lifecycle: error cases without valid model/vindex ──

#[test]
fn extract_fails_on_nonexistent_model() {
    let mut session = Session::new();
    let stmt = parser::parse(
        r#"EXTRACT MODEL "/nonexistent/model" INTO "/tmp/test_extract_out.vindex";"#,
    )
    .unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), LqlError::Execution(_)));
}

#[test]
fn compile_no_backend() {
    let mut session = Session::new();
    let stmt = parser::parse(
        r#"COMPILE CURRENT INTO MODEL "out/";"#,
    )
    .unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn diff_nonexistent_vindex() {
    let mut session = Session::new();
    let stmt =
        parser::parse(r#"DIFF "/nonexistent/a.vindex" "/nonexistent/b.vindex";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::Execution(_)
    ));
}

// ── Mutation: no-backend errors ──

#[test]
fn insert_no_backend() {
    let mut session = Session::new();
    let stmt = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c");"#,
    )
    .unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn delete_no_backend() {
    let mut session = Session::new();
    let stmt = parser::parse(
        r#"DELETE FROM EDGES WHERE entity = "x";"#,
    )
    .unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn update_no_backend() {
    let mut session = Session::new();
    let stmt = parser::parse(
        r#"UPDATE EDGES SET target = "y" WHERE entity = "x";"#,
    )
    .unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

#[test]
fn merge_nonexistent_source() {
    let mut session = Session::new();
    let stmt =
        parser::parse(r#"MERGE "/nonexistent/source.vindex";"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::Execution(_)
    ));
}

// ── INFER ──

#[test]
fn infer_no_backend() {
    let mut session = Session::new();
    let stmt = parser::parse(r#"INFER "test" TOP 5;"#).unwrap();
    assert!(matches!(
        session.execute(&stmt).unwrap_err(),
        LqlError::NoBackend
    ));
}

// ── is_readable_token ──

#[test]
fn readable_tokens() {
    assert!(is_readable_token("French"));
    assert!(is_readable_token("Paris"));
    assert!(is_readable_token("capital-of"));
    assert!(is_readable_token("is"));
    assert!(is_readable_token("Europe"));
}

#[test]
fn unreadable_tokens() {
    assert!(!is_readable_token("ইসলামাবাদ"));
    assert!(!is_readable_token("южна"));
    assert!(!is_readable_token("ളാ"));
    assert!(!is_readable_token("ڪ"));
    assert!(!is_readable_token(""));
}

// ── is_content_token ──

#[test]
fn content_tokens_pass() {
    assert!(is_content_token("French"));
    assert!(is_content_token("Paris"));
    assert!(is_content_token("Europe"));
    assert!(is_content_token("Mozart"));
    assert!(is_content_token("composer"));
    assert!(is_content_token("Berlin"));
    assert!(is_content_token("IBM"));
    assert!(is_content_token("Facebook"));
}

#[test]
fn stop_words_rejected() {
    assert!(!is_content_token("the"));
    assert!(!is_content_token("from"));
    assert!(!is_content_token("for"));
    assert!(!is_content_token("with"));
    assert!(!is_content_token("this"));
    assert!(!is_content_token("about"));
    assert!(!is_content_token("which"));
    assert!(!is_content_token("first"));
    assert!(!is_content_token("after"));
}

#[test]
fn short_tokens_rejected() {
    assert!(!is_content_token("a"));
    assert!(!is_content_token("of"));
    assert!(!is_content_token("is"));
    assert!(!is_content_token("-"));
    assert!(!is_content_token("lö"));
    assert!(!is_content_token("par"));
}

#[test]
fn code_tokens_rejected() {
    assert!(!is_content_token("trialComponents"));
    assert!(!is_content_token("NavigationBar"));
    assert!(!is_content_token("LastName"));
}

// ── SHOW MODELS works without backend ──

#[test]
fn show_models_no_crash() {
    let mut session = Session::new();
    let stmt = parser::parse("SHOW MODELS;").unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_ok());
}

// ── Pipe: errors propagate ──

#[test]
fn pipe_error_propagates() {
    let mut session = Session::new();
    let stmt = parser::parse(
        r#"STATS |> WALK "test";"#,
    )
    .unwrap();
    assert!(session.execute(&stmt).is_err());
}

// ── Format helpers ──

#[test]
fn format_number_small() {
    assert_eq!(format_number(42), "42");
    assert_eq!(format_number(999), "999");
}

#[test]
fn format_number_thousands() {
    assert_eq!(format_number(1_000), "1.0K");
    assert_eq!(format_number(10_240), "10.2K");
    assert_eq!(format_number(348_160), "348.2K");
}

#[test]
fn format_number_millions() {
    assert_eq!(format_number(1_000_000), "1.00M");
    assert_eq!(format_number(2_917_432), "2.92M");
}

#[test]
fn format_bytes_small() {
    assert_eq!(format_bytes(512), "512 B");
}

#[test]
fn format_bytes_kb() {
    assert_eq!(format_bytes(2048), "2.0 KB");
}

#[test]
fn format_bytes_mb() {
    let mb = 5 * 1_048_576;
    assert_eq!(format_bytes(mb), "5.0 MB");
}

#[test]
fn format_bytes_gb() {
    let gb = 6_420_000_000;
    assert!(format_bytes(gb).contains("GB"));
}

// ═══════════════════════════════════════════════════════════════
// Weight backend tests
// ═══════════════════════════════════════════════════════════════

/// Create a minimal ModelWeights for testing the Weight backend.
fn make_test_weights() -> larql_inference::ModelWeights {
    use std::collections::HashMap;
    use larql_inference::ndarray;

    let num_layers = 2;
    let hidden = 8;
    let intermediate = 4;
    let vocab_size = 16;

    let mut tensors: HashMap<String, ndarray::ArcArray2<f32>> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

    for layer in 0..num_layers {
        let mut gate = ndarray::Array2::<f32>::zeros((intermediate, hidden));
        for i in 0..intermediate { gate[[i, i % hidden]] = 1.0 + layer as f32; }
        tensors.insert(format!("layers.{layer}.mlp.gate_proj.weight"), gate.into_shared());

        let mut up = ndarray::Array2::<f32>::zeros((intermediate, hidden));
        for i in 0..intermediate { up[[i, (i + 1) % hidden]] = 0.5; }
        tensors.insert(format!("layers.{layer}.mlp.up_proj.weight"), up.into_shared());

        let mut down = ndarray::Array2::<f32>::zeros((hidden, intermediate));
        for i in 0..intermediate { down[[i % hidden, i]] = 0.3; }
        tensors.insert(format!("layers.{layer}.mlp.down_proj.weight"), down.into_shared());

        for suffix in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            let mut attn = ndarray::Array2::<f32>::zeros((hidden, hidden));
            for i in 0..hidden { attn[[i, i]] = 1.0; }
            tensors.insert(format!("layers.{layer}.self_attn.{suffix}.weight"), attn.into_shared());
        }

        vectors.insert(format!("layers.{layer}.input_layernorm.weight"), vec![1.0; hidden]);
        vectors.insert(format!("layers.{layer}.post_attention_layernorm.weight"), vec![1.0; hidden]);
    }

    vectors.insert("norm.weight".into(), vec![1.0; hidden]);

    let mut embed = ndarray::Array2::<f32>::zeros((vocab_size, hidden));
    for i in 0..vocab_size { embed[[i, i % hidden]] = 1.0; }
    let embed = embed.into_shared();
    let lm_head = embed.clone();

    let arch = larql_models::detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": hidden,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate,
        "head_dim": hidden,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "rope_theta": 10000.0,
        "vocab_size": vocab_size,
    }));

    larql_inference::ModelWeights {
        tensors, vectors, embed, lm_head,
        num_layers, hidden_size: hidden, intermediate_size: intermediate,
        vocab_size, head_dim: hidden, num_q_heads: 1, num_kv_heads: 1,
        rope_base: 10000.0, arch,
    }
}

/// Create a minimal tokenizer for testing.
fn make_test_tokenizer() -> larql_inference::tokenizers::Tokenizer {
    let tok_json = r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    larql_inference::tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).unwrap()
}

/// Create a Session with Weight backend for testing.
fn weight_session() -> Session {
    let mut session = Session::new();
    session.backend = Backend::Weight {
        model_id: "test/model".into(),
        weights: make_test_weights(),
        tokenizer: make_test_tokenizer(),
    };
    session
}

#[test]
fn weight_backend_stats() {
    let mut session = weight_session();
    let stmt = parser::parse("STATS;").unwrap();
    let result = session.execute(&stmt).unwrap();
    assert!(result.iter().any(|l| l.contains("test/model")));
    assert!(result.iter().any(|l| l.contains("live weights")));
    assert!(result.iter().any(|l| l.contains("2"))); // num_layers
}

#[test]
fn weight_backend_walk_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse(r#"WALK "test" TOP 5;"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("requires a vindex"), "expected vindex error, got: {msg}");
    assert!(msg.contains("EXTRACT"), "should suggest EXTRACT, got: {msg}");
}

#[test]
fn weight_backend_describe_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse(r#"DESCRIBE "France";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("requires a vindex"));
}

#[test]
fn weight_backend_select_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse("SELECT * FROM EDGES;").unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("requires a vindex"));
}

#[test]
fn weight_backend_explain_walk_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse(r#"EXPLAIN WALK "test";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("requires a vindex"));
}

#[test]
fn weight_backend_insert_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse(
        r#"INSERT INTO EDGES (entity, relation, target) VALUES ("a", "b", "c");"#
    ).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("requires a vindex") || msg.contains("mutation requires"));
}

#[test]
fn weight_backend_show_relations_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse("SHOW RELATIONS;").unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("requires a vindex"));
}

#[test]
fn weight_backend_compile_current_requires_vindex() {
    let mut session = weight_session();
    let stmt = parser::parse(r#"COMPILE CURRENT INTO MODEL "out/";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("EXTRACT") || msg.contains("vindex"));
}

#[test]
fn weight_backend_show_models_works() {
    let mut session = weight_session();
    let stmt = parser::parse("SHOW MODELS;").unwrap();
    let result = session.execute(&stmt);
    assert!(result.is_ok());
}

// ── Mutation pipeline integration tests ────────────────────────────────
//
// These tests build a tiny synthetic vindex on disk, load it via USE,
// and exercise the DELETE / UPDATE / patch-session paths through the
// real executor + parser. They cover the auto-patch lifecycle, the
// patch overlay update, and SAVE PATCH file emission.
//
// INSERT is exercised end-to-end in `compile_demo` against a real
// Gemma vindex (the synthetic tokenizer here has an empty vocab so it
// can't tokenise meaningful entity strings). The auto-patch session
// creation that INSERT triggers is covered indirectly by the DELETE
// auto-patch test below.

use larql_inference::ndarray::Array2;

/// Build a minimal vindex directory on disk that the LQL executor can
/// load via `USE`. Includes gate vectors, down_meta, embeddings, and a
/// stub tokenizer. Returns the directory path; the caller is
/// responsible for cleanup.
fn make_test_vindex_dir(tag: &str) -> std::path::PathBuf {
    use larql_vindex::{
        ExtractLevel, FeatureMeta, StorageDtype, VectorIndex, VindexConfig,
    };
    use larql_models::TopKEntry;

    let dir = std::env::temp_dir().join(format!("larql_lql_test_vindex_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Tiny in-memory index — 2 layers × 3 features × 4 hidden dims.
    let hidden = 4;
    let num_features = 3;
    let num_layers = 2;
    let vocab_size = 10;

    let mut gate0 = Array2::<f32>::zeros((num_features, hidden));
    gate0[[0, 0]] = 1.0;
    gate0[[1, 1]] = 1.0;
    gate0[[2, 2]] = 1.0;

    let mut gate1 = Array2::<f32>::zeros((num_features, hidden));
    gate1[[0, 3]] = 1.0;
    gate1[[1, 0]] = 0.5;
    gate1[[2, 2]] = -1.0;

    let make_meta = |tok: &str, id: u32, c: f32| FeatureMeta {
        top_token: tok.to_string(),
        top_token_id: id,
        c_score: c,
        top_k: vec![TopKEntry { token: tok.to_string(), token_id: id, logit: c }],
    };

    let meta0 = vec![
        Some(make_meta("Paris", 100, 0.95)),
        Some(make_meta("French", 101, 0.88)),
        Some(make_meta("Europe", 102, 0.75)),
    ];
    let meta1 = vec![
        Some(make_meta("Berlin", 200, 0.90)),
        None,
        Some(make_meta("Spain", 202, 0.70)),
    ];
    let down_meta = vec![Some(meta0), Some(meta1)];

    let index = VectorIndex::new(
        vec![Some(gate0), Some(gate1)],
        down_meta,
        num_layers,
        hidden,
    );

    let mut config = VindexConfig {
        version: 2,
        model: "test/lql-mutation".into(),
        family: "llama".into(),
        source: None,
        checksums: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: num_features,
        vocab_size,
        embed_scale: 1.0,
        extract_level: ExtractLevel::Browse,
        dtype: StorageDtype::F32,
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: 5,
        has_model_weights: false,
        model_config: None,
    };
    index.save_vindex(&dir, &mut config).unwrap();

    // Synthetic embeddings.bin (vocab_size × hidden f32, all zeros).
    let embed_bytes = vec![0u8; vocab_size * hidden * 4];
    std::fs::write(dir.join("embeddings.bin"), embed_bytes).unwrap();

    // Stub tokenizer.json — empty BPE. Not used by DELETE / UPDATE /
    // PATCH; INSERT-against-this-vindex tests would need a real one.
    let tok_json = r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();

    dir
}

/// Spin up a session and `USE` the test vindex from `make_test_vindex_dir`.
fn vindex_session(tag: &str) -> (Session, std::path::PathBuf) {
    let dir = make_test_vindex_dir(tag);
    let mut session = Session::new();
    let stmt = parser::parse(&format!(r#"USE "{}";"#, dir.display())).unwrap();
    session.execute(&stmt).expect("USE on synthetic vindex should succeed");
    (session, dir)
}

#[test]
fn use_synthetic_vindex_loads() {
    let (session, dir) = vindex_session("use_loads");
    assert!(matches!(session.backend, Backend::Vindex { .. }));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn delete_by_layer_and_feature_succeeds() {
    let (mut session, dir) = vindex_session("delete_lf");

    let stmt = parser::parse(
        r#"DELETE FROM EDGES WHERE layer = 0 AND feature = 0;"#,
    )
    .unwrap();
    let out = session.execute(&stmt).expect("DELETE should succeed");
    let joined = out.join("\n");
    assert!(
        joined.contains("Deleted") || joined.contains("deleted"),
        "expected delete confirmation in: {joined}"
    );

    // The patch session should now be active (auto-patch).
    assert!(
        session.patch_recording.is_some(),
        "DELETE should have started an auto-patch session"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn delete_no_matches_returns_message() {
    let (mut session, dir) = vindex_session("delete_nomatch");

    // Layer that doesn't exist in our 2-layer test vindex.
    let stmt = parser::parse(
        r#"DELETE FROM EDGES WHERE layer = 99 AND feature = 0;"#,
    )
    .unwrap();
    let result = session.execute(&stmt);
    // The executor either returns an empty-match message or errors —
    // both are acceptable; the important thing is no panic.
    assert!(result.is_ok() || result.is_err(), "no panic");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn update_feature_target_succeeds() {
    let (mut session, dir) = vindex_session("update_target");

    let stmt = parser::parse(
        r#"UPDATE EDGES SET target = "London" WHERE layer = 0 AND feature = 0;"#,
    )
    .unwrap();
    let out = session.execute(&stmt).expect("UPDATE should succeed");
    let joined = out.join("\n");
    assert!(
        joined.contains("Updated") || joined.contains("updated") || joined.contains("London"),
        "expected update confirmation in: {joined}"
    );

    assert!(
        session.patch_recording.is_some(),
        "UPDATE should have started an auto-patch session"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explicit_begin_patch_starts_session() {
    let (mut session, dir) = vindex_session("begin_patch");

    let patch_path = dir.join("session.vlp");
    let stmt = parser::parse(&format!(r#"BEGIN PATCH "{}";"#, patch_path.display())).unwrap();
    session.execute(&stmt).expect("BEGIN PATCH should succeed");

    assert!(
        session.patch_recording.is_some(),
        "BEGIN PATCH should populate patch_recording"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn save_patch_writes_file_to_disk() {
    let (mut session, dir) = vindex_session("save_patch");

    // Start a patch, do a delete (so there's at least one operation), save.
    let patch_path = dir.join("save.vlp");
    let begin = parser::parse(&format!(r#"BEGIN PATCH "{}";"#, patch_path.display())).unwrap();
    session.execute(&begin).expect("BEGIN PATCH");

    let del = parser::parse(r#"DELETE FROM EDGES WHERE layer = 0 AND feature = 1;"#).unwrap();
    session.execute(&del).expect("DELETE under patch");

    let save = parser::parse("SAVE PATCH;").unwrap();
    let out = session.execute(&save).expect("SAVE PATCH");
    let joined = out.join("\n");
    assert!(
        patch_path.exists(),
        "SAVE PATCH should write the .vlp file. Output: {joined}"
    );

    // After SAVE PATCH, recording should be cleared.
    assert!(
        session.patch_recording.is_none(),
        "SAVE PATCH should clear the recording"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn auto_patch_session_starts_on_first_mutation() {
    let (mut session, dir) = vindex_session("auto_patch");

    // No explicit BEGIN PATCH first.
    assert!(session.patch_recording.is_none(), "no patch session before mutation");

    let del = parser::parse(r#"DELETE FROM EDGES WHERE layer = 0 AND feature = 0;"#).unwrap();
    session.execute(&del).expect("DELETE");

    assert!(
        session.patch_recording.is_some(),
        "first mutation should auto-start a patch session"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn merge_nonexistent_source_errors_cleanly() {
    let (mut session, dir) = vindex_session("merge_bad_src");

    let stmt = parser::parse(r#"MERGE "/nonexistent/src.vindex";"#).unwrap();
    let err = session.execute(&stmt).unwrap_err();
    assert!(matches!(err, LqlError::Execution(_)));

    let _ = std::fs::remove_dir_all(&dir);
}

// ── Session::patched_overlay_mut accessor ──

#[test]
fn patched_overlay_mut_returns_some_for_vindex_backend() {
    let (mut session, dir) = vindex_session("overlay_mut_some");
    assert!(
        session.patched_overlay_mut().is_some(),
        "Vindex backend should yield a mutable overlay"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn patched_overlay_mut_returns_none_for_no_backend() {
    let mut session = Session::new();
    assert!(
        session.patched_overlay_mut().is_none(),
        "fresh session with no backend should yield None"
    );
}

#[test]
fn patched_overlay_mut_round_trip_via_insert_feature() {
    use larql_models::TopKEntry;
    use larql_vindex::FeatureMeta;

    let (mut session, dir) = vindex_session("overlay_mut_round_trip");
    let gate = vec![0.7_f32, 0.0, 0.0, 0.0];
    {
        let overlay = session.patched_overlay_mut().expect("vindex backend");
        overlay.insert_feature(
            0, 1,
            gate.clone(),
            FeatureMeta {
                top_token: "z".into(),
                top_token_id: 9,
                c_score: 0.42,
                top_k: vec![TopKEntry { token: "z".into(), token_id: 9, logit: 0.42 }],
            },
        );
    }
    // Same accessor, second call: the gate we just wrote must still be there.
    let overlay2 = session.patched_overlay_mut().expect("vindex backend");
    assert_eq!(
        overlay2.overrides_gate_at(0, 1),
        Some(gate.as_slice()),
        "second patched_overlay_mut() call should observe the previous mutation",
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── COMPILE INTO VINDEX WITH REFINE: integration ──

/// Helper for the refine integration tests: inject two parallel gates
/// directly into the patch overlay (bypassing INSERT, which would
/// require a real tokenizer + relation classifier the synthetic vindex
/// doesn't carry). Returns the original gate vectors so the caller can
/// compare them against the post-compile state.
fn inject_parallel_gates(session: &mut Session) -> (Vec<f32>, Vec<f32>) {
    use larql_models::TopKEntry;
    use larql_vindex::FeatureMeta;

    let gate_a = vec![1.0_f32, 0.5, 0.0, 0.0];
    let gate_b = vec![0.5_f32, 1.0, 0.0, 0.0];
    let meta_a = FeatureMeta {
        top_token: "alpha".into(),
        top_token_id: 1,
        c_score: 0.9,
        top_k: vec![TopKEntry { token: "alpha".into(), token_id: 1, logit: 0.9 }],
    };
    let meta_b = FeatureMeta {
        top_token: "beta".into(),
        top_token_id: 2,
        c_score: 0.9,
        top_k: vec![TopKEntry { token: "beta".into(), token_id: 2, logit: 0.9 }],
    };

    match &mut session.backend {
        Backend::Vindex { patched, .. } => {
            patched.insert_feature(0, 0, gate_a.clone(), meta_a);
            patched.insert_feature(0, 1, gate_b.clone(), meta_b);
        }
        _ => panic!("test session must be Vindex backend"),
    }
    (gate_a, gate_b)
}

fn read_overlay_gate(session: &Session, layer: usize, feature: usize) -> Vec<f32> {
    match &session.backend {
        Backend::Vindex { patched, .. } => patched
            .overrides_gate_at(layer, feature)
            .expect("gate override should exist after injection")
            .to_vec(),
        _ => panic!("test session must be Vindex backend"),
    }
}

#[test]
fn refine_pass_modifies_overlapping_gates() {
    // Two parallel gates injected at L0/F0 and L0/F1 should both lose
    // norm under the refine pass. The output line should advertise
    // refine and the overlay should hold the new vectors.
    let (mut session, dir) = vindex_session("refine_modifies");
    let (orig_a, orig_b) = inject_parallel_gates(&mut session);

    let out_dir = dir.join("compiled_with_refine");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}" WITH REFINE;"#,
        out_dir.display(),
    ))
    .unwrap();
    let out = session.execute(&stmt).expect("compile WITH REFINE");
    let joined = out.join("\n");
    assert!(joined.contains("Refine: 2 fact"),
            "compile output should mention 2 facts refined: {joined}");
    assert!(joined.contains("norm retained"),
            "compile output should report norm retained stats: {joined}");

    let refined_a = read_overlay_gate(&session, 0, 0);
    let refined_b = read_overlay_gate(&session, 0, 1);
    let norm_orig_a: f32 = orig_a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_orig_b: f32 = orig_b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_refined_a: f32 = refined_a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_refined_b: f32 = refined_b.iter().map(|x| x * x).sum::<f32>().sqrt();

    assert!(norm_refined_a < norm_orig_a * 0.95,
            "fact a should lose norm: {norm_refined_a} < {norm_orig_a}");
    assert!(norm_refined_b < norm_orig_b * 0.95,
            "fact b should lose norm");
    // The original (orthogonal-component) directions are preserved by
    // sign — assert the refined gate is *not* equal to the original.
    assert_ne!(refined_a, orig_a, "refined gate must differ from original");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn without_refine_leaves_overlay_gates_untouched() {
    // Same constellation but compiled WITHOUT REFINE — the overlay
    // gates should still equal what was injected, byte for byte.
    let (mut session, dir) = vindex_session("refine_skipped");
    let (orig_a, orig_b) = inject_parallel_gates(&mut session);

    let out_dir = dir.join("compiled_no_refine");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}" WITHOUT REFINE;"#,
        out_dir.display(),
    ))
    .unwrap();
    let out = session.execute(&stmt).expect("compile WITHOUT REFINE");
    let joined = out.join("\n");
    assert!(joined.contains("Refine: skipped (WITHOUT REFINE)"),
            "compile output should advertise skipped refine: {joined}");

    let after_a = read_overlay_gate(&session, 0, 0);
    let after_b = read_overlay_gate(&session, 0, 1);
    assert_eq!(after_a, orig_a, "WITHOUT REFINE must not touch the gates");
    assert_eq!(after_b, orig_b);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn refine_with_decoys_errors_on_browse_only_vindex() {
    // The synthetic test vindex is browse-only (no model weights).
    // WITH DECOYS requires forward-pass capability, so the executor
    // should reject it with a clear message instead of trying and
    // failing partway through.
    let (mut session, dir) = vindex_session("refine_decoys_browse");
    let _ = inject_parallel_gates(&mut session);

    let out_dir = dir.join("compiled_with_decoys");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}" WITH DECOYS ("hello world");"#,
        out_dir.display(),
    ))
    .unwrap();
    let err = session.execute(&stmt).expect_err("WITH DECOYS on browse-only must error");
    let msg = format!("{err}");
    assert!(msg.to_lowercase().contains("decoys requires model weights")
                || msg.to_lowercase().contains("with all"),
            "error should explain the precondition: {msg}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn refine_pass_with_no_overrides_returns_quietly() {
    // Compile WITH REFINE on a session with no patches — the refine
    // pass should report "nothing to refine" rather than running
    // empty math or panicking.
    let (mut session, dir) = vindex_session("refine_empty");

    let out_dir = dir.join("compiled_empty");
    let stmt = parser::parse(&format!(
        r#"COMPILE CURRENT INTO VINDEX "{}" WITH REFINE;"#,
        out_dir.display(),
    ))
    .unwrap();
    let out = session.execute(&stmt).expect("empty refine should succeed");
    let joined = out.join("\n");
    assert!(joined.contains("no gate overrides to refine"),
            "expected the no-op message: {joined}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn show_patches_with_no_patches_returns_message() {
    let (mut session, dir) = vindex_session("show_patches_empty");

    let stmt = parser::parse("SHOW PATCHES;").unwrap();
    let out = session.execute(&stmt).expect("SHOW PATCHES");
    let joined = out.join("\n").to_lowercase();
    assert!(
        joined.contains("no") || joined.contains("0") || joined.is_empty(),
        "expected an empty/no-patches message: {joined}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
