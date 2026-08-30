//! Disk-path detection: `require_config_fields` + missing `config.json`.

use crate::detect::config_io::{
    CONFIG_KEY_HIDDEN_SIZE, CONFIG_KEY_INTERMEDIATE_SIZE, CONFIG_KEY_NUM_HIDDEN_LAYERS,
    REQUIRED_CONFIG_FIELDS,
};
use crate::detect::*;

// ── Disk-path tests: require_config_fields + missing config.json ──

fn write_config_json(dir: &std::path::Path, body: &serde_json::Value) {
    std::fs::write(
        dir.join(CONFIG_FILE_NAME),
        serde_json::to_string(body).unwrap(),
    )
    .unwrap();
}

fn expect_detect_err(model_dir: &std::path::Path) -> ModelError {
    // `Box<dyn ModelArchitecture>` isn't Debug, so `Result::expect_err`
    // doesn't apply. Match instead.
    match detect_architecture(model_dir) {
        Ok(_) => panic!("expected detect_architecture to fail"),
        Err(e) => e,
    }
}

#[test]
fn detect_architecture_errors_when_config_json_is_missing() {
    let tmp = tempfile::tempdir().unwrap();
    // No config.json at all — the failure mode reported in issue #22
    // (user pointed extract-index at a directory containing only
    // safetensors + tokenizer.json).
    let err = expect_detect_err(tmp.path());
    match err {
        ModelError::ConfigMissing(p) => {
            assert_eq!(p, tmp.path().join(CONFIG_FILE_NAME));
        }
        other => panic!("expected ConfigMissing, got {other:?}"),
    }
}

#[test]
fn detect_architecture_errors_when_required_fields_are_missing() {
    let tmp = tempfile::tempdir().unwrap();
    // config.json exists but is empty — previously the silent
    // `unwrap_or(2048)` / `unwrap_or(32)` defaults made this look like
    // a 32-layer 2048-hidden model and panicked on broadcast against
    // the real embed shape (issue #22).
    write_config_json(tmp.path(), &serde_json::json!({}));
    let err = expect_detect_err(tmp.path());
    match err {
        ModelError::ConfigFieldsMissing { path, missing } => {
            assert_eq!(path, tmp.path().join(CONFIG_FILE_NAME));
            // Every required field should be reported as missing, in
            // declared order, so the user sees the full set to fix.
            // REQUIRED_CONFIG_FIELDS is a list of alias lists; the
            // validator reports the canonical (first-listed) name from
            // each.
            let expected: Vec<&str> = REQUIRED_CONFIG_FIELDS.iter().map(|a| a[0]).collect();
            assert_eq!(missing, expected);
        }
        other => panic!("expected ConfigFieldsMissing, got {other:?}"),
    }
}

#[test]
fn detect_architecture_reports_only_the_missing_required_fields() {
    let tmp = tempfile::tempdir().unwrap();
    // Two of three required fields present — only the one absent
    // should be reported, so the user can fix one entry at a time.
    write_config_json(
        tmp.path(),
        &serde_json::json!({
            "model_type": "llama",
            CONFIG_KEY_HIDDEN_SIZE: 4096,
            CONFIG_KEY_INTERMEDIATE_SIZE: 11008,
        }),
    );
    let err = expect_detect_err(tmp.path());
    match err {
        ModelError::ConfigFieldsMissing { missing, .. } => {
            assert_eq!(missing, vec![CONFIG_KEY_NUM_HIDDEN_LAYERS]);
        }
        other => panic!("expected ConfigFieldsMissing, got {other:?}"),
    }
}

#[test]
fn detect_architecture_accepts_nested_text_config() {
    let tmp = tempfile::tempdir().unwrap();
    // Multimodal layout (Gemma 3 IT): required fields live under
    // `text_config`. Must not be reported as missing.
    write_config_json(
        tmp.path(),
        &serde_json::json!({
            "model_type": "gemma3",
            CONFIG_KEY_TEXT_CONFIG: {
                "model_type": "gemma3_text",
                CONFIG_KEY_HIDDEN_SIZE: 2560,
                CONFIG_KEY_NUM_HIDDEN_LAYERS: 34,
                CONFIG_KEY_INTERMEDIATE_SIZE: 10240,
            }
        }),
    );
    let arch = detect_architecture(tmp.path()).expect("nested text_config must resolve");
    assert_eq!(arch.config().hidden_size, 2560);
    assert_eq!(arch.config().num_layers, 34);
    assert_eq!(arch.config().intermediate_size, 10240);
}

#[test]
fn detect_architecture_accepts_flat_config() {
    let tmp = tempfile::tempdir().unwrap();
    // Text-only model with required fields at the top level (no
    // text_config wrapper). Must also be accepted.
    write_config_json(
        tmp.path(),
        &serde_json::json!({
            "model_type": "llama",
            CONFIG_KEY_HIDDEN_SIZE: 4096,
            CONFIG_KEY_NUM_HIDDEN_LAYERS: 32,
            CONFIG_KEY_INTERMEDIATE_SIZE: 11008,
            "num_attention_heads": 32,
        }),
    );
    let arch = detect_architecture(tmp.path()).expect("flat config must resolve");
    assert_eq!(arch.config().hidden_size, 4096);
    assert_eq!(arch.config().num_layers, 32);
    assert_eq!(arch.config().intermediate_size, 11008);
}

#[test]
fn detect_architecture_falls_back_to_top_level_when_text_config_omits_field() {
    let tmp = tempfile::tempdir().unwrap();
    // Mixed layout: `text_config` carries some required fields, the
    // rest sit at the top level. The presence check accepts either
    // location so users assembling configs by hand aren't tripped up.
    write_config_json(
        tmp.path(),
        &serde_json::json!({
            "model_type": "gemma3",
            CONFIG_KEY_INTERMEDIATE_SIZE: 10240,
            CONFIG_KEY_TEXT_CONFIG: {
                "model_type": "gemma3_text",
                CONFIG_KEY_HIDDEN_SIZE: 2560,
                CONFIG_KEY_NUM_HIDDEN_LAYERS: 34,
            }
        }),
    );
    let arch = detect_architecture(tmp.path()).expect("mixed layout must resolve required fields");
    assert_eq!(arch.config().hidden_size, 2560);
    assert_eq!(arch.config().num_layers, 34);
    assert_eq!(arch.config().intermediate_size, 10240);
}

#[test]
fn detect_architecture_validated_propagates_missing_config_error() {
    // The validated entrypoint is what the streaming extractor calls
    // (`build_streaming_index` in larql-vindex). It must surface the
    // same clean error rather than panic deeper down.
    let tmp = tempfile::tempdir().unwrap();
    let err = match detect_architecture_validated(tmp.path()) {
        Ok(_) => panic!("expected validated detect to fail"),
        Err(e) => e,
    };
    assert!(matches!(err, ModelError::ConfigMissing(_)));
}
