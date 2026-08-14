//! `norm_eps` parsing — covers bug 2 from
//! docs/diagnoses/shannon-cross-engine-divergence.md (rms_norm_eps was
//! hardcoded to 1e-6 in `ModelArchitecture::norm_eps()`, ignoring the
//! model's config.json).

use super::support::minimal_llama_config;
use crate::detect::*;

#[test]
fn norm_eps_from_rms_norm_eps_field() {
    // Llama / Mistral / Gemma all ship `rms_norm_eps`. Parser must read it
    // and `arch.norm_eps()` must return the parsed value, not the 1e-6
    // default. This is the root cause of the +8.2% Mistral 7B drift.
    let arch = detect_from_json(&minimal_llama_config(serde_json::json!({
        "rms_norm_eps": 1e-5,
    })));
    assert_eq!(arch.config().norm_eps, Some(1e-5));
    assert_eq!(arch.norm_eps(), 1e-5);
}

#[test]
fn norm_eps_from_layer_norm_epsilon_field() {
    // GPT-2 family uses `layer_norm_epsilon`. Same parser, same trait
    // method, same outcome.
    let arch = detect_from_json(&minimal_llama_config(serde_json::json!({
        "model_type": "gpt2",
        "layer_norm_epsilon": 1e-5,
    })));
    assert_eq!(arch.config().norm_eps, Some(1e-5));
    assert_eq!(arch.norm_eps(), 1e-5);
}

#[test]
fn norm_eps_from_norm_epsilon_field() {
    // StarCoder2 uses `norm_epsilon`. This was the unfixed bug surfaced
    // by the multi-arch diagnostic sweep on 2026-05-16.
    let arch = detect_from_json(&minimal_llama_config(serde_json::json!({
        "model_type": "starcoder2",
        "norm_epsilon": 1e-5,
    })));
    assert_eq!(arch.config().norm_eps, Some(1e-5));
    assert_eq!(arch.norm_eps(), 1e-5);
}

#[test]
fn norm_eps_falls_back_to_default_when_absent() {
    // No eps field in config → trait fallback (`DEFAULT_NORM_EPS = 1e-6`).
    // Older models (Llama 1, Gemma 1, BERT) relied on this.
    let arch = detect_from_json(&minimal_llama_config(serde_json::json!({})));
    assert!(arch.config().norm_eps.is_none());
    assert_eq!(arch.norm_eps(), crate::defaults::DEFAULT_NORM_EPS);
}
