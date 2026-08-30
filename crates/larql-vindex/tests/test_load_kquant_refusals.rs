//! `load_model_weights_kquant`'s two preconditions, and that each names
//! which one failed.
//!
//! Both are checked before a single weight byte is mapped, and that
//! ordering is the point: the loader goes on to build byte-range records
//! into mmaps sized from `config`, so a vindex that does not carry model
//! weights, or carries them in a format this loader does not decode, must
//! be turned away while "turn away" is still cheap and unambiguous.
//!
//! The refusals must also be distinguishable. "Rebuild with --level all"
//! and "this is the wrong quant" send an operator to different commands,
//! and a loader that answered both with one message would send half of
//! them to the wrong one.

use larql_vindex::index::types::SilentLoadCallbacks;
use larql_vindex::load_model_weights_kquant;

/// The loader's error text, or a panic naming what should have failed.
/// `ModelWeights` is not `Debug`, so `unwrap_err` is unavailable.
fn refusal(dir: &std::path::Path, why: &str) -> String {
    match load_model_weights_kquant(dir, &mut SilentLoadCallbacks) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("loaded successfully, but {why}"),
    }
}

/// The two fields the loader inspects before touching weights, over an
/// otherwise well-formed v2 index.
fn write_index(dir: &std::path::Path, has_model_weights: bool, quant: &str) {
    std::fs::create_dir_all(dir).unwrap();
    let idx = serde_json::json!({
        "version": 2,
        "model": "synthetic/kquant-refusals",
        "family": "synthetic",
        "num_layers": 1,
        "hidden_size": 64,
        "intermediate_size": 64,
        "vocab_size": 16,
        "embed_scale": 1.0,
        "extract_level": if has_model_weights { "inference" } else { "browse" },
        "dtype": "f32",
        "quant": quant,
        "layers": [
            {"layer": 0, "num_features": 2, "offset": 0, "length": 256},
        ],
        "down_top_k": 1,
        "has_model_weights": has_model_weights,
    });
    std::fs::write(
        dir.join("index.json"),
        serde_json::to_string_pretty(&idx).unwrap(),
    )
    .unwrap();
}

#[test]
fn a_vindex_without_model_weights_is_refused_with_the_rebuild_hint() {
    let dir = tempfile::tempdir().unwrap();
    let vindex = dir.path().join("browse.vindex");
    write_index(&vindex, false, "q4k");

    let err = refusal(&vindex, "a browse-level vindex carries no weights to load");
    assert!(
        err.contains("does not contain model weights"),
        "refusal must say what is absent: {err}"
    );
    assert!(
        err.contains("--level all"),
        "refusal must say how to get them: {err}"
    );
}

#[test]
fn a_non_q4k_vindex_is_refused_and_names_the_quant_it_found() {
    let dir = tempfile::tempdir().unwrap();
    let vindex = dir.path().join("f32.vindex");
    write_index(&vindex, true, "none");

    let err = refusal(&vindex, "this loader decodes Q4_K and nothing else");
    assert!(err.contains("expects a Q4_K vindex"), "{err}");
    // Naming the quant it actually found is what makes the message
    // actionable — without it the operator cannot tell a mislabelled
    // vindex from one built at the wrong quant.
    assert!(
        err.contains("quant="),
        "refusal must report the quant it saw: {err}"
    );
}

/// The two refusals are distinct texts, not one message reused.
#[test]
fn the_two_preconditions_report_themselves_separately() {
    let dir = tempfile::tempdir().unwrap();
    let no_weights = dir.path().join("a.vindex");
    let wrong_quant = dir.path().join("b.vindex");
    write_index(&no_weights, false, "q4k");
    write_index(&wrong_quant, true, "none");

    let a = refusal(&no_weights, "no weights");
    let b = refusal(&wrong_quant, "wrong quant");
    assert_ne!(a, b);
}
