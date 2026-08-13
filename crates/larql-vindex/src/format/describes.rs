//! Reading a vindex's own description of itself, without loading it.
//!
//! A vindex records the checkpoint it was built from. Anything that needs to
//! know *which model this is* — a tokenizer to fetch, a config to resolve, a
//! bench label to print — should ask the artifact rather than carry a default
//! model id of its own.
//!
//! That is not a style preference. A hardcoded default silently disagrees
//! with the vindex the user actually passed, and the disagreement surfaces
//! far away: the wrong tokenizer produces plausible token ids, the wrong
//! config produces plausible shapes, and the first honest error arrives deep
//! inside a dequant parser complaining about byte counts.

use std::path::Path;

use super::filenames::INDEX_JSON;
use crate::config::VindexConfig;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// The model id a vindex was built from, or `None` when the directory has no
/// readable `index.json` or records no model.
///
/// Deliberately `Option` rather than an error: callers are usually choosing a
/// *default*, and "the artifact does not say" is a normal answer that should
/// fall through to whatever the caller was going to do anyway.
pub fn model_id_at(vindex_dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(vindex_dir.join(INDEX_JSON)).ok()?;
    let config: VindexConfig = serde_json::from_str(&text).ok()?;
    (!config.model.is_empty()).then_some(config.model)
}

#[cfg(test)]
mod tests {
    use super::model_id_at;
    use crate::config::VindexConfig;

    const MODEL_ID: &str = "acme/some-model-7b";

    fn write_index(dir: &std::path::Path, model: &str) {
        let config = VindexConfig {
            model: model.to_string(),
            ..Default::default()
        };
        std::fs::write(
            dir.join(super::INDEX_JSON),
            serde_json::to_string(&config).expect("serialise"),
        )
        .expect("write index.json");
    }

    #[test]
    fn reads_the_model_the_vindex_records() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_index(dir.path(), MODEL_ID);
        assert_eq!(model_id_at(dir.path()).as_deref(), Some(MODEL_ID));
    }

    #[test]
    fn a_directory_with_no_index_says_nothing() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(model_id_at(dir.path()).is_none());
    }

    #[test]
    fn an_empty_model_field_says_nothing() {
        // Distinguished from "recorded the empty string", because a caller
        // choosing a default must not end up asking a hub for `""`.
        let dir = tempfile::tempdir().expect("tempdir");
        write_index(dir.path(), "");
        assert!(model_id_at(dir.path()).is_none());
    }

    #[test]
    fn unparseable_json_says_nothing_rather_than_panicking() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join(super::INDEX_JSON), "{ not json").expect("write");
        assert!(model_id_at(dir.path()).is_none());
    }
}
