//! Resolution of clustering vocabulary data files.
//!
//! The clustering/labeling code is generic engine code — its vocabularies
//! (category words, stop words, entity-pattern word lists, reference
//! relation databases) are DATA, not code, and live as JSON files in a
//! data directory. This module owns the one search chain used to find
//! them:
//!
//! 1. `LARQL_DATA_DIR` env var, joined with the filename;
//! 2. the workspace `data/` directory recorded at compile time
//!    (`CARGO_MANIFEST_DIR/../../data` — valid for dev checkouts).
//!
//! Callers that hold an explicit path (config-provided) bypass the chain
//! via the `*_from(path)` loaders. The chain never consults the process
//! working directory — cwd-relative probing was a 2026-07-30 review
//! finding (the loaded vocabulary silently depended on the directory the
//! binary was launched from).
//!
//! Every loader in this module falls back to a minimal built-in
//! vocabulary when its file cannot be found or parsed, and does so
//! LOUDLY (a `warning:` line on stderr) so a degraded vocabulary is
//! never mistaken for the real one.

use std::path::{Path, PathBuf};

/// Env var naming the directory that holds the clustering data files.
pub const DATA_DIR_ENV: &str = "LARQL_DATA_DIR";

/// Workspace `data/` directory as recorded at compile time. Exists in
/// dev checkouts; deployed binaries should set [`DATA_DIR_ENV`].
fn workspace_data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data")
}

/// Resolve `filename` through the search chain. Returns the first
/// candidate that exists as a file; `None` if nothing matches.
pub fn resolve_data_file(filename: &str) -> Option<PathBuf> {
    resolve_data_file_with_env(filename, std::env::var_os(DATA_DIR_ENV).map(PathBuf::from))
}

/// Pure version of [`resolve_data_file`] — the env-var value is passed
/// in instead of read from the process environment, so tests can drive
/// the precedence chain without mutating shared env state.
pub fn resolve_data_file_with_env(filename: &str, env_dir: Option<PathBuf>) -> Option<PathBuf> {
    if let Some(dir) = env_dir {
        let candidate = dir.join(filename);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    let candidate = workspace_data_dir().join(filename);
    if candidate.is_file() {
        return Some(candidate);
    }
    None
}

/// Loud-fallback warning, the one message format for every vocabulary
/// loader in `clustering/`.
pub(crate) fn warn_builtin_fallback(filename: &str, reason: &str) {
    eprintln!(
        "warning: clustering data file `{filename}` unavailable ({reason}); \
         falling back to the minimal built-in vocabulary — set {DATA_DIR_ENV} \
         or pass an explicit path for the full set"
    );
}

/// Read + parse a JSON word list (`["word", ...]`). `Err` carries the
/// reason for the loud-fallback message.
pub(crate) fn load_word_list(path: &Path) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read failed: {e}"))?;
    let words: Vec<String> =
        serde_json::from_str(&text).map_err(|e| format!("parse failed: {e}"))?;
    if words.is_empty() {
        return Err("file parsed to an empty list".to_string());
    }
    Ok(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_dir_takes_precedence_over_workspace() {
        let tmp = tempfile::tempdir().unwrap();
        let name = "wikidata_categories.json"; // exists in workspace data/
        std::fs::write(tmp.path().join(name), "[\"only\"]").unwrap();
        let p = resolve_data_file_with_env(name, Some(tmp.path().to_path_buf())).unwrap();
        assert_eq!(p, tmp.path().join(name));
    }

    #[test]
    fn falls_back_to_workspace_data_dir() {
        // Env dir set but file absent there → workspace copy wins.
        let tmp = tempfile::tempdir().unwrap();
        let p =
            resolve_data_file_with_env("wikidata_categories.json", Some(tmp.path().to_path_buf()))
                .expect("workspace data/wikidata_categories.json should exist in a checkout");
        assert!(p.ends_with("data/wikidata_categories.json"));
        assert!(p.is_file());
    }

    #[test]
    fn workspace_dir_is_anchored_at_compile_time_not_cwd() {
        // The 2026-07-30 review finding: the old code probed
        // `data`/`../data`/`../../data` relative to the process cwd, so the
        // loaded vocabulary silently depended on where the binary was
        // launched from. Anchoring at CARGO_MANIFEST_DIR makes every
        // candidate absolute — this holds whether or not the file exists,
        // so it stays a real guard in a checkout without `data/`.
        assert!(
            workspace_data_dir().is_absolute(),
            "workspace data dir {:?} is cwd-relative",
            workspace_data_dir()
        );
    }

    #[test]
    fn missing_everywhere_is_none() {
        assert!(resolve_data_file_with_env("no_such_vocab_file.json", None).is_none());
    }

    #[test]
    fn load_word_list_reads_json_array() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("w.json");
        std::fs::write(&path, "[\"alpha\", \"beta\"]").unwrap();
        assert_eq!(load_word_list(&path).unwrap(), vec!["alpha", "beta"]);
    }

    #[test]
    fn load_word_list_errors_on_missing_invalid_and_empty() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(load_word_list(&tmp.path().join("absent.json"))
            .unwrap_err()
            .contains("read failed"));

        let bad = tmp.path().join("bad.json");
        std::fs::write(&bad, "not json{").unwrap();
        assert!(load_word_list(&bad).unwrap_err().contains("parse failed"));

        let empty = tmp.path().join("empty.json");
        std::fs::write(&empty, "[]").unwrap();
        assert!(load_word_list(&empty).unwrap_err().contains("empty list"));
    }
}
