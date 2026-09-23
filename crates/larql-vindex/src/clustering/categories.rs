//! Category vocabulary and stop words for cluster labeling.
//!
//! Both vocabularies are DATA, not code: they load from JSON files
//! resolved through [`super::data_files`] (env-var dir → workspace
//! `data/`; never cwd-relative). A minimal built-in core set is kept
//! only as a loud fallback — if the file cannot be found or parsed a
//! `warning:` line goes to stderr and labeling quality degrades
//! visibly rather than silently.
//!
//! To regenerate the category file:
//! `python3 knowledge/scripts/fetch_wikidata_properties.py`

use std::collections::HashSet;
use std::path::Path;
use std::sync::OnceLock;

use super::data_files::{load_word_list, resolve_data_file, warn_builtin_fallback};

/// Filename of the category vocabulary (Wikidata property labels).
pub const CATEGORIES_FILE: &str = "wikidata_categories.json";

/// Filename of the stop-word list used by TF-IDF labeling.
pub const STOP_WORDS_FILE: &str = "stop_words.json";

/// Load category words from the resolved data file, or fall back
/// (loudly) to the minimal built-in core set.
pub fn category_words() -> Vec<String> {
    match resolve_data_file(CATEGORIES_FILE) {
        Some(path) => category_words_from(&path),
        None => {
            warn_builtin_fallback(CATEGORIES_FILE, "not found in search chain");
            builtin_categories()
        }
    }
}

/// Load categories from an explicit (config-provided) path. Falls back
/// (loudly) to the built-in core set if the file cannot be loaded.
pub fn category_words_from(path: &Path) -> Vec<String> {
    match load_word_list(path) {
        Ok(words) => words,
        Err(reason) => {
            warn_builtin_fallback(CATEGORIES_FILE, &reason);
            builtin_categories()
        }
    }
}

/// Minimal built-in category core set — a fallback, not the vocabulary.
/// The real set is `data/wikidata_categories.json` (~800 Wikidata
/// property labels).
fn builtin_categories() -> Vec<String> {
    [
        "country",
        "city",
        "place",
        "language",
        "nationality",
        "person",
        "animal",
        "plant",
        "company",
        "organization",
        "product",
        "capital",
        "currency",
        "leader",
        "occupation",
        "genre",
        "science",
        "technology",
        "music",
        "literature",
        "film",
        "sport",
        "politics",
        "religion",
        "food",
        "art",
        "history",
        "number",
        "time",
        "month",
        "color",
        "emotion",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

/// Is `tok` a stop word (excluded from cluster labels)?
///
/// The list loads once per process from [`STOP_WORDS_FILE`] via the
/// data-file search chain; on failure it falls back (loudly) to the
/// minimal built-in list.
pub fn is_stop_word(tok: &str) -> bool {
    static SET: OnceLock<HashSet<String>> = OnceLock::new();
    SET.get_or_init(|| {
        let words = match resolve_data_file(STOP_WORDS_FILE) {
            Some(path) => stop_words_from(&path),
            None => {
                warn_builtin_fallback(STOP_WORDS_FILE, "not found in search chain");
                builtin_stop_words()
            }
        };
        words.into_iter().collect()
    })
    .contains(tok)
}

/// Load the stop-word list from an explicit path. Falls back (loudly)
/// to the built-in list if the file cannot be loaded. Uncached — the
/// process-wide cache lives in [`is_stop_word`].
pub fn stop_words_from(path: &Path) -> Vec<String> {
    match load_word_list(path) {
        Ok(words) => words,
        Err(reason) => {
            warn_builtin_fallback(STOP_WORDS_FILE, &reason);
            builtin_stop_words()
        }
    }
}

/// Minimal built-in stop-word fallback. The real list is
/// `data/stop_words.json` (~150 words).
fn builtin_stop_words() -> Vec<String> {
    [
        "the", "and", "for", "but", "not", "you", "all", "can", "was", "are", "has", "his", "her",
        "its", "from", "have", "been", "will", "with", "this", "that", "they", "were", "some",
        "them", "than", "when", "what", "your", "which", "their", "would", "there", "could",
        "other", "about", "because", "through", "however",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn category_words_load_from_data_file() {
        // The workspace checkout carries data/wikidata_categories.json
        // (~800 entries) — the resolved vocabulary must be the file, not
        // the ~32-entry builtin fallback.
        let words = category_words();
        assert!(
            words.len() > builtin_categories().len() * 2,
            "expected the data-file vocabulary, got {} words (builtin-sized)",
            words.len()
        );
    }

    #[test]
    fn category_words_contains_key_terms() {
        let words = category_words();
        for term in &["country", "language", "food", "music"] {
            assert!(
                words.iter().any(|w| w == term || w.contains(term)),
                "missing category: {term}"
            );
        }
    }

    #[test]
    fn category_words_from_explicit_path_wins() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("cats.json");
        std::fs::write(&path, "[\"custom-only-category\"]").unwrap();
        assert_eq!(category_words_from(&path), vec!["custom-only-category"]);
    }

    #[test]
    fn category_words_from_missing_path_falls_back_to_builtin() {
        let words = category_words_from(Path::new("/nonexistent/path.json"));
        assert_eq!(words, builtin_categories());
        assert!(!words.is_empty());
    }

    #[test]
    fn category_words_from_invalid_json_falls_back_to_builtin() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bad.json");
        std::fs::write(&path, "not json{").unwrap();
        assert_eq!(category_words_from(&path), builtin_categories());
    }

    #[test]
    fn stop_words_detected() {
        assert!(is_stop_word("the"));
        assert!(is_stop_word("and"));
        assert!(is_stop_word("from"));
        assert!(is_stop_word("because"));
        assert!(is_stop_word("however"));
    }

    #[test]
    fn content_words_not_stop() {
        assert!(!is_stop_word("country"));
        assert!(!is_stop_word("music"));
        assert!(!is_stop_word("France"));
        assert!(!is_stop_word("president"));
    }

    #[test]
    fn stop_words_from_explicit_path_wins() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("sw.json");
        std::fs::write(&path, "[\"zzz\"]").unwrap();
        assert_eq!(stop_words_from(&path), vec!["zzz"]);
    }

    #[test]
    fn stop_words_from_missing_path_falls_back_to_builtin() {
        let words = stop_words_from(Path::new("/nonexistent/sw.json"));
        assert_eq!(words, builtin_stop_words());
    }

    #[test]
    fn data_file_stop_word_list_is_the_full_set() {
        // is_stop_word resolves the ~150-word data file, which includes
        // words absent from the minimal builtin fallback.
        let builtin = builtin_stop_words();
        assert!(!builtin.contains(&"following".to_string()));
        assert!(is_stop_word("following"));
    }
}
